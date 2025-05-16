import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import method.layers as layers
from flash_pytorch import GAU
from globals import args_global
from torch_geometric.nn import GIN

# Manually add CATS functionality switches to args_global
if not hasattr(args_global, 'use_cats_contrast'):
    args_global.use_cats_contrast = True  # Default: enable contrastive learning
if not hasattr(args_global, 'use_cats_attention'):
    args_global.use_cats_attention = True  # Default: enable attention mechanism


def _coo_scipy2torch(adj):
    """
    Convert from scipy sparse COO matrix to PyTorch sparse tensor format
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i,v, torch.Size(adj.shape))


class Similarity(nn.Module):
    """
    Class for computing dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y, temp=None):
        if temp is None:
            temp = self.temp
        return self.cos(x, y) / temp


class GraphSAINT(nn.Module):
    def __init__(self, num_classes, arch_gcn, train_params, feat_full, label_full, cpu_eval=False):
        """
        Build multi-layer GNN architecture of GraphSAINT model
        
        Args:
            num_classes: int, number of classes a node can belong to
            arch_gcn: dict, config for each GNN layer
            train_params: dict, training hyperparameters (e.g., learning rate)
            feat_full: np array, shape N x f, N is the total num of nodes, f is input node feature dimension
            label_full: np array, shape N x c for multi-class classification, N x 1 for single-class classification
            cpu_eval: bool, if True, put model on CPU for evaluation
        """
        super(GraphSAINT,self).__init__()
        
        # === Text processing initialization ===
        # Vocabulary size: determined by the maximum value in the input features (plus 1 because indexing starts from 0)
        self.vocab_size = np.max(feat_full)+1  
        # Create mask: for handling padding values, set mask to 1 at positions where feature value is 0
        self.mask = np.zeros_like(feat_full)  
        self.mask[feat_full==0]=1  
        self.mask = torch.from_numpy(self.mask.astype(np.bool))  

        # === Device configuration ===
        # Determine whether to use CUDA based on global parameters
        self.use_cuda = (args_global.gpu >= 0)  
        if cpu_eval:
            self.use_cuda = False  # If CPU evaluation is specified, disable CUDA

        # === Aggregator configuration ===
        # Select appropriate aggregator type and number of multi-head attention based on configuration
        self.use_GAaN = args_global.use_GAaN
        self.use_CATS = args_global.use_CATS if hasattr(args_global, 'use_CATS') else False  # CATS flag
        self.use_TGCA = args_global.use_TGCA
        if self.use_GAaN:
            self.aggregator_cls = layers.GatedAttentionAggregator
            self.mulhead = 2
        else:
            if "attention" in arch_gcn:
                if "gated_attention" in arch_gcn:
                    if arch_gcn['gated_attention']:
                        # Use gated attention aggregator: can adaptively adjust the importance of neighbor nodes
                        self.aggregator_cls = layers.GatedAttentionAggregator  
                        self.mulhead = int(arch_gcn['attention'])  # Set the number of multi-head attention
                else:
                    # Use regular attention aggregator: aggregate neighbor features through attention mechanism
                    self.aggregator_cls = layers.AttentionAggregator  
                    self.mulhead = int(arch_gcn['attention'])  # Set the number of multi-head attention
            else:
                # Use high-order aggregator: can aggregate neighbor information at different distances
                self.aggregator_cls = layers.HighOrderAggregator  
                self.mulhead = 1  # Don't use multi-head attention mechanism

        # === Model base parameter settings ===
        # Parse network architecture string to determine number of layers
        self.num_layers = len(arch_gcn['arch'].split('-'))  
        self.weight_decay = train_params['weight_decay']  # Weight decay rate for regularization
        self.dropout = train_params['dropout']  # Dropout ratio to prevent overfitting
        self.lr = train_params['lr']  # Learning rate
        self.arch_gcn = arch_gcn  # Save GCN architecture configuration
        # Determine the type of loss function used
        self.sigmoid_loss = (arch_gcn['loss'] == 'sigmoid')  # True for multi-label classification, False for single-label
        
        # === Data transformation processing ===
        # Convert NumPy arrays to PyTorch tensors
        self.feat_full = torch.from_numpy(feat_full.astype(np.float32))  
        self.label_full = torch.from_numpy(label_full.astype(np.float32))  
        # Get sentence embedding method name
        self.sentence_embed_method = train_params["sentence_embed"]  
        # Set sentence embedding and get its dimension
        self.sentence_embedding_dim = self.set_sentence_embedding(train_params["sentence_embed"])  

        # === Device migration ===
        # If using GPU, move data to CUDA device
        if self.use_cuda:
            self.feat_full = self.feat_full.cuda()
            self.label_full = self.label_full.cuda()
            self.mask = self.mask.cuda()
        
        # Special handling for single-label classification
        if not self.sigmoid_loss:
            # For single-class classification, need to convert one-hot encoding to category index
            self.label_full_cat = torch.from_numpy(label_full.argmax(axis=1).astype(np.int64))
            if self.use_cuda:
                self.label_full_cat = self.label_full_cat.cuda()

        # === Network structure settings ===
        # Set number of classes
        self.num_classes = num_classes  
        # Parse layer configuration to get dimensions, aggregation method, activation function, etc.
        _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer \
                        = parse_layer_yml(arch_gcn, self.sentence_embedding_dim)
        
        # Set convolution layer index, used for JK network and other special structures
        self.set_idx_conv()  
        # Set dimensions for each layer
        self.set_dims(_dims)  

        # === Model state initialization ===
        self.loss = 0  # Initialize loss value
        self.opt_op = None  # Initialize optimization operation

        # === Build network model ===
        self.num_params = 0  # Initialize parameter count
        # Get aggregator list and parameter count
        self.aggregators, num_param = self.get_aggregators()  
        self.num_params += num_param  # Accumulate parameter count
        # Build aggregators into a sequential model
        self.conv_layers = nn.Sequential(*self.aggregators)  
        # Get hidden layer dimension
        self.hidden_dim = train_params["hidden_dim"]  
        # Whether to use graph structure
        self.no_graph = train_params["no_graph"]  
        # Create dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)  
        self.use_GIN = args_global.use_GIN
        if self.use_GIN and not self.no_graph:
            # If graph-level embedding is enabled, initialize GIN model
            self.graph_level_embedding = self._build_graph_level_embedding()
        self.use_gau = args_global.use_gau
        self.gate_layer = nn.Linear(self.dims_feat[-1]*2+self.sentence_embedding_dim, 1)  # emb + feat + graph_emb concatenated
        self.use_TGCA = args_global.use_TGCA
        self.contrastive_weight = 0.1  # Reduce contrastive learning loss weight, originally 0.1
        self.use_triplet_loss = args_global.use_triplet_loss
        self.triplet_weight = 0.1  # Adjustable

        # SMOTE related additions
        # Priority: Command-line args > config file > default values
        if args_global.use_smote:
            self.use_smote = True
            self.smote_k_neighbors = args_global.smote_k_neighbors
            self.smote_random_state = args_global.smote_random_state
            self.smote_synthetic_batch_size = args_global.synthetic_batch_size # This will be used by Minibatch
            self.smote_loss_weight = args_global.smote_loss_weight
            printf("GraphSAINT: SMOTE explicitly enabled via command-line argument.", style="green")
        else:
            self.use_smote = train_params.get('use_smote', False)
            self.smote_k_neighbors = train_params.get('smote_k_neighbors', 5)
            self.smote_random_state = train_params.get('smote_random_state', 42)
            # synthetic_batch_size from train_params is primarily for Minibatch, but store it if needed.
            self.smote_synthetic_batch_size = train_params.get('synthetic_batch_size', 64)
            self.smote_loss_weight = train_params.get('smote_loss_weight', 0.5)
            if self.use_smote:
                printf("GraphSAINT: SMOTE enabled via configuration file.", style="green")

        self.synthetic_processor = None
        if self.use_smote:
            # printf("GraphSAINT: SMOTE enabled. Initializing synthetic processor.", style="green") # Moved message above
            # This processor takes already embedded synthetic features
            # Its architecture should be comparable to the classifier part that processes real node embeddings
            if self.hidden_dim == -1:
                 # Directly to num_classes if no main hidden GNN layer
                self.synthetic_processor = layers.HighOrderAggregator(self.sentence_embedding_dim, 
                                                                      self.num_classes,
                                                                      act='I', order=0, 
                                                                      dropout=self.dropout, bias='bias')
                self.num_params += self.synthetic_processor.num_param
            else:
                # If there's a hidden layer in the main classifier, synthetic data might also go through a similar transform
                self.synthetic_processor_hidden = layers.HighOrderAggregator(self.sentence_embedding_dim, 
                                                                            self.hidden_dim, 
                                                                            act='relu', order=0, 
                                                                            dropout=self.dropout, bias='norm-nn')
                self.synthetic_processor_final = nn.Linear(self.hidden_dim, self.num_classes)
                self.num_params += self.synthetic_processor_hidden.num_param + self.num_classes*self.hidden_dim
                # Need to register these new parameters if they are separate nn.Modules
                # If synthetic_processor_hidden and synthetic_processor_final are part of a nn.Sequential, that's handled.
                # Otherwise, ensure they are added to self.parameters() e.g. by making them attributes like self.classifier_
            printf(f"GraphSAINT: Added synthetic processor. Total params now: {self.num_params}")

        # CATS module initialization
        if self.use_CATS:
            # CATS functionality switch: read whether to use contrastive learning and attention mechanism
            self.use_cats_contrast = True  # Default: enable contrastive learning
            self.use_cats_attention = True # Default: enable attention mechanism
            self.arc_margin = 0.3
            self.arc_scale = 30.0

            # If global parameters are set, use the values of global parameters
            if hasattr(args_global, 'use_cats_contrast'):
                self.use_cats_contrast = args_global.use_cats_contrast
            if hasattr(args_global, 'use_cats_attention'):
                self.use_cats_attention = args_global.use_cats_attention
                
            # SimCSE contrastive learning temperature parameter
            self.simcse_temp = 0.05
            # Projection layer for contrastive learning
            self.text_proj = nn.Sequential(
                nn.Linear(self.sentence_embedding_dim, self.sentence_embedding_dim),
                nn.ReLU(),
                nn.Linear(self.sentence_embedding_dim, self.sentence_embedding_dim)
            )
            
            # Feature projection layer - project two types of features into the same dimensional space
            # Choose common dimension: use the smaller of text feature dimension and graph feature dimension, and ensure divisible by number of attention heads
            self.common_dim = min(self.sentence_embedding_dim, self.dims_feat[-1])
            # Ensure divisible by number of attention heads (8)
            self.common_dim = (self.common_dim // 8) * 8
            self.graph_proj = nn.Linear(self.dims_feat[-1], self.common_dim)
            self.text_attn_proj = nn.Linear(self.sentence_embedding_dim, self.common_dim)
            
            # Attention layer - using projected features of the same dimension
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.common_dim,
                num_heads=8,
                dropout=self.dropout
            )
            
            # Fusion layer - determine input dimension based on whether attention mechanism is used
            if self.use_cats_attention:
                # When using attention, input is [graph feature + text feature + attention output]
                self.fusion_layer = nn.Linear(
                    self.dims_feat[-1] + self.sentence_embedding_dim + self.common_dim,
                    self.dims_feat[-1] + self.sentence_embedding_dim
                )
            else:
                # When not using attention, input is only [graph feature + text feature]
                self.fusion_layer = nn.Linear(
                    self.dims_feat[-1] + self.sentence_embedding_dim,
                    self.dims_feat[-1] + self.sentence_embedding_dim
                )
            self.layer_norm = nn.LayerNorm(self.dims_feat[-1] + self.sentence_embedding_dim)

        if self.use_TGCA:
            self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.sentence_embedding_dim, nhead=1)
            self.transformer_encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=1)

            self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.dims_weight[self.num_layers - 1][1], nhead=2)
            self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=1)
            self.encoder = layers.Encoder(self.dims_weight[self.num_layers - 1][1], self.sentence_embedding_dim)
            self.att_struct = nn.MultiheadAttention(embed_dim=self.sentence_embedding_dim, num_heads=8, dropout=self.dropout)
            self.att_text = nn.MultiheadAttention(embed_dim=self.sentence_embedding_dim, num_heads=8,dropout=self.dropout)
            self.decoder = layers.Decoder(self.sentence_embedding_dim * 4, self.dims_weight[self.num_layers - 1][1])
            self.att = nn.MultiheadAttention(embed_dim=self.dims_weight[self.num_layers - 1][1], num_heads=8, dropout=self.dropout)
        # === Classifier settings ===
        if self.hidden_dim == -1:
            if self.no_graph:
                print("NO GRAPH")  # Output prompt for not using graph structure
                # Use only sentence embedding feature for classification
                self.classifier = layers.HighOrderAggregator(
                    self.sentence_embedding_dim, self.num_classes, 
                    act='I',  # 'I' for identity activation function
                    order=0,  # 0th order for not aggregating neighbors
                    dropout=self.dropout, 
                    bias='bias'  # Use bias term
                )
                self.num_params += self.classifier.num_param  # Accumulate classifier parameter count
            else:
                if self.use_GIN and self.use_CATS and self.use_cats_attention:
                    # Combine graph feature and text feature for classification
                    # Initialize GAU model
                    self.gau = GAU(
                        dim=self.dims_feat[-1]*3+self.sentence_embedding_dim,
                        query_key_dim=(self.dims_feat[-1]+self.sentence_embedding_dim) // 4,
                        expansion_factor=2,
                        causal=False,
                        laplace_attn_fn=True
                    )
                    self.classifier = layers.HighOrderAggregator(
                        self.dims_feat[-1]*3+self.sentence_embedding_dim, 
                        self.num_classes,
                        act='I', 
                        order=0, 
                        dropout=self.dropout, 
                        bias='bias'
                    )
                elif self.use_GIN:
                    # Combine graph feature and text feature for classification
                    # Initialize GAU model
                    self.gau = GAU(
                        dim=self.dims_feat[-1]*2+self.sentence_embedding_dim,
                        query_key_dim=(self.dims_feat[-1]*2+self.sentence_embedding_dim) // 4,
                        expansion_factor=2,
                        causal=False,
                        laplace_attn_fn=True
                    )
                    self.classifier = layers.HighOrderAggregator(
                        self.dims_feat[-1]*2+self.sentence_embedding_dim, 
                        self.num_classes,
                        act='I', 
                        order=0, 
                        dropout=self.dropout, 
                        bias='bias'
                    )
                elif self.use_CATS and self.use_cats_attention:
                    self.gau = GAU(
                        dim=self.dims_feat[-1]*2+self.sentence_embedding_dim,
                        query_key_dim=(self.dims_feat[-1]*2+self.sentence_embedding_dim)//4,
                        expansion_factor=2,
                        causal=False,
                        laplace_attn_fn=True
                    )
                    self.classifier = layers.HighOrderAggregator(
                        self.dims_feat[-1]*2+self.sentence_embedding_dim, 
                        self.num_classes,
                        act='I', 
                        order=0, 
                        dropout=self.dropout, 
                        bias='bias'
                    )
                else:
                    self.gau = GAU(
                        dim=self.dims_feat[-1]+self.sentence_embedding_dim,
                        query_key_dim=(self.dims_feat[-1]+self.sentence_embedding_dim) // 4,
                        expansion_factor=2,
                        causal=False,
                        laplace_attn_fn=True
                    )
                    self.classifier = layers.HighOrderAggregator(
                        self.dims_feat[-1]+self.sentence_embedding_dim, 
                        self.num_classes,
                        act='I', 
                        order=0, 
                        dropout=self.dropout, 
                        bias='bias'
                    )
                self.num_params += self.classifier.num_param  # Accumulate classifier parameter count
        else:
            # Use additional hidden layer
            # First map features to hidden layer dimension
            self.classifier_ = layers.HighOrderAggregator(
                self.dims_feat[-1]+self.sentence_embedding_dim, 
                self.hidden_dim,
                act='relu',  # Use ReLU activation
                order=0, 
                dropout=self.dropout, 
                bias='norm-nn'  # Use normalized neural network bias
            )
            # Then map from hidden layer to output category
            self.classifier = nn.Linear(self.hidden_dim, self.num_classes)  
            # Accumulate two-layer classifier parameter count
            self.num_params += self.classifier_.num_param + self.num_classes*self.hidden_dim

        # === Regularization and optimizer ===
        # Batch normalization for sentence embedding to improve training stability
        self.sentence_embed_norm = nn.BatchNorm1d(
            self.sentence_embedding_dim, 
            eps=1e-9,  # Numerical stability parameter
            track_running_stats=True  # Track running time statistics
        )
        # Use Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.triplet_loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    def _build_graph_level_embedding(self):
        """Build graph-level embedding layer"""
        return GIN(
            in_channels=self.dims_feat[-1],  # Input feature dimension
            hidden_channels=self.dims_feat[-1],  # Keep same dimension as input
            num_layers=2,  # GIN layer count
            dropout=0.1,  # Dropout rate
            jk='last',  # Use output of last layer
            train_eps=True  # Whether to train epsilon parameter
        )

    def set_dims(self, dims):
        """
        Set feature dimension/weight dimension for each GNN or MLP layer
        These dimensions will be used to initialize PyTorch layers
        
        Args:
            dims: list, node feature length for each hidden layer
            
        Returns:
            None
        """
        # Set feature dimension for each layer, considering aggregation method and order
        self.dims_feat = [dims[0]] + [
            ((self.aggr_layer[l]=='concat') * self.order_layer[l] + 1) * dims[l+1]
            for l in range(len(dims) - 1)
        ]
        # Set weight dimension for each layer
        self.dims_weight = [(self.dims_feat[l],dims[l+1]) for l in range(len(dims)-1)]

    def set_idx_conv(self):
        """
        Set index for each layer in complete neural network
        For example, if complete neural network structure is 1-0-1-0 (1-hop graph convolution, followed by 0-hop MLP, ...),
        then layer indices will be 0, 2
        """
        # Find all layers where graph convolution layer exists
        idx_conv = np.where(np.array(self.order_layer) >= 1)[0]
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer) - 1)
        _o_arr = np.array(self.order_layer)[idx_conv]
        # Ensure index layers satisfy conditions, otherwise use default index
        if np.prod(np.ediff1d(_o_arr)) == 0:
            self.idx_conv = idx_conv
        else:
            self.idx_conv = list(np.where(np.array(self.order_layer) == 1)[0])


    def cos_sim(self, x, is_training):
        """
        Compute cosine similarity between input vectors
        
        Args:
            x: Input vectors
            is_training: Whether in training mode
            
        Returns:
            Cosine similarity matrix
        """
        if is_training:
            # Directly compute cosine similarity between all vector pairs in training mode
            return nn.functional.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
        B, L = x.shape
        sims = torch.zeros(size=(B,B))
        # Compute cosine similarity one by one in non-training mode (possible for memory saving)
        for i in range(B):
            sims[i,:] = nn.functional.cosine_similarity(x, x[i].unsqueeze(0))
        return sims.to(x.device)

    def top_sim(self, sims, topk=3):
        """
        Extract top k highest similarities for each node with other nodes
        
        Args:
            sims: Similarity matrix
            topk: Number of highest similarities to retain for each node
            
        Returns:
            Sparse similarity matrix, only retain top k highest similarities
        """
        sims_, indices_ = sims.sort(descending=True)  # Sort similarities in descending order
        B,_ = sims.shape
        indices = torch.zeros(size=(2,B*topk))  # Initialize index tensor
        values = torch.zeros(size=(1,B*topk))  # Initialize value tensor
        for i,inds in enumerate(indices_):
            indices[0, i*topk:(i+1)*topk] = i*torch.ones(size=(1,topk))  # Set row index
            indices[1, i * topk:(i + 1) * topk] = inds[:topk]  # Set column index
            values[0, i * topk:(i + 1) * topk] = sims[i][:topk]  # Set similarity value
        return torch.sparse_coo_tensor(indices,values.squeeze(0),size=(B,B))  # Return sparse tensor
        # return indices, values


    # Sentence embedding, CNN is the best method
    def set_sentence_embedding(self, method="cnn"):
        """
        Set sentence embedding method
        
        Args:
            method: Embedding method name, such as "cnn", "maxpool", "rnn", etc.
            
        Returns:
            Embedding dimension size
        """
        if method=="cnn":
            # CNN embedding method
            embed_size = 128  # Embedding dimension
            filter_size = [3, 4, 5]  # Convolution kernel size
            filter_num = 128  # Convolution kernel count
            self.embedding = nn.Embedding(self.vocab_size, embed_size)  # Word embedding layer
            self.cnn_list = nn.ModuleList()  # CNN module list
            for size in filter_size:
                self.cnn_list.append(nn.Conv1d(embed_size, filter_num, size))  # Add different size convolution layers
            self.relu = nn.ReLU()  # ReLU activation function
            self.max_pool = nn.AdaptiveMaxPool1d(1)  # Adaptive max pooling
            self.sentence_embed=self.cnn_embed  # Set sentence embedding function
            return len(filter_size) * filter_num  # Return embedding dimension
        '''
        one layer FCN equals only pool
        '''
        if method == "maxpool":
            # Max pooling embedding method
            self.embed_size = 128
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.relu = nn.ReLU()
            self.embed_pool = nn.AdaptiveMaxPool1d(1)
            self.sentence_embed=self.pool_embed
            return self.embed_size
        if method == "avgpool":
            # Average pooling embedding method
            self.embed_size = 128
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.relu = nn.ReLU()
            self.embed_pool = nn.AdaptiveAvgPool1d(1)
            self.sentence_embed = self.pool_embed
            return self.embed_size
        if method == "rnn":
            # RNN embedding method
            self.embed_size = 128
            hidden_dim = 64
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.rnn = nn.LSTM(self.embed_size, hidden_dim, 1, dropout=self.dropout, bidirectional=True)
            self.sentence_embed=self.RNN_embed
            return 2 * hidden_dim
        if method == "lstm":
            # LSTM embedding method
            self.embed_size = 128
            hidden_dim = 64
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.rnn = nn.LSTM(self.embed_size, hidden_dim, 1, dropout=self.dropout, bidirectional=True)
            self.sentence_embed=self.LSTM_embed
            return 2 * hidden_dim * 2
        if method == "lstmatt":
            # LSTM embedding method with attention
            self.embed_size = 128
            hidden_dim = 64
            self.LSTMATT_DP=nn.Dropout(self.dropout)
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.lstm = nn.LSTM(self.embed_size, hidden_dim, 1, dropout=self.dropout, bidirectional=True)
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=1, dropout=self.dropout)
            self.sentence_embed = self.LSTMATT_embed
            self.relu = nn.ReLU()
            self.embed_pool = nn.AdaptiveAvgPool1d(1)
            return hidden_dim*2
        if method == "Transformer":
            # Transformer embedding method
            self.embed_size = 128
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            self.Trans_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=1)
            self.sentence_embed =self.Transformer_embed
            return self.embed_size
        if method == "gnn":
            # Graph neural network embedding method
            embed_size = 128
            self.edge_weight = nn.Embedding((self.vocab_size) * (self.vocab_size)+1, 1, padding_idx=0)
            self.node_embedding = nn.Embedding(self.vocab_size, embed_size, padding_idx=0)
            self.node_weight = nn.Embedding(self.vocab_size, 1, padding_idx=0)
            # nn.init.xavier_uniform_(self.edge_weight.weight)
            # nn.init.xavier_uniform_(self.node_weight.weight)
            self.sentence_embed = self.gnn_embed
            return embed_size
            # self.fc = nn.Sequential(
            #     nn.Linear(embed_size, 2),
            #     nn.ReLU(),
            #     nn.Dropout(self.dropout),
            #     nn.LogSoftmax(dim=1)
            # )


    def forward(self, node_subgraph, adj_subgraph,current_epoch=10, is_training=True, synthetic_feat_batch=None):
        """
        Forward propagation function: integrate text, graph structure, CATS, TGCA (baseline) and SwAV, GAU, Graph-Level (optional components)
        """
        # === 1. Get original input features ===
        feat_subg = self.feat_full[node_subgraph]
        label_subg = self.label_full[node_subgraph]
        mask_subg = self.mask[node_subgraph]

        feat_subg = self.sentence_embed(tokens=feat_subg, padding_mask=mask_subg, is_training=is_training)
        label_subg_converted = label_subg if self.sigmoid_loss else self.label_full_cat[node_subgraph]

        pred_synthetic = None
        # === 2. Special branch: no graph scenario ===
        if self.no_graph:
            pred_subg = self.classifier((None, feat_subg))[1]
            return pred_subg, label_subg_converted, pred_synthetic

        # === 3. Text feature preprocessing ===
        feat_subg_ = self.sentence_embed_norm(feat_subg)
        feat_subg_ = self.dropout_layer(feat_subg_)

        if current_epoch >= 0:

            # === 5. baseline branch ===
            if self.use_TGCA:
                print("use_TGCA")
                trans_embed = self.transformer_encoder1(feat_subg_.unsqueeze(0)).squeeze(0)
                _, emb_subg = self.conv_layers((adj_subgraph, trans_embed))
                transGNN_embed = self.transformer_encoder2(emb_subg.unsqueeze(0)).squeeze(0)
                emb_subg = self.encoder(transGNN_embed)

                att_struct, _ = self.att_struct(emb_subg.unsqueeze(0), feat_subg_.unsqueeze(0), emb_subg.unsqueeze(0))
                att_text, _ = self.att_text(feat_subg_.unsqueeze(0), emb_subg.unsqueeze(0), feat_subg_.unsqueeze(0))

                combine = torch.cat([emb_subg, feat_subg_, att_struct.squeeze(0), att_text.squeeze(0)], dim=1)
                emb_subg = self.decoder(combine)
                emb_subg, _ = self.att(emb_subg.unsqueeze(0), emb_subg.unsqueeze(0), emb_subg.unsqueeze(0))
                emb_subg = emb_subg.squeeze(0)
                
                fused_features = torch.cat([emb_subg, feat_subg], dim=1)
                if self.use_gau:
                    fused_features = self.gau(fused_features.unsqueeze(1)).squeeze(1)


            elif self.use_CATS:
                # === 4. Graph structure encoding ===
                _, emb_subg = self.conv_layers((adj_subgraph, feat_subg_))
                emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)
                fused_features, contrastive_loss = self.apply_cats(feat_subg_, emb_subg_norm)
                if is_training:
                    self.contrastive_loss = contrastive_loss
                if self.use_GIN:
                    if adj_subgraph.is_sparse:
                        adj_subgraph = adj_subgraph.coalesce()
                    graph_emb = self.graph_level_embedding(emb_subg_norm, adj_subgraph)
                    graph_emb = graph_emb.mean(dim=0, keepdim=True).expand(emb_subg_norm.size(0), -1)
                    fused_features = torch.cat([fused_features, graph_emb], dim=1)
                if self.use_gau:
                    fused_features = self.gau(fused_features.unsqueeze(1)).squeeze(1)

            else:
                # === 4. Graph structure encoding ===
                _, emb_subg = self.conv_layers((adj_subgraph, feat_subg_))
                emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)
                # === 6. Default feature concatenation ===
                fused_features = torch.cat([emb_subg_norm, feat_subg], dim=1)
                # === 7. Optional enhanced component ===
                if self.use_GIN:
                    print("use_GIN")
                    if adj_subgraph.is_sparse:
                        adj_subgraph = adj_subgraph.coalesce()
                    graph_emb = self.graph_level_embedding(emb_subg_norm, adj_subgraph)
                    graph_emb = graph_emb.mean(dim=0, keepdim=True).expand(emb_subg_norm.size(0), -1)
                    fused_features = torch.cat([fused_features, graph_emb], dim=1)

                if self.use_gau:
                    fused_features = self.gau(fused_features.unsqueeze(1)).squeeze(1)
            
            self.last_fused_features = fused_features

            if self.hidden_dim == -1:
                pred_subg = self.classifier((None, fused_features))[1]
            else:
                pred_subg = self.classifier_((None, fused_features))[1]
                pred_subg = self.classifier(pred_subg)
        else:
            pred_subg = self.classifier2((None, feat_subg))[1]
        print("use_smote", self.use_smote)
        print("synthetic_feat_batch", synthetic_feat_batch)
        print("self.synthetic_processor", self.synthetic_processor)
        if self.use_smote and synthetic_feat_batch is not None and self.synthetic_processor is not None and is_training:
            print("use_smote")
            if self.hidden_dim == -1:
                pred_synthetic = self.synthetic_processor((None, synthetic_feat_batch))[1]
            else: # Assumes synthetic_processor_hidden and synthetic_processor_final are defined
                hidden_output_synthetic = self.synthetic_processor_hidden((None, synthetic_feat_batch))[1]
                pred_synthetic = self.synthetic_processor_final(hidden_output_synthetic)
        elif self.use_smote and synthetic_feat_batch is not None and self.hidden_dim != -1 and self.synthetic_processor_hidden is not None and is_training:
             hidden_output_synthetic = self.synthetic_processor_hidden((None, synthetic_feat_batch))[1]
             pred_synthetic = self.synthetic_processor_final(hidden_output_synthetic)
        
        return pred_subg, label_subg_converted, pred_synthetic


    def compute_contrastive_loss(self, text_features):
        """
        Compute unsupervised contrastive learning loss (SimCSE method)
        
        Args:
            text_features: Text features, size [B, D_text]
            
        Returns:
            enhanced_text: Enhanced text representation
            contrastive_loss: Contrastive loss
        """
        batch_size = text_features.size(0)

        if self.training:
            # Training stage: Compute contrastive loss
            proj1 = self.text_proj(text_features)
            # proj1 = F.normalize(proj1, p=2, dim=1)

            proj2 = self.text_proj(F.dropout(text_features, p=0.1, training=True))
            # proj2 = F.normalize(proj2, p=2, dim=1)

            cos_sim = torch.matmul(proj1, proj2.transpose(0, 1)) / self.simcse_temp
            sim_labels = torch.arange(batch_size, device=text_features.device)


            # ArcCSE part
            # margin = getattr(self, 'arc_margin', 0.3)
            # scale = getattr(self, 'arc_scale', 30.0)

            # one_hot = F.one_hot(sim_labels, num_classes=batch_size).float()
            # theta = torch.acos(torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7))  # Avoid NaN
            # cos_theta_m = torch.cos(theta + margin)

            # # Apply margin angle, only replace logit for correct class
            # logits = scale * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_sim)

            # # Loss calculation
            # contrastive_loss = F.cross_entropy(logits, sim_labels)


            # Compute contrastive loss  
            contrastive_loss = F.cross_entropy(cos_sim, sim_labels)

            # Average two views as enhanced representation
            enhanced_text = (proj1 + proj2) / 2
        else:
            # Validation stage: Skip contrastive loss calculation for efficiency
            proj = self.text_proj(text_features)
            # enhanced_text = F.normalize(proj, p=2, dim=1)
            enhanced_text = proj
            # Create an empty loss, no impact on backpropagation
            contrastive_loss = torch.tensor(0.0, device=text_features.device)
            
        return enhanced_text, contrastive_loss
    
    def apply_cross_attention(self, text_features, graph_features, enhanced_text):
        """
        Apply cross-modal attention mechanism to fuse text and graph features
        
        Args:
            text_features: Original text features, size [B, D_text]
            graph_features: Graph structure features, size [B, D_graph]
            enhanced_text: Text feature enhanced by contrastive learning
            
        Returns:
            fused_features: Merged features, size [B, D_fused]
        """
        # Project to unified dimension
        graph_common = self.graph_proj(graph_features)
        text_common = self.text_attn_proj(enhanced_text)


        # Cross-modal attention: Graph Q, Text KV
        graph_query = graph_common.unsqueeze(0)
        text_key_value = text_common.unsqueeze(0)
        attn_output, _ = self.cross_attention(graph_query, text_key_value, text_key_value)
        attn_output = attn_output.squeeze(0)

        # Feature concatenation and mapping
        concat_features = torch.cat([graph_features, text_features, attn_output], dim=1)
        # fused_features = self.fusion_layer(concat_features)
        # fused_features = F.relu(self.layer_norm(fused_features))
        
        return concat_features
    
    def apply_cats(self, text_features, graph_features):
        """
        CATS method: Combine contrastive learning and cross-modal attention fusion
        
        Args:
            text_features: Text features, size [B, D_text]
            graph_features: Graph structure features, size [B, D_graph]
            
        Returns:
            fused_features: Merged features, size [B, D_fused]
            contrastive_loss: Contrastive loss
        """
        # Default no contrastive loss
        contrastive_loss = torch.tensor(0.0, device=text_features.device)
        enhanced_text = text_features  # Default use original text features
        
        # Step 1: If contrastive learning is enabled, compute contrastive loss and get enhanced text
        if self.use_cats_contrast:
            enhanced_text, contrastive_loss = self.compute_contrastive_loss(text_features)
        
        # Step 2: If attention mechanism is enabled, apply cross-modal attention; otherwise directly concatenate
        if self.use_cats_attention:
            fused_features = self.apply_cross_attention(text_features, graph_features, enhanced_text)
        else:
            # Directly concatenate features (no attention)
            concat_features = torch.cat([graph_features, enhanced_text], dim=1)
            # Use fusion layer and layer normalization
            # fused_features = self.fusion_layer(concat_features)
            # fused_features = F.relu(self.layer_norm(fused_features))
            fused_features = concat_features

        return fused_features, contrastive_loss





    def cnn_embed(self, tokens, padding_mask=None, is_training = None):
        '''
        CNN sentence embedding
        
        Args:
            tokens: Input token sequence
            padding_mask: Padding mask
            is_training: Whether in training mode
            
        Returns:
            CNN encoded sentence representation
        '''
        x = tokens.long()  # Convert to long integer
        _ = self.embedding(x)  # Word embedding
        _ = _.permute(0, 2, 1)  # Adjust dimension order to (batch, embed_size, seq_len)
        result = []
        for self.cnn in self.cnn_list:
            __ = self.cnn(_)  # Apply convolution
            __ = self.max_pool(__)  # Max pooling
            __ = self.relu(__)  # ReLU activation
            result.append(__.squeeze(dim=2))  # Add to result list
        _ = torch.cat(result, dim=1)  # Concatenate results from different convolution kernels
        return _
        # return self.relu(self.avg_pool(self.embed(tokens.long()).permute(0,2,1)).squeeze())
        # return self.max_pool(self.sentence_encoder(self.embed(tokens.long()).permute(1,0,2), src_key_padding_mask=padding_mask).permute(1,0,2).permute(0,2,1)).squeeze()

    def pool_embed(self, tokens, padding_mask=None, is_training=None):
        """
        Pooling embedding method
        
        Args:
            tokens: Input token sequence
            padding_mask: Padding mask
            is_training: Whether in training mode
            
        Returns:
            Pooled sentence representation
        """
        return self.relu(self.embed_pool(self.embed(tokens.long()).permute(0, 2, 1)).squeeze())

    def RNN_embed(self, tokens, padding_mask=None, is_training=None):
        """
        RNN embedding method
        
        Args:
            tokens: Input token sequence
            padding_mask: Padding mask
            is_training: Whether in training mode
            
        Returns:
            RNN encoded sentence representation
        """
        x = tokens.long()  # Convert to long integer
        _ = self.embed(x)  # Word embedding
        _ = _.permute(1, 0, 2)  # Adjust dimension order to (seq_len, batch, embed_size)
        # h_out = self.rnn(_)
        hidden = self.rnn(_)  # RNN encoding
        # Get representation from last hidden state
        hidden = hidden[1][-1].permute(1, 0, 2).reshape((-1, self.sentence_embedding_dim))
        return hidden

    def LSTM_embed(self, tokens, padding_mask=None, is_training=None):
        """
        LSTM embedding method
        
        Args:
            tokens: Input token sequence
            padding_mask: Padding mask
            is_training: Whether in training mode
            
        Returns:
            LSTM encoded sentence representation
        """
        x = tokens.long()  # Convert to long integer
        _ = self.embed(x)  # Word embedding
        _ = _.permute(1, 0, 2)  # Adjust dimension order
        __, h_out = self.rnn(_)  # LSTM encoding, get hidden state
        # if self._cell in ["lstm", "bi-lstm"]:
        #     h_out = torch.cat([h_out[0], h_out[1]], dim=2)
        h_out = torch.cat([h_out[0], h_out[1]], dim=2)  # Concatenate bidirectional LSTM hidden states
        h_out = h_out.permute(1, 0, 2)  # Adjust dimension order
        h_out = h_out.reshape(-1, h_out.shape[1] * h_out.shape[2])  # Reshape to 2D tensor
        return h_out

    def LSTMATT_embed(self, tokens, padding_mask=None, is_training=None):
        """
        LSTM embedding method with attention
        
        Args:
            tokens: Input token sequence
            padding_mask: Padding mask
            is_training: Whether in training mode
            
        Returns:
            LSTM encoded representation with attention
        """
        input_lstm = self.embed(tokens.long())  # Word embedding
        input_lstm = input_lstm.permute(1,0,2)  # Adjust dimension order to (seq_len, batch, embed_size)
        output, _ = self.lstm(input_lstm)  # LSTM encoding, get hidden state
        output = self.LSTMATT_DP(output)  # Apply dropout
        # sentence_output = torch.cat([_[0], _[1]], dim=2)
        # scc, _ = self.attn(output, output, output, key_padding_mask=padding_mask)
        # scc, _ = self.attn(sentence_output.mean(dim=0).unsqueeze(0), output, output, key_padding_mask=padding_mask)
        # return scc.squeeze()
        # Use multi-head attention mechanism to compute context vector, and get sentence representation by pooling
        return self.embed_pool(self.attn(output, output, output, key_padding_mask=padding_mask)[0].permute(1,0,2).permute(0,2,1)).squeeze()
        # return self.attn(output, output, output, key_padding_mask=padding_mask)[0].mean(dim=0)

    def Transformer_embed(self, tokens, padding_mask=None, is_training=None):
        """
        Transformer embedding method
        
        Args:
            tokens: Input token sequence
            padding_mask: Padding mask
            is_training: Whether in training mode
            
        Returns:
            Transformer encoded sentence representation
        """
        # Use Transformer encoder to encode sentence, and get representation by average pooling
        return self.Trans_encoder(self.embed(tokens.long()).permute(1, 0, 2), src_key_padding_mask=padding_mask).mean(dim=0)

    def gnn_embed(self, tokens, padding_mask=None, is_training=None):
        """
        Graph neural network embedding method
        
        Args:
            tokens: Input token sequence
            padding_mask: Padding mask
            is_training: Whether in training mode
            
        Returns:
            GNN encoded sentence representation
        """
        X = tokens.long()  # Convert to long integer
        NX, EW = self.get_neighbors(X, nb_neighbor=2)  # Get neighbor nodes and edge weights
        NX = NX.long()
        EW = EW.long()
        # NX = input_ids
        # EW = input_ids
        Ra = self.node_embedding(NX)  # Neighbor node embedding
        # edge weight  (bz, seq_len, neighbor_num, 1)
        Ean = self.edge_weight(EW)  # Edge weight
        # neighbor representation  (bz, seq_len, embed_dim)
        if not is_training:
            B, L, N, E = Ra.shape
            # Mn = torch.zeros(size=(B,L,E)).to(Ra.device)
            y = torch.zeros(size=(B,E)).to(Ra.device)
            Rn = self.node_embedding(X)  # Self node embedding
            # self node weight  (bz, seq_len, 1)
            Nn = self.node_weight(X)  # Self node weight
            # Aggregate node features
            for i in range(B):
                tmp = (Ra[i]*Ean[i]).max(dim=1)[0]  # Max pooling aggregate neighbor features
                # Mn[i,:,:] = tmp
                y[i] = ((1-Nn[i])*tmp + Nn[i] * Rn[i]).sum(dim=0)  # Weighted combination of self and neighbor features
            return y
        Mn = (Ra * Ean).max(dim=2)[0]  # Max pooling aggregate neighbor features
        # self representation (bz, seq_len, embed_dim)
        Rn = self.node_embedding(X)  # Self node embedding
        # self node weight  (bz, seq_len, 1)
        Nn = self.node_weight(X)  # Self node weight
        # Aggregate node features
        y = (1 - Nn) * Mn + Nn * Rn  # Weighted combination of self and neighbor features
        return y.sum(dim=1)  # Sum along sequence dimension, get sentence representation


    def get_neighbors(self, x_ids, nb_neighbor=2):
        """
        Get neighbors for each token in input sequence
        
        Args:
            x_ids: Input token sequence
            nb_neighbor: Number of neighbors to consider on each side
            
        Returns:
            neighbours: Neighbor token indices
            ew_ids: Edge weight indices
        """
        B, L = x_ids.size()
        neighbours = torch.zeros(size=(L, B, 2 * nb_neighbor))
        ew_ids = torch.zeros(size=(L, B, 2 * nb_neighbor))
        # pad = [0] * nb_neighbor
        pad = torch.zeros(size=(B, nb_neighbor)).to(x_ids.device)
        # x_ids_ = pad + list(x_ids) + pad
        x_ids_ = torch.cat([pad, x_ids, pad], dim=-1)  # Add padding on both sides of sequence
        for i in range(nb_neighbor, L + nb_neighbor):
            # x = x_ids_[i - nb_neighbor: i] + x_ids_[i + 1: i + nb_neighbor + 1]
            # Get previous nb_neighbor and next nb_neighbor neighbors
            neighbours[i - nb_neighbor, :, :] = torch.cat(
                [x_ids_[:, i - nb_neighbor: i], x_ids_[:, i + 1: i + nb_neighbor + 1]], dim=-1)
        # ew_ids[i-nb_neighbor,:,:] = (x_ids[i-nb_neighbor,:] -1) * self.vocab_size + nb_neighbor[i-nb_neighbor,:,:]
        # Compute edge weight indices
        neighbours = neighbours.permute(1, 0, 2).to(x_ids.device)
        ew_ids = ((x_ids) * (self.vocab_size)).reshape(B, L, 1) + neighbours
        ew_ids[neighbours == 0] = 0  # Set edge weight indices to 0 for padding positions
        return neighbours, ew_ids

    def generate_corrupted(self, emb_subg):
        """Scramble node representation to generate negative samples"""
        idx = torch.randperm(emb_subg.size(0))
        corrupted_emb = emb_subg[idx]
        return corrupted_emb

    
    

    def sample_triplets(self, features, labels):
        """
        Sample triplets (anchor, positive, negative) for triplet loss calculation
        
        Args:
            features: Feature matrix
            labels: Label vector 
            
        Returns:
            anchors, positives, negatives: Triplet features
        """
        anchors, positives, negatives = [], [], []
        for i in range(labels.size(0)):
            label = labels[i]
            pos_mask = (labels == label)
            neg_mask = (labels != label)

            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]
            # Ensure not selecting self as positive sample
            pos_indices = pos_indices[pos_indices != i]

            if len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]
                neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,)).item()]

                anchors.append(i)
                positives.append(pos_idx.item())
                negatives.append(neg_idx.item())

        # If no valid triplets found, return None
        if len(anchors) == 0:
            return None, None, None

        # Convert to tensor and get corresponding features
        anchor_indices = torch.tensor(anchors, device=features.device)
        positive_indices = torch.tensor(positives, device=features.device)
        negative_indices = torch.tensor(negatives, device=features.device)
        
        anchor_features = features[anchor_indices]
        positive_features = features[positive_indices]
        negative_features = features[negative_indices]

        return anchor_features, positive_features, negative_features

    def sample_triplets_with_mining(self, features, labels, mining_strategy='semi-hard'):
        """
        Use hard sample mining strategy to sample triplets (anchor, positive, negative)
        
        Args:
            features: Feature matrix
            labels: Label vector 
            mining_strategy: Mining strategy, 'random', 'semi-hard', or 'hard'
            
        Returns:
            anchors, positives, negatives: Triplet features
        """
        device = features.device
        batch_size = features.size(0)
        
        print(f"Sampling triplets - Feature dimension: {features.shape}, Label dimension: {labels.shape}")
        
        # Check if labels are 2D, if so, need to convert to 1D class index
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            # Try taking max value index of each row as class
            labels = labels.argmax(dim=1)
            print(f"Converting labels to 1D class index: {labels.shape}")
        
        # Handle small batch
        if batch_size < 3:
            return self.sample_triplets(features, labels)
        
        # Random sampling processing
        if mining_strategy == 'random':
            return self.sample_triplets(features, labels)
        
        # Compute distance matrix
        dist_matrix = torch.cdist(features, features, p=2)
        
        anchors, positives, negatives = [], [], []
        
        # Process each sample as anchor
        for anchor_idx in range(batch_size):
            anchor_label = labels[anchor_idx]
            
            # Get positive and negative sample masks
            pos_mask = (labels == anchor_label) & (torch.arange(batch_size, device=device) != anchor_idx)
            neg_mask = (labels != anchor_label)
            
            # Ensure positive and negative samples are available
            if not torch.any(pos_mask) or not torch.any(neg_mask):
                continue
            
            # Get all positive distances
            pos_distances = dist_matrix[anchor_idx][pos_mask]
            
            # 1. First select closest positive sample
            closest_pos_idx = torch.argmin(pos_distances).item()
            pos_indices = torch.where(pos_mask)[0]
            positive_idx = pos_indices[closest_pos_idx].item()
            
            # Compute distance from anchor to positive sample
            ap_distance = dist_matrix[anchor_idx, positive_idx]
            
            # Get all negative distances and indices
            neg_distances = dist_matrix[anchor_idx][neg_mask]
            neg_indices = torch.where(neg_mask)[0]
            
            # Negative index
            negative_idx = None
            
            if mining_strategy == 'hard':
                # Hard mining: Select closest negative sample (most challenging)
                closest_neg_idx = torch.argmin(neg_distances).item()
                negative_idx = neg_indices[closest_neg_idx].item()
                
            elif mining_strategy == 'semi-hard':
                # Semi-hard mining: Select negative sample farther than positive, but as close as possible
                # d(a,n) > d(a,p)
                semi_hard_mask = neg_distances > ap_distance
                
                if torch.any(semi_hard_mask):
                    # Find closest in satisfying negative samples
                    semi_hard_neg_distances = neg_distances[semi_hard_mask]
                    semi_hard_indices = torch.where(semi_hard_mask)[0]
                    closest_semi_hard_idx = torch.argmin(semi_hard_neg_distances).item()
                    negative_idx = neg_indices[semi_hard_indices[closest_semi_hard_idx]].item()
                else:
                    # If no semi-hard samples, select farthest negative sample
                    farthest_neg_idx = torch.argmax(neg_distances).item()
                    negative_idx = neg_indices[farthest_neg_idx].item()
            
            # Add to list
            if negative_idx is not None:
                anchors.append(anchor_idx)
                positives.append(positive_idx)
                negatives.append(negative_idx)
        
        # If no valid triplets found, return None
        if len(anchors) == 0:
            return None, None, None

        # Convert to tensor and get corresponding features
        anchor_indices = torch.tensor(anchors, device=device)
        positive_indices = torch.tensor(positives, device=device)
        negative_indices = torch.tensor(negatives, device=device)
        
        anchor_features = features[anchor_indices]
        positive_features = features[positive_indices]
        negative_features = features[negative_indices]

        return anchor_features, positive_features, negative_features

    def _loss(self, preds, labels, norm_loss, preds_synthetic=None, labels_synthetic=None):
        """
        Compute loss function
        
        Args:
            preds: Predicted values
            labels: Labels
            norm_loss: Loss weight
            
        Returns:
            total_loss: Total loss
        """
        # Compute classification loss
        if self.sigmoid_loss:
            print("Using sigmoid loss function")
            norm_loss = norm_loss.unsqueeze(1)
            classification_loss = torch.nn.BCEWithLogitsLoss(weight=norm_loss, reduction='sum')(preds, labels)
        else:
            _ls = torch.nn.CrossEntropyLoss(reduction='none')(preds, labels)
            classification_loss = (norm_loss * _ls).sum()

        # Initialize total loss
        total_loss = classification_loss
        
        # Add contrastive loss (if exists)
        if hasattr(self, 'contrastive_loss') and self.training:
            total_loss = total_loss + self.contrastive_weight * self.contrastive_loss

        # Compute triplet loss - Modify this part to handle label format code
        if self.training and self.use_triplet_loss:
            features = self.last_fused_features
            
            # Debug information
            print(f"Feature dimension: {features.shape}, Label dimension: {labels.shape}")
            
            # Ensure labels_for_triplet is 1D tensor
            if len(labels.shape) > 1 and labels.shape[1] > 1:  # If one-hot format
                if self.sigmoid_loss:
                    # Multi-label case, select first non-zero or max value label
                    labels_for_triplet = labels.argmax(dim=1)
                else:
                    # Already category index
                    labels_for_triplet = labels
            else:
                # Single category label, ensure 1D
                labels_for_triplet = labels.squeeze()
            
            # Ensure label is long integer
            labels_for_triplet = labels_for_triplet.long()
            
            # Debug information
            print(f"Processed triplet label dimension: {labels_for_triplet.shape}")
            
            # Use sample mining or random sampling
            if hasattr(args_global, 'use_hard_mining') and args_global.use_hard_mining:
                mining_strategy = args_global.mining_strategy if hasattr(args_global, 'mining_strategy') else 'semi-hard'
                anchors, positives, negatives = self.sample_triplets_with_mining(features, labels_for_triplet, mining_strategy)
            else:
                anchors, positives, negatives = self.sample_triplets(features, labels_for_triplet)
            
            # Compute triplet loss
            if anchors is not None:
                triplet_loss = self.triplet_loss_fn(anchors, positives, negatives)
                total_loss = total_loss + self.triplet_weight * triplet_loss

        loss_synthetic = 0
        if self.use_smote and preds_synthetic is not None and labels_synthetic is not None:
            if self.sigmoid_loss:
                # For synthetic data, typically no per-sample weighting like norm_loss_real is needed for BCE
                # Ensure labels_synthetic is in the correct one-hot/multi-hot float format for BCEWithLogitsLoss
                # If labels_synthetic are class indices, they need to be converted to one-hot float for BCEWithLogitsLoss
                # Assuming labels_synthetic might be 1D from SMOTE, convert to one-hot if needed for BCE:
                if labels_synthetic.ndim == 1 or labels_synthetic.shape[-1] != preds_synthetic.shape[-1]:
                    pass # Assuming labels_synthetic are already correctly formatted for BCE

                loss_synthetic = torch.nn.BCEWithLogitsLoss(reduction='mean')(preds_synthetic, labels_synthetic.float()) # Use mean reduction for SMOTE part
            else:
                # For CrossEntropyLoss, labels_synthetic should be 1D tensor of class indices
                # Ensure labels_synthetic is 1D Long tensor
                loss_synthetic = torch.nn.CrossEntropyLoss(reduction='mean')(preds_synthetic, labels_synthetic.long()) # Use mean for SMOTE part
        
        if self.use_smote and preds_synthetic is not None: # Only add if there were synthetic predictions
            total_loss = total_loss + self.smote_loss_weight * loss_synthetic
        return total_loss



    def get_aggregators(self):
        """
        Get aggregator instance list for model building
        
        Returns:
            aggregators: Aggregator list
            num_param: Total aggregator parameter count
        """
        num_param = 0
        aggregators = []
        for l in range(self.num_layers):
            # Create aggregator instance for each layer
            aggr = self.aggregator_cls(
                    *self.dims_weight[l],
                    dropout=self.dropout,
                    act=self.act_layer[l],
                    order=self.order_layer[l],
                    aggr=self.aggr_layer[l],
                    bias=self.bias_layer[l],
                    mulhead=self.mulhead,
            )
            num_param += aggr.num_param  # Accumulate parameter count
            aggregators.append(aggr)  # Add to aggregator list
        return aggregators, num_param


    def predict(self, preds):
        """
        Convert model output to probability
        
        Args:
            preds: Model original output
            
        Returns:
            Converted probability prediction
        """
        return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)


    def train_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph, synthetic_feat_batch=None, synthetic_label_batch=None, current_epoch=0):
        """
        Training step, handle graph data and synthetic samples
        
        Args:
            node_subgraph: Node ID in subgraph
            adj_subgraph: Subgraph adjacency matrix
            norm_loss_subgraph: Loss normalization coefficient
            features: Node features
            synthetic_features: SMOTE generated synthetic features
            synthetic_labels: Labels corresponding to synthetic features
        """
        self.train()
        self.optimizer.zero_grad()
        
        # Forward propagation
        preds, labels_real_converted, preds_synthetic = self(
            node_subgraph, adj_subgraph, current_epoch=current_epoch, is_training=True, synthetic_feat_batch=synthetic_feat_batch)
        
        loss = self._loss(preds, labels_real_converted, norm_loss_subgraph, 
                          preds_synthetic, synthetic_label_batch) 
        
        if isinstance(loss, torch.Tensor): # Ensure loss is a tensor before backward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5) # Corrected function name
            self.optimizer.step()
        else:
            # Handle cases where loss might not be a tensor (e.g., if both real and synthetic parts are None)
            # This might happen if a batch has no valid real or synthetic data, though unlikely with current setup.
            printf("Warning: Loss is not a tensor, skipping backward/step. Check data flow.", style="yellow")
            # Ensure loss is a scalar float for return, or handle appropriately
            if not isinstance(loss, float):
                 loss = 0.0 # Default to 0 if loss calculation was skipped and not a float

        # For evaluation metrics during training (like F1), typically use only real predictions and labels
        return loss, self.predict(preds) if preds is not None else None, labels_real_converted


    def eval_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph, synthetic_features=None, synthetic_labels=None):
        """
        Evaluation step, handle graph data and synthetic samples
        
        Args:
            node_subgraph: Node ID in subgraph
            adj_subgraph: Subgraph adjacency matrix
            norm_loss_subgraph: Loss normalization coefficient
            features: Node features
            synthetic_features: SMOTE generated synthetic features
            synthetic_labels: Labels corresponding to synthetic features
        """
        self.eval()
        with torch.no_grad():
            # Forward propagation
            preds, labels_real_converted, _ = self(
                node_subgraph, adj_subgraph, is_training=False, synthetic_feat_batch=None
            )
            
            loss = self._loss(preds, labels_real_converted, norm_loss_subgraph, 
                              preds_synthetic=None, labels_synthetic=None)
            
            
            
        return loss, self.predict(preds), self.label_full[node_subgraph]