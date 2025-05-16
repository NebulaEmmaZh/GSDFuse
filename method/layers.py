import torch
from torch import nn
import scipy.sparse as sp
from .kan import KANLinear as effKanLayer
from torch_geometric.nn import APPNP
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GINConv


F_ACT = {'relu': nn.ReLU(),
         'I': lambda x:x}

"""
NOTE
    For the various GNN layers, we optionally support batch normalization. Yet, due to the
    non-IID nature of GNN samplers (whether it is graph-sampling or layer sampling based),
    we may need some modification to the standard batch-norm layer operations to achieve
    optimal accuracy on graphs.

    The study of optimal GNN-based batch-norm is out-of-scope for the current version of
    GraphSAINT. So as a compromise, we provide multiple implementations of batch-norm
    layer which can be optionally inserted at the output of the GNN layers.

Specifically, we have the various choices for the <bias> field in the layer classes
    'bias'          means no batch-norm is applied. Only add the bias to the hidden
                    features of the GNN layer.
    'norm'          means we calculate the mean and variance of the GNN hidden features,
                    and then scale the hidden features manually by the mean and variance.
                    In this case, we need explicitly create the 'offset' and 'scale' params.
    'norm-nn'       means we use the torch.nn.BatchNorm1d layer implemented by torch.
                    In this case, no need to explicitly maintain the BN internal params.
"""


class HighOrderAggregator(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0., act='relu', \
            order=1, aggr='mean', bias='norm-nn', **kwargs):
        """
        Implementation of a higher-order graph aggregation layer, 
        combining ideas from GraphSAGE-mean and MixHop.
        
        Core idea:
        - Defines "order": A k-order layer aggregates information from 0-hop to k-hop neighbors.
        - Operation is approximately: X W_0 [+] A X W_1 [+] ... [+] A^k X W_k
          where [+] is an aggregation op (e.g., sum or concat), A is adj matrix, X is node features.
        
        Special cases:
        - Order = 0: Equivalent to a standard MLP layer.
        - Order = 1: Equivalent to a standard GraphSAGE layer.
        
        Args:
            dim_in      Input node feature dimension.
            dim_out     Output node feature dimension.
            dropout     Dropout rate for weight matrices W_0 to W_k.
            act         Activation function (see F_ACT dict at top of file).
            order       Order of the layer.
            aggr        Aggregation method:
                         - 'mean': Sum features from different hops.
                         - 'concat': Concatenate features from different hops.
            bias        Bias type:
                         - 'bias': Apply bias vector to features of each hop.
                         - 'norm': Manual batch-normalization on output features.
                         - 'norm-nn': Use PyTorch's BatchNorm1d layer.
        """
        super(HighOrderAggregator,self).__init__()
        assert bias in ['bias', 'norm', 'norm-nn']
        self.order, self.aggr = order, aggr
        self.act, self.bias = F_ACT[act], bias
        self.dropout = dropout
        self.f_lin, self.f_bias = [], []
        self.offset, self.scale = [], []
        self.num_param = 0
        for o in range(self.order + 1):
            self.f_lin.append(nn.Linear(dim_in, dim_out, bias=False))
            nn.init.xavier_uniform_(self.f_lin[-1].weight)
            self.f_bias.append(nn.Parameter(torch.zeros(dim_out)))
            self.num_param += dim_in * dim_out
            self.num_param += dim_out
            self.offset.append(nn.Parameter(torch.zeros(dim_out)))
            self.scale.append(nn.Parameter(torch.ones(dim_out)))
            if self.bias == 'norm' or self.bias == 'norm-nn':
                self.num_param += 2 * dim_out
        self.f_lin = nn.ModuleList(self.f_lin)
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.params = nn.ParameterList(self.f_bias + self.offset + self.scale)
        self.f_bias = self.params[:self.order + 1]
        if self.bias == 'norm':
            self.offset = self.params[self.order + 1 : 2 * self.order + 2]
            self.scale = self.params[2 * self.order + 2 : ]
        elif self.bias == 'norm-nn':
            final_dim_out = dim_out * ((aggr=='concat') * (order + 1) + (aggr=='mean'))
            self.f_norm = nn.BatchNorm1d(final_dim_out, eps=1e-9, track_running_stats=True)
        self.num_param = int(self.num_param)

    def _spmm(self, adj_norm, _feat):
        """
        Sparse matrix multiplication: calculates the product of the adjacency matrix and the feature matrix
        
        Args:
            adj_norm: Normalized adjacency matrix (sparse format)
            _feat: Node feature matrix
            
        Returns:
            The result of sparse matrix multiplication, equivalent to aggregating neighbor features for each node
        """
        return torch.sparse.mm(adj_norm, _feat)

    def _f_feat_trans(self, _feat, _id):
        """
        Linearly transform features, apply activation and normalization if needed.
        
        Args:
            _feat       Input features.
            _id         Which linear transformation to use (corresponding hop).
            
        Returns:
            Transformed features.
        """
        feat = self.act(self.f_lin[_id](_feat) + self.f_bias[_id])
        if self.bias == 'norm':
            mean = feat.mean(dim=1).view(feat.shape[0],1)
            var = feat.var(dim=1, unbiased=False).view(feat.shape[0], 1) + 1e-9
            feat_out = (feat - mean) * self.scale[_id] * torch.rsqrt(var) + self.offset[_id]
        else:
            feat_out = feat
        return feat_out

    def forward(self, inputs):
        """
        Forward propagation function.
        
        Args:
            inputs      Tuple or list with two elements:
                        1. adj_norm: Normalized adjacency matrix of the subgraph.
                        2. feat_in: 2D matrix of input node features.
                        
        Returns:
            adj_norm    Same as input (to support nn.Sequential).
            feat_out    2D matrix of output node features.
        """
        adj_norm, feat_in = inputs
        feat_in = self.f_dropout(feat_in)
        feat_hop = [feat_in]
        for o in range(self.order):
            feat_hop.append(self._spmm(adj_norm, feat_hop[-1]))
        feat_partial = [self._f_feat_trans(ft, idf) for idf, ft in enumerate(feat_hop)]
        
        if self.aggr == 'mean':
            feat_out = feat_partial[0]
            for o in range(len(feat_partial) - 1):
                feat_out += feat_partial[o + 1]
        elif self.aggr == 'concat':
            feat_out = torch.cat(feat_partial, 1)
        else:
            raise NotImplementedError
            
        if self.bias == 'norm-nn':
            feat_out = self.f_norm(feat_out)
            
        return adj_norm, feat_out


class JumpingKnowledge(nn.Module):
    def __init__(self, arch_gcn, dim_input_jk, mode=None, **kwargs):
        """
        To be added soon. For now please see the tensorflow version for JK layers
        """
        pass
        # FIXME
        #  JK of my style, may be wrong


class AttentionAggregator(nn.Module):
    """
    This layer follows the design of Graph Attention Network (GAT: https://arxiv.org/abs/1710.10903).
    We extend GAT to higher order as well (see the HighOrderAggregator class above), even though most
    of the time, order-1 layer should be sufficient. The enhancement to SAGE-mean architecture is
    that GAT performs *weighted* aggregation on neighbor features. The edge weight is generated by
    additional learnable MLP layer. Such weight means "attention". GAT proposed multi-head attention
    so that there can be multiple weights for each edge. The k-head attention can be speficied by the
    `mulhead` parameter.

    Note that
     1. In GraphSAINT minibatch training, we remove the softmax normalization across the neighbors.
        Reason: since the minibatch does not see the full neighborhood, softmax does not make much
        sense now. We see significant accuracy improvement by removing the softmax step. See also
        Equations 8 and 9, Appendix C.3 of GraphSAINT (https://arxiv.org/pdf/1907.04931.pdf).
     2. For order > 1, we obtain attention from neighbors from lower order up to higher order.

     Inputs:
        dim_in      int, feature dimension for input nodes
        dim_out     int, feature dimension for output nodes
        dropout     float, dropout on weight matrices W_0 to W_k
        act         str, activation function. See F_ACT at the top of this file
        order       int, see definition in HighOrderAggregator
        aggr        str, if 'mean' then [+] operation adds features of various hops
                         if 'concat' then [+] concatenates features of various hops
        bias        str, if 'bias' then apply a bias vector to features of each hop
                         if 'norm' then perform batch-normalization on output features
        mulhead     int, the number of heads for attention

    Outputs:
        None
    """
    def __init__(self, dim_in, dim_out, dropout=0., act='relu', \
            order=1, aggr='mean', bias='norm', mulhead=1):
        super(AttentionAggregator,self).__init__()
        assert bias in ['bias', 'norm', 'norm-nn']
        self.num_param = 0
        self.mulhead = mulhead
        self.order, self.aggr = order, aggr
        self.act, self.bias = F_ACT[act], bias
        self.att_act = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = dropout
        self._f_lin = []
        self._offset, self._scale = [], []
        self._attention = []
        # mostly we have order = 1 for GAT
        # "+1" since we do batch norm of order-0 and order-1 outputs separately
        for o in range(self.order+1):
            for i in range(self.mulhead):
                self._f_lin.append(nn.Linear(dim_in, int(dim_out / self.mulhead), bias=True))
                nn.init.xavier_uniform_(self._f_lin[-1].weight)
                # _offset and _scale are for 'norm' type of batch norm
                self._offset.append(nn.Parameter(torch.zeros(int(dim_out / self.mulhead))))
                self._scale.append(nn.Parameter(torch.ones(int(dim_out / self.mulhead))))
                self.num_param += dim_in * dim_out / self.mulhead + 2 * dim_out / self.mulhead
                if o < self.order:
                    self._attention.append(nn.Parameter(torch.ones(1, int(dim_out / self.mulhead * 2))))
                    nn.init.xavier_uniform_(self._attention[-1])
                    self.num_param += dim_out / self.mulhead * 2
        self.mods = nn.ModuleList(self._f_lin)
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.params = nn.ParameterList(self._offset + self._scale + self._attention)
        self.f_lin = []
        self.offset, self.scale = [], []
        self.attention = []
        # We need traverse order and mulhead the second time, just because we want to support
        # higher order. Reason: if we have torch parameters in a python list, i.e.:
        #       [nn.Parameter(), nn.Parameter(), ...]
        # PyTorch cannot automically add these parameters into the learnable parameters.
        for o in range(self.order+1):
            self.f_lin.append([])
            self.offset.append([])
            self.scale.append([])
            self.attention.append([])
            for i in range(self.mulhead):
                self.f_lin[-1].append(self.mods[o * self.mulhead + i])
                if self.bias == 'norm':     # not used in 'norm-nn' mode
                    self.offset[-1].append(self.params[o * self.mulhead + i])
                    self.scale[-1].append(self.params[len(self._offset) + o * self.mulhead + i])
                if o < self.order:      # excluding the order-0 part
                    self.attention[-1].append(self.params[len(self._offset) * 2 + o * self.mulhead + i])
        if self.bias == 'norm-nn':
            final_dim_out = dim_out*((aggr=='concat')*(order+1) + (aggr=='mean'))
            self.f_norm = nn.BatchNorm1d(final_dim_out, eps=1e-9, track_running_stats=True)
        self.num_param = int(self.num_param)

    def _spmm(self, adj_norm, _feat):
        """
        Sparse matrix multiplication: calculates the product of the adjacency matrix and the feature matrix
        
        Args:
            adj_norm: Normalized adjacency matrix (sparse format)
            _feat: Node feature matrix
            
        Returns:
            The result of sparse matrix multiplication, equivalent to aggregating neighbor features for each node
        """
        return torch.sparse.mm(adj_norm, _feat)

    def _f_feat_trans(self, _feat, f_lin):
        """
        Apply multi-head linear transformation and activation function to input features
        
        Each attention head has its own linear transformation. This function passes
        the input features through the linear layer of each attention head and applies
        the activation function.
        
        Args:
            _feat: Input features, shape [num_nodes, feature_dimension]
            f_lin: List of linear transformation layers (one for each head)
            
        Returns:
            List of transformed features, each element corresponds to the output of one attention head
        """
        feat_out = []
        for i in range(self.mulhead):
            # For each attention head, apply linear transformation and activation function
            feat_out.append(self.act(f_lin[i](_feat)))
        return feat_out

    def _aggregate_attention(self, adj, feat_neigh, feat_self, attention):
        """
        Aggregate neighbor node features using attention mechanism
        
        Calculation process:
        1. Calculate attention score for self node
        2. Calculate attention score for neighbor nodes
        3. Apply attention scores to the adjacency matrix
        4. Aggregate neighbor features using the weighted adjacency matrix
        
        Args:
            adj: Sparse adjacency matrix
            feat_neigh: Neighbor node features, shape [num_nodes, feature_dimension/num_heads]
            feat_self: Self node features, shape [num_nodes, feature_dimension/num_heads]
            attention: Attention weight parameters
            
        Returns:
            Neighbor feature aggregation weighted by attention
        """
        # Calculate attention score for self node: dot product of the first half of attention parameters and self features
        attention_self = self.att_act(
            attention[:, : feat_self.shape[1]].mm(feat_self.t())
        ).squeeze()
        
        # Calculate attention score for neighbor nodes: dot product of the second half of attention parameters and neighbor features
        attention_neigh = self.att_act(
            attention[:, feat_neigh.shape[1] :].mm(feat_neigh.t())
        ).squeeze()
        
        # Create adjacency matrix with attention weights
        # For each edge (i,j), attention score = self score of node i + neighbor score of node j
        att_adj = torch.sparse.FloatTensor(
            adj._indices(),
            (attention_self[adj._indices()[0]] + attention_neigh[adj._indices()[1]])
            * adj._values(),
            torch.Size(adj.shape),
        )
        
        # Aggregate neighbor features using the weighted adjacency matrix
        return self._spmm(att_adj, feat_neigh)

    def _batch_norm(self, feat):
        """
        Perform batch normalization on features
        
        Normalize features for each order separately:
        1. Calculate the mean and variance of features for each sample
        2. Apply learnable scaling and offset parameters
        
        Args:
            feat: List of features, each element is a feature matrix for an order
            
        Returns:
            List of normalized features
        """
        for i in range(self.order + 1):
            # Calculate the mean of each sample (along the feature dimension)
            mean = feat[i].mean(dim=1).unsqueeze(1)
            # Calculate the variance of each sample (along the feature dimension) and add a numerical stability term
            var = feat[i].var(dim=1, unbiased=False).unsqueeze(1) + 1e-9
            # Apply normalization, scaling, and offset
            # (x - mean) / sqrt(var) * scale + offset
            feat[i] = (feat[i] - mean) * self.scale[i] * torch.rsqrt(var) \
                    + self.offset[i]
        return feat

    def forward(self, inputs):
        """
        Inputs:
            inputs          tuple / list of two elements:
                            1. feat_in: 2D matrix of node features input to the layer
                            2. adj_norm: normalized subgraph adj. Normalization should
                               consider both the node degree and aggregation normalization

        Outputs:
            feat_out        2D matrix of features for output nodes of the layer
            adj_norm        normalized adj same as the input. We have to return it to
                            support nn.Sequential called in models.py
        """
        adj_norm, feat_in = inputs
        feat_in = self.f_dropout(feat_in)
        # generate A^i X
        feat_partial = []
        for o in range(self.order + 1):
            feat_partial.append(self._f_feat_trans(feat_in, self.f_lin[o]))
        for o in range(1,self.order + 1):
            for s in range(o):
                for i in range(self.mulhead):
                    feat_partial[o][i] = self._aggregate_attention(
                        adj_norm,
                        feat_partial[o][i],
                        feat_partial[o - s - 1][i],
                        self.attention[o-1][i],
                    )
        if self.bias == 'norm':
            # normalize per-order, per-head
            for o in range(self.order + 1):
                for i in range(self.mulhead):
                    mean = feat_partial[o][i].mean(dim=1).unsqueeze(1)
                    var = feat_partial[o][i].var(dim=1, unbiased=False).unsqueeze(1) + 1e-9
                    feat_partial[o][i] = (feat_partial[o][i] - mean) \
                                * self.scale[o][i] * torch.rsqrt(var) + self.offset[o][i]

        for o in range(self.order + 1):
            feat_partial[o] = torch.cat(feat_partial[o], 1)
        if self.aggr == 'mean':
            feat_out = feat_partial[0]
            for o in range(len(feat_partial) - 1):
                feat_out += feat_partial[o + 1]
        elif self.aggr == 'concat':
            feat_out = torch.cat(feat_partial, 1)
        else:
            raise NotImplementedError
        if self.bias == 'norm-nn':
            feat_out = self.f_norm(feat_out)
        return adj_norm, feat_out


class GatedAttentionAggregator(nn.Module):
    """
    Gated attentionn network (GaAN: https://arxiv.org/pdf/1803.07294.pdf).
    The general idea of attention is similar to GAT. The main difference is that GaAN adds
    a gated weight for each attention head. Therefore, we can selectively pick important
    heads for better expressive power. Note that this layer is quite expensive to execute,
    since the operations to compute attention are complicated. Therefore, we only support
    order <= 1 (See HighOrderAggregator for definition of order).

    Inputs:
        dim_in      int, feature dimension for input nodes
        dim_out     int, feature dimension for output nodes
        dropout     float, dropout on weight matrices W_0 to W_k
        act         str, activation function. See F_ACT at the top of this file
        order       int, see definition in HighOrderAggregator
        aggr        str, if 'mean' then [+] operation adds features of various hops
                         if 'concat' then [+] concatenates features of various hops
        bias        str, if 'bias' then apply a bias vector to features of each hop
                         if 'norm' then perform batch-normalization on output features
        mulhead     int, the number of heads for attention
        dim_gate    int, output dimension of theta_m during gate value calculation

    Outputs:
        None
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        dropout=0.0,
        act="relu",
        order=1,
        aggr="mean",
        bias="norm",
        mulhead=1,
        dim_gate=64,
    ):
        """
        Initialize Gated Attention Aggregator
        
        Detailed explanation of parameters:
            dim_in: Input feature dimension
            dim_out: Output feature dimension, must be divisible by mulhead
            dropout: Dropout ratio for weight matrices
            act: Activation function type (e.g., 'relu', 'elu')
            order: Aggregation order, indicating the number of neighbor hops considered
            aggr: Aggregation method ('mean' or 'concat')
            bias: Bias type ('bias' or 'norm')
            mulhead: Number of heads for multi-head attention
            dim_gate: Dimension of the intermediate layer in the gating network
        """
        super(GatedAttentionAggregator, self).__init__()
        self.num_param = 0      # TODO: Update parameter count
        self.multi_head = mulhead
        # Ensure the number of multi-head attention heads is reasonable, and the output dimension is divisible by the number of heads
        assert self.multi_head > 0 and dim_out % self.multi_head == 0
        self.order, self.aggr = order, aggr
        # Set activation function and bias type
        self.act, self.bias = F_ACT[act], bias
        # LeakyReLU for activation in attention calculation
        self.att_act = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = dropout
        self.dim_gate = dim_gate
        # Initialize lists to store various network parameters
        self._f_lin = []  # Linear transformation layers
        self._offset, self._scale = [], []  # Offset and scale parameters for batch normalization
        self._attention = []  # Attention weights
        
        # Create corresponding parameters for each order and each attention head
        for i in range(self.order + 1):
            # Batch normalization parameters
            self._offset.append(nn.Parameter(torch.zeros(dim_out)))
            self._scale.append(nn.Parameter(torch.ones(dim_out)))
            
            # Create linear transformation layer and attention weights for each attention head
            for _j in range(self.multi_head):
                # Create linear transformation layer to map input features to output feature space
                self._f_lin.append(
                    nn.Linear(dim_in, int(dim_out / self.multi_head), bias=True)
                )
                # Initialize weights using Xavier initialization
                nn.init.xavier_uniform_(self._f_lin[-1].weight)
                
                # Create attention weights for each order (except the highest order)
                if i < self.order:
                    # Attention weight is a vector of 2 times the feature dimension (for self and neighbors)
                    self._attention.append(
                        nn.Parameter(torch.ones(1, int(dim_out / self.multi_head * 2)))
                    )
                    nn.init.xavier_uniform_(self._attention[-1])
        
        # Gating weight matrix: used to calculate the importance score of each attention head
        # Input dimension is [node_features + pooled_neighbor_features + averaged_neighbor_features]
        self._weight_gate = nn.Parameter(
            torch.ones(dim_in * 2 + dim_gate, self.multi_head)
        )
        nn.init.xavier_uniform_(self._weight_gate)
        
        # Weight matrix for pooling neighbor features: used to generate neighbor representations
        self._weight_pool_gate = nn.Parameter(torch.ones(dim_in, dim_gate))
        nn.init.xavier_uniform_(self._weight_pool_gate)
        
        # Organize all components into PyTorch module lists and parameter lists
        self.mods = nn.ModuleList(self._f_lin)
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.params = nn.ParameterList(
            self._offset
            + self._scale
            + self._attention
            + [self._weight_gate, self._weight_pool_gate]
        )
        
        # Create conveniently accessible references
        self.f_lin = []
        self.offset, self.scale = [], []
        self.attention = []
        
        # Organize parameters for easy access
        for i in range(self.order + 1):
            self.f_lin.append([])
            self.attention.append([])
            self.offset.append(self.params[i])
            self.scale.append(self.params[len(self._offset) + i])
            
            for j in range(self.multi_head):
                self.f_lin[-1].append(self.mods[i * self.multi_head + j])
                if i < self.order:
                    self.attention[-1].append(
                        self.params[len(self._offset) * 2 + i * self.multi_head + j]
                    )
        
        # Set references for gating weight parameters
        self.weight_gate = self.params[-2]
        self.weight_pool_gate = self.params[-1]

    def _spmm(self, adj_norm, _feat):
        """
        Sparse matrix multiplication: calculates the product of the adjacency matrix and the feature matrix
        
        Args:
            adj_norm: Normalized adjacency matrix (sparse format)
            _feat: Node feature matrix
            
        Returns:
            The result of sparse matrix multiplication, equivalent to aggregating neighbor features for each node
        """
        return torch.sparse.mm(adj_norm, _feat)

    def _f_feat_trans(self, _feat, f_lin):
        """
        Apply multi-head linear transformation and activation function to input features
        
        Each attention head has its own linear transformation. This function passes
        the input features through the linear layer of each attention head and applies
        the activation function.
        
        Args:
            _feat: Input features, shape [num_nodes, feature_dimension]
            f_lin: List of linear transformation layers (one for each head)
            
        Returns:
            List of transformed features, each element corresponds to the output of one attention head
        """
        feat_out = []
        for i in range(self.multi_head):
            # For each attention head, apply linear transformation and activation function
            feat_out.append(self.act(f_lin[i](_feat)))
        return feat_out

    def _aggregate_attention(self, adj, feat_neigh, feat_self, attention):
        """
        Aggregate neighbor node features using attention mechanism
        
        Calculation process:
        1. Calculate attention score for self node
        2. Calculate attention score for neighbor nodes
        3. Apply attention scores to the adjacency matrix
        4. Aggregate neighbor features using the weighted adjacency matrix
        
        Args:
            adj: Sparse adjacency matrix
            feat_neigh: Neighbor node features, shape [num_nodes, feature_dimension/num_heads]
            feat_self: Self node features, shape [num_nodes, feature_dimension/num_heads]
            attention: Attention weight parameters
            
        Returns:
            Neighbor feature aggregation weighted by attention
        """
        # Calculate attention score for self node: dot product of the first half of attention parameters and self features
        attention_self = self.att_act(
            attention[:, : feat_self.shape[1]].mm(feat_self.t())
        ).squeeze()
        
        # Calculate attention score for neighbor nodes: dot product of the second half of attention parameters and neighbor features
        attention_neigh = self.att_act(
            attention[:, feat_neigh.shape[1] :].mm(feat_neigh.t())
        ).squeeze()
        
        # Create adjacency matrix with attention weights
        # For each edge (i,j), attention score = self score of node i + neighbor score of node j
        att_adj = torch.sparse.FloatTensor(
            adj._indices(),
            (attention_self[adj._indices()[0]] + attention_neigh[adj._indices()[1]])
            * adj._values(),
            torch.Size(adj.shape),
        )
        
        # Aggregate neighbor features using the weighted adjacency matrix
        return self._spmm(att_adj, feat_neigh)

    def _batch_norm(self, feat):
        """
        Perform batch normalization on features
        
        Normalize features for each order separately:
        1. Calculate the mean and variance of features for each sample
        2. Apply learnable scaling and offset parameters
        
        Args:
            feat: List of features, each element is a feature matrix for an order
            
        Returns:
            List of normalized features
        """
        for i in range(self.order + 1):
            # Calculate the mean of each sample (along the feature dimension)
            mean = feat[i].mean(dim=1).unsqueeze(1)
            # Calculate the variance of each sample (along the feature dimension) and add a numerical stability term
            var = feat[i].var(dim=1, unbiased=False).unsqueeze(1) + 1e-9
            # Apply normalization, scaling, and offset
            # (x - mean) / sqrt(var) * scale + offset
            feat[i] = (feat[i] - mean) * self.scale[i] * torch.rsqrt(var) \
                    + self.offset[i]
        return feat

    def _compute_gate_value(self, adj, feat, adj_sp_csr):
        """
        Calculate gate values to weight the importance of each attention head
        
        Implements formula (3) from the GaAN paper. Gate values are applied before each attention head.
        Notation like zj follows the formula in the paper.
        
        Calculation process:
        1. Calculate projection of node features (zj)
        2. For each node, calculate the max aggregation of its neighbor features
        3. Calculate the mean aggregation of neighbor features
        4. Concatenate node features, neighbor max features, and neighbor mean features
        5. Apply gating weight matrix to calculate the weight for each attention head
        
        Args:
            adj: Sparse adjacency matrix (PyTorch format)
            feat: Node feature matrix
            adj_sp_csr: Sparse adjacency matrix (SciPy CSR format)
            
        Returns:
            Gate value matrix of shape [num_nodes, num_attention_heads]
        """
        # Calculate projection of features (zj in GaAN paper)
        zj = feat.mm(self.weight_pool_gate)
        neigh_zj = []
        
        # Use a loop because PyTorch does not support sparse tensor slicing
        # For each node, find the maximum feature value of its neighbors
        for i in range(adj.shape[0]):
            if adj_sp_csr.indptr[i] < adj_sp_csr.indptr[i + 1]:
                # Get all neighbor indices of the current node
                neigh_zj.append(
                    torch.max(
                        zj[
                            adj_sp_csr.indices[
                                adj_sp_csr.indptr[i] : adj_sp_csr.indptr[i + 1]
                            ]
                        ],
                        0,
                    )[0].unsqueeze(0)
                )
            else:
                # If the node has no neighbors, use a zero vector
                if zj.is_cuda:
                    neigh_zj.append(torch.zeros(1, self.dim_gate).cuda())
                else:
                    neigh_zj.append(torch.zeros(1, self.dim_gate))
        
        # Concatenate the neighbor max features of all nodes
        neigh_zj = torch.cat(neigh_zj, 0)
        
        # Calculate the mean of neighbor features
        neigh_mean = self._spmm(adj, feat)
        
        # Concatenate [node_features, neighbor_max_features, neighbor_mean_features] as input to the gating network
        gate_feat = torch.cat([feat, neigh_zj, neigh_mean], 1)
        
        # Apply gating weight matrix to calculate the gate value for each attention head
        return gate_feat.mm(self.weight_gate)

    def forward(self, inputs):
        """
        Forward propagation function
        
        Calculation process:
        1. Extract normalized adjacency matrix and input features
        2. Apply dropout to input features
        3. Calculate gate values for each attention head
        4. For different orders and different attention heads, calculate feature transformations
        5. For higher-order features, apply attention aggregation and multiply by gate values
        6. Apply batch normalization (if set)
        7. Merge features of different orders according to the aggregation method (mean or concat)
        
        Args:
            inputs: Tuple/list containing two elements:
                   1. adj_norm: Normalized subgraph adjacency matrix. Normalization should consider both node degree and aggregation normalization
                   2. feat_in: 2D matrix of node features input to this layer

        Returns:
            feat_out: 2D matrix of features for output nodes of this layer
            adj_norm: Same normalized adjacency matrix as input (returned to support nn.Sequential called in models.py)
        """
        adj_norm, feat_in = inputs
        
        # Apply dropout to input features
        feat_in = self.f_dropout(feat_in)
        
        # Convert PyTorch sparse tensor to SciPy sparse matrix (for efficient neighbor access)
        adj_norm_cpu = adj_norm.cpu()
        adj_norm_sp_csr = sp.coo_matrix(
            (
                adj_norm_cpu._values().numpy(),
                (
                    adj_norm_cpu._indices()[0].numpy(),
                    adj_norm_cpu._indices()[1].numpy(),
                ),
            ),
            shape=(adj_norm.shape[0], adj_norm.shape[0]),
        ).tocsr()
        
        # Calculate gate values for each attention head
        gate_value = self._compute_gate_value(adj_norm, feat_in, adj_norm_sp_csr)
        
        # Create a list to store features of different orders
        feat_partial = []
        
        # For each order, apply feature transformation
        for i in range(self.order + 1):
            feat_partial.append(self._f_feat_trans(feat_in, self.f_lin[i]))
        
        # For higher-order features (i>=1), apply attention aggregation and multiply by gate values
        for i in range(1, self.order + 1):
            for j in range(i):
                for k in range(self.multi_head):
                    # Aggregate neighbor information using attention mechanism
                    # i is the current order, i-j-1 is the previous order to attend to
                    feat_partial[i][k] = self._aggregate_attention(
                        adj_norm,
                        feat_partial[i][k],
                        feat_partial[i - j - 1][k],
                        self.attention[i - 1][k],
                    )
                    # Apply gate values, assigning different weights to each attention head
                    feat_partial[i][k] *= gate_value[:, k].unsqueeze(1)
        
        # For each order, concatenate the results of multi-head attention
        for i in range(self.order + 1):
            feat_partial[i] = torch.cat(feat_partial[i], 1)
        
        # If normalization is set, apply batch normalization before concatenation
        # Note: If normalized before concatenation, the effect of gate values will disappear
        if self.bias == "norm":
            feat_partial = self._batch_norm(feat_partial)
        
        # Merge features of different orders according to the aggregation method
        if self.aggr == "mean":
            # Sum aggregation: add features of all orders
            feat_out = feat_partial[0]
            for i in range(len(feat_partial) - 1):
                feat_out += feat_partial[i + 1]
        elif self.aggr == "concat":
            # Concatenation aggregation: concatenate features of all orders
            feat_out = torch.cat(feat_partial, 1)
        else:
            raise NotImplementedError
        
        # Return adjacency matrix and output features
        return adj_norm, feat_out

# TGCA
class Encoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Encoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.transfer = effKanLayer(self.in_size, self.out_size)
        # self.acti = nn.ReLU()
        # self.tranformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.out_size, nhead=2), num_layers=1)

    def forward(self, x):
        x = self.transfer(x)
        # x = self.acti(x)
        # x = self.drop(x)
        # x = self.tranformer(x.unsqueeze(0)).squeeze(0)
        return x


class Decoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Decoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.transfer = effKanLayer(self.in_size, self.out_size)
        # self.acti = nn.ReLU()
        # self.tranformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.in_size, nhead=2), num_layers=1)

    def forward(self, x):
        # x = self.tranformer(x.unsqueeze(0)).squeeze(0)
        x = self.transfer(x)
        # x = self.acti(x)
        # x = self.drop(x)
        return x
