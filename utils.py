import json
import scipy.sparse
import yaml
import scipy.sparse as sp
from globals import *
from collections import Counter


def load_data(prefix, normalize=True):
    """
    Load various data files located in the 'prefix' directory.
    
    Required files include:
        adj_full.npz        Sparse CSR matrix stored as scipy.sparse.csr_matrix
                           Shape: N×N. Non-zero elements correspond to all edges in the full graph.
                           Edges are included regardless of whether they connect training, validation, or test nodes.
                           For unweighted graphs, all non-zero elements are 1.
                           
        adj_train.npz       Sparse CSR matrix stored as scipy.sparse.csr_matrix
                           Shape: Also N×N. Non-zero elements only correspond to edges connecting two training nodes.
                           Graph samplers only select nodes/edges from adj_train, not adj_full.
                           
        role.json           Dictionary with three keys:
                              'tr': List of indices for all training nodes
                              'va': List of indices for all validation nodes
                              'te': List of indices for all test nodes
                           Note: Nodes in the original data may have string IDs, 
                           which need to be reassigned to numeric IDs (0 to N-1).
                           
        class_map.json      Dictionary of length N. Each key is a node index, each value is:
                           - A binary list of length C (for multi-class classification)
                           - A scalar integer from 0 to C-1 (for single-class classification)
                           
        feats.npz          Numpy array of shape N×F. Row i corresponds to the feature vector for node i.

    Parameters:
        prefix              String, directory containing the graph-related files described above
        normalize          Boolean, whether to normalize node features

    Returns:
        adj_full           Scipy sparse CSR matrix (shape N×N, |E| non-zero elements), adjacency matrix of the full graph
        adj_train          Scipy sparse CSR matrix (shape N×N, |E'| non-zero elements), adjacency matrix of the training graph
        feats              Numpy array (shape N×f), node feature matrix
        class_map          Dictionary, keys are node IDs, values are the class to which the node belongs
        role              Dictionary, keys are 'tr'(train),'va'(validation),'te'(test), values are lists of node IDs for each set
    """
    # Load adjacency matrix of the full graph, convert to boolean type
    adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.bool)
    # Load adjacency matrix of the training graph, convert to boolean type
    adj_train = scipy.sparse.load_npz('./{}/adj_train.npz'.format(prefix)).astype(np.bool)
    # Load role information (train/validation/test set division)
    role = json.load(open('./{}/role.json'.format(prefix)))
    # Load node features
    feats = np.load('./{}/feats.npy'.format(prefix))
    # Load class map and convert keys to integer type
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    # Ensure class map size matches the number of rows in the feature matrix
    assert len(class_map) == feats.shape[0]
    
    # ---- Feature normalization (commented out) ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    # scaler = StandardScaler()
    # scaler.fit(train_feats)
    # feats = scaler.transform(feats)
    # -------------------------
    return adj_full, adj_train, feats, class_map, role


def process_graph_data(adj_full, adj_train, feats, class_map, role):
    """
    Set up vertex attribute mappings, including output classes, train/validation/test masks, and features
    
    Parameters:
        adj_full: Adjacency matrix of the full graph
        adj_train: Adjacency matrix of the training graph
        feats: Node feature matrix
        class_map: Class mapping dictionary
        role: Role information dictionary
    """
    num_vertices = adj_full.shape[0]
    # Handle multi-class case
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    # Handle single-class case
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
    return adj_full, adj_train, feats, class_arr, role


def parse_layer_yml(arch_gcn, dim_input):
    """
    Parse *.yml config file to get GNN structure.
    
    Parameters:
        arch_gcn: GCN architecture configuration dictionary
        dim_input: Input dimension
    """
    # Get number of layers
    num_layers = len(arch_gcn['arch'].split('-'))
    # Set default values, then update through arch_gcn
    bias_layer = [arch_gcn['bias']]*num_layers  # Bias layer configuration
    act_layer = [arch_gcn['act']]*num_layers    # Activation function configuration
    aggr_layer = [arch_gcn['aggr']]*num_layers  # Aggregation method configuration
    dims_layer = [arch_gcn['dim']]*num_layers   # Layer dimension configuration
    order_layer = [int(o) for o in arch_gcn['arch'].split('-')]  # Layer order configuration
    return [dim_input]+dims_layer, order_layer, act_layer, bias_layer, aggr_layer


def parse_n_prepare(flags):
    """
    Parse training configuration and prepare training data
    
    Parameters:
        flags: Command line argument object
    """
    # Load training configuration file
    with open(flags.train_config) as f_train_config:
        train_config = yaml.load(f_train_config, Loader=yaml.FullLoader)
    
    # Set default GCN architecture configuration
    arch_gcn = {
        'dim': -1,           # Layer dimension
        'aggr': 'concat',    # Aggregation method
        'loss': 'softmax',   # Loss function
        'arch': '1',         # Architecture configuration
        'act': 'I',          # Activation function
        'bias': 'norm'       # Bias type
    }
    # Update architecture settings from configuration file
    arch_gcn.update(train_config['network'][0])
    
    # Set default training parameter values
    train_params = {
        'lr': 0.01,              # Learning rate
        'weight_decay': 0.,      # Weight decay
        'norm_loss': True,       # Whether to normalize loss
        'norm_aggr': True,       # Whether to normalize aggregation
        'q_threshold': 50,       # Queue threshold
        'q_offset': 0            # Queue offset
    }
    # Update training parameters from configuration file
    train_params.update(train_config['params'][0])
    train_params["sentence_embed"] = flags.sentence_embed
    train_params["hidden_dim"] = flags.hidden_dim
    train_params["no_graph"] = flags.no_graph
    
    # Get training phase configuration
    train_phases = train_config['phase']
    for ph in train_phases:
        assert 'end' in ph
        assert 'sampler' in ph
        
    print("Loading training data..")
    # Load and process training data
    temp_data = load_data(flags.data_prefix)
    train_data = process_graph_data(*temp_data)
    
    # Process sentence embedding
    if flags.sentence_embed in ["gnn", "cnn", "rnn", "maxpool", "avgpool", "lstmatt", "lstm"]:
        def delete_low_freq(tokens, train_roles):
            """Delete low-frequency words and replace them with UNK token"""
            # Count word frequency in training set
            count = Counter(tokens[train_roles].reshape(-1).tolist())
            count = sorted(count.items(), key=lambda items: items[1])[::-1]
            old2new = {}
            new2old = {}
            unk_tokens = []
            id = 0
            # Process words with frequency >= 10
            for k, v in count:
                if v >= 10:
                    old2new[k] = id
                    new2old[id] = k
                    id += 1
                else:
                    unk_tokens.append(k)
            # Map low-frequency words to UNK token
            unk_id = id
            for k in unk_tokens:
                old2new[k] = unk_id
            # Replace vocabulary
            new_tokens = []
            for i, tokens_ in enumerate(tokens):
                new_tokens.append([])
                for token in tokens_:
                    new_tokens[i].append(old2new.get(token, unk_id))
            return np.array(new_tokens)
            
        # Process low-frequency words
        tmp = delete_low_freq(train_data[2], train_data[4]["tr"])
        train_data = (temp_data[0], train_data[1], tmp, train_data[3], train_data[4])
    print("Done loading training data..")
    return train_params, train_phases, train_data, arch_gcn


def log_dir(f_train_config, prefix, git_branch, git_rev, timestamp):
    """
    Create and return log directory path
    
    Parameters:
        f_train_config: Training configuration file path
        prefix: Data prefix
        git_branch: Git branch name
        git_rev: Git revision number
        timestamp: Timestamp
    """
    import getpass
    # Build log directory path
    log_dir = args_global.dir_log+"/log_train/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}/".format(
            model='graphsaint',
            gitrev=git_rev.strip(),
            ts=timestamp)
    # Create directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Copy training configuration file to log directory
    if f_train_config != '':
        from shutil import copyfile
        copyfile(f_train_config, '{}/{}'.format(log_dir, f_train_config.split('/')[-1]))
    return log_dir


def sess_dir(dims, train_config, prefix, git_branch, git_rev, timestamp):
    """
    Create and return session directory path
    
    Parameters:
        dims: Layer dimension list
        train_config: Training configuration
        prefix: Data prefix
        git_branch: Git branch name
        git_rev: Git revision number
        timestamp: Timestamp
    """
    import getpass
    # Build session directory path
    log_dir = "saved_models/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}-{layer}/".format(
            model='graphsaint',
            gitrev=git_rev.strip(),
            layer='-'.join(dims),
            ts=timestamp)
    # Create directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return sess_dir


def adj_norm(adj, deg=None, sort_indices=True):
    """
    Normalize adjacency matrix using random walk normalization method.
    
    Note: The original GCN paper (kipf) uses symmetric normalization,
    while GraphSAGE and some other variants use random walk normalization.
    We do not use symmetric normalization here as it doesn't seem to help improve accuracy.

    Steps:
        1. Add self-connections to adj --> adj'
        2. Calculate degree matrix D' from adj'
        3. Normalize using D^{-1} x adj'
        
    Parameters:
        adj: Adjacency matrix
        deg: Degree list (optional)
        sort_indices: Whether to reorder indices
    """
    diag_shape = (adj.shape[0], adj.shape[1])
    # Calculate degrees
    D = adj.sum(1).flatten() if deg is None else deg
    # Create normalized diagonal matrix
    norm_diag = sp.dia_matrix((1/D, 0), shape=diag_shape)
    # Normalize adjacency matrix
    adj_norm = norm_diag.dot(adj)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm


##################
# Printing Utility Functions #
#----------------#

# Define ANSI escape sequence color codes
_bcolors = {'header': '\033[95m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m'}


def printf(msg, style=''):
    """
    Print message with color
    
    Parameters:
        msg: Message to print
        style: Print style (color)
    """
    if not style or style == 'black':
        print(msg)
    else:
        print("{color1}{msg}{color2}".format(color1=_bcolors[style], msg=msg, color2='\033[0m'))
