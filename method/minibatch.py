from globals import *
from utils import *
from graph_samplers import *
from norm_aggr import *
import torch
import scipy.sparse as sp

import numpy as np
import time



def _coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i,v, torch.Size(adj.shape))



class Minibatch:
    """
    Provides minibatches for the trainer or evaluator. This class is responsible for
    calling the proper graph sampler and estimating normalization coefficients.
    """
    def __init__(self, adj_full_norm, adj_train, role, train_params, cpu_eval=False, **kwargs):
        """
        Inputs:
            adj_full_norm       scipy CSR, adj matrix for the full graph (row-normalized)
            adj_train           scipy CSR, adj matrix for the traing graph. Since we are
                                under transductive setting, for any edge in this adj,
                                both end points must be training nodes.
            role                dict, key 'tr' -> list of training node IDs;
                                      key 'va' -> list of validation node IDs;
                                      key 'te' -> list of test node IDs.
            train_params        dict, additional parameters related to training. e.g.,
                                how many subgraphs we want to get to estimate the norm
                                coefficients.
            cpu_eval            bool, whether or not we want to run full-batch evaluation
                                on the CPU.
            **kwargs            additional keyword arguments

        Outputs:
            None
        """
        self.use_cuda = (args_global.gpu >= 0)
        if cpu_eval:
            self.use_cuda=False

        self.current_device = torch.device("cuda" if self.use_cuda else "cpu")

        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])

        self.adj_full_norm = _coo_scipy2torch(adj_full_norm.tocoo())
        self.adj_train = adj_train
        # -----------------------
        # sanity check (optional)
        # -----------------------
        #for role_set in [self.node_val, self.node_test]:
        #    for v in role_set:
        #        assert self.adj_train.indptr[v+1] == self.adj_train.indptr[v]
        #_adj_train_T = sp.csr_matrix.tocsc(self.adj_train)
        #assert np.abs(_adj_train_T.indices - self.adj_train.indices).sum() == 0
        #assert np.abs(_adj_train_T.indptr - self.adj_train.indptr).sum() == 0
        #_adj_full_T = sp.csr_matrix.tocsc(adj_full_norm)
        #assert np.abs(_adj_full_T.indices - adj_full_norm.indices).sum() == 0
        #assert np.abs(_adj_full_T.indptr - adj_full_norm.indptr).sum() == 0
        #printf("SANITY CHECK PASSED", style="yellow")
        if self.use_cuda:
            # now i put everything on GPU. Ideally, full graph adj/feat
            # should be optionally placed on CPU
            self.adj_full_norm = self.adj_full_norm.cuda()

        # below: book-keeping for mini-batch
        self.node_subgraph = None
        self.batch_num = -1

        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        # norm_loss_test is used in full batch evaluation (without sampling).
        # so neighbor features are simply averaged.
        self.norm_loss_test = np.zeros(self.adj_full_norm.shape[0])
        _denom = len(self.node_train) + len(self.node_val) +  len(self.node_test)
        self.norm_loss_test[self.node_train] = 1. / _denom
        self.norm_loss_test[self.node_val] = 1. / _denom
        self.norm_loss_test[self.node_test] = 1. / _denom
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))
        if self.use_cuda:
            self.norm_loss_test = self.norm_loss_test.cuda()
        self.norm_aggr_train = np.zeros(self.adj_train.size)

        self.sample_coverage = train_params['sample_coverage']
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()

        # SMOTE related initializations
        self.synthetic_feature_pool = kwargs.get('synthetic_feature_pool', None)
        self.synthetic_labels_pool = kwargs.get('synthetic_labels_pool', None)
        self.synthetic_batch_size = train_params.get('synthetic_batch_size', 0)
        self.use_smote_sampling = train_params.get('use_smote', False) and \
                                  self.synthetic_feature_pool is not None and \
                                  self.synthetic_labels_pool is not None and \
                                  self.synthetic_feature_pool.shape[0] > 0 and \
                                  self.synthetic_labels_pool.shape[0] > 0 and \
                                  self.synthetic_batch_size > 0

        if self.use_smote_sampling:
            printf(f"Minibatch: SMOTE sampling enabled with batch size {self.synthetic_batch_size} and pool size {self.synthetic_feature_pool.shape[0]}.", style="green")
            # Move synthetic pool to device once
            if self.synthetic_feature_pool is not None:
                self.synthetic_feature_pool = self.synthetic_feature_pool.to(self.current_device)
            if self.synthetic_labels_pool is not None:
                self.synthetic_labels_pool = self.synthetic_labels_pool.to(self.current_device)
        else:
            if train_params.get('use_smote', False): # Only print warning if SMOTE was intended but not activated
                printf("Minibatch: SMOTE sampling NOT enabled (check config, pool size, or batch size).", style="yellow")

    def set_sampler(self, train_phases):
        """
        Pick the proper graph sampler. Run the warm-up phase to estimate
        loss / aggregation normalization coefficients.

        Inputs:
            train_phases       dict, config / params for the graph sampler

        Outputs:
            None
        """
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.method_sample = train_phases['sampler']
        if self.method_sample == 'mrw':
            if 'deg_clip' in train_phases:
                _deg_clip = int(train_phases['deg_clip'])
            else:
                _deg_clip = 100000      # setting this to a large number so essentially there is no clipping in probability
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = mrw_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
                train_phases['size_frontier'],
                _deg_clip,
            )
        elif self.method_sample == 'rw':
            self.size_subg_budget = train_phases['num_root'] * train_phases['depth']
            self.graph_sampler = rw_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
                int(train_phases['num_root']),
                int(train_phases['depth']),
            )
        elif self.method_sample == 'edge':
            self.size_subg_budget = train_phases['size_subg_edge'] * 2
            self.graph_sampler = edge_sampling(
                self.adj_train,
                self.node_train,
                train_phases['size_subg_edge'],
            )
        elif self.method_sample == 'node':
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = node_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        elif self.method_sample == 'full_batch':
            self.size_subg_budget = self.node_train.size
            self.graph_sampler = full_batch_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        elif self.method_sample == "vanilla_node_python":
            self.size_subg_budget = train_phases["size_subgraph"]
            self.graph_sampler = NodeSamplingVanillaPython(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )
        else:
            raise NotImplementedError

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        self.norm_aggr_train = np.zeros(self.adj_train.size).astype(np.float32)

        # -------------------------------------------------------------
        # BELOW: estimation of loss / aggregation normalization factors
        # -------------------------------------------------------------
        # For some special sampler, no need to estimate norm factors, we can calculate
        # the node / edge probabilities directly.
        # However, for integrity of the framework, we follow the same procedure
        # for all samplers:
        #   1. sample enough number of subgraphs
        #   2. update the counter for each node / edge in the training graph
        #   3. estimate norm factor alpha and lambda
        tot_sampled_nodes = 0
        while True:
            self.par_graph_sample('train')
            tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_nodes])
            if tot_sampled_nodes > self.sample_coverage * self.node_train.size:
                break
        print()
        num_subg = len(self.subgraphs_remaining_nodes)
        for i in range(num_subg):
            self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1
            self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1
        assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0
        for v in range(self.adj_train.shape[0]):
            i_s = self.adj_train.indptr[v]
            i_e = self.adj_train.indptr[v + 1]
            val = np.clip(self.norm_loss_train[v] / self.norm_aggr_train[i_s : i_e], 0, 1e4)
            val[np.isnan(val)] = 0.1
            self.norm_aggr_train[i_s : i_e] = val
        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1
        self.norm_loss_train[self.node_val] = 0
        self.norm_loss_train[self.node_test] = 0
        self.norm_loss_train[self.node_train] = num_subg / self.norm_loss_train[self.node_train] / self.node_train.size
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))
        if self.use_cuda:
            self.norm_loss_train = self.norm_loss_train.cuda()

    def par_graph_sample(self,phase):
        """
        Perform graph sampling in parallel. A wrapper function for graph_samplers.py
        """
        t0 = time.time()
        _indptr, _indices, _data, _v, _edge_index = self.graph_sampler.par_sample(phase)
        t1 = time.time()
        print('sampling 200 subgraphs:   time = {:.3f} sec'.format(t1 - t0), end="\r")
        self.subgraphs_remaining_indptr.extend(_indptr)
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)

    def one_batch(self, mode='train'):
        """
        Generate one minibatch for trainer. In the 'train' mode, one minibatch corresponds
        to one subgraph of the training graph. In the 'val' or 'test' mode, one batch
        corresponds to the full graph (i.e., full-batch rather than minibatch evaluation
        for validation / test sets).

        Inputs:
            mode                str, can be 'train', 'val', 'test' or 'valtest'

        Outputs:
            node_subgraph       np array, IDs of the subgraph / full graph nodes
            adj                 scipy CSR, adj matrix of the subgraph / full graph
            norm_loss           np array, loss normalization coefficients. In 'val' or
                                'test' modes, we don't need to normalize, and so the values
                                in this array are all 1.
            synthetic_feat_batch torch.Tensor or None, batch of synthetic features.
            synthetic_label_batch torch.Tensor or None, batch of synthetic labels.
        """
        synthetic_feat_batch = None
        synthetic_label_batch = None

        if mode in ['val','test','valtest']:
            self.node_subgraph = np.arange(self.adj_full_norm.shape[0])
            adj = self.adj_full_norm
            # adj is already on the correct device if use_cuda
        else:
            assert mode == 'train'
            if len(self.subgraphs_remaining_nodes) == 0:
                self.par_graph_sample('train')
                print() # This print might be for the progress bar end

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()
            self.size_subgraph = len(self.node_subgraph)
            adj = sp.csr_matrix(
                (
                    self.subgraphs_remaining_data.pop(),
                    self.subgraphs_remaining_indices.pop(),
                    self.subgraphs_remaining_indptr.pop()),
                    shape=(self.size_subgraph,self.size_subgraph,
                )
            )
            adj_edge_index = self.subgraphs_remaining_edge_index.pop()
            norm_aggr(adj.data, adj_edge_index, self.norm_aggr_train, num_proc=args_global.num_cpu_core)
            adj = adj_norm(adj, deg=self.deg_train[self.node_subgraph])
            adj = _coo_scipy2torch(adj.tocoo())
            if self.use_cuda: # This moves adj to GPU
                adj = adj.cuda()
            self.batch_num += 1

            # SMOTE data sampling for training mode
            if self.use_smote_sampling:
                num_available_synthetic = self.synthetic_feature_pool.shape[0]
                current_synthetic_batch_size = min(self.synthetic_batch_size, num_available_synthetic)

                if current_synthetic_batch_size > 0:
                    # Randomly sample indices without replacement
                    # If pool is smaller than batch_size, it samples all unique available ones.
                    sampled_indices = np.random.choice(num_available_synthetic, size=current_synthetic_batch_size, replace=False)
                    synthetic_feat_batch = self.synthetic_feature_pool[sampled_indices]
                    synthetic_label_batch = self.synthetic_labels_pool[sampled_indices]
                    # Synthetic pool is already on self.current_device from __init__
                else: # Should not happen if use_smote_sampling is true, but as a safeguard
                    printf("Minibatch: SMOTE active but no synthetic samples to draw from or batch size is zero.", style="yellow")


        norm_loss = self.norm_loss_test if mode in ['val','test', 'valtest'] else self.norm_loss_train
        norm_loss = norm_loss[self.node_subgraph] # This should be a Torch tensor
        # Ensure norm_loss is on the same device as adj
        if isinstance(adj, torch.Tensor) and adj.device != norm_loss.device:
             norm_loss = norm_loss.to(adj.device)


        return self.node_subgraph, adj, norm_loss, synthetic_feat_batch, synthetic_label_batch


    def num_training_batches(self):
        return math.ceil(self.node_train.shape[0] / float(self.size_subg_budget))

    def shuffle(self):
        self.node_train = np.random.permutation(self.node_train)
        self.batch_num = -1

    def end(self):
        return (self.batch_num + 1) * self.size_subg_budget >= self.node_train.shape[0]