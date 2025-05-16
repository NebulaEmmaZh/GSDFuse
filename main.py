from globals import *
from method.models import GraphSAINT
from method.minibatch import Minibatch
from utils import *
from metric import *
from method.utils import *

import warnings
warnings.filterwarnings("ignore")
import torch
import time
import json
import numpy as np
from method import smote_sampler

def evaluate_full_batch(model, minibatch, mode='val'):
    """
    Full batch evaluation function: only used for validation and test sets
    
    When calculating F1 scores, relevant root nodes will be masked
    (e.g., nodes belonging to validation/test sets)
    
    Parameters:
        model: Model to be evaluated
        minibatch: Mini-batch generator
        mode: Evaluation mode, options:
            - 'val': Evaluate validation set only
            - 'test': Evaluate test set only
            - 'valtest': Evaluate both validation and test sets
    
    Returns:
        loss: Loss value
        f1mic: Micro-average F1 score
        f1mac: Macro-average F1 score
        acc: Accuracy
        prec: Precision
        rec: Recall
        f1: F1 score
        f1m: Macro F1 score
    """
    # Perform a forward pass to get loss and prediction results
    loss, preds, labels = model.eval_step(*minibatch.one_batch(mode=mode))
    
    # Select target nodes based on mode
    if mode == 'val':
        node_target = [minibatch.node_val]
    elif mode == 'test':
        node_target = [minibatch.node_test]
    else:
        assert mode == 'valtest'
        node_target = [minibatch.node_val, minibatch.node_test]
    
    # Initialize evaluation metric lists
    f1mic, f1mac = [], []
    acc, f1m, prec, rec, f1 = [], [], [], [], []
    
    # Calculate evaluation metrics for each target node set
    for n in node_target:
        # Calculate F1 scores
        f1_scores = calc_f1(to_numpy(labels[n]), to_numpy(preds[n]), model.sigmoid_loss)
        # Calculate other metrics
        accuracy, macro_f1, precision, recall, f1_score = calc_metrics(
            to_numpy(labels[n]), to_numpy(preds[n]), model.sigmoid_loss)
        
        # Collect all metrics
        f1mic.append(f1_scores[0])
        f1mac.append(f1_scores[1])
        acc.append(accuracy)
        prec.append(precision)
        rec.append(recall)
        f1.append(f1_score)
        f1m.append(macro_f1)
    
    # If there's only one target set, return single value instead of list
    f1mic = f1mic[0] if len(f1mic)==1 else f1mic
    f1mac = f1mac[0] if len(f1mac)==1 else f1mac
    acc = acc[0] if len(acc) == 1 else acc
    prec = prec[0] if len(prec) ==1 else prec
    rec = rec[0] if len(rec) == 1 else rec
    f1 = f1[0] if len(f1) ==1 else f1
    f1m = f1m[0] if len(f1m) == 1 else f1m

    # Note: The loss value here is not very accurate, as the loss also includes contributions from training nodes
    # But for validation/testing, we mainly care about accuracy, so the loss issue is not significant
    return loss, f1mic, f1mac, acc, prec, rec, f1, f1m


def prepare(train_data, train_params, arch_gcn):
    """
    Prepare data structures and initialize models and mini-batch processors before actual iterative training begins
    
    Parameters:
        train_data: Training data tuple, containing:
            - adj_full: Adjacency matrix of the complete graph
            - adj_train: Adjacency matrix of the training graph
            - feat_full: Complete node feature matrix
            - class_arr: Class array
            - role: Node role dictionary
        train_params: Training parameters dictionary
        arch_gcn: GCN architecture configuration
    
    Returns:
        model: Model instance for training
        minibatch: Mini-batch generator for training
        minibatch_eval: Mini-batch generator for evaluation
        model_eval: Model instance for evaluation
    """
    # Unpack training data
    adj_full, adj_train, feat_full, class_arr, role = train_data
    
    # Convert adjacency matrices to 32-bit integer type
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    
    # Normalize adjacency matrix of the complete graph
    adj_full_norm = adj_norm(adj_full)
    
    # Get number of classes
    num_classes = class_arr.shape[1]
    
    # Determine final SMOTE parameters (Command-line > config > defaults)
    final_use_smote = False
    if hasattr(args_global, 'use_smote') and args_global.use_smote:
        final_use_smote = True
        final_smote_k_neighbors = args_global.smote_k_neighbors
        final_smote_random_state = args_global.smote_random_state
        final_synthetic_batch_size = args_global.synthetic_batch_size
        # final_smote_loss_weight is used by the model, not directly here for data prep
        printf("Prepare: SMOTE explicitly enabled via command-line argument.", style="yellow")
    else:
        final_use_smote = train_params.get('use_smote', False)
        final_smote_k_neighbors = train_params.get('smote_k_neighbors', 5)
        final_smote_random_state = train_params.get('smote_random_state', 42)
        final_synthetic_batch_size = train_params.get('synthetic_batch_size', 64)
        if final_use_smote:
            printf("Prepare: SMOTE enabled via configuration file.", style="yellow")

    synthetic_feature_pool = None
    synthetic_labels_pool = None

    if final_use_smote:
        printf("SMOTE enabled. Generating synthetic feature pool...", style="yellow")
        train_idx = role['tr']
        train_feat_tokens_np = feat_full[train_idx]
        train_labels_one_hot_np = class_arr[train_idx]
        
        # Convert one-hot labels to 1D for SMOTE
        if train_labels_one_hot_np.ndim > 1 and train_labels_one_hot_np.shape[1] > 1:
            train_labels_1d_np = np.argmax(train_labels_one_hot_np, axis=1)
        else:
            train_labels_1d_np = train_labels_one_hot_np.flatten()


        printf(f"  Number of training samples for SMOTE: {len(train_idx)}")

        # Create a temporary GraphSAINT model for embedding
        # Ensure feat_full and class_arr are in the format GraphSAINT expects for vocab_size and mask creation
        temp_model_for_embedding = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)
        temp_model_for_embedding.eval() # Set to evaluation mode

        # Prepare mask for the training features
        # The mask is generated based on the original feat_full in GraphSAINT.__init__
        # So, we extract the relevant part of the mask for training nodes.
        # Ensure the mask is boolean as expected by some embedding layers.
        train_mask_np_bool = temp_model_for_embedding.mask.cpu().numpy()[train_idx]
        
        train_feat_tokens_tensor = torch.from_numpy(train_feat_tokens_np.astype(np.int64)) # Tokens are usually Long
        train_mask_tensor = torch.from_numpy(train_mask_np_bool)

        printf(f"  Shape of training tokens for embedding: {train_feat_tokens_tensor.shape}")
        printf(f"  Shape of training mask for embedding: {train_mask_tensor.shape}")
        
        # Perform sentence embedding
        with torch.no_grad():
            # Move tensors to the device where temp_model_for_embedding's feat_full resides (which is CPU due to cpu_eval=True)
            device = temp_model_for_embedding.feat_full.device
            embedded_train_features = temp_model_for_embedding.sentence_embed(
                tokens=train_feat_tokens_tensor.to(device),
                padding_mask=train_mask_tensor.to(device),
                is_training=False
            )
        embedded_train_features_np = embedded_train_features.cpu().numpy()
        printf(f"  Shape of embedded training features: {embedded_train_features_np.shape}")

        # Apply SMOTE
        smote_k_neighbors_to_use = final_smote_k_neighbors
        # Ensure k_neighbors is valid for the number of samples in the smallest class
        unique_labels, counts = np.unique(train_labels_1d_np, return_counts=True)
        min_class_count = counts.min() if len(counts)>0 else 0
        
        if min_class_count <= smote_k_neighbors_to_use and min_class_count > 0 : # k_neighbors must be < n_samples in smallest class for imblearn SMOTE
            printf(f"  Adjusting SMOTE k_neighbors from {smote_k_neighbors_to_use} to {max(1,min_class_count -1)} due to small class size ({min_class_count}).", style='yellow')
            smote_k_neighbors_to_use = max(1, min_class_count - 1) # k_neighbors must be at least 1
        
        if min_class_count == 0 or smote_k_neighbors_to_use == 0 : # Cannot apply SMOTE if a class has 0 samples or k_neighbors is 0
             printf(f"  Skipping SMOTE due to min_class_count ({min_class_count}) or k_neighbors ({smote_k_neighbors_to_use}) being zero.", style='yellow')
             synthetic_features_np, synthetic_labels_1d_np = np.array([]).reshape(0, embedded_train_features_np.shape[1]), np.array([])
        else:
            synthetic_features_np, synthetic_labels_1d_np = smote_sampler.apply_smote(
                embedded_train_features_np,
                train_labels_1d_np,
                k_neighbors=smote_k_neighbors_to_use,
                random_state=final_smote_random_state
            )
        printf(f"  Generated synthetic features shape: {synthetic_features_np.shape}, labels shape: {synthetic_labels_1d_np.shape}")

        if synthetic_features_np.shape[0] > 0:
            synthetic_feature_pool = torch.from_numpy(synthetic_features_np).float()
            synthetic_labels_pool = torch.from_numpy(synthetic_labels_1d_np).long() # SMOTE usually returns 1D labels
        else: # Handle case where SMOTE produces no samples
            synthetic_feature_pool = torch.empty((0, embedded_train_features_np.shape[1])).float()
            synthetic_labels_pool = torch.empty((0,)).long()
        printf("SMOTE processing finished.", style="yellow")

    # Create a mutable copy of train_params to pass to Minibatch, reflecting final SMOTE decisions
    effective_train_params_for_minibatch = train_params.copy()
    effective_train_params_for_minibatch['use_smote'] = final_use_smote
    effective_train_params_for_minibatch['synthetic_batch_size'] = final_synthetic_batch_size
    
    minibatch = Minibatch(adj_full_norm, adj_train, role, effective_train_params_for_minibatch, 
                          synthetic_feature_pool=synthetic_feature_pool, 
                          synthetic_labels_pool=synthetic_labels_pool)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)
    
    # Print total number of model parameters
    printf("TOTAL NUM OF PARAMS = {}".format(sum(p.numel() for p in model.parameters())), style="yellow")
    
    # Create mini-batch generator and model for evaluation
    minibatch_eval = Minibatch(adj_full_norm, adj_train, role, train_params, cpu_eval=True)
    model_eval = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)
    
    # If GPU is available, move model to GPU
    if args_global.gpu >= 0:
        model = model.cuda()
        
    return model, minibatch, minibatch_eval, model_eval


def train(train_phases, model, minibatch, minibatch_eval, model_eval, eval_val_every):
    """
    Main function for model training
    
    Parameters:
        train_phases: Training phase configuration
        model: Model for training
        minibatch: Mini-batch generator for training
        minibatch_eval: Mini-batch generator for evaluation
        model_eval: Model for evaluation
        eval_val_every: How many epochs between each validation
    """
    # If not using CPU for evaluation, evaluation mini-batch generator is the same as training
    if not args_global.cpu_eval:
        minibatch_eval = minibatch
        
    epoch_ph_start = 0  # Starting epoch of current phase
    f1mic_best, ep_best = 0, -1  # Record best F1 score and corresponding epoch
    f1_best = 0  # Record best F1 score
    time_train = 0  # Training time statistics
    
    # Set model save path
    dir_saver = '{}/pytorch_models'.format(args_global.dir_log)
    path_saver = '{}/pytorch_models/saved_model_{}.pkl'.format(args_global.dir_log, timestamp)
    early_stop = 0  # Early stopping counter

    # Iterate through each training phase
    for ip, phase in enumerate(train_phases):
        printf('START PHASE {:4d}'.format(ip), style='underline')
        # Set sampler for current phase
        minibatch.set_sampler(phase)
        num_batches = minibatch.num_training_batches()

        # Iterate through each epoch
        for e in range(epoch_ph_start, int(phase['end'])):
            printf('Epoch {:4d}'.format(e), style='bold')
            minibatch.shuffle()  # Shuffle training data
            
            # Initialize training metric lists
            l_loss_tr, l_f1mic_tr, l_f1mac_tr = [], [], []
            l_acc_tr, l_f1m_tr, l_prec_tr, l_rec_tr, l_f1_tr = [], [], [], [], []
            time_train_ep = 0  # Training time for current epoch

            # Train for one epoch
            while not minibatch.end():
                t1 = time.time()
                # Perform one training step
                # loss_train, preds_train, labels_train = model.train_step(*minibatch.one_batch(mode='train'), current_epoch=e)
                loss_train, preds_train, labels_train = model.train_step(*minibatch.one_batch(mode='train'), current_epoch=e)
                time_train_ep += time.time() - t1

                # Periodically evaluate training performance
                if not minibatch.batch_num % args_global.eval_train_every:
                    # Calculate various evaluation metrics
                    f1_mic, f1_mac = calc_f1(to_numpy(labels_train), to_numpy(preds_train), model.sigmoid_loss)
                    accuracy, macro_f1, precision, recall, f1_score = calc_metrics(
                        to_numpy(labels_train), to_numpy(preds_train), model.sigmoid_loss)
                    
                    # Record training metrics
                    l_loss_tr.append(loss_train)
                    l_f1mic_tr.append(f1_mic)
                    l_f1mac_tr.append(f1_mac)
                    l_acc_tr.append(accuracy)
                    l_prec_tr.append(precision)
                    l_rec_tr.append(recall)
                    l_f1_tr.append(f1_score)
                    l_f1m_tr.append(macro_f1)

            # Periodically perform validation evaluation
            if (e+1) % eval_val_every == 0:
                # Handle CPU evaluation case
                if args_global.cpu_eval:
                    torch.save(model.state_dict(), 'tmp.pkl')
                    model_eval.load_state_dict(torch.load('tmp.pkl', map_location=lambda storage, loc: storage))
                else:
                    model_eval = model

                # Evaluate on validation set
                loss_val, f1mic_val, f1mac_val, acc_val, prec_val, rec_val, f1_val, f1m_val = evaluate_full_batch(
                    model_eval, minibatch_eval, mode='val')

                # Print training and validation results
                printf(' TRAIN (Ep avg): loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\tacc = {:.4f}\tprec = {:.4f}\trec = {:.4f}\tf1 = {:.4f}\tf1m = {:.4f}\ttrain time = {:.4f} sec'
                    .format(f_mean(l_loss_tr), f_mean(l_f1mic_tr), f_mean(l_f1mac_tr), f_mean(l_acc_tr),
                            f_mean(l_prec_tr), f_mean(l_rec_tr), f_mean(l_f1_tr), f_mean(l_f1m_tr), time_train_ep))
                printf(' VALIDATION:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\tacc = {:.4f}\tprec = {:.4f}\trec = {:.4f}\tf1 = {:.4f}\tf1m = {:.4f}'
                    .format(loss_val, f1mic_val, f1mac_val, acc_val, prec_val, rec_val, f1_val, f1m_val), style='yellow')

                # Model saving logic
                if f1_val > f1_best:  # If a better F1 score is obtained
                    f1_best = f1_val
                    ep_best = e
                    # Create save directory (if it doesn't exist)
                    if not os.path.exists(dir_saver):
                        os.makedirs(dir_saver)
                    printf('  Saving model ...', style='yellow')
                    torch.save(model.state_dict(), path_saver)
                    early_stop = 0  # Reset early stopping counter
                else:
                    early_stop += 1  # Increase early stopping counter

                # Early stopping check
                if early_stop >= 20:
                    print("     Early Stop   ")
                    break

            time_train += time_train_ep
        epoch_ph_start = int(phase['end'])

    # Evaluation after training is complete
    printf("Optimization Finished!", style="yellow")
    if ep_best >= 0:
        # Load best model
        if args_global.cpu_eval:
            model_eval.load_state_dict(torch.load(path_saver, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load(path_saver))
            model_eval = model
        printf('  Restoring model ...', style='yellow')

    # Perform final evaluation on validation and test sets
    loss, f1mic_both, f1mac_both, acc_both, prec_both, rec_both, f1_both, f1m_both = evaluate_full_batch(
        model_eval, minibatch_eval, mode='valtest')
    
    # Separate validation and test results
    f1mic_val, f1mic_test = f1mic_both
    f1mac_val, f1mac_test = f1mac_both
    acc_val, acc_test = acc_both
    prec_val, prec_test = prec_both
    rec_val, rec_test = rec_both
    f1_val, f1_test = f1_both
    f1m_val, f1m_test = f1m_both

    # Print final results
    printf("Full validation (Epoch {:4d}): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\tAcc = {:.4f}\tPrec = {:.4f}\tRec = {:.4f}\tF1 = {:.4f}\tF1m = {:.4f}"
        .format(ep_best, f1mic_val, f1mac_val, acc_val, prec_val, rec_val, f1_val, f1m_val), style='red')
    printf("Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}\tAcc = {:.4f}\tPrec = {:.4f}\tRec = {:.4f}\tF1 = {:.4f}\tF1m = {:.4f}"
        .format(f1mic_test, f1mac_test, acc_test, prec_test, rec_test, f1_test, f1m_test), style='red')
    printf("Total training time: {:6.2f} sec".format(time_train), style='red')

    return f1mic_test, f1mac_test, acc_test, prec_test, rec_test, f1_test, f1m_test


if __name__ == '__main__':
    # 1. Create log directory
    log_dir = log_dir(args_global.train_config, args_global.data_prefix, git_branch, git_rev, timestamp)
    result_json_path = log_dir + "result.json"
    
    # 2. Parse configuration and prepare training data
    train_params, train_phases, train_data, arch_gcn = parse_n_prepare(args_global)
    
    # 3. Set validation evaluation frequency
    if 'eval_val_every' not in train_params:
        train_params['eval_val_every'] = EVAL_VAL_EVERY_EP
    
    # 4. Initialize result lists for storing metrics of multiple experiments
    f1mic_tests, f1mac_tests, acc_tests, prec_tests, rec_tests, f1_tests, f1m_tests = [], [], [], [], [], [], []
    
    # 5. Repeat experiments multiple times to obtain statistical significance
    for _ in range(args_global.repeat_time):
        # 5.1 Prepare model and data batches
        model, minibatch, minibatch_eval, model_eval = prepare(train_data, train_params, arch_gcn)
        # 5.2 Train model and get test results
        f1mic_test, f1mac_test, acc_test, prec_test, rec_test, f1_test, f1m_test = train(
            train_phases, model, minibatch, minibatch_eval, model_eval, train_params['eval_val_every'])
        # 5.3 Save results from this experiment
        f1mic_tests.append(f1mic_test)
        f1mac_tests.append(f1mac_test)
        acc_tests.append(acc_test)
        prec_tests.append(prec_test)
        rec_tests.append(rec_test)
        f1_tests.append(f1_test)
        f1m_tests.append(f1m_test)

    # 6. Save statistical results from multiple experiments to JSON file
    json.dump({
        "repeat_time": args_global.repeat_time,
        "f1mac": f_mean(f1mac_tests), "f1mac_std": np.std(f1mac_tests),
        "f1mic": f_mean(f1mic_tests), "f1mic_std": np.std(f1mic_tests),
        "acc": f_mean(acc_tests), "acc_std": np.std(acc_tests),
        "prec": f_mean(prec_tests), "prec_std": np.std(prec_tests),
        "rec": f_mean(rec_tests), "rec_std": np.std(rec_tests),
        "f1": f_mean(f1_tests), "f1_std": np.std(f1_tests),
        "f1m": f_mean(f1m_tests), "f1m_std": np.std(f1m_tests)
    }, open(result_json_path, "w", encoding="utf-8"))

