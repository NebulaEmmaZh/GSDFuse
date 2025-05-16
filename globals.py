# Import Python standard random number generation module, for generating random numbers and random selection
import random

# Import NumPy library, for scientific computing and array operations
import numpy as np
# Import system and time related modules, for file operations, system interaction and timestamp generation
import os, sys, time, datetime
# Import function for expanding user directory (e.g., expanding ~ to user's home directory)
from os.path import expanduser
# Import command line argument parsing module
import argparse

# Import subprocess management module, for executing shell commands and getting output
import subprocess

# Import PyTorch random number generation module
import torch.random

# Get Git version information for experiment traceability
# Execute git command to get short hash value (first 7 characters) of current commit
git_rev = subprocess.Popen("git rev-parse --short HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]
# Execute git command to get current Git branch name, for recording which development branch the experiment was performed on
git_branch = subprocess.Popen("git symbolic-ref --short -q HEAD", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]

# Generate timestamp for marking experiment start time
timestamp = time.time()
# Convert timestamp to human-readable format: year-month-day hour-minute-second
timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')




# Set random seeds for reproducibility
# Fixed random seed value, so that the same random sequence is generated each time the program runs
seed = 42
# Set random seed for NumPy random number generator
np.random.seed(seed)
# Set random seed for PyTorch CPU random number generator
torch.manual_seed(seed)
# Set random seeds for all PyTorch GPU random number generators
torch.cuda.manual_seed_all(seed)
# Set random seed for Python built-in random number generator
random.seed(seed)
# Set random seed for TensorFlow random number generator (commented out, as TensorFlow may not be used in the project)
#tf.set_random_seed(seed)


# !Create command line argument parser, set program description
parser = argparse.ArgumentParser(description="argument for GraphSAINT training")
# Set number of CPU cores for parallel sampling, default is 4
parser.add_argument("--num_cpu_core", default=4, type=int, help="Number of CPU cores for parallel sampling")
# Whether to log device placement information flag, default is False
parser.add_argument("--log_device_placement", default=False, action="store_true", help="Whether to log device placement")
# Training data prefix path, required parameter, used to locate training data files
parser.add_argument("--data_prefix", required=True, type=str, help="prefix identifying training data")
# Log and embedding vector saving base directory, default is "./test"
parser.add_argument("--dir_log", default="./test", type=str, help="base directory for logging and saving embeddings")
# Specify which GPU to use, default is "0"
parser.add_argument("--gpu", default="0", type=str, help="which GPU to use")
# Frequency of evaluating training subgraph accuracy, default is every 1 epoch
parser.add_argument("--eval_train_every", default=1, type=int, help="How often to evaluate training subgraph accuracy")
# Training configuration file path, required parameter, specifies YAML configuration file for training parameters
parser.add_argument("--train_config", required=True, type=str, help="path to the configuration of training (*.yml)")
# Floating point precision type, 's' for single precision (float32), 'd' for double precision (float64), default is 's'
parser.add_argument("--dtype", default="s", type=str, help="d for double, s for single precision floating point")
# Whether to save timeline.json file flag, for performance analysis, default is False
parser.add_argument("--timeline", default=False, action="store_true", help="to save timeline.json or not")
# Whether to save data to TensorBoard flag, for visualizing training process, default is False
parser.add_argument("--tensorboard", default=False, action="store_true", help="to save data to tensorboard or not")
# Whether to distribute the model to two GPUs flag, for model parallel training, default is False
parser.add_argument("--dualGPU", default=False, action="store_true", help="whether to distribute the model to two GPUs")
# Whether to use CPU for evaluation flag, default is False (use GPU evaluation)
parser.add_argument("--cpu_eval", default=False, action="store_true", help="whether to use CPU to do evaluation")
# Pretrained model file path, used to load trained model weights, default is empty string
parser.add_argument("--saved_model_path", default="", type=str, help="path to pretrained model file")
# Number of repeat experiments, for obtaining statistical significance, default is 10 times
parser.add_argument("--repeat_time", default=10, type=int)
# Sentence embedding method, specify how to convert text to vector, default is "cnn"
parser.add_argument("--sentence_embed", default="cnn", type=str)
# Hidden layer dimension, -1 means using default value from configuration file
parser.add_argument("--hidden_dim", type=int, default=-1)
# Whether to not use graph structure flag, default is False (use graph structure)
parser.add_argument("--no_graph", default=False, action="store_true",)
# Whether to use GAU (Gated Attention Unit) flag, default is False
parser.add_argument("--use_gau", default=False, action="store_true", help="whether to use Gated Attention Unit")

parser.add_argument("--use_GAaN", default=False, action="store_true", help="whether to use GraphTransformer")
# Whether to use graph level embedding flag, default is False
parser.add_argument("--use_GIN", default=False, action="store_true", help="whether to use GIN")
# Whether to use TGCA flag, default is False
parser.add_argument("--use_TGCA", default=False, action="store_true", help="whether to use TransGNN")
# Whether to use CATS flag, default is False
parser.add_argument("--use_CATS", default=False, action="store_true", help="whether to use CATS")

parser.add_argument("--use_triplet_loss", default=False, action="store_true", help="whether to use triplet loss")
# Whether to use contrastive learning part in CATS
parser.add_argument("--use_cats_contrast", default=True, action="store_false", dest="use_cats_contrast", help="whether to use contrastive learning in CATS")
# Whether to use attention mechanism in CATS
parser.add_argument("--use_cats_attention", default=True, action="store_false", dest="use_cats_attention", help="whether to use attention in CATS")

# Add SMOTE related command line parameters
parser.add_argument("--use_smote", default=False, action="store_true", help="whether to use SMOTE for oversampling")
parser.add_argument("--smote_k_neighbors", default=5, type=int, help="Number of k-nearest neighbors for SMOTE")
parser.add_argument("--smote_random_state", default=42, type=int, help="Random state seed for SMOTE")
parser.add_argument("--synthetic_batch_size", default=64, type=int, help="Number of synthetic samples per mini-batch for SMOTE")
parser.add_argument("--smote_loss_weight", default=0.5, type=float, help="Weighting factor for SMOTE-generated samples in loss")

# Add hard sample mining switches and strategy parameters
parser.add_argument("--use_hard_mining", default=False, action="store_true", help="whether to use hard mining for triplet loss")
parser.add_argument("--mining_strategy", default="semi-hard", type=str, choices=["random", "semi-hard", "hard"], 
                    help="triplet mining strategy: random, semi-hard, or hard")
# !Parse command line arguments and store in global variable args_global
args_global = parser.parse_args()
# Example command line arguments (commented out), can be used for debugging or default runs
# args_global = parser.parse_args(["--data_prefix=./data/reddit-ac-1-onlyends_with_isolated_bi_10percent_hop1", "--gpu=0","--train_config=./train_config/gat_192_8_with_graph.yml", "--repeat_time=1",
#                                ])
# print(args_global.hidden_dim)

# Set the number of parallel samplers based on CPU cores specified in command line arguments
NUM_PAR_SAMPLER = args_global.num_cpu_core
# Calculate samples per processor (ceiling division)
SAMPLES_PER_PROC = -(-200 // NUM_PAR_SAMPLER) # round up division

# Set validation set evaluation frequency
EVAL_VAL_EVERY_EP = 1       # get accuracy on the validation set every this # epochs


# Logic for automatically selecting available NVIDIA GPUs
gpu_selected = args_global.gpu
if gpu_selected == '-1234':
    # When GPU parameter is '-1234', automatically detect available GPUs
    gpu_stat = subprocess.Popen("nvidia-smi", shell=True, stdout=subprocess.PIPE, universal_newlines=True).communicate()[0]
    # Create a set of possible GPU IDs (assuming system has at most 8 GPUs, numbered 0-7)
    gpu_avail = set([str(i) for i in range(8)])
    # Parse nvidia-smi output to find GPUs running Python processes
    for line in gpu_stat.split('\n'):
        if 'python' in line:  # If the line contains 'python', it means that GPU has Python processes
            if line.split()[1] in gpu_avail:
                # Remove GPU with running Python processes from available GPU set
                gpu_avail.remove(line.split()[1])
            if len(gpu_avail) == 0:
                # If no GPUs are available, set gpu_selected to -2
                gpu_selected = -2
            else:
                # Select the GPU with the smallest ID
                gpu_selected = sorted(list(gpu_avail))[0]
    if gpu_selected == -1:
        # If gpu_selected is still -1, default to GPU 0
        gpu_selected = '0'
    # Update args_global.gpu to the selected GPU ID
    args_global.gpu = int(gpu_selected)

# Set environment variables based on selected GPU
if str(gpu_selected).startswith('nvlink'):
    # If GPU ID starts with 'nvlink', it means using NVIDIA NVLink connected GPUs
    # Extract GPU ID from nvlink string
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_selected).split('nvlink')[1]
elif int(gpu_selected) >= 0:
    # If a valid GPU ID is specified (greater than or equal to 0)
    # Set CUDA device order to PCI bus ID to ensure GPU ID matches nvidia-smi display
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Set visible CUDA devices to the selected GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_selected)
    # Set GPU memory usage proportion to 80% to avoid occupying all GPU memory
    GPU_MEM_FRACTION = 0.8
else:
    # If no valid GPU ID is specified, disable all GPUs and use CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Ensure args_global.gpu is of integer type
args_global.gpu = int(args_global.gpu)

# Global variables and helper functions

# Define lambda function to calculate list average
f_mean = lambda l: sum(l)/len(l)

# Set data type precision based on command line arguments
# 's' for single precision (float32), 'd' for double precision (float64)
DTYPE = "float32" if args_global.dtype == 's' else "float64"      # NOTE: currently not supporting float64 yet
