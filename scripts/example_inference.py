import subprocess
import sys
import os
import torch_geometric
import scipy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import pickle
import pandas as pd
from VortexNet import VortexNetUtils
from VortexNet.model import GNN4
from VortexNet.MFData import MFData
from dataParser import DataParser
from plot_field_distribution import plot_field_distribution


weights_name = '20260217_022011'

# Specify the path to the trained weights file
weights_path = "..."    #*.pth
scaler_path_pkl = "..." #*.pth

HIDDEN_CHANNELS = 63
HEADS= 4
PENALTY_WEIGHT=0.025689113034518467
DROPOUT_WEIGHT=0.08504544001617964
LAMBDA=5.8267381175451374e-05
ALPHA=0.009705552281881635
HOP=10
MAX_PHY_LOSS=0.4175220124520909


# fixed hyperparameters
noise = 0.03
CLIP_VALUE = 30
node_in_channels = 11
edge_in_channels = 1
out_channels = 1
device = 'cuda'
REF_RE = 10**7
Ref_Length = 0.435762
random_seeds = 21
model = GNN4(node_in_channels, edge_in_channels, HIDDEN_CHANNELS, out_channels, num_coarse=None, num_fine=None, dropout_rate=DROPOUT_WEIGHT, HEADS = HEADS, ALPHA=ALPHA, HOP=HOP ).to(device)

model.load_state_dict(torch.load(weights_path))

print("***Model loaded from \n {}".format(weights_path))


# Load and process data using DataParser
directory_path = '...'

# DataParser configuration parameters (matching training configuration)
root_chord_ref = 0.6536436
apex_x, apex_y, apex_z = 0.0, 0.0, 0.0

# Create DataParser instance
parser = DataParser(
    directory_path=directory_path,
    ref_re=REF_RE,
    ref_length=Ref_Length,
    root_chord_ref=root_chord_ref,
    apex_x=apex_x,
    apex_y=apex_y,
    apex_z=apex_z,
    test_size=0.3,
    random_state=random_seeds,  # Matching random_seeds=21 from original code
    device=device,
    use_robust_scaler=False,  # Using new standardization with RobustScaler
    normalize_coordinates=True
)

# Try to load scaler from training if it exists

scaler_data = None
if os.path.exists(scaler_path_pkl):
    print(f"Loading scaler from {scaler_path_pkl}")
    scaler_data = DataParser.load_scaler(scaler_path_pkl)
    # Apply loaded scaler configuration to parser
    parser.scaler = scaler_data['scaler']
    parser.use_robust_scaler = scaler_data['use_robust_scaler']
    parser.features_to_normalize_indices = scaler_data['features_to_normalize_indices']
    parser.features_to_keep_indices = scaler_data['features_to_keep_indices']
    parser.normalize_coordinates = scaler_data['normalize_coordinates']
    parser.root_chord_ref = scaler_data['root_chord_ref']
    parser.apex_x = scaler_data['apex_x']
    parser.apex_y = scaler_data['apex_y']
    parser.ref_re = scaler_data['ref_re']
    parser.ref_length = scaler_data['ref_length']
    print("Scaler loaded successfully")
else:
    print(f"Scaler file not found at {scaler_path_pkl}, will fit new scaler from data")

# Process data: load, split, prepare, and normalize
print("\nProcessing data with DataParser...")
parser.load_data()
parser.split_data()
parser.prepare_datasets()

# If scaler was loaded, apply it; otherwise normalize_data will fit a new one
if scaler_data is not None:
    # Apply loaded scaler to datasets
    print("Applying loaded scaler to datasets...")
    # Convert each tensor in the list to NumPy arrays
    nn_training_set_np = [parser.to_numpy(data) for data in parser.nn_training_set]
    nn_test_set_np = [parser.to_numpy(data) for data in parser.nn_test_set]
    
    # Apply scaler to training set
    nn_training_set_scaled_np = []
    for data in nn_training_set_np:
        features_to_normalize = data[:, parser.features_to_normalize_indices]
        features_normalized = parser.scaler.transform(features_to_normalize)
        
        if parser.use_robust_scaler:
            features_normalized_processed = features_normalized.copy()
            features_normalized_processed[:, 1:9] = np.tanh(features_normalized[:, 1:9])
        else:
            features_normalized_processed = features_normalized
        
        features_to_keep = data[:, parser.features_to_keep_indices]
        reconstructed = np.hstack([features_normalized_processed, features_to_keep])
        nn_training_set_scaled_np.append(reconstructed)
    
    # Apply scaler to test set
    nn_test_set_scaled_np = []
    for data in nn_test_set_np:
        features_to_normalize = data[:, parser.features_to_normalize_indices]
        features_normalized = parser.scaler.transform(features_to_normalize)
        
        if parser.use_robust_scaler:
            features_normalized_processed = features_normalized.copy()
            features_normalized_processed[:, 1:9] = np.tanh(features_normalized[:, 1:9])
        else:
            features_normalized_processed = features_normalized
        
        features_to_keep = data[:, parser.features_to_keep_indices]
        reconstructed = np.hstack([features_normalized_processed, features_to_keep])
        nn_test_set_scaled_np.append(reconstructed)
    
    # Update Data objects with normalized node features
    for i, data in enumerate(parser.nn_training_set):
        data.x = torch.tensor(nn_training_set_scaled_np[i], device=data.x.device, dtype=torch.float)
    
    for i, data in enumerate(parser.nn_test_set):
        data.x = torch.tensor(nn_test_set_scaled_np[i], device=data.x.device, dtype=torch.float)
else:
    # Fit new scaler and normalize
    parser.normalize_data()

# Get the processed datasets
nn_training_set = parser.nn_training_set
nn_test_set = parser.nn_test_set
train_set = parser.training_set
test_set = parser.test_set
mf_results = parser.mf_results  # Make mf_results available for later cells

# Print summary of the training and test set 
print("\n******Summary of the Training and Test Datasets******")
print(f"Number of training samples: {len(nn_training_set)}")
print(f"Number of test samples: {len(nn_test_set)}")
print(f"Number of features: {nn_training_set[0].num_features}")
print(f"Number of edge features: {nn_training_set[0].num_edge_features}")
print(f"Number of nodes: {nn_training_set[0].num_nodes}")
print(f"Number of edges: {nn_training_set[0].num_edges}")

model.eval()
###############################################################
## Evaluate the model on the test set
test_loss = 0
for data in nn_test_set:
    data = data.to(device)
    with torch.no_grad():
        output = model(data)
        test_loss += F.mse_loss(output, data.y).item()

test_loss /= len(test_set)
print(f'Test loss: {test_loss:.4f}')
################################################################
## Plot the results for a random index
test_index = np.random.randint(len(test_set))
data = nn_test_set[test_index].to(device)

with torch.no_grad():
    output = model(data).cpu().numpy().reshape(-1)
    print(output.shape)
    print(f"Test {test_index} Predicted, AoA: {test_set[test_index].alpha}, Mach: {test_set[test_index].mach} dcp")
    plot_field_distribution(test_set[test_index].vlm_vd, np.array(output), 
                            title=f" ",
                            min=0, max=1.5)
    plot_field_distribution(test_set[test_index].vlm_vd, np.array(test_set[test_index].cfd_data),
                            title=f" ",
                            min=0, max=1.5)
    plot_field_distribution(test_set[test_index].vlm_vd, np.array(test_set[test_index].vlm_data),
                            title=f" ",
                            min=0, max=1.5, )
    print(f"Re: {test_set[test_index].Re:.2e}")

