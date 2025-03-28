"""
Hyper-parameter sweep for the delta wing 15 geom dataset, results will be saved in the "optuna_results" folder.
Author: Yiren Shen 
Initial Date: 2024-11-03
Modification: 2025-03-10 code clean up

"""
import subprocess
import sys
import os
try:
    import torch_geometric
    print("torch_geometric is already installed.")
except ImportError:
    print("torch_geometric is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
    print("torch_geometric has been installed.")
# add path 

module_path = './VortexNet'
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)    
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VortexNet import GNN4, train_model_k_fold, MFData
from VortexNet.VortexNetUtils import VortexNetUtils
import numpy as np
import pickle
import math
import pandas as pd
from datetime import datetime
import itertools
import json
import random
from sklearn.model_selection import train_test_split
from plotly.offline import plot
from torch_geometric.data import Data
import multiprocessing
from multiprocessing import Queue
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn as nn
import optuna
import matplotlib.pyplot as plt

#############################################################
# USER INPUT 
#############################################################
RANDOM_STATES = 3407    # Random seed
TEST_SIZE = 0.3         # 30% test set
# - reference variables 
REF_RE = 10**7          # Reynolds number
Ref_Length = 0.435762   # chrod length of the wing
# - device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# - Hyperparameter bounds for Optuna
hyperparameter_bounds = {
    "hidden_channels": (3, 64),                 # Integer range 
    "learning_rate": (1e-2, 1),                 # Log-uniform range
    "decay": (1e-5, 1e-2),                      # Log-uniform range
    "HEADS": (2, 8),                            # Integer range
    "penalty_weight": (1e-2, 1e-1),             # Log-uniform range
    "dropout_rate": (0.1, 0.5),                 # Uniform range
    "LAMBDA": (1e-5, 1e-2),                     # Log-uniform range
    "ALPHA": (0.0, 1.0),                        # Uniform range
    "HOP": (3, 20),                             # Integer range 
    "max_phy_loss": (0, 0.5)                    # Uniform range
}
N_TRAILS = 10       # Number of hyperparameter trials
EPOCHS = 40         # Number of training epochs
K_FOLDS = 4         # Number of cross-validation folds
NOISE = 0.02        # Noise level for training
CLIP_VALUE = 30     # Gradient clipping value

# - Data file path 
directory_path = './dataset/train_set'

#############################################################
## Load data 
#############################################################

# load data from pickle file
mf_results = []

for filename in os.listdir(directory_path):
    if filename.endswith(".pkl"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            mf_results.extend(data)  

# Convert to a DataFrame if each entry has is a dictionary
if isinstance(mf_results[0], dict):  
    combined_df = pd.DataFrame(mf_results)
    print(combined_df.head())
else:
    print(f"Total number of runs: {len(mf_results)}")

#############################################################
## Prepare data for training
#############################################################
# Total number of samples
total_samples = len(mf_results)
# Generate indices for all samples
all_indices = list(range(total_samples))
# Randomly split indices into training and test sets (20% test set)
train_indices, test_indices = train_test_split(all_indices, test_size=TEST_SIZE, random_state=RANDOM_STATES)
# Create training and test sets based on the indices
training_set = [mf_results[i] for i in train_indices]
test_set = [mf_results[i] for i in test_indices]
print(f"Size of the training set: {len(training_set)}")
print(f"Size of the test set: {len(test_set)}")

## Prasing data 

# Assemble training and test datasets
training_data = VortexNetUtils.assemble_dataset(training_set, REF_RE)
test_data = VortexNetUtils.assemble_dataset(test_set, REF_RE)

# Print assembled training dataset shapes
print("******Assembled Training Dataset Shapes******")
for key, value in training_data.items():
    print(f"{key} shape : {value.shape}")

# Print assembled test dataset shapes
print("******Assembled Test Dataset Shapes******")
for key, value in test_data.items():
    print(f"{key} shape : {value.shape}")

# Prepare the training dataset
nn_training_set = [
    VortexNetUtils.prepare_dataset_with_standarization(
        cp, vlm_cp, cfd_cp, thickness, curvature_u, curvature_l,
        slope_u, slope_l, ff, aic_matrix, rhs_matrix, dcpsid, 
        factor, chord, rnmax, Ref_Length
    )
    for cp, vlm_cp, cfd_cp, thickness, curvature_u, curvature_l, slope_u, slope_l, ff, aic_matrix, 
        rhs_matrix, dcpsid, factor, chord, rnmax in zip(
        training_data["control_points"], training_data["vlm_dcp"], training_data["cfd_dcp"],
        training_data["vlm_thickness"], training_data["vlm_curvature_u"], training_data["vlm_curvature_l"],
        training_data["vlm_slope_chord_u"], training_data["vlm_slope_chord_l"], 
        training_data["cfd_ff"], training_data["aic_matrices"], training_data["rhs_matrices"],
        training_data["dcpsid_list"], training_data["factor_list"], training_data["chord_list"],
        training_data["rnmax_list"]
    )
]

# Prepare the test dataset
nn_test_set = [
    VortexNetUtils.prepare_dataset_with_standarization(
        cp, vlm_cp, cfd_cp, thickness, curvature_u, curvature_l, slope_u, slope_l, ff,
        aic_matrix, rhs_matrix, dcpsid, factor, chord, rnmax, Ref_Length
    )
    for cp, vlm_cp, cfd_cp, thickness, curvature_u, curvature_l, slope_u, slope_l, ff, aic_matrix, 
        rhs_matrix, dcpsid, factor, chord, rnmax in zip(
        test_data["control_points"], test_data["vlm_dcp"], test_data["cfd_dcp"],
        test_data["vlm_thickness"], test_data["vlm_curvature_u"], test_data["vlm_curvature_l"],
        test_data["vlm_slope_chord_u"], test_data["vlm_slope_chord_l"],
        test_data["cfd_ff"], test_data["aic_matrices"], test_data["rhs_matrices"],
        test_data["dcpsid_list"], test_data["factor_list"], test_data["chord_list"],
        test_data["rnmax_list"]
    )
]

# Print summary of the training and test set 
print("******Summary of the Training and Test Datasets******")
print(f"Number of training samples: {len(nn_training_set)}")
print(f"Number of test samples: {len(nn_test_set)}")
print(f"Number of features: {nn_training_set[0].num_features}")
print(f"Number of edge features: {nn_training_set[0].num_edge_features}")
print(f"Number of nodes: {nn_training_set[0].num_nodes}")
print(f"Number of edges: {nn_training_set[0].num_edges}")


## Normalize the data 
def to_numpy(data):
    if isinstance(data, Data):
        return data.x.cpu().numpy()  
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    else:
        raise TypeError(f"Expected torch_geometric.data.data.Data or torch.Tensor but got {type(data)}")

# Convert each tensor in the list to NumPy arrays
nn_training_set_np = [to_numpy(data) for data in nn_training_set]
nn_test_set_np = [to_numpy(data) for data in nn_test_set]

# Standardize each tensor in the training set
scaler = StandardScaler()
nn_training_set_scaled_np = [scaler.fit_transform(data) for data in nn_training_set_np]

# Standardize each tensor in the test set
nn_test_set_scaled_np = [scaler.transform(data) for data in nn_test_set_np]

# Convert back to tensors
nn_training_set_scaled = [torch.tensor(data, device=nn_training_set[0].x.device) for data in nn_training_set_scaled_np]
nn_test_set_scaled = [torch.tensor(data, device=nn_test_set[0].x.device) for data in nn_test_set_scaled_np]

#############################################################
## Hyper-parameter search
#############################################################

# Objective function for Optuna with external hyperparameter bounds
def objective(trial):

    # define the output file 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"optuna_results/train_output_{timestamp}.txt"
    model_filename = f"optuna_results/model_weights_{timestamp}.pth"
    os.makedirs("optuna_results", exist_ok=True)

    with open(log_filename, "w") as log_file:
        sys.stdout = log_file
        sys.stderr = log_file

        try:
            # Use the bounds defined in hyperparameter_bounds
            hidden_channels = trial.suggest_int("hidden_channels", *hyperparameter_bounds["hidden_channels"])
            learning_rate = trial.suggest_loguniform("learning_rate", *hyperparameter_bounds["learning_rate"])
            decay = trial.suggest_loguniform("decay", *hyperparameter_bounds["decay"])
            HEADS = trial.suggest_int("HEADS", *hyperparameter_bounds["HEADS"])
            penalty_weight = trial.suggest_loguniform("penalty_weight", *hyperparameter_bounds["penalty_weight"])
            #noise_level = trial.suggest_uniform("noise_level", *hyperparameter_bounds["noise_level"])
            dropout_rate = trial.suggest_uniform("dropout_rate", *hyperparameter_bounds["dropout_rate"])
            LAMBDA = trial.suggest_loguniform("LAMBDA", *hyperparameter_bounds["LAMBDA"])
            ALPHA = trial.suggest_uniform("ALPHA", *hyperparameter_bounds["ALPHA"])
            HOP = trial.suggest_int("HOP", *hyperparameter_bounds["HOP"])
            #CLIP_VALUE = trial.suggest_uniform("CLIP_VALUE", *hyperparameter_bounds["CLIP_VALUE"])
            max_phy_loss = trial.suggest_uniform("max_phy_loss", *hyperparameter_bounds["max_phy_loss"])

            # Initialize the model with the selected hyperparameters
            model = GNN4(
                node_in_channels=11, #11 input features, see paper
                edge_in_channels=1,  #1 edge feature, distance
                hidden_channels=hidden_channels,
                out_channels=1,      #1 output feature, DCP
                dropout_rate=dropout_rate,
                HEADS=HEADS,
                ALPHA=ALPHA,
                HOP=HOP
            ).to(DEVICE)

            # Train the model
            val_loss, trained_model, ave_val_loss = train_model_k_fold(
                model,
                nn_training_set,
                k=K_FOLDS,
                learning_rate=learning_rate,
                epochs=EPOCHS,
                clip_value=CLIP_VALUE,
                noise_level=NOISE,
                penalty_weight=penalty_weight,
                device=DEVICE,
                Lambda=LAMBDA,
                decay=decay,
                max_phy_loss=max_phy_loss
            )

            # Evaluate the model on the test set
            test_loss = 0
            trained_model.eval()
            with torch.no_grad():
                for data in nn_test_set:
                    data = data.to(DEVICE)
                    output = trained_model(data)
                    test_loss += nn.functional.mse_loss(output, data.y).item()
            test_loss /= len(nn_test_set)
            
            # save training val loss 
            plt.figure()
            plt.plot(val_loss)
            plt.xlabel('Folds')
            plt.ylabel('Validation Loss')
            plt.title(f"Validation Loss vs. Folds{timestamp}")
            plt.savefig(f'optuna_results/val_loss_{timestamp}.png')
            
            # Save the trained model weights
            torch.save(trained_model.state_dict(), model_filename)
            print(f"Model weights saved to {model_filename}")

            # Log the result
            result = {
                "timestamp": timestamp,
                "test_loss": test_loss,
                "average_validation_loss": ave_val_loss,
                "hyperparameters": trial.params
            }
            os.makedirs("optuna_results", exist_ok=True)
            with open(f"optuna_results/tuning_results_{timestamp}.json", "w") as f:
                json.dump(result, f)

            # release GPU memory
            del model, trained_model, data, output
            torch.cuda.empty_cache()

        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"Training log saved to {log_filename}")
            torch.cuda.empty_cache()
    return test_loss  # Optuna minimizes this value

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRAILS)  
