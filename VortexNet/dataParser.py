#%%
import subprocess
import sys
import os
import torch_geometric

# add path 

module_path = 'path/to/vortexnet'
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)    
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from VortexNet import GNN4, train_model_k_fold, MFData
import scipy
from scipy.spatial import distance_matrix
from scipy import stats
import matplotlib.pyplot as plt
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
from torch_geometric.nn import NNConv
import multiprocessing
from multiprocessing import Queue
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn as nn
import seaborn as sns


class DataParser:
    """
    A class to parse and prepare data for neural network training.
    Handles data loading, preprocessing, normalization, and graph construction.
    """
    
    def __init__(self, 
                 directory_path,
                 ref_re=10**7,
                 ref_length=0.435762,
                 root_chord_ref=0.6536436,
                 apex_x=0.0,
                 apex_y=0.0,
                 apex_z=0.0,
                 test_size=0.3,
                 random_state=3407,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 use_robust_scaler=True,
                 normalize_coordinates=True):
        """
        Initialize the DataParser with configuration parameters.
        
        Parameters:
        - directory_path: Path to directory containing pickle files
        - ref_re: Reference Reynolds number for normalization
        - ref_length: Reference length for freestream density calculation
        - root_chord_ref: Reference root chord for coordinate normalization
        - apex_x, apex_y, apex_z: Apex coordinates for coordinate offset
        - test_size: Fraction of data to use for testing
        - random_state: Random seed for train/test split
        - device: Device to use for tensors ('cuda' or 'cpu')
        - use_robust_scaler: If True, use RobustScaler; if False, use StandardScaler
        - normalize_coordinates: If True, normalize coordinates by root_chord_ref and offset by apex
        """
        self.directory_path = directory_path
        self.ref_re = ref_re
        self.ref_length = ref_length
        self.root_chord_ref = root_chord_ref
        self.apex_x = apex_x
        self.apex_y = apex_y
        self.apex_z = apex_z
        self.test_size = test_size
        self.random_state = random_state
        self.device = device
        self.use_robust_scaler = use_robust_scaler
        self.normalize_coordinates = normalize_coordinates
        
        # Will be set after loading data
        self.mf_results = None
        self.training_set = None
        self.test_set = None
        self.training_data = None
        self.test_data = None
        self.nn_training_set = None
        self.nn_test_set = None
        self.scaler = None
        
        # Feature indices for normalization
        self.features_to_normalize_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.features_to_keep_indices = [9, 10]
    
    def load_data(self):
        """Load data from pickle files in the specified directory."""
        import re
        mf_results = []
        # Iterate over each file in the directory
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".pkl"):
                file_path = os.path.join(self.directory_path, filename)
                
                # Extract the sweep number from the filename (e.g., _sweep_65_)
                match = re.search(r'_sweep_(\d+)_', filename)
                if match:
                    sweep_number = int(match.group(1))
                else:
                    sweep_number = None

                # Extract the NACA number from the filename (e.g., _naca_0010)
                match = re.search(r'_naca_(\d+)', filename)
                if match:
                    naca_number = match.group(1).zfill(4)
                    naca_dict = {
                        'm': int(naca_number[0]),
                        'p': int(naca_number[1]),
                        't': int(naca_number[2:]),
                        'chord_length': 1.0
                    }
                else:
                    naca_number = None
                    naca_dict = None
                
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    # Set geom attribute for each data object
                    for d in data:
                        d.geom = [sweep_number, naca_dict]
                    mf_results.extend(data)  # Assuming data is a list, extend to add multiple entries
        
        # Convert to a DataFrame if each entry has a structured format (e.g., dictionary)
        if len(mf_results) > 0 and isinstance(mf_results[0], dict):
            combined_df = pd.DataFrame(mf_results)
            print(combined_df.head())
        else:
            # If it's not dictionary-structured, you may just use the list `all_results`
            print(f"Total number of runs: {len(mf_results)}")
        
        self.mf_results = mf_results
        return mf_results
    
    def split_data(self):
        """Split data into training and test sets."""
        if self.mf_results is None:
            raise ValueError("Data must be loaded first. Call load_data() before split_data().")
        
        # Total number of samples
        total_samples = len(self.mf_results)
        # Generate indices for all samples
        all_indices = list(range(total_samples))
        # Randomly split indices into training and test sets
        train_indices, test_indices = train_test_split(
            all_indices, test_size=self.test_size, random_state=self.random_state
        )
        # Create training and test sets based on the indices
        self.training_set = [self.mf_results[i] for i in train_indices]
        self.test_set = [self.mf_results[i] for i in test_indices]
        # Print count of training set and test set
        print(f"Size of the training set: {len(self.training_set)}")
        print(f"Size of the test set: {len(self.test_set)}")
        
        return self.training_set, self.test_set 

    def assemble_dataset(self, data_set):
        """
        Assemble dataset by extracting control points, DCP, far field conditions, slopes, curvatures, thickness, 
        and CPU/CPL data from the provided dataset.

        Parameters:
        - data_set: List of data objects (training or test set)

        Returns:
        - Dictionary containing assembled arrays for all necessary data components
        """
        control_points = []
        vlm_dcp = []
        cfd_dcp = []
        cfd_ff = []
        vlm_slope_span_u = []
        vlm_slope_chord_u = []
        vlm_slope_span_l = []
        vlm_slope_chord_l = []
        vlm_curvature_u = []
        vlm_curvature_l = []
        vlm_thickness = []
        cfd_cpu = []
        cfd_cpl = []
        aic_matrix = []
        rhs_matrix = []
        DCPSID_list = []
        FACTOR_list = []
        CHORD_list = []
        RNMAX_list = []

        # Process each data object in the dataset
        for j in range(len(data_set)):
            cur_re = data_set[j].Re / self.ref_re
            alpha = data_set[j].alpha
            mach = data_set[j].mach

            # Extract control points
            vd = data_set[j].vlm_vd
            control_point_list = np.column_stack((vd.XC, vd.YC, vd.ZC))
            control_points.append(control_point_list)

            # extract lattice area
            lattice_area = np.array(vd.panel_areas).reshape(1, -1)
            #print(lattice_area)

            # Extract DCP
            vlm_dcp.append(np.array(data_set[j].vlm_data))
            cfd_dcp.append(np.array(data_set[j].cfd_data).reshape(1, -1))
            

            # Extract far field conditions
            current_ff = np.tile([alpha, mach, cur_re], (len(control_point_list), 1))
            cfd_ff.append(current_ff)

            # Extract slopes, curvatures, thickness
            vlm_slope_span_u.append(np.array(data_set[j].spanwise_slope_u).reshape(1, -1))
            vlm_slope_chord_u.append(np.array(data_set[j].chordwise_slope_u).reshape(1, -1))
            vlm_curvature_u.append(np.array(data_set[j].gaussian_curvature_u).reshape(1, -1))
            vlm_slope_span_l.append(np.array(data_set[j].spanwise_slope_l).reshape(1, -1))
            vlm_slope_chord_l.append(np.array(data_set[j].chordwise_slope_l).reshape(1, -1))
            vlm_curvature_l.append(np.array(data_set[j].gaussian_curvature_l).reshape(1, -1))
            vlm_thickness.append(np.array(data_set[j].thickness).reshape(1, -1))

            # Extract CPU and CPL
            cfd_cpu.append(np.array(data_set[j].cfd_cpu).reshape(1, -1))
            cfd_cpl.append(np.array(data_set[j].cfd_cpl).reshape(1, -1))

            # Compute AIC and RHS matrix for VLM
            aic_matrix.append(np.array(data_set[j].vlm_A))
            rhs_matrix.append(np.array(data_set[j].vlm_RHS))    

            # Store additional variables
            DCPSID_list.append(np.array(data_set[j].vlm_DCPSID))
            FACTOR_list.append(np.array(data_set[j].vlm_FACTOR))
            CHORD_list.append(np.array(data_set[j].vlm_CHORD))
            RNMAX_list.append(np.array(data_set[j].vlm_RNMAX))

        # Convert all lists to numpy arrays
        return {
            "control_points": np.array(control_points),
            "vlm_dcp": np.array(vlm_dcp),
            "cfd_dcp": np.array(cfd_dcp),
            "cfd_ff": np.array(cfd_ff),
            "vlm_slope_span_u": np.array(vlm_slope_span_u),
            "vlm_slope_chord_u": np.array(vlm_slope_chord_u),
            "vlm_curvature_u": np.array(vlm_curvature_u),
            "vlm_slope_span_l": np.array(vlm_slope_span_l),
            "vlm_slope_chord_l": np.array(vlm_slope_chord_l),
            "vlm_curvature_l": np.array(vlm_curvature_l),
            "vlm_thickness": np.array(vlm_thickness),
            "cfd_cpu": np.array(cfd_cpu),
            "cfd_cpl": np.array(cfd_cpl),
            "aic_matrices": np.array(aic_matrix),
            "rhs_matrices": np.array(rhs_matrix),
            "dcpsid_list": np.array(DCPSID_list),
            "factor_list": np.array(FACTOR_list),
            "chord_list": np.array(CHORD_list),
            "rnmax_list": np.array(RNMAX_list)
        }

    @staticmethod
    def tanh_standardization(arr):
        """
        Apply tanh standardization to the input array.
        """
        return np.tanh(arr)

    @staticmethod
    def compute_freestream_density(reynolds_number, mach, viscosity, characteristic_length, TInfinity):
        """
        Compute the freestream density using the Reynolds number definition.
        
        """
        gamma = 1.4
        R = 287.05
        speed_of_sound = math.sqrt(gamma * R * TInfinity)
        velocity = mach * speed_of_sound
        density = (reynolds_number * viscosity) / (velocity * characteristic_length)
        return density

    def prepare_dataset_with_standarization(
        self, control_points, vlm_cp, cfd_cp, vlm_thickness, vlm_curvature_u, vlm_curvature_l, vlm_slope_u, vlm_slope_l, ff,
        aic_matrix, rhs_matrix, dcpsid, factor, chord, rnmax):
        """
        Parameters:
        - control_points: Control points for the wing
        - vlm_cp: VLM DCP data
        - cfd_cp: CFD DCP data
        - vlm_thickness: Thickness data
        - vlm_curvature_u: Upper surface curvature data
        - vlm_curvature_l: Lower surface curvature data
        - ff: Far field conditions (AOA, Mach, Re)
        - aic_matrix: AIC matrix from VLM
        - rhs_matrix: RHS matrix from VLM
        - dcpsid: DCPSID list
        - factor: FACTOR list
        - chord: CHORD list
        - rnmax: RNMAX list    

        Returns:
        - PyTorch geometric data object containing node features, edge index, edge attributes, and freestream properties
        """

        NUM_POINTS = control_points.shape[0]
        coordinates = control_points[:, :2]
        
        # Offset by apex and normalize by root_chord_ref if enabled
        if self.normalize_coordinates:
            coordinates = coordinates - np.array([self.apex_x, self.apex_y])
            coordinates = coordinates / self.root_chord_ref

        # Calculate the distance matrix between all pairs of points
        dist_matrix = distance_matrix(coordinates, coordinates)

        # Initialize the list for edges
        edges = []

        # Iterate over each point to find its 4 nearest neighbors
        for i in range(NUM_POINTS):
            # Get indices of the 4 nearest neighbors (excluding the point itself)
            nearest_neighbors = np.argsort(dist_matrix[i])[1:5]
            for neighbor in nearest_neighbors:
                edges.append([i, neighbor])

        # Convert edges to a tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Prepare raw features for normalization
        aoa = ff[:, 0].reshape(-1, 1)
        mach = ff[:, 1].reshape(-1, 1)
        re = ff[:, 2].reshape(-1, 1)
        aoa_rad = aoa * math.pi / 180
        # Use raw features (will be normalized later)
        ff_raw = np.hstack((aoa_rad, mach, re))
        vlm_thickness_raw = vlm_thickness.reshape(-1, 1)
        vlm_curvature_u_raw = vlm_curvature_u.reshape(-1, 1)
        vlm_curvature_l_raw = vlm_curvature_l.reshape(-1, 1)
        vlm_slope_u_raw = vlm_slope_u.reshape(-1, 1)
        vlm_slope_l_raw = vlm_slope_l.reshape(-1, 1)

        # Node features: vlm_cp, raw features (will be normalized), and coordinates
        node_features = np.hstack((
            vlm_cp.reshape(-1, 1),             # Column 0: vlm_cp
            ff_raw,                             # Columns 1-3: ff
            vlm_thickness_raw,                  # Column 4: vlm_thickness
            vlm_curvature_u_raw,                 # Column 5: vlm_curvature_u
            vlm_curvature_l_raw,                 # Column 6: vlm_curvature_l
            vlm_slope_u_raw,                     # Column 7: vlm_slope_u
            vlm_slope_l_raw,                     # Column 8: vlm_slope_l
            coordinates                          # Columns 9-10: coordinates
        ))

        node_features_tensor = torch.tensor(node_features, dtype=torch.float)

        # High fidelity node features
        high_fidelity_node_features_tensor = torch.tensor(cfd_cp, dtype=torch.float)

        # Edge attributes: (we can use distance between nodes as edge attributes)
        edge_attr = []
        for edge in edges:
            node1 = edge[0]
            node2 = edge[1]
            x1, y1 = coordinates[node1, :2]
            x2, y2 = coordinates[node2, :2]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            edge_attr.append([distance])

        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Compute freestream properties
        gamma = 1.4  # Specific heat ratio for air
        R = 287.05   # Specific gas constant for air in J/(kg·K)
        T_inf = 322  # Freestream temperature in K (adjust if necessary)
        P_inf = 101325  # Freestream pressure in Pa (standard atmospheric pressure)
        # Sutherland's constants for air
        mu_ref = 1.71e-5
        T_ref = 273.11
        S = 110.56      # Sutherland's temperature, K
        viscosity = mu_ref * (T_inf / T_ref) ** 1.5 * (T_ref + S) / (T_inf + S)
        mach_inf = ff[:, 1].mean()  # Assuming Mach number is constant per sample
        rho_inf = self.compute_freestream_density(re[0], mach_inf, viscosity, self.ref_length, T_inf)
        a_inf = np.sqrt(gamma * R * T_inf)
        V_inf = mach_inf * a_inf

        # Pack data into a torch geometric data object
        data = Data(
            x=node_features_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=high_fidelity_node_features_tensor.view(-1, 1)
        )

        # Add freestream properties to data object
        data.V_inf = torch.tensor(V_inf, dtype=torch.float)
        data.a_inf = torch.tensor(a_inf, dtype=torch.float)
        data.rho_inf = torch.tensor(rho_inf, dtype=torch.float)

        # Add additional variables as attributes
        data.aic_matrix = torch.tensor(aic_matrix, dtype=torch.float)
        data.rhs_matrix = torch.tensor(rhs_matrix, dtype=torch.float)
        data.dcpsid = torch.tensor(dcpsid, dtype=torch.float)
        data.factor = torch.tensor(factor, dtype=torch.float)
        data.chord = torch.tensor(chord, dtype=torch.float)
        data.rnmax = torch.tensor(rnmax, dtype=torch.float)
        return data

    def prepare_datasets(self):
        """Assemble and prepare training and test datasets."""
        if self.training_set is None or self.test_set is None:
            raise ValueError("Data must be split first. Call split_data() before prepare_datasets().")
        
        # Assemble training and test datasets
        self.training_data = self.assemble_dataset(self.training_set)
        self.test_data = self.assemble_dataset(self.test_set)

        # Print assembled training dataset shapes
        print("******Assembled Training Dataset Shapes******")
        for key, value in self.training_data.items():
            print(f"{key} shape : {value.shape}")

        # Print assembled test dataset shapes
        print("******Assembled Test Dataset Shapes******")
        for key, value in self.test_data.items():
            print(f"{key} shape : {value.shape}")

        # Prepare the training dataset
        self.nn_training_set = [
            self.prepare_dataset_with_standarization(
                cp, vlm_cp, cfd_cp, thickness, curvature_u, curvature_l,
                slope_u, slope_l, ff, aic_matrix, rhs_matrix, dcpsid, 
                factor, chord, rnmax
            )
            for cp, vlm_cp, cfd_cp, thickness, curvature_u, curvature_l, slope_u, slope_l, ff, aic_matrix, 
                rhs_matrix, dcpsid, factor, chord, rnmax in zip(
                self.training_data["control_points"], self.training_data["vlm_dcp"], self.training_data["cfd_dcp"],
                self.training_data["vlm_thickness"], self.training_data["vlm_curvature_u"], self.training_data["vlm_curvature_l"],
                self.training_data["vlm_slope_chord_u"], self.training_data["vlm_slope_chord_l"], 
                self.training_data["cfd_ff"], self.training_data["aic_matrices"], self.training_data["rhs_matrices"],
                self.training_data["dcpsid_list"], self.training_data["factor_list"], self.training_data["chord_list"],
                self.training_data["rnmax_list"]
            )
        ]

        # Prepare the test dataset
        self.nn_test_set = [
            self.prepare_dataset_with_standarization(
                cp, vlm_cp, cfd_cp, thickness, curvature_u, curvature_l, slope_u, slope_l, ff,
                aic_matrix, rhs_matrix, dcpsid, factor, chord, rnmax
            )
            for cp, vlm_cp, cfd_cp, thickness, curvature_u, curvature_l, slope_u, slope_l, ff, aic_matrix, 
                rhs_matrix, dcpsid, factor, chord, rnmax in zip(
                self.test_data["control_points"], self.test_data["vlm_dcp"], self.test_data["cfd_dcp"],
                self.test_data["vlm_thickness"], self.test_data["vlm_curvature_u"], self.test_data["vlm_curvature_l"],
                self.test_data["vlm_slope_chord_u"], self.test_data["vlm_slope_chord_l"],
                self.test_data["cfd_ff"], self.test_data["aic_matrices"], self.test_data["rhs_matrices"],
                self.test_data["dcpsid_list"], self.test_data["factor_list"], self.test_data["chord_list"],
                self.test_data["rnmax_list"]
            )
        ]

        # Print summary of the training and test set 
        print("******Summary of the Training and Test Datasets******")
        print(f"Number of training samples: {len(self.nn_training_set)}")
        print(f"Number of test samples: {len(self.nn_test_set)}")
        print(f"Number of features: {self.nn_training_set[0].num_features}")
        print(f"Number of edge features: {self.nn_training_set[0].num_edge_features}")
        print(f"Number of nodes: {self.nn_training_set[0].num_nodes}")
        print(f"Number of edges: {self.nn_training_set[0].num_edges}")
        
        return self.nn_training_set, self.nn_test_set

    @staticmethod
    def to_numpy(data):
        """Convert Data object or tensor to numpy array."""
        if isinstance(data, Data):
            return data.x.cpu().numpy()
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        else:
            raise TypeError(f"Expected torch_geometric.data.data.Data or torch.Tensor but got {type(data)}")

    def normalize_data(self):
        """Normalize the training and test datasets."""
        if self.nn_training_set is None or self.nn_test_set is None:
            raise ValueError("Datasets must be prepared first. Call prepare_datasets() before normalize_data().")
        
        # Convert each tensor in the list to NumPy arrays
        nn_training_set_np = [self.to_numpy(data) for data in self.nn_training_set]
        nn_test_set_np = [self.to_numpy(data) for data in self.nn_test_set]

        # Extract features to normalize from all training graphs and pool them
        all_training_features_to_normalize = []
        for data in nn_training_set_np:
            all_training_features_to_normalize.append(data[:, self.features_to_normalize_indices])

        # Stack all training features across all graphs and nodes
        all_training_features_to_normalize = np.vstack(all_training_features_to_normalize)

        # Fit scaler on training set
        if self.use_robust_scaler:
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        self.scaler.fit(all_training_features_to_normalize)

        # Apply scaler to training set
        # Apply tanh post-processing only to columns 1-8 (not vlm_cp at column 0) if using RobustScaler
        nn_training_set_scaled_np = []
        for data in nn_training_set_np:
            # Extract features to normalize (including vlm_cp)
            features_to_normalize = data[:, self.features_to_normalize_indices]
            # Normalize them
            features_normalized = self.scaler.transform(features_to_normalize)
            
            if self.use_robust_scaler:
                # Apply tanh only to columns 1-8 (indices 1-8), NOT to vlm_cp (index 0)
                features_normalized_processed = features_normalized.copy()
                features_normalized_processed[:, 1:9] = np.tanh(features_normalized[:, 1:9])
            else:
                features_normalized_processed = features_normalized
            
            # Extract features to keep (only coordinates)
            features_to_keep = data[:, self.features_to_keep_indices]
            # Reconstruct node features
            reconstructed = np.hstack([
                features_normalized_processed,
                features_to_keep  # coordinates (columns 9-10)
            ])
            nn_training_set_scaled_np.append(reconstructed)

        # Apply scaler to test set
        nn_test_set_scaled_np = []
        for data in nn_test_set_np:
            # Extract features to normalize (including vlm_cp)
            features_to_normalize = data[:, self.features_to_normalize_indices]
            # Normalize them
            features_normalized = self.scaler.transform(features_to_normalize)
            
            if self.use_robust_scaler:
                # Apply tanh only to columns 1-8 (indices 1-8), NOT to vlm_cp (index 0)
                features_normalized_processed = features_normalized.copy()
                features_normalized_processed[:, 1:9] = np.tanh(features_normalized[:, 1:9])
            else:
                features_normalized_processed = features_normalized
            
            # Extract features to keep (only coordinates)
            features_to_keep = data[:, self.features_to_keep_indices]
            # Reconstruct node features
            reconstructed = np.hstack([
                features_normalized_processed,
                features_to_keep  # coordinates (columns 9-10)
            ])
            nn_test_set_scaled_np.append(reconstructed)

        # Update Data objects with normalized node features
        for i, data in enumerate(self.nn_training_set):
            data.x = torch.tensor(nn_training_set_scaled_np[i], device=data.x.device, dtype=torch.float)

        for i, data in enumerate(self.nn_test_set):
            data.x = torch.tensor(nn_test_set_scaled_np[i], device=data.x.device, dtype=torch.float)
        
        return self.nn_training_set, self.nn_test_set
    
    def get_datasets(self):
        """Get the prepared and normalized training and test datasets."""
        if self.nn_training_set is None or self.nn_test_set is None:
            raise ValueError("Datasets must be prepared and normalized first.")
        return self.nn_training_set, self.nn_test_set
    
    def process_all(self):
        """Run the complete data processing pipeline."""
        self.load_data()
        self.split_data()
        self.prepare_datasets()
        self.normalize_data()
        return self.get_datasets()
    
    def save_scaler(self, filepath):
        """
        Save the fitted scaler to disk.
        
        Parameters:
        - filepath: Path where to save the scaler (should end with .pkl)
        """
        if self.scaler is None:
            raise ValueError("Scaler must be fitted first. Call normalize_data() before save_scaler().")
        
        scaler_data = {
            'scaler': self.scaler,
            'use_robust_scaler': self.use_robust_scaler,
            'features_to_normalize_indices': self.features_to_normalize_indices,
            'features_to_keep_indices': self.features_to_keep_indices,
            'normalize_coordinates': self.normalize_coordinates,
            'root_chord_ref': self.root_chord_ref,
            'apex_x': self.apex_x,
            'apex_y': self.apex_y,
            'ref_re': self.ref_re,
            'ref_length': self.ref_length
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
        print(f"Scaler saved to {filepath}")
    
    @staticmethod
    def load_scaler(filepath):
        """
        Load a fitted scaler from disk.
        
        Parameters:
        - filepath: Path to the saved scaler (should end with .pkl)
        
        Returns:
        - Dictionary containing scaler and configuration
        """
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        print(f"Scaler loaded from {filepath}")
        return scaler_data
    
    def transform_single_graph(self, graph_data, scaler_data=None):
        """
        Transform a single graph for inference using the fitted scaler.
        
        Parameters:
        - graph_data: PyTorch Geometric Data object with raw (un-normalized) features
        - scaler_data: Dictionary from load_scaler() containing scaler and config.
                      If None, uses self.scaler (must have called normalize_data() first)
        
        Returns:
        - PyTorch Geometric Data object with normalized features ready for inference
        """
        if scaler_data is None:
            if self.scaler is None:
                raise ValueError("Either provide scaler_data or call normalize_data() first.")
            scaler = self.scaler
            use_robust_scaler = self.use_robust_scaler
            features_to_normalize_indices = self.features_to_normalize_indices
            features_to_keep_indices = self.features_to_keep_indices
        else:
            scaler = scaler_data['scaler']
            use_robust_scaler = scaler_data['use_robust_scaler']
            features_to_normalize_indices = scaler_data['features_to_normalize_indices']
            features_to_keep_indices = scaler_data['features_to_keep_indices']
        
        # Convert node features to numpy
        node_features = graph_data.x.cpu().numpy()
        
        # Extract features to normalize
        features_to_normalize = node_features[:, features_to_normalize_indices]
        # Normalize them
        features_normalized = scaler.transform(features_to_normalize)
        
        if use_robust_scaler:
            # Apply tanh only to columns 1-8 (indices 1-8), NOT to vlm_cp (index 0)
            features_normalized_processed = features_normalized.copy()
            features_normalized_processed[:, 1:9] = np.tanh(features_normalized[:, 1:9])
        else:
            features_normalized_processed = features_normalized
        
        # Extract features to keep (only coordinates)
        features_to_keep = node_features[:, features_to_keep_indices]
        # Reconstruct node features
        reconstructed = np.hstack([
            features_normalized_processed,
            features_to_keep  # coordinates (columns 9-10)
        ])
        
        # Update the graph data with normalized features
        graph_data.x = torch.tensor(reconstructed, device=graph_data.x.device, dtype=torch.float)
        
        return graph_data

    def inverse_transform_graph(self, data):
        """
        Inverse transform a normalized graph to recover original raw feature values.
        
        Parameters:
        - data: PyTorch Geometric Data object with normalized features
        
        Returns:
        - Dictionary containing original raw features:
            - 'node_features': Original node features array (n_nodes, 11)
            - 'vlm_cp': vlm_cp values (column 0)
            - 'ff_raw': Far field conditions (columns 1-3: aoa_rad, mach, re)
            - 'vlm_thickness_raw': Thickness (column 4)
            - 'vlm_curvature_u_raw': Upper curvature (column 5)
            - 'vlm_curvature_l_raw': Lower curvature (column 6)
            - 'vlm_slope_u_raw': Upper slope (column 7)
            - 'vlm_slope_l_raw': Lower slope (column 8)
            - 'coordinates': Original coordinates (columns 9-10, denormalized)
        """
        if self.scaler is None:
            raise ValueError("Scaler must be fitted first. Call normalize_data() before inverse_transform_graph().")
        
        # Convert node features to numpy
        node_features = data.x.cpu().numpy()
        
        # Extract coordinates (columns 9-10, not normalized by scaler)
        coordinates_normalized = node_features[:, self.features_to_keep_indices]  # Columns 9-10: coordinates
        
        # Extract normalized features (columns 0-8, including vlm_cp)
        features_normalized_processed = node_features[:, self.features_to_normalize_indices]
        
        # Step 1: Inverse tanh only for columns 1-8 (not vlm_cp at column 0) if using RobustScaler
        features_normalized = features_normalized_processed.copy()
        if self.use_robust_scaler:
            # Apply inverse tanh only to columns 1-8
            features_normalized[:, 1:9] = np.arctanh(np.clip(features_normalized_processed[:, 1:9], -0.999999, 0.999999))
        
        # Step 2: Inverse scaler to get raw values
        features_raw = self.scaler.inverse_transform(features_normalized)
        
        # Step 3: Denormalize coordinates if they were normalized
        if self.normalize_coordinates:
            coordinates_raw = coordinates_normalized * self.root_chord_ref + np.array([self.apex_x, self.apex_y])
        else:
            coordinates_raw = coordinates_normalized
        
        # Reconstruct original node features
        node_features_original = np.hstack([
            features_raw,       # Columns 0-8: raw features (including vlm_cp at column 0)
            coordinates_raw      # Columns 9-10: coordinates
        ])
        
        # Extract individual features for convenience
        vlm_cp = features_raw[:, [0]]  # Column 0: vlm_cp
        ff_raw = features_raw[:, [1, 2, 3]]  # Columns 1-3: aoa_rad, mach, re
        vlm_thickness_raw = features_raw[:, [4]]
        vlm_curvature_u_raw = features_raw[:, [5]]
        vlm_curvature_l_raw = features_raw[:, [6]]
        vlm_slope_u_raw = features_raw[:, [7]]
        vlm_slope_l_raw = features_raw[:, [8]]
        
        return {
            'node_features': node_features_original,
            'vlm_cp': vlm_cp,
            'ff_raw': ff_raw,
            'vlm_thickness_raw': vlm_thickness_raw,
            'vlm_curvature_u_raw': vlm_curvature_u_raw,
            'vlm_curvature_l_raw': vlm_curvature_l_raw,
            'vlm_slope_u_raw': vlm_slope_u_raw,
            'vlm_slope_l_raw': vlm_slope_l_raw,
            'coordinates': coordinates_raw
        }


if __name__ == "__main__":
    """
    Unit test to verify inverse transformation correctness.
    Tests: original -> scaled -> inverse_scaled should recover original values.
    """
    # Configuration for testing
    directory_path = './data'
    root_chord_ref = 0.6536436
    apex_x, apex_y, apex_z = 0.0, 0.0, 0.0
    REF_RE = 10**7
    Ref_Length = 0.435762
    TEST_SIZE = 0.3
    RANDOM_STATES = 3407
    
    # Create parser and process data
    parser = DataParser(
        directory_path=directory_path,
        ref_re=REF_RE,
        ref_length=Ref_Length,
        root_chord_ref=root_chord_ref,
        apex_x=apex_x,
        apex_y=apex_y,
        apex_z=apex_z,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATES,
        use_robust_scaler=True,
        normalize_coordinates=True
    )
    
    # Process all data
    nn_training_set, nn_test_set = parser.process_all()
    
    print("\n" + "="*80)
    print("UNIT TEST: Testing inverse transformation (original -> scaled -> inverse_scaled)")
    print("="*80)
    
    # Store original node features before normalization
    nn_training_set_np = [parser.to_numpy(data) for data in nn_training_set]
    
    # Select 10 graphs from training set for testing
    num_test_graphs = min(10, len(nn_training_set))
    test_indices = list(range(num_test_graphs))
    
    # Store original node features before normalization
    original_features_list = []
    for idx in test_indices:
        original_features_list.append(nn_training_set_np[idx].copy())
    
    print(f"\nTesting {num_test_graphs} graphs...")
    print(f"Each graph has {nn_training_set[0].num_nodes} nodes and {nn_training_set[0].num_features} features")
    
    # Test each graph
    all_residuals = []
    max_residuals = []
    mean_residuals = []
    
    for i, idx in enumerate(test_indices):
        # Get the normalized graph
        normalized_graph = nn_training_set[idx]
        
        # Apply inverse transformation
        recovered_features = parser.inverse_transform_graph(normalized_graph)
        recovered_node_features = recovered_features['node_features']
        
        # Get original features
        original_node_features = original_features_list[i]
        
        # Compute residual: original - recovered
        residual = original_node_features - recovered_node_features
        
        # Store statistics
        max_residual = np.max(np.abs(residual))
        mean_residual = np.mean(np.abs(residual))
        all_residuals.append(residual)
        max_residuals.append(max_residual)
        mean_residuals.append(mean_residual)
        
        print(f"\nGraph {i+1} (index {idx}):")
        print(f"  Max absolute residual: {max_residual:.2e}")
        print(f"  Mean absolute residual: {mean_residual:.2e}")
        print(f"  Residual shape: {residual.shape}")
        
        # Check per-feature residuals
        print(f"  Per-feature max residuals:")
        feature_names = ['vlm_cp', 'ff_aoa_rad', 'ff_mach', 'ff_re', 'thickness', 
                        'curvature_u', 'curvature_l', 'slope_u', 'slope_l', 'coord_x', 'coord_y']
        for feat_idx, feat_name in enumerate(feature_names):
            feat_max_residual = np.max(np.abs(residual[:, feat_idx]))
            print(f"    {feat_name:15s}: {feat_max_residual:.2e}")
    
    # Overall statistics
    all_residuals_array = np.vstack(all_residuals)
    overall_max_residual = np.max(np.abs(all_residuals_array))
    overall_mean_residual = np.mean(np.abs(all_residuals_array))
    
    print("\n" + "-"*80)
    print("OVERALL STATISTICS:")
    print(f"  Overall max absolute residual: {overall_max_residual:.2e}")
    print(f"  Overall mean absolute residual: {overall_mean_residual:.2e}")
    print(f"  Number of nodes tested: {all_residuals_array.shape[0]}")
    print(f"  Number of features: {all_residuals_array.shape[1]}")
    
    # Check if residuals are acceptably small (within numerical precision)
    # For float32, we expect residuals around 1e-6 to 1e-5 due to tanh/arctanh numerical precision
    tolerance = 1e-4
    if overall_max_residual < tolerance:
        print(f"\n✓ TEST PASSED: All residuals < {tolerance:.2e}")
    else:
        print(f"\n✗ TEST FAILED: Some residuals >= {tolerance:.2e}")
        print(f"  This might be due to:")
        print(f"  - Numerical precision limits in tanh/arctanh")
        print(f"  - Extreme values near tanh boundaries")
    
    print("="*80 + "\n")


#%%

