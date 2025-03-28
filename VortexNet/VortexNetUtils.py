"""
VortexNet/VortexNetUtils.py
Utility functions for VortexNet project such as loading data, plotting, etc.

Yiren Shen 
Nov, 12, 2021
"""
import numpy as np
from scipy.spatial import distance_matrix
from scipy.interpolate import griddata

import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

import math
import os
import re
import sys
from VortexNet.MFData import MFData




# Creating a utility class based on the identified reusable functions
class VortexNetUtils:
    """
    A utility class for common functions used in aerodynamic analysis and data processing for delta wing cases.
    """


    @staticmethod
    def load_pickle_data(directory_path):
        """
        Loads .pkl files from a directory and aggregates them into a list or DataFrame if structured as dictionaries.
        """
        import os
        import pickle
        import pandas as pd
        import re
        from VortexNet.MFData import MFData
        print(f"MFData class: {MFData}")

        import pickle

        class customUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Map the old module path to the new one
                if module == "MFData":  # Replace with the old module path
                    module = "VortexNet.MFData"  # Replace with the new module path
                return super().find_class(module, name)


        mf_results = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".pkl"):
                file_path = os.path.join(directory_path, filename)
                
                # Extract the number 65 from the filename
                match = re.search(r'_sweep_(\d+)_', filename)
                if match:
                    sweep_number = int(match.group(1))
                else:
                    sweep_number = None

                # Extract the number 0010 from the filename
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
                    
                    data = customUnpickler(f).load()
                    for d in data:
                        d.geom = [sweep_number, naca_dict]
                    mf_results.extend(data)  # Assuming data is a list
        
        if isinstance(mf_results[0], dict):
            return pd.DataFrame(mf_results)
        
        print(f"Total number of runs: {len(mf_results)}")
        return mf_results
    
    @staticmethod
    def split_mfdata(mf_results, test_size = 0.3, random_seeds = 21):
        from sklearn.model_selection import train_test_split
        # Total number of samples
        total_samples = len(mf_results)

        # Generate indices for all samples
        all_indices = list(range(total_samples))

        # Randomly split indices into training and test sets (20% test set)
        train_indices, test_indices = train_test_split(all_indices, test_size=test_size, random_state=random_seeds)

        # Create training and test sets based on the indices
        training_set = [mf_results[i] for i in train_indices]
        test_set = [mf_results[i] for i in test_indices]

        # Print count of training set and test set
        print(f"Size of the training set: {len(training_set)}")
        print(f"Size of the test set: {len(test_set)}")

        return training_set, test_set
    
    @staticmethod
    def tanh_standardization(arr):
        """
        Apply tanh standardization to the input array.
        """
        return np.tanh(arr)
    
    @staticmethod
    def assemble_dataset(data_set, ref_re):
        """
        Assemble dataset by extracting control points, DCP, far field conditions, slopes, curvatures, thickness, 
        and CPU/CPL data from the provided dataset.

        Parameters:
        - data_set: List of data objects (training or test set)
        - ref_re: Reference Reynolds number for normalization

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
            cur_re = data_set[j].Re / ref_re
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
    def prepare_dataset_with_standarization(
    control_points, vlm_cp, cfd_cp, vlm_thickness, vlm_curvature_u, vlm_curvature_l, vlm_slope_u, vlm_slope_l, ff,
    aic_matrix, rhs_matrix, dcpsid, factor, chord, rnmax, Ref_Length):
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

        # Calculate the distance matrix between all pairs of points
        dist_matrix = distance_matrix(coordinates, coordinates)

        # Initialize the list for edges
        edges = []

        # Iterate over each point to find its 4 nearest neighbors
        # usually, it corresponds to the left, right, front, and back for any central nodes.
        # for edge vertex, this definition may leads to different connectivity topology. 
        for i in range(NUM_POINTS):
            # Get indices of the 4 nearest neighbors (excluding the point itself)
            nearest_neighbors = np.argsort(dist_matrix[i])[1:5]
            for neighbor in nearest_neighbors:
                edges.append([i, neighbor])

        # Convert edges to a tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Apply tanh standardization to arrays that are not vlm_cp
        # Apply tanh to ff, vlm_thickness, vlm_curvature_u, vlm_curvature_l, and coordinates
        aoa = ff[:, 0].reshape(-1, 1)
        mach = ff[:, 1].reshape(-1, 1)
        re = ff[:, 2].reshape(-1, 1)
        aoa_rad = aoa * math.pi / 180
        mach_tanh = VortexNetUtils.tanh_standardization(mach)
        aoa_tanh = VortexNetUtils.tanh_standardization(aoa_rad)
        ff_tanh = np.hstack((aoa_tanh, mach_tanh, re))
        vlm_thickness_tanh = VortexNetUtils.tanh_standardization(vlm_thickness)
        vlm_curvature_u_tanh = VortexNetUtils.tanh_standardization(vlm_curvature_u)
        vlm_curvature_l_tanh = VortexNetUtils.tanh_standardization(vlm_curvature_l)
        vlm_slope_u_tanh = VortexNetUtils.tanh_standardization(vlm_slope_u)
        vlm_slope_l_tanh = VortexNetUtils.tanh_standardization(vlm_slope_l)

        # Node features: vlm_cp (without tanh), and tanh-transformed features
        node_features = np.hstack((
            vlm_cp.reshape(-1, 1),             
            ff_tanh,                           # Tanh-transformed
            vlm_thickness_tanh.reshape(-1, 1), # Tanh-transformed
            vlm_curvature_u_tanh.reshape(-1, 1), # Tanh-transformed
            vlm_curvature_l_tanh.reshape(-1, 1), # Tanh-transformed
            vlm_slope_u_tanh.reshape(-1, 1), # Tanh-transformed
            vlm_slope_l_tanh.reshape(-1, 1), # Tanh-transformed
            coordinates                   
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
        rho_inf = VortexNetUtils.compute_freestream_density(re[0], mach_inf, viscosity, Ref_Length, T_inf)
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

    @staticmethod
    def plot_with_surface(ax, mach_np, aoa_np, vlm_data_np, cfd_data_np, maped_cfd_data_np, 
                          label_vlm, label_cfd, label_maped_cfd, zlabel):
        """
        Plots a 3D surface fit for VLM, CFD, and mapped CFD data against Mach number and AOA.
        """
        from scipy.interpolate import griddata
        
        # Define grid for plotting
        mach_grid, aoa_grid = np.meshgrid(
            np.linspace(mach_np.min(), mach_np.max(), 50), 
            np.linspace(aoa_np.min(), aoa_np.max(), 50)
        )
        
        # Surface fitting for each dataset
        vlm_surface = griddata((mach_np, aoa_np), vlm_data_np, (mach_grid, aoa_grid), method='linear')
        maped_cfd_surface = griddata((mach_np, aoa_np), maped_cfd_data_np, (mach_grid, aoa_grid), method='linear')
        cfd_surface = griddata((mach_np, aoa_np), cfd_data_np, (mach_grid, aoa_grid), method='linear')
        
        # Scatter plot and surface plotting
        ax.scatter(mach_np, aoa_np, vlm_data_np, label=label_vlm, c='b')
        ax.scatter(mach_np, aoa_np, maped_cfd_data_np, label=label_maped_cfd, c='g')
        ax.scatter(mach_np, aoa_np, cfd_data_np, label=label_cfd, c='r')
        
        # Plot surfaces
        ax.plot_surface(mach_grid, aoa_grid, vlm_surface, color='blue', alpha=0.5, rstride=100, cstride=100)
        ax.plot_surface(mach_grid, aoa_grid, maped_cfd_surface, color='green', alpha=0.5, rstride=100, cstride=100)
        ax.plot_surface(mach_grid, aoa_grid, cfd_surface, color='red', alpha=0.5, rstride=100, cstride=100)
        
        # Labels and legend
        ax.set_xlabel('Mach Number')
        ax.set_ylabel('AOA [deg]')
        ax.set_zlabel(zlabel)
        ax.legend()

    @staticmethod
    def split_data_and_plot_3d(mf_results, test_size=0.3, random_state=21):
        """
        Splits the dataset into training and test sets, then creates a 3D scatter plot for each set.
        """
        from sklearn.model_selection import train_test_split
        import plotly.graph_objs as go
        from plotly.offline import plot
        
        # Split data
        total_samples = len(mf_results)
        all_indices = list(range(total_samples))
        train_indices, test_indices = train_test_split(all_indices, test_size=test_size, random_state=random_state)
        
        training_set = [mf_results[i] for i in train_indices]
        test_set = [mf_results[i] for i in test_indices]
        
        # Extract parameters for plotting
        train_AOA = [sample.alpha for sample in training_set]
        train_Mach = [sample.mach for sample in training_set]
        train_Re = [sample.Re for sample in training_set]
        
        test_AOA = [sample.alpha for sample in test_set]
        test_Mach = [sample.mach for sample in test_set]
        test_Re = [sample.Re for sample in test_set]
        
        # Create Plotly traces for 3D plotting
        trace_train = go.Scatter3d(
            x=train_AOA, y=train_Mach, z=train_Re,
            mode='markers', marker=dict(size=5, color='blue', opacity=0.8), name='Training Set'
        )
        trace_test = go.Scatter3d(
            x=test_AOA, y=test_Mach, z=test_Re,
            mode='markers', marker=dict(size=5, color='red', opacity=0.8), name='Test Set'
        )
        
        layout = go.Layout(
            title='3D Scatter Plot of Training and Test Sets',
            scene=dict(
                xaxis_title='Angle of Attack (AOA) [deg]',
                yaxis_title='Mach Number',
                zaxis_title='Reynolds Number (Re)'
            ),
            legend=dict(x=0.7, y=0.9)
        )
        
        fig = go.Figure(data=[trace_train, trace_test], layout=layout)
        plot(fig)


    @staticmethod
    def split_data_and_plot_3d(mf_results, test_size=0.3, random_state=21):
        """
        Splits the dataset into training and test sets, then creates a 3D scatter plot for each set.
        """
        from sklearn.model_selection import train_test_split
        import plotly.graph_objs as go
        from plotly.offline import plot
        
        # Split data
        total_samples = len(mf_results)
        all_indices = list(range(total_samples))
        train_indices, test_indices = train_test_split(all_indices, test_size=test_size, random_state=random_state)
        
        training_set = [mf_results[i] for i in train_indices]
        test_set = [mf_results[i] for i in test_indices]
        
        # Extract parameters for plotting
        train_AOA = [sample.alpha for sample in training_set]
        train_Mach = [sample.mach for sample in training_set]
        train_Re = [sample.Re for sample in training_set]
        
        test_AOA = [sample.alpha for sample in test_set]
        test_Mach = [sample.mach for sample in test_set]
        test_Re = [sample.Re for sample in test_set]
        
        # Create Plotly traces for 3D plotting
        trace_train = go.Scatter3d(
            x=train_AOA, y=train_Mach, z=train_Re,
            mode='markers', marker=dict(size=5, color='blue', opacity=0.8), name='Training Set'
        )
        trace_test = go.Scatter3d(
            x=test_AOA, y=test_Mach, z=test_Re,
            mode='markers', marker=dict(size=5, color='red', opacity=0.8), name='Test Set'
        )
        
        layout = go.Layout(
            title='3D Scatter Plot of Training and Test Sets',
            scene=dict(
                xaxis_title='Angle of Attack (AOA) [deg]',
                yaxis_title='Mach Number',
                zaxis_title='Reynolds Number (Re)'
            ),
            legend=dict(x=0.7, y=0.9)
        )
        
        fig = go.Figure(data=[trace_train, trace_test], layout=layout)
        plot(fig)

    @staticmethod
    def plot_nn_percentage_error(aoa_np, mach_np, re_np, nn_cm_np, maped_cfd_cm_np):
        """
        Plots the percentage error of NN predictions relative to mapped CFD data on the AoA-Mach,
        AoA-Re, and Mach-Re planes.
        """

        
        # Compute the percentage error
        percentage_error = 100 * (nn_cm_np - maped_cfd_cm_np) / maped_cfd_cm_np
        percentage_error = np.clip(percentage_error, -200, 200)  # Limit to ±300%
        
        # Define grid parameters
        num_aoa_points, num_mach_points, num_re_points = 500, 500, 500
        aoa_grid = np.linspace(aoa_np.min(), aoa_np.max(), num_aoa_points)
        mach_grid = np.linspace(mach_np.min(), mach_np.max(), num_mach_points)
        re_grid_scaled = np.linspace((re_np / 1e7).min(), (re_np / 1e7).max(), num_re_points) #normalize Re by 10^7

        # Plot AoA-Mach plane
        AOA_grid, Mach_grid = np.meshgrid(aoa_grid, mach_grid)
        grid_percentage_error = griddata((aoa_np, mach_np), percentage_error, (AOA_grid, Mach_grid), method='nearest')
        grid_percentage_error = np.ma.array(grid_percentage_error, mask=np.isnan(grid_percentage_error))

        plt.figure(figsize=(12, 8))
        contour = plt.contourf(AOA_grid, Mach_grid, grid_percentage_error, levels=100, cmap='RdBu_r', vmin=-200, vmax=200)
        plt.colorbar(contour, label='Percentage Error (%)')
        scatter = plt.scatter(aoa_np, mach_np, c=re_np, cmap='viridis', edgecolors='k', s=50)
        plt.colorbar(scatter, label='Reynolds Number')
        plt.xlabel('AOA [deg]')
        plt.ylabel('Mach Number')
        plt.title('Percentage Error of NN Prediction Relative to Mapped CFD Data on AoA-Mach Plane')
        plt.tight_layout()
        plt.show()

        # Plot AoA-Re plane
        AOA_grid, Re_grid = np.meshgrid(aoa_grid, re_grid_scaled)
        grid_percentage_error = griddata((aoa_np, re_np / 1e7), percentage_error, (AOA_grid, Re_grid), method='linear')
        grid_percentage_error = np.ma.array(grid_percentage_error, mask=np.isnan(grid_percentage_error))

        plt.figure(figsize=(12, 8))
        contour = plt.contourf(AOA_grid, Re_grid * 1e7, grid_percentage_error, levels=100, cmap='RdBu_r', vmin=-200, vmax=200)
        plt.colorbar(contour, label='Percentage Error (%)')
        scatter = plt.scatter(aoa_np, re_np, c=mach_np, cmap='viridis', edgecolors='k', s=50)
        plt.colorbar(scatter, label='Mach Number')
        plt.xlabel('AOA [deg]')
        plt.ylabel('Reynolds Number')
        plt.title('Percentage Error of NN Prediction Relative to Mapped CFD Data on AoA-Re Plane')
        plt.tight_layout()
        plt.show()

        # Plot Mach-Re plane
        Mach_grid, Re_grid = np.meshgrid(mach_grid, re_grid_scaled)
        grid_percentage_error = griddata((mach_np, re_np / 1e7), percentage_error, (Mach_grid, Re_grid), method='linear')
        grid_percentage_error = np.ma.array(grid_percentage_error, mask=np.isnan(grid_percentage_error))

        plt.figure(figsize=(12, 8))
        contour = plt.contourf(Mach_grid, Re_grid * 1e7, grid_percentage_error, levels=100, cmap='RdBu_r', vmin=-200, vmax=200)
        plt.colorbar(contour, label='Percentage Error (%)')
        scatter = plt.scatter(mach_np, re_np, c=aoa_np, cmap='viridis', edgecolors='k', s=50)
        plt.colorbar(scatter, label='AoA [deg]')
        plt.xlabel('Mach Number')
        plt.ylabel('Reynolds Number')
        plt.title('Percentage Error of NN Prediction Relative to Mapped CFD Data on Mach-Re Plane')
        plt.tight_layout()
        plt.show()
    
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
    
    @staticmethod
    def plot_field_distribution(VD, cp, title='Pressure Distribution', min=None, max=None):
        # Ensure cp is a 1D array
        if cp.ndim > 1:
            cp = cp.squeeze(0)

        if min is None:
            vmin = np.min(cp)
        else:
            vmin = min    

        if max is None:
            vmax = np.max(cp)     
        else:
            vmax = max	
        
        fig, ax = plt.subplots()
        
        # Normalize the pressure coefficient values to the range [0, 1]
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('viridis')
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        # Plot the panels and fill with cp data
        for i in range(len(VD.XA1)):
            x = [VD.XA1[i], VD.XB1[i], VD.XB2[i], VD.XA2[i], VD.XA1[i]]
            y = [VD.YA1[i], VD.YB1[i], VD.YB2[i], VD.YA2[i], VD.YA1[i]]
            
            # Get the color for the current pressure coefficient value
            cp_value = cp[i]
            color = scalar_map.to_rgba(cp_value)
            
            # Fill the panel with the corresponding cp value
            polygon = plt.Polygon(np.column_stack((x, y)), closed=True, facecolor=color, edgecolor=(0, 0, 0, 0.1))
            ax.add_patch(polygon)
        
        plt.colorbar(scalar_map, label=title)
        plt.title(title)
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis('equal')
        plt.show()