"""
Created:  Oct 30 2021, Yiren Shen
# ``
# ----------------------------------------------------------------------
#   Script
# ----------------------------------------------------------------------
#   prepare data pkl for deltawing_sweep_55_naca_0010
#
#   The script reads in the configurations from a text file, and then performs VLM analysis
#   for each configuration. The script then reads in the CFD surface_flow.vtu for each configuration and
#   projects the CFD data to the VLM lattice. The results are saved to a pickle file.
#
#   The script uses the SUAVE package for the VLM analysis..
#   The script is designed to be run from the root directory of the project.
#
#   The script saves the results to a pickle file in the current directory.
#   
#   User shall change the configuration setting in line 85-122 to indicate the correct path for CFD fiels, 
#   and match the simulation constants to those used in CFD.
# ----------------------------------------------------------------------

"""
# General Python Imports
import sys
import numpy as np
import matplotlib.pyplot as plt

# SUAVE Imports
import SUAVE
assert SUAVE.__version__=='2.5.2', 'These tutorials only work with the SUAVE 2.5.2 release'
from SUAVE.Core import Data, Units 
# The Data import here is a native SUAVE data structure that functions similarly to a dictionary.
#   However, iteration directly returns values, and values can be retrieved either with the 
#   typical dictionary syntax of "entry['key']" or the more class-like "entry.key". For this to work
#   properly, all keys must be strings.
# The Units import is used to allow units to be specified in the vehicle setup (or elsewhere).
#   This is because SUAVE functions generally operate using metric units, so inputs must be 
#   converted. To use a length of 20 feet, set l = 20 * Units.ft . Additionally, to convert to SUAVE
#   output back to a desired units, use l_ft = l_m / Units.ft
from SUAVE.Plots.Performance.Mission_Plots import *
# These are a variety of plotting routines that simplify the plotting process for commonly 
# requested metrics. Plots of specifically desired metrics can also be manually created.
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Propulsion.turbojet_sizing import turbojet_sizing
from SUAVE.Input_Output.OpenVSP import write
from SUAVE.Input_Output.OpenVSP.vsp_read  import vsp_read
from SUAVE.Input_Output.OpenVSP import get_vsp_measurements

# import i/o functions
from SUAVE.Input_Output.Results import  print_parasite_drag,  \
     print_compress_drag, \
     print_engine_data,   \
     print_mission_breakdown, \
     print_weight_breakdown
VLM_path = './scripts'
sys.path.append(VLM_path)
from VLM import VLM
from torch_geometric.data import DataLoader
from copy import deepcopy
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
# import geometry form SUAVE
from generateDeltawing import full_setup, point_analysis

# import data structure 
sys.path.append("./VortexNet/")
from VortexNet import MFData
from DataLoader import DataLoader
import pickle
import pandas as pd
from sklearn.neighbors import KDTree
import math
import os
from multiprocessing import Pool


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    #######################INPUT SETUP################################
    # DESIGN PARAMETERS for CURRENT GEOM 
    LE_SWEEP = 55
    NACA_4DIGITS = {'m': 0, 'p': 0, 't': 16, 'chord_length': 1.0}
    current_dir = os.getcwd()
    config_name = f"deltawing_sweep_{LE_SWEEP}_naca_{NACA_4DIGITS['m']}{NACA_4DIGITS['p']}{NACA_4DIGITS['t']}"
    print(f"Current Configuration: {config_name}")
    # extract CFD configurations 
    # a configuration file with free-stream conditions tabulated, 
    # see ./dataset/freestream_configurations for example
    configuration_file_path = f"../data/{config_name}/configuration.txt"
    # root path for CFD data
    # Contains multiple folders of each CFD run where #TESTRUN_surface_flow.vtu, from SU2, is located
    # each sub folders shall have the folder name as the config_name
    cfd_root_path = f"../data/{config_name}/"
    # file corresponding to the vertex area defined at the vertex location
    # A .csv file with columns: Global Index, Vertex Area, X, Y, Z
    vertex_area_file_path = f"../mesh_data/cell_area_{config_name}.csv"

    # define data save path 
    PKL_PATH = os.path.join(current_dir, f"pklData_{config_name}_example.pkl")

    # SU2 parameters 
    CHAR_LENGTH = 0.43576
    CHAR_AREA = 0.19927702
    tInfinity = 322
    pTInfinity = 101325.3
    TInfinity = 322
    R = 287.05
    gamma = 1.4
    viscosity = 1.835E-5
    MOM_REF_XYZ = [0.4357624, 0.0, 0.0]
    # Sutherland's constants for air
    C1 = 0.0000139992 # kg/(m·s·√K)
    mu_ref = 1.71e-5
    T_ref = 273.11
    S = 110.56      # Sutherland's temperature, K

    #######################EOF INPUT SETUP############################


    # SUAVE components 
    configs, analyses = full_setup(LE_SWEEP, NACA_4DIGITS)
    configs.finalize()
    analyses.finalize()
    vlm_reference_area = configs[config_name]['reference_area']
    # find coefficient scaling ratio 
    coeff_scaling_ratio = CHAR_AREA / vlm_reference_area
    
    # initalize array for results 
    mf_results = []

    # read in farfield configurations 
    ff_configurations = read_configurations(configuration_file_path)
    print(f"{len(configs)} configurations loaded")

    # initialize np array for global index and vertex area that will be passed after 1st pass
    GLOBAL_INDEX = np.array([])
    VERTEX_AREA = np.array([])

    # loop over all configurations
    for current_config in ff_configurations:
        # extract configuration data
        test_number = int(current_config[0])
        AOA = float(current_config[1])
        Ma = float(current_config[2])
        Re = float(current_config[3])

        # perform point analysis in SUAVE, VLM re = inf 
        point_result = point_analysis(configs[config_name], LE_SWEEP, NACA_4DIGITS, AOA, Ma, if_plot = False)
        # save results 
        current_result = MFData()
        current_result.test_number = test_number
        current_result.alpha = extract_float(AOA)
        current_result.mach = extract_float(Ma)
        current_result.Re = extract_float(Re)
        current_result.vlm_data = point_result.cp
        current_result.vlm_cm = extract_float(point_result.CM)
        current_result.vlm_cl = extract_float(point_result.CL)
        current_result.vlm_cd = extract_float(point_result.CDi)
        current_result.vlm_vd = point_result.VD
        current_result.alpha_local = np.squeeze(point_result.alpha_local)
        current_result.beta_local = np.squeeze(point_result.beta_local)
        current_result.gamma_local = np.squeeze(point_result.gamma_local)
        current_result.theta_x = np.squeeze(point_result.theta_x)
        current_result.theta_y = np.squeeze(point_result.theta_y)
        current_result.theta_z = np.squeeze(point_result.theta_z)
        current_result.vlm_vx = np.squeeze(point_result.v_x)
        current_result.vlm_vy = np.squeeze(point_result.v_y)
        current_result.vlm_vz = np.squeeze(point_result.v_z)
        current_result.vlm_A = point_result.A.squeeze(0)
        current_result.vlm_RHS = point_result.RHS
        current_result.vlm_RNMAX = point_result.RNMAX
        current_result.vlm_CHORD = point_result.CHORD
        current_result.vlm_DCPSID = point_result.DCPSID
        current_result.vlm_FACTOR = point_result.FACTOR

        # validate the residual 
        GNET = (current_result.vlm_data - current_result.vlm_DCPSID) / 2
        GAMMA = (GNET * (current_result.vlm_CHORD / current_result.vlm_RNMAX) ) \
                / current_result.vlm_FACTOR
        residual =  current_result.vlm_A @ GAMMA.T - current_result.vlm_RHS.T
        print(f"Average Residual: {np.average(residual)}")

        # compute curvature and thickness
        compute_lattice_slopes_and_gaussian_curvature(current_result.vlm_vd)

        current_result.thickness = np.squeeze(current_result.vlm_vd.thickness)
        current_result.chordwise_slope_u = np.squeeze(current_result.vlm_vd.chordwise_slope_upper)
        current_result.spanwise_slope_u = np.squeeze(current_result.vlm_vd.chordwise_slope_upper)
        current_result.gaussian_curvature_u = np.squeeze(current_result.vlm_vd.gaussian_curvature_upper)
        current_result.chordwise_slope_l = np.squeeze(current_result.vlm_vd.chordwise_slope_lower)
        current_result.spanwise_slope_l = np.squeeze(current_result.vlm_vd.spanwise_slope_lower)
        current_result.gaussian_curvature_l = np.squeeze(current_result.vlm_vd.gaussian_curvature_lower)

        # compute viscosity used for CFD 
        viscosity = mu_ref * (TInfinity / T_ref) ** 1.5 * (T_ref + S) / (TInfinity + S)
        rhoInfinity = compute_freestream_density(Re, Ma, viscosity, CHAR_LENGTH, TInfinity)
        reference_pressure = 0.5 * rhoInfinity * (Ma ** 2) * gamma * R * TInfinity
        print(f"Reference Pressure: {reference_pressure:.2f} Pa")
        current_result.ref_pressure = reference_pressure

        # analyze cfd results 
        configuration_folder_path = \
            f"Test_{current_result.test_number:03}_AOA_{current_result.alpha:.2f}_Mach_{current_result.mach:.2f}_Re_{current_result.Re:.2e}/"
        cfd_file_path = cfd_root_path + configuration_folder_path + \
                        f"{current_result.test_number:03}_surface_flow.vtu"

        force_breakdown_file_path = cfd_root_path + configuration_folder_path + \
                                    f"{current_result.test_number:03}_forces_breakdown.dat"
        
        dataloader = DataLoader(cfd_file_path)

        # extract intergated quantities
        # the coefficient scaling is only used if CFD reference area was not scaled by geometries, 
        # if you are using correct area in SU2 configuration file, set coeff_scaling_ratio = 1
        cfd_cl, cfd_cd, cfd_cm, cfd_csf, cfd_cfx, cfd_cfy, cfd_cfz  = \
            dataloader.extract_aero_coefficients(force_breakdown_file_path)
        current_result.cfd_cl = cfd_cl * coeff_scaling_ratio
        current_result.cfd_cd = cfd_cd * coeff_scaling_ratio
        current_result.cfd_cm = cfd_cm * coeff_scaling_ratio
        current_result.cfd_csf = cfd_csf * coeff_scaling_ratio
        current_result.cfd_cfx = cfd_cfx * coeff_scaling_ratio
        current_result.cfd_cfy = cfd_cfy * coeff_scaling_ratio
        current_result.cfd_cfz = cfd_cfz * coeff_scaling_ratio

        # load CFD data
        data, mesh_info = dataloader.load_cfd_data()
        print(f"CFD data for test {current_result.test_number} loaded, mesh info: {mesh_info}")
        xyz_coordinates, connectivity = dataloader.get_xyz_and_connectivity(data)

        # project CFD data to VLM lattice
        # sort and bin vertex into panels using KDTree
        if current_result.test_number == 1:
            # read the area data
            df = pd.read_csv(vertex_area_file_path)
            global_index = []
            vertex_areas = []
            csv_coords = df[['X', 'Y', 'Z']].values
            tree = KDTree(csv_coords)

            # iterate over all xyz coordinates
            for i, coord in enumerate(xyz_coordinates):
                # Query the KD-tree to find the closest point
                dist, idx = tree.query([coord], k=1)  # k=1 means we want the single closest point
                
                # dist and idx are returned as arrays, so we extract the first element
                min_distance = dist[0][0]
                closest_index = idx[0][0]

                if min_distance > 1E-7: #indicating a potentially false vertex identification
                    print(f"Warning: Closest point distance is {min_distance} for point {coord}.")
                
                # Retrieve the Global index and Vertex area for the closest point
                global_index.append(df.iloc[closest_index]['Global Index'])
                vertex_areas.append(df.iloc[closest_index]['Vertex Area'])
            
            global_index = np.array(global_index)
            vertex_areas = np.array(vertex_areas)
            GLOBAL_INDEX = global_index
            VERTEX_AREA = vertex_areas    

        # project CFD data to VLM lattice
        pressure = dataloader.get_array(data, array_name='Pressure')
        lattice = dataloader.parse_lattice_corners(current_result.vlm_vd)
        cfd_delta_p, pu, pl = dataloader.project_cfd_to_lattice(lattice, xyz_coordinates, pressure,
                                                                   VERTEX_AREA)
        current_result.cfd_data = list(np.array(cfd_delta_p) / reference_pressure)
        current_result.cfd_cpu = list(np.array(pu) / reference_pressure)
        current_result.cfd_cpl = list(np.array(pl) / reference_pressure)
        current_result.cfd_refP = reference_pressure


        mf_results.append(current_result)

    # save results
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(mf_results, f)
    
    # read results
    with open(PKL_PATH, 'rb') as f:
        read_results = pickle.load(f)

    # check if two results are the same length
    assert len(mf_results) == len(read_results), "Results are not the same length."
    print(f"Results saved and read successfully, {len(mf_results)} results saved.")

    return



def compute_lattice_slopes_and_gaussian_curvature(VD):
    """
    Computes the slopes in the chordwise and spanwise directions,
    Gaussian curvature, and thickness at each lattice (panel) for all
    wings/surfaces in the VD data structure, for both upper and lower surfaces.

    Parameters:
    -----------
    VD : Data
        Vortex distribution data structure containing panel and control point information.

    Modifies:
    ---------
    VD : Data
        Adds the following attributes to VD:
            - VD.chordwise_slope_upper : Flat array of chordwise slope (upper surface).
            - VD.spanwise_slope_upper  : Flat array of spanwise slope (upper surface).
            - VD.gaussian_curvature_upper  : Flat array of Gaussian curvature (upper surface).
            - VD.chordwise_slope_lower : Flat array of chordwise slope (lower surface).
            - VD.spanwise_slope_lower  : Flat array of spanwise slope (lower surface).
            - VD.gaussian_curvature_lower  : Flat array of Gaussian curvature (lower surface).
            - VD.thickness             : Flat array of thickness at each control point.
    """
    import numpy as np

    # Initialize flat arrays to store slope and curvature data
    total_panels = len(VD.XC)  # Total number of panels across all wings/surfaces

    # For upper surface
    chordwise_slope_upper = np.zeros(total_panels)
    spanwise_slope_upper = np.zeros(total_panels)
    gaussian_curvature_upper = np.zeros(total_panels)

    # For lower surface
    chordwise_slope_lower = np.zeros(total_panels)
    spanwise_slope_lower = np.zeros(total_panels)
    gaussian_curvature_lower = np.zeros(total_panels)

    # Thickness
    thickness = np.zeros(total_panels)

    # Index to keep track of position in the flat arrays
    panel_start_index = 0

    # Number of wings/surfaces in VD
    num_wings = len(VD.n_cw)

    # Loop over each wing/surface
    for wing_index in range(num_wings):
        # Extract n_cw and n_sw for the wing
        n_cw = VD.n_cw[wing_index]
        n_sw = VD.n_sw[wing_index]

        # Get start and end indices for the panels of this wing
        start_strip = VD.spanwise_breaks[wing_index]
        end_strip = VD.spanwise_breaks[wing_index + 1] if wing_index + 1 < len(VD.spanwise_breaks) else len(VD.chordwise_breaks)
        start_panel = VD.chordwise_breaks[start_strip]
        end_panel = VD.chordwise_breaks[end_strip] if end_strip < len(VD.chordwise_breaks) else len(VD.XC)

        # Number of panels in this wing
        num_panels = n_cw * n_sw

        # Extract control point coordinates
        XC = VD.XC[start_panel:end_panel]
        YC = VD.YC[start_panel:end_panel]
        ZC = VD.ZC[start_panel:end_panel]
        ZCU = VD.ZCU[start_panel:end_panel]  # Upper surface Z-coordinate
        ZCL = VD.ZCL[start_panel:end_panel]  # Lower surface Z-coordinate

        # Ensure the number of panels matches n_cw * n_sw
        if len(XC) != num_panels:
            raise ValueError(f"Mismatch in number of panels for wing {wing_index}")

        # Reshape into grids
        XC_grid = XC.reshape((n_sw, n_cw))
        YC_grid = YC.reshape((n_sw, n_cw))
        ZC_grid = ZC.reshape((n_sw, n_cw))
        Z_upper = ZCU.reshape((n_sw, n_cw))
        Z_lower = ZCL.reshape((n_sw, n_cw))

        # Compute thickness at each control point
        thickness_grid = Z_upper - Z_lower

        # Compute partial derivatives using central differences

        # Chordwise direction (axis=1)
        dXc = np.gradient(XC_grid, axis=1)
        Zx_upper = np.gradient(Z_upper, axis=1) / dXc
        Zx_lower = np.gradient(Z_lower, axis=1) / dXc

        # Spanwise direction (axis=0)
        dYs = np.gradient(YC_grid, axis=0)
        Zy_upper = np.gradient(Z_upper, axis=0) / dYs
        Zy_lower = np.gradient(Z_lower, axis=0) / dYs

        # Second derivatives for upper surface
        Zxx_upper = np.gradient(Zx_upper, axis=1) / dXc
        Zyy_upper = np.gradient(Zy_upper, axis=0) / dYs
        Zxy_upper = np.gradient(Zx_upper, axis=0) / dYs

        # Second derivatives for lower surface
        Zxx_lower = np.gradient(Zx_lower, axis=1) / dXc
        Zyy_lower = np.gradient(Zy_lower, axis=0) / dYs
        Zxy_lower = np.gradient(Zx_lower, axis=0) / dYs

        # Compute curvature for the upper surface

        # First Fundamental Form Coefficients (Upper Surface)
        E_upper = 1 + Zx_upper ** 2
        F_upper = Zx_upper * Zy_upper
        G_upper = 1 + Zy_upper ** 2

        # Second Fundamental Form Coefficients (Upper Surface)
        L_upper = Zxx_upper
        M_upper = Zxy_upper
        N_upper = Zyy_upper

        # Denominator for curvature formulas (Upper Surface)
        denominator_upper = E_upper * G_upper - F_upper ** 2

        # Compute Gaussian Curvature for Upper Surface
        K_upper = (L_upper * N_upper - M_upper ** 2) / denominator_upper

        # Compute curvature for the lower surface

        # First Fundamental Form Coefficients (Lower Surface)
        E_lower = 1 + Zx_lower ** 2
        F_lower = Zx_lower * Zy_lower
        G_lower = 1 + Zy_lower ** 2

        # Second Fundamental Form Coefficients (Lower Surface)
        L_lower = Zxx_lower
        M_lower = Zxy_lower
        N_lower = Zyy_lower

        # Denominator for curvature formulas (Lower Surface)
        denominator_lower = E_lower * G_lower - F_lower ** 2

        # Compute Gaussian Curvature for Lower Surface
        K_lower = (L_lower * N_lower - M_lower ** 2) / denominator_lower

        # Flatten the grids back into flat arrays
        chordwise_slope_flat_upper = Zx_upper.flatten()
        spanwise_slope_flat_upper = Zy_upper.flatten()
        gaussian_curv_flat_upper = K_upper.flatten()

        chordwise_slope_flat_lower = Zx_lower.flatten()
        spanwise_slope_flat_lower = Zy_lower.flatten()
        gaussian_curv_flat_lower = K_lower.flatten()

        thickness_flat = thickness_grid.flatten()

        # Store the slope and curvature data back into the flat arrays
        panel_end_index = panel_start_index + num_panels

        chordwise_slope_upper[panel_start_index:panel_end_index] = chordwise_slope_flat_upper
        spanwise_slope_upper[panel_start_index:panel_end_index] = spanwise_slope_flat_upper
        gaussian_curvature_upper[panel_start_index:panel_end_index] = gaussian_curv_flat_upper

        chordwise_slope_lower[panel_start_index:panel_end_index] = chordwise_slope_flat_lower
        spanwise_slope_lower[panel_start_index:panel_end_index] = spanwise_slope_flat_lower
        gaussian_curvature_lower[panel_start_index:panel_end_index] = gaussian_curv_flat_lower

        thickness[panel_start_index:panel_end_index] = thickness_flat

        # Update the panel_start_index for the next wing
        panel_start_index = panel_end_index

    # Add the slope and curvature data to the VD data structure
    VD.chordwise_slope_upper = chordwise_slope_upper
    VD.spanwise_slope_upper = spanwise_slope_upper
    VD.gaussian_curvature_upper = gaussian_curvature_upper

    VD.chordwise_slope_lower = chordwise_slope_lower
    VD.spanwise_slope_lower = spanwise_slope_lower
    VD.gaussian_curvature_lower = gaussian_curvature_lower

    VD.thickness = thickness



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


def euclidean_distance(point1, point2):
    # Function to calculate the Euclidean distance between two 3D points

    return np.sqrt((point1[0] - point2[0]) ** 2 + 
                   (point1[1] - point2[1]) ** 2 + 
                   (point1[2] - point2[2]) ** 2)

def extract_float(value):
    # Base case: if the value is a float, return it
    if isinstance(value, float):
        return value
    
    # If the value is a list (or other iterable), keep extracting
    if isinstance(value, (list, tuple, np.ndarray)):
        return extract_float(value[0])
    
    # If the value is neither a list nor a float, raise an error
    raise ValueError("The structure does not contain a float.")

def read_configurations(filepath):
    # read list of AOA and Mach configurations from text file 
    # return a list of configurations
    with open(filepath, 'r') as f:
        lines = f.readlines()
        configs = []
        for line in lines:
            config = line.split(',')
            configs.append(config)
    return configs





if __name__ == '__main__': 
    
    main()
