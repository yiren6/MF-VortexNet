"""
Data class for the GNN model.
"""

import numpy as np
import pickle

class MFData:
    def __init__(self, vlm_data=None, vlm_cm=0.0, vlm_cl=0.0, vlm_cd=0.0, cfd_data=None, \
                 cfd_cm=0.0, cfd_cl= 0.0, cfd_cd = 0.0, vlm_vd = None):
        self.test_number = -999                 # test  number
        self.mach = -999                        # mach number
        self.alpha = -999                       # angle of attack [deg]
        self.Re = -999                          # Reynolds number
        self.ref_pressure = -999                # reference pressure [Pa]
        self.vlm_data = np.array(vlm_data) if vlm_data is not None else np.array([])    # vlm DCP
        self.vlm_cm = vlm_cm                # moment coefficient
        self.vlm_cl = vlm_cl                # lift coefficient
        self.vlm_cd = vlm_cd                # drag coefficient
        self.vlm_cdi = 0.0                  # induced drag coefficient
        self.vlm_cytot = 0.0                # total side force coefficient
        self.vlm_crtot = 0.0                # total rolling moment coefficient
        self.vlm_crmtot = 0.0               # total pitching moment coefficient
        self.vlm_cntot = 0.0                # total normal force coefficient
        self.vlm_cymtot = 0.0               # total yawing moment coefficient
        self.vlm_v_distribution = []        # vortex distribution
        self.vlm_gamma = []                 # VLM Gamma array
        self.vlm_cp = []                    # VLM Cp array
        self.alpha_local = []               # panel-wise local angle of attack from VLM panels
        self.beta_local = []                # panel-wise local side slip angle from VLM panels
        self.gamma_local = []               # panel-wise local circulation from VLM panels
        self.theta_x =[]                    # panel-wise vlm theta_x
        self.theta_y = []                   # panel-wise vlm theta_y
        self.theta_z = []                   # panel-wise vlm theta_z
        self.vlm_vx = []                    # panel-wise vlm vx
        self.vlm_vy = []                    # panel-wise vlm vy
        self.vlm_vz = []                    # panel-wise vlm vz
        self.thickness = []                 # wing thickness at control points
        self.spanwise_slope_u = []          # spanwise slope at upper surface at control points
        self.chordwise_slope_u = []         # chordwise slope at upper surface at control points        
        self.gaussian_curvature_u = []      # gaussian curvature at upper surface at control points
        self.spanwise_slope_l = []          # spanwise slope at lower surface at control points
        self.chordwise_slope_l = []         # chordwise slope at lower surface at control points
        self.gaussian_curvature_l = []      # gaussian curvature at lower surface at control points


        self.cfd_data = np.array(cfd_data) if cfd_data is not None else np.array([]) 
        self.cfd_cpl = np.array(cfd_data) if cfd_data is not None else np.array([])  # average pressure coefficient  for VLM panels at upper surface at VLM control points
        self.cfd_cpu = np.array(cfd_data) if cfd_data is not None else np.array([])  # average pressure coefficient for VLM panels at lower surface at VLM control points
        self.cfd_cm = cfd_cm                # moment coefficient
        self.cfd_cl = cfd_cl                # lift coefficient
        self.cfd_cd = cfd_cd                # drag coefficient
        self.cfd_cfx = 0.0                  # force coefficient in x direction
        self.cfd_cfy = 0.0                  # force coefficient in y direction
        self.cfd_cfz = 0.0                  # force coefficient in z direction
        self.cfd_csf = 0.0                  # skin friction coefficient
        self.cfd_refP = 0.0                 # reference pressure used in CFD
        self.vlm_vd = vlm_vd                # vortex distribution in SUAVE (VD)
        self.vlm_A = []                     # A matrix solved in VLM
        self.vlm_RHS = []                   # RHS vector solved in VLM
        self.vlm_RNMAX = []                 # RNMAX in SUAVE VLM
        self.vlm_CHORD = []                 # CHRRD in SUAVE VLM
        self.vlm_DCPSID = []                # DCPSID in SUAVE VLM
        self.vlm_FACTOR = []                # FACTOR in SUAVE VLM
        # array to store the geometry data
        self.geom = []


    def get_array_size(self, input_data):
        return len(input_data)    

    def save_with_pickle(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def __repr__(self):
        return (f"Test Number: {self.test_number}, Mach: {self.mach}, AOA: {self.alpha}, "
                f"Data(vlm_data={self.vlm_data}, vlm_cm={self.vlm_cm}, "
                f"vlm_cl={self.vlm_cl}, vlm_cd={self.vlm_cd}, "
                f"cfd_data={self.cfd_data}, cfd_cm={self.cfd_cm})"
                f"cfd_cl={self.cfd_cl}, cfd_cd={self.cfd_cd}")
    
    # method to extract key names and data type 
    def get_key_names(self):
        return self.__dict__.keys()
    
