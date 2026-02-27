import numpy as np
import pickle

class MFData:
    def __init__(self, vlm_data=None, vlm_cm=0.0, vlm_cl=0.0, vlm_cd=0.0, cfd_data=None, \
                 cfd_cm=0.0, cfd_cl= 0.0, cfd_cd = 0.0, vlm_vd = None):
        self.test_number = -999
        self.mach = -999
        self.alpha = -999
        self.Re = -999
        self.ref_pressure = -999
        self.vlm_data = np.array(vlm_data) if vlm_data is not None else np.array([])
        self.vlm_cm = vlm_cm
        self.vlm_cl = vlm_cl
        self.vlm_cd = vlm_cd
        self.vlm_cdi = 0.0
        self.vlm_cytot = 0.0
        self.vlm_crtot = 0.0
        self.vlm_crmtot = 0.0
        self.vlm_cntot = 0.0
        self.vlm_cymtot = 0.0
        self.vlm_v_distribution = []
        self.vlm_gamma = []
        self.vlm_cp = []
        self.alpha_local = []
        self.beta_local = []
        self.gamma_local = []
        self.theta_x =[]
        self.theta_y = []
        self.theta_z = []
        self.vlm_vx = []
        self.vlm_vy = []
        self.vlm_vz = []
        self.thickness = []
        self.spanwise_slope_u = []
        self.chordwise_slope_u = []
        self.gaussian_curvature_u = []
        self.spanwise_slope_l = []
        self.chordwise_slope_l = []
        self.gaussian_curvature_l = []


        self.cfd_data = np.array(cfd_data) if cfd_data is not None else np.array([])
        self.cfd_cpl = np.array(cfd_data) if cfd_data is not None else np.array([])
        self.cfd_cpu = np.array(cfd_data) if cfd_data is not None else np.array([])
        self.cfd_cm = cfd_cm
        self.cfd_cl = cfd_cl
        self.cfd_cd = cfd_cd
        self.cfd_cfx = 0.0
        self.cfd_cfy = 0.0
        self.cfd_cfz = 0.0
        self.cfd_csf = 0.0
        self.cfd_refP = 0.0
        self.vlm_vd = vlm_vd
        self.vlm_A = []
        self.vlm_RHS = []
        self.vlm_RNMAX = []
        self.vlm_CHORD = []
        self.vlm_DCPSID = []
        self.vlm_FACTOR = []


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
    
