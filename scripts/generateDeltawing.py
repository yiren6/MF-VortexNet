"""
Code to build Delta wing and setup analyses using SUAVE library 
(c) Yiren Shen

Initial Date: Oct 20, 2024

Modification: 
"""

import SUAVE
assert SUAVE.__version__ == '2.5.2', 'These tutorials only work with the SUAVE 2.5.2 release'
from SUAVE.Core import Units, Data 
import sys
from SUAVE.Plots.Performance.Mission_Plots import *
from SUAVE.Methods.Propulsion.turbojet_sizing import turbojet_sizing
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_segmented_planform
from SUAVE.Plots.Geometry import *
from SUAVE.Input_Output.OpenVSP import write
from SUAVE.Input_Output.OpenVSP.vsp_read import vsp_read
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
VLM_path = './scripts'
sys.path.append(VLM_path)
from VLM import VLM
import pylab as plt
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class NACA4DigitAirfoil:
    def __init__(self, m=0, p=0, t=12, chord_length=1.0):
        """
        Parameters:
        - m: Maximum camber as percentage of chord (e.g., 2 for 0.02 or 2%)
        - p: Position of maximum camber as tenth of chord (e.g., 4 for 0.4 or 40%)
        - t: Maximum thickness as percentage of chord (e.g., 12 for 0.12 or 12%)
        - chord_length: Length of the airfoil chord (default: 1.0)
        """
        self.m = m / 100.0
        self.p = p / 10.0
        self.t = t / 100.0
        self.chord_length = chord_length

    def thickness_distribution(self, x):
        """
        Thickness distribution for the NACA 4-digit airfoil.
        """
        t = self.t
        return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 +
                        0.2843 * x**3 - 0.1036 * x**4)

    def camber_line(self, x):
        """
        Camber line for the NACA 4-digit airfoil.
        """
        m, p = self.m, self.p
        yc = np.where(
            x < p,
            m * (2 * p * x - x**2) / (p**2) if p != 0 else 0,
            m * (1 - 2 * p + 2 * p * x - x**2) / ((1 - p)**2) if p != 0 else 0
        )
        return yc

    def camber_slope(self, x):
        """
        Derivative of the camber line.
        """
        m, p = self.m, self.p
        dyc_dx = np.where(
            x < p,
            2 * m * (p - x) / (p**2) if p != 0 else 0,
            2 * m * (p - x) / ((1 - p)**2) if p != 0 else 0
        )
        return dyc_dx

    def generate_coordinates(self, num_points=100):
        """
        Generate upper and lower surface coordinates for the NACA 4-digit airfoil.
        """
        x = np.linspace(0, 1, num_points)
        yt = self.thickness_distribution(x)
        yc = self.camber_line(x)
        dyc_dx = self.camber_slope(x)

        theta = np.arctan(dyc_dx)

        # Upper and lower surface coordinates
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Combine coordinates for airfoil plot
        x_coords = np.concatenate([xu[::-1], xl[1:]])
        y_coords = np.concatenate([yu[::-1], yl[1:]])

        return x_coords * self.chord_length, y_coords * self.chord_length

    def write_coordinates(self, filename):
        """
        Write airfoil coordinates to a file.
        """
        x, y = self.generate_coordinates()
        with open(filename, 'w') as f:
            f.write(f'NACA{int(self.m * 100)}{int(self.p * 10)}{int(self.t * 100)}\n')
            for xi, yi in zip(x, y):
                f.write(f'{xi:.6f} {yi:.6f}\n')

def vehicle_setup(LE_SWEEP, NACA_4DIGITS):
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'DeltaWing'

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------

    # basic parameters
    vehicle.mass_properties.center_of_gravity =  [[0.00001, 0.0, 0.0]] 

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    # wing geom
    wing.sweeps.leading_edge = LE_SWEEP * Units.deg
    wing.taper                   = 0.0
    wing.chords.root             = 25.734 * Units.inches
    wing.total_length            = 25.734 * Units.inches
    wing.chords.tip              = 0.0 * Units.meter
    wing.chords.mean_aerodynamic = 17.156 * Units.inches
    wing.origin                  = [[0.0, 0.0, 0.0]] 
    wing.aerodynamic_center      = [0.4357624, 0.0, 0.0]  # 2/3 root chord, TM4645 pp 4
    semi_span = wing.chords.root / np.tan(LE_SWEEP * Units.deg)
    wing.spans.projected         = semi_span * 2
    wing.areas.reference         = wing.chords.root * semi_span
    wing.aspect_ratio            = (wing.spans.projected)**2 / wing.areas.reference
    wing.thickness_to_chord      = NACA_4DIGITS['t'] / 100.0
    wing.areas.wetted            = 2 * wing.areas.reference
    
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False
    wing.vortex_lift             = True
    wing.high_mach               = True
    wing.dynamic_pressure_ratio  = 1.0
    # compute quarter chord sweep
    wing_sweeps_quarter_chord = convert_sweep(wing, old_ref_chord_fraction=0.0, new_ref_chord_fraction=0.25)

    # Generate the NACA 4-digit airfoil
    airfoil_generator = NACA4DigitAirfoil(
        m=NACA_4DIGITS['m'],
        p=NACA_4DIGITS['p'],
        t=NACA_4DIGITS['t'],
        chord_length=NACA_4DIGITS['chord_length']
    )

    # Write airfoil coordinates to a file
    airfoil_filename = f"NACA{NACA_4DIGITS['m']}{NACA_4DIGITS['p']}{NACA_4DIGITS['t']}.dat"
    airfoil_generator.write_coordinates(airfoil_filename)

    # Create an airfoil object
    airfoil = SUAVE.Components.Airfoils.Airfoil()
    airfoil.coordinate_file = airfoil_filename

    # Wing Segments
    # Segment 1: Root
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                             = 'Root'
    segment.percent_span_location           = 0.0
    segment.twist                           = 0.0 * Units.deg
    segment.root_chord_percent              = 1.0
    segment.dihedral_outboard               = 0.0 * Units.deg
    segment.sweeps.quarter_chord            = wing_sweeps_quarter_chord
    segment.thickness_to_chord              = wing.thickness_to_chord
    segment.append_airfoil(airfoil)
    wing.append_segment(segment)

    # Segment 2: Tip
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                           = 'Tip'
    segment.percent_span_location         = 1.0
    segment.twist                         = 0.0 * Units.deg
    segment.root_chord_percent            = 0.0
    segment.dihedral_outboard             = 0.0 * Units.degrees
    segment.sweeps.quarter_chord          = wing_sweeps_quarter_chord
    segment.thickness_to_chord            = wing.thickness_to_chord
    segment.append_airfoil(airfoil)
    wing.append_segment(segment)

    # Update wing planform
    wing = wing_segmented_planform(wing)

    # converted from vsp file assuming original unit in inch
    vehicle.reference_area               = wing.areas.reference 
    vehicle.total_length                 = wing.chords.root

    # Add wing to vehicle
    vehicle.append_component(wing)

    return vehicle

# ----------------------------------------------------------------------
#   Define the Configurations
# ----------------------------------------------------------------------
def full_setup(LE_SWEEP, NACA_4DIGITS):
    """
    Setup the vehicle and analyses for the deltawing configuration
    """
    # vehicle data
    vehicle = vehicle_setup(LE_SWEEP, NACA_4DIGITS)
    configs  = configs_setup(vehicle, LE_SWEEP, NACA_4DIGITS)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)
    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses

    return configs, analyses


def configs_setup(vehicle, LE_SWEEP, NACA_4DIGITS):
    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = f"deltawing_sweep_{LE_SWEEP}_naca_{NACA_4DIGITS['m']}{NACA_4DIGITS['p']}{NACA_4DIGITS['t']}"
    configs.append(base_config)
    write(vehicle, base_config.tag, write_igs=True)

    return configs


def generate_family_of_delta_wings(le_sweep_angles, naca_variations):
    for le_sweep in le_sweep_angles:
        for naca_params in naca_variations:

            # Define LE_SWEEP and NACA_4DIGITS
            LE_SWEEP = le_sweep
            NACA_4DIGITS = naca_params

            # Setup vehicle with specified sweep angle and airfoil
            vehicle = vehicle_setup(LE_SWEEP, NACA_4DIGITS)
            configs = configs_setup(vehicle, LE_SWEEP, NACA_4DIGITS)

            config_tag = f"deltawing_sweep_{LE_SWEEP}_naca_{NACA_4DIGITS['m']}{NACA_4DIGITS['p']}{NACA_4DIGITS['t']}"
            base_config = configs[config_tag]
            print(f"Generated configuration: {config_tag}")

    return 0

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------
def analyses_setup(configs):
    """
    Function to setup the analyses environment for the vehicle
    """
    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis

    return analyses
    
# ----------------------------------------------------------------------
#   VLM Settings
# ----------------------------------------------------------------------
def get_settings():
    """
    Function to set the settings for the VLM analysis
    """
    settings = SUAVE.Analyses.Aerodynamics.Vortex_Lattice().settings
    settings.number_spanwise_vortices        = 15
    settings.number_chordwise_vortices       = 30   
    settings.propeller_wake_model            = None
    settings.spanwise_cosine_spacing         = True
    settings.model_fuselage                  = True
    settings.model_nacelle                   = True
    settings.leading_edge_suction_multiplier = 1
    settings.discretize_control_surfaces     = False
    settings.use_VORLAX_matrix_calculation   = True    
                
    #misc settings
    settings.show_prints = False
    
    return settings

def base_analysis(vehicle):
    """
    Function to setup the base analysis for the vehicle
    """
    #   Initialize the Analyses
    analyses = SUAVE.Analyses.Vehicle()

    #  Basic Geometry Relations
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)
    
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Supersonic_Zero()    # Aerodynamics.
    aerodynamics.geometry = vehicle    
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    aerodynamics.settings.span_efficiency            = .8    
    analyses.append(aerodynamics)

    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks 
    analyses.append(energy)

    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)   

    # done!
    return analyses    


def get_conditions(machs, altitudes, aoas, PSIs, PITCHQs, YAWQs, ROLLQs):
    """
    Function to set the conditions for the VLM analysis
    """

    conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    atmosphere                              = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    speeds_of_sound                         = atmosphere.compute_values(altitudes).speed_of_sound
    v_infs                                  = machs * speeds_of_sound.flatten() 
    conditions.freestream.velocity          = np.atleast_2d(v_infs).T 
    conditions.freestream.mach_number       = np.atleast_2d(machs).T 
    conditions.aerodynamics.angle_of_attack = np.atleast_2d(aoas).T 
    conditions.aerodynamics.side_slip_angle = np.atleast_2d(PSIs).T 
    conditions.stability.dynamic.pitch_rate = np.atleast_2d(PITCHQs).T 
    conditions.stability.dynamic.roll_rate  = np.atleast_2d(ROLLQs).T 
    conditions.stability.dynamic.yaw_rate   = np.atleast_2d(YAWQs).T 
    
    return conditions


def point_analysis(vehicle, LE_SWEEP, NACA_4DIGITS, AOA, Ma, if_plot = False, DCP_overwrite=None, SPC_enforce = -1):
    """
    Function perform analysis for a single point (free stream conditions)
    Return and examine the panel lift distribution 
    """

    alpha = AOA * Units.deg
    pitch_rate = 0.0 * Units.deg    # pitch rate
    yaw_rate = 0.0 * Units.deg      # yaw rate
    roll_rate = 0.0 * Units.deg     # roll rate
    M_inf = Ma                      # Mach number
    Alt = 5000 * Units.ft           # Altitude
    PSIs = 0.0 * Units.deg          # side slip angle

    # get settings and conditions
    conditions = get_conditions(M_inf, Alt, alpha, PSIs, pitch_rate, yaw_rate, roll_rate)
    settings   = get_settings()

    # run VLM
    geometry    = vehicle_setup(LE_SWEEP, NACA_4DIGITS)
    data        = VLM(conditions, settings, geometry, DCP_overwrite=DCP_overwrite, SPC_enforce = SPC_enforce)
    plot_title  = geometry.tag

    # save/load results
    results = Data()
    results.CL         =  data.CL
    results.CDi        =  data.CDi
    results.CM         =  data.CM
    results.CYTOT      =  data.CYTOT        # Total y force coeff
    results.CRTOT      =  data.CRTOT        # Rolling moment coeff (unscaled)
    results.CRMTOT     =  data.CRMTOT       # Rolling moment coeff (scaled by w_span)
    results.CNTOT      =  data.CNTOT        # Yawing  moment coeff (unscaled)
    results.CYMTOT     =  data.CYMTOT       # Yawing  moment coeff (scaled by w_span)
    results.VD         =  data.VD           # Votext location coordinate
    results.V_distribution = data.V_distribution # Free stream velocity 
    results.gamma       = data.gamma            # circulation distribution
    results.cp          = data.CP               # pressure coefficient distribution   
    results.alpha_local = data.alpha_local
    results.beta_local  = data.beta_local
    results.gamma_local = data.gamma_local
    results.theta_x     = data.theta_x
    results.theta_y     = data.theta_y
    results.theta_z     = data.theta_z
    results.v_x          = data.v_total_x
    results.v_y          = data.v_total_y
    results.v_z          = data.v_total_z
    results.A = data.A
    results.RHS = data.RHS
    results.RNMAX = data.RNMAX
    results.CHORD = data.CHORD
    results.DCPSID = data.DCPSID
    results.FACTOR = data.FACTOR

    return results

def main():

    all_sweep = [55, 65, 75]
    all_airfoil = [
        {'m': 0, 'p': 0, 't': 10, 'chord_length': 1.0},
        {'m': 0, 'p': 0, 't': 16, 'chord_length': 1.0},
        {'m': 0, 'p': 0, 't': 24, 'chord_length': 1.0},
        {'m': 2, 'p': 4, 't': 16, 'chord_length': 1.0},
        {'m': 4, 'p': 4, 't': 16, 'chord_length': 1.0}
    ]

    generate_family_of_delta_wings(all_sweep, all_airfoil)

if __name__ == '__main__':
    main()
