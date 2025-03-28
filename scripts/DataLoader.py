"""
Import CFD runs to extract aerodynamic coefficients and project CFD data to the lattice.
Y. Shen, Oct. 2024
"""

import vtk
import os
import numpy as np
import pickle
from vtk.util.numpy_support import vtk_to_numpy     # type: ignore

class DataLoader:
    """
    DataLoader class to load CFD data and extract aerodynamic coefficients. 
    Relies on VTK package for data loading.
    Args:
        cfd_path (str): Path to the CFD data file.
    """
    def __init__(self, cfd_path):
        self.cfd_path = cfd_path
        self.reader = None

    def extract_aero_coefficients(self, filepath):
        """
        Read SU2 force_breakdown.dat file and extract aerodynamic coefficients.
        Args:
            filepath (str): Path to the force_breakdown.dat file.
        Returns:
            tuple: Cl, Cd, Cm, Cs, Cfx, Cfy, Cfz (lift, drag, moment, side force, x-force, y-force, z-force)
        """
        with open(filepath, 'r') as file:
            lines = file.readlines()

        cl, cd, cm, csf, cfx, cfy, cfz = None, None, None, None, None, None, None
        in_body_section = False

        for line in lines:
            if "Surface name: body" in line:
                in_body_section = True
            if in_body_section:
                if "Total CL    (" in line:
                    cl = float(line.split()[4])
                if "Total CD" in line:
                    cd = float(line.split()[4])
                if "Total CSF" in line:
                    csf = float(line.split()[4])
                if "Total CMy" in line:
                    cm = float(line.split()[4])
                if "Total CFx" in line:
                    cfx = float(line.split()[4])
                if "Total CFy" in line:
                    cfy = float(line.split()[4])
                if "Total CFz" in line:
                    cfz = float(line.split()[4])                
                
                    break  # Exit after finding Cm to avoid unnecessary parsing

        if cl is None or cd is None or cm is None:
            raise ValueError("Failed to extract aerodynamic coefficients from the file.")
        
        return cl, cd, cm, csf, cfx, cfy, cfz    

    def load_single_file(self, filename):
        """
        Load a single VTK file.
        Args:
            filename (str): The filename to load.
        Returns:
            data (vtkUnstructuredGrid): The loaded VTK data
        """
        # get extension of filename 
        file_extension = os.path.splitext(filename)[-1].lower()
        if file_extension == ".vtu":
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif file_extension == ".vtk":
            reader = vtk.vtkUnstructuredGridReader()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        self.reader = reader
        reader.SetFileName(filename)
        reader.Update()
        data = reader.GetOutput()

        print("Point Data Arrays:")
        point_data = data.GetPointData()
        for i in range(point_data.GetNumberOfArrays()):
            print(f"{i}: {point_data.GetArrayName(i)}")
        return data

    def load_cfd_data(self):
        """
        Loading a specific surface_flow.vtu data to cell and vertex data arrays
        Returns:
            data (vtkUnstructuredGrid): The loaded VTK data.
            mesh_info (dict): Basic information about the mesh.
        """
        # Load the .vtu file data
        curdir = os.getcwd()
        filepath = os.path.join(curdir, self.cfd_path)
        data = self.load_single_file(filepath)

        # Extract basic information about the mesh
        points = data.GetPoints()
        num_points = points.GetNumberOfPoints()
        num_cells = data.GetNumberOfCells()

        # Extract point data arrays
        point_data = data.GetPointData()
        point_data_arrays = {point_data.GetArrayName(i): point_data.GetArray(i) \
                             for i in range(point_data.GetNumberOfArrays())}

        # Extract cell data arrays
        cell_data = data.GetCellData()
        cell_data_arrays = {cell_data.GetArrayName(i): cell_data.GetArray(i) \
                            for i in range(cell_data.GetNumberOfArrays())}

        mesh_info = {
            "num_points": num_points,
            "num_cells": num_cells,
            "point_data_arrays": list(point_data_arrays.keys()),
            "cell_data_arrays": list(cell_data_arrays.keys())
        }

        return data, mesh_info


    def get_array(self, data, array_name='Pressure_Coefficient'):
        array = vtk_to_numpy(data.GetPointData().GetArray(array_name))
        if array is None:
            raise ValueError(f"No '{array_name}' array found in the point data")
        return array
    
    def get_cell_array(self, data, array_name='CellArea'):
        cell_data = data.GetCellData()
        if not cell_data:
            raise ValueError("No cell data found in the dataset")

        array = cell_data.GetArray(array_name)
        if array is None:
            available_arrays = [cell_data.GetArrayName(i) \
                                for i in range(cell_data.GetNumberOfArrays())]
            raise ValueError(f"No '{array_name}' array found in the cell data. Available arrays: {available_arrays}")
        
        return vtk_to_numpy(array)

    
    def parse_lattice_corners(self, VD):
        lattices = []

        for i in range(len(VD.XA1)):
            lattice = [
                [VD.XA1[i], VD.YA1[i], VD.ZA1[i]],
                [VD.XB1[i], VD.YB1[i], VD.ZB1[i]],
                [VD.XB2[i], VD.YB2[i], VD.ZB2[i]],
                [VD.XA2[i], VD.YA2[i], VD.ZA2[i]]
            ]
            lattices.append(lattice)

        return lattices
    
    def get_xyz_and_connectivity(self, data):
        points = data.GetPoints()
        num_points = points.GetNumberOfPoints()
        xyz_coordinates = [points.GetPoint(i) for i in range(num_points)]

        cells = data.GetCells()
        connectivity = []
        cells.InitTraversal()
        id_list = vtk.vtkIdList()
        while cells.GetNextCell(id_list):
            connectivity.append([id_list.GetId(j) for j in range(id_list.GetNumberOfIds())])

        return xyz_coordinates, connectivity

    def find_cells_in_lattice(self, lattice, xyz_coordinates, pressure_coeff, vertex_area):
        from matplotlib.path import Path
        path = Path([corner[:2] for corner in lattice])
        upper_cp, lower_cp, lower_area, upper_area = [], [], [], []

        for i, coord in enumerate(xyz_coordinates):
            if path.contains_point(coord[:2]):
                if coord[2] > 0:
                    upper_cp.append(pressure_coeff[i])
                    upper_area.append(vertex_area[i])
                else:
                    lower_cp.append(pressure_coeff[i])
                    lower_area.append(vertex_area[i])

        average_pressure = np.average(pressure_coeff)

        total_upper_area = np.sum(upper_area)
        total_lower_area = np.sum(lower_area)
        if total_lower_area * total_upper_area == 0:
            Warning("No cells found in the lattice ")
        avg_cp_upper = np.sum(np.array(upper_cp) * np.array(upper_area) ) / total_upper_area if total_upper_area != 0 else average_pressure
        avg_cp_lower = np.sum(np.array(lower_cp) * np.array(lower_area) ) / total_lower_area if total_lower_area != 0 else average_pressure
        if avg_cp_upper * avg_cp_lower == 0:
            Warning("0 pressure found in the lattice")

        return avg_cp_upper, avg_cp_lower

    def project_cfd_to_lattice(self, lattices, xyz_coordinates, pressure_coeff, vertex_area):
        results, cp_u, cp_l = [], [], []
        for i, lattice in enumerate(lattices):
            print(f"Processing lattice {i}")
            avg_cp_upper, avg_cp_lower = self.find_cells_in_lattice(lattice, xyz_coordinates, pressure_coeff,
                                                                    vertex_area)
            
            delta_cp = avg_cp_lower - avg_cp_upper
            print(f"Average Cp upper: {avg_cp_upper}, Average Cp lower: {avg_cp_lower}, Delta Cp: {delta_cp}")
            results.append(delta_cp)
            cp_u.append(avg_cp_upper)
            cp_l.append(avg_cp_lower)

        return results, cp_u, cp_l


# Usage example:
# cfd_path = "GNN_Test/surface_flow.vtu"
# dataloader = DataLoader(cfd_path)
# data, mesh_info = dataloader.load_cfd_data() # singleblock data (.vtu file) 
# xyz_coordinates = [...]  # Extract from data
# pressure_coeff = vtk_to_numpy(data.GetPointData().GetArray('Pressure_Coefficient'))
# mfdata = MFData(...)
# lattices = dataloader.parse_lattice_corners(mfdata)
# cfd_results = dataloader.project_cfd_to_lattice(lattices, xyz_coordinates, pressure_coeff)
# mfdata.cfd_data = cfd_results
# mfdata.save_with_pickle("mfdata.pkl")