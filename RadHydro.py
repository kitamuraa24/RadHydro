import numpy as np


# 1D RadHydro for Slab, Spherical, and Cylindrical
class Mesh:
    """
    Object to handle the mesh for slab, spherical, and cylindrical
        Inputs:
            - grid_type (str): the coord. system id.
                        - SLAB
                        - CYL
                        - SPH
            - dh (1d np.array(float64)): an array of the max cell widths per region.
            - hb (1d np.array(float64)): an array containing the bounds of each region.
            - verbosity (bool): Verbosity of the process (Default: False)
        Outputs: Generate the mesh vertices, cell centers, areas, and volumes 
    """
    def __init__(self, grid_type, dh, hb, verbosity=False):
        # Keys for acceptable grids
        valid_grids = ['SLAB', 'CYL', 'SPH']
        try:
            idx = valid_grids.index(grid_type)
            self.grid_type = valid_grids[idx]
        except:
            self.grid_type = None
            valid_grid_str = ', '.join(valid_grids)
            raise ValueError(f"Grid type {grid_type} not supported! Acceptable values are: {valid_grid_str}.")
        self.dh = dh
        self.hb = hb
        self.verbose = verbosity
    
    def generate_uniform_submesh(self):
        """
        Member method to generate uniform grids across multiple zones.
        """
        if len(self.dh) != len(self.hb):
            raise ValueError("Provided hb and dh do not match! Ensure the no. of dh matches the no. of bounds!") 
        else:
            # Create (start, end) pairs
            h_start = np.concatenate(([0], self.hb[:-1]))
            h_end = self.hb
            # Array of cells per zone
            num_cells_per_zone = np.floor((h_end - h_start)/ self.dh).astype(int)
            self.ncells = np.sum(num_cells_per_zone)
            temp_verts = np.concatenate([np.linspace(start, end, num + 1) for start, end, num in zip(h_start, h_end, num_cells_per_zone)])
            # Need to remove duplicate nodes
            self.vertices = np.unique(temp_verts)
            cell_w = np.diff(self.vertices)
            self.centroids = self.vertices[:-1] + cell_w / 2 
            # Handle Areas and Volumes
            if self.grid_type == 'SLAB':
                self.A = np.ones(self.ncells + 1)
                self.V = cell_w
            elif self.grid_type == 'CYL':
                self.A = 2 * np.pi * self.vertices
                self.A = np.insert(self.A, 0, 0)
                self.V = np.pi * (self.vertices[1:]**2 - self.vertices[:-1]**2)
                self.V = np.insert(self.V, 0, 0)
            else: # self.grid_type == 'SPH'
                self.A = 4 * np.pi * self.vertices**2
                self.A = np.insert(self.A, 0, 0)
                self.V = 4/3 * np.pi * (self.vertices[1:]**3 - self.vertices[:-1]**3)
                self.V = np.insert(self.V, 0, 0)
            if self.verbose:
                header = f"{'Region':<10}{'Start':<10}{'End':<10}{'Cell Count':<12}"
                print(header)
                print("-" * len(header))
                # Iterate over each region and print the details in a formatted row
                for i, (start, end, count) in enumerate(zip(h_start, h_end, num_cells_per_zone), start=1):
                    print(f"{i:<10}{start:<10.3f}{end:<10.3f}{count:<12}")
                print('\n')

class MaterialProperties:
    """
    Object to handle the material properties of the problem.
        Inputs:
            - hb (np.array(float64)): array of bounds for each zone.
            - centers (np.array(float64)): array of cell centers (from Mesh).
            - ncells (int): total no. of cells (from Mesh).
            - Cv_list (np.array(float64)): array of specific heat capacities per zone.
            - Ka_list (np.array(float64)): array of absoption opacities per zone.
            - Ks_list (np.array(float64)): array of scattering opacities per zone.
            - verbosity (bool): verbosity of the process (Default: False)
    """
    def __init__(self, hb, centroids, ncells, Cv_list, Ka_list, Ks_list, verbosity=False):
        if not (len(Cv_list) == len(Ka_list) == len(Ks_list) == len(hb)):
            raise ValueError(
                f"Material property lists must have length {len(hb)} (len(hb)) , "
                f"but got lengths: Cv_list={len(Cv_list)}, Ka_list={len(Ka_list)}, Ks_list={len(Ks_list)}"
            )
        
        self.hb = hb
        self.centroids = centroids
        self.ncells = ncells
        self.Cv_list = Cv_list
        self.Ka_list = Ka_list
        self.Ks_list = Ks_list
        self.verbose = verbosity

    def generate_matProps(self):
        """
        Member method to generate the material idx and properties global mapping
        """
        # Create material mapping for cells
        mat_idx = np.searchsorted(self.hb, self.centroids, side='right')
        self.Cv = self.Cv_list[mat_idx]
        self.Ka = self.Ka_list[mat_idx]
        self.Ks = self.Ks_list[mat_idx]
        if self.verbose:
            # Get unique material indices and count the number of cells per material
            unique_materials, counts = np.unique(mat_idx, return_counts=True)
            header = f"{'Material Id':<20}{'Number of Cells':<20}"
            print(header)
            print("-" * len(header))
            for material, count in zip(unique_materials, counts):
                print(f"{material:<20}{count:<20}")
            print('\n')






if __name__ == "__main__":
    hb = np.array([1])
    dh = np.array([.02])
    Cv_list = np.array([0.1])
    Ka_list = np.array([20])
    Ks_list = np.array([0.5])
    verbose = True
    mesh = Mesh("SLAB", dh, hb, verbose)
    mesh.generate_uniform_submesh()
    matProps = MaterialProperties(hb, mesh.centroids, mesh.ncells, Cv_list, Ka_list, Ks_list, verbose)
    matProps.generate_matProps()

