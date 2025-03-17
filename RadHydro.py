import numpy as np


# 1D RadHydro for Slab, Spherical, and Cylindrical
class Mesh:
    """
    Object to handle the mesh for slab, spherical, and cylindrical
        Inputs:
            grid_type: (str) the coord. system id.
                        - SLAB
                        - CYL
                        - SPH
            dh: (1d np.array(float64)) an array of the max cell widths per region.
            hb: (1d np.array(float64)) an array containing the bounds of each region.
            verbosity: (bool) Verbosity of the simulation (Default: False)
        Outputs: Generate the mesh vertices, cell centers, areas, and volumes 
    """
    def __init__(self, grid_type, dh, hb, verbosity=False):
        # Keys for acceptable grids
        acceptable_grids = ['SLAB', 'CYL', 'SPH']
        try:
            idx = acceptable_grids.index(grid_type)
            self.grid_type = acceptable_grids[idx]
        except:
            self.grid_type = None
            acceptable_grid_str = ', '.join(acceptable_grids)
            raise ValueError(f"Grid type {grid_type} not supported! Acceptable values are: {acceptable_grid_str}.")
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
            self.x_centers = self.vertices[:-1] + self.dh / 2 
            # Handle Areas and Volumes
            if self.grid_type == 'SLAB':
                self.A = np.ones(self.ncells + 1)
                self.V = np.diff(self.vertices)
            elif self.grid_type == 'CYL':
                self.A = 2 * np.pi * self.vertices
                self.V = np.pi * (self.vertices[1:]**2 - self.vertices[:-1]**2)
            else: # self.grid_type == 'SPH'
                self.A = 4 * np.pi * self.vertices**2
                self.V = 4/3 * np.pi * (self.vertices[1:]**3 - self.vertices[:-1]**3)
            if self.verbose == True:
                header = f"{'Region':<10}{'Start':<10}{'End':<10}{'Cell Count':<12}"
                print(header)
                print("-" * len(header))
                # Iterate over each region and print the details in a formatted row
                for i, (start, end, count) in enumerate(zip(h_start, h_end, num_cells_per_zone), start=1):
                    print(f"{i:<10}{start:<10.3f}{end:<10.3f}{count:<12}")



if __name__ == "__main__":
    mesh = Mesh(grid_type="SLAB", hb=np.array([1]), dh=np.array([.02]), verbosity=True)
    mesh.generate_uniform_submesh()

