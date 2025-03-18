import numpy as np
from scipy.linalg import solve_banded # Thomas Algorithm
# Lagrangian Frame

# 1D RadHydro for Slab, Spherical, and Cylindrical
class Mesh:
    """
    Object to handle the mesh for slab, spherical, and cylindrical
        Inputs:
            - geom (str): the coord. system id.
                        - SLAB
                        - CYL
                        - SPH
            - dh (1d np.array(float64)): an array of the max cell widths per region.
            - hb (1d np.array(float64)): an array containing the bounds of each region.
            - verbosity (bool): Verbosity of the process (Default: False)
        Outputs: Generate the mesh vertices, cell centers, areas, and volumes 
    """
    def __init__(self, geom, dh, hb, verbosity=False):
        # Keys for acceptable grids
        valid_geoms = ['SLAB', 'CYL', 'SPH']
        try:
            idx = valid_geoms.index(geom)
            self.geom = valid_geoms[idx]
        except:
            self.geom = None
            valid_geom_str = ', '.join(valid_geoms)
            raise ValueError(f"Grid type {geom} not supported! Acceptable values are: {valid_geom_str}.")
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
            if self.geom == 'SLAB':
                self.A = np.ones(self.ncells + 1)
                self.V = cell_w
            elif self.geom == 'CYL':
                self.A = 2 * np.pi * self.vertices
                self.V = np.pi * (self.vertices[1:]**2 - self.vertices[:-1]**2)
            else: # self.geom == 'SPH'
                self.A = 4 * np.pi * self.vertices**2
                self.V = 4/3 * np.pi * (self.vertices[1:]**3 - self.vertices[:-1]**3)
            if self.verbose:
                header = f"{'Region':<10}{'Start':<10}{'End':<10}{'Cell Count':<12}"
                print(header)
                print("-" * len(header))
                # Iterate over each region and print the details in a formatted row
                for i, (start, end, count) in enumerate(zip(h_start, h_end, num_cells_per_zone), start=1):
                    print(f"{i:<10}{start:<10.3f}{end:<10.3f}{count:<12}")
                print('\n')

class RadHydroSolver:
    def __init__(self, mesh, matProps, bc, init_cond, CFL, max_dt, max_rel_rad_e, gamma, max_iter, xtol, solver_type):
        """
        Object that handles solving the 1D RadHydro problem.
        """
        # Acceptable solvers
        solvers = ['Hydro', 'Rad', 'RadHydro']
        try:
            idx = solvers.index(solver_type)
            self.solver = solvers[idx]
        except:
            self.solver = None
            valid_solver_str = ', '.join(solvers)
            raise ValueError(f"Solver type {solver_type} not supported! Acceptable values are: {valid_solver_str}.")
        self.mesh = mesh
        self.Cv = matProps['Cv']
        self.Ks = matProps['Ks']
        self.K1, self.K2, self.K3, self.n = matProps['Ka']
        self.CFL, self.max_dt, self.max_rel_rad_e = CFL, max_dt, max_rel_rad_e
        self.gamma = gamma
        self.max_iter, self.xtol = max_iter, xtol
        # Physical constants
        self.c = 299.792 # light speed
        self.a = 0.01372 # rad constant
        # Pressure, Energy, and Temperature are stored at cell centers
        self.P = init_cond['Pressure']
        self.rad_e = init_cond['Radiation Energy']
        self.mat_e = init_cond['Material Energy']
        self.T = init_cond["Temperature"]
        self.rho = init_cond['Density']
        self.m = self.rho * self.mesh.V
        # velocities are stored at cell vertices
        self.u = init_cond['Velocity']
        # Values of rad_e and pressure stored at boundaries
        self.rad_e_edge = np.zeros(2)
        self.P_edge = bc["Pressure"]
        self.u_edge = bc["Velocity"]
        self.rad_extr = bc["Radiation Energy"]

    def predictor_step(self):
        # Compute the edge rad_e
        Ka_0, Ka_N = self.compute_absorption_opacity(self.T[0]), self.compute_absorption_opacity(self.T[-1])
        Kt_0, Kt_N = Ka_0 + self.Ks, Ka_N + self.Ks
        self.rad_e_edge[0] = (3*self.rho[0]*np.diff(self.vertices[1] - self.vertices[0]) * Kt_0 * self.rad_extr[0]\
                              + 4*self.rad_e[0])/(3*self.rho[0]*np.diff(self.vertices[1] - self.vertices[0]) * Kt_0 + 4)
        self.rad_e_edge[-1] = (3*self.rho[-1]*np.diff(self.vertices[-1] - self.vertices[-2]) * Kt_N * self.rad_extr[-1]\
                              + 4*self.rad_e[-1])/(3*self.rho[-1]*np.diff(self.vertices[-1] - self.vertices[-2]) * Kt_N + 4)
        # Generate the half idx masses.
        m_halfs = 0.5 * (self.m[:-1] + self.m[1:])
        self.u_p = np.copy(self.u)
        dt = self.compute_time_step()
        # Calculate predictor u (u_p) #TODO: Treat boundary conditions
        self.u_p[1:-1] = -dt/m_halfs * (self.A[1:-1] \
                        * (self.P[1:] + 1/3*self.rad_e[1:] - self.P[:-1] - 1/3*self.rad_e[:-1]))
        if self.u_edge is None:
            self.u_p[0] = -self.mesh.A[0]*dt/(0.5 * self.rho[0]*self.mesh.V[0])*\
                (self.P[0] + 1/3*self.rad_e[0] - self.P_edge["Pressure"][0] - 1/3 * self.rad_e_edge[0])
            self.u_p[-1] = -self.mesh.A[-1]*dt/(0.5 * self.rho[-1]*self.mesh.V[-1])*\
                (self.P_edge["Pressure"][-1] + 1/3*self.rad_e_edge[-1] - self.P[-1] - 1/3 * self.rad_e[-1])
        else:
            self.u_p[0] = self.u_edge[0]
            self.u_p[-1] = self.u_edge[-1]
        # Compute a half time velocity
        u_k = 0.5 * (self.u + self.u_p)
        # Update vertices
        vertices_p = np.copy(self.mesh.vertices)
        vertices_p = self.mesh.vertices + u_k * dt
        # Update areas and volumes
        A_p, V_p = self.recompute_volumes_and_areas(vertices_p)
        # Update densities
        rho_p = self.recompute_densities(V_p)
        # Update radiaiton and material energy, but first we compute time avgs.
        A_k = 0.5 * (self.mesh.A + A_p)
        rho_k = 0.5 * (self.rho + rho_p)
        dh_k = 0.5 * (self.vertices + vertices_p)
        # Compute the predictor energy densities
        rad_e_k, mat_e_p = self.compute_energy_densities(dt, u_k, A_k, rho_k, dh_k, rho_p)
        # Update Temperature and Pressure
        T_p = mat_e_p/self.Cv
        P_p = (self.gamma - 1)*rho_p*mat_e_p
        # Compute time avg
        T_k = 0.5 * (self.T + T_p)
        P_k = 0.5 * (self.P + P_p)
        return A_k, rad_e_k, T_k, P_k
    
    def corrector_step(self, A_pk, rad_e_pk, T_pk, P_pk):
        """
        
        """
        



    # Utility methods
    def compute_absorption_opacity(self, T):
        """
        Member method to compute the absorption opacity
        """
        Ka = self.K1 / (self.K2*T**self.n + self.K3)
        return Ka

    def compute_time_step(self):
        """
        member method to solve for tor the optimal time step
        """
        dt_eps = self.max_rel_rad_e * self.rad_e * (self.dt) / (self.rad_e - self.rad_e_old)
        sound_speed = np.sqrt(self.gamma * self.P/self.rho)
        cell_widths = np.diff(self.mesh.vertices)
        #TODO Not sure if we should do aritmetic avg for cell-center speed
        propagation = cell_widths * self.CFL/(0.5 * (self.u[:-1] + self.u[1:]))
        sound_propagation = cell_widths * self.CFL / sound_speed
        min1, min2 = np.min(propagation), np.min(sound_propagation)
        return min(self.max_dt, dt_eps, min1, min2)

    def recompute_volumes_and_areas(self, vertices_new):
        """
        Member method to compute new areas and volumes at predictor step 
        """
        if self.mesh.geom == 'SLAB':
            A = np.ones(self.mesh.ncells + 1)
            V = np.diff(vertices_new)
        elif self.mesh.geom == 'CYL':
            A = 2 * np.pi * vertices_new
            V = np.pi * (vertices_new[1:]**2 - vertices_new[:-1]**2)
        else: # self.mesh.geom == 'SPH'
            A = 4 * np.pi * vertices_new**2
            V = 4/3 * np.pi * (vertices_new[1:]**3 - vertices_new[:-1]**3)
        return A, V
    
    def recompute_densities(self, V_new):
        """
        Member method to compute new densities at predictor step
        """
        rho_new = self.m / V_new
        return rho_new
    
    def compute_xi(self, u_k):
        """
        Member method to compute xi during predictor step for rad_e and mat_e
        """
        xi = -self.P * (self.mesh.A[1:]*u_k[1:] - self.mesh.A[:-1]*u_k[:-1])
        return xi

    def compute_nu(self, Ka, dt):
        """
        Member method to compute nu during predictor step for rad_e and mat_e
        """
        nu = (dt * Ka * 2* self.c * self.a * self.T**3)\
            /(self.Cv + dt * Ka * 2 *self.a * self.c * self.T**3)
        return nu

    def compute_energy_densities(self, dt, u_k, A_k, rho_k, dh_k, rho_new):
        """

        """
        # Compute absorp. opacities at cell center
        Ka = self.compute_absorption_opacity(T=self.T)
        # Compute nu 
        nu = self.compute_nu(Ka, dt)
        # Compute xi
        xi = self.compute_xi(u_k)
        
        # Compute avg T at interior vertices
        T_vert = ((self.T[:-1]**4 + self.T[1:]**4)/2)**0.25
        #TODO Treat bc here
        T_l = ((1/self.a*self.rad_e_edge[0] + self.T[0]**4)/2)**0.25
        T_r = ((1/self.a * self.rad_e_edge[1] + self.T[-1]**4)/2)**0.25
        T_vert = np.concatenate(([T_l], T_vert, [T_r]))
        Ka_t = self.compute_absorption_opacity(T_vert)
        # These are edge evalueated total opacities
        Kt = self.Ks + Ka_t
        # Build the tridiag system as banded matrix (Thomas Alg)
        Mb = np.zeros((3, self.mesh.ncells))
        y = np.zeros(self.mesh.ncells)
        # Do interior nodes first
        # superdiagonal
        Mb[0, 2:] = -(A_k[2:-1]*self.c)/(3*(rho_k[2:]*dh_k[2:]*Kt[2:-1] + rho_k[1:-1]*dh_k[1:-1]*Kt[2:-1]))
        # main diagonal #TODO: Might be using the wrong idx for Kt
        Mb[1, 1:-1] = self.m[1:-1]/(dt*rho_new[1:-1]) +  A_k[2:-1] *self.c / (3*(rho_k[1:-1]*dh_k[1:-1]*Kt[2:-1] + rho_k[2:]*dh_k[2:]*Kt[2:-1])) +\
            A_k[1:-2]*self.c/(3*(rho_k[1:-1]*dh_k[1:-1]*Kt[1:-2] + rho_k[:-2]*dh_k[:-2]*Kt[1:-2]))
        # subdiagonal  
        Mb[2, :-2] = -(A_k[1:-2]*self.c)/(3*(rho_k[1:-1]*dh_k[1:-2]*Kt[1:-2] + rho_k[:-2]*dh_k[:-2]*Kt[1:-2]))
        # rhs
        y[1:-1] =  self.m[1:-1]*self.rad_e[1:-1]/(dt * self.rho[1:-1]) + A_k[2:-1]*self.c*(self.rad_e[2:] - self.rad_e[1:-1])/\
            (3*(rho_k[1:-1]*dh_k[1:-1]*Kt[2:-1] + rho_k[2:]*dh_k[2:]*Kt[2:-1])) - A_k[1:-2]*self.c*(self.rad_e[1:-1] - self.rad_e[:-2])/\
                (3*(rho_k[1:-1]*dh_k[1:-1]*Kt[1:-2] + rho_k[:-2]*dh_k[:-2]*Kt[1:-2])) + self.m[1:-1]*Ka[1:-1]*self.c*(1-nu[1:-1])*(self.a*self.T[1:-1]**4 - 0.5 * self.rad_e[1:-1])\
                    + nu[1:-1]*xi[1:-1] - 1/3*self.rad_e[1:-1]*(self.mesh.A[2:]*u_k[2:] - self.mesh.A[1:-2]*u_k[1:-2])
        # Boundary Conditions
        # superdiagonal
        Mb[0, 1] = -A_k[1]*self.c/(3*(rho_k[0]*dh_k[0]*Kt[1] + rho_k[1]*dh_k[1]*Kt[1]))
        # main diagonal
        Mb[1, 0] = self.m[0]/(dt*rho_new[0]) + A_k[1]*self.c/(3*(rho_k[0]*dh_k[0]*Kt[1] + rho_k[1]*dh_k[1]*Kt[1])) +\
                    A_k[0]*self.c/(3*rho_k[0]*dh_k[0]*Kt[0] + 4) + 0.5*self.m[0]*Ka[0]*self.c*(1-nu[0])
        Mb[1, -1] = self.m[-1]/(dt*rho_new[-1]) + A_k[-1]*self.c/(3*rho_k[-1]*dh_k[-1]*Kt[-1]+4) + A_k[-2]*self.c/\
                    (3*(rho_k[-1]*dh_k[-1]*Kt[-2]+rho_k[-2]*dh_k[-2]*Kt[-2])) + 0.5*self.m[-1]*Ka[-1]*self.c*(1-nu[-1])
        # subdiagonal
        Mb[2, -2] = -A_k[-2]*self.c/(3*(rho_k[-1]*dh_k[-1]*Kt[-2]+rho_k[-2]*dh_k[-2]*Kt[-2]))
        y[0] = self.m[0]*self.rad_e[0]/(dt*self.rho[0]) + A_k[1]*self.c*(self.rad_e[1]-self.rad_e[0])/(3*(rho_k[0]*dh_k[0]*Kt[1] + rho_k[1]*dh_k[1]*Kt[1])) -\
                A_k[0]*self.c*(self.rad_e[0]-2*self.rad_extr[0])/(3*rho_k[0]*dh_k[0]*Kt[0] + 4) + self.m[0]*Ka[0]*self.c*(1-nu[0])*(self.a*self.T[0]**4-0.5*self.rad_e[0])+\
                nu[0]*xi[0] - 1/3*self.rad_e[0]*(self.mesh.A[1]*u_k[1]-self.mesh.A[0]*u_k[0])
        y[-1] = self.m[-1]*self.rad_e[-1]/(dt*self.rho[-1]) + A_k[-1]*self.c*(2*self.rad_extr[-1]-self.rad_e[-1])/(3*rho_k[-1]*dh_k[-1]*Kt[-1]+4) -\
                A_k[-2]*self.c*(self.rad_e[-1]-self.rad_e[-2])/(3*(rho_k[-1]*dh_k[-1]*Kt[-2]+rho_k[-2]*dh_k[-2]*Kt[-2])) +\
                self.m[-1]*Ka[-1]*self.c*(1-nu[-1])*(self.a*self.T[-1]**4-0.5*self.rad_e[-1]) + nu[-1]*xi[-1] -\
                    1/3*self.rad_e[-1]*(self.mesh.A[-1]*u_k[-1]-self.mesh.A[-2]*u_k[-2])
        # Solve for predictor rad_e
        rad_e_p = solve_banded((1, 1), Mb, y)
        # Compute time avg for rad_e
        rad_e_k = 0.5 * (self.rad_e + rad_e_p)
        mat_e_p = self.mat_e + (dt*self.Cv*(self.m*Ka*self.c*(rad_e_k - self.a*self.T**4) + xi))/\
            (self.m*self.Cv + dt*self.m*Ka*2*self.a*self.c*self.T**3)
        return rad_e_k, mat_e_p




         

        



        


if __name__ == "__main__":
    hb = np.array([1])
    dh = np.array([.02])
    Cv_list = np.array([0.1])
    Ka_list = np.array([20])
    Ks_list = np.array([0.5])
    matProps = {"Cv": 0.1,
                "Ka": [20, 0, 1, 1], #K1, K2, K3, n
                "Ks": 0.5}
    
    verbose = True
    mesh = Mesh("SLAB", dh, hb, verbose)
    mesh.generate_uniform_submesh()


