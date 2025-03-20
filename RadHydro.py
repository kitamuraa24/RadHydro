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
    def __init__(self, mesh, matProps, bc, init_cond, CFL, max_dt, max_rel_rad_e, time, solver_type, verbose=False):
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
        self.gamma = matProps['gamma']
        self.K1, self.K2, self.K3, self.n = matProps['Ka']
        self.CFL, self.max_dt, self.max_rel_rad_e = CFL, max_dt, max_rel_rad_e
        self.start_time, self.end_time = time
        self.verbose = verbose
        # Physical constants
        self.c = 299.792 # light speed
        self.a = 0.01372 # rad constant
        # Pressure, Energy, and Temperature are stored at cell centers
        self.rad_e = init_cond['Radiation Energy']
        self.T = init_cond["Temperature"]
        self.rho = init_cond['Density']
        self.mat_e = self.Cv * self.T
        self.P = self.rho*self.mat_e * (self.gamma - 1)
        self.m = self.rho * self.mesh.V
        # velocities are stored at cell vertices
        self.u = init_cond['Velocity']
        # Values of rad_e and pressure stored at boundaries
        self.rad_e_edge = np.zeros(2)
        self.P_edge = bc.get("Pressure")
        self.u_edge = bc.get("Velocity")
        self.rad_extr = bc.get("Radiation Energy")

    def predictor_step(self):
        # Compute the edge rad_e
        Kt_0, Kt_N = self.compute_absorption_opacity(self.T[0]) + self.Ks, self.compute_absorption_opacity(self.T[-1]) + self.Ks
        self.rad_e_edge[0] = (3*self.rho[0]*(self.mesh.vertices[1] - self.mesh.vertices[0]) * Kt_0 * self.rad_extr[0]\
                              + 4*self.rad_e[0])/(3*self.rho[0]*(self.mesh.vertices[1] - self.mesh.vertices[0]) * Kt_0 + 4)
        self.rad_e_edge[-1] = (3*self.rho[-1]*(self.mesh.vertices[-1] - self.mesh.vertices[-2]) * Kt_N * self.rad_extr[-1]\
                              + 4*self.rad_e[-1])/(3*self.rho[-1]*(self.mesh.vertices[-1] - self.mesh.vertices[-2]) * Kt_N + 4)
        # Generate the half idx masses.
        self.m_halfs = 0.5 * (self.m[:-1] + self.m[1:])
        # Add the half cells
        self.m_halfs = np.concatenate([0.5 * self.rho[0]*self.mesh.V[0]], self.m_halfs, [0.5 * self.rho[-1]*self.mesh.V[-1]])
        self.u_p = np.copy(self.u)
        self.dt = self.compute_time_step()
        # Calculate predictor u (u_p) #TODO: Treat boundary conditions
        self.u_p[1:-1] = -self.dt/self.m_halfs[1:-1] * (self.A[1:-1] \
                        * (self.P[1:] + 1/3*self.rad_e[1:] - self.P[:-1] - 1/3*self.rad_e[:-1]))
        if self.u_edge is None:
            self.u_p[0] = -self.mesh.A[0]*self.dt/(self.m_halfs[0])*\
                (self.P[0] + 1/3*self.rad_e[0] - self.P_edge["Pressure"][0] - 1/3 * self.rad_e_edge[0])
            self.u_p[-1] = -self.mesh.A[-1]*self.dt/(self.m_halfs[-1])*\
                (self.P_edge["Pressure"][-1] + 1/3*self.rad_e_edge[-1] - self.P[-1] - 1/3 * self.rad_e[-1])
        else:
            self.u_p[0] = self.u_edge[0]
            self.u_p[-1] = self.u_edge[-1]
        # Compute a half time velocity
        u_k = 0.5 * (self.u + self.u_p)
        # Update vertices
        vertices_p = np.copy(self.mesh.vertices)
        vertices_p = self.mesh.vertices + u_k * self.dt
        # Update areas and volumes
        A_p, V_p = self.recompute_volumes_and_areas(vertices_p)
        # Update densities
        rho_p = self.recompute_densities(V_p)
        # Update radiation and material energy, but first we compute time avgs.
        A_k = 0.5 * (self.mesh.A + A_p)
        rho_k = 0.5 * (self.rho + rho_p)
        dh_k = 0.5 * (np.diff(self.mesh.vertices) + np.diff(vertices_p))
        # Compute the predictor energy densities
        rad_e_k, mat_e_p = self.compute_predictor_energy_densities(u_k, A_k, rho_k, dh_k, rho_p)
        # Update Temperature and Pressure
        T_p = mat_e_p/self.Cv
        P_p = (self.gamma - 1)*rho_p*mat_e_p
        # Compute time avg
        T_k = 0.5 * (self.T + T_p)
        P_k = 0.5 * (self.P + P_p)
        # Update edge values for rad_e
        Kt_0, Kt_N = self.compute_absorption_opacity(T_k[0]) + self.Ks, self.compute_absorption_opacity(T_k[-1]) + self.Ks
        self.rad_e_edge[0] = (3*rho_k[0]*dh_k[0] * Kt_0 * self.rad_extr[0]\
                              + 4*rad_e_k[0])/(3*rho_k[0]*dh_k[0] * Kt_0  + 4)
        self.rad_e_edge[-1] = (3*rho_k[-1]*dh_k[-1] * Kt_N * self.rad_extr[-1]\
                              + 4*rad_e_k[-1])/(3*rho_k[-1]*dh_k[-1] * Kt_N  + 4)
        return A_k, rad_e_k, T_k, T_p, P_k, mat_e_p, u_k
    
    def corrector_step(self, A_pk, rad_e_pk, T_p, T_pk, P_pk, mat_e_p, u_pk):
        """
        
        """
        # Update velocities
        u_c = np.copy(self.u)
        u_c[1:-1] = -A_pk[1:-1]/self.m_half[1:-1]*self.dt*(P_pk[1:] + 1/3*rad_e_pk[1:] - P_pk[:-1] - 1/3*rad_e_pk[:-1]) + self.u[1:-1]
        if self.u_edge is None:
            u_c[0] = -A_pk[0]/self.m_half[0]*self.dt*(P_pk[0] + 1/3*rad_e_pk[0] - self.P_edge[0] - 1/3*self.rad_e_edge[0]) + self.u[0]
            u_c[-1] = -A_pk[-1]/self.m_half[-1]*self.dt*(self.P_edge[-1] + 1/3*self.rad_e_edge[-1] - P_pk[-1] - 1/3*rad_e_pk[-1]) + self.u[-1]
        else:
            u_c[0] = self.u_edge[0]
            u_c[-1] = self.u_edge[-1]
        # Update vertices, volumes, and areas
        vertices_c = self.mesh.vertices + self.u * self.dt
        A_c, self.mesh.V = self.recompute_volumes_and_areas(vertices_c)
        rho_c = self.recompute_densities(V=self.mesh.V)
        # Compute time avgs. 
        A_k = 0.5 * (self.mesh.A + A_c)
        rho_k = 0.5*(self.rho + rho_c)
        dh_k = 0.5 * (np.diff(vertices_c) + np.diff(self.mesh.vertices))
        u_k = 0.5 * (u_c + self.u)
        # Compute energy densities for corrector step
        self.compute_corrector_energy_densities(u_k, A_k, rho_k, dh_k, rho_c, T_p, T_pk, mat_e_p, P_pk, A_pk, rad_e_pk, u_pk)
        # Update the values at end of time step
        self.rho = rho_c
        self.mesh.vertices = vertices_c
        self.mesh.A = A_c
        self.u = u_c
        self.T = self.mat_e/self.Cv
        self.P = (self.gamma - 1)*self.rho * self.mat_e

    def solve(self):
        t, t_end = self.start_time, self.end_time
        iter = 0
        while t < t_end:
            A_pk, rad_e_pk, T_pk, T_p, P_pk, mat_e_p, u_pk = self.predictor_step()
            self.corrector_step(A_pk, rad_e_pk, T_p, T_pk, P_pk, mat_e_p, u_pk)
            t += self.dt
            self.dt_old = self.dt
            iter+=1
            if self.verbose:
                print(f"Iteration: {iter} | Time Step: {self.dt:.3f} | Cuurent Time: {t:.3f}")



    # Utility methods
    def compute_absorption_opacity(self, T):
        """
        Member method to compute the absorption opacity
        """
        Ka = self.K1 / (self.K2*T**self.n + self.K3)
        return Ka

    def compute_time_step(self):
        """
        Member method to solve for tor the optimal time step
        """
        dt_eps = self.max_rel_rad_e * self.rad_e * (self.dt_old) / (self.rad_e - self.rad_e_old)
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
    
    def compute_predictor_xi(self, u_k):
        """
        Member method to compute xi during predictor step for rad_e and mat_e
        """
        xi = -self.P * (self.mesh.A[1:]*u_k[1:] - self.mesh.A[:-1]*u_k[:-1])
        return xi
    
    def compute_corrector_xi(self, mat_e_p, P_pk, A_pk, u_k):
        """
        Member method to compute xi during corrector step for rad_e and mat_e
        """
        xi = self.m/self.dt*(mat_e_p - self.mat_e) - P_pk*(A_pk[1:]*u_k[1:] - A_pk[:-1]*u_k[:-1])
        return xi

    def compute_nu(self, Ka, T):
        """
        Member method to compute nu during step for rad_e and mat_e
        """
        nu = (self.dt * Ka * 2* self.c * self.a * T**3)\
            /(self.Cv + self.dt * Ka * 2 *self.a * self.c * T**3)
        return nu

    def compute_predictor_energy_densities(self, u_k, A_k, rho_k, dh_k, rho_p):
        """
        Member method to compute rad_e and mat_e for the predictor step
        """
        # Compute absorp. opacities at cell center
        Ka = self.compute_absorption_opacity(T=self.T)
        # Compute nu 
        nu = self.compute_nu(Ka, T=self.T)
        # Compute xi
        xi = self.compute_predictor_xi(u_k)
        
        # Compute avg T at interior vertices
        T_vert = ((self.T[:-1]**4 + self.T[1:]**4)/2)**0.25
        #TODO Treat bc here
        T_l = ((1/self.a * self.rad_e_edge[0] + self.T[0]**4)/2)**0.25
        T_r = ((1/self.a * self.rad_e_edge[1] + self.T[-1]**4)/2)**0.25
        T_vert = np.concatenate(([T_l], T_vert, [T_r]))
        Ka_t = self.compute_absorption_opacity(T_vert)
        # These are edge evaluated total opacities
        Kt = self.Ks + Ka_t
        # Build the tridiag system as banded matrix (Thomas Alg)
        Mb = np.zeros((3, self.mesh.ncells))
        y = np.zeros(self.mesh.ncells)
        # Do interior nodes first
        # superdiagonal
        Mb[0, 2:] = -(A_k[2:-1]*self.c)/(3*(rho_k[2:]*dh_k[2:]*Kt[2:-1] + rho_k[1:-1]*dh_k[1:-1]*Kt[2:-1]))
        # main diagonal #TODO: Might be using the wrong idx for Kt
        Mb[1, 1:-1] = self.m[1:-1]/(self.dt*rho_p[1:-1]) +  A_k[2:-1] *self.c / (3*(rho_k[1:-1]*dh_k[1:-1]*Kt[2:-1] + rho_k[2:]*dh_k[2:]*Kt[2:-1])) +\
            A_k[1:-2]*self.c/(3*(rho_k[1:-1]*dh_k[1:-1]*Kt[1:-2] + rho_k[:-2]*dh_k[:-2]*Kt[1:-2]))
        # subdiagonal  
        Mb[2, :-2] = -(A_k[1:-2]*self.c)/(3*(rho_k[1:-1]*dh_k[1:-2]*Kt[1:-2] + rho_k[:-2]*dh_k[:-2]*Kt[1:-2]))
        # rhs
        y[1:-1] =  self.m[1:-1]*self.rad_e[1:-1]/(self.dt * self.rho[1:-1]) + A_k[2:-1]*self.c*(self.rad_e[2:] - self.rad_e[1:-1])/\
            (3*(rho_k[1:-1]*dh_k[1:-1]*Kt[2:-1] + rho_k[2:]*dh_k[2:]*Kt[2:-1])) - A_k[1:-2]*self.c*(self.rad_e[1:-1] - self.rad_e[:-2])/\
                (3*(rho_k[1:-1]*dh_k[1:-1]*Kt[1:-2] + rho_k[:-2]*dh_k[:-2]*Kt[1:-2])) + self.m[1:-1]*Ka[1:-1]*self.c*(1-nu[1:-1])*(self.a*self.T[1:-1]**4 - 0.5 * self.rad_e[1:-1])\
                    + nu[1:-1]*xi[1:-1] - 1/3*self.rad_e[1:-1]*(self.mesh.A[2:]*u_k[2:] - self.mesh.A[1:-2]*u_k[1:-2])
        # Boundary Conditions
        # superdiagonal
        Mb[0, 1] = -A_k[1]*self.c/(3*(rho_k[0]*dh_k[0]*Kt[1] + rho_k[1]*dh_k[1]*Kt[1]))
        # main diagonal
        Mb[1, 0] = self.m[0]/(self.dt*rho_p[0]) + A_k[1]*self.c/(3*(rho_k[0]*dh_k[0]*Kt[1] + rho_k[1]*dh_k[1]*Kt[1])) +\
                    A_k[0]*self.c/(3*rho_k[0]*dh_k[0]*Kt[0] + 4) + 0.5*self.m[0]*Ka[0]*self.c*(1-nu[0])
        Mb[1, -1] = self.m[-1]/(self.dt*rho_p[-1]) + A_k[-1]*self.c/(3*rho_k[-1]*dh_k[-1]*Kt[-1]+4) + A_k[-2]*self.c/\
                    (3*(rho_k[-1]*dh_k[-1]*Kt[-2]+rho_k[-2]*dh_k[-2]*Kt[-2])) + 0.5*self.m[-1]*Ka[-1]*self.c*(1-nu[-1])
        # subdiagonal
        Mb[2, -2] = -A_k[-2]*self.c/(3*(rho_k[-1]*dh_k[-1]*Kt[-2]+rho_k[-2]*dh_k[-2]*Kt[-2]))
        y[0] = self.m[0]*self.rad_e[0]/(self.dt*self.rho[0]) + A_k[1]*self.c*(self.rad_e[1]-self.rad_e[0])/(3*(rho_k[0]*dh_k[0]*Kt[1] + rho_k[1]*dh_k[1]*Kt[1])) -\
                A_k[0]*self.c*(self.rad_e[0]-2*self.rad_extr[0])/(3*rho_k[0]*dh_k[0]*Kt[0] + 4) + self.m[0]*Ka[0]*self.c*(1-nu[0])*(self.a*self.T[0]**4-0.5*self.rad_e[0])+\
                nu[0]*xi[0] - 1/3*self.rad_e[0]*(self.mesh.A[1]*u_k[1]-self.mesh.A[0]*u_k[0])
        y[-1] = self.m[-1]*self.rad_e[-1]/(self.dt*self.rho[-1]) + A_k[-1]*self.c*(2*self.rad_extr[-1]-self.rad_e[-1])/(3*rho_k[-1]*dh_k[-1]*Kt[-1]+4) -\
                A_k[-2]*self.c*(self.rad_e[-1]-self.rad_e[-2])/(3*(rho_k[-1]*dh_k[-1]*Kt[-2]+rho_k[-2]*dh_k[-2]*Kt[-2])) +\
                self.m[-1]*Ka[-1]*self.c*(1-nu[-1])*(self.a*self.T[-1]**4-0.5*self.rad_e[-1]) + nu[-1]*xi[-1] -\
                    1/3*self.rad_e[-1]*(self.mesh.A[-1]*u_k[-1]-self.mesh.A[-2]*u_k[-2])
        # Solve for predictor rad_e
        rad_e_p = solve_banded((1, 1), Mb, y)
        # Compute time avg for rad_e
        rad_e_k = 0.5 * (self.rad_e + rad_e_p)
        mat_e_p = self.mat_e + (self.dt*self.Cv*(self.m*Ka*self.c*(rad_e_k - self.a*self.T**4) + xi))/\
            (self.m*self.Cv + self.dt*self.m*Ka*2*self.a*self.c*self.T**3)
        return rad_e_k, mat_e_p

    def compute_corrector_energy_densities(self, u_k, A_k, rho_k, dh_k, rho_c, T_p, T_pk, mat_e_p, P_pk, A_pk, rad_e_pk, u_pk):
        """
        
        """
        # Compute cell-centered opacities
        Ka = self.compute_absorption_opacity(T_pk)
        # Compute nu
        nu = self.compute_nu(Ka, T_p)
        # Compute xi
        xi = self.compute_corrector_xi(mat_e_p, P_pk, A_pk, u_k)
        # Compute vertex Tempeartures from time avg predictor temp
        T_vert = ((T_pk[:-1]**4 + T_pk[1:]**4)/2)**0.25
        T_l = ((1/self.a * self.rad_e_edge[0] + T_pk[0]**4)/2)**0.25
        T_r = ((1/self.a * self.rad_e_edge[1] + T_pk[-1]**4)/2)**0.25
        T_vert = np.concatenate(([T_l], T_vert, [T_r]))
        Kt_p = self.compute_absorption_opacity(T_vert)
        # Build the banded tridiagonal matrix (Thomas Alg.)
        Mb = np.zeros((3, self.mesh.ncells))
        y = np.zeros(self.mesh.ncells)
        # Do interior nodes first
        # superdiagonal
        Mb[0, 2:] = -(A_k[2:-1]*self.c)/(3*(rho_k[2:]*dh_k[2:]*Kt_p[2:-1] + rho_k[1:-1]*dh_k[1:-1]*Kt_p[2:-1]))
        # main diagonal #TODO: Might be using the wrong idx for Kt_p
        Mb[1, 1:-1] = self.m[1:-1]/(self.dt*rho_c[1:-1]) +  A_k[2:-1] *self.c / (3*(rho_k[1:-1]*dh_k[1:-1]*Kt_p[2:-1] + rho_k[2:]*dh_k[2:]*Kt_p[2:-1])) +\
            A_k[1:-2]*self.c/(3*(rho_k[1:-1]*dh_k[1:-1]*Kt_p[1:-2] + rho_k[:-2]*dh_k[:-2]*Kt_p[1:-2]))
        # subdiagonal  
        Mb[2, :-2] = -(A_k[1:-2]*self.c)/(3*(rho_k[1:-1]*dh_k[1:-2]*Kt_p[1:-2] + rho_k[:-2]*dh_k[:-2]*Kt_p[1:-2]))
        # rhs
        y[1:-1] =  self.m[1:-1]*self.rad_e[1:-1]/(self.dt * self.rho[1:-1]) + A_k[2:-1]*self.c*(self.rad_e[2:] - self.rad_e[1:-1])/\
            (3*(rho_k[1:-1]*dh_k[1:-1]*Kt_p[2:-1] + rho_k[2:]*dh_k[2:]*Kt_p[2:-1])) - A_k[1:-2]*self.c*(self.rad_e[1:-1] - self.rad_e[:-2])/\
                (3*(rho_k[1:-1]*dh_k[1:-1]*Kt_p[1:-2] + rho_k[:-2]*dh_k[:-2]*Kt_p[1:-2])) + self.m[1:-1]*Ka[1:-1]*self.c*(1-nu[1:-1])*(self.a*T_pk[1:-1]**4 - 0.5 * self.rad_e[1:-1])\
                    + nu[1:-1]*xi[1:-1] - rad_e_pk[1:-1]*(A_pk[2:]*u_k[2:] - A_pk[1:-2]*u_k[1:-2])
        # Boundary Conditions
        # superdiagonal
        Mb[0, 1] = -A_k[1]*self.c/(3*(rho_k[0]*dh_k[0]*Kt_p[1] + rho_k[1]*dh_k[1]*Kt_p[1]))
        # main diagonal
        Mb[1, 0] = self.m[0]/(self.dt*rho_c[0]) + A_k[1]*self.c/(3*(rho_k[0]*dh_k[0]*Kt_p[1] + rho_k[1]*dh_k[1]*Kt_p[1])) +\
                    A_k[0]*self.c/(3*rho_k[0]*dh_k[0]*Kt_p[0] + 4) + 0.5*self.m[0]*Ka[0]*self.c*(1-nu[0])
        Mb[1, -1] = self.m[-1]/(self.dt*rho_c[-1]) + A_k[-1]*self.c/(3*rho_k[-1]*dh_k[-1]*Kt_p[-1]+4) + A_k[-2]*self.c/\
                    (3*(rho_k[-1]*dh_k[-1]*Kt_p[-2]+rho_k[-2]*dh_k[-2]*Kt_p[-2])) + 0.5*self.m[-1]*Ka[-1]*self.c*(1-nu[-1])
        # subdiagonal
        Mb[2, -2] = -A_k[-2]*self.c/(3*(rho_k[-1]*dh_k[-1]*Kt_p[-2]+rho_k[-2]*dh_k[-2]*Kt_p[-2]))
        y[0] = self.m[0]*self.rad_e[0]/(self.dt*self.rho[0]) + A_k[1]*self.c*(self.rad_e[1]-self.rad_e[0])/(3*(rho_k[0]*dh_k[0]*Kt_p[1] + rho_k[1]*dh_k[1]*Kt_p[1])) -\
                A_k[0]*self.c*(self.rad_e[0]-2*self.rad_extr[0])/(3*rho_k[0]*dh_k[0]*Kt_p[0] + 4) + self.m[0]*Ka[0]*self.c*(1-nu[0])*(self.a*T_pk[0]**4-0.5*self.rad_e[0])+\
                nu[0]*xi[0] - rad_e_pk[0]*(A_pk[1]*u_k[1]-A_pk[0]*u_k[0])
        y[-1] = self.m[-1]*self.rad_e[-1]/(self.dt*self.rho[-1]) + A_k[-1]*self.c*(2*self.rad_extr[-1]-self.rad_e[-1])/(3*rho_k[-1]*dh_k[-1]*Kt_p[-1]+4) -\
                A_k[-2]*self.c*(self.rad_e[-1]-self.rad_e[-2])/(3*(rho_k[-1]*dh_k[-1]*Kt_p[-2]+rho_k[-2]*dh_k[-2]*Kt_p[-2])) +\
                self.m[-1]*Ka[-1]*self.c*(1-nu[-1])*(self.a*T_pk[-1]**4-0.5*self.rad_e[-1]) + nu[-1]*xi[-1] -\
                    rad_e_pk[-1]*(A_pk[-1]*u_k[-1]-A_pk[-2]*u_k[-2])
        # Solve for corector rad_e
        rad_e_c = solve_banded((1, 1), Mb, y)
        rad_e_k = 0.5 * (rad_e_c + self.rad_e)
        # Calculate mat_e
        mat_e_p = mat_e_p + (self.dt*self.Cv*(self.m*Ka*self.c*(rad_e_k - self.a*T_pk**4) + xi))/\
            (self.m*self.Cv + self.dt*self.m*Ka*2*self.a*self.c*T_p**3)
        # Update our energy densities values
        self.rad_e_old = self.rad_e
        self.rad_e = rad_e_c
        self.mat_e = mat_e_p


if __name__ == "__main__":
    verbose = True
    hb = np.array([1])
    dh = np.array([.02])
    mesh = Mesh("SLAB", dh, hb, verbose)
    mesh.generate_uniform_submesh()
    ncells = mesh.ncells
    Cv_list = np.array([0.1])
    Ka_list = np.array([20])
    Ks_list = np.array([0.5])
    matProps = {"Cv": 0.1,
                "Ka": [20, 0, 1, 1], #K1, K2, K3, n
                "Ks": 0.5,
                "gamma": 5/3}
    time = [0, 10]
    bc = {"Pressure": [0, 0],
          "Radiation Energy": [0, 0]}
    a = 0.01372
    T_0 = 0.2
    rho_0 = 1
    u_0 = 0
    esp_0 = a*T_0**4
    init_cond = {"Density": np.repeat(rho_0, ncells),
                 "Temperature": np.repeat(T_0, ncells),
                 "Velocity": np.repeat(u_0, ncells + 1),
                 "Radiation Energy": np.repeat(esp_0, ncells)}
    CFL = 0.5
    max_dt = 0.01
    max_rel_rad_e = esp_0/max_dt
    Solver = RadHydroSolver(mesh, matProps, bc, init_cond, CFL, max_dt, max_rel_rad_e, time, "RadHydro", verbose)
    Solver.solve()


