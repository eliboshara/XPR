"""
Author: Elias Boshara
"""

import os
import numpy as np
from juliacall import Main as jl, convert
import Bio.PDB

class XPRsim:
    """
    Python interface for the Julia XPR module X-ray Reflectivity forward model.
    Handles data loading, parameter masking, and bounds checking, function evals
    """
    def __init__(self, 
                rrf_file, 
                num_boxes, 
                pdb_file=None, 
                fit_sig=False, 
                fit_yscl=False, 
                fit_bkg=False, 
                vol_calc="approx", 
                rho_top=0.0, 
                rho_bottom=0.334,
                spacez=0.5):
    
        # assign spacez and existing parameters
        self.spacez = spacez
        self.rrf_file = rrf_file
        self.pdb_file = pdb_file
        self.num_boxes = num_boxes
        self.include_protein = (pdb_file is not None)
        self.fit_sig = fit_sig
        self.fit_yscl = fit_yscl
        self.fit_bkg = fit_bkg
        self.vol_calc = vol_calc
        self.rho_top = rho_top
        self.rho_bottom = rho_bottom
        
        # fixed baseline parameters
        self.fixed_sig = 3.4
        self.fixed_yscl = 1.0
        self.fixed_bkg = 0.0

        # initialize Julia environment
        jl.seval("using Pkg; Pkg.activate(\"XPR\")")
        jl.seval("using XPR")
        
        # configure parameter mapping/bounds first
        self._configure_parameters()

        # load data and allocate Julia memory using those bounds
        self.reload_data(rrf_file)

        self._print_configuration()

    def _configure_parameters(self):
        self.param_names = []
        self.bounds = []
        self.active_indices = []  # maps the Python input vector to the 8+2n Julia vector
        
        # q_offset (always active)
        self.param_names.append("q_offset")
        self.bounds.append((-1e-2, 1e-2))
        self.active_indices.append(0)
        
        # protein parameters (1-4)
        if self.include_protein:
            self.param_names.extend(["theta (deg)", "phi (deg)", "d_protein", "coverage_C"])
            self.bounds.extend([(0.0, 180.0), (0.0, 360.0), (-20.0, 20.0), (0.0, 10.0)])
            self.active_indices.extend([1, 2, 3, 4])
            
        # optional fitting parameters (5-7)
        if self.fit_sig:
            self.param_names.append("sigma")
            self.bounds.append((1.0, 5.0))
            self.active_indices.append(5)
            
        if self.fit_yscl:
            self.param_names.append("yscl")
            self.bounds.append((0.0, 5.0))
            self.active_indices.append(6)
            
        if self.fit_bkg:
            self.param_names.append("bkg")
            self.bounds.append((-5.0, 5.0))
            self.active_indices.append(7)
            
        # box lengths (8 to 8+n-1)
        for i in range(self.num_boxes):
            self.param_names.append(f"box_length_{i+1}")
            self.bounds.append((0.0, 30.0))
            self.active_indices.append(8 + i)
            
        # box densities (8+n to 8+2n-1)
        for i in range(self.num_boxes):
            self.param_names.append(f"box_density_{i+1}")
            self.bounds.append((0.0, 1.0))
            self.active_indices.append(8 + self.num_boxes + i)
            
        self.bounds = np.array(self.bounds)
        self.total_julia_params = 8 + 2 * self.num_boxes
        
        # map parameter names to array indices for fast lookup
        self.param_map = {name: idx for idx, name in enumerate(self.param_names)}

    def _get_allocation_bounds(self):

        # calculate minimum/max possible box length sum for preallocation limits
        min_box_length_sum = sum(
            float(self.bounds[self.param_map[f"box_length_{i+1}"], 0]) for i in range(self.num_boxes)
        )
        max_box_length = max(
            float(self.bounds[self.param_map[f"box_length_{i+1}"], 1]) for i in range(self.num_boxes)
        )

        if self.include_protein:
            idx = self.param_map["d_protein"]
            d_protein_bounds = (float(self.bounds[idx, 0]), float(self.bounds[idx, 1]))
        else:
            d_protein_bounds = (0.0, 0.0)

        if self.fit_sig:
            idx = self.param_map["sigma"]
            sig_max = float(self.bounds[idx, 1])
        else:
            sig_max = float(self.fixed_sig)

        return (max_box_length, d_protein_bounds, sig_max, min_box_length_sum)

    def update_bounds(self, bounds_dict):

        # allocation requirements before changing bounds
        old_allocation_bounds = self._get_allocation_bounds()

        for name, bound in bounds_dict.items():

            if name not in self.param_map:
                raise KeyError(f"parameter '{name}' is inactive or undefined.")
            if len(bound) != 2:
                raise ValueError(f"Bounds for '{name}' must be (lower, upper).")

            lower, upper = bound
            if lower > upper:
                raise ValueError(f"Invalid bounds for '{name}': "f"lower bound {lower} exceeds upper bound {upper}.")

            self.bounds[self.param_map[name]] = (lower, upper)

        # allocation requirements after changing bounds
        new_allocation_bounds = self._get_allocation_bounds()

        # reload Julia only if the fixed-mesh domain actually changed
        if new_allocation_bounds != old_allocation_bounds:
            self.reload_data(self.rrf_file)

    # outputs the expected parameter order for the user
    def _print_configuration(self):
       
        print("XPRsim Initialized Successfully.")
        print("-" * 40)
        print(f"Total Active Parameters: {len(self.active_indices)}")
        print("Expected Input Vector Order & Bounds:")
        for idx, (name, bound) in enumerate(zip(self.param_names, self.bounds)):
            print(f"  [{idx}] {name:<15}: {bound}")
        print("-" * 40)

    # pads the active parameter array into the full vector expected by Julia
    def _pad_params(self, params):
        
        params = np.asarray(params, dtype=float)
        
        if len(params) != len(self.active_indices):
            raise ValueError(f"Input length mismatch. Expected {len(self.active_indices)} parameters, got {len(params)}.")
            
        if not np.all((params >= self.bounds[:, 0]) & (params <= self.bounds[:, 1])):
            raise ValueError("One or more parameters are out of bounds.")

        full_params = np.zeros(self.total_julia_params, dtype=float)
        
        # apply defaults for optional params (overwritten if active)
        full_params[5] = self.fixed_sig
        full_params[6] = self.fixed_yscl
        full_params[7] = self.fixed_bkg
        
        # inject active parameters
        full_params[self.active_indices] = params
        return full_params

    # loads data from file and triggers Julia memory reallocation
    def reload_data(self, rrf_file):

        if not os.path.isfile(rrf_file):
            raise FileNotFoundError(f"Data file '{rrf_file}' not found.")

        self.rrf_file = rrf_file
        self.data_Q, self.data_R, self.data_err = self._import_rrf(rrf_file)
        
        coord, electrons, radii = None, None, None
        if self.include_protein:
            if not os.path.isfile(self.pdb_file):
                raise FileNotFoundError(f"PDB file '{self.pdb_file}' not found.")
            coord, electrons, radii, masses = self._import_pdb(self.pdb_file)
            
        # convert to Julia arrays
        j_Q = convert(jl.Array, self.data_Q)
        j_R = convert(jl.Array, self.data_R)
        j_err = convert(jl.Array, self.data_err)
        
        # determine parameter-domain bounds needed for fixed Julia allocation
        max_box_length, d_protein_bounds, sig_max, min_box_length_sum = self._get_allocation_bounds()

        kwargs = {
            "vol_calc": self.vol_calc,
            "rho_top": self.rho_top,
            "rho_bottom": self.rho_bottom,
            "max_box_length": max_box_length,
            "d_protein_bounds": d_protein_bounds,
            "sig_max": sig_max,
            "spacez": self.spacez,
            "min_box_length_sum": min_box_length_sum
        }

        if self.include_protein:
            kwargs["coordinates_py"] = convert(jl.Array, coord)
            kwargs["electrons_py"] = convert(jl.Array, electrons)
            kwargs["radii_py"] = convert(jl.Array, radii)
            kwargs["masses_py"] = convert(jl.Array, masses)
            
        jl.load_data(j_Q, j_R, j_err, self.num_boxes, self.include_protein, 
                     self.fit_sig, self.fit_yscl, self.fit_bkg, **kwargs)

    # returns the log-likelihood and simulated reflectivity
    def eval_logL(self, params):
        full_params = self._pad_params(params)
        logL, R, protein_height = jl.XPR_sim_ref(full_params)
        return logL, np.array(R), protein_height

    # returns the gradient vector, stripped of fixed parameters
    def eval_grad(self, params):
        full_params = self._pad_params(params)
        grad = np.array(jl.XPR_grad(full_params))
        return grad[self.active_indices]

    # returns the Hessian matrix, stripped of fixed parameters
    def eval_hess(self, params):
        full_params = self._pad_params(params)
        hess = np.array(jl.XPR_hess(full_params))
        # extract only the intersections of active rows and columns
        return hess[np.ix_(self.active_indices, self.active_indices)]

    # rrf import method
    @staticmethod
    def _import_rrf(fname):
        values = np.genfromtxt(fname, unpack=True)
        rrf = np.transpose(np.array(values))
        rrf_proc = rrf[np.where(rrf[:, 0] >= 0.026), :][0,:,:]
        return rrf_proc[:,0], rrf_proc[:,1], rrf_proc[:,2]

    # pdg import method
    @staticmethod
    def _import_pdb(fname):
        try:
            from mendeleev.fetch import fetch_table
            ptable = fetch_table('elements')[['symbol', 'atomic_number', 'vdw_radius', 'atomic_weight']]
        except ImportError:
            from mendeleev import get_table
            ptable = get_table('elements')[['symbol', 'atomic_number', 'vdw_radius', 'atomic_weight']]
            
        atominfo = ptable.set_index('symbol').T.to_dict('list')
        pdbparser = Bio.PDB.PDBParser(QUIET=True)
        struct = pdbparser.get_structure('struct', fname)

        atoms = list(struct.get_atoms())
        n_atoms = len(atoms)
        coord = np.zeros((n_atoms, 3))
        radii = np.zeros(n_atoms)
        electrons = np.zeros(n_atoms)
        masses = np.zeros(n_atoms)

        for ind, atom in enumerate(atoms):
            coord[ind, :] = atom.coord
            element_key = 'Ca' if atom.name == 'CAL' else atom.element.capitalize()
            electrons[ind] = atominfo[element_key][0]
            radii[ind] = atominfo[element_key][1] / 100.0
            masses[ind] = atominfo[element_key][2]

        return coord, electrons, radii, masses
