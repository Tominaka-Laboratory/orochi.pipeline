import os
import json
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np

from orochi.base.core import OrochiBase
from orochi.phase import Phase
from orochi.pdfsim import Simulator
from orochi.fit import LeastSquaresOptimizer


class RefinementPipeline(OrochiBase):
    """Automated pipeline for real-space pair distribution function (PDF) structural refinement.

    Integrates `orochi.phase` (crystallographic unit cell, directional PBCs, Wyckoff constraints),
    `orochi.pdfsim` (Numba-accelerated G(r) simulation with exact Q-dependent scattering factors),
    and `orochi.fit` (bounded least-squares optimization).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.work_dir = kwargs.get("work_dir", None)
        self.study_name = kwargs.get("study_name", "StructureRefinement")

    def refine(
        self,
        gr_data: Union[str, np.ndarray],
        cif_file: str,
        pbc: Tuple[bool, bool, bool] = (True, True, True),
        shape_type: str = "analytical",
        diameter: float = 1000.0,
        qdepend: bool = True,
        rambda: float = 0.18033,
        qmin: float = 0.8,
        qmax: float = 25.0,
        fit_min: float = 1.5,
        fit_max: float = 30.0,
        rstep: float = 0.01,
        refine_lattice: bool = True,
        refine_adp: bool = True,
        refine_scale: bool = True,
        refine_qdamp: bool = True,
        refine_qbroad: bool = True,
        refine_delta2: bool = True,
        refine_sphere: bool = False,
        initial_scale: float = 0.9,
        initial_qdamp: float = 0.03,
        initial_qbroad: float = 0.015,
        initial_delta2: float = 2.0,
        output_dir: Optional[str] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Executes full real-space structural refinement of a CIF model against experimental G(r).

        Parameters
        ----------
        gr_data : str or np.ndarray
            Path to experimental .gr file or numpy array with shape (N, 2) [r, G(r)].
        cif_file : str
            Path to crystal structure CIF file.
        pbc : tuple of 3 bool
            Directional periodic boundary conditions [a, b, c], e.g. (True, True, False) for 2D nanosheets.
        shape_type : str
            Morphology model: "analytical" (spherical envelope), "sphere" (explicit cluster), etc.
        diameter : float
            Nanoparticle diameter or coherence length in Angstroms.
        qdepend : bool
            Whether to use exact Q-dependent DABAX form-factor kernels (True) or constant-Z (False).
        rambda : float
            Radiation wavelength in Angstroms.
        qmin, qmax : float
            Scattering vector Q range in Angstrom^-1.
        fit_min, fit_max : float
            Real-space fitting interval in Angstroms.
        rstep : float
            Radial distance step size in Angstroms.
        refine_lattice : bool
            Whether to optimize lattice constants conforming to space group constraints.
        refine_adp : bool
            Whether to optimize atomic displacement parameters (U_iso).
        refine_scale : bool
            Whether to optimize overall scale factor.
        refine_qdamp : bool
            Whether to optimize instrumental damp factor Q_damp.
        refine_qbroad : bool
            Whether to optimize instrumental peak broadening Q_broad.
        refine_delta2 : bool
            Whether to optimize dynamic thermal correlation factor delta2.
        refine_sphere : bool
            Whether to optimize particle diameter / coherence length.
        output_dir : str, optional
            Output directory for saving reports and refined CIFs.
        save_results : bool
            Whether to persist results JSON and plot.

        Returns
        -------
        dict
            Refinement results including best parameters, uncertainties, residuals, and profile arrays.
        """
        out_dir = output_dir or self.work_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(cif_file):
            raise FileNotFoundError(f"CIF file not found: {cif_file}")

        # 1. Initialize Phase
        phase = Phase()
        phase.work_dir = out_dir
        phase.study_name = self.study_name
        phase.read_file(cif_file)
        phase["shape_type"] = shape_type
        phase["pbc"] = list(pbc)
        phase["sphere"] = [diameter, diameter, diameter]

        phase.set_spherical_constraints()
        phase.set_lattice_constraints()
        phase.set_elemental_adp_constraints()

        # Lattice refinement setup
        lat0 = np.array(phase["lattice"], dtype=float)
        phase["lattice", "min"] = (lat0 * 0.95).tolist()
        phase["lattice", "max"] = (lat0 * 1.05).tolist()
        if refine_lattice:
            # Respect symmetry constraints (e.g. cubic a=b=c -> refine only a)
            phase["lattice", "optimize"] = [True, False, False, False, False, False]
        else:
            phase["lattice", "optimize"] = [False] * 6

        # ADP refinement setup
        num_atoms = len(phase["adp"])
        phase["adp", "min"] = [0.0002] * num_atoms
        phase["adp", "max"] = [0.08] * num_atoms
        phase["adp", "optimize"] = [refine_adp] * num_atoms

        if refine_sphere:
            phase["sphere", "optimize"] = [True, False, False]
            phase["sphere", "min"] = [5.0, 5.0, 5.0]
            phase["sphere", "max"] = [diameter * 3.0, diameter * 3.0, diameter * 3.0]

        phase()

        # 2. Initialize Simulator
        sim = Simulator()
        sim.work_dir = out_dir
        sim.study_name = self.study_name
        sim.phase = phase
        sim["rambda"] = rambda
        sim["qmin"] = qmin
        sim["qmax"] = qmax
        sim["rmin"] = fit_min
        sim["rmax"] = fit_max
        sim["rstep"] = rstep
        sim["qdepend"] = qdepend

        sim["scale"] = initial_scale
        sim["scale", "min"] = 0.1
        sim["scale", "max"] = 3.0
        sim["scale", "optimize"] = refine_scale

        sim["qdamp"] = initial_qdamp
        sim["qdamp", "min"] = 0.001
        sim["qdamp", "max"] = 0.1
        sim["qdamp", "optimize"] = refine_qdamp

        sim["qbroad"] = initial_qbroad
        sim["qbroad", "min"] = 0.0001
        sim["qbroad", "max"] = 0.08
        sim["qbroad", "optimize"] = refine_qbroad

        sim["delta2"] = initial_delta2
        sim["delta2", "min"] = 0.0
        sim["delta2", "max"] = 6.0
        sim["delta2", "optimize"] = refine_delta2

        # 3. Initialize Optimizer
        ls = LeastSquaresOptimizer()
        ls.work_dir = out_dir
        ls.study_name = self.study_name
        ls.simulators = [sim]
        ls["fit_min"] = fit_min
        ls["fit_max"] = fit_max

        if isinstance(gr_data, str) and os.path.exists(gr_data):
            ls.read_file(gr_data)
        elif isinstance(gr_data, np.ndarray):
            ls.set_data(gr_data)

        # Run refinement
        opt_results = ls()

        # Extract profiles
        r_fit = ls.r if hasattr(ls, "r") else np.array([])
        g_exp = ls.data if hasattr(ls, "data") else np.array([])
        g_calc = ls.calculated if hasattr(ls, "calculated") else np.array([])
        g_diff = g_exp - g_calc if len(g_exp) == len(g_calc) and len(g_exp) > 0 else np.array([])

        scores = getattr(ls, "best_scores", {})
        rw = scores.get("adj", None)
        r2 = scores.get("rsquared", None)
        chisqr = scores.get("chisqr", None)

        best_params = getattr(ls, "best_params", {})

        summary = {
            "cif_file": cif_file,
            "pbc": list(pbc),
            "qdepend": qdepend,
            "qmin": qmin,
            "qmax": qmax,
            "fit_min": fit_min,
            "fit_max": fit_max,
            "rw": float(rw) if rw is not None else None,
            "r2": float(r2) if r2 is not None else None,
            "chisqr": float(chisqr) if chisqr is not None else None,
            "best_params": best_params,
            "scores": scores,
            "lattice_refined": np.asarray(phase["lattice"]).tolist(),
            "adp_refined": np.asarray(phase["adp"]).tolist(),
            "scale_refined": float(sim["scale"]),
            "qdamp_refined": float(sim["qdamp"]),
            "qbroad_refined": float(sim["qbroad"]),
            "delta2_refined": float(sim["delta2"]),
        }

        if save_results:
            json_path = os.path.join(out_dir, f"{self.study_name}_results.json")
            # Ensure all values are JSON serializable
            def to_json_serializable(val):
                if isinstance(val, (np.ndarray, list)):
                    return [to_json_serializable(x) for x in val]
                elif isinstance(val, (np.integer, int)):
                    return int(val)
                elif isinstance(val, (np.floating, float)):
                    return float(val)
                elif isinstance(val, dict):
                    return {k: to_json_serializable(v) for k, v in val.items()}
                return val

            serializable_summary = to_json_serializable(summary)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(serializable_summary, f, indent=4)
            summary["json_path"] = json_path

            # Save refined CIF
            refined_cif_path = os.path.join(out_dir, f"{self.study_name}_refined.cif")
            try:
                phase.save_file(refined_cif_path)
                summary["refined_cif_path"] = refined_cif_path
            except Exception:
                pass

        return {
            "summary": summary,
            "r": r_fit,
            "g_exp": g_exp,
            "g_calc": g_calc,
            "g_diff": g_diff,
            "phase": phase,
            "simulator": sim,
            "optimizer": ls,
        }
