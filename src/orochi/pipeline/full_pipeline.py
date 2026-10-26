import os
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

from orochi.base.core import OrochiBase
from .reduction_1d import Reduction1DPipeline
from .reduction import ReductionPipeline as Reduction2DPipeline
from .refinement import RefinementPipeline


class FullAnalysisPipeline(OrochiBase):
    """End-to-end total scattering analysis pipeline chaining reduction and structural refinement.

    Supports both 1D total scattering profiles and 2D area detector images.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.work_dir = kwargs.get("work_dir", None)
        self.study_name = kwargs.get("study_name", "FullAnalysis")

    def from_1d_profile(
        self,
        sample_data: Union[str, np.ndarray],
        cif_file: str,
        composition: str,
        background_data: Optional[Union[str, np.ndarray]] = None,
        rambda: float = 0.18033,
        qmin: float = 0.8,
        qmax: float = 25.0,
        fit_min: float = 1.5,
        fit_max: float = 30.0,
        pbc: Tuple[bool, bool, bool] = (True, True, True),
        qdepend: bool = True,
        tune: bool = False,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Executes complete 1D profile reduction followed by real-space CIF refinement.

        Parameters
        ----------
        sample_data : str or np.ndarray
            Path to 1D diffraction file or data array.
        cif_file : str
            Path to crystal structure CIF file.
        composition : str
            Chemical formula.
        background_data : str or np.ndarray, optional
            Path to blank capillary or background data.
        rambda : float
            Wavelength in Angstroms.
        qmin, qmax : float
            Q range in Angstrom^-1.
        fit_min, fit_max : float
            Real-space fitting interval in Angstroms.
        pbc : tuple of 3 bool
            Directional periodic boundary conditions.
        qdepend : bool
            Whether to use exact Q-dependent atomic form factors.
        tune : bool
            Whether to run automated ripple minimization.

        Returns
        -------
        dict
            Combined results from 1D reduction and structural refinement.
        """
        out_dir = output_dir or self.work_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        # 1. 1D Reduction stage
        red_pipe = Reduction1DPipeline(work_dir=out_dir, study_name=f"{self.study_name}_Red1D")
        red_res = red_pipe.process(
            sample_data=sample_data,
            background_data=background_data,
            composition=composition,
            rambda=rambda,
            qmin=qmin,
            qmax=qmax,
            rmin=0.0,
            rmax=fit_max * 1.5,
            tune=tune,
            output_dir=out_dir,
            save_gr=True,
        )

        gr_path = red_res.get("gr_path", None)
        if gr_path is None or not os.path.exists(gr_path):
            # Pass array directly
            r = red_res["r"]
            gr = red_res["gr"]
            gr_data = np.column_stack((r, gr))
        else:
            gr_data = gr_path

        # 2. Refinement stage
        ref_pipe = RefinementPipeline(work_dir=out_dir, study_name=f"{self.study_name}_Refine")
        ref_res = ref_pipe.refine(
            gr_data=gr_data,
            cif_file=cif_file,
            pbc=pbc,
            qdepend=qdepend,
            rambda=rambda,
            qmin=qmin,
            qmax=qmax,
            fit_min=fit_min,
            fit_max=fit_max,
            output_dir=out_dir,
            save_results=True,
            **kwargs,
        )

        return {
            "reduction": red_res,
            "refinement": ref_res,
            "summary": ref_res["summary"],
        }

