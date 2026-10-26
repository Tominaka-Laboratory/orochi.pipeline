import os
import json
from typing import Dict, Any, Optional, Union
import numpy as np

from orochi.base.core import OrochiBase
from orochi.pdfget import PDFget


class Reduction1DPipeline(OrochiBase):
    """Automated pipeline for converting 1D total scattering profiles into pair distribution functions (PDF).

    Handles background subtraction, Lorch window filtering, composition normalization,
    and automated low-r ripple tuning via `orochi.pdfget`.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.work_dir = kwargs.get("work_dir", None)
        self.study_name = kwargs.get("study_name", "Reduction1D")

    def process(
        self,
        sample_data: Union[str, np.ndarray],
        background_data: Optional[Union[str, np.ndarray]] = None,
        composition: str = "Ni 1",
        rambda: float = 0.18033,
        qmin: float = 0.8,
        qmax: float = 25.0,
        fitmin: Optional[float] = None,
        fitmax: Optional[float] = None,
        rmin: float = 0.0,
        rmax: float = 50.0,
        rstep: float = 0.01,
        lorch_width: float = 0.0,
        zero_padding: int = 0,
        tune: bool = False,
        output_dir: Optional[str] = None,
        save_gr: bool = True,
        save_json: bool = True,
    ) -> Dict[str, Any]:
        """Executes 1D-to-PDF data reduction.

        Parameters
        ----------
        sample_data : str or np.ndarray
            Path to 1D diffraction file (two columns: Q and I, or 2theta and I) or numpy array.
        background_data : str or np.ndarray, optional
            Path to blank/container 1D diffraction file or numpy array.
        composition : str
            Chemical formula (e.g. "CeO2", "Ni 1", "Pt O2").
        rambda : float
            X-ray wavelength in Angstroms.
        qmin, qmax : float
            Scattering vector Q range in Angstrom^-1.
        fitmin, fitmax : float, optional
            High-Q fitting window for coherent normalization. Defaults to [qmax/2, qmax].
        rmin, rmax, rstep : float
            Real-space radial distance grid in Angstroms.
        lorch_width : float
            Lorch modification damping parameter (0.0 for raw, unbroadened).
        zero_padding : int
            Zero-padding factor for high-resolution Fourier grid.
        tune : bool
            Whether to run automated polynomial ripple minimization for low-r cleanup.
        output_dir : str, optional
            Directory to save output files.
        save_gr : bool
            Whether to write output .gr file.
        save_json : bool
            Whether to save reduced parameters and diagnostic report to JSON.

        Returns
        -------
        dict
            Dictionary containing r, G(r), Q, S(Q), F(Q), and reduction parameters.
        """
        out_dir = output_dir or self.work_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        pdfget = PDFget()
        pdfget.work_dir = out_dir
        pdfget.study_name = self.study_name

        if isinstance(sample_data, str) and os.path.exists(sample_data):
            pdfget.read_file(sample_data, target="sample")
        elif isinstance(sample_data, np.ndarray):
            pdfget.set_data(sample_data, target="sample")

        if background_data is not None:
            if isinstance(background_data, str) and os.path.exists(background_data):
                pdfget.read_file(background_data, target="background")
            elif isinstance(background_data, np.ndarray):
                pdfget.set_data(background_data, target="background")

        pdfget["composition"] = composition
        pdfget["rambda"] = rambda
        pdfget["qmin"] = qmin
        pdfget["qmax"] = qmax
        pdfget["fitmin"] = fitmin if fitmin is not None else max(qmin, qmax * 0.5)
        pdfget["fitmax"] = fitmax if fitmax is not None else qmax
        pdfget["rmin"] = rmin
        pdfget["rmax"] = rmax
        pdfget["rstep"] = rstep
        pdfget["lorch_width"] = lorch_width
        pdfget["zero_padding"] = zero_padding

        if tune:
            pdfget.tune()
        else:
            pdfget()

        results = {
            "r": pdfget.r if hasattr(pdfget, "r") else None,
            "gr": pdfget.gr if hasattr(pdfget, "gr") else None,
            "q": pdfget.q if hasattr(pdfget, "q") else None,
            "sq": pdfget.sq if hasattr(pdfget, "sq") else None,
            "fq": pdfget.fq if hasattr(pdfget, "fq") else None,
            "composition": composition,
            "qmin": qmin,
            "qmax": qmax,
            "parameters": dict(pdfget.par) if hasattr(pdfget, "par") else {},
        }

        if save_gr and results["gr"] is not None:
            gr_filename = os.path.join(out_dir, f"{self.study_name}.gr")
            pdfget.save_file(gr_filename)
            results["gr_path"] = gr_filename

        if save_json:
            json_filename = os.path.join(out_dir, f"{self.study_name}_reduction.json")
            save_dict = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()
                if k not in ["r", "gr", "q", "sq", "fq"]
            }
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(save_dict, f, indent=4)
            results["json_path"] = json_filename

        return results

