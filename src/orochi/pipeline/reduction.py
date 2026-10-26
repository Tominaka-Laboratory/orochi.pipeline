import os
import json
from typing import List, Dict, Any

from orochi.base.core import OrochiBase
from orochi.pixia import Pixia
from orochi.pdfget import PDFget


class ReductionPipeline(OrochiBase):
    """Automated high-throughput pipeline for batch 2D image reduction to 1D pair distribution functions."""

    def __init__(self, calibration: Dict[str, Any] | str | None = None, **kwargs):
        super().__init__()
        self.calib = {}
        if isinstance(calibration, str) and os.path.isfile(calibration):
            with open(calibration, "r", encoding="utf-8") as f:
                self.calib = json.load(f)
        elif isinstance(calibration, dict):
            self.calib = calibration

    def process_dataset(
        self,
        sample_file: str,
        background_file: str | None = None,
        composition: str = "Ni 1",
        qmax: float = 29.0,
        qmin: float = 0.8,
        output_dir: str | None = None,
        azimuthal_mode: str = "auto",
        save_json: bool = True,
        save_gr: bool = True,
    ) -> Dict[str, Any]:
        """Processes a single 2D total scattering pattern into reduced F(Q) and G(r)."""
        pix = Pixia()
        pix["sample_file"] = sample_file
        if background_file:
            pix["background_file"] = background_file

        # Apply calibrated parameters if present
        if "stages" in self.calib:
            st = self.calib["stages"]
            if "center_tilt" in st and "refined_center_px" in st["center_tilt"]:
                pix["x0"] = st["center_tilt"]["refined_center_px"][0]
                pix["y0"] = st["center_tilt"]["refined_center_px"][1]
                pix["tiltx"] = st["center_tilt"]["refined_tilt_deg"][0]
                pix["tilty"] = st["center_tilt"]["refined_tilt_deg"][1]
            if "geometry_refinement" in st:
                pix["distance"] = st["geometry_refinement"]["calibrated_distance_mm"]
                pix["rambda"] = st["geometry_refinement"]["calibrated_wavelength_A"]
            if "polarization" in st:
                pix["polar"] = st["polarization"]["calibrated_polar"]

        # Azimuthal mode selection
        if azimuthal_mode == "auto":
            pix.auto_azimuthal_mode(qmax=qmax)
        elif azimuthal_mode in ["diagonals", "diagonal"]:
            pix.set_azimuthal_mode("diagonals", width=18.0)
        else:
            pix.set_azimuthal_mode("full")

        pix.read()
        if background_file:
            pix.bg_subtraction()
        pix.convert_selected()
        pix.twotheta_intensity()
        pix.get_background_curve()

        pdf = PDFget()
        pdf.from_pixia(pix)
        pdf["composition"] = composition
        pdf["qmin"] = qmin
        pdf["qmax"] = qmax
        pdf.run()

        out_dir = output_dir or os.path.dirname(os.path.abspath(sample_file))
        base_name = os.path.splitext(os.path.basename(sample_file))[0]

        result_paths = {}
        if save_gr:
            gr_path = os.path.join(out_dir, f"{base_name}.gr")
            pdf.save_file(gr_path)
            result_paths["gr_path"] = gr_path

        if save_json:
            json_path = os.path.join(out_dir, f"{base_name}_pdfget.json")
            pdf.save_json(json_path)
            result_paths["json_path"] = json_path

        return {
            "sample_file": sample_file,
            "qmax": qmax,
            "outputs": result_paths,
            "r": pdf.data["r"].tolist(),
            "Gr": pdf.data["Gr"].tolist()
        }
