import os
import json
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks

from orochi.base.core import OrochiBase


class CalibrationPipeline(OrochiBase):
    """End-to-end multi-stage instrument and total scattering calibration pipeline.
    
    Orchestrates `orochi.pixia`, `orochi.pdfget`, `orochi.pdfsim`, and `orochi.fit`
    without polluting individual domain packages with cross-package dependencies.
    """

    def __init__(self, standard: str = "ni", **kwargs):
        super().__init__()
        self.standard = standard.lower()
        self.target_lattice = kwargs.get("target_lattice", 3.5238 if self.standard == "ni" else 5.4111)
        self.distance = kwargs.get("distance", 227.575)
        self.wavelength = kwargs.get("wavelength", 0.18033)
        self.fix_wavelength = kwargs.get("fix_wavelength", False)
        self.fix_distance = kwargs.get("fix_distance", False)
        self.pixel_size = kwargs.get("pixel_size", 0.2)  # mm
        self.qmin = kwargs.get("qmin", 0.8)
        self.qmax = kwargs.get("qmax", 29.0 if self.standard == "ni" else 25.0)
        self.work_dir = kwargs.get("work_dir", None)
        self.calibrated_params = {}

    def get_standard_d_spacings(self) -> np.ndarray:
        """Retrieves theoretical d-spacings for certified calibration standards."""
        if self.standard == "ni":
            # FCC Ni (Fm-3m)
            a0 = self.target_lattice
            hkl_list = [
                (1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2),
                (4, 0, 0), (3, 3, 1), (4, 2, 0), (4, 2, 2), (5, 1, 1),
                (4, 4, 0), (5, 3, 1), (6, 0, 0), (6, 2, 0), (5, 3, 3),
                (6, 2, 2), (4, 4, 4), (7, 1, 1), (6, 4, 0), (6, 4, 2)
            ]
            d_list = [a0 / np.sqrt(h**2 + k**2 + l**2) for h, k, l in hkl_list]
            return np.array(d_list)
        elif self.standard in ["ceo2", "ceria"]:
            # Fluorite CeO2 (Fm-3m)
            a0 = self.target_lattice
            # Miller indices for fluorite
            hkl_list = [
                (1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2),
                (4, 0, 0), (3, 3, 1), (4, 2, 0), (4, 2, 2), (5, 1, 1),
                (4, 4, 0), (5, 3, 1), (6, 0, 0), (6, 2, 0), (5, 3, 3),
                (6, 2, 2), (4, 4, 4), (7, 1, 1), (6, 4, 0), (6, 4, 2)
            ]
            d_list = [a0 / np.sqrt(h**2 + k**2 + l**2) for h, k, l in hkl_list]
            return np.array(d_list)
        else:
            raise ValueError(f"Unsupported calibration standard: '{self.standard}'. Supported: 'ni', 'ceo2'.")

    def refine_distance_wavelength(
        self,
        pixia,
        fix_wavelength: bool | None = None,
        fix_distance: bool | None = None,
        max_peaks: int = 18,
    ) -> dict:
        """Refines sample-to-detector distance L and incident wavelength lambda from Bragg peak positions.

        Parameters
        ----------
        pixia : Pixia
            Configured Pixia instance after 1D integration.
        fix_wavelength : bool, optional
            Whether to hold wavelength fixed to nominal value (e.g. beamline specification).
        fix_distance : bool, optional
            Whether to hold detector distance fixed.
        max_peaks : int
            Maximum number of Bragg peaks to fit.

        Returns
        -------
        dict
            Refinement summary including fitted L, lambda, uncertainties, and peak-by-peak table.
        """
        if fix_wavelength is None:
            fix_wavelength = self.fix_wavelength
        if fix_distance is None:
            fix_distance = self.fix_distance

        if hasattr(pixia, "tt1d") and pixia.tt1d is not None:
            tt_raw = np.array(pixia.tt1d)
            i_raw = np.array(pixia.i1d)
        elif hasattr(pixia, "data") and isinstance(pixia.data, dict) and "tt" in pixia.data:
            tt_raw = np.array(pixia.data["tt"])
            i_raw = np.array(pixia.data["i"])
        else:
            raise AttributeError("Pixia instance has no 1D profile data ('tt1d', 'i1d'). Run twotheta_intensity() first.")

        d_all = self.get_standard_d_spacings()
        L_nom = float(pixia["distance"])
        lam_nom = float(pixia["rambda"])

        def gaussian(x, a, x0, sigma, bg):
            return a * np.exp(-(x - x0)**2 / (2.0 * sigma**2)) + bg

        fitted_tt = []
        valid_dsp = []
        peak_details = []

        for idx, dsp in enumerate(d_all):
            if len(valid_dsp) >= max_peaks:
                break
            sin_th = lam_nom / (2.0 * dsp)
            if sin_th >= 0.95:
                continue
            th_approx = 2.0 * np.degrees(np.arcsin(sin_th))
            if th_approx > np.max(tt_raw) or th_approx < np.min(tt_raw):
                continue

            window = 0.28  # degrees around nominal Bragg position
            mask = (tt_raw > th_approx - window) & (tt_raw < th_approx + window)
            if np.sum(mask) < 6:
                continue

            sub_tt = tt_raw[mask]
            sub_i = i_raw[mask]
            p0 = [np.max(sub_i) - np.min(sub_i), th_approx, 0.05, np.min(sub_i)]
            try:
                popt, _ = curve_fit(gaussian, sub_tt, sub_i, p0=p0, maxfev=2000)
                # Sanity check peak center within window
                if np.abs(popt[1] - th_approx) < window * 0.8:
                    fitted_tt.append(popt[1])
                    valid_dsp.append(dsp)
                    peak_details.append({
                        "peak_index": idx + 1,
                        "d_spacing_A": float(dsp),
                        "nominal_2theta_deg": float(th_approx),
                        "observed_2theta_deg": float(popt[1]),
                        "peak_intensity": float(popt[0]),
                        "fwhm_deg": float(2.355 * np.abs(popt[2]))
                    })
            except Exception:
                continue

        if len(fitted_tt) < 3:
            raise RuntimeError(f"Insufficient Bragg peaks detected ({len(fitted_tt)}) to refine geometry.")

        fitted_tt = np.array(fitted_tt)
        valid_dsp = np.array(valid_dsp)

        # Measured radial position on detector (in mm from beam center)
        R_obs_mm = L_nom * np.tan(np.radians(fitted_tt))

        if fix_wavelength and fix_distance:
            L_fit = L_nom
            lam_fit = lam_nom
            res_mm = 0.0
        elif fix_wavelength:
            # Fix lambda, refine L
            lam_fit = lam_nom
            def residual_L(p):
                L = p[0]
                s = lam_fit / (2.0 * valid_dsp)
                tt_c = 2.0 * np.arcsin(s)
                return L * np.tan(tt_c) - R_obs_mm

            sol = least_squares(residual_L, [L_nom], bounds=([L_nom * 0.7], [L_nom * 1.3]))
            L_fit = float(sol.x[0])
            res_mm = float(np.mean(np.abs(sol.fun)))
        elif fix_distance:
            # Fix L, refine lambda
            L_fit = L_nom
            def residual_lam(p):
                lam = p[0]
                s = lam / (2.0 * valid_dsp)
                if np.any(s >= 1.0): return np.ones_like(valid_dsp) * 1e4
                tt_c = 2.0 * np.arcsin(s)
                return L_fit * np.tan(tt_c) - R_obs_mm

            sol = least_squares(residual_lam, [lam_nom], bounds=([lam_nom * 0.8], [lam_nom * 1.2]))
            lam_fit = float(sol.x[0])
            res_mm = float(np.mean(np.abs(sol.fun)))
        else:
            # Simultaneous refinement of L and lambda
            def residual_both(p):
                L, lam = p
                s = lam / (2.0 * valid_dsp)
                if np.any(s >= 1.0): return np.ones_like(valid_dsp) * 1e4
                tt_c = 2.0 * np.arcsin(s)
                return L * np.tan(tt_c) - R_obs_mm

            sol = least_squares(
                residual_both,
                [L_nom, lam_nom],
                bounds=([L_nom * 0.8, lam_nom * 0.8], [L_nom * 1.2, lam_nom * 1.2])
            )
            L_fit = float(sol.x[0])
            lam_fit = float(sol.x[1])
            res_mm = float(np.mean(np.abs(sol.fun)))

        res_px = res_mm / self.pixel_size

        # Apply calibrated parameters back to Pixia
        pixia["distance"] = L_fit
        pixia["rambda"] = lam_fit
        self.distance = L_fit
        self.wavelength = lam_fit

        summary = {
            "mode": "fixed_wavelength" if fix_wavelength else ("fixed_distance" if fix_distance else "simultaneous"),
            "nominal_distance_mm": L_nom,
            "calibrated_distance_mm": L_fit,
            "distance_shift_mm": L_fit - L_nom,
            "nominal_wavelength_A": lam_nom,
            "calibrated_wavelength_A": lam_fit,
            "wavelength_shift_A": lam_fit - lam_nom,
            "num_peaks_fitted": len(fitted_tt),
            "mean_peak_residual_mm": res_mm,
            "mean_peak_residual_px": res_px,
            "peaks": peak_details
        }
        return summary

    def run(
        self,
        pixia,
        empty_pixia=None,
        fix_wavelength: bool = False,
        fix_distance: bool = False,
        run_pdf_instrumental: bool = True,
    ) -> dict:
        """Executes the complete multi-stage calibration pipeline.

        Stages:
        1. Beamstop shadow detection & masking
        2. Beam center & detector tilt optimization via opposing sector azimuthal symmetry
        3. Bragg peak detection & (L, lambda) refinement with flexible fixing options
        4. Synchrotron polarization purity calibration
        5. PDF instrumental resolution refinement (Qdamp, Qbroad) & lattice validation
        """
        report = {
            "standard": self.standard,
            "target_lattice_A": self.target_lattice,
            "stages": {}
        }

        # Stage 1: Beamstop mask
        try:
            stop_r = pixia.auto_beamstop_mask()
            report["stages"]["beamstop_mask"] = {"beamstop_radius_px": stop_r}
        except Exception as e:
            report["stages"]["beamstop_mask"] = {"status": "skipped", "reason": str(e)}

        # Stage 2: Beam center & tilt
        try:
            center_tilt = pixia.align_center_tilt()
            report["stages"]["center_tilt"] = center_tilt
        except Exception as e:
            report["stages"]["center_tilt"] = {"status": "skipped", "reason": str(e)}

        # Execute 1D integration to prepare Bragg profile
        pixia.read()
        if empty_pixia is not None:
            pixia.bg_subtraction()
        pixia.convert_selected()
        pixia.twotheta_intensity()

        # Stage 3: Bragg peak (L, lambda) refinement
        geom_report = self.refine_distance_wavelength(
            pixia,
            fix_wavelength=fix_wavelength,
            fix_distance=fix_distance
        )
        report["stages"]["geometry_refinement"] = geom_report

        # Re-integrate with updated L and lambda
        pixia.twotheta_intensity()
        pixia.get_background_curve()

        # Stage 4: Polarization tuning
        try:
            pol = pixia.tune_polarization()
            report["stages"]["polarization"] = {"calibrated_polar": pol}
        except Exception as e:
            report["stages"]["polarization"] = {"calibrated_polar": float(pixia.get("polar", 1.0))}

        # Stage 5: PDF Instrumental calibration
        if run_pdf_instrumental:
            try:
                from orochi.pdfget import PDFget
                from orochi.pdfsim import Simulator
                from orochi.phase import Phase
                from orochi.fit import LeastSquaresOptimizer

                phase = Phase()
                db_dir = os.path.join(os.path.dirname(__file__), "database")
                cif_path = os.path.join(db_dir, f"{self.standard}.cif")
                if os.path.isfile(cif_path):
                    phase.read_file(cif_path)
                phase["lattice"] = [self.target_lattice] * 3 + [90.0] * 3
                phase["shape_type"] = "analytical"
                phase["sphere"] = [1000.0, 1000.0, 1000.0]
                phase.set_lattice_constraints()
                phase()

                pdfget = PDFget()
                pdfget.from_pixia(pixia)
                pdfget.set_phase(phase)
                pdfget["source"] = "synchrotron"
                pdfget["qmin"] = self.qmin
                pdfget["qmax"] = self.qmax
                pdfget["fitmin"] = 12.0
                pdfget["fitmax"] = self.qmax - 0.5
                pdfget["padding"] = 5
                pdfget.run()

                work_dir = self.work_dir or os.path.dirname(os.path.abspath(pixia["sample_file"]))
                gr_path = os.path.join(work_dir, f"{self.standard}_calib_fit.gr")
                pdfget.save_file(gr_path)

                sim = Simulator()
                sim.phase = phase
                sim["rambda"] = float(pixia["rambda"])
                sim["qmin"] = self.qmin
                sim["qmax"] = self.qmax
                sim["rmin"] = 2.0
                sim["rmax"] = 15.0
                sim["rstep"] = 0.01
                sim["qdepend"] = True
                sim()

                phase["lattice", "min"] = [self.target_lattice * 0.95] * 3 + [89.0] * 3
                phase["lattice", "max"] = [self.target_lattice * 1.05] * 3 + [91.0] * 3
                phase["lattice", "optimize"] = [True, False, False, False, False, False]
                phase.set_lattice_constraints()
                phase["adp", "min"] = [0.0005]
                phase["adp", "max"] = [0.03]
                phase["adp", "optimize"] = [True]
                phase()

                sim["scale"] = 0.9
                sim["scale", "optimize"] = True
                sim["qdamp"] = 0.026
                sim["qdamp", "optimize"] = True
                sim["qbroad"] = 0.015
                sim["qbroad", "optimize"] = True
                sim["delta2"] = 2.4
                sim["delta2", "optimize"] = True

                opt = LeastSquaresOptimizer()
                opt.work_dir = work_dir
                opt.study_name = f"{self.standard}_lattice_calib"
                opt.simulators = [sim]
                opt["fit_min"] = 2.0
                opt["fit_max"] = 15.0
                opt.read_file(gr_path)
                opt()

                fitted_a = float(phase["lattice"][0])
                report["stages"]["pdf_instrumental"] = {
                    "Rw_percent": float(opt.Rw * 100.0),
                    "refined_lattice_a_A": fitted_a,
                    "target_lattice_a_A": self.target_lattice,
                    "lattice_discrepancy_percent": float(np.abs(fitted_a - self.target_lattice) / self.target_lattice * 100.0),
                    "refined_qdamp_A_inv": float(sim["qdamp"]),
                    "refined_qbroad_A_inv": float(sim["qbroad"]),
                    "refined_delta2": float(sim["delta2"]),
                    "refined_adp_A2": float(phase["adp"][0])
                }
            except Exception as e:
                report["stages"]["pdf_instrumental"] = {"status": "skipped", "reason": str(e)}

        self.calibrated_params = report
        return report

    def save_calibration(self, filepath: str):
        """Exports the calibrated pipeline parameter dictionary to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.calibrated_params, f, indent=2, ensure_ascii=False)
