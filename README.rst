.. image:: images/orochi_icon.png
   :align: center
   :width: 80px
   :target: https://github.com/Tominaka-Laboratory/orochi.pipeline
   :alt: Orochi Logo

===============
orochi.pipeline
===============

.. |title| replace:: orochi.pipeline
.. _title: https://Tominaka-Laboratory.github.io/orochi.pipeline

.. |Icon| image:: images/orochi_icon.png
   :target: https://github.com/Tominaka-Laboratory/orochi.pipeline
   :height: 100px

|PyPi| |Forge| |PythonVersion| |ChemRxiv| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |ChemRxiv| image:: https://img.shields.io/badge/ChemRxiv-10.26434%2Fchemrxiv.15010802-blue.svg
        :target: https://doi.org/10.26434/chemrxiv.15010802

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/Tominaka-Laboratory/orochi.pipeline/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/Tominaka-Laboratory/orochi.pipeline/actions/workflows/tests.yml

.. |Codecov| image:: https://codecov.io/gh/Tominaka-Laboratory/orochi.pipeline/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/Tominaka-Laboratory/orochi.pipeline

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/orochi.pipeline
        :target: https://anaconda.org/conda-forge/orochi.pipeline

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff

.. |PyPi| image:: https://img.shields.io/pypi/v/orochi.pipeline
        :target: https://pypi.org/project/orochi.pipeline/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/orochi.pipeline
        :target: https://pypi.org/project/orochi.pipeline/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/Tominaka-Laboratory/orochi.pipeline/issues

---
Orochi Pipeline: Automated Workflows for Synchrotron Total Scattering
---

``orochi.pipeline`` orchestrates specialized Orochi 4.0 modules (``pixia``, ``pdfget``, ``phase``, ``pdfsim``, and ``fit``) into high-level, production-ready workflows for synchrotron and laboratory total scattering experiments.

Available Pipelines
-------------------

1. **``CalibrationPipeline``**: Automated detector geometry refinement (sample distance, beam center, detector tilts, wavelength) using certified powder diffraction standards (CeO₂, Ni, LaB₆).
2. **``Reduction2DPipeline``**: High-throughput transformation of 2D area detector frames into 1D profiles *I(Q)* and pair distribution functions *G(r)*, applying polarization, solid angle, and oblique sensor absorption corrections.
3. **``Reduction1DPipeline``**: 1D profile total scattering reduction with Faber--Ziman normalization, background subtraction, and automated Fourier ripple suppression via polynomial optimization.
4. **``RefinementPipeline``**: Real-space structural refinement against crystallographic CIF models with anisotropic directional PBCs (1D/2D/3D), nanoparticle shape envelopes, and exact Q-dependent form-factor simulations.
5. **``FullAnalysisPipeline``**: End-to-end analysis chaining raw experimental reduction directly into structural refinement in a single script.

Quickstart: End-to-End Analysis Pipeline
----------------------------------------

The following example executes complete 1D profile reduction followed by real-space CIF refinement in a single call:

.. code-block:: python

    from orochi.pipeline import FullAnalysisPipeline

    pipeline = FullAnalysisPipeline(work_dir="./results", study_name="CeO2_Experiment")

    # Run complete workflow: reduction + structural refinement
    results = pipeline.from_1d_profile(
        sample_data="CeO2_raw_profile.xy",
        background_data="Capillary_background.xy",
        composition="CeO2",
        cif_file="CeO2.cif",
        rambda=0.18033,           # X-ray wavelength in Angstroms
        qmin=0.8,
        qmax=25.0,
        fit_min=1.5,
        fit_max=20.0,
        pbc=(True, True, True),   # 3D bulk periodicity
        qdepend=True,             # Exact Q-dependent atomic form factors
        tune=True,                # Automated low-r ripple minimization
    )

    # Inspect refined parameters and goodness-of-fit
    summary = results["summary"]
    print("Refined Scale:", summary["best_params"].get("sim_0_scale"))
    print("Goodness of fit (Rw):", summary["rw"])

Installation
------------

Install via ``pip``:

.. code-block:: bash

    pip install orochi.pipeline

Or install in development mode from source:

.. code-block:: bash

    git clone https://github.com/Tominaka-Laboratory/orochi.pipeline.git
    cd orochi.pipeline
    pip install -e .

To confirm successful installation:

.. code-block:: bash

    python -c "import orochi.pipeline; print(orochi.pipeline.__version__)"

Citation
--------

If you use ``orochi.pipeline`` in your research, please cite:

    Orochi Software Suite, Tominaka Laboratory, National Institute for Materials Science (NIMS).
    https://github.com/Tominaka-Laboratory/orochi.pipeline



License
-------

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See the `LICENSE.rst` file for details.

This copyleft license guarantees that Orochi remains freely accessible, open, and transparent to the entire scientific community, and that all derivative works must likewise remain open source.

Citation
--------

If you use this package in academic research, please cite:

.. code-block:: bibtex

    @article{tominaka2026orochi,
      title   = {Orochi: A Physics-Grounded Python Package for Pair Distribution Function Analysis of Crystalline and Low-Dimensional Materials},
      author  = {Tominaka, Satoshi},
      journal = {ChemRxiv},
      year    = {2026},
      doi     = {10.26434/chemrxiv.15010802},
      url     = {https://doi.org/10.26434/chemrxiv.15010802}
    }

Contact
-------

For bug reports and questions, visit `GitHub Issues <https://github.com/Tominaka-Laboratory/orochi.pipeline/issues>`_ or contact Dr. Satoshi Tominaka at ``tominaka.satoshi@nims.go.jp``.
