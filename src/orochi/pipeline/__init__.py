from .calibration import CalibrationPipeline
from .reduction import ReductionPipeline
from .reduction_1d import Reduction1DPipeline
from .refinement import RefinementPipeline
from .full_pipeline import FullAnalysisPipeline

# Convenient aliases
Reduction2DPipeline = ReductionPipeline
ProfileReductionPipeline = Reduction1DPipeline
StructureRefinementPipeline = RefinementPipeline

__all__ = [
    "CalibrationPipeline",
    "ReductionPipeline",
    "Reduction2DPipeline",
    "Reduction1DPipeline",
    "ProfileReductionPipeline",
    "RefinementPipeline",
    "StructureRefinementPipeline",
    "FullAnalysisPipeline",
]
