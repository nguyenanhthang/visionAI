"""GUI widgets."""
from .acquisition_panel import AcquisitionPanel
from .collapsible import CollapsibleSection, HRule, SectionLabel
from .image_canvas import ImageCanvas
from .operator_panel import OperatorSidebar
from .param_dialog import ParamDialog
from .pipeline_panel import PipelinePanel
from .results_view import ResultsView

__all__ = [
    "AcquisitionPanel",
    "CollapsibleSection",
    "HRule",
    "SectionLabel",
    "ImageCanvas",
    "OperatorSidebar",
    "ParamDialog",
    "PipelinePanel",
    "ResultsView",
]
