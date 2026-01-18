"""
Model adapters for integrating specific AI models into the framework.

This package contains adapters that bridge between our model specifications
and actual AI model implementations.
"""

from .hunyuan3d_adapter_v21 import (
    Hunyuan3DV21ImageMeshPaintingAdapter,
    Hunyuan3DV21ImageToRawMeshAdapter,
    Hunyuan3DV21ImageToTexturedMeshAdapter,
)
from .partfield_adapter import PartFieldSegmentationAdapter
from .partpacker_adapter import PartPackerImageToRawMeshAdapter
from .trellis_adapter import (
    TrellisImageMeshPaintingAdapter,
    TrellisImageToTexturedMeshAdapter,
    TrellisTextMeshPaintingAdapter,
    TrellisTextToTexturedMeshAdapter,
)
from .trellis2_adapter import (
    Trellis2ImageMeshPaintingAdapter,
    Trellis2ImageToTexturedMeshAdapter,

)
from .unirig_adapter import UniRigAdapter

__all__ = [
    "Hunyuan3DV21ImageMeshPaintingAdapter",
    "Hunyuan3DV21ImageToRawMeshAdapter",
    "Hunyuan3DV21ImageToTexturedMeshAdapter",
    "PartFieldSegmentationAdapter",
    "PartPackerImageToRawMeshAdapter",
    "TrellisImageMeshPaintingAdapter",
    "TrellisImageToTexturedMeshAdapter",
    "TrellisTextMeshPaintingAdapter",
    "TrellisTextToTexturedMeshAdapter",
    "Trellis2ImageMeshPaintingAdapter",
    "Trellis2ImageToTexturedMeshAdapter",
    "UniRigAdapter",
]
