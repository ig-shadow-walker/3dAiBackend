"""
Mesh editing models for local 3D mesh editing tasks.

This module provides base classes for mesh editing operations,
particularly for local region editing using VoxHammer.
"""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)


class MeshEditingModel(BaseModel):
    """
    Base class for mesh editing models.
    
    Common functionality for all mesh editing models (text/image-guided).
    """
    
    def __init__(
        self,
        model_id: str,
        model_path: str,
        vram_requirement: int,
        feature_type: str = "mesh_editing",
        supported_output_formats: Optional[List[str]] = None,
    ):
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type=feature_type,
        )
        
        self.supported_output_formats = supported_output_formats or ["glb"]
    
    def _validate_common_inputs(self, inputs: Dict[str, Any]) -> str:
        """Validate common inputs and return output format."""
        # Get output format
        output_format = inputs.get("output_format", "glb")
        if output_format not in self.supported_output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return output_format
    
    def _create_common_response(
        self, inputs: Dict[str, Any], output_format: str
    ) -> Dict[str, Any]:
        """Create common response structure."""
        return {
            "output_mesh_path": f"edited_mesh_{self.model_id}.{output_format}",
            "success": True,
        }


class TextMeshEditingModel(MeshEditingModel):
    """
    Text-guided mesh editing model.
    
    Inputs: Mesh + mask region + text prompts (source and target)
    Outputs: Edited mesh with local modifications
    """
    
    def __init__(
        self,
        model_id: str,
        model_path: str,
        vram_requirement: int,
        feature_type: str = "text_mesh_editing",
        supported_output_formats: Optional[List[str]] = None,
    ):
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type=feature_type,
            supported_output_formats=supported_output_formats,
        )
    
    def _load_model(self):
        """Load the text-guided mesh editing model. To be implemented by adapters."""
        logger.info(f"Loading text mesh editing model: {self.model_id}")
        pass
    
    def _unload_model(self):
        """Unload the text mesh editing model."""
        logger.info(f"Unloading text mesh editing model: {self.model_id}")
        pass
    
    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text-guided mesh editing request.
        
        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input mesh (required)
                - mask_type: Type of mask ("bbox" or "ellipsoid") (required)
                - mask_center: Center of mask region [x, y, z] (required)
                - mask_params: Dimensions/radii for mask [p1, p2, p3] (required)
                - source_prompt: Text describing original region (required)
                - target_prompt: Text describing desired edited region (required)
                - output_format: Output format (default: "glb")
                - num_views: Number of rendering views (default: 150)
                - resolution: Rendering resolution (default: 512)
        
        Returns:
            Dictionary containing:
                - output_mesh_path: Path to edited mesh file
                - editing_info: Additional editing metadata
        """
        # Validate required inputs
        if "mesh_path" not in inputs:
            raise ValueError("mesh_path is required for mesh editing")
        if "mask_type" not in inputs:
            raise ValueError("mask_type is required for mesh editing")
        if "mask_center" not in inputs:
            raise ValueError("mask_center is required for mesh editing")
        if "mask_params" not in inputs:
            raise ValueError("mask_params is required for mesh editing")
        if "source_prompt" not in inputs:
            raise ValueError("source_prompt is required for text-guided editing")
        if "target_prompt" not in inputs:
            raise ValueError("target_prompt is required for text-guided editing")
        
        output_format = self._validate_common_inputs(inputs)
        
        logger.info(f"Processing text-guided mesh editing request: {inputs['mesh_path']}")
        
        # This will be implemented by specific adapters
        response = self._create_common_response(inputs, output_format)
        response.update(
            {
                "source_prompt": inputs["source_prompt"],
                "target_prompt": inputs["target_prompt"],
                "editing_info": {
                    "input_type": "text",
                    "mask_type": inputs["mask_type"],
                    "success": True,
                },
            }
        )
        
        return response
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats."""
        return {
            "input": ["glb", "obj", "ply"],
            "output": self.supported_output_formats
        }


class ImageMeshEditingModel(MeshEditingModel):
    """
    Image-guided mesh editing model.
    
    Inputs: Mesh + mask region + reference images
    Outputs: Edited mesh with local modifications
    """
    
    def __init__(
        self,
        model_id: str,
        model_path: str,
        vram_requirement: int,
        feature_type: str = "image_mesh_editing",
        supported_output_formats: Optional[List[str]] = None,
    ):
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type=feature_type,
            supported_output_formats=supported_output_formats,
        )
    
    def _load_model(self):
        """Load the image-guided mesh editing model. To be implemented by adapters."""
        logger.info(f"Loading image mesh editing model: {self.model_id}")
        pass
    
    def _unload_model(self):
        """Unload the image mesh editing model."""
        logger.info(f"Unloading image mesh editing model: {self.model_id}")
        pass
    
    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-guided mesh editing request.
        
        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input mesh (required)
                - mask_type: Type of mask ("bbox" or "ellipsoid") (required)
                - mask_center: Center of mask region [x, y, z] (required)
                - mask_params: Dimensions/radii for mask [p1, p2, p3] (required)
                - source_image_path: Path to source reference image (required)
                - target_image_path: Path to target reference image (required)
                - mask_image_path: Path to 2D mask image (required)
                - output_format: Output format (default: "glb")
                - num_views: Number of rendering views (default: 150)
                - resolution: Rendering resolution (default: 512)
        
        Returns:
            Dictionary containing:
                - output_mesh_path: Path to edited mesh file
                - editing_info: Additional editing metadata
        """
        # Validate required inputs
        if "mesh_path" not in inputs:
            raise ValueError("mesh_path is required for mesh editing")
        if "mask_type" not in inputs:
            raise ValueError("mask_type is required for mesh editing")
        if "mask_center" not in inputs:
            raise ValueError("mask_center is required for mesh editing")
        if "mask_params" not in inputs:
            raise ValueError("mask_params is required for mesh editing")
        if "source_image_path" not in inputs:
            raise ValueError("source_image_path is required for image-guided editing")
        if "target_image_path" not in inputs:
            raise ValueError("target_image_path is required for image-guided editing")
        if "mask_image_path" not in inputs:
            raise ValueError("mask_image_path is required for image-guided editing")
        
        output_format = self._validate_common_inputs(inputs)
        
        logger.info(f"Processing image-guided mesh editing request: {inputs['mesh_path']}")
        
        # This will be implemented by specific adapters
        response = self._create_common_response(inputs, output_format)
        response.update(
            {
                "source_image": inputs["source_image_path"],
                "target_image": inputs["target_image_path"],
                "mask_image": inputs["mask_image_path"],
                "editing_info": {
                    "input_type": "image",
                    "mask_type": inputs["mask_type"],
                    "success": True,
                },
            }
        )
        
        return response
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats."""
        return {
            "input": ["glb", "obj", "ply"],
            "output": self.supported_output_formats
        }

