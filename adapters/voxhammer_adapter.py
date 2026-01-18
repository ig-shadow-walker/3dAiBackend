"""
VoxHammer model adapters for local mesh editing.

This adapter integrates VoxHammer with TRELLIS for text/image-guided
local 3D mesh editing operations.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from utils.voxhammer_pipeline_helper import VoxHammerInferenceHelper
from core.models.base import ModelStatus
from core.models.mesh_editing_models import TextMeshEditingModel, ImageMeshEditingModel
from core.utils.mask_generator import MaskGenerator
from core.utils.file_utils import OutputPathGenerator
from core.utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class VoxHammerTextMeshEditingAdapter(TextMeshEditingModel):
    """
    Adapter for VoxHammer text-guided mesh editing.
    
    Performs local mesh editing using text prompts to guide the editing region.
    """
    
    FEATURE_TYPE = "text_mesh_editing"
    MODEL_ID = "voxhammer_text_mesh_editing"
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        vram_requirement: int = 40960,  # 40GB VRAM (TRELLIS + rendering overhead)
        voxhammer_root: Optional[str] = None,
        trellis_root: Optional[str] = None,
        feature_type: Optional[str] = None,
        supported_output_formats: Optional[List[str]] = None,
    ):
        if model_id is None:
            model_id = self.MODEL_ID
        if model_path is None:
            model_path = os.path.abspath(
                os.path.join(os.getcwd(), "pretrained", "VoxHammer")
            )
        if voxhammer_root is None:
            voxhammer_root = os.path.abspath(
                os.path.join(os.getcwd(), "thirdparty", "VoxHammer")
            )

        if trellis_root is None:
            trellis_root = os.path.abspath(
                os.path.join(os.getcwd(), "thirdparty", "TRELLIS")
            )
        if feature_type is None:
            feature_type = self.FEATURE_TYPE
        if supported_output_formats is None:
            supported_output_formats = ["glb"]
        
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_output_formats=supported_output_formats,
            feature_type=feature_type,
        )
        
        self.model_path = Path(model_path)
        self.voxhammer_root = Path(voxhammer_root)
        self.trellis_root = Path(trellis_root)
        
        # Pipeline components
        self.trellis_pipeline = None
        self.voxhammer_helper = None
        self.mask_generator = MaskGenerator()
        
        # Utilities
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")
        
        # Add paths to sys.path
        if str(self.voxhammer_root) not in sys.path:
            sys.path.insert(0, str(self.voxhammer_root))
        if str(self.trellis_root) not in sys.path:
            sys.path.insert(0, str(self.trellis_root))
        
        logger.info(f"Initialized VoxHammer text editing adapter with root: {voxhammer_root}")
    
    def _load_model(self):
        """Load TRELLIS pipeline for text-guided editing."""
        try:
            logger.info("Loading TRELLIS text-to-3D pipeline for VoxHammer...")
            
            from trellis.pipelines import TrellisTextTo3DPipeline
            
            # Load TRELLIS text pipeline
            self.trellis_pipeline = TrellisTextTo3DPipeline.from_pretrained(
                "fishwowater/TRELLIS-text-large-voxhammer", 
                cache_dir=str(self.model_path / "TRELLIS-text-large-voxhammer")
            )
            self.trellis_pipeline.cuda()
            
            # Create VoxHammer helper
            self.voxhammer_helper = VoxHammerInferenceHelper(
                voxhammer_root=str(self.voxhammer_root),
                trellis_pipeline=self.trellis_pipeline,
                is_text_mode=True,
            )
            
            logger.info("VoxHammer text editing models loaded successfully")
            return {"trellis_pipeline": self.trellis_pipeline}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Failed to load VoxHammer text editing models: {str(e)}")
            raise Exception(f"Failed to load VoxHammer models: {str(e)}")
    
    def _unload_model(self):
        """Unload models."""
        try:
            if self.trellis_pipeline is not None:
                del self.trellis_pipeline
                self.trellis_pipeline = None
            
            if self.voxhammer_helper is not None:
                del self.voxhammer_helper
                self.voxhammer_helper = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("VoxHammer text editing models unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading VoxHammer models: {str(e)}")
    
    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text-guided mesh editing request using VoxHammer.
        
        Args:
            inputs: Dictionary with mesh_path, mask params, prompts
        
        Returns:
            Dictionary with editing results
        """
        try:
            # Validate inputs
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required")
            if "mask_type" not in inputs:
                raise ValueError("mask_type is required")
            if "mask_center" not in inputs:
                raise ValueError("mask_center is required")
            if "mask_params" not in inputs:
                raise ValueError("mask_params is required")
            if "source_prompt" not in inputs:
                raise ValueError("source_prompt is required")
            if "target_prompt" not in inputs:
                raise ValueError("target_prompt is required")
            
            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")
            
            # Extract parameters
            output_format = inputs.get("output_format", "glb")
            mask_type = inputs["mask_type"]
            mask_center = inputs["mask_center"]
            mask_params = inputs["mask_params"]
            source_prompt = inputs["source_prompt"]
            target_prompt = inputs["target_prompt"]
            num_views = inputs.get("num_views", 150)
            resolution = inputs.get("resolution", 512)
            
            logger.info(
                f"Editing mesh with VoxHammer (text-guided): {mesh_path}"
            )
            logger.info(f"Source: '{source_prompt}' -> Target: '{target_prompt}'")
            
            # Create output directory
            base_name = f"{self.model_id}_{mesh_path.stem}"
            output_dir = self.path_generator.base_output_dir / "voxhammer" / f"{mesh_path.stem}_{int(__import__('time').time())}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mask mesh
            logger.info(f"Creating {mask_type} mask at center {mask_center}...")
            mask_glb_path = os.path.join(output_dir, "mask.glb")
            self.mask_generator.create_mask_from_params(
                mask_type=mask_type,
                center=mask_center,
                params=mask_params,
                output_path=mask_glb_path,
            )
            
            # Generate output path
            output_mesh_path = str(self.path_generator.generate_mesh_path(
                self.model_id, base_name, output_format
            ))
            
            # Run VoxHammer editing pipeline
            logger.info("Running VoxHammer text-guided editing pipeline...")
            self.voxhammer_helper.edit_mesh_with_text(
                input_mesh_path=str(mesh_path),
                mask_glb_path=mask_glb_path,
                output_path=output_mesh_path,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                num_views=num_views,
                resolution=resolution,
            )
            
            # Load final mesh for statistics
            final_mesh = self.mesh_processor.load_mesh(output_mesh_path)
            mesh_stats = self.mesh_processor.get_mesh_stats(final_mesh)
            
            # Create response
            response = {
                "output_mesh_path": output_mesh_path,
                "success": True,
                "editing_info": {
                    "model": self.model_id,
                    "input_mesh": str(mesh_path),
                    "output_format": output_format,
                    "vertex_count": mesh_stats["vertex_count"],
                    "face_count": mesh_stats["face_count"],
                    "mask_type": mask_type,
                    "mask_center": mask_center,
                    "mask_params": mask_params,
                    "source_prompt": source_prompt,
                    "target_prompt": target_prompt,
                    "num_views": num_views,
                    "resolution": resolution,
                },
            }
            
            logger.info(f"VoxHammer text editing completed: {output_mesh_path}")
            self.status = ModelStatus.LOADED
            return response
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"VoxHammer text editing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"VoxHammer text editing failed: {str(e)}")
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for VoxHammer text editing."""
        return {
            "input": ["glb", "obj"],
            "output": ["glb"]
        }
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Return JSON Schema describing model-specific parameters.
        
        Returns:
            Parameter schema dictionary
        """
        return {
            "parameters": {
                "num_views": {
                    "type": "integer",
                    "description": "Number of rendered views for 3D optimization",
                    "default": 150,
                    "minimum": 50,
                    "maximum": 300,
                    "required": False
                },
                "resolution": {
                    "type": "integer",
                    "description": "Rendering resolution for each view",
                    "default": 512,
                    "enum": [256, 512, 1024],
                    "required": False
                }
            }
        }


class VoxHammerImageMeshEditingAdapter(ImageMeshEditingModel):
    """
    Adapter for VoxHammer image-guided mesh editing.
    
    Performs local mesh editing using reference images to guide the editing region.
    """
    
    FEATURE_TYPE = "image_mesh_editing"
    MODEL_ID = "voxhammer_image_mesh_editing"
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        vram_requirement: int = 40960,  # 40GB VRAM
        voxhammer_root: Optional[str] = None,
        trellis_root: Optional[str] = None,
        feature_type: Optional[str] = None,
        supported_output_formats: Optional[List[str]] = None,
    ):
        if model_id is None:
            model_id = self.MODEL_ID
        if model_path is None:
            model_path = os.path.abspath(
                os.path.join(os.getcwd(), "pretrained", "VoxHammer")
            )
        if voxhammer_root is None:
            voxhammer_root = os.path.abspath(
                os.path.join(os.getcwd(), "thirdparty", "VoxHammer")
            )
        if trellis_root is None:
            trellis_root = os.path.abspath(
                os.path.join(os.getcwd(), "thirdparty", "TRELLIS")
            )
        if feature_type is None:
            feature_type = self.FEATURE_TYPE
        if supported_output_formats is None:
            supported_output_formats = ["glb"]
        
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_output_formats=supported_output_formats,
            feature_type=feature_type,
        )
        
        self.model_path = Path(model_path)
        self.voxhammer_root = Path(voxhammer_root)
        self.trellis_root = Path(trellis_root)
        
        # Pipeline components
        self.trellis_pipeline = None
        self.voxhammer_helper = None
        self.mask_generator = MaskGenerator()
        
        # Utilities
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")
        
        # Add paths to sys.path
        if str(self.voxhammer_root) not in sys.path:
            sys.path.insert(0, str(self.voxhammer_root))
        if str(self.trellis_root) not in sys.path:
            sys.path.insert(0, str(self.trellis_root))
        
        logger.info(f"Initialized VoxHammer image editing adapter with root: {voxhammer_root}")
    
    def _load_model(self):
        """Load TRELLIS pipeline for image-guided editing."""
        try:
            logger.info("Loading TRELLIS image-to-3D pipeline for VoxHammer...")
            
            from trellis.pipelines import TrellisImageTo3DPipeline
            
            # Load TRELLIS image pipeline
            self.trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "fishwowater/TRELLIS-image-large-voxhammer", 
                cache_dir=str(self.model_path / "TRELLIS-image-large-voxhammer")
            )
            self.trellis_pipeline.cuda()
            
            # Create VoxHammer helper
            self.voxhammer_helper = VoxHammerInferenceHelper(
                voxhammer_root=str(self.voxhammer_root),
                trellis_pipeline=self.trellis_pipeline,
                is_text_mode=False,
            )
            
            logger.info("VoxHammer image editing models loaded successfully")
            return {"trellis_pipeline": self.trellis_pipeline}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Failed to load VoxHammer image editing models: {str(e)}")
            raise Exception(f"Failed to load VoxHammer models: {str(e)}")
    
    def _unload_model(self):
        """Unload models."""
        try:
            if self.trellis_pipeline is not None:
                del self.trellis_pipeline
                self.trellis_pipeline = None
            
            if self.voxhammer_helper is not None:
                del self.voxhammer_helper
                self.voxhammer_helper = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("VoxHammer image editing models unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading VoxHammer models: {str(e)}")
    
    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-guided mesh editing request using VoxHammer.
        
        Args:
            inputs: Dictionary with mesh_path, mask params, image paths or target_prompt
        
        Returns:
            Dictionary with editing results
        """
        try:
            # Validate inputs
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required")
            if "mask_type" not in inputs:
                raise ValueError("mask_type is required")
            if "mask_center" not in inputs:
                raise ValueError("mask_center is required")
            if "mask_params" not in inputs:
                raise ValueError("mask_params is required")
            
            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")
            
            # Extract guidance info
            source_image_path = inputs.get("source_image_path")
            target_image_path = inputs.get("target_image_path")
            mask_image_path = inputs.get("mask_image_path")
            target_prompt = inputs.get("target_prompt", "")
            
            if not target_prompt and not (source_image_path and target_image_path and mask_image_path):
                raise ValueError(
                    "Either target_prompt must be provided for automated guidance generation, "
                    "or all three image paths (source_image_path, target_image_path, mask_image_path) must be provided."
                )
            
            # Extract parameters
            output_format = inputs.get("output_format", "glb")
            mask_type = inputs["mask_type"]
            mask_center = inputs["mask_center"]
            mask_params = inputs["mask_params"]
            num_views = inputs.get("num_views", 150)
            resolution = inputs.get("resolution", 512)
            
            logger.info(
                f"Editing mesh with VoxHammer (image-guided): {mesh_path}"
            )
            
            # Create output directory
            base_name = f"{self.model_id}_{mesh_path.stem}"
            output_dir = self.path_generator.base_output_dir / "voxhammer" / f"{mesh_path.stem}_{int(__import__('time').time())}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mask mesh
            logger.info(f"Creating {mask_type} mask at center {mask_center}...")
            mask_glb_path = os.path.join(output_dir, "mask.glb")
            self.mask_generator.create_mask_from_params(
                mask_type=mask_type,
                center=mask_center,
                params=mask_params,
                output_path=mask_glb_path,
            )
            
            # Setup image directory for VoxHammer
            image_dir = os.path.join(output_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            
            # If images provided, copy them
            if source_image_path and target_image_path and mask_image_path:
                source_image_path = Path(source_image_path)
                target_image_path = Path(target_image_path)
                mask_image_path = Path(mask_image_path)
                
                for img_path in [source_image_path, target_image_path, mask_image_path]:
                    if not img_path.exists():
                        raise FileNotFoundError(f"Input image not found: {img_path}")
                
                import shutil
                shutil.copy(source_image_path, os.path.join(image_dir, "2d_render.png"))
                shutil.copy(target_image_path, os.path.join(image_dir, "2d_edit.png"))
                shutil.copy(mask_image_path, os.path.join(image_dir, "2d_mask.png"))
            
            # Generate output path
            output_mesh_path = str(self.path_generator.generate_mesh_path(
                self.model_id, base_name, output_format
            ))
            
            # Run VoxHammer editing pipeline
            logger.info("Running VoxHammer image-guided editing pipeline...")
            self.voxhammer_helper.edit_mesh_with_image(
                input_mesh_path=str(mesh_path),
                mask_glb_path=mask_glb_path,
                output_path=output_mesh_path,
                image_dir=image_dir,
                target_prompt=target_prompt,
                num_views=num_views,
                resolution=resolution,
            )
            
            # Load final mesh for statistics
            final_mesh = self.mesh_processor.load_mesh(output_mesh_path)
            mesh_stats = self.mesh_processor.get_mesh_stats(final_mesh)
            
            # Create response
            response = {
                "output_mesh_path": output_mesh_path,
                "success": True,
                "editing_info": {
                    "model": self.model_id,
                    "input_mesh": str(mesh_path),
                    "output_format": output_format,
                    "vertex_count": mesh_stats["vertex_count"],
                    "face_count": mesh_stats["face_count"],
                    "mask_type": mask_type,
                    "mask_center": mask_center,
                    "mask_params": mask_params,
                    "source_image": str(source_image_path) if source_image_path else "generated",
                    "target_image": str(target_image_path) if target_image_path else "generated",
                    "mask_image": str(mask_image_path) if mask_image_path else "generated",
                    "target_prompt": target_prompt,
                    "num_views": num_views,
                    "resolution": resolution,
                },
            }
            
            logger.info(f"VoxHammer image editing completed: {output_mesh_path}")
            self.status = ModelStatus.LOADED
            return response
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"VoxHammer image editing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"VoxHammer image editing failed: {str(e)}")
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for VoxHammer image editing."""
        return {
            "input": ["glb", "obj"],
            "output": ["glb"]
        }
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Return JSON Schema describing model-specific parameters.
        
        Returns:
            Parameter schema dictionary
        """
        return {
            "parameters": {
                "target_prompt": {
                    "type": "string",
                    "description": "Text prompt for automated guidance image generation (inpainting)",
                    "required": False
                },
                "num_views": {
                    "type": "integer",
                    "description": "Number of rendered views for 3D optimization",
                    "default": 150,
                    "minimum": 50,
                    "maximum": 300,
                    "required": False
                },
                "resolution": {
                    "type": "integer",
                    "description": "Rendering resolution for each view",
                    "default": 512,
                    "enum": [256, 512, 1024],
                    "required": False
                }
            }
        }

