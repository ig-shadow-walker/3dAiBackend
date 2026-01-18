"""
UltraShape model adapter for high-quality mesh generation.

This adapter integrates UltraShape refinement with Hunyuan3D-2.1 coarse mesh generation,
providing a complete pipeline for generating high-quality 3D meshes from images.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from utils.ultrashape_pipeline_helper import UltraShapeInferenceHelper
from core.models.base import ModelStatus
from core.models.mesh_models import ImageToMeshModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class UltraShapeImageToRawMeshAdapter(ImageToMeshModel):
    """
    Adapter for UltraShape image-to-refined-mesh generation.
    
    This adapter chains Hunyuan3D-2.1 coarse mesh generation with UltraShape
    refinement to produce high-quality geometry.
    """
    
    FEATURE_TYPE = "image_to_raw_mesh"
    MODEL_ID = "ultrashape_image_to_raw_mesh"
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        vram_requirement: int = 20480,  # 20GB VRAM (8GB Hunyuan + 12GB UltraShape)
        ultrashape_root: Optional[str] = None,
        hunyuan3d_root: Optional[str] = None,
        feature_type: Optional[str] = None,
        supported_output_formats: Optional[List[str]] = None,
    ):
        if model_id is None:
            model_id = self.MODEL_ID
        if model_path is None:
            model_path = os.path.abspath(
                os.path.join(os.getcwd(), "thirdparty", "UltraShape")
            )
        if ultrashape_root is None:
            ultrashape_root = model_path
        if hunyuan3d_root is None:
            hunyuan3d_root = os.path.abspath(
                os.path.join(os.getcwd(), "thirdparty", "Hunyuan3D-2.1")
            )
        if feature_type is None:
            feature_type = self.FEATURE_TYPE
        if supported_output_formats is None:
            supported_output_formats = ["glb", "obj", "ply"]
        
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_output_formats=supported_output_formats,
            feature_type=feature_type,
        )
        
        self.ultrashape_root = Path(ultrashape_root)
        self.hunyuan3d_root = Path(hunyuan3d_root)
        
        # Pipeline components
        self.ultrashape_helper = None
        self.hunyuan_pipeline = None
        self.hunyuan_bg_remover = None
        
        # Utilities
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")
        
        # Checkpoint paths
        self.ultrashape_checkpoint = os.path.join(
            os.getcwd(), "pretrained", "UltraShape", "ultrashape_v1.pt"
        )
        self.ultrashape_config = os.path.join(
            ultrashape_root, "configs", "infer_dit_refine.yaml"
        )
        self.hunyuan_model_path = os.path.join(
            os.getcwd(), "pretrained", "tencent", "Hunyuan3D-2.1"
        )
        
        # Add paths to sys.path
        if str(self.ultrashape_root) not in sys.path:
            sys.path.insert(0, str(self.ultrashape_root))
        if str(self.hunyuan3d_root) not in sys.path:
            sys.path.insert(0, str(self.hunyuan3d_root))
        if str(self.hunyuan3d_root / "hy3dshape") not in sys.path:
            sys.path.insert(0, str(self.hunyuan3d_root / "hy3dshape"))
        
        logger.info(f"Initialized UltraShape adapter with root: {ultrashape_root}")
    
    def _load_model(self):
        """Load UltraShape and Hunyuan3D-2.1 pipelines."""
        try:
            logger.info("Loading UltraShape and Hunyuan3D-2.1 models...")
            
            # Apply torchvision fix if available
            try:
                from utils.torchvision_fix import apply_fix
                apply_fix()
            except ImportError:
                logger.warning("torchvision_fix module not found, proceeding without fix")
            except Exception as e:
                logger.warning(f"Failed to apply torchvision fix: {e}")
            
            loaded_models = {}
            
            # Load Hunyuan3D-2.1 for coarse mesh generation
            logger.info("Loading Hunyuan3D-2.1 shape generation pipeline...")
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dshape.rembg import BackgroundRemover
            
            self.hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                str(self.hunyuan_model_path)
            )
            self.hunyuan_bg_remover = BackgroundRemover()
            loaded_models["hunyuan_pipeline"] = self.hunyuan_pipeline
            loaded_models["hunyuan_bg_remover"] = self.hunyuan_bg_remover
            
            # Load UltraShape refinement pipeline
            logger.info("Loading UltraShape refinement pipeline...")
            self.ultrashape_helper = UltraShapeInferenceHelper(
                ultrashape_root=str(self.ultrashape_root),
                config_path=self.ultrashape_config,
                checkpoint_path=self.ultrashape_checkpoint,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16,
            )
            self.ultrashape_helper.load_models()
            loaded_models["ultrashape_helper"] = self.ultrashape_helper
            
            logger.info("UltraShape models loaded successfully")
            return loaded_models
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Failed to load UltraShape models: {str(e)}")
            raise Exception(f"Failed to load UltraShape models: {str(e)}")
    
    def _unload_model(self):
        """Unload UltraShape and Hunyuan3D models."""
        try:
            if self.ultrashape_helper is not None:
                self.ultrashape_helper.unload_models()
                self.ultrashape_helper = None
            
            if self.hunyuan_pipeline is not None:
                del self.hunyuan_pipeline
                self.hunyuan_pipeline = None
            
            if self.hunyuan_bg_remover is not None:
                del self.hunyuan_bg_remover
                self.hunyuan_bg_remover = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("UltraShape models unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading UltraShape models: {str(e)}")
    
    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-to-refined-mesh generation using UltraShape.
        
        Args:
            inputs: Dictionary containing:
                - image_path: Path to input image (required)
                - output_format: Output format (default: "glb")
                - num_inference_steps: Diffusion steps (default: 50)
                - num_latents: Number of latent tokens (default: 32768)
                - octree_res: Marching cubes resolution (default: 1024)
                - chunk_size: Inference chunk size (default: 8000)
                - scale: Mesh normalization scale (default: 0.99)
                - seed: Random seed (default: 42)
        
        Returns:
            Dictionary with generation results
        """
        try:
            # Validate inputs
            if "image_path" not in inputs:
                raise ValueError("image_path is required for image-to-mesh generation")
            
            image_path = Path(inputs["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Input image file not found: {image_path}")
            
            # Extract parameters
            output_format = inputs.get("output_format", "glb")
            if output_format not in self.supported_output_formats:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            num_inference_steps = inputs.get("num_inference_steps", 50)
            num_latents = inputs.get("num_latents", 16384)
            octree_res = inputs.get("octree_res", 512)
            chunk_size = inputs.get("chunk_size", 8000)
            scale = inputs.get("scale", 0.99)
            seed = inputs.get("seed", 42)
            
            logger.info(
                f"Generating refined mesh with UltraShape from image: {image_path}"
            )
            
            # Create output directory
            base_name = f"{self.model_id}_{image_path.stem}"
            output_dir = self.path_generator.base_output_dir / "ultrashape" / f"{image_path.stem}_{int(__import__('time').time())}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run complete pipeline: Hunyuan3D coarse + UltraShape refinement
            logger.info("Running integrated Hunyuan3D + UltraShape pipeline...")
            refined_mesh_path, generation_info = self.ultrashape_helper.generate_and_refine(
                image_path=str(image_path),
                hunyuan_pipeline=self.hunyuan_pipeline,
                hunyuan_bg_remover=self.hunyuan_bg_remover,
                output_dir=str(output_dir),
                num_inference_steps=num_inference_steps,
                num_latents=num_latents,
                octree_res=octree_res,
                chunk_size=chunk_size,
                scale=scale,
                seed=seed,
            )
            
            # Generate final output path with correct format
            final_output_path = self.path_generator.generate_mesh_path(
                self.model_id, base_name, output_format
            )
            
            # Convert format if needed
            if not refined_mesh_path.endswith(output_format):
                logger.info(f"Converting mesh to {output_format} format...")
                mesh = self.mesh_processor.load_mesh(refined_mesh_path)
                self.mesh_processor.save_mesh(mesh, final_output_path)
            else:
                # Just move to final location
                import shutil
                shutil.move(refined_mesh_path, final_output_path)
            
            # Load final mesh for statistics
            final_mesh = self.mesh_processor.load_mesh(final_output_path)
            mesh_stats = self.mesh_processor.get_mesh_stats(final_mesh)
            
            # Create response
            response = {
                "output_mesh_path": str(final_output_path),
                "success": True,
                "generation_info": {
                    "model": self.model_id,
                    "input_image": str(image_path),
                    "output_format": output_format,
                    "vertex_count": mesh_stats["vertex_count"],
                    "face_count": mesh_stats["face_count"],
                    "has_texture": False,
                    "refinement_steps": num_inference_steps,
                    "num_latents": num_latents,
                    "octree_resolution": octree_res,
                    "coarse_mesh_path": generation_info.get("coarse_mesh_path"),
                },
            }
            
            logger.info(f"UltraShape mesh generation completed: {final_output_path}")
            self.status = ModelStatus.LOADED
            return response
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"UltraShape mesh generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"UltraShape mesh generation failed: {str(e)}")
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for UltraShape."""
        return {
            "input": ["png", "jpg", "jpeg"],
            "output": ["glb", "obj", "ply"]
        }
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Return JSON Schema describing model-specific parameters.
        
        Returns:
            Parameter schema dictionary
        """
        return {
            "parameters": {
                "num_inference_steps": {
                    "type": "integer",
                    "description": "Number of diffusion inference steps for refinement",
                    "default": 50,
                    "minimum": 10,
                    "maximum": 200,
                    "required": False
                },
                "num_latents": {
                    "type": "integer",
                    "description": "Number of latent tokens for shape representation",
                    "default": 16384,# 32768,
                    "minimum": 8192,
                    "maximum": 65536,
                    "required": False
                },
                "octree_res": {
                    "type": "integer",
                    "description": "Octree resolution for marching cubes extraction",
                    "default": 512,
                    "enum": [512, 1024, 2048],
                    "required": False
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Chunk size for inference to manage memory usage",
                    "default": 8000,
                    "minimum": 1000,
                    "maximum": 16000,
                    "required": False
                },
                "scale": {
                    "type": "number",
                    "description": "Mesh normalization scale factor",
                    "default": 0.99,
                    "minimum": 0.5,
                    "maximum": 1.0,
                    "required": False
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility",
                    "default": 42,
                    "minimum": 0,
                    "required": False
                }
            }
        }

