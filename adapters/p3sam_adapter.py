"""
P3-SAM model adapter for automatic mesh segmentation.

This adapter integrates P3-SAM for semantic part segmentation of 3D meshes.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from utils.p3sam_utils import P3SAMRunner, create_segmented_parts_scene
from core.models.base import ModelStatus
from core.models.segment_models import MeshSegmentationModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class P3SAMSegmentationAdapter(MeshSegmentationModel):
    """
    Adapter for P3-SAM mesh segmentation model.
    
    Integrates P3-SAM for automatic semantic part segmentation with
    axis-aligned bounding box (AABB) detection.
    """
    
    FEATURE_TYPE = "mesh_segmentation"
    MODEL_ID = "p3sam_mesh_segmentation"
    
    def __init__(
        self,
        model_id: str = "p3sam_mesh_segmentation",
        model_path: Optional[str] = None,
        vram_requirement: int = 6144,  # 6GB VRAM
        p3sam_root: Optional[str] = None,
    ):
        if model_path is None:
            model_path = "pretrained/P3-SAM/p3sam.safetensors"
        
        if p3sam_root is None:
            p3sam_root = "thirdparty/Hunyuan3DPart/P3SAM"
        
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
        )
        
        self.p3sam_root = Path(p3sam_root)
        self.p3sam_runner: Optional[P3SAMRunner] = None
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")
        
        # Model checkpoint path
        self.checkpoint_path = model_path
    
    def _load_model(self):
        """Load P3-SAM segmentation model."""
        try:
            logger.info(f"Loading P3-SAM model from {self.checkpoint_path}")
            
            # Add P3-SAM to Python path
            if str(self.p3sam_root) not in sys.path:
                sys.path.insert(0, str(self.p3sam_root))
            
            # Initialize P3-SAM runner
            self.p3sam_runner = P3SAMRunner(
                checkpoint_path=self.checkpoint_path,
                p3sam_root=str(self.p3sam_root),
                device="cuda"
            )
            
            # Pre-load the model
            self.p3sam_runner._load_model()
            
            logger.info("P3-SAM model loaded successfully")
            return self.p3sam_runner
        
        except Exception as e:
            logger.error(f"Failed to load P3-SAM model: {str(e)}")
            raise Exception(f"Failed to load P3-SAM model: {str(e)}")
    
    def _unload_model(self):
        """Unload P3-SAM model."""
        try:
            if self.p3sam_runner is not None:
                self.p3sam_runner.cleanup()
                self.p3sam_runner = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("P3-SAM model unloaded successfully")
        
        except Exception as e:
            logger.error(f"Error unloading P3-SAM model: {str(e)}")
    
    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process mesh segmentation request using P3-SAM.
        
        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input mesh (required)
                - point_num: Number of points to sample (default: 100000)
                - prompt_num: Number of prompt points (default: 400)
                - threshold: Post-processing threshold (default: 0.95)
                - post_process: Enable post-processing (default: True)
                - seed: Random seed (default: 42)
                - prompt_bs: Prompt batch size (default: 32)
                - save_mid_res: Save intermediate results (default: False)
                - output_format: Output format (default: "glb")
        
        Returns:
            Dictionary with segmentation results including AABB
        """
        try:
            if self.p3sam_runner is None:
                raise RuntimeError("P3-SAM runner is not loaded")
            
            # Validate inputs
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for mesh segmentation")
            
            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")
            
            # Extract parameters
            point_num = inputs.get("point_num", 100000)
            prompt_num = inputs.get("prompt_num", 400)
            threshold = inputs.get("threshold", 0.95)
            post_process = inputs.get("post_process", True)
            seed = inputs.get("seed", 42)
            prompt_bs = inputs.get("prompt_bs", 32)
            save_mid_res = inputs.get("save_mid_res", False)
            output_format = inputs.get("output_format", "glb")
            
            logger.info(f"Segmenting mesh with P3-SAM: {mesh_path}")
            
            # Load and validate mesh
            mesh = self.mesh_processor.load_mesh(mesh_path)
            
            # Setup temporary directory for P3-SAM
            base_name = mesh_path.stem
            temp_base = (
                self.path_generator.base_output_dir
                / "temp"
                / f"p3sam_{int(time.time())}"
            )
            temp_base.mkdir(parents=True, exist_ok=True)
            
            # Run P3-SAM segmentation
            aabb, face_ids, processed_mesh = self.p3sam_runner.segment_mesh(
                mesh_path=str(mesh_path),
                point_num=point_num,
                prompt_num=prompt_num,
                threshold=threshold,
                post_process=post_process,
                save_path=str(temp_base) if save_mid_res else None,
                save_mid_res=save_mid_res,
                show_info=False,
                clean_mesh_flag=True,
                seed=seed,
                prompt_bs=prompt_bs
            )
            
            if aabb is None or face_ids is None:
                raise Exception("P3-SAM segmentation failed to produce results")
            
            # Determine number of parts
            num_parts = len(np.unique(face_ids[face_ids >= 0]))
            
            # Generate output paths
            output_path = self.path_generator.generate_segmentation_path(
                self.model_id, base_name, output_format
            )
            info_path = self.path_generator.generate_info_path(output_path)
            
            # Create scene where each segmented part is a geometry
            scene = create_segmented_parts_scene(processed_mesh, face_ids)
            
            # Save segmented mesh scene
            self.mesh_processor.save_scene(scene, output_path, do_normalise=True)
            
            # Compute part statistics
            part_statistics = self._compute_part_statistics(face_ids, num_parts)
            
            # Create segmentation info
            segmentation_info = {
                "num_parts": num_parts,
                "point_num": point_num,
                "prompt_num": prompt_num,
                "threshold": threshold,
                "post_process": post_process,
                "seed": seed,
                "mesh_stats": self.mesh_processor.get_mesh_stats(mesh),
                "part_statistics": part_statistics,
                "processing_parameters": {
                    "prompt_bs": prompt_bs,
                    "save_mid_res": save_mid_res,
                    "model_id": self.model_id,
                },
            }
            
            # Save segmentation info
            self.mesh_processor.export_segmentation_info(segmentation_info, info_path)
            
            # Cleanup temporary files if not saving intermediate results
            if not save_mid_res:
                self._cleanup_temp_files(temp_base)
            
            # Create response
            response = {
                "output_mesh_path": str(output_path),
                "segmentation_info_path": str(info_path),
                "num_parts": num_parts,
                "segmentation_info": segmentation_info,
                "success": True,
                "generation_info": {
                    "model": self.model_id,
                    "input_mesh": str(mesh_path),
                    "vertex_count": len(mesh.vertices),
                    "face_count": len(mesh.faces),
                    "segmentation_method": "P3-SAM",
                },
            }
            
            logger.info(f"P3-SAM segmentation completed: {output_path}")
            self.status = ModelStatus.LOADED
            return response
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status = ModelStatus.ERROR
            logger.error(f"P3-SAM segmentation failed: {str(e)}")
            raise Exception(f"P3-SAM segmentation failed: {str(e)}")
    
    def _compute_part_statistics(
        self, face_ids: np.ndarray, num_parts: int
    ) -> Dict[str, Any]:
        """Compute statistics for segmented parts."""
        unique_ids, counts = np.unique(face_ids, return_counts=True)
        
        # Filter out invalid IDs (< 0)
        valid_mask = unique_ids >= 0
        unique_ids = unique_ids[valid_mask]
        counts = counts[valid_mask]
        
        return {
            "num_parts_actual": len(unique_ids),
            "num_parts_requested": num_parts,
            "part_sizes": {
                int(label): int(count) for label, count in zip(unique_ids, counts)
            },
            "average_part_size": float(np.mean(counts)) if len(counts) > 0 else 0,
            "part_size_std": float(np.std(counts)) if len(counts) > 0 else 0,
        }
    
    def _cleanup_temp_files(self, temp_dir: Path):
        """Clean up temporary files created during processing."""
        try:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {str(e)}")
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for P3-SAM."""
        return {"input": ["glb", "obj", "ply"], "output": ["glb"]}
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Return JSON Schema describing model-specific parameters.
        
        Returns:
            Parameter schema dictionary
        """
        return {
            "parameters": {
                "point_num": {
                    "type": "integer",
                    "description": "Number of points to sample from the mesh surface",
                    "default": 100000,
                    "minimum": 10000,
                    "maximum": 500000,
                    "required": False
                },
                "prompt_num": {
                    "type": "integer",
                    "description": "Number of prompt points for segmentation",
                    "default": 400,
                    "minimum": 50,
                    "maximum": 1000,
                    "required": False
                },
                "threshold": {
                    "type": "number",
                    "description": "Post-processing threshold for merging small parts",
                    "default": 0.95,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "required": False
                },
                "post_process": {
                    "type": "boolean",
                    "description": "Whether to apply post-processing to refine segmentation",
                    "default": True,
                    "required": False
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility",
                    "default": 42,
                    "minimum": 0,
                    "required": False
                },
                "prompt_bs": {
                    "type": "integer",
                    "description": "Batch size for processing prompt points",
                    "default": 32,
                    "minimum": 1,
                    "maximum": 128,
                    "required": False
                },
                "save_mid_res": {
                    "type": "boolean",
                    "description": "Save intermediate results for debugging",
                    "default": False,
                    "required": False
                }
            }
        }

