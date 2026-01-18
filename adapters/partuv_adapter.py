"""
PartUV model adapter for mesh UV unwrapping.

This adapter integrates PartUV for generating optimized UV coordinates
through part-based unwrapping and packing.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from utils.partuv_utils import PartUVRunner
from core.models.base import ModelStatus
from core.models.uv_models import UVUnwrappingModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class PartUVUnwrappingAdapter(UVUnwrappingModel):
    """
    Adapter for PartUV UV unwrapping.

    Integrates PartUV for generating optimized UV coordinates using
    part-based unwrapping with minimal distortion.
    """

    def __init__(
        self,
        model_id: str = "partuv_uv_unwrapping",
        model_path: Optional[str] = None,
        vram_requirement: int = 6144,  # 6GB VRAM
        partuv_root: Optional[str] = None,
        config_path: Optional[str] = None,
        distortion_threshold: float = 1.25,
    ):
        if model_path is None:
            model_path = "pretrained/PartField/model_objaverse.ckpt"

        if partuv_root is None:
            partuv_root = "thirdparty/PartUV"

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            distortion_threshold=distortion_threshold,
        )

        self.partuv_root = Path(partuv_root)
        self.config_path = config_path
        self.partuv_runner = None
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

    def _load_model(self):
        """Load PartUV model."""
        try:
            logger.info(f"Loading PartUV model from {self.partuv_root}")

            # Add PartUV to Python path
            # if str(self.partuv_root) not in sys.path:
                # sys.path.insert(0, str(self.partuv_root))

            # Initialize PartUV runner
            self.partuv_runner = PartUVRunner(
                config_path=self.config_path,
                partfield_checkpoint=self.model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                partuv_root=str(self.partuv_root),
                distortion_threshold=self.distortion_threshold,
            )

            logger.info("PartUV model loaded successfully")
            return self.partuv_runner

        except Exception as e:
            logger.error(f"Failed to load PartUV model: {str(e)}")
            raise Exception(f"Failed to load PartUV model: {str(e)}")

    def _unload_model(self):
        """Unload PartUV model."""
        try:
            if self.partuv_runner is not None:
                self.partuv_runner.cleanup()
                self.partuv_runner = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("PartUV model unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading PartUV model: {str(e)}")

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process UV unwrapping request using PartUV.

        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input mesh (required)
                - output_format: Output format (default: "obj")
                - distortion_threshold: Maximum distortion threshold (optional)
                - pack_method: UV packing method (optional: 'blender', 'uvpackmaster', 'none')
                - save_individual_parts: Save individual part meshes (default: True)
                - save_visuals: Save visualization images (default: False)
                - hierarchy_path: Pre-computed hierarchy file path (optional)

        Returns:
            Dictionary with UV unwrapping results
        """
        try:
            # Validate inputs
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for UV unwrapping")

            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

            # Extract parameters
            output_format = inputs.get("output_format", "obj")
            distortion_threshold = inputs.get(
                "distortion_threshold", self.distortion_threshold
            )
            pack_method = inputs.get("pack_method", "blender")
            save_individual_parts = inputs.get("save_individual_parts", True)
            save_visuals = inputs.get("save_visuals", False)
            hierarchy_path = inputs.get("hierarchy_path", None)

            if output_format not in ["obj", "glb"]:
                raise ValueError(f"Unsupported output format: {output_format}")

            if pack_method not in ["blender", "uvpackmaster", "none"]:
                raise ValueError(
                    f"Invalid pack method: {pack_method}. Must be 'blender', 'uvpackmaster', or 'none'"
                )

            logger.info(
                f"Generating UV coordinates with PartUV for mesh: {mesh_path}"
            )

            # Load original mesh for stats
            original_mesh = self.mesh_processor.load_mesh(mesh_path)
            original_stats = self.mesh_processor.get_mesh_stats(original_mesh)

            # Generate output path
            base_name = f"{self.model_id}_{mesh_path.stem}"
            output_dir = self.path_generator.base_output_dir / "partuv" / base_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Run PartUV UV unwrapping
            generation_result = self.partuv_runner.generate_uv_from_mesh(
                str(mesh_path),
                output_path=str(output_dir),
                hierarchy_path=hierarchy_path,
                pack_method=pack_method,
                save_visuals=save_visuals,
                save_individual_parts=save_individual_parts,
            )

            if generation_result is None:
                raise Exception("PartUV generation failed to produce results")

            # Get the output mesh path
            output_mesh_path = generation_result.get("output_mesh_path")
            packed_mesh_path = generation_result.get("packed_mesh_path")
            individual_parts_dir = generation_result.get("individual_parts_dir")

            if not output_mesh_path:
                raise Exception("No output mesh path generated")

            # Load and get stats from output mesh
            output_mesh = self.mesh_processor.load_mesh(output_mesh_path)
            output_stats = self.mesh_processor.get_mesh_stats(output_mesh)

            # Create generation info
            metadata = generation_result.get("metadata", {})
            generation_info = {
                "original_stats": original_stats,
                "output_stats": output_stats,
                "num_components": metadata.get("num_components", 0),
                "num_individual_parts": metadata.get("num_individual_parts", 0),
                "distortion": metadata.get("distortion", 0.0),
                "distortion_threshold": distortion_threshold,
                "pack_method": pack_method if pack_method != "none" else None,
                "has_packed_mesh": packed_mesh_path is not None,
                "model_info": self.partuv_runner.get_model_info(),
            }

            # Save generation info
            info_path = self.path_generator.generate_info_path(
                Path(output_mesh_path)
            )
            self.mesh_processor.export_generation_info(generation_info, info_path)

            # Create response
            response = {
                "output_mesh_path": str(output_mesh_path),
                "packed_mesh_path": str(packed_mesh_path) if packed_mesh_path else None,
                "individual_parts_dir": str(individual_parts_dir)
                if individual_parts_dir
                else None,
                "generation_info_path": str(info_path),
                "num_components": metadata.get("num_components", 0),
                "distortion": metadata.get("distortion", 0.0),
                "success": True,
                "uv_info": {
                    "model": self.model_id,
                    "input_mesh": str(mesh_path),
                    "output_format": output_format,
                    "original_vertices": original_stats["vertex_count"],
                    "original_faces": original_stats["face_count"],
                    "num_uv_components": metadata.get("num_components", 0),
                    "num_parts": metadata.get("num_individual_parts", 0),
                    "final_distortion": metadata.get("distortion", 0.0),
                    "distortion_threshold": distortion_threshold,
                    "pack_method": pack_method if pack_method != "none" else None,
                    "components_info": metadata.get("components", []),
                },
            }

            logger.info(f"PartUV UV unwrapping completed: {output_mesh_path}")
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"PartUV UV unwrapping failed: {str(e)}")
            raise Exception(f"PartUV UV unwrapping failed: {str(e)}")

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for PartUV."""
        return {
            "input": ["obj", "glb"],
            "output": ["obj"],  # PartUV primarily outputs OBJ with UV
        }
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Return JSON Schema describing model-specific parameters.
        
        Returns:
            Parameter schema dictionary
        """
        return {
            "parameters": {
                "distortion_threshold": {
                    "type": "number",
                    "description": "Maximum distortion threshold for UV unwrapping",
                    "default": 1.25,
                    "minimum": 1.0,
                    "maximum": 2.0,
                    "required": False
                },
                "pack_method": {
                    "type": "string",
                    "description": "UV packing method to use",
                    "default": "blender",
                    "enum": ["blender", "uvpackmaster", "none"],
                    "required": False
                },
                "save_individual_parts": {
                    "type": "boolean",
                    "description": "Save individual part meshes separately",
                    "default": True,
                    "required": False
                },
                "save_visuals": {
                    "type": "boolean",
                    "description": "Save visualization images of UV layout",
                    "default": False,
                    "required": False
                }
            }
        }

