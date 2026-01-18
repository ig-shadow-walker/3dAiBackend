"""
FastMesh model adapter for mesh retopology.

This adapter integrates FastMesh for optimizing mesh topology through
point cloud sampling and neural reconstruction.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from utils.fastmesh_utils import FastMeshRunner
from core.models.base import ModelStatus
from core.models.retopo_models import MeshRetopologyModel
from core.utils.file_utils import OutputPathGenerator
from core.utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class FastMeshRetopologyAdapter(MeshRetopologyModel):
    """
    Adapter for FastMesh retopology.

    Integrates FastMesh for generating optimized, lower-polygon meshes
    from high-resolution input meshes.
    """

    def __init__(
        self,
        model_id: str = "fastmesh_v1k_retopology",
        variant: str = "V1K",
        model_path: Optional[str] = None,
        vram_requirement: int = 8192,  # 8GB VRAM
        fastmesh_root: Optional[str] = None,
        input_pc_num: int = 8192,
    ):
        if model_path is None:
            model_path = f"pretrained/FastMesh-{variant}"

        if fastmesh_root is None:
            fastmesh_root = "thirdparty/FastMesh"

        # Determine target vertex count based on variant
        target_vertex_count = 1000 if variant == "V1K" else 4000

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            target_vertex_count=target_vertex_count,
        )

        self.variant = variant
        self.fastmesh_root = Path(fastmesh_root)
        self.input_pc_num = input_pc_num
        self.fastmesh_runner = None
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

    def _load_model(self):
        """Load FastMesh model."""
        try:
            logger.info(
                f"Loading FastMesh {self.variant} model from {self.fastmesh_root}"
            )

            # Add FastMesh to Python path
            if str(self.fastmesh_root) not in sys.path:
                sys.path.insert(0, str(self.fastmesh_root))

            # Initialize FastMesh runner
            self.fastmesh_runner = FastMeshRunner(
                variant=self.variant,
                model_path=self.model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision="fp16",
                input_pc_num=self.input_pc_num,
                fastmesh_root=str(self.fastmesh_root),
            )

            logger.info("FastMesh model loaded successfully")
            return self.fastmesh_runner

        except Exception as e:
            logger.error(f"Failed to load FastMesh model: {str(e)}")
            raise Exception(f"Failed to load FastMesh model: {str(e)}")

    def _unload_model(self):
        """Unload FastMesh model."""
        try:
            if self.fastmesh_runner is not None:
                self.fastmesh_runner.cleanup()
                self.fastmesh_runner = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("FastMesh model unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading FastMesh model: {str(e)}")

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process mesh retopology request using FastMesh.

        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input mesh (required)
                - output_format: Output format (default: "obj")
                - seed: Random seed for reproducibility (optional)
                - target_vertex_count: Target vertex count (optional, overrides variant default)
                - poly_type: the type of polygon in the input mesh, "tri" or "quad", defaults to "tri"

        Returns:
            Dictionary with retopology results
        """
        try:
            # Validate inputs
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for mesh retopology")

            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

            # Extract parameters
            output_format = inputs.get("output_format", "obj")
            poly_type = inputs.get("poly_type", "tri")
            seed = inputs.get("seed", None)

            if output_format not in ["obj", "glb", "ply"]:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.info(
                f"Generating retopologized mesh with FastMesh {self.variant} from: {mesh_path}"
            )

            # Load original mesh for stats
            original_mesh = self.mesh_processor.load_mesh(mesh_path)
            original_stats = self.mesh_processor.get_mesh_stats(original_mesh)

            # Run FastMesh retopology
            generation_result = self.fastmesh_runner.generate_from_mesh(
                str(mesh_path), seed=seed, poly_type=poly_type
            )

            if generation_result is None or "mesh" not in generation_result:
                raise Exception("FastMesh generation failed to produce results")

            # Generate output path
            base_name = f"{self.model_id}_{mesh_path.stem}"
            output_path = self.path_generator.generate_mesh_path(
                self.model_id, base_name, output_format
            )
            info_path = self.path_generator.generate_info_path(output_path)

            # Save retopologized mesh
            retopo_mesh = generation_result["mesh"]
            if retopo_mesh is not None:
                self.mesh_processor.save_mesh(retopo_mesh, output_path)
                if poly_type == "quad":
                    self.mesh_processor.tri2quad(output_path)
            else:
                raise Exception("No mesh generated")

            # Get mesh statistics
            final_mesh = self.mesh_processor.load_mesh(output_path)
            output_stats = self.mesh_processor.get_mesh_stats(final_mesh)

            # Create generation info
            generation_info = {
                "original_stats": original_stats,
                "output_stats": output_stats,
                "variant": self.variant,
                "input_pc_num": self.input_pc_num,
                "seed": seed,
                "poly_type": poly_type,
                "reduction_ratio": {
                    "vertices": output_stats["vertex_count"]
                    / max(original_stats["vertex_count"], 1),
                    "faces": output_stats["face_count"]
                    / max(original_stats["face_count"], 1),
                },
                "model_info": self.fastmesh_runner.get_model_info(),
            }

            # Save generation info
            self.mesh_processor.export_generation_info(generation_info, info_path)

            # Create response
            response = {
                "output_mesh_path": str(output_path),
                "generation_info_path": str(info_path),
                "original_stats": original_stats,
                "output_stats": output_stats,
                "success": True,
                "retopology_info": {
                    "model": self.model_id,
                    "variant": self.variant,
                    "input_mesh": str(mesh_path),
                    "output_format": output_format,
                    "original_vertices": original_stats["vertex_count"],
                    "original_faces": original_stats["face_count"],
                    "output_vertices": output_stats["vertex_count"],
                    "output_faces": output_stats["face_count"],
                    "vertex_reduction": f"{(1 - generation_info['reduction_ratio']['vertices']) * 100:.1f}%",
                    "face_reduction": f"{(1 - generation_info['reduction_ratio']['faces']) * 100:.1f}%",
                    "seed": seed,
                },
            }

            logger.info(f"FastMesh retopology completed: {output_path}")
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"FastMesh retopology failed: {str(e)}")
            raise Exception(f"FastMesh retopology failed: {str(e)}")

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for FastMesh."""
        return {
            "input": ["obj", "glb", "ply", "stl"],
            "output": ["obj", "glb", "ply"],
        }
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Return JSON Schema describing model-specific parameters.
        
        Returns:
            Parameter schema dictionary
        """
        return {
            "parameters": {
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility",
                    "default": None,
                    "minimum": 0,
                    "required": False
                },
                "poly_type": {
                    "type": "string",
                    "description": "Polygon type for output mesh",
                    "default": "tri",
                    "enum": ["tri", "quad"],
                    "required": False
                }
            }
        }

