"""
FastMesh utility for mesh retopology from point clouds.

This module provides a clean interface to the FastMesh model adapted for our framework.
FastMesh performs mesh retopology by converting high-resolution meshes to optimized,
low-polygon versions suitable for real-time rendering and game assets.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
import trimesh
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

logger = logging.getLogger(__name__)


class FastMeshError(Exception):
    """Custom exception for FastMesh-related errors."""

    pass


class FastMeshRunner:
    """
    A utility class for FastMesh retopology.

    This class encapsulates the FastMesh functionality for generating optimized
    meshes from input meshes through point cloud sampling and reconstruction.
    """

    def __init__(
        self,
        variant: str = "V1K",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        precision: str = "fp16",
        input_pc_num: int = 8192,
        fastmesh_root: Optional[str] = None,
    ):
        """
        Initialize the FastMesh runner.

        Args:
            variant: Model variant to use ('V1K' for 1000 vertices, 'V4K' for 4000 vertices)
            model_path: Path to the model checkpoint directory
            device: Device to use for inference. If None, automatically selects GPU if available
            precision: Precision for inference ('fp16', 'fp32', 'bf16')
            input_pc_num: Number of points to sample from input mesh
            fastmesh_root: Root directory of FastMesh code

        Raises:
            FastMeshError: If model loading fails or required files are missing
        """
        self.variant = variant
        self.precision = precision
        self.input_pc_num = input_pc_num

        if variant not in ["V1K", "V4K"]:
            raise FastMeshError(f"Invalid variant: {variant}. Must be 'V1K' or 'V4K'")

        if fastmesh_root is None:
            fastmesh_root = "thirdparty/FastMesh"
        self.fastmesh_root = Path(fastmesh_root)

        # Add FastMesh to Python path
        if str(self.fastmesh_root) not in sys.path:
            sys.path.insert(0, str(self.fastmesh_root))

        # Model path setup
        if model_path is None:
            model_path = f"pretrained/FastMesh-{variant}"
        self.model_path = Path(model_path)

        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        logger.info(f"FastMesh using device: {self.device}, variant: {variant}")

        # Model will be loaded when needed
        self.model = None
        self.accelerator = None

        self._load_model()
        logger.info("FastMesh runner initialized")

    def _load_model(self) -> None:
        """Load the FastMesh model and weights."""
        if self.model is not None:
            return  # Already loaded

        try:
            logger.info(f"Loading FastMesh {self.variant} model from {self.model_path}")

            # Import FastMesh modules
            from thirdparty.FastMesh.models import MODELS

            # Setup accelerator
            kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
            self.accelerator = Accelerator(
                mixed_precision=self.precision, kwargs_handlers=[kwargs]
            )

            # Load model from pretrained
            model_name = "MeshGen"  # Default model name in FastMesh
            if self.variant == "V1K":
                self.model = MODELS[model_name].from_pretrained(
                    "WopperSet/FastMesh-V1K",
                    local_dir=str(self.model_path),
                    # cache_dir=str(self.model_path),
                )
            elif self.variant == "V4K":
                self.model = MODELS[model_name].from_pretrained(
                    "WopperSet/FastMesh-V4K",
                    local_dir=str(self.model_path),
                    # cache_dir=str(self.model_path),
                )

            # Prepare model with accelerator
            self.model = self.accelerator.prepare(self.model)
            self.model.eval()

            logger.info("FastMesh model loaded successfully")

        except Exception as e:
            raise FastMeshError(f"Failed to load FastMesh model: {e}")

    def _apply_normalize(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Normalize mesh to [-1, 1] coordinate space.

        Args:
            mesh: Input trimesh mesh

        Returns:
            Normalized trimesh mesh
        """
        bbox = mesh.bounds
        center = (bbox[1] + bbox[0]) / 2
        scale = (bbox[1] - bbox[0]).max()

        mesh.apply_translation(-center)
        mesh.apply_scale(1 / scale * 2 * 0.95)

        return mesh

    def _sample_pc_with_normal(
        self, mesh: trimesh.Trimesh, pc_num: int
    ) -> np.ndarray:
        """
        Sample point cloud with normals from mesh.

        Args:
            mesh: Input trimesh mesh
            pc_num: Number of points to sample

        Returns:
            Point cloud array of shape (pc_num, 6) containing [x, y, z, nx, ny, nz]
        """
        mesh = self._apply_normalize(mesh)

        # Sample points with face indices
        points, face_idx = mesh.sample(50000, return_index=True)
        normals = mesh.face_normals[face_idx]

        # Concatenate points and normals
        pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)

        # Random sample to target number
        if pc_normal.shape[0] > pc_num:
            ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
            pc_normal = pc_normal[ind]

        return pc_normal

    @torch.inference_mode()
    def generate_from_mesh(
        self,
        mesh_path: Union[str, Path, trimesh.Trimesh],
        seed: Optional[int] = None,
        poly_type: Optional[Literal["tri", "quad"]] = "tri",
        batch_size: int = 1,
    ) -> Dict:
        """
        Generate a retopologized mesh from an input mesh.

        Args:
            mesh_path: Path to input mesh file or trimesh.Trimesh object
            seed: Random seed for reproducibility
            batch_size: Batch size for inference (usually 1)

        Returns:
            Dictionary containing generated mesh and metadata

        Raises:
            FastMeshError: If generation fails
        """
        try:
            # Ensure model is loaded
            self._load_model()

            # Set seed if provided
            if seed is not None:
                set_seed(seed, device_specific=True)

            # Load mesh
            if isinstance(mesh_path, (str, Path)):
                if not os.path.exists(mesh_path):
                    raise FastMeshError(f"Mesh file not found: {mesh_path}")
                input_mesh = trimesh.load(str(mesh_path), force="mesh")
            elif isinstance(mesh_path, trimesh.Trimesh):
                input_mesh = mesh_path
            else:
                raise FastMeshError("Invalid mesh input type")

            logger.info(
                f"Generating retopologized mesh with FastMesh {self.variant}..."
            )

            # Normalize and sample point cloud
            pc_normal = self._sample_pc_with_normal(input_mesh, self.input_pc_num)

            # Convert to tensor
            pc_normal_tensor = (
                torch.from_numpy(pc_normal).unsqueeze(0).to(self.device)
            )

            # Prepare model input
            input_dict = {"pc_normal": pc_normal_tensor}

            # Run model inference
            with self.accelerator.autocast():
                recon_meshes = self.model(input_dict, is_eval=True)

            # Process output mesh
            if not recon_meshes or len(recon_meshes) == 0:
                raise FastMeshError("Model failed to generate mesh")

            output_mesh = recon_meshes[0]

            # Get mesh statistics
            num_vertices = len(output_mesh.vertices)
            num_faces = len(output_mesh.faces)

            # Create output dictionary
            output = {
                "mesh": output_mesh,
                "metadata": {
                    "variant": self.variant,
                    "num_vertices": num_vertices,
                    "num_faces": num_faces,
                    "input_pc_num": self.input_pc_num,
                    "seed": seed,
                    "input_mesh_stats": {
                        "vertices": len(input_mesh.vertices),
                        "faces": len(input_mesh.faces),
                    },
                },
            }

            logger.info(
                f"FastMesh retopology completed: {num_vertices} vertices, {num_faces} faces"
            )
            return output

        except Exception as e:
            raise FastMeshError(f"Failed to generate mesh with FastMesh: {e}")

    def batch_generate_from_meshes(
        self,
        mesh_paths: List[Union[str, Path]],
        seed: Optional[int] = None,
        batch_size: int = 2,
    ) -> List[Dict]:
        """
        Generate retopologized meshes from multiple input meshes in batches.

        Args:
            mesh_paths: List of paths to input mesh files
            seed: Random seed for reproducibility
            batch_size: Batch size for inference

        Returns:
            List of dictionaries containing generated meshes and metadata

        Raises:
            FastMeshError: If generation fails
        """
        try:
            # Ensure model is loaded
            self._load_model()

            # Set seed if provided
            if seed is not None:
                set_seed(seed, device_specific=True)

            results = []
            current_batch = []
            current_paths = []

            logger.info(
                f"Processing {len(mesh_paths)} meshes in batches of {batch_size}"
            )

            for mesh_path in mesh_paths:
                # Load and process mesh
                if not os.path.exists(mesh_path):
                    logger.warning(f"Mesh file not found: {mesh_path}, skipping")
                    continue

                input_mesh = trimesh.load(str(mesh_path), force="mesh")
                pc_normal = self._sample_pc_with_normal(input_mesh, self.input_pc_num)
                pc_normal_tensor = torch.from_numpy(pc_normal).unsqueeze(0)

                current_batch.append(pc_normal_tensor)
                current_paths.append((mesh_path, input_mesh))

                # Process batch when full
                if len(current_batch) >= batch_size:
                    batch_tensor = torch.cat(current_batch).to(self.device)
                    input_dict = {"pc_normal": batch_tensor}

                    with self.accelerator.autocast():
                        recon_meshes = self.model(input_dict, is_eval=True)

                    # Process results
                    for (path, orig_mesh), recon_mesh in zip(
                        current_paths, recon_meshes
                    ):
                        results.append(
                            {
                                "input_path": str(path),
                                "mesh": recon_mesh,
                                "metadata": {
                                    "variant": self.variant,
                                    "num_vertices": len(recon_mesh.vertices),
                                    "num_faces": len(recon_mesh.faces),
                                    "input_mesh_stats": {
                                        "vertices": len(orig_mesh.vertices),
                                        "faces": len(orig_mesh.faces),
                                    },
                                },
                            }
                        )

                    current_batch = []
                    current_paths = []

            # Process remaining meshes
            if current_batch:
                batch_tensor = torch.cat(current_batch).to(self.device)
                input_dict = {"pc_normal": batch_tensor}

                with self.accelerator.autocast():
                    recon_meshes = self.model(input_dict, is_eval=True)

                for (path, orig_mesh), recon_mesh in zip(current_paths, recon_meshes):
                    results.append(
                        {
                            "input_path": str(path),
                            "mesh": recon_mesh,
                            "metadata": {
                                "variant": self.variant,
                                "num_vertices": len(recon_mesh.vertices),
                                "num_faces": len(recon_mesh.faces),
                                "input_mesh_stats": {
                                    "vertices": len(orig_mesh.vertices),
                                    "faces": len(orig_mesh.faces),
                                },
                            },
                        }
                    )

            logger.info(f"Batch retopology completed: {len(results)} meshes processed")
            return results

        except Exception as e:
            raise FastMeshError(f"Failed to batch generate meshes with FastMesh: {e}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "variant": self.variant,
            "device": self.device,
            "precision": self.precision,
            "input_pc_num": self.input_pc_num,
            "model_path": str(self.model_path),
            "target_vertices": 1000 if self.variant == "V1K" else 4000,
        }

    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.accelerator is not None:
            del self.accelerator
            self.accelerator = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("FastMesh runner cleaned up")

