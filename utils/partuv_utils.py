"""
PartUV utility for part-based UV unwrapping of 3D meshes.

This module provides a clean interface to the PartUV model adapted for our framework.
PartUV performs part-based UV unwrapping, creating optimized UV layouts for texturing.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import trimesh

logger = logging.getLogger(__name__)


class PartUVError(Exception):
    """Custom exception for PartUV-related errors."""

    pass


class PartUVRunner:
    """
    A utility class for PartUV mesh UV unwrapping.

    This class encapsulates the PartUV functionality for generating optimized
    UV coordinates for 3D meshes using part-based unwrapping.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        partfield_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        partuv_root: Optional[str] = None,
        distortion_threshold: float = 1.25,
    ):
        """
        Initialize the PartUV runner.

        Args:
            config_path: Path to PartUV config file
            partfield_checkpoint: Path to PartField model checkpoint
            device: Device to use for inference. If None, automatically selects GPU if available
            partuv_root: Root directory of PartUV code
            distortion_threshold: Maximum allowed distortion for UV unwrapping

        Raises:
            PartUVError: If model loading fails or required files are missing
        """
        if partuv_root is None:
            partuv_root = "thirdparty/PartUV"
        self.partuv_root = Path(partuv_root)

        # Add PartUV to Python path
        # if str(self.partuv_root) not in sys.path:
            # sys.path.insert(0, str(self.partuv_root))

        # Configuration setup
        if config_path is None:
            config_path = str(self.partuv_root / "config" / "config.yaml")
        self.config_path = config_path

        if partfield_checkpoint is None:
            partfield_checkpoint = "pretrained/PartField/model_objaverse.ckpt"
        self.partfield_checkpoint = partfield_checkpoint

        self.distortion_threshold = distortion_threshold

        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        logger.info(f"PartUV using device: {self.device}")

        # Initialize PartField model
        self.pf_model = None
        self._load_partfield()

        logger.info("PartUV runner initialized")

    def _load_partfield(self) -> None:
        """Load the PartField model for part segmentation."""
        try:
            from thirdparty.PartUV.preprocess_utils.partfield_official.run_PF import PFInferenceModel
            logger.info("Loading PartField model for part segmentation")

            # Initialize PartField model
            self.pf_model = PFInferenceModel(device=self.device)

            logger.info("PartField model loaded successfully")

        except Exception as e:
            raise PartUVError(f"Failed to load PartField model: {e}")

    def _preprocess_mesh(
        self,
        mesh_path: Union[str, Path],
        output_path: Optional[Path] = None,
        save_tree_file: bool = True,
        save_processed_mesh: bool = True,
        sample_on_faces: int = 10,
        sample_batch_size: int = 100000,
        merge_vertices_epsilon: float = 1e-7,
    ) -> tuple:
        """
        Preprocess mesh for UV unwrapping.

        Args:
            mesh_path: Path to input mesh
            output_path: Output directory for preprocessed files
            save_tree_file: Whether to save the part hierarchy tree
            save_processed_mesh: Whether to save the processed mesh
            sample_on_faces: Number of samples per face for part assignment
            sample_batch_size: Batch size for sampling
            merge_vertices_epsilon: Epsilon for merging overlapping vertices

        Returns:
            Tuple of (mesh, tree_filename, tree_dict, preprocess_times)
        """
        try:
            from partuv.preprocess import preprocess
            logger.info(f"Preprocessing mesh: {mesh_path}")

            mesh, tree_filename, tree_dict, preprocess_times = preprocess(
                str(mesh_path),
                self.pf_model,
                str(output_path) if output_path else None,
                save_tree_file=save_tree_file,
                save_processed_mesh=save_processed_mesh,
                sample_on_faces=sample_on_faces,
                sample_batch_size=sample_batch_size,
                merge_vertices_epsilon=merge_vertices_epsilon,
            )

            logger.info(
                f"Mesh preprocessing completed: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
            )

            return mesh, tree_filename, tree_dict, preprocess_times

        except Exception as e:
            raise PartUVError(f"Failed to preprocess mesh: {e}")

    def generate_uv_from_mesh(
        self,
        mesh_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        hierarchy_path: Optional[Union[str, Path]] = None,
        pack_method: str = "blender",
        save_visuals: bool = False,
        save_individual_parts: bool = True,
    ) -> Dict:
        """
        Generate UV coordinates for a mesh using PartUV.

        Args:
            mesh_path: Path to input mesh file
            output_path: Output directory for UV unwrapped mesh
            hierarchy_path: Optional pre-computed hierarchy file path
            pack_method: UV packing method ('blender', 'uvpackmaster', or 'none')
            save_visuals: Whether to save visualization images
            save_individual_parts: Whether to save individual part meshes

        Returns:
            Dictionary containing UV unwrapping results and metadata

        Raises:
            PartUVError: If UV generation fails
        """
        try:
            import partuv
            from thirdparty.PartUV.pack.pack import pack_mesh
            from partuv.preprocess import save_results

            # Validate input
            mesh_path = Path(mesh_path)
            if not mesh_path.exists():
                raise PartUVError(f"Mesh file not found: {mesh_path}")

            # Setup output path
            if output_path is None:
                mesh_name = mesh_path.stem
                output_path = Path("outputs") / "partuv" / mesh_name
            else:
                output_path = Path(output_path)

            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Generating UV coordinates for mesh: {mesh_path}")

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Process based on whether hierarchy is provided
            if hierarchy_path is None:
                # Full preprocessing and UV unwrapping
                mesh, tree_filename, tree_dict, preprocess_times = self._preprocess_mesh(
                    mesh_path,
                    output_path,
                    save_tree_file=True,
                    save_processed_mesh=True,
                )

                V = mesh.vertices
                F = mesh.faces

                logger.info(
                    f"Running PartUV pipeline: V={V.shape}, F={F.shape}, threshold={self.distortion_threshold}"
                )

                # Run PartUV pipeline with numpy arrays
                final_parts, individual_parts = partuv.pipeline_numpy(
                    V=V,
                    F=F,
                    tree_dict=tree_dict,
                    configPath=self.config_path,
                    threshold=self.distortion_threshold,
                )

            else:
                # Use provided hierarchy file
                logger.info(f"Using provided hierarchy file: {hierarchy_path}")
                tree_filename = str(hierarchy_path)
                mesh_filename = str(mesh_path)

                final_parts, individual_parts = partuv.pipeline(
                    tree_filename=tree_filename,
                    mesh_filename=mesh_filename,
                    configPath=self.config_path,
                    threshold=self.distortion_threshold,
                )

            # Save results
            logger.info("Saving UV unwrapping results")
            save_results(str(output_path), final_parts, individual_parts)

            # Get output file paths
            final_mesh_path = output_path / "final_components.obj"
            individual_parts_dir = output_path / "individual_parts"

            # Collect metadata
            num_components = final_parts.num_components
            final_distortion = final_parts.distortion
            num_individual_parts = len(individual_parts)

            logger.info(
                f"UV unwrapping completed: {num_components} charts, distortion={final_distortion:.3f}"
            )

            # Optional: Pack UVs
            packed_mesh_path = None
            if pack_method in ["uvpackmaster", "blender"]:
                try:
                    logger.info(f"Packing UVs with {pack_method} method")

                    pack_mesh(
                        str(output_path),
                        uvpackmaster=(pack_method == "uvpackmaster"),
                        save_visuals=save_visuals,
                    )
                    packed_mesh_path = str(output_path / "final_packed.obj")
                except Exception as e:
                    logger.warning(f"UV packing failed: {e}")

            # Prepare output dictionary
            output = {
                # "output_mesh_path": str(final_mesh_path),
                "output_mesh_path": packed_mesh_path, 
                "packed_mesh_path": packed_mesh_path,
                "individual_parts_dir": str(individual_parts_dir)
                if save_individual_parts
                else None,
                "metadata": {
                    "num_components": num_components,
                    "num_individual_parts": num_individual_parts,
                    "distortion": final_distortion,
                    "distortion_threshold": self.distortion_threshold,
                    "pack_method": pack_method if pack_method != "none" else None,
                    "config_path": self.config_path,
                },
            }

            # Get UV coordinates if available
            if num_components > 0:
                uv_coords = final_parts.getUV()
                output["metadata"]["uv_shape"] = uv_coords.shape

                # Add component details
                components_info = []
                for i, component in enumerate(final_parts.components):
                    components_info.append(
                        {
                            "chart_id": i,
                            "num_faces": component.F.shape[0],
                            "distortion": component.distortion,
                        }
                    )
                output["metadata"]["components"] = components_info

            logger.info("PartUV UV generation completed successfully")
            return output

        except Exception as e:
            raise PartUVError(f"Failed to generate UV coordinates with PartUV: {e}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "device": self.device,
            "config_path": self.config_path,
            "partfield_checkpoint": self.partfield_checkpoint,
            "distortion_threshold": self.distortion_threshold,
            "partuv_root": str(self.partuv_root),
        }

    def cleanup(self):
        """Clean up resources."""
        if self.pf_model is not None:
            del self.pf_model
            self.pf_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("PartUV runner cleaned up")

