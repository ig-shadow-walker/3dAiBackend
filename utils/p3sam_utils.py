"""
P3-SAM utility wrapper for automatic mesh segmentation.

This module provides a high-level interface to P3-SAM (Part-aware Point-cloud Segmentation and Annotation Model),
wrapping the inference logic from thirdparty/Hunyuan3D-Part/P3-SAM/demo/auto_mask.py
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import trimesh

logger = logging.getLogger(__name__)


class P3SAMRunner:
    """
    Wrapper for P3-SAM automatic mesh segmentation.
    
    P3-SAM performs automatic semantic part segmentation on 3D meshes,
    returning axis-aligned bounding boxes (AABB) and face-level part IDs.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        p3sam_root: Optional[str] = None,
        point_num: int = 100000,
        prompt_num: int = 400,
        threshold: float = 0.95,
        post_process: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize P3-SAM runner.
        
        Args:
            checkpoint_path: Path to P3-SAM model checkpoint (.ckpt file)
            p3sam_root: Path to P3-SAM directory
            point_num: Number of points to sample from mesh
            prompt_num: Number of prompt points for segmentation
            threshold: Post-processing threshold
            post_process: Whether to apply post-processing
            device: Device to run on ('cuda' or 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.point_num = point_num
        self.prompt_num = prompt_num
        self.threshold = threshold
        self.post_process = post_process
        
        if p3sam_root is None:
            p3sam_root = os.path.join(
                os.getcwd(), "thirdparty", "Hunyuan3DPart", "P3SAM"
            )
        
        self.p3sam_root = Path(p3sam_root)
        
        # Add P3-SAM to Python path
        if str(self.p3sam_root) not in sys.path:
            sys.path.insert(0, str(self.p3sam_root))
        
        # Add demo directory to path
        demo_path = str(self.p3sam_root / "demo")
        if demo_path not in sys.path:
            sys.path.insert(0, demo_path)
        
        # Model (lazy loaded)
        self.auto_mask_model = None
    
    def _load_model(self):
        """Load P3-SAM model."""
        if self.auto_mask_model is not None:
            return
        
        try:
            logger.info(f"Loading P3-SAM model from {self.checkpoint_path}")
            
            # Import AutoMask class from P3-SAM
            from thirdparty.Hunyuan3DPart.P3SAM.demo.auto_mask import AutoMask
            
            # Initialize model
            self.auto_mask_model = AutoMask(
                ckpt_path=self.checkpoint_path,
                point_num=self.point_num,
                prompt_num=self.prompt_num,
                threshold=self.threshold,
                post_process=self.post_process
            )
            
            logger.info("P3-SAM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load P3-SAM model: {e}")
            raise
    
    def segment_mesh(
        self,
        mesh_path: str,
        point_num: Optional[int] = None,
        prompt_num: Optional[int] = None,
        threshold: Optional[float] = None,
        post_process: Optional[bool] = None,
        save_path: Optional[str] = None,
        save_mid_res: bool = False,
        show_info: bool = False,
        clean_mesh_flag: bool = True,
        seed: int = 42,
        prompt_bs: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, trimesh.Trimesh]:
        """
        Segment a 3D mesh into semantic parts.
        
        Args:
            mesh_path: Path to input mesh file
            point_num: Number of points to sample (overrides init value)
            prompt_num: Number of prompt points (overrides init value)
            threshold: Post-processing threshold (overrides init value)
            post_process: Whether to post-process (overrides init value)
            save_path: Optional path to save intermediate results
            save_mid_res: Whether to save intermediate results
            show_info: Whether to show processing info
            clean_mesh_flag: Whether to clean mesh before processing
            seed: Random seed for reproducibility
            prompt_bs: Batch size for prompt processing
        
        Returns:
            Tuple of (aabb, face_ids, mesh):
                - aabb: Axis-aligned bounding boxes for each part [N, 2, 3]
                - face_ids: Part ID for each face in the mesh [num_faces]
                - mesh: Processed trimesh object
        """
        self._load_model()
        
        # Override parameters if provided
        point_num = point_num if point_num is not None else self.point_num
        prompt_num = prompt_num if prompt_num is not None else self.prompt_num
        threshold = threshold if threshold is not None else self.threshold
        post_process = post_process if post_process is not None else self.post_process
        
        # Load mesh
        logger.info(f"Loading mesh from {mesh_path}")
        mesh = trimesh.load(mesh_path, force="mesh")
        
        # Create temporary save path if not provided
        if save_path is None:
            import tempfile
            save_path = tempfile.mkdtemp(prefix="p3sam_")
        
        # Run segmentation
        logger.info(f"Segmenting mesh with P3-SAM (points={point_num}, prompts={prompt_num})")
        
        try:
            aabb, face_ids, mesh = self.auto_mask_model.predict_aabb(
                mesh=mesh,
                point_num=point_num,
                prompt_num=prompt_num,
                threshold=threshold,
                post_process=post_process,
                save_path=save_path,
                save_mid_res=save_mid_res,
                show_info=show_info,
                clean_mesh_flag=clean_mesh_flag,
                seed=seed,
                is_parallel=True,  # Use DataParallel for faster inference
                prompt_bs=prompt_bs
            )
            
            logger.info(f"Segmentation completed: {len(np.unique(face_ids))} parts found")
            
            return aabb, face_ids, mesh
            
        except Exception as e:
            logger.error(f"P3-SAM segmentation failed: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources and free GPU memory."""
        if self.auto_mask_model is not None:
            # The AutoMask model has internal cleanup
            del self.auto_mask_model
            self.auto_mask_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("P3-SAM runner cleaned up")


def create_colored_segmented_mesh(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    color_seed: int = 42
) -> trimesh.Trimesh:
    """
    Create a colored mesh based on face IDs.
    
    Args:
        mesh: Input trimesh object
        face_ids: Part ID for each face [num_faces]
        color_seed: Random seed for color generation
    
    Returns:
        Trimesh with colored faces
    """
    # Set random seed for consistent colors
    np.random.seed(color_seed)
    
    # Get unique part IDs
    unique_ids = np.unique(face_ids)
    
    # Generate random colors for each part
    color_map = {}
    for part_id in unique_ids:
        if part_id < 0:  # Invalid parts
            color_map[part_id] = np.array([128, 128, 128, 255], dtype=np.uint8)  # Gray
        else:
            color_map[part_id] = np.concatenate([
                np.random.randint(50, 255, size=3),
                [255]
            ]).astype(np.uint8)
    
    # Assign colors to faces
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    for i, face_id in enumerate(face_ids):
        face_colors[i] = color_map[face_id]
    
    # Create colored mesh
    colored_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        face_colors=face_colors
    )
    
    return colored_mesh


def create_aabb_scene(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    aabb: np.ndarray,
    color_seed: int = 42
) -> trimesh.Scene:
    """
    Create a scene with colored mesh and bounding boxes.
    
    Args:
        mesh: Input trimesh object
        face_ids: Part ID for each face
        aabb: Axis-aligned bounding boxes [N, 2, 3]
        color_seed: Random seed for color generation
    
    Returns:
        Trimesh Scene with mesh and bounding boxes
    """
    # Create colored mesh
    colored_mesh = create_colored_segmented_mesh(mesh, face_ids, color_seed)
    
    # Create scene
    scene = trimesh.Scene()
    scene.add_geometry(colored_mesh, node_name="segmented_mesh")
    
    # Add bounding boxes
    np.random.seed(color_seed)
    unique_ids = np.unique(face_ids)
    
    for i, part_id in enumerate(unique_ids):
        if part_id < 0 or i >= len(aabb):
            continue
        
        # Get AABB for this part
        min_xyz, max_xyz = aabb[i]
        center = (min_xyz + max_xyz) / 2
        size = max_xyz - min_xyz
        
        # Create box outline
        box = trimesh.path.creation.box_outline()
        box.vertices *= size
        box.vertices += center
        
        # Assign color (same as mesh part)
        color = np.random.randint(50, 255, size=3)
        box_color = np.array([[color[0], color[1], color[2], 255]])
        box_color = np.repeat(box_color, len(box.entities), axis=0).astype(np.uint8)
        box.colors = box_color
        
        scene.add_geometry(box, node_name=f"bbox_part_{part_id}")
    
    return scene


def create_segmented_parts_scene(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray
) -> trimesh.Scene:
    """
    Create a scene where each segmented part is a separate geometry.

    Args:
        mesh: Input trimesh object
        face_ids: Part ID for each face [num_faces]

    Returns:
        Trimesh Scene with separate meshes for each part
    """
    scene = trimesh.Scene()
    unique_ids = np.unique(face_ids)

    for part_id in unique_ids:
        if part_id < 0:
            continue

        # Get face indices for this part
        mask = (face_ids == part_id)
        face_indices = np.where(mask)[0]

        # Create submesh
        # append=True ensures we get a single mesh object for these faces
        try:
             submesh = mesh.submesh([face_indices], append=True)
        except Exception as e:
             logger.warning(f"Failed to create submesh for part {part_id}: {e}")
             continue
        
        if isinstance(submesh, list):
             if len(submesh) > 0:
                 submesh = submesh[0]
             else:
                 continue

        # Add to scene with descriptive name
        scene.add_geometry(submesh, node_name=f"part_{int(part_id)}")

    return scene