"""
UltraShape Pipeline Helper

Wrapper for UltraShape inference pipeline that integrates with Hunyuan3D-2.1
for coarse mesh generation followed by refinement.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)


class UltraShapeInferenceHelper:
    """
    Helper class for UltraShape mesh refinement pipeline.
    
    This class wraps the complete UltraShape inference process:
    1. Generate coarse mesh using Hunyuan3D-2.1
    2. Apply UltraShape refinement to produce high-quality mesh
    """
    
    def __init__(
        self,
        ultrashape_root: str,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize UltraShape helper.
        
        Args:
            ultrashape_root: Root directory of UltraShape codebase
            config_path: Path to inference config YAML
            checkpoint_path: Path to UltraShape checkpoint (.pt file)
            device: Device to run inference on
            dtype: Data type for inference
        """
        self.ultrashape_root = Path(ultrashape_root)
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.dtype = dtype
        
        # Add UltraShape to Python path
        if str(self.ultrashape_root) not in sys.path:
            sys.path.insert(0, str(self.ultrashape_root))
        
        # Components will be loaded on demand
        self.vae = None
        self.dit = None
        self.conditioner = None
        self.scheduler = None
        self.image_processor = None
        self.pipeline = None
        self.surface_loader = None
        self.rembg = None
        
        logger.info(f"Initialized UltraShape helper with config: {config_path}")
    
    def load_models(self) -> None:
        """Load UltraShape components from checkpoint."""
        try:
            from omegaconf import OmegaConf
            from ultrashape.utils.misc import instantiate_from_config
            from ultrashape.pipelines import UltraShapePipeline
            from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
            from ultrashape.rembg import BackgroundRemover
            
            logger.info(f"Loading UltraShape config from {self.config_path}...")
            config = OmegaConf.load(self.config_path)
            
            logger.info("Instantiating VAE...")
            self.vae = instantiate_from_config(config.model.params.vae_config)
            
            logger.info("Instantiating DiT...")
            self.dit = instantiate_from_config(config.model.params.dit_cfg)
            
            logger.info("Instantiating Conditioner...")
            self.conditioner = instantiate_from_config(config.model.params.conditioner_config)
            
            logger.info("Instantiating Scheduler & Processor...")
            self.scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
            self.image_processor = instantiate_from_config(config.model.params.image_processor_cfg)
            
            logger.info(f"Loading weights from {self.checkpoint_path}...")
            weights = torch.load(self.checkpoint_path, map_location='cpu')
            
            self.vae.load_state_dict(weights['vae'], strict=True)
            self.dit.load_state_dict(weights['dit'], strict=True)
            self.conditioner.load_state_dict(weights['conditioner'], strict=True)
            
            # Move to device and set to eval mode
            self.vae.eval().to(self.device)
            self.dit.eval().to(self.device)
            self.conditioner.eval().to(self.device)
            
            # Enable flashvdm decoder if available
            if hasattr(self.vae, 'enable_flashvdm_decoder'):
                self.vae.enable_flashvdm_decoder()
            
            # Create pipeline
            self.pipeline = UltraShapePipeline(
                vae=self.vae,
                model=self.dit,
                scheduler=self.scheduler,
                conditioner=self.conditioner,
                image_processor=self.image_processor
            )
            
            # Initialize surface loader
            logger.info("Initializing Surface Loader...")
            self.surface_loader = SharpEdgeSurfaceLoader(
                num_sharp_points=204800,
                num_uniform_points=204800,
            )
            
            # Initialize background remover
            self.rembg = BackgroundRemover()
            
            logger.info("UltraShape models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load UltraShape models: {str(e)}")
            raise
    
    def unload_models(self) -> None:
        """Unload UltraShape models to free memory."""
        if self.vae is not None:
            del self.vae
            self.vae = None
        
        if self.dit is not None:
            del self.dit
            self.dit = None
        
        if self.conditioner is not None:
            del self.conditioner
            self.conditioner = None
        
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        if self.surface_loader is not None:
            del self.surface_loader
            self.surface_loader = None
        
        if self.rembg is not None:
            del self.rembg
            self.rembg = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("UltraShape models unloaded")
    
    def refine_mesh(
        self,
        image_path: str,
        coarse_mesh_path: str,
        num_inference_steps: int = 50,
        num_latents: int = 32768,
        octree_res: int = 1024,
        chunk_size: int = 8000,
        scale: float = 0.99,
        seed: int = 42,
        remove_bg: bool = False,
    ) -> trimesh.Trimesh:
        """
        Refine a coarse mesh using UltraShape.
        
        Args:
            image_path: Path to input image
            coarse_mesh_path: Path to coarse mesh (from Hunyuan3D-2.1)
            num_inference_steps: Number of diffusion steps
            num_latents: Number of latent tokens
            octree_res: Marching cubes resolution
            chunk_size: Chunk size for inference
            scale: Mesh normalization scale
            seed: Random seed
            remove_bg: Force background removal
        
        Returns:
            Refined mesh as trimesh.Trimesh
        """
        if self.pipeline is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        try:
            from ultrashape.utils import voxelize_from_point
            
            # Load and process image
            logger.info(f"Processing image: {image_path}")
            image = Image.open(image_path)
            
            if remove_bg or image.mode != 'RGBA':
                logger.info("Removing background...")
                image = self.rembg(image)
            
            # Load and process coarse mesh
            logger.info(f"Loading coarse mesh: {coarse_mesh_path}")
            surface = self.surface_loader(
                coarse_mesh_path,
                normalize_scale=scale
            ).to(self.device, dtype=torch.float16)
            
            pc = surface[:, :, :3]  # [B, N, 3]
            
            # Voxelize
            logger.info("Voxelizing mesh...")
            from omegaconf import OmegaConf
            config = OmegaConf.load(self.config_path)
            voxel_res = config.model.params.vae_config.params.voxel_query_res
            _, voxel_idx = voxelize_from_point(pc, num_latents, resolution=voxel_res)
            
            # Run diffusion
            logger.info("Running UltraShape diffusion process...")
            generator = torch.Generator(self.device).manual_seed(seed)
            
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                mesh, _ = self.pipeline(
                    image=image,
                    voxel_cond=voxel_idx,
                    generator=generator,
                    box_v=1.0,
                    mc_level=0.0,
                    octree_resolution=octree_res,
                    num_inference_steps=num_inference_steps,
                    num_chunks=chunk_size,
                )
            
            logger.info("UltraShape refinement completed successfully")
            return mesh[0]
            
        except Exception as e:
            logger.error(f"UltraShape refinement failed: {str(e)}")
            raise
    
    def generate_and_refine(
        self,
        image_path: str,
        hunyuan_pipeline,
        hunyuan_bg_remover,
        output_dir: Optional[str] = None,
        **refine_kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Complete pipeline: Generate coarse mesh + refine.
        
        Args:
            image_path: Path to input image
            hunyuan_pipeline: Loaded Hunyuan3D-2.1 shape generation pipeline
            hunyuan_bg_remover: Hunyuan3D background remover
            output_dir: Directory to save output (temp dir if None)
            **refine_kwargs: Additional arguments for refine_mesh()
        
        Returns:
            Tuple of (output_mesh_path, generation_info)
        """
        if self.pipeline is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Create temp directory if needed
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="ultrashape_")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Step 1: Generate coarse mesh using Hunyuan3D-2.1
            logger.info("Step 1: Generating coarse mesh with Hunyuan3D-2.1...")
            image = Image.open(image_path).convert("RGBA")
            if image.mode == "RGB":
                image = hunyuan_bg_remover(image)
            
            coarse_mesh = hunyuan_pipeline(image=image)[0]
            
            # Save coarse mesh temporarily
            coarse_mesh_path = os.path.join(output_dir, "coarse_mesh.glb")
            coarse_mesh.export(coarse_mesh_path)
            logger.info(f"Coarse mesh saved to: {coarse_mesh_path}")
            
            # Step 2: Refine mesh using UltraShape
            logger.info("Step 2: Refining mesh with UltraShape...")
            refined_mesh = self.refine_mesh(
                image_path=image_path,
                coarse_mesh_path=coarse_mesh_path,
                **refine_kwargs,
            )
            
            # Save refined mesh
            base_name = Path(image_path).stem
            refined_mesh_path = os.path.join(output_dir, f"{base_name}_refined.glb")
            refined_mesh.export(refined_mesh_path)
            logger.info(f"Refined mesh saved to: {refined_mesh_path}")
            
            # Gather statistics
            generation_info = {
                "coarse_mesh_path": coarse_mesh_path,
                "refined_mesh_path": refined_mesh_path,
                "vertex_count": len(refined_mesh.vertices),
                "face_count": len(refined_mesh.faces),
                "refinement_params": refine_kwargs,
            }
            
            return refined_mesh_path, generation_info
            
        except Exception as e:
            logger.error(f"Generate and refine failed: {str(e)}")
            raise


def create_ultrashape_helper(
    checkpoint_path: str,
    ultrashape_root: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = "cuda",
) -> UltraShapeInferenceHelper:
    """
    Convenience function to create UltraShapeInferenceHelper.
    
    Args:
        checkpoint_path: Path to UltraShape checkpoint
        ultrashape_root: Root directory of UltraShape (auto-detected if None)
        config_path: Path to config file (auto-detected if None)
        device: Device to run on
    
    Returns:
        Initialized UltraShapeInferenceHelper
    """
    if ultrashape_root is None:
        # Auto-detect from current working directory
        ultrashape_root = os.path.join(os.getcwd(), "thirdparty", "UltraShape")
    
    if config_path is None:
        # Use default inference config
        config_path = os.path.join(ultrashape_root, "configs", "infer_dit_refine.yaml")
    
    return UltraShapeInferenceHelper(
        ultrashape_root=ultrashape_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

