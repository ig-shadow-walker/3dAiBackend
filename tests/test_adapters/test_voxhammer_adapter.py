"""
Real GPU inference tests for VoxHammer adapters.

Tests the VoxHammer adapters with actual model inference on GPU.
No mocks - this performs real local mesh editing.
"""

from pathlib import Path

import pytest
import torch

from adapters.voxhammer_adapter import (
    VoxHammerTextMeshEditingAdapter,
    VoxHammerImageMeshEditingAdapter,
)
from tests.test_adapters.gpu_memory_tracker import track_gpu_memory


class TestVoxHammerTextEditingAdapter:
    """Test VoxHammer text-guided editing adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a VoxHammer text editing adapter instance"""
        return VoxHammerTextMeshEditingAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(1800)  # 30 minutes timeout
    def test_text_guided_editing_bbox(self, adapter):
        """Test text-guided editing with bounding box mask"""
        # Check if example mesh exists
        example_mesh = Path("assets/example_mesh/typical_creature_dragon.obj")
        if not example_mesh.exists():
            pytest.skip("Example mesh not found")

        # Load the model
        adapter.load(0)

        try:
            inputs = {
                "mesh_path": str(example_mesh),
                "mask_type": "bbox",
                "mask_center": [0.0, 0.5, 0.0],
                "mask_params": [0.3, 0.3, 0.3],
                "source_prompt": "a dragon head",
                "target_prompt": "a dragon head with horns",
                "num_views": 30,  # Reduced for testing
                "resolution": 256,  # Lower resolution for testing
                "output_format": "glb",
            }

            result = adapter.process(inputs)

            # Verify the result
            assert result is not None
            assert "output_mesh_path" in result
            assert "editing_info" in result

            # Check the output file exists
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix == ".glb"

            # Verify editing info
            edit_info = result["editing_info"]
            assert edit_info["mask_type"] == "bbox"
            assert edit_info["source_prompt"] == inputs["source_prompt"]
            assert edit_info["target_prompt"] == inputs["target_prompt"]

            print(f"Successfully edited mesh: {output_path}")
            print(f"Vertices: {edit_info['vertex_count']}, Faces: {edit_info['face_count']}")

        finally:
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(1800)
    def test_text_guided_editing_ellipsoid(self, adapter):
        """Test text-guided editing with ellipsoid mask"""
        example_mesh = Path("assets/example_mesh/typical_creature_dragon.obj")
        if not example_mesh.exists():
            pytest.skip("Example mesh not found")

        adapter.load(0)

        try:
            inputs = {
                "mesh_path": str(example_mesh),
                "mask_type": "ellipsoid",
                "mask_center": [0.0, 0.3, 0.0],
                "mask_params": [0.2, 0.2, 0.2],
                "source_prompt": "dragon scales",
                "target_prompt": "smooth skin",
                "num_views": 30,  # Reduced
                "resolution": 256,
                "output_format": "glb",
            }

            result = adapter.process(inputs)

            assert result is not None
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()

            edit_info = result["editing_info"]
            assert edit_info["mask_type"] == "ellipsoid"

            print(f"Successfully edited mesh with ellipsoid mask: {output_path}")

        finally:
            adapter.unload()


class TestVoxHammerImageEditingAdapter:
    """Test VoxHammer image-guided editing adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a VoxHammer image editing adapter instance"""
        return VoxHammerImageMeshEditingAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(1800)
    def test_image_guided_editing(self, adapter):
        """Test image-guided editing"""
        # Check if example files exist
        example_mesh = Path("assets/example_mesh/typical_creature_dragon.obj")
        example_image_dir = Path("assets/example/images")
        
        if not example_mesh.exists():
            pytest.skip("Example mesh not found")
        
        # Check for required images
        # required_images = ["2d_render.png", "2d_edit.png", "2d_mask.png"]
        # if not all((example_image_dir / img).exists() for img in required_images):
        #     pytest.skip("Example images not found")

        adapter.load(0)

        try:
            inputs = {
                "mesh_path": str(example_mesh),
                "mask_type": "bbox",
                "mask_center": [0.0, 0.5, 0.0],
                "mask_params": [0.4, 0.4, 0.4],
                # you can either use existing images or a single target prompt
                # "source_image_path": str(example_image_dir / "2d_render.png"),
                # "target_image_path": str(example_image_dir / "2d_edit.png"),
                # "mask_image_path": str(example_image_dir / "2d_mask.png"),
                "target_prompt": "a dog", 
                "num_views": 30,
                "resolution": 256,
                "output_format": "glb",
            }

            result = adapter.process(inputs)

            assert result is not None
            assert "output_mesh_path" in result

            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix == ".glb"

            edit_info = result["editing_info"]
            assert edit_info["mask_type"] == "bbox"
            assert "source_image" in edit_info
            assert "target_image" in edit_info

            print(f"Successfully edited mesh with images: {output_path}")

        finally:
            adapter.unload()


class TestVoxHammerErrorHandling:
    """Test error handling for VoxHammer adapters"""

    @pytest.fixture
    def text_adapter(self):
        """Create text editing adapter for error testing"""
        return VoxHammerTextMeshEditingAdapter()

    def test_missing_mesh_file(self, text_adapter):
        """Test handling of missing mesh file"""
        text_adapter.load(0)

        try:
            inputs = {
                "mesh_path": "nonexistent_mesh.obj",
                "mask_type": "bbox",
                "mask_center": [0, 0, 0],
                "mask_params": [1, 1, 1],
                "source_prompt": "test",
                "target_prompt": "test2",
            }

            with pytest.raises(Exception):
                text_adapter.process(inputs)

            print("Missing mesh file correctly handled")

        finally:
            text_adapter.unload()

    def test_invalid_mask_params(self, text_adapter):
        """Test handling of invalid mask parameters"""
        example_mesh = Path("assets/example_mesh/typical_creature_dragon.obj")
        if not example_mesh.exists():
            pytest.skip("Example mesh not found")

        text_adapter.load(0)

        try:
            inputs = {
                "mesh_path": str(example_mesh),
                "mask_type": "bbox",
                "mask_center": [0, 0],  # Invalid: only 2 values
                "mask_params": [1, 1, 1],
                "source_prompt": "test",
                "target_prompt": "test2",
            }

            with pytest.raises(Exception):
                text_adapter.process(inputs)

            print("Invalid mask parameters correctly handled")

        finally:
            text_adapter.unload()


class TestVoxHammerMemoryManagement:
    """Test GPU memory management for VoxHammer adapters"""

    @track_gpu_memory
    def test_memory_cleanup(self):
        """Test memory cleanup after mesh editing"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        adapter = VoxHammerTextMeshEditingAdapter()

        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        adapter.load(0)

        try:
            # Memory should increase after loading
            loaded_memory = torch.cuda.memory_allocated()
            assert loaded_memory > initial_memory
            print(f"Memory after loading: {loaded_memory / 1024**2:.1f}MB")

        finally:
            adapter.unload()

        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        print(
            f"Memory usage - Initial: {initial_memory / 1024**2:.1f}MB, "
            f"Final: {final_memory / 1024**2:.1f}MB"
        )

        # Memory should be released (allow some tolerance)
        assert final_memory <= initial_memory + 200 * 1024**2  # 200MB tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

