"""
Real GPU inference tests for UltraShape adapter.

Tests the UltraShape adapter with actual model inference on GPU.
No mocks - this performs real image-to-refined-mesh generation.
"""

from pathlib import Path

import pytest
import torch

from adapters.ultrashape_adapter import UltraShapeImageToRawMeshAdapter
from tests.test_adapters.gpu_memory_tracker import track_gpu_memory


class TestUltraShapeAdapter:
    """Test UltraShape adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create an UltraShape adapter instance"""
        return UltraShapeImageToRawMeshAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(1200)  # 20 minutes timeout
    def test_image_to_refined_mesh_generation(self, adapter):
        """Test image-to-refined-mesh generation with real inference"""
        # Check if example image exists
        example_image = Path("assets/example_image/073.png")
        if not example_image.exists():
            pytest.skip("Example image not found")

        # Check if checkpoints exist
        if not Path(adapter.ultrashape_checkpoint).exists():
            pytest.skip("UltraShape checkpoint not found")
        if not Path(adapter.hunyuan_model_path).exists():
            pytest.skip("Hunyuan3D model not found")

        # Load the model
        adapter.load(0)

        try:
            inputs = {
                "image_path": str(example_image),
                "output_format": "glb",
                "num_inference_steps": 20,  # Reduced for testing
                "num_latents": 16384,  # Reduced for testing
                "octree_res": 512,  # Reduced for testing
                "chunk_size": 4000,
                "seed": 42,
            }

            result = adapter.process(inputs)

            # Verify the result
            assert result is not None
            assert "output_mesh_path" in result
            assert "generation_info" in result

            # Check the output file exists
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix == ".glb"

            # Verify generation info
            gen_info = result["generation_info"]
            assert "vertex_count" in gen_info
            assert "face_count" in gen_info
            assert not gen_info["has_texture"]
            assert gen_info["refinement_steps"] == 20
            assert "coarse_mesh_path" in gen_info

            print(f"Successfully generated refined mesh: {output_path}")
            print(
                f"Vertices: {gen_info['vertex_count']}, Faces: {gen_info['face_count']}"
            )

        finally:
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(1200)
    def test_different_output_formats(self, adapter):
        """Test generation with different output formats"""
        example_image = Path("assets/example_image/075.png")
        if not example_image.exists():
            pytest.skip("Example image not found")
        
        if not Path(adapter.ultrashape_checkpoint).exists():
            pytest.skip("UltraShape checkpoint not found")

        adapter.load(0)

        try:
            base_inputs = {
                "image_path": str(example_image),
                "num_inference_steps": 15,  # Quick test
                "num_latents": 8192,  # Minimal
                "octree_res": 384,
                "seed": 42,
            }

            for output_format in ["glb", "obj"]:
                inputs = {**base_inputs, "output_format": output_format}

                result = adapter.process(inputs)

                assert result is not None
                output_path = Path(result["output_mesh_path"])
                assert output_path.exists()
                assert output_path.suffix == f".{output_format}"

                print(f"Generated {output_format} format: {output_path}")

        finally:
            adapter.unload()


class TestUltraShapeErrorHandling:
    """Test error handling for UltraShape adapter"""

    @pytest.fixture
    def adapter(self):
        """Create adapter for error testing"""
        return UltraShapeImageToRawMeshAdapter()

    def test_missing_image_file(self, adapter):
        """Test handling of missing image file"""
        adapter.load(0)

        try:
            inputs = {
                "image_path": "nonexistent_image.png",
                "output_format": "glb",
            }

            with pytest.raises(Exception):
                adapter.process(inputs)

            print("Missing image file correctly handled")

        finally:
            adapter.unload()

    def test_unsupported_format(self, adapter):
        """Test handling of unsupported output format"""
        example_image = Path("assets/example_image/073.png")
        if not example_image.exists():
            pytest.skip("Example image not found")

        adapter.load(0)

        try:
            inputs = {
                "image_path": str(example_image),
                "output_format": "stl",  # Not supported
            }

            with pytest.raises(Exception):
                adapter.process(inputs)

            print("Unsupported format correctly handled")

        finally:
            adapter.unload()


class TestUltraShapeMemoryManagement:
    """Test GPU memory management for UltraShape adapter"""

    @track_gpu_memory
    def test_memory_cleanup(self):
        """Test memory cleanup after mesh generation"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        adapter = UltraShapeImageToRawMeshAdapter()

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

