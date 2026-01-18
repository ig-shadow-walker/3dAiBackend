"""
Real GPU inference tests for TRELLIS.2 adapters.

Tests the TRELLIS.2 adapters with actual model inference on GPU.
No mocks - this performs real mesh generation and texturing.
"""

from pathlib import Path

import pytest
import torch

from adapters.trellis2_adapter import (
    Trellis2ImageMeshPaintingAdapter,
    Trellis2ImageToTexturedMeshAdapter,
)
from tests.test_adapters.gpu_memory_tracker import track_gpu_memory


class TestTrellis2ImageToMeshAdapter:
    """Test TRELLIS.2 image-to-mesh adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a TRELLIS.2 image-to-mesh adapter instance"""
        return Trellis2ImageToTexturedMeshAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(600)  # 10 minutes timeout
    def test_image_to_mesh_generation(self, adapter):
        """Test image-to-mesh generation with default parameters"""
        # Load the model
        adapter.load(0)

        try:
            # Find a sample image
            sample_images = [
                Path("assets/example_image/typical_creature_dragon.png"),
                Path("assets/example_image/typical_humanoid_mech.png"),
                Path("thirdparty/TRELLIS.2/assets/example_image/T.png"),
            ]

            sample_image = None
            for image_path in sample_images:
                if image_path.exists():
                    sample_image = image_path
                    break

            if sample_image is None:
                pytest.skip("No sample image files found for testing")

            # Test with default parameters
            inputs = {
                "image_path": str(sample_image),
                "decimation_target": 500000,  # Reduce for faster testing
                "texture_size": 2048,  # Reduce for faster testing
                "output_format": "glb",
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
            assert gen_info["model"] == "TRELLIS.2"
            assert gen_info["image_path"] == str(sample_image)
            assert gen_info["seed"] == 42
            assert "vertex_count" in gen_info
            assert "face_count" in gen_info

            print(f"Successfully generated mesh: {output_path}")
            print(f"Vertices: {gen_info['vertex_count']}, Faces: {gen_info['face_count']}")

        finally:
            # Unload the model
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_image_to_mesh_with_custom_parameters(self, adapter):
        """Test image-to-mesh generation with custom parameters"""
        adapter.load(0)

        try:
            sample_image = Path("thirdparty/TRELLIS.2/assets/example_image/T.png")
            if not sample_image.exists():
                pytest.skip("Sample image not found")

            # Test with custom parameters
            inputs = {
                "image_path": str(sample_image),
                "decimation_target": 100000,
                "texture_size": 1024,
                "remesh": False,
                "output_format": "obj",
                "seed": 123,
            }

            result = adapter.process(inputs)

            assert result is not None
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix == ".obj"

            gen_info = result["generation_info"]
            assert gen_info["decimation_target"] == 100000
            assert gen_info["texture_size"] == 1024
            assert gen_info["remesh"] == False

            print(f"Generated mesh with custom parameters: {output_path}")

        finally:
            adapter.unload()

    def test_error_handling(self, adapter):
        """Test error handling for invalid inputs"""
        adapter.load(0)

        try:
            # Test with non-existent file
            with pytest.raises(Exception):
                adapter.process({"image_path": "nonexistent_image.png"})

            print("Error handling test passed")

        finally:
            adapter.unload()

    def test_get_parameter_schema(self, adapter):
        """Test parameter schema retrieval"""
        schema = adapter.get_parameter_schema()

        assert "parameters" in schema
        params = schema["parameters"]

        # Verify key parameters are present
        assert "decimation_target" in params
        assert "texture_size" in params
        assert "remesh" in params
        assert "seed" in params

        # Verify parameter structure
        assert params["decimation_target"]["type"] == "integer"
        assert params["decimation_target"]["default"] == 1000000
        assert "minimum" in params["decimation_target"]

        assert params["texture_size"]["type"] == "integer"
        assert "enum" in params["texture_size"]

        print("Parameter schema test passed")
        print(f"Available parameters: {list(params.keys())}")


class TestTrellis2MeshPaintingAdapter:
    """Test TRELLIS.2 mesh painting adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a TRELLIS.2 mesh painting adapter instance"""
        return Trellis2ImageMeshPaintingAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_mesh_texturing(self, adapter):
        """Test mesh texturing with image guidance"""
        adapter.load(0)

        try:
            # Find sample mesh and image
            sample_mesh = Path("thirdparty/TRELLIS.2/assets/example_texturing/the_forgotten_knight.ply")
            sample_image = Path("thirdparty/TRELLIS.2/assets/example_texturing/image.webp")

            if not sample_mesh.exists() or not sample_image.exists():
                pytest.skip("Sample files not found for texturing test")

            inputs = {
                "mesh_path": str(sample_mesh),
                "image_path": str(sample_image),
                "output_format": "glb",
                "extension_webp": True,
            }

            result = adapter.process(inputs)

            # Verify the result
            assert result is not None
            assert "output_mesh_path" in result
            assert "generation_info" in result

            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix == ".glb"

            gen_info = result["generation_info"]
            assert gen_info["model"] == "TRELLIS.2-Texturing"
            assert gen_info["mesh_path"] == str(sample_mesh)
            assert gen_info["image_path"] == str(sample_image)

            print(f"Successfully textured mesh: {output_path}")

        finally:
            adapter.unload()

    def test_get_parameter_schema(self, adapter):
        """Test parameter schema for painting adapter"""
        schema = adapter.get_parameter_schema()

        assert "parameters" in schema
        params = schema["parameters"]

        # Verify painting-specific parameters
        assert "extension_webp" in params
        assert params["extension_webp"]["type"] == "boolean"
        assert params["extension_webp"]["default"] == True

        print("Painting parameter schema test passed")


class TestTrellis2AdapterIntegration:
    """Integration tests for TRELLIS.2 adapters"""

    def test_both_adapters_coexist(self):
        """Test that both TRELLIS.2 adapters can be instantiated"""
        mesh_gen_adapter = Trellis2ImageToTexturedMeshAdapter()
        painting_adapter = Trellis2ImageMeshPaintingAdapter()

        assert mesh_gen_adapter.MODEL_ID == "trellis2_image_to_textured_mesh"
        assert painting_adapter.MODEL_ID == "trellis2_image_mesh_painting"

        assert mesh_gen_adapter.FEATURE_TYPE == "image_to_textured_mesh"
        assert painting_adapter.FEATURE_TYPE == "image_mesh_painting"

        print("Adapter coexistence test passed")

    def test_supported_formats(self):
        """Test supported formats for both adapters"""
        mesh_gen_adapter = Trellis2ImageToTexturedMeshAdapter()
        painting_adapter = Trellis2ImageMeshPaintingAdapter()

        mesh_gen_formats = mesh_gen_adapter.get_supported_formats()
        assert "input" in mesh_gen_formats
        assert "output" in mesh_gen_formats
        assert "glb" in mesh_gen_formats["output"]

        painting_formats = painting_adapter.get_supported_formats()
        assert "mesh" in painting_formats["input"] or "image" in painting_formats["input"]

        print("Supported formats test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

