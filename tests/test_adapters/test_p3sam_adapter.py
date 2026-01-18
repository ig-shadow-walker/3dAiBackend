"""
Real GPU inference tests for P3-SAM mesh segmentation adapter.

Tests the P3-SAM adapter with actual model inference on GPU.
No mocks - this performs real mesh segmentation.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from adapters.p3sam_adapter import P3SAMSegmentationAdapter
from tests.test_adapters.gpu_memory_tracker import track_gpu_memory


class TestP3SAMAdapterRealInference:
    """Test P3-SAM adapter with real GPU inference"""

    @pytest.fixture
    def adapter(self):
        """Create a P3-SAM adapter instance"""
        return P3SAMSegmentationAdapter()

    @track_gpu_memory
    @pytest.mark.timeout(600)  # 10 minutes timeout for segmentation
    def test_mesh_segmentation_default(self, adapter):
        """Test mesh segmentation with default parameters"""
        # Load the model
        adapter.load(0)

        try:
            # Find a sample mesh
            sample_meshes = [
                Path("assets/example_mesh/typical_creature_dragon.obj"),
                Path("assets/example_mesh/typical_creature_elephant.obj"),
                Path("assets/example_mesh/typical_humanoid_mech.obj"),
            ]

            sample_mesh = None
            for mesh_path in sample_meshes:
                if mesh_path.exists():
                    sample_mesh = mesh_path
                    break

            if sample_mesh is None:
                pytest.skip("No sample mesh files found for testing")

            # Test with default parameters
            inputs = {
                "mesh_path": str(sample_mesh),
                "point_num": 50000,  # Reduce for faster testing
                "prompt_num": 200,  # Reduce for faster testing
                "threshold": 0.95,
                "post_process": True,
                "seed": 42,
                "output_format": "glb",
            }

            result = adapter.process(inputs)

            # Verify the result
            assert result is not None
            assert "output_mesh_path" in result
            assert "segmentation_info" in result
            assert "aabb_path" in result
            assert result["success"] is True

            # Check the output file exists
            output_path = Path(result["output_mesh_path"])
            assert output_path.exists()
            assert output_path.suffix == ".glb"

            # Check AABB file exists
            aabb_path = Path(result["aabb_path"])
            assert aabb_path.exists()
            assert aabb_path.suffix == ".npy"

            # Load and verify AABB
            aabb = np.load(aabb_path)
            assert aabb.ndim == 3  # [N, 2, 3]
            assert aabb.shape[1] == 2  # min, max
            assert aabb.shape[2] == 3  # x, y, z

            # Verify segmentation info
            seg_info = result["segmentation_info"]
            assert "num_parts" in seg_info
            assert "point_num" in seg_info
            assert "prompt_num" in seg_info
            assert seg_info["point_num"] == inputs["point_num"]
            assert seg_info["prompt_num"] == inputs["prompt_num"]

            # Check generation info
            gen_info = result["generation_info"]
            assert gen_info["input_mesh"] == str(sample_mesh)
            assert gen_info["segmentation_method"] == "P3-SAM"
            assert "vertex_count" in gen_info
            assert "face_count" in gen_info

            print(f"Successfully segmented mesh: {output_path}")
            print(f"Number of parts: {result['num_parts']}")
            print(f"AABB shape: {aabb.shape}")

        finally:
            # Unload the model
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_mesh_segmentation_custom_parameters(self, adapter):
        """Test mesh segmentation with custom parameters"""
        adapter.load(0)

        try:
            sample_mesh = Path("assets/example_mesh/typical_creature_dragon.obj")
            if not sample_mesh.exists():
                pytest.skip("Sample mesh not found")

            # Test with custom parameters
            inputs = {
                "mesh_path": str(sample_mesh),
                "point_num": 30000,
                "prompt_num": 100,
                "threshold": 0.9,
                "post_process": False,
                "seed": 123,
                "prompt_bs": 16,
                "save_mid_res": False,
                "output_format": "glb",
            }

            result = adapter.process(inputs)

            assert result is not None
            assert result["success"] is True

            seg_info = result["segmentation_info"]
            assert seg_info["point_num"] == 30000
            assert seg_info["prompt_num"] == 100
            assert seg_info["threshold"] == 0.9
            assert seg_info["post_process"] == False

            print(f"Segmentation with custom parameters completed")
            print(f"Parts found: {result['num_parts']}")

        finally:
            adapter.unload()

    @track_gpu_memory
    @pytest.mark.timeout(600)
    def test_mesh_segmentation_different_mesh_types(self, adapter):
        """Test segmentation with different mesh file formats"""
        adapter.load(0)

        try:
            # Test with different mesh formats
            test_meshes = [
                Path("assets/example_mesh/sample.obj"),
                Path("assets/example_mesh/sample.glb"),
                Path("assets/example_mesh/sample.ply"),
            ]

            tested = False
            for mesh_path in test_meshes:
                if mesh_path.exists():
                    inputs = {
                        "mesh_path": str(mesh_path),
                        "point_num": 30000,
                        "prompt_num": 100,
                        "seed": 42,
                    }

                    result = adapter.process(inputs)
                    assert result is not None
                    assert result["success"] is True

                    print(f"Tested mesh format: {mesh_path.suffix}")
                    tested = True
                    break

            if not tested:
                pytest.skip("No test meshes found for format testing")

        finally:
            adapter.unload()

    def test_error_handling(self, adapter):
        """Test error handling for invalid inputs"""
        adapter.load(0)

        try:
            # Test with non-existent file
            with pytest.raises(Exception):
                adapter.process({"mesh_path": "nonexistent_file.obj"})

            print("Error handling test passed")

        finally:
            adapter.unload()

    def test_get_parameter_schema(self, adapter):
        """Test parameter schema retrieval"""
        schema = adapter.get_parameter_schema()

        assert "parameters" in schema
        params = schema["parameters"]

        # Verify key parameters are present
        assert "point_num" in params
        assert "prompt_num" in params
        assert "threshold" in params
        assert "post_process" in params
        assert "seed" in params
        assert "prompt_bs" in params
        assert "save_mid_res" in params

        # Verify parameter structure
        assert params["point_num"]["type"] == "integer"
        assert params["point_num"]["default"] == 100000
        assert "minimum" in params["point_num"]
        assert "maximum" in params["point_num"]

        assert params["prompt_num"]["type"] == "integer"
        assert params["prompt_num"]["default"] == 400

        assert params["threshold"]["type"] == "number"
        assert params["threshold"]["default"] == 0.95
        assert params["threshold"]["minimum"] == 0.0
        assert params["threshold"]["maximum"] == 1.0

        assert params["post_process"]["type"] == "boolean"
        assert params["seed"]["type"] == "integer"
        assert params["prompt_bs"]["type"] == "integer"
        assert params["save_mid_res"]["type"] == "boolean"

        print("Parameter schema test passed")
        print(f"Available parameters: {list(params.keys())}")

    def test_part_statistics(self, adapter):
        """Test that part statistics are properly computed"""
        adapter.load(0)

        try:
            sample_mesh = Path("assets/example_mesh/typical_creature_dragon.obj")
            if not sample_mesh.exists():
                pytest.skip("Sample mesh not found")

            inputs = {
                "mesh_path": str(sample_mesh),
                "point_num": 30000,
                "prompt_num": 100,
                "seed": 42,
            }

            result = adapter.process(inputs)
            seg_info = result["segmentation_info"]

            assert "part_statistics" in seg_info
            stats = seg_info["part_statistics"]

            assert "num_parts_actual" in stats
            assert "num_parts_requested" in stats
            assert "part_sizes" in stats
            assert "average_part_size" in stats
            assert "part_size_std" in stats

            assert isinstance(stats["part_sizes"], dict)
            assert stats["num_parts_actual"] > 0

            print("Part statistics test passed")
            print(f"Actual parts: {stats['num_parts_actual']}")
            print(f"Average part size: {stats['average_part_size']:.2f}")

        finally:
            adapter.unload()


class TestP3SAMAdapterIntegration:
    """Integration tests for P3-SAM adapter"""

    def test_adapter_metadata(self):
        """Test adapter metadata and configuration"""
        adapter = P3SAMSegmentationAdapter()

        assert adapter.MODEL_ID == "p3sam_mesh_segmentation"
        assert adapter.FEATURE_TYPE == "mesh_segmentation"
        assert adapter.vram_requirement == 6144  # 6GB

        print("Adapter metadata test passed")

    def test_supported_formats(self):
        """Test supported input/output formats"""
        adapter = P3SAMSegmentationAdapter()

        formats = adapter.get_supported_formats()
        assert "input" in formats
        assert "output" in formats

        # P3-SAM should support multiple input formats
        assert "glb" in formats["input"]
        assert "obj" in formats["input"]
        assert "ply" in formats["input"]

        # Output format
        assert "glb" in formats["output"]

        print("Supported formats test passed")
        print(f"Input formats: {formats['input']}")
        print(f"Output formats: {formats['output']}")

    def test_coexistence_with_partfield(self):
        """Test that P3-SAM can coexist with PartField"""
        from adapters.partfield_adapter import PartFieldSegmentationAdapter

        p3sam_adapter = P3SAMSegmentationAdapter()
        partfield_adapter = PartFieldSegmentationAdapter()

        # Both should have mesh_segmentation feature
        assert p3sam_adapter.FEATURE_TYPE == "mesh_segmentation"
        assert partfield_adapter.FEATURE_TYPE == "mesh_segmentation"

        # But different model IDs
        assert p3sam_adapter.MODEL_ID != partfield_adapter.MODEL_ID
        assert p3sam_adapter.MODEL_ID == "p3sam_mesh_segmentation"
        assert partfield_adapter.MODEL_ID == "partfield_mesh_segmentation"

        print("Coexistence test passed - both segmentation models can be used")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

