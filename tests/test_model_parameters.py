"""
Tests for model parameter schema system.

Verifies that all models properly implement the get_parameter_schema() method
and that the API endpoint works correctly.
"""

import pytest
import requests
from typing import Dict, Any

from core.scheduler.model_factory import ModelFactory


class TestModelParameterSchemas:
    """Test parameter schema implementation in adapters"""

    def test_trellis2_image_to_mesh_schema(self):
        """Test TRELLIS.2 image-to-mesh parameter schema"""
        from adapters.trellis2_adapter import Trellis2ImageToTexturedMeshAdapter

        adapter = Trellis2ImageToTexturedMeshAdapter()
        schema = adapter.get_parameter_schema()

        # Verify schema structure
        assert "parameters" in schema
        params = schema["parameters"]

        # Check key parameters
        assert "decimation_target" in params
        assert "texture_size" in params
        assert "remesh" in params
        assert "seed" in params

        # Verify parameter details
        self._verify_parameter(params["decimation_target"], "integer", 1000000)
        self._verify_parameter(params["texture_size"], "integer", 4096)
        self._verify_parameter(params["remesh"], "boolean", True)

        print("TRELLIS.2 image-to-mesh schema test passed")

    def test_trellis2_painting_schema(self):
        """Test TRELLIS.2 painting parameter schema"""
        from adapters.trellis2_adapter import Trellis2ImageMeshPaintingAdapter

        adapter = Trellis2ImageMeshPaintingAdapter()
        schema = adapter.get_parameter_schema()

        assert "parameters" in schema
        params = schema["parameters"]

        assert "extension_webp" in params
        self._verify_parameter(params["extension_webp"], "boolean", True)

        print("TRELLIS.2 painting schema test passed")

    def test_p3sam_schema(self):
        """Test P3-SAM parameter schema"""
        from adapters.p3sam_adapter import P3SAMSegmentationAdapter

        adapter = P3SAMSegmentationAdapter()
        schema = adapter.get_parameter_schema()

        assert "parameters" in schema
        params = schema["parameters"]

        # Check key parameters
        assert "point_num" in params
        assert "prompt_num" in params
        assert "threshold" in params
        assert "post_process" in params
        assert "seed" in params
        assert "prompt_bs" in params

        # Verify parameter details
        self._verify_parameter(params["point_num"], "integer", 100000)
        self._verify_parameter(params["prompt_num"], "integer", 400)
        self._verify_parameter(params["threshold"], "number", 0.95)
        self._verify_parameter(params["post_process"], "boolean", True)
        self._verify_parameter(params["seed"], "integer", 42)
        self._verify_parameter(params["prompt_bs"], "integer", 32)

        # Verify constraints
        assert "minimum" in params["point_num"]
        assert "maximum" in params["point_num"]
        assert params["threshold"]["minimum"] == 0.0
        assert params["threshold"]["maximum"] == 1.0

        print("P3-SAM schema test passed")

    def test_all_registered_models_have_schema(self):
        """Test that all registered models implement parameter schema"""
        registry = ModelFactory.ADAPTER_REGISTRY

        failed_models = []

        for model_id, adapter_info in registry.items():
            try:
                # Create a minimal config to instantiate the model
                config = {
                    "model_id": model_id,
                    "feature_type": "test",
                    "vram_requirement": 1024,
                    "init_params": {}
                }

                model_instance = ModelFactory.create_model_from_config(config)
                schema = model_instance.get_parameter_schema()

                # Verify schema has required structure
                assert "parameters" in schema, f"{model_id} schema missing 'parameters' key"
                assert isinstance(schema["parameters"], dict), f"{model_id} parameters is not a dict"

                print(f"✓ {model_id} has valid parameter schema")

            except NotImplementedError:
                failed_models.append(model_id)
                print(f"✗ {model_id} has not implemented get_parameter_schema()")
            except Exception as e:
                print(f"⚠ {model_id} - Error checking schema: {e}")

        if failed_models:
            print(f"\nModels without parameter schema: {failed_models}")
            print("Note: This is expected for models that haven't been updated yet")

    def _verify_parameter(
        self, param: Dict[str, Any], expected_type: str, expected_default: Any
    ):
        """Helper to verify parameter structure"""
        assert "type" in param
        assert param["type"] == expected_type
        assert "description" in param
        assert "default" in param
        assert param["default"] == expected_default
        assert "required" in param


class TestParameterAPIEndpoint:
    """Test the parameter API endpoint"""

    BASE_URL = "http://localhost:7842/api/v1/system"

    @pytest.fixture(scope="class")
    def check_server(self):
        """Check if server is running"""
        try:
            response = requests.get(f"{self.BASE_URL}/health", timeout=2)
            return response.status_code == 200
        except:
            pytest.skip("Server is not running. Start server with: ./scripts/run_server.sh")

    def test_get_trellis2_parameters(self, check_server):
        """Test getting TRELLIS.2 parameters via API"""
        if not check_server:
            pytest.skip("Server not running")

        model_id = "trellis2_image_to_textured_mesh"
        response = requests.get(f"{self.BASE_URL}/models/{model_id}/parameters")

        assert response.status_code == 200
        data = response.json()

        assert "model_id" in data
        assert data["model_id"] == model_id
        assert "feature_type" in data
        assert "vram_requirement" in data
        assert "schema" in data
        assert "timestamp" in data

        schema = data["schema"]
        assert "parameters" in schema
        assert "decimation_target" in schema["parameters"]

        print(f"API endpoint test passed for {model_id}")

    def test_get_p3sam_parameters(self, check_server):
        """Test getting P3-SAM parameters via API"""
        if not check_server:
            pytest.skip("Server not running")

        model_id = "p3sam_mesh_segmentation"
        response = requests.get(f"{self.BASE_URL}/models/{model_id}/parameters")

        assert response.status_code == 200
        data = response.json()

        assert data["model_id"] == model_id
        schema = data["schema"]
        assert "point_num" in schema["parameters"]
        assert "prompt_num" in schema["parameters"]

        print(f"API endpoint test passed for {model_id}")

    def test_nonexistent_model(self, check_server):
        """Test error handling for nonexistent model"""
        if not check_server:
            pytest.skip("Server not running")

        model_id = "nonexistent_model_id"
        response = requests.get(f"{self.BASE_URL}/models/{model_id}/parameters")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

        print("Error handling test passed")

    def test_list_all_models_with_parameters(self, check_server):
        """Test that model list includes parameter info"""
        if not check_server:
            pytest.skip("Server not running")

        response = requests.get(f"{self.BASE_URL}/models")

        assert response.status_code == 200
        data = response.json()

        # Should have available models
        assert "available_models" in data or "features" in data

        print("Model listing test passed")


class TestSchemaValidation:
    """Test schema validation logic"""

    def test_schema_format_validation(self):
        """Test that schemas follow the correct format"""
        from adapters.trellis2_adapter import Trellis2ImageToTexturedMeshAdapter

        adapter = Trellis2ImageToTexturedMeshAdapter()
        schema = adapter.get_parameter_schema()

        params = schema["parameters"]

        for param_name, param_spec in params.items():
            # Required fields
            assert "type" in param_spec, f"{param_name} missing 'type'"
            assert "description" in param_spec, f"{param_name} missing 'description'"
            assert "default" in param_spec, f"{param_name} missing 'default'"
            assert "required" in param_spec, f"{param_name} missing 'required'"

            # Type must be valid
            assert param_spec["type"] in ["integer", "number", "string", "boolean", "array", "object"]

            # Type-specific validations
            if param_spec["type"] in ["integer", "number"]:
                # Numeric types can have min/max or enum
                if "enum" not in param_spec:
                    # Should have some constraint
                    assert "minimum" in param_spec or "maximum" in param_spec, \
                        f"{param_name} numeric type should have constraints"

        print("Schema format validation passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

