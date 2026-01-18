# UltraShape & VoxHammer Integration Documentation

This document describes the integration of UltraShape and VoxHammer into the 3DAIGC-API backend system.

## Overview

### UltraShape
**Purpose**: High-quality mesh generation via refinement  
**Type**: Extends `image_to_raw_mesh` feature  
**Pipeline**: Hunyuan3D-2.1 (coarse) → UltraShape (refinement)  
**VRAM**: 20GB (8GB + 12GB)  

### VoxHammer
**Purpose**: Local mesh editing with text/image guidance  
**Type**: New `mesh_editing` feature  
**Pipeline**: 3D Rendering → Feature Extraction → Voxel Masking → TRELLIS Editing  
**VRAM**: 26GB  

## Architecture

### UltraShape Integration
```
User Request (Image)
    ↓
API Endpoint (/api/v1/mesh-generation/image-to-raw-mesh)
    ↓
Job Scheduler
    ↓
UltraShapeAdapter
    ↓
UltraShapeHelper
    ├─→ Hunyuan3D-2.1 (coarse mesh)
    └─→ UltraShape Pipeline (refinement)
    ↓
Refined Mesh Output
```

### VoxHammer Integration
```
User Request (Mesh + Mask + Text/Images)
    ↓
API Endpoint (/api/v1/mesh-editing/text-mesh-editing or image-mesh-editing)
    ↓
Job Scheduler
    ↓
VoxHammerAdapter
    ├─→ MaskGenerator (create 3D mask from parameters)
    ├─→ VoxHammerHelper
    │   ├─→ Blender Rendering (multi-view)
    │   ├─→ DINOv2 Feature Extraction
    │   ├─→ Voxel Masking
    │   └─→ TRELLIS Editing
    └─→ Edited Mesh Output
```

## Implementation Details

### File Structure
```
3DAIGC-API/
├── adapters/
│   ├── ultrashape_adapter.py          # UltraShape mesh refinement
│   └── voxhammer_adapter.py           # VoxHammer text/image editing
├── core/
│   ├── models/
│   │   └── mesh_editing_models.py     # Base classes for editing
│   └── utils/
│       ├── ultrashape_pipeline_helper.py   # UltraShape wrapper
│       ├── voxhammer_pipeline_helper.py    # VoxHammer wrapper
│       └── mask_generator.py               # 3D mask creation
├── api/
│   └── routers/
│       └── mesh_editing.py            # Mesh editing REST endpoints
└── tests/
    └── test_adapters/
        ├── test_ultrashape_adapter.py
        └── test_voxhammer_adapter.py
```

### Key Design Decisions

1. **UltraShape Integration Approach**: Chose integrated pipeline (single API call) where UltraShape adapter internally calls Hunyuan3D-2.1 to generate coarse mesh, then refines it. This provides better user experience than requiring two separate API calls.

2. **VoxHammer Mask Generation**: Implemented procedural mask generation from geometric parameters (bounding box or ellipsoid). The API accepts center coordinates and dimensions/radii, then generates mask GLB on-the-fly. This is simpler than requiring users to provide pre-made mask files.

3. **TRELLIS Reuse**: VoxHammer adapters reuse existing TRELLIS pipeline instances (text or image mode) from the adapter registry, avoiding duplicate model loading.

4. **Feature Separation**: Created separate features for text (`text_mesh_editing`) and image (`image_mesh_editing`) guided editing to allow independent scheduling and VRAM management.

## API Usage Examples

### UltraShape - Refined Mesh Generation

```bash
# 1. Upload image file
curl -X POST "http://localhost:7842/api/v1/file-upload/image" \
  -F "file=@/path/to/your/image.jpg"
# Response: {"file_id": "abc123def456", ...}

# 2. Generate refined mesh using UltraShape
curl -X POST "http://localhost:7842/api/v1/mesh-generation/image-to-raw-mesh" \
  -H "Content-Type: application/json" \
  -d '{
    "image_file_id": "abc123def456",
    "output_format": "glb",
    "model_preference": "ultrashape_image_to_raw_mesh",
    "num_inference_steps": 50,
    "num_latents": 32768,
    "octree_res": 1024
  }'
# Response: {"job_id": "job_123", "status": "queued", ...}

# 3. Check job status and download
curl "http://localhost:7842/api/v1/system/jobs/job_123"
curl "http://localhost:7842/api/v1/system/jobs/job_123/download" -o refined_mesh.glb
```

### VoxHammer - Text-Guided Mesh Editing

```bash
# 1. Upload mesh file
curl -X POST "http://localhost:7842/api/v1/file-upload/mesh" \
  -F "file=@/path/to/mesh.glb"
# Response: {"file_id": "mesh_xyz789", ...}

# 2. Edit mesh with text guidance and bounding box mask
curl -X POST "http://localhost:7842/api/v1/mesh-editing/text-mesh-editing" \
  -H "Content-Type: application/json" \
  -d '{
    "mesh_file_id": "mesh_xyz789",
    "mask_bbox": {
      "center": [0.0, 0.5, 0.0],
      "dimensions": [0.3, 0.3, 0.3]
    },
    "source_prompt": "dragon head",
    "target_prompt": "dragon head with horns",
    "num_views": 150,
    "resolution": 512,
    "model_preference": "voxhammer_text_mesh_editing"
  }'
# Response: {"job_id": "edit_job_456", ...}

# Alternative: Use ellipsoid mask
curl -X POST "http://localhost:7842/api/v1/mesh-editing/text-mesh-editing" \
  -H "Content-Type: application/json" \
  -d '{
    "mesh_file_id": "mesh_xyz789",
    "mask_ellipsoid": {
      "center": [0.0, 0.3, 0.0],
      "radii": [0.2, 0.2, 0.2]
    },
    "source_prompt": "smooth surface",
    "target_prompt": "textured surface",
    "model_preference": "voxhammer_text_mesh_editing"
  }'

# 3. Download edited mesh
curl "http://localhost:7842/api/v1/system/jobs/edit_job_456/download" -o edited_mesh.glb
```

### VoxHammer - Image-Guided Mesh Editing

```bash
# 1. Upload mesh and images
curl -X POST "http://localhost:7842/api/v1/file-upload/mesh" -F "file=@mesh.glb"
# Response: {"file_id": "mesh_abc", ...}

curl -X POST "http://localhost:7842/api/v1/file-upload/image" -F "file=@source.png"
# Response: {"file_id": "src_img", ...}

curl -X POST "http://localhost:7842/api/v1/file-upload/image" -F "file=@target.png"
# Response: {"file_id": "tgt_img", ...}

curl -X POST "http://localhost:7842/api/v1/file-upload/image" -F "file=@mask.png"
# Response: {"file_id": "mask_img", ...}

# 2. Edit mesh with image guidance
curl -X POST "http://localhost:7842/api/v1/mesh-editing/image-mesh-editing" \
  -H "Content-Type: application/json" \
  -d '{
    "mesh_file_id": "mesh_abc",
    "source_image_file_id": "src_img",
    "target_image_file_id": "tgt_img",
    "mask_image_file_id": "mask_img",
    "mask_bbox": {
      "center": [0.0, 0.5, 0.0],
      "dimensions": [0.4, 0.4, 0.4]
    },
    "num_views": 150,
    "resolution": 512,
    "model_preference": "voxhammer_image_mesh_editing"
  }'
```

## Configuration

### UltraShape Configuration
```yaml
# config/models.yaml
image_to_raw_mesh:
  ultrashape_image_to_raw_mesh:
    vram_requirement: 20480  # 20GB
    supported_inputs: ["image"]
    supported_outputs: ["glb", "obj", "ply"]
    model_path: "thirdparty/UltraShape"
    enabled: true
    max_workers: 1
```

### VoxHammer Configuration
```yaml
# config/models.yaml
text_mesh_editing:
  voxhammer_text_mesh_editing:
    vram_requirement: 26624  # 26GB
    supported_inputs: ["mesh", "text"]
    supported_outputs: ["glb"]
    model_path: "thirdparty/VoxHammer"
    enabled: true
    max_workers: 1

image_mesh_editing:
  voxhammer_image_mesh_editing:
    vram_requirement: 26624  # 26GB
    supported_inputs: ["mesh", "image"]
    supported_outputs: ["glb"]
    model_path: "thirdparty/VoxHammer"
    enabled: true
    max_workers: 1
```

## Parameters Reference

### UltraShape Parameters
- `num_inference_steps` (default: 50): Number of diffusion steps
- `num_latents` (default: 32768): Number of latent tokens
- `octree_res` (default: 1024): Marching cubes resolution
- `chunk_size` (default: 8000): Chunk size for inference
- `scale` (default: 0.99): Mesh normalization scale
- `seed` (default: 42): Random seed

### VoxHammer Parameters
- `num_views` (default: 150, range: 50-300): Number of rendering views
- `resolution` (default: 512, range: 256-1024): Rendering resolution
- `mask_type`: "bbox" or "ellipsoid"
- `mask_center`: [x, y, z] coordinates
- `mask_params`: For bbox: [width, height, depth]; for ellipsoid: [rx, ry, rz]
- `source_prompt`/`target_prompt`: Text descriptions (text mode)
- Source/target/mask images (image mode)

## Known Limitations

### UltraShape
1. **Sequential Processing**: Cannot run simultaneously with standalone Hunyuan3D-2.1 jobs due to shared model components
2. **Memory Requirements**: Requires 20GB VRAM, may need GPU offloading on smaller GPUs
3. **Processing Time**: Slower than direct Hunyuan3D-2.1 due to refinement step (typical: 2-3 minutes per mesh)

### VoxHammer
1. **Rendering Overhead**: Blender-based rendering adds significant processing time
2. **Intermediate Files**: Generates many temporary files (renders, features, voxels) - automatic cleanup implemented
3. **Mask Complexity**: Simple geometric masks (bbox/ellipsoid) only - no arbitrary mesh masks yet
4. **TRELLIS Dependency**: Requires TRELLIS models to be available and loaded

## Troubleshooting

### UltraShape Issues

**Problem**: "Failed to load UltraShape checkpoint"
- **Solution**: Download checkpoint manually from UltraShape repository to `pretrained/UltraShape/ultrashape_v1.pt`

**Problem**: "Out of memory" during refinement
- **Solution**: Reduce `num_latents` (try 16384) or `octree_res` (try 512), or enable CPU offloading

**Problem**: "Coarse mesh generation failed"
- **Solution**: Ensure Hunyuan3D-2.1 models are properly installed and accessible

### VoxHammer Issues

**Problem**: "Blender rendering failed"
- **Solution**: Ensure Blender/bpy is installed correctly (required for rendering)

**Problem**: "Invalid mask parameters"
- **Solution**: Verify mask center/params are lists of 3 float values, all positive for dimensions/radii

**Problem**: "DINOv2 features extraction failed"
- **Solution**: Ensure DINOv2 model is downloaded (`pretrained/dinov2-giant/`)

**Problem**: "TRELLIS pipeline not loaded"
- **Solution**: Ensure TRELLIS models are installed and enabled in config

## Testing

```bash
# Test UltraShape adapter
PYTHONPATH=. pytest tests/test_adapters/test_ultrashape_adapter.py -v -s -r s

# Test VoxHammer adapters
PYTHONPATH=. pytest tests/test_adapters/test_voxhammer_adapter.py -v -s -r s

# Test via API (after starting server)
python tests/run_test_client.py --server-url http://localhost:7842
```

## Performance Notes

### UltraShape
- Typical processing time: 120-180 seconds per image
- Peak VRAM usage: ~18GB (can vary based on parameters)
- Recommended for: High-quality mesh generation where detail matters

### VoxHammer
- Typical processing time: 300-600 seconds per edit
- Peak VRAM usage: ~13GB
- Recommended for: Targeted local edits, preserving most of original mesh

## Future Enhancements

1. **UltraShape**: Add batch processing support, expose more refinement parameters
2. **VoxHammer**: Support arbitrary mesh masks, add region detection/suggestion
3. **Both**: Optimize temporary file management, add preview/validation endpoints

