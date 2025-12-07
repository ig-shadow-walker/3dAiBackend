@api/routers/system.py @api/routers/system.py:729-816
I can see that  an example status of a finished job is
``` text
{
  "job_id": "d53cef2a-d6b9-4a7a-8565-9cfc7baa9d14",
  "feature": "image_to_textured_mesh",
  "inputs": {
    "image_path": "uploads/image/91/upload_22ff1918-da70-4991-a1e6-6a0245f2b836.png",
    "texture_image_path": null,
    "output_format": "glb",
    "texture_resolution": 1024
  },
  "model_preference": "trellis_image_to_textured_mesh",
  "priority": 1,
  "status": "completed",
  "created_at": "2025-12-02T15:11:54.655500",
  "metadata": {
    "feature_type": "image_to_textured_mesh"
  },
  "user_id": "user_dd253d7fe66009af",
  "model_id": "trellis_image_to_textured_mesh",
  "started_at": "2025-12-02T15:13:50.408413",
  "completed_at": "2025-12-02T15:15:05.403650",
  "result": {
    "output_mesh_path": "/mnt/afs/project/3DAIGC-API/outputs/meshes/trellis_upload_22ff1918-da70-4991-a1e6-6a0245f2b836_1764688504.glb",
    "success": true,
    "thumbnail_path": "/mnt/afs/project/3DAIGC-API/outputs/thumbnails/trellis_upload_22ff1918-da70-4991-a1e6-6a0245f2b836_1764688504_thumb.png",
    "generation_info": {
      "model": "TRELLIS",
      "image_path": "uploads/image/91/upload_22ff1918-da70-4991-a1e6-6a0245f2b836.png",
      "seed": 42,
      "vertex_count": 9208,
      "face_count": 11604,
      "thumbnail_generated": true
    },
    "mesh_url": "http://localhost:7842/api/v1/system/jobs/d53cef2a-d6b9-4a7a-8565-9cfc7baa9d14/download",
    "mesh_file_info": {
      "filename": "trellis_upload_22ff1918-da70-4991-a1e6-6a0245f2b836_1764688504.glb",
      "file_size_bytes": 2087548,
      "file_size_mb": 1.99,
      "content_type": "model/gltf-binary",
      "file_extension": ".glb"
    },
    "thumbnail_url": "http://localhost:7842/api/v1/system/jobs/d53cef2a-d6b9-4a7a-8565-9cfc7baa9d14/thumbnail",
    "thumbnail_file_info": {
      "filename": "trellis_upload_22ff1918-da70-4991-a1e6-6a0245f2b836_1764688504_thumb.png",
      "file_size_bytes": 19080,
      "file_size_mb": 0.02,
      "content_type": "image/png",
      "file_extension": ".png"
    }
  }
}
``` 
And an example of a processing job is 
``` text
{"job_id":"d53cef2a-d6b9-4a7a-8565-9cfc7baa9d14","feature":"image_to_textured_mesh","inputs":{"image_path":"uploads/image/91/upload_22ff1918-da70-4991-a1e6-6a0245f2b836.png","texture_image_path":null,"output_format":"glb","texture_resolution":1024},"model_preference":"trellis_image_to_textured_mesh","priority":1,"status":"processing","created_at":"2025-12-02T15:11:54.655500","metadata":{"feature_type":"image_to_textured_mesh"},"user_id":"user_dd253d7fe66009af","model_id":"trellis_image_to_textured_mesh","started_at":"2025-12-02T15:13:50.408413"}
```
For the input image, could you also port it to a public url? just as the `mesh_url`?