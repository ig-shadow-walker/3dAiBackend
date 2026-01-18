You are an expert at Python, AI inference and system design. Now in the current codebase I am developing a backend server for 3D generative models. You can check the framework and system architecture in @README.md  and @assets/system_arch.drawio  .  
The backend service supports a number of different 3DAIGC algorithms(e.g. mesh generation, mesh segmentation, etc.).  Now I need you to implement two other modules,  @thirdparty/UltraShape for raw mesh generation. @voxhammer for mesh local editing. Notice that ultrashape belongs to the feature of mesh generation, while the local mesh editing is a new feature.

## General Guidelines and Steps for the integration of new Modules
1. Wrap the inference example into a inference-friendly class, this helper class should be put under @utils. Notice that it's NOT necessary to implement this helper class. If the reference code(class) is already clear enough, you can just use it in later steps.
2. Implement specific adapters under @adapters  according to the task type of the target model (e.g. mesh generation)
3. Register the model in the default configuration and scheduler factory
4. Implement specific test scripts under @tests/test_adapters. Notice that the adapters should be simple and include minimal important tests.
5. Add the installation steps of the target model in @scripts/install.sh  and @scripts/install.bat. If there are any models that can be pre-downloaded, also integrate the model downloading logics in @scripts/download_models.sh  and @scripts/download_models.bat 

## Additional Details:
### UltraShape
You can refer to @thirdparty/UltraShape/scripts/run.sh for an inference example. Notice that ultrashape requires a coarse mesh from Hunyuan3D-2.1 as the input, which you can refer to @thirdparty/Hunyuan3D-2.1/demo.py to further integrate it as a single pipeline.

### VoxHammer
VoxHammer is a model specification of local 3D editing, thus you should first implement basic framework for local 3D editing (including @routers, basic request models @models etc.). Generally this feature accepts a mesh file and a mask as input. The client will provide 2 types of masks (i.) 3D center and dimensions of a bounding box (ii.) 3D center and 3D radius of an ellipsoid, both cases you should first create a 3D mask on the fly, which adapts the input of VoxHammer inference @inference.py
You can refer to @inference.py for an inference example.

### Note
Make good plans before you begin, and don't create too many complicated readme files (either in txt or markdown), if necessary, carefully organize them under @docs/vibe_coding/cursor_created  