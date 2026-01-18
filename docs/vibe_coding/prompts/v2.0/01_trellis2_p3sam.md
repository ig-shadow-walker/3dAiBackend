You are an expert at Python, AI inference and system design. Now in the current codebase I am developing a backend server for 3D generative models. You can check the framework and system architecture in @README.md  and @assets/system_arch.drawio  .  
The backend service supports a number of different 3DAIGC algorithms(e.g. mesh generation, mesh segmentation, etc.).  Now I need you to implement two other modules,  @thirdparty/TRELLIS.2  for mesh generation and mesh painting. @thirdparty/Hunyuan3D-Part/P3-SAM  for mesh segmentation.

General Guidelines and Steps for the integration of new Modules
1. Wrap the inference example into a inference-friendly class, this helper class should be put under @utils 
2. Implement specific adapters under @adapters  according to the task type of the target model (e.g. mesh generation)
3. Register the model in the default configuration and scheduler factory
4. Implement specific test scripts under @tests/test_adapters  
5. Add the installation steps of the target model in @scripts/install.sh  and @scripts/install.bat  . If there are any models that can be pre-downloaded, also integrate the model downloading logics in @scripts/download_models.sh  and @scripts/download_models.bat 

Additional Details
TRELLIS.2:
You can refer to @thirdparty/TRELLIS.2/example.py  for mesh generation, and @thirdparty/TRELLIS.2/example_texturing.py  for mesh painting. 
Notice that besides the general guidelines for integration, you should also use TRELLIS.2 (replacing TRELLIS) as the base environment 

P3SAM: 
You can refer to @thirdparty/Hunyuan3D-Part/P3-SAM/demo/auto_mask.py  for an inference example. 

Besides, I have noticed that different model(e.g. trellis vs trellis.2) can have different input parameters. I need you to expose such model-specific parameters to the endpoint routers. I think you should design proper ways for the client to get, for each model, the model-specific parameters and the parameter type.

Make good plans before you begin, and don't create too many complicated readme files (either in txt or markdown), if necessary, carefully organize them under @docs/vibe_coding/cursor_created  