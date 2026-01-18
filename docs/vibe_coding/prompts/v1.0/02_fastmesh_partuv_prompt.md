You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase.

I need you to integrate two additional features.
(1) Mesh Retopology 
(2) Mesh UV unwraping 

For each feature, you need to first create specific routers under @routers, and then create data structures (base models) under @models. Note that the implementation should roughly follow the design of current routers and models.

Then, for each feature, I need you to integrate one specification, FastMesh @FastMesh for mesh re-topology and @PartUV for uv unwarping.

For the integration, you should first read corresponding 3rdparpty codebase @FastMesh @PartUV and corresponding demo scripts @demo.sh @demo.sh. Then, create inference-friendly wrappers for each codebase, just like @partpacker_utils.py etc. Then, implement adapters under @adapters.

Finally, ensure that the overall design is correct and follow the best practice of current codebase. Create necessary unit tests under @test_adapters.

You are an expert, thus make good plans before you begin. Don't create too many extra dummy markdown files or test scripts.