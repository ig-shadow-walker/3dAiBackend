
General Guidelines and Steps for the integration of new Modules

1. Wrap the inference example into a inference-friendly class, this helper class should be put under @utils. Notice that it's NOT necessary to implement this helper class. If the reference code(class) is already clear enough, you can just use it in later steps.
2. Implement specific adapters under @adapters according to the task type of the target model (e.g. mesh generation).
3. Register the model in the default configuration and scheduler factory.
4. Implement specific test scripts under @test_adapters, notice that the adapters should be simple and include minimal important tests.
5. Add the installation steps of the target model in @install.sh and @install.bat. If there are any models that can be pre-downloaded, also integrate the model downloading logics in @download_models.sh and @download_models.bat .
