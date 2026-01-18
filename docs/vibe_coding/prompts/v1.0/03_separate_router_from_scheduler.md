### Separate the router layer with the scheduler layer 
You are an expert python backend engineer.
In the current codebase I am developing a scalable 3DAIGC backend service based on Python and FastAPI, the main entry point is @api/main.py , with the GPU scheduler @thirdparty/UniRig/src/model/pointcept/utils/scheduler.py @core/scheduler/scheduler_factory.py@core/scheduler/multiprocess_scheduler.py . I have some architectural concerns about the concurrency. 
As you can see in @api/main.py , If I start multiple threads serving the FastAPI uvicorn, each thread will launch/create a new scheduler, and multiple schedulers will conflict. 
Is there any possibility for different uvicorn workers to use the SAME scheduler or any other solutions to solve this? Maybe separate the uvicorn layer from the scheduler layer (e.g. make the scheduler a service?)

