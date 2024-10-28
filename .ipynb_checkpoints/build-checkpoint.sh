docker build . -t llm_env --no-cache
docker run --name llm_env_container -it -v $(pwd):/mnt --gpus all llm_env:latest