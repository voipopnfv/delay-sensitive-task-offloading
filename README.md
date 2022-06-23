# Delay Sensitive Task Offloading

## Build Images
```
% docker build -t visual-navigation-img -f Dockerfile-visualNav . --no-cache
% docker build -t object-detect-img -f Dockerfile-objectDetect . --no-cache
% docker build -t agent-img -f Dockerfile-agent . --no-cache
```

## Run Containers
```
% docker run -d --gpus all -v /tmp/offloading:/tmp/offloading -p 50051:50051 --name visual-navigation-container visual-navigation-img:latest
% docker run -d --gpus all -v /tmp/offloading:/tmp/offloading -p 50052:50052 --name object-detect-container object-detect-img:latest
% docker run -d -v /tmp/offloading:/tmp/offloading --net=host --name agent-container agent-img:latest
```

## Use `docker compose`
Build images and run containers within one command. The version of docker compose is v2.6.
```
% docker compose up -d
```
To stop all the running container and remove them, use command `docker compose down` in this folder. This command won't remove the images.
