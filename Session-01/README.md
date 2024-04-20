# MLOPS - 1

## Assignment

1. Create Dockerfile that uses https://github.com/rwightman/pytorch-image-modelsLinks to an external site.
2. Build the image for this
3. Create an Inference Python Script that takes a model name and image path/url and outputs json like

```
{"predicted": "dog", "confidence": "0.89"}
```

4. MODEL and IMAGE must be configurable while inferencing
5. Model Inference will be done like:

```
docker run $IMAGE_NAME --model $MODEL --image $IMAGE
```

6. Push the Image to Docker Hub
7. Try to bring the docker image size as less as possible (maybe try out slim/alpine images?) (use architecture and linux specific CPU wheel from here https://download.pytorch.org/whl/torch_stable.htmlLinks to an external site.)
8. Pull from DockerHub and run on Play With Docker to verify yourself
9. Submit the Dockerfile contents and your complete image name with tag that was uploaded to DockerHub, also the link to the github classroom repository
10. Tests can be run with

```
bash ./tests/all_tests.sh
```

# Image Sizes

```
REPOSITORY                   TAG       IMAGE ID       CREATED          SIZE

mlops-01-small              latest    b94a6856361a   10 minutes ago   817MB
mlops-01-medium             latest    2ffc04e387f4   25 minutes ago   864MB
mlops-01-huge               latest    16d4b8ee71a0   31 minutes ago   1.09GB

```
