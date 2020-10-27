## === Stage 1 --- Build dlib wheel ===
FROM python:3.8

## Build 'dlib' separately - it takes a long time and needs a lot of space
RUN apt-get update && apt-get install -y build-essential cmake
RUN mkdir -p /tmp/dlib
RUN pip wheel --use-pep517 --wheel-dir /tmp/dlib dlib
RUN ln /tmp/dlib/dlib-*.whl /tmp/dlib.whl


## === Stage 2 --- Build the actual container ===
FROM python:3.8

WORKDIR /app
EXPOSE 5000

# Use 'dlib.whl' from the above container
COPY --from=0 /tmp/dlib /tmp/dlib
RUN pip install /tmp/dlib/dlib-*.whl
RUN rm -rf /tmp/dlib

# libGL1 is required for opencv-python
RUN apt-get update && apt-get install -y libgl1 && apt-get clean

# Download YOLOv4 model files (also change mlapiconfig.docker.ini
# if you need some other models)
COPY get_models.sh .
RUN INSTALL_YOLOV3=no INSTALL_TINYYOLOV3=no INSTALL_YOLOV4=yes INSTALL_TINYYOLOV4=no ./get_models.sh

# Prevent re-installing if source has changed but not requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt && rm -rf /root/.cache

# Copy all the files
COPY . .
RUN rm -rf .git

# Use Docker config customised for Docker
RUN cp mlapiconfig.docker.ini mlapiconfig.ini

CMD [ "python3", "./mlapi.py", "-c", "mlapiconfig.ini" ]
