FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /workspace

# TensorFlow 2.17 comes with the base image, so requirements.txt only fills in the rest.
# The protobuf constraint stops pip from upgrading past what this TF build supports.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "protobuf<5.0.0dev,>=3.20.3"

# The repo is bind-mounted at run time (docker run -v "$(pwd):/workspace"), so host edits
# show up without rebuilding the image.
