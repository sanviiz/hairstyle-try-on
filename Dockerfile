# Dockerfile for deployment
# Stage 1: Builder/Compiler
FROM python:3.8-slim as builder
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc
COPY requirements.txt /requirements.txt

RUN python3.8 -m pip install --upgrade pip
RUN pip install --default-timeout=100 -r /requirements.txt

# # Stage 2: Runtime
# FROM nvidia/cuda:10.2-base-ubuntu18.04

# RUN apt update && \
#     apt install --no-install-recommends -y build-essential software-properties-common && \
#     add-apt-repository -y ppa:deadsnakes/ppa && \
#     apt install --no-install-recommends -y python3.8 python3-distutils && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 && \
#     apt clean && rm -rf /var/lib/apt/lists/*
# COPY --from=builder /root/.local/lib/python3.8/site-packages /usr/local/lib/python3.8/dist-packages
# COPY . /workspace
# CMD [ 'streamlit', 'run', '/workspace/app.py' ]
# EXPOSE 8501