---
title: "Building Docker Containers for GPU Applications" 
# date: 2023-09-07
tags: ["Docker","GPU"]
author: ["Guillermo Alfaro"]
description: "Description on some hacks to build a Docker containers that use GPU "
summary: "Building Docker containers can be tricky, and let alone if they use GPUs. However, there are some 'hacks' or techinques that I have found to be useful and I would lime to share with y'all to save you some time if you face these challenges."
cover:
    image: "docker.png"
    alt: "Cute Docker Container"
    relative: True
editPost:
    # URL: "https://github.com/pmichaillat/squareroot-uv"
    # Text: "GitHub repository"
# showToc: true
# disableAnchoredHeadings: false

---

## Overview

Docker is a great tool for isolating pieces of code or a whole app. You can think of it as a super tiny but efficient virtual machine that has the purpose a running a specific piece of code. Many Docker containers start by downloading a base image and start building of top of it. Similar to downloading an operative system and adding the programs and application you want to use. 

In AI development, building Docker containers plays a fundamental role in production. In practice, it saves us a lot of work having *Docker containers* that can readily run our code instead of running settings and `pip install -r requirements.txt` every single time, and on top of that, Docker containers are production-friendly!

It gets a little bit tricky when our code utilizes GPU. Since we are a building an isolated container that can run in any computer with any specifications, we try to generalize many pieces of code. What I mean by this is that the our code should be able to work in any machine, and thus, we have to build our code as machine-agnostic as possible. If you have worked with applications that use CUDA, more often than not, we have to specify versions, directories, etc., so that our code can run. CUDA and cuDNN can be hard to get used to at first. So, I decided to do a hands-on example of building a Docker container where CUDA is needed.  

---

## Task

We will containerize a really nice library used for inference of LLMs: [vLLM](https://github.com/vllm-project/vllm)
If you are not familiarized with this library, tldr: It is a library that leverages [paged attention](https://arxiv.org/pdf/2309.06180.pdf) for high throughput by strategically allocating KV caching in memory.

---

## Containerization

The first step for containerizng vLLM is knowing the requirements needed to run it. Upon close inspection in the library we see that we need CUDA $18.1$ or CUDA $12.1$. I'd suggest we go with CUDA $12.1$ as vLLM is more optimized in that environment as well as any $8.$something version of cuDNN compatible with CUDA. Lucky for us, NVIDIA already has an image for us to download.

```Docker
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 
```

Next is setting the CUDA environment variables so that we can use the GPU in an isolated space. 

```Docker
ENV DEBIAN_FRONTEND noninteractive
ENV FORCE_CUDA="1"
```
Here, `ENV DEBIAN_FRONTEND noninteractive` allows us to not have to deal with interactives prompts such as setting a geographical location or accepting clauses of any library. `ENV FORCE_CUDA="1"` is used to skip checking CUDA in the machine where docker is being built. It implies that CUDA support is available.

Then we install the basics needed to download the requirements such as python, git, ninja, etc., and since we do not know which GPU will utilize, we select a list of possible architectures. Here you can see why [those numbers.](https://stackoverflow.com/questions/68496906/pytorch-installation-for-different-cuda-architectures)

```Docker
RUN apt-get update && \
apt full-upgrade -y && \
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev python3-venv python-is-python3 git curl wget sudo ninja-build && \
rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
```

Next step is copying the needed directories from the repo and installing the libraries (we assume we are building the Dockerfile inside the repo).

```Docker
WORKDIR /app

COPY . /app/vllm

WORKDIR /app/vllm

RUN python -m venv --upgrade-deps /app/vllm/venv && \
   . /app/vllm/venv/bin/activate && \
   pip install --upgrade pip && \
   pip install -r requirements.txt && \
   pip install transformers && \
   pip install accelerate fschat setuptools numpy scipy --upgrade && \
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 --upgrade

RUN python -m venv --upgrade-deps /app/vllm/venv && \
   . /app/vllm/venv/bin/activate && \
   pip install wheel && \
   pip install vllm

RUN . /app/vllm/venv/bin/activate && pip install -e .
```

Finally, we allow the container to interact with outside directories. Essentially, it is used to set up a persistent and shareable data storage space specifically for HuggingFace 🤗 cache within the Docker container.

```Docker
VOLUME [ "/root/.cache/huggingface" ]
```


---
## Dockerfile

```Docker
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 

ENV DEBIAN_FRONTEND noninteractive
ENV FORCE_CUDA="1"

RUN apt-get update && \
apt full-upgrade -y && \
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev python3-venv python-is-python3 git curl wget sudo ninja-build && \
rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

WORKDIR /app

COPY . /app/vllm

WORKDIR /app/vllm

RUN python -m venv --upgrade-deps /app/vllm/venv && \
   . /app/vllm/venv/bin/activate && \
   pip install --upgrade pip && \
   pip install -r requirements.txt && \
   pip install transformers && \
   pip install accelerate fschat setuptools numpy scipy --upgrade && \
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 --upgrade

RUN python -m venv --upgrade-deps /app/vllm/venv && \
   . /app/vllm/venv/bin/activate && \
   pip install wheel && \
   pip install vllm

RUN . /app/vllm/venv/bin/activate && pip install -e .

VOLUME [ "/root/.cache/huggingface" ]

```

Run it:
```bash
sudo docker build -t <tag> -f Dockerfile .
```
---

## Docker Compose

The very last step is using docker compose for deploying this Dockerfile.

```yaml
version: "3.8"

services:
vllm-1:
   image: <tag>
   ports:
     - "8000:8000"
   ipc: host
   ulimits:
     memlock: -1
     stack: 67108864
   volumes:
     - ./volumes/cache/huggingface/:/root/.cache/huggingface/
   command: bash -c ". /app/vllm/venv/bin/activate && cd vllm && python -m vllm.entrypoints.openai.api_server --model TheBloke/llama-2-7B-Guanaco-QLoRA-AWQ --tokenizer hf-internal-testing/llama-tokenizer --quantization awq --dtype half --max-model-len 2048 --disable-log-requests" 
   # mistral 7b awq
   # command: bash -c ". /app/vllm/venv/bin/activate && cd vllm && python -m vllm.entrypoints.openai.api_server --model TheBloke/Mistral-7B-v0.1-AWQ --quantization awq --dtype half --max-model-len 8096 --disable-log-requests" 
   stdin_open: true
   tty: true
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             device_ids: [ '0' ]
             capabilities: [ gpu ]
```

---

## Conclusion

Docker is a powerful tool for AI in production. Knowing the right techniques for its usage goes a long way in simplifying how we run applications and avoid redundancy.


