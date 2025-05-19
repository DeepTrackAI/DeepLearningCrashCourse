# Deep Learning Crash Course Docker Image

[![Early Access - Use Code PREORDER for 25% Off](https://img.shields.io/badge/Early%20Access%20Now%20Available-Use%20Code%20PREORDER%20for%2025%25%20Off-orange)](https://nostarch.com/deep-learning-crash-course)  
by Benjamin Midtvedt, Jesús Pineda, Henrik Klein Moberg, Harshith Bachimanchi, Joana B. Pereira, Carlo Manzo, Giovanni Volpe  
No Starch Press, San Francisco (CA), 2025  
ISBN-13: 9781718503922  
[https://nostarch.com/deep-learning-crash-course](https://nostarch.com/deep-learning-crash-course)

---

![Docker Image CI](https://github.com/DeepTrackAI/DeepLearningCrashCourse/actions/workflows/docker-publish.yml/badge.svg)

A ready-to-run JupyterLab environment with all notebooks and dependencies baked in.  
Works on Intel & Apple-Silicon Macs, Linux ×86_64 & ARM64.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) (macOS, Windows) or Docker Engine (Linux)  
- (Optional) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/) on Linux for GPU support  
- (Optional) VS Code + [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension


---
## General use

### 1. Pull the latest image

```bash
docker pull ghcr.io/deeptrackai/deep-learning-crash-course:latest
```
Docker will automatically select the correct architecture slice (amd64 or arm64).

### 2. Run JupyterLab
```docker run --rm -it \
  -p 8888:8888 \
  -v "$(pwd)":/home/jovyan/work \
  ghcr.io/deeptrackai/deep-learning-crash-course:latest
  ```
After startup you’ll see a URL with a token (e.g. http://127.0.0.1:8888/lab?token=…). Paste it into your browser.

--- 

## Use in VS Code (Dev Containers)

### 1. Install the “Dev Containers” extension in VS Code.

### 2. Command Palette → Dev Containers: Open Folder in Container…

### 3. Select your local DeepLearningCrashCourse folder.

### 4. In the new window:

   - Install Python & Jupyter extensions when prompted.

   - Open Folder → /home/jovyan/work (the repo inside the container).

   - Select Interpreter in the bottom bar → the one at `/opt/conda/bin/python` (Python 3.11).

### 5. Use the integrated terminal (`Ctrl`+``) or Run & Debug scripts (F5).

### 6. Open any `.ipynb` and run cells.