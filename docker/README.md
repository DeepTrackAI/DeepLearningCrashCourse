# Deep Learning Crash Course

[![Early Access - Use Code PREORDER for 25% Off](https://img.shields.io/badge/Early%20Access%20Now%20Available-Use%20Code%20PREORDER%20for%2025%25%20Off-orange)](https://nostarch.com/deep-learning-crash-course)  
by Benjamin Midtvedt, Jesús Pineda, Henrik Klein Moberg, Harshith Bachimanchi, Joana B. Pereira, Carlo Manzo, Giovanni Volpe  
No Starch Press, San Francisco (CA), 2025  
ISBN-13: 9781718503922  
[https://nostarch.com/deep-learning-crash-course](https://nostarch.com/deep-learning-crash-course)

---

# Deep Learning Crash Course Docker Image

![Docker Image CI](https://github.com/DeepTrackAI/DeepLearningCrashCourse/actions/workflows/docker-publish.yml/badge.svg)

A ready-to-run JupyterLab environment with all notebooks and dependencies baked in.  
Works on Intel & Apple-Silicon Macs, Linux ×86_64 & ARM64; also provides an NVIDIA-CUDA-enabled variant for GPU hosts.

---

## Prerequisites

- **Docker**  
  - macOS / Windows → [Docker Desktop](https://www.docker.com/products/docker-desktop)  
  - Linux → Docker Engine + (optional) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)  
- (Optional) **VS Code** + [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

---

## Quick Start

### 1. Pull the image

**CPU-only (multi-arch)**  
```bash
docker pull ghcr.io/deeptrackai/deep-learning-crash-course:latest
```

**GPU-enabled (amd64 + CUDA)**
```bash
docker pull ghcr.io/deeptrackai/deep-learning-crash-course-gpu:latest
```

### 2. Start JupyterLab

**CPU-only (multi-arch)**  
```bash
docker run --rm -it \
  -p 8888:8888 \
  ghcr.io/deeptrackai/deep-learning-crash-course:latest
```

**GPU-enabled (amd64 + CUDA)**
```bash
docker run --rm -it --gpus all \
  -p 8888:8888 \
  ghcr.io/deeptrackai/deep-learning-crash-course-gpu:latest
```
### 3. Run in JupyterLab ...

After startup, copy the URL with token (e.g., http://127.0.0.1:8888/lab?token=…) into your browser to access JupyterLab.

### ... or attach in VS Code (Dev Containers)

   1. In VS Code, open Command Palette (`Ctrl+Shift+P`).
   
   2. Run **Dev Containers: Attach to Running Container...**

   3. Select your **CPU** or **GPU** container from the list. A new VS Code window will pop up.

   4. Install Python & Jupyter extensions when prompted.

   5. **Open Folder** → `/home/jovyan/work` and  **Select Kernel** → `/opt/conda/bin/python` (Python 3.11).

   6. Open any `.ipynb` and run cells. If `ipywidgets` fails, Reload window.