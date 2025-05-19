# ---------------------------------------------------------------------------------
# Universal multi-arch image (works on both Apple Silicon & Intel)
# ---------------------------------------------------------------------------------
FROM quay.io/jupyter/datascience-notebook:python-3.11
# ---------- system utilities ------------------------------------------------------
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
USER ${NB_UID}

# ---------- copy the course -------------------------------------------------------
WORKDIR /home/jovyan/work
RUN git clone --depth 1 https://github.com/DeepTrackAI/DeepLearningCrashCourse.git .

# ---------- ensure correct ownership ---------------------------------------------
    USER root
    RUN chown -R ${NB_UID}:${NB_GID} /home/jovyan/work
    USER ${NB_UID}

# ---------- extra Python deps -----------------------------------------------------
# CPU-only PyTorch (wheels from CPU index)
RUN pip install --no-cache-dir \
        torch==2.7.0 torchvision==0.22.0\
        --index-url https://download.pytorch.org/whl/cpu

# CPU-only PyG (and its C/C++ extensions)
RUN pip install --no-cache-dir \
        torch-geometric \
        -f https://data.pyg.org/whl/torch-2.7.0+cpu.html

COPY ./requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Enable Jupyter widget support
RUN pip install --no-cache-dir ipywidgets

# For JupyterLab (inside the container), install the widget manager
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager \
  --no-build && \
  jupyter lab build

EXPOSE 8888
CMD ["start-notebook.sh"]