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
COPY .. .

# ---------- extra Python deps -----------------------------------------------------
# CPU wheels only
RUN pip install --no-cache-dir \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cpu

COPY docker/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8888
CMD ["start-notebook.sh"]