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
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8888
CMD ["start-notebook.sh"]