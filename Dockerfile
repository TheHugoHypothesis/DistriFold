# ============================================================
# DistriFold — Docker Image
# Rede distribuída de K-Fold Cross-Validation com OpenMPI
# ============================================================

FROM python:3.11-slim AS base

# Metadados
LABEL maintainer="DistriFold Team"
LABEL description="Distributed K-Fold Cross-Validation training with OpenMPI"

# Evita prompts interativos na instalação de pacotes
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# 1. Instala dependências de sistema (OpenMPI + build tools)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    openmpi-bin \
    libopenmpi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 2. Cria diretório de trabalho
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# 3. Instala dependências Python
# ------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# 4. Copia o código-fonte
# ------------------------------------------------------------
COPY src/ ./src/
COPY tests/ ./tests/

# ------------------------------------------------------------
# 5. Garante que o dataset inicial (seed) existe no Rank 0
#    O breast_cancer.npz é gerado pelo sklearn e precisa
#    estar presente para o líder iniciar a distribuição P2P.
# ------------------------------------------------------------
RUN mkdir -p /app/src/Locals/Rank\ 0

# Gera o dataset breast_cancer caso não exista no build
RUN python -c "\
import os, numpy as np;\
from sklearn.datasets import load_breast_cancer;\
data = load_breast_cancer();\
path = '/app/src/Locals/Rank 0/breast_cancer.npz';\
np.savez(path, X=data.data, y=data.target);\
print(f'Dataset salvo em {path} — shape: {data.data.shape}')\
"

# ------------------------------------------------------------
# 6. Configuração de ambiente
# ------------------------------------------------------------
# Permite rodar mpiexec como root dentro do container
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Garante que os imports do src/ funcionem
ENV PYTHONPATH=/app/src
# Desativa buffering para ver os logs em tempo real
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# 7. Número de nós MPI (pode ser sobrescrito com -e NUM_NODES=N)
# ------------------------------------------------------------
ENV NUM_NODES=6

# Expõe a porta do servidor de visualização web
EXPOSE 8000

# ------------------------------------------------------------
# 8. Entrypoint padrão
# ------------------------------------------------------------
# Usa shell form para expandir $NUM_NODES
CMD mpiexec --allow-run-as-root -n $NUM_NODES python -B src/MPI_start.py
