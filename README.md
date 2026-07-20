# DistriFold

Rede distribuída de treinamento descentralizado via K-Fold Cross-Validation com OpenMPI.

---

## Como Rodar (Passo a Passo)

### Pré-requisitos
- Docker e Docker Compose (Método recomendado)
- Podman e Podman Compose (Suportado no Linux)
- OpenMPI 4.x e Python 3.11 (Para execução local sem containers)

---

### 1. Executando a Aplicação Distribuída

#### Método 1: Docker (Recomendado)

1. Construir a imagem da aplicação:
   ```bash
   docker build -t distrifold .
   ```

2. Executar com a configuração padrão (6 nós MPI):
   ```bash
   docker run --rm -v $(pwd)/output:/app/src/Locals distrifold
   ```

3. Executar alterando a quantidade de nós MPI (ex: 4 nós):
   ```bash
   docker run --rm -e NUM_NODES=4 -v $(pwd)/output:/app/src/Locals distrifold
   ```

#### Método 2: Docker Compose

```bash
# Executar a aplicação principal com build automático (6 nós)
docker compose up --build

# Executar alterando o número de nós via variável de ambiente
NUM_NODES=4 docker compose up --build
```

#### Método 3: Podman (Compatibilidade Linux / SELinux)

Ao utilizar o Podman no Linux, adicione a flag `--userns=keep-id` e o sufixo `:U,z` nos volumes para sincronização de permissões:

```bash
# Build no Podman
podman build -t distrifold .

# Execução no Podman com volume gravável pelo usuário
podman run --rm --userns=keep-id \
  -v ./output:/app/src/Locals:U,z \
  distrifold

# Execução via Podman Compose
NUM_NODES=4 podman compose up --build
```

#### Método 4: Execução Local no Host (Sem Container)

```bash
# 1. Instalar as dependências Python
pip install -r requirements.txt

# 2. Limpar dados locais de execuções anteriores
python src/clear_locals.py

# 3. Disparar o processo MPI com o número desejado de nós (ex: 4 nós)
mpiexec -n 4 python -B src/MPI_start.py
```

---

### 2. Executando os Testes Automatizados

A suíte de testes valida 9 cenários de integração distribuída (eleição de líder, difusão torrent P2P, falhas de workers/líder, reatribuição de tarefas e escalabilidade).

```bash
# Executar a suíte de testes via Docker Compose (Recomendado)
docker compose --profile test up --build tests

# Executar via Docker direto
docker run --rm -v $(pwd)/test-output:/app/src/Locals distrifold python -B tests/run_tests.py

# Executar via Podman
podman run --rm --userns=keep-id -v ./test-output:/app/src/Locals:U,z distrifold python -B tests/run_tests.py

# Executar localmente no hospedeiro
python tests/run_tests.py
```

---

### 3. Visualização Web Interativa em Tempo Real

O servidor HTTP disponibiliza um painel gráfico em D3.js na porta 8000 para acompanhar a topologia da rede, o tráfego P2P, eleições de líder e o status dos folds.

1. Subir o servidor de visualização via Docker Compose:
   ```bash
   docker compose --profile vis up --build visualizer
   ```
   *(Ou localmente via Python: `python src/visualizer_server.py 8000`)*

2. Abrir no navegador:
   Acesse http://localhost:8000

3. Acompanhar a execução:
   Dispare uma rodada da aplicação em outro terminal para visualizar os eventos em tempo real.

---

### 4. Benchmarks e Análise de Eficiência Paralela

#### 1. Gerador de Gráficos (plot_benchmark.py)

Gera gráficos comparativos (`speedup_comparison.png` e `execution_phases_breakdown.png`) com fundo transparente.

```bash
# Modo Rápido (Gera os gráficos a partir dos dados do último teste ou dados históricos)
python src/plot_benchmark.py --plot-only

# Modo Ativo via Docker (Executa o benchmark completo variando de 1 a 6 nós e gera os gráficos)
docker run --rm \
  -v $(pwd)/benchmark_results:/app/benchmark_results \
  -v $(pwd)/output:/app/src/Locals \
  distrifold python src/plot_benchmark.py --run
```

#### 2. Analisador de Métricas Textual (analyze_metrics.py)

Gera um relatório detalhado no console calculando a Eficiência Paralela e a porcentagem gasta com Overhead de Comunicação:

```bash
python src/analyze_metrics.py
```

---

## Como Alterar Configurações

O comportamento do sistema pode ser customizado via variáveis de ambiente ou editando `src/MPI_start.py`.

### Variáveis de Ambiente Suportadas

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `NUM_NODES` | `6` | Número de nós MPI ativos iniciados pelo container |
| `DISTRIFOLD_N_SPLITS` | `64` | Quantidade de folds na divisão do K-Fold Cross Validation |
| `DISTRIFOLD_EPOCHS` | `600` | Número máximo de épocas de treinamento da MLP por fold |
| `ALLOW_LEADER_EARLY_TRAINING` | `False` | Se `True`, o líder treina localmente sem esperar que os workers fiquem prontos |
| `DISTRIFOLD_BENCHMARK` | `False` | Se `True`, desativa a injeção simulada de falhas para medições puras de performance |

### Exemplos de Execução com Parâmetros Customizados

```bash
# Executar com 128 Folds e 400 Épocas via Docker
docker run --rm \
  -e NUM_NODES=4 \
  -e DISTRIFOLD_N_SPLITS=128 \
  -e DISTRIFOLD_EPOCHS=400 \
  -v $(pwd)/output:/app/src/Locals \
  distrifold

# Executar habilitando o Treinamento Precoce do Líder
docker run --rm \
  -e ALLOW_LEADER_EARLY_TRAINING=True \
  -v $(pwd)/output:/app/src/Locals \
  distrifold
```
