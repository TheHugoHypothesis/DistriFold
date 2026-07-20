<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Podman-Compatible-892CA0?style=for-the-badge&logo=podman&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenMPI-4.x-4E9A06?style=for-the-badge&logo=openmpi&logoColor=white" />
  <img src="https://img.shields.io/badge/mpi4py-4.1-005F9E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
</p>

<h1 align="center">⚡ DistriFold</h1>

<p align="center">
  <strong>Rede distribuída de treinamento descentralizado via K-Fold Cross-Validation com OpenMPI</strong>
</p>

<p align="center">
  <em>Acelere a validação cruzada de modelos de IA distribuindo os K folds entre múltiplos nós MPI, com tolerância a falhas, eleição de líder e distribuição P2P de dataset.</em>
</p>

---

## 📖 Sobre o Projeto

O **DistriFold** é uma arquitetura distribuída de alta performance para paralelização de **K-Fold Cross-Validation** em modelos de Machine Learning e Inteligência Artificial. Em vez de treinar todos os $K$ folds de maneira sequencial em uma única máquina, o sistema distribui os folds dinamicamente entre múltiplos workers em uma rede MPI.

### 🌟 Destaques do Sistema
- **⚡ Agendamento Dinâmico de Folds**: O líder atribui tarefas sob demanda para os workers disponíveis assim que eles ficam ociosos.
- **🌊 Distribuição P2P de Dataset (Torrent)**: O dataset é fatiado em chunks e compartilhado descentralizadamente entre os nós através de um protocolo torrent próprio sobre MPI.
- **🛡️ Tolerância a Falhas & Resiliência**: Detecção ativa de falhas por heartbeat com timeout, reatribuição transparente de tarefas e re-eleição automática de líder pelo menor rank ativo com dataset.
- **⚡ Otimização de Treinamento Precoce**: O líder pode iniciar o treinamento de folds localmente enquanto a propagação P2P ocorre entre os demais workers.
- **🌐 Dashboard Interativo Web**: Servidor HTTP embutido com interface em tempo real utilizando D3.js para visualizar a topologia da rede, fluxo P2P e status dos folds.
- **📊 Análise de Desempenho & Métricas**: Ferramentas integradas para gerar gráficos de Speedup vs. Ideal e relatórios detalhados de Eficiência Paralela ($E_N$).

---

## 🏗️ Arquitetura do Sistema

O sistema opera sob um modelo **Líder-Seguidor** flexível sobre o runtime do OpenMPI:

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        CLUSTER MPI                                  │
│                                                                     │
│   ┌──────────────┐     Heartbeat / Sincronização   ┌──────────────┐ │
│   │   Nó 0       │◄──────────────────────────────► │   Nó 1       │ │
│   │  (Líder)     │       TAG_HELLO / TAG_ACK       │  (Worker)    │ │
│   │              │                                 │              │ │
│   │  • Agenda    │       TAG_TASK ─────────────────►│  • Treina    │ │
│   │    folds     │ ◄───── TAG_RESULT               │    fold      │ │
│   │  • Coleta    │                                 │  • Retorna   │ │
│   │    métricas  │                                 │    pesos     │ │
│   └──────────────┘                                 └──────────────┘ │
│                                                                     │
│   ◄──── Torrent P2P (TAG_TORRENT_*) ──────────────────────────────►  │
│   Distribuição descentralizada do dataset entre todos os nós        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Estrutura do Projeto

```text
DistriFold/
├── 📦 Dockerfile              # Imagem base (Python 3.11 + OpenMPI + dependências)
├── 📦 docker-compose.yml      # Orquestração de serviços (App, Testes e Visualização)
├── 📋 requirements.txt        # Dependências Python (mpi4py, numpy, scikit-learn, matplotlib)
├── 📋 .dockerignore            # Exclusões do contexto de build containerizado
│
├── 📁 src/
│   ├── 🚀 MPI_start.py        # Ponto de entrada e orquestrador principal de nós
│   ├── 👑 leader.py            # Lógica do papel de Líder (agendamento e agregação)
│   ├── ⚙️ worker.py           # Lógica do papel de Worker (download P2P e treino MLP)
│   ├── 🧠 MLP.py              # Classificador MLP (Backpropagation manual em NumPy)
│   ├── 🌐 node_context.py     # Gerenciador de estado e contexto global do nó
│   ├── 📝 logger.py            # Logger formatado para arquivo e console
│   ├── 🧹 clear_locals.py     # Utilitário de limpeza de estado local
│   ├── 📊 plot_benchmark.py   # Gerador de gráficos de speedup e divisão por fases
│   ├── 📈 analyze_metrics.py  # Analisador de eficiência paralela e overhead de rede
│   ├── 🖥️ visualizer_server.py # Servidor HTTP REST/Static da interface web
│   ├── 📜 visualizer_logger.py # Logger de eventos JSONL para o visualizador
│   ├── 🎨 index.html          # Dashboard interativo em Vanilla JS + D3.js
│   └── 📁 communication/      # Camada de rede, abstração MPI e engine torrent P2P
│       ├── 📡 communication.py      # Serviço de heartbeat e eleição de líder
│       ├── 🏷️ communication_tags.py # Definição de tags numéricas das mensagens MPI
│       ├── 🔌 network.py            # Wrapper thread-safe para Send/Recv no MPI
│       └── 🌊 torrent.py            # Engine P2P (inventário HAVE, requests e seeding)
│
└── 📁 tests/
    ├── 🧪 run_tests.py         # Suíte de 9 testes de integração distribuídos
    ├── 🧪 test_MPI_start.py    # Entry point para injeção e monkey-patching nos testes
    └── 🔧 utils.py             # Helpers para sub-processos MPI e parsing de logs
```

---

## 🚀 Como Rodar (Passo a Passo)

### 📋 Pré-requisitos
- **Docker** & **Docker Compose** *(Método principal recomendado)*
- **Podman** & **Podman Compose** *(Suportado nativamente no Linux)*
- **OpenMPI 4.x** & **Python 3.11** *(Para execução local direta sem containers)*

---

### 1️⃣ Executando a Aplicação Distribuída

#### 🐳 Método 1: Docker (Recomendado)

1. **Construir a imagem da aplicação**:
   ```bash
   docker build -t distrifold .
   ```

2. **Executar com a configuração padrão (6 nós MPI)**:
   ```bash
   docker run --rm -v $(pwd)/output:/app/src/Locals distrifold
   ```

3. **Executar alterando a quantidade de nós MPI (ex: 4 nós)**:
   ```bash
   docker run --rm -e NUM_NODES=4 -v $(pwd)/output:/app/src/Locals distrifold
   ```

#### 🐙 Método 2: Docker Compose

```bash
# Executar a aplicação principal com build automático (6 nós)
docker compose up --build

# Executar alterando o número de nós via variável de ambiente
NUM_NODES=4 docker compose up --build
```

#### 🦭 Método 3: Podman (Compatibilidade Linux / SELinux)

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

#### 💻 Método 4: Execução Local no Host (Sem Container)

```bash
# 1. Instalar as dependências Python
pip install -r requirements.txt

# 2. Limpar dados locais de execuções anteriores
python src/clear_locals.py

# 3. Disparar o processo MPI com o número desejado de nós (ex: 4 nós)
mpiexec -n 4 python -B src/MPI_start.py
```

---

### 2️⃣ Executando os Testes Automatizados

A suíte de testes valida **9 cenários críticos de sistemas distribuídos**:
- 1. **Eleição de Líder**: Valida a re-eleição automática após a queda do nó principal.
- 2. **Elegibilidade**: Garante que nós sem dataset não assumam a liderança.
- 3. **Difusão P2P**: Confirma que todos os nós reconstroem o dataset completo via torrent.
- 4. **Balanceamento Dinâmico**: Garante a distribuição proporcional de tarefas entre os workers.
- 5. **Falha de Worker**: Verifica se tarefas de um worker morto retornam à fila de pendentes.
- 6. **Falha do Líder**: Testa a recuperação do contexto global pelo novo líder.
- 7. **Sincronização de Estado**: Confirma que todos os nós mantêm réplicas atualizadas.
- 8. **Reintegração de Nós**: Testa se nós recuperados sincronizam estado com o líder.
- 9. **Escalabilidade**: Avalia a redução no tempo de treino com a adição de nós.

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

### 3️⃣ Visualização Web Interativa em Tempo Real

O sistema grava visualmente todos os pacotes trocados e o progresso dos folds num log JSONL. O servidor HTTP disponibiliza um painel gráfico em D3.js na porta **8000**.

1. **Subir o servidor de visualização via Docker Compose**:
   ```bash
   docker compose --profile vis up --build visualizer
   ```
   *(Ou localmente via Python: `python src/visualizer_server.py 8000`)*

2. **Abrir no navegador**:
   Acesse **[http://localhost:8000](http://localhost:8000)**

3. **Acompanhar a execução**:
   Dispare uma rodada da aplicação em outro terminal. O painel exibirá as conexões da rede, animações de transferência de pacotes P2P, alertas de eleição de líder e barras de progresso do treinamento K-Fold!

---

### 4️⃣ Benchmarks e Análise de Eficiência Paralela

#### 📊 1. Gerador de Gráficos (`plot_benchmark.py`)

Gera gráficos comparativos (`speedup_comparison.png` e `execution_phases_breakdown.png`) com fundo transparente prontos para artigos e apresentações.

```bash
# Modo Rápido (Gera os gráficos a partir dos dados do último teste ou dados históricos)
python src/plot_benchmark.py --plot-only

# Modo Ativo via Docker (Executa o benchmark completo variando de 1 a 6 nós e gera os gráficos)
docker run --rm \
  -v $(pwd)/benchmark_results:/app/benchmark_results \
  -v $(pwd)/output:/app/src/Locals \
  distrifold python src/plot_benchmark.py --run
```

#### 📈 2. Analisador de Métricas Textual (`analyze_metrics.py`)

Gera um relatório detalhado no console calculando a **Eficiência Paralela ($E_N$)** e a porcentagem gasta com **Overhead de Comunicação**:

```bash
python src/analyze_metrics.py
```

---

## ⚙️ Como Alterar Configurações

O comportamento do sistema pode ser customizado via **variáveis de ambiente** ou editando `src/MPI_start.py`.

### 🛠️ Variáveis de Ambiente Suportadas

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `NUM_NODES` | `6` | Número de nós MPI ativos iniciados pelo container |
| `DISTRIFOLD_N_SPLITS` | `64` | Quantidade de folds na divisão do K-Fold Cross Validation |
| `DISTRIFOLD_EPOCHS` | `600` | Número máximo de épocas de treinamento da MLP por fold |
| `ALLOW_LEADER_EARLY_TRAINING` | `False` | Se `True`, o líder treina localmente sem esperar que os workers fiquem prontos |
| `DISTRIFOLD_BENCHMARK` | `False` | Se `True`, desativa a injeção simulada de falhas para medições puras de performance |

### 💡 Exemplos de Execução com Parâmetros Customizados

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

---

## 🛡️ Tolerância a Falhas

| Cenário | Mecanismo de Recuperação |
|---------|--------------------------|
| **Worker cai** | O Líder detecta timeout via heartbeat → o fold em andamento é devolvido para a fila de pendentes → outro worker assume a tarefa. |
| **Líder cai** | Os Workers detectam timeout do Líder → iniciam eleição descentralizada → o menor rank ativo contendo o dataset é eleito novo Líder e recupera o estado global. |
| **Nó retorna** | O nó recuperado faz broadcast perguntando pela identidade do Líder → sincroniza o contexto de execução → retoma o papel de worker. |

---

## 🔧 Stack Tecnológica

| Tecnologia | Versão | Função no Projeto |
|-----------|--------|-------------------|
| **Python** | 3.11 | Linguagem base da aplicação |
| **Docker / Docker Compose** | Latest | Containerização primária e orquestração |
| **Podman** | Latest | Runtime de containers alternativo com suporte rootless |
| **OpenMPI** | 4.x | Runtime de comunicação paralela e mensageria de alta velocidade |
| **mpi4py** | 4.1 | Interface Python para bindings MPI |
| **NumPy** | 2.x | Computação matricial e implementação manual da MLP |
| **scikit-learn** | 1.9 | Divisão do K-Fold Cross Validation e métricas de avaliação |
| **Matplotlib** | 3.x | Visualização gráfica de benchmarks e escalabilidade |
| **D3.js / HTML5** | Latest | Interface web interativa de monitoramento da rede |

---

<p align="center">
  Desenvolvido para a disciplina de <strong>Sistemas Distribuídos (DSID)</strong>
</p>
