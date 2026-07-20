# DistriFold

Rede distribuída de treinamento descentralizado via K-Fold Cross-Validation com OpenMPI, tolerância a falhas, eleição de líder e distribuição P2P de dataset.

---

## 1. Sobre o Projeto

O **DistriFold** é uma arquitetura distribuída para paralelização de **K-Fold Cross-Validation** em modelos de Inteligência Artificial. Em vez de treinar todos os $K$ folds de maneira sequencial em um único nó, o sistema distribui os folds dinamicamente entre múltiplos workers em uma rede MPI.

### Principais Recursos
- **Agendamento Dinâmico**: O líder distribui folds sob demanda para os workers disponíveis.
- **Distribuição P2P (Torrent)**: O dataset é dividido em chunks e compartilhado entre os nós usando um protocolo P2P descentralizado.
- **Tolerância a Falhas**: Detecção de queda de nós via heartbeat, reatribuição de tarefas e re-eleição automática de líder por menor rank ativo.
- **Otimização de Treino Precoce**: Suporte a treinamento antecipado pelo líder enquanto a distribuição de dataset ocorre nos demais nós.
- **Interface Visual e Métricas**: Servidor HTTP embutido para monitoramento web em tempo real e gerador de gráficos de escalabilidade/speedup.

---

## 2. Arquitetura do Sistema

O sistema segue o modelo **Líder-Seguidor** sob o tempo de execução do OpenMPI:

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

## 3. Estrutura de Diretórios

```text
DistriFold/
├── Dockerfile              # Containerização base (Python 3.11 + OpenMPI)
├── docker-compose.yml      # Orquestração de serviços (App, Testes e Visualização)
├── requirements.txt        # Dependências de runtime (mpi4py, numpy, scikit-learn, etc.)
│
├── src/
│   ├── MPI_start.py        # Ponto de entrada e orquestrador principal de nós
│   ├── leader.py           # Lógica do papel de Líder (agendamento e agregação)
│   ├── worker.py           # Lógica do papel de Worker (download P2P e treino MLP)
│   ├── MLP.py              # Classificador MLP (Backpropagation manual em NumPy)
│   ├── node_context.py     # Gerenciador de estado e contexto de nó
│   ├── logger.py            # Sistema de log para arquivo e console
│   ├── clear_locals.py     # Utilitário de limpeza de dados locais
│   ├── plot_benchmark.py   # Gerador de gráficos de speedup e eficiência
│   ├── analyze_metrics.py  # Analisador textual de métricas e overheads
│   ├── visualizer_server.py # Servidor HTTP da interface de visualização
│   ├── visualizer_logger.py # Logger de eventos JSONL para o visualizador
│   ├── index.html          # Dashboard interativo visual em Vanilla JS + D3
│   └── communication/      # Camada de rede, abstração MPI e engine torrent P2P
│
└── tests/
    ├── run_tests.py         # Suíte de 9 testes de integração distribuídos
    ├── test_MPI_start.py    # Entry point para injeção de testes
    └── utils.py             # Helpers para execução e parsing de logs
```

---

## 4. Passo a Passo de Execução

### Pré-requisitos
- **Docker** (Método recomendado)
- **Docker Compose**
- *(Opcional)* **Podman** / **Podman Compose**
- *(Opcional para execução local)* **OpenMPI** (v4.x+) e **Python 3.11**

---

### Passo 4.1: Executando a Aplicação Distribuída

#### Método 1: Docker (Recomendado)

1. **Construir a imagem Docker**:
   ```bash
   docker build -t distrifold .
   ```

2. **Rodar com a configuração padrão (6 nós MPI)**:
   ```bash
   docker run --rm -v $(pwd)/output:/app/src/Locals distrifold
   ```

3. **Rodar alterando o número de nós MPI (ex: 4 nós)**:
   ```bash
   docker run --rm -e NUM_NODES=4 -v $(pwd)/output:/app/src/Locals distrifold
   ```

#### Método 2: Docker Compose

```bash
# Iniciar o serviço principal com build automático
docker compose up --build

# Iniciar alterando o número de nós MPI
NUM_NODES=4 docker compose up --build
```

#### Método 3: Compatibilidade com Podman

No Linux (com SELinux ou User Namespaces), utilize o mapeamento de permissões `:U,z` e `--userns=keep-id`:

```bash
# Build no Podman
podman build -t distrifold .

# Execução no Podman
podman run --rm --userns=keep-id \
  -v ./output:/app/src/Locals:U,z \
  distrifold

# Execução via Podman Compose
NUM_NODES=4 podman compose up --build
```

#### Método 4: Execução Local no Hospedeiro (Sem Container)

```bash
# 1. Instalar dependências Python
pip install -r requirements.txt

# 2. Limpar cache de execuções anteriores
python src/clear_locals.py

# 3. Executar via OpenMPI
mpiexec -n 4 python -B src/MPI_start.py
```

---

### Passo 4.2: Executando os Testes Automatizados

A suíte contempla **9 cenários de integração distribuída**:
1. Eleição de Líder após queda
2. Elegibilidade (nós sem dataset não assumem liderança)
3. Difusão descentralizada do dataset via Torrent P2P
4. Balanceamento dinâmico de carga
5. Reatribuição após falha de Worker
6. Recuperação de contexto após falha do Líder
7. Sincronização de réplicas de estado
8. Reintegração de nós recuperados
9. Escalabilidade de speedup

#### Rodar Testes via Docker Compose (Recomendado)
```bash
docker compose --profile test up --build tests
```

#### Rodar Testes via Docker direto
```bash
docker run --rm -v $(pwd)/test-output:/app/src/Locals distrifold python -B tests/run_tests.py
```

#### Rodar Testes via Podman
```bash
podman run --rm --userns=keep-id -v ./test-output:/app/src/Locals:U,z distrifold python -B tests/run_tests.py
```

#### Rodar Testes Localmente
```bash
python tests/run_tests.py
```

---

### Passo 4.3: Interface de Visualização Web em Tempo Real

 O sistema registra a troca de mensagens e estado da rede em formato JSONL. O servidor `visualizer_server.py` expõe esses eventos para a interface web interativa.

#### Método 1: Via Docker / Docker Compose

1. **Subir o servidor de visualização**:
   ```bash
   docker compose --profile vis up --build visualizer
   ```
   *(Ou via Docker direto: `docker run --rm -p 8000:8000 -v $(pwd)/output:/app/src/Locals distrifold python src/visualizer_server.py 8000`)*

2. **Abrir no navegador**:
   Acesse [http://localhost:8000](http://localhost:8000)

3. **Visualizar a simulação**:
   Execute uma rodada da aplicação em outro terminal. O painel D3.js exibirá a topologia, o tráfego P2P, a eleição de líder e o status do treinamento dos folds em tempo real.

#### Método 2: Via Python Local
```bash
python src/visualizer_server.py 8000
```
Navegue para `http://localhost:8000`.

---

### Passo 4.4: Benchmarks, Gráficos e Análise de Eficiência

O projeto inclui scripts para medir a escalabilidade e o impacto do tempo de comunicação no cluster.

#### 1. Gerador de Gráficos (`plot_benchmark.py`)

Gera os gráficos `speedup_comparison.png` e `execution_phases_breakdown.png` com fundo transparente.

- **Modo Histórico (Gera os gráficos usando logs/dados existentes sem re-executar)**:
  ```bash
  python src/plot_benchmark.py --plot-only
  ```

- **Modo Ativo (Executa automaticamente rodadas de 1 a 6 nós e plota)**:
  ```bash
  docker run --rm \
    -v $(pwd)/benchmark_results:/app/benchmark_results \
    -v $(pwd)/output:/app/src/Locals \
    distrifold python src/plot_benchmark.py --run
  ```

#### 2. Analisador de Métricas Textual (`analyze_metrics.py`)

Lê o arquivo `raw_benchmark_data.json` e imprime a tabela comparativa de **Eficiência Paralela ($E_N$)** e **Overhead de Comunicação (%)**:

```bash
python src/analyze_metrics.py
```

---

## 5. Como Alterar Configurações

As configurações do sistema podem ser ajustadas através de **variáveis de ambiente** ou editando `src/MPI_start.py`.

### Variáveis de Ambiente Suportadas

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `NUM_NODES` | `6` | Número total de processos MPI iniciados no container. |
| `DISTRIFOLD_N_SPLITS` | `64` | Quantidade de folds na divisão do K-Fold Cross Validation. |
| `DISTRIFOLD_EPOCHS` | `600` | Número máximo de épocas de treinamento da MLP por fold. |
| `ALLOW_LEADER_EARLY_TRAINING` | `False` | Se `True`, o líder começa a treinar folds localmente antes que todos os workers estejam prontos. |
| `DISTRIFOLD_BENCHMARK` | `False` | Se `True`, desativa a injeção simulada de falhas para medições puras de tempo. |

### Exemplos de Ajuste via Linha de Comando (Docker)

```bash
# Executar com 128 Folds e 400 Épocas por Fold
docker run --rm \
  -e NUM_NODES=4 \
  -e DISTRIFOLD_N_SPLITS=128 \
  -e DISTRIFOLD_EPOCHS=400 \
  -v $(pwd)/output:/app/src/Locals \
  distrifold

# Habilitar Treinamento Precoce do Líder
docker run --rm \
  -e ALLOW_LEADER_EARLY_TRAINING=True \
  -v $(pwd)/output:/app/src/Locals \
  distrifold
```

### Configuração Manual em Código (`src/MPI_start.py`)

Para alterar a arquitetura da rede neural ou parâmetros de falha diretamente no código:

```python
# Configuração do K-Fold
CONFIG_FOLD = {
    "n_splits": int(os.getenv("DISTRIFOLD_N_SPLITS", "64")),
    "shuffle": True,
    "random_state": 42
}

# Configuração da MLP
CONFIG_MLP = {
    "h1": 64,          # Neurônios na camada oculta 1
    "h2": 16,          # Neurônios na camada oculta 2
    "lr": 0.001,       # Taxa de aprendizado
    "epochs": int(os.getenv("DISTRIFOLD_EPOCHS", "600")),
    "batch_size": 4
}
```

---

## 6. Pilha Tecnológica

- **Linguagem**: Python 3.11
- **Comunicação Distribuída**: OpenMPI 4.x + `mpi4py` 4.1
- **Computação Científica**: NumPy 2.x & scikit-learn 1.9
- **Containerização**: Docker & Docker Compose (com suporte a Podman)
- **Visualização Frontend**: HTML5, Vanilla JavaScript, D3.js, Matplotlib
