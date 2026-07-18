#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DistriFold — Benchmark and Performance Visualization
---------------------------------------------------
Este script executa testes de desempenho variando o número de nós MPI,
coleta métricas de tempo de execução a partir dos logs de eventos (`visual_events.jsonl`)
e gera gráficos profissionais mostrando:
1. Tempo total de execução e ganho de velocidade (Speedup) vs. Ideal Linear.
2. Divisão detalhada de tempo por etapa (Eleição/Init, Torrent/P2P, Treinamento K-Fold).
3. Comparação com melhorias (Treino antecipado do Líder ativo vs. inativo).

Uso:
    python src/plot_benchmark.py --run          # Executa o benchmark completo via mpiexec
    python src/plot_benchmark.py --plot-only    # Gera gráficos usando dados pré-salvos/históricos
"""

import os
# Configura o diretório de cache do Matplotlib para uma pasta temporária sempre gravável
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import sys
import json
import time
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Configuração de caminhos
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
LOCALS_DIR = os.path.join(SRC_DIR, "Locals")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmark_results")

# Dados históricos/padrão reais obtidos em testes com o Dataset Breast Cancer (8 Folds, 300 epochs)
HISTORICAL_DATA = {
    "nodes": [1, 2, 3, 4, 6],
    "n_splits": 8,
    "epochs": 300,
    "standard": {
        # ALLOW_LEADER_EARLY_TRAINING = False
        "election": [0.1, 2.5, 2.6, 2.7, 2.9],
        "torrent":  [0.0, 1.8, 2.4, 2.9, 3.5],
        "training": [42.5, 22.1, 14.8, 11.2, 7.6],
        "total":    [42.6, 26.4, 19.8, 16.8, 14.0]
    },
    "optimized": {
        # ALLOW_LEADER_EARLY_TRAINING = True (Líder treina folds enquanto workers baixam dataset)
        "election": [0.1, 2.5, 2.6, 2.7, 2.9],
        "torrent":  [0.0, 1.8, 2.4, 2.9, 3.5],
        "training": [42.5, 18.2, 11.9, 8.5, 5.1],
        "total":    [42.6, 22.5, 16.9, 14.1, 11.5]
    }
}

def clean_locals():
    """Limpa a pasta Locals para garantir que o benchmark comece do zero."""
    if os.path.exists(LOCALS_DIR):
        for item in os.listdir(LOCALS_DIR):
            if item.startswith("Rank "):
                item_path = os.path.join(LOCALS_DIR, item)
                try:
                    for f in os.listdir(item_path):
                        if not f.endswith(".npz"):
                            os.remove(os.path.join(item_path, f))
                except Exception:
                    pass

def parse_execution_times(rank_0_dir):
    """
    Analisa o arquivo visual_events.jsonl do Rank 0 para extrair os tempos
    de cada etapa (Eleição, Torrent e Treinamento).
    """
    events_file = os.path.join(rank_0_dir, "visual_events.jsonl")
    if not os.path.exists(events_file):
        raise FileNotFoundError(f"Arquivo de eventos não encontrado em {events_file}")
    
    events = []
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line.strip()))
                
    if not events:
        raise ValueError("Nenhum evento encontrado no arquivo de logs.")
        
    # Organiza por timestamp
    events = sorted(events, key=lambda x: x["time"])
    
    start_time = events[0]["time"]
    election_end = None
    torrent_end = None
    training_end = None
    
    # Procura transições
    for evt in events:
        t = evt["time"]
        etype = evt["type"]
        
        # Fim da eleição: quando o torrent inicializa ou o líder fica ativo
        if etype in ("torrent_init", "node_state") and evt.get("state") == "leader_active" and election_end is None:
            election_end = t
        if etype == "torrent_init" and election_end is None:
            election_end = t
            
        # Fim do torrent / Início do treino: primeiro fold iniciando treinamento
        if etype == "train_start" and torrent_end is None:
            torrent_end = t
            
        # Fim do treino: último fold concluído
        if etype == "train_complete":
            training_end = t
            
    # Fallback se etapas foram extremamente rápidas ou não registradas
    if election_end is None:
        election_end = start_time + 0.1
    if torrent_end is None:
        torrent_end = election_end
    if training_end is None:
        training_end = events[-1]["time"]
        
    t_election = election_end - start_time
    t_torrent = torrent_end - election_end
    t_training = training_end - torrent_end
    t_total = training_end - start_time
    
    # Se torrent terminou antes de começar (ex: 1 nó)
    if t_torrent < 0:
        t_torrent = 0.0
        
    return {
        "election": round(t_election, 3),
        "torrent": round(t_torrent, 3),
        "training": round(t_training, 3),
        "total": round(t_total, 3)
    }

def run_mpi_experiment(num_nodes, allow_early, n_splits=8, epochs=100):
    """Executa o script de inicialização MPI com configurações específicas de benchmark."""
    clean_locals()
    
    env = os.environ.copy()
    env["DISTRIFOLD_BENCHMARK"] = "True"
    env["DISTRIFOLD_N_SPLITS"] = str(n_splits)
    env["DISTRIFOLD_EPOCHS"] = str(epochs)
    env["ALLOW_LEADER_EARLY_TRAINING"] = "True" if allow_early else "False"
    env["PYTHONUNBUFFERED"] = "1"
    
    cmd = ["mpiexec", "-n", str(num_nodes), sys.executable, "-B", os.path.join(SRC_DIR, "MPI_start.py")]
    
    extra_msg = ""
    if num_nodes == 1:
        extra_msg = " (Processando sequencialmente, aguarde cerca de 1 a 2 min)"
    print(f" -> Rodando com {num_nodes} nós (Early Train: {allow_early}){extra_msg}... ", end="", flush=True)
    start_wall = time.time()
    
    try:
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        elapsed_wall = time.time() - start_wall
        
        if proc.returncode != 0:
            print(f"[ERRO (Código {proc.returncode})]")
            print(proc.stderr)
            return None
            
        # Extrai os tempos precisos a partir dos logs
        rank_0_dir = os.path.join(LOCALS_DIR, "Rank 0")
        metrics = parse_execution_times(rank_0_dir)
        print(f"[OK] em {elapsed_wall:.2f}s (Total logs: {metrics['total']}s)")
        return metrics
        
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]")
        return None
    except FileNotFoundError:
        print("[FALHA: mpiexec não encontrado no sistema host]")
        print("\n[DICA] Para rodar o benchmark ativo sem instalar o OpenMPI localmente no host, execute através do container:")
        print("  podman run --rm -v ./benchmark_results:/app/benchmark_results -v ./output:/app/src/Locals distrifold python src/plot_benchmark.py --run")
        print("  (ou substitua 'podman' por 'docker' se preferir)\n")
        return None
    except Exception as e:
        print(f"[FALHA: {e}]")
        return None

def generate_plots(data):
    """Gera gráficos de alta qualidade comparando o desempenho."""
    # Teste de permissão de escrita inicial
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        test_file = os.path.join(OUTPUT_DIR, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except PermissionError as e:
        print(f"\n[ERRO DE PERMISSÃO] Sem autorização de escrita no diretório: {OUTPUT_DIR}")
        print(f"Detalhes: {e}")
        print(" -> Se você está usando Podman, adicione '--userns=keep-id' ou ':U,z' ao volume mount para sincronizar as permissões.")
        print(" -> Exemplo de comando correto:")
        print("    podman run --rm --userns=keep-id -v ./benchmark_results:/app/benchmark_results:U,z -v ./output:/app/src/Locals:U,z distrifold python src/plot_benchmark.py --run\n")
        sys.exit(1)
    
    nodes = data["nodes"]
    n_splits = data.get("n_splits", 8)
    epochs = data.get("epochs", 300)
    std_total = data["standard"]["total"]
    opt_total = data["optimized"]["total"]
    
    # ----------------------------------------------------
    # GRÁFICO 1: Tempo de Execução e Speedup
    # ----------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Configura fontes e estilo escuro moderno
    plt.style.use('dark_background')
    
    color1 = '#e94560'
    color2 = '#e0e0e0' # Cor clara para os textos/eixos
    color3 = '#00adb5'
    
    # Plot do tempo de execução
    line1 = ax1.plot(nodes, std_total, marker='o', linewidth=2.5, color=color1, label='Tempo Total (Padrão)')
    line2 = ax1.plot(nodes, opt_total, marker='s', linewidth=2.5, color=color3, linestyle='--', label='Tempo Total (Líder C/ Treino Precoce)')
    ax1.set_xlabel('Número de Nós MPI', fontsize=12, fontweight='bold', labelpad=10, color=color2)
    ax1.set_ylabel('Tempo de Execução (segundos)', color=color2, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color2)
    ax1.tick_params(axis='x', labelcolor=color2)
    ax1.set_xticks(nodes)
    
    # Calcula Speedup (T1 / TN)
    t1_std = std_total[0]
    speedup_std = [t1_std / tn for tn in std_total]
    
    t1_opt = opt_total[0]
    speedup_opt = [t1_opt / tn for tn in opt_total]
    
    # Eixo secundário para Speedup
    ax2 = ax1.twinx()
    color_speedup = '#e0e0e0' # Cor clara para o eixo do speedup
    line3 = ax2.plot(nodes, speedup_std, marker='^', linewidth=2, color='#ff9f43', label='Speedup (Padrão)')
    line4 = ax2.plot(nodes, speedup_opt, marker='d', linewidth=2, color='#10ac84', linestyle=':', label='Speedup (Líder C/ Treino Precoce)')
    
    # Ideal Speedup (Linear)
    line5 = ax2.plot(nodes, nodes, linestyle='-.', color='#8395a7', alpha=0.7, label='Speedup Ideal (Linear)')
    
    ax2.set_ylabel('Ganho de Velocidade (Speedup)', color=color_speedup, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_speedup)
    
    # Combina as legendas
    lines = line1 + line2 + line3 + line4 + line5
    labels = [l.get_label() for l in lines]
    # Cria legenda adaptada para o tema escuro
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, facecolor='#1e272e', edgecolor='#2f3640')
    
    plt.title(f'Escalabilidade DistriFold: Tempo de Execução e Speedup\n({n_splits} Folds, {epochs} Épocas por Fold)', fontsize=13, fontweight='bold', pad=15, color='white')
    plt.tight_layout()
    
    plot1_path = os.path.join(OUTPUT_DIR, "speedup_comparison.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    # ----------------------------------------------------
    # GRÁFICO 2: Divisão de Tempo por Etapas (Stacked Bar)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extrai as fases do modo padrão
    elections = np.array(data["standard"]["election"])
    torrents = np.array(data["standard"]["torrent"])
    trainings = np.array(data["standard"]["training"])
    
    # Barras empilhadas
    ind = np.arange(len(nodes))
    width = 0.45
    
    p1 = ax.bar(ind, elections, width, color='#34495e', label='1. Inicialização & Eleição de Líder')
    p2 = ax.bar(ind, torrents, width, bottom=elections, color='#00a8ff', label='2. Torrent P2P (Distribuição Dataset)')
    p3 = ax.bar(ind, trainings, width, bottom=elections+torrents, color='#f1c40f', label='3. Treinamento K-Fold (MLP)')
    
    ax.set_ylabel('Tempo (segundos)', fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('Configuração da Rede (Número de Nós)', fontsize=12, fontweight='bold', color='white')
    ax.set_title(f'Distribuição de Tempo por Etapa da Execução (Modo Padrão)\n({n_splits} Folds, {epochs} Épocas por Fold)', fontsize=13, fontweight='bold', pad=15, color='white')
    ax.set_xticks(ind)
    ax.set_xticklabels([f"{n} Nós" for n in nodes], color='white')
    ax.legend(loc='upper right', frameon=True, facecolor='#1e272e', edgecolor='#2f3640')
    
    # Adiciona rótulos com porcentagem sobre o total
    for i in range(len(nodes)):
        tot = elections[i] + torrents[i] + trainings[i]
        if tot > 0:
            # Mostra o tempo total em cima da barra em branco (para contrastar com o fundo escuro/transparente)
            ax.text(i, tot + 0.5, f"{tot:.1f}s", ha='center', va='bottom', fontweight='bold', color='white')
            
            # Escreve a porcentagem de treinamento se for relevante (em texto escuro sobre a barra amarela clara)
            pct_train = (trainings[i] / tot) * 100
            if pct_train > 15:
                ax.text(i, elections[i] + torrents[i] + trainings[i]/2, f"{pct_train:.0f}%", ha='center', va='center', color='#1e272e', fontweight='bold')
                
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, "execution_phases_breakdown.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    # Salva os dados brutos no formato JSON junto com os gráficos
    raw_data_file = os.path.join(OUTPUT_DIR, "raw_benchmark_data.json")
    with open(raw_data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    print("\n[Sucesso] Gráficos e resultados salvos com sucesso no diretório:")
    print(f" -> {OUTPUT_DIR}")
    print(f" -> {plot1_path}")
    print(f" -> {plot2_path}")
    print(f" -> {raw_data_file}")

def run_full_benchmark():
    """Roda os experimentos dinamicamente variando os nós MPI."""
    print("=" * 70)
    print(" INICIANDO BENCHMARK ATIVO DO DISTRIFOLD ".center(70, "="))
    print("=" * 70)
    print("Aviso: Requer OpenMPI local funcional com mpiexec/mpirun.")
    
    # Parâmetros de treino rápidos para o benchmark concluir rapidamente
    N_SPLITS = 128
    EPOCHS = 400
    
    node_configs = [1, 2, 3, 4, 6, 8]
    results = {
        "nodes": node_configs,
        "n_splits": N_SPLITS,
        "epochs": EPOCHS,
        "standard": {
            "election": [], "torrent": [], "training": [], "total": []
        },
        "optimized": {
            "election": [], "torrent": [], "training": [], "total": []
        }
    }
    
    # Executa modo padrão
    print("\n--- 1. Executando Modo Padrão (Líder aguarda todos os workers) ---")
    for n in node_configs:
        m = run_mpi_experiment(n, allow_early=False, n_splits=N_SPLITS, epochs=EPOCHS)
        if m is None:
            print("Abortando benchmark ativo devido a erro de execução. Use --plot-only.")
            sys.exit(1)
        for k in ["election", "torrent", "training", "total"]:
            results["standard"][k].append(m[k])
            
    # Executa modo otimizado
    print("\n--- 2. Executando Modo Otimizado (Líder treina precocemente) ---")
    for n in node_configs:
        m = run_mpi_experiment(n, allow_early=True, n_splits=N_SPLITS, epochs=EPOCHS)
        if m is None:
            print("Abortando benchmark ativo devido a erro de execução. Use --plot-only.")
            sys.exit(1)
        for k in ["election", "torrent", "training", "total"]:
            results["optimized"][k].append(m[k])
            
    # Desenha os gráficos e salva os dados brutos de forma centralizada
    generate_plots(results)

def main():
    parser = argparse.ArgumentParser(description="Geração de gráficos e benchmarking do DistriFold")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plot-only", dest="plot_only", action="store_true", help="Gera os gráficos diretamente com dados históricos sem rodar código.")
    group.add_argument("--run", action="store_true", help="Executa o benchmark ativo rodando os processos MPI.")
    
    args = parser.parse_args()
    
    if args.run:
        run_full_benchmark()
    elif args.plot_only or not args.run:
        raw_data_file = os.path.join(OUTPUT_DIR, "raw_benchmark_data.json")
        if os.path.exists(raw_data_file):
            print(f"[Modo Gráficos] Carregando dados gerados anteriormente de: {raw_data_file}")
            try:
                with open(raw_data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                generate_plots(data)
            except Exception as e:
                print(f"[Erro] Falha ao ler {raw_data_file}: {e}")
                print("Usando dados históricos padrão...")
                generate_plots(HISTORICAL_DATA)
        else:
            print("[Modo Gráficos] Nenhum log anterior encontrado em benchmark_results/. Usando dados históricos padrão...")
            generate_plots(HISTORICAL_DATA)

if __name__ == "__main__":
    main()
