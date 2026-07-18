#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DistriFold — Analisador de Métricas de Comunicação e Eficiência
--------------------------------------------------------------
Este script lê o arquivo `raw_benchmark_data.json` e gera relatórios
detalhados mostrando a eficiência paralela da rede e o impacto dos
overheads de comunicação (eleição + torrent P2P) no tempo total de execução.
"""

import os
import json
import sys

def main():
    # Define caminhos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "benchmark_results", "raw_benchmark_data.json")

    if not os.path.exists(json_path):
        print(f"[Erro] Arquivo de dados brutos não encontrado em: {json_path}")
        print("Certifique-se de executar o benchmark ativo (--run) ou histórico (--plot-only) primeiro.")
        sys.exit(1)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Erro] Falha ao ler os dados do arquivo JSON: {e}")
        sys.exit(1)

    nodes = data["nodes"]
    print("=" * 90)
    print(" RELATÓRIO DE EFICIÊNCIA PARALELA E IMPACTO DE COMUNICAÇÃO ".center(90, "="))
    print("=" * 90)
    print(f"Dataset: Breast Cancer | Parâmetros: {data.get('n_splits', 8)} Folds, {data.get('epochs', 300)} Épocas por Fold\n")

    for mode in ["standard", "optimized"]:
        mode_title = "MODO PADRÃO (LÍDER AGUARDA WORKERS)" if mode == "standard" else "MODO OTIMIZADO (LÍDER TREINA PRECOCEMENTE)"
        print(f"--- {mode_title} ---")
        print(f"{'Nós':<6} | {'Tempo (s)':<10} | {'Speedup Obt.':<12} | {'Speedup Id.':<12} | {'Eficiência':<10} | {'Treino (Comp.)':<16} | {'Comunicação (Overhead)':<22}")
        print("-" * 90)

        t1 = data[mode]["total"][0]
        
        for i, n in enumerate(nodes):
            t_total = data[mode]["total"][i]
            t_election = data[mode]["election"][i]
            t_torrent = data[mode]["torrent"][i]
            t_train = data[mode]["training"][i]
            
            # Cálculos de Speedup e Eficiência
            speedup_obtained = t1 / t_total if t_total > 0 else 0
            speedup_ideal = float(n)
            efficiency = (speedup_obtained / speedup_ideal) * 100 if speedup_ideal > 0 else 0
            
            # Cálculos de Comunicação vs Computação
            t_comm = t_election + t_torrent
            pct_comm = (t_comm / t_total) * 100 if t_total > 0 else 0
            pct_train = (t_train / t_total) * 100 if t_total > 0 else 0

            # Formatação de string
            speedup_str = f"{speedup_obtained:.2f}x"
            ideal_str = f"{speedup_ideal:.1f}x"
            efficiency_str = f"{efficiency:.1f}%"
            train_str = f"{t_train:.1f}s ({pct_train:.1f}%)"
            comm_str = f"{t_comm:.1f}s ({pct_comm:.1f}%)"

            print(f"{n:<6} | {t_total:<10.2f} | {speedup_str:<12} | {ideal_str:<12} | {efficiency_str:<10} | {train_str:<16} | {comm_str:<22}")
        print("\n" + "=" * 90 + "\n")

    # Análise interpretativa simplificada
    print("ANÁLISE SINTETIZADA:")
    print(" 1. Impacto da Eleição + Torrent (Comunicação):")
    print("    - Em redes pequenas (1-2 nós), a computação (treino da MLP) domina a execução.")
    print("    - À medida que adicionamos nós (ex: 4-6 nós), a comunicação passa a ocupar uma fração maior")
    print("      do tempo total. Isso ilustra o limite da Lei de Amdahl em sistemas distribuídos.")
    print("\n 2. Ganho com o Treinamento Precoce (Modo Otimizado):")
    print("    - O modo otimizado reduz consideravelmente a latência inicial de torrenting do líder,")
    print("      pois ele inicia o treinamento local imediatamente enquanto os workers realizam o P2P.")
    print("    - Isso reflete diretamente em uma maior Eficiência Paralela na rede.")
    print("=" * 90)

if __name__ == "__main__":
    main()
