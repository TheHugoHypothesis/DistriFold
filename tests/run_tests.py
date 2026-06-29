import sys
import os
import time
import re
import matplotlib.pyplot as plt

# Adiciona o diretório raiz ao path
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from tests.utils import run_mpi, search_log, get_node_logs

def print_banner(title):
    print("\n" + "=" * 60)
    print(f" {title.upper()} ".center(60, "="))
    print("=" * 60)

def print_result(name, passed, detail=""):
    status = " [OK] " if passed else " [FALHA] "
    color_start = "\033[92m" if passed else "\033[91m"
    color_end = "\033[0m"
    print(f"{color_start}{status}{color_end} {name:<40} {detail}")

# ==========================================
# 1. Teste de Eleição de Líder
# ==========================================
def test_leader_election():
    print_banner("Teste 1: Eleição de Líder")
    code, stdout, stderr = run_mpi("eleicao", num_nodes=3, timeout=20)
    
    # Assertions
    timeout_detected = search_log(1, "detectou timeout do líder") or search_log(2, "detectou timeout do líder")
    election_started = search_log(1, "Iniciando eleição") or search_log(2, "Iniciando eleição")
    new_leader_elected = search_log(1, "Novo líder eleito")
    follower_discovered = search_log(2, "Descobri líder, ele é 1") or search_log(2, "sincronizou contexto")
    
    passed = timeout_detected and election_started and new_leader_elected and follower_discovered
    
    details = []
    if not timeout_detected: details.append("Timeout do líder não detectado")
    if not election_started: details.append("Eleição não iniciada")
    if not new_leader_elected: details.append("Rank 1 não foi eleito líder")
    if not follower_discovered: details.append("Rank 2 não descobriu o novo líder")
    
    detail_str = ", ".join(details) if not passed else "Líder inicial caiu; Rank 1 foi eleito com sucesso."
    print_result("Eleição de Líder", passed, detail_str)
    return passed

# ==========================================
# 2. Teste de Elegibilidade dos Nós
# ==========================================
def test_node_eligibility():
    print_banner("Teste 2: Elegibilidade dos Nós")
    code, stdout, stderr = run_mpi(
        "elegibilidade", 
        num_nodes=3, 
        env_overrides={"DISTRIFOLD_NO_DATASET_WORKER": "2"},
        timeout=20
    )
    
    # Assertions
    inelegible_refused = search_log(2, "Não tenho dataset (ou simulado), não vou me candidatar")
    leader_elected = search_log(1, "Novo líder eleito")
    
    # Verifica que Rank 2 não foi eleito como líder
    leader_is_1 = True
    logs_rank_2 = get_node_logs(2)
    for line in logs_rank_2:
        if "Novo líder eleito" in line and "Sou o menor" in line:
            leader_is_1 = False
            
    passed = inelegible_refused and leader_elected and leader_is_1
    
    details = []
    if not inelegible_refused: details.append("Rank 2 não recusou candidatura")
    if not leader_elected: details.append("Nenhum líder foi eleito")
    if not leader_is_1: details.append("Rank 2 (inelegível) foi eleito líder erroneamente")
    
    detail_str = ", ".join(details) if not passed else "Nó sem dataset recusou candidatura; Rank 1 eleito líder."
    print_result("Elegibilidade dos Nós", passed, detail_str)
    return passed

# ==========================================
# 3. Teste de Distribuição P2P do Dataset
# ==========================================
def test_p2p_distribution():
    print_banner("Teste 3: Distribuição P2P do Dataset")
    code, stdout, stderr = run_mpi("p2p", num_nodes=3, timeout=20)
    
    # Assertions
    rank1_ok = search_log(1, "Dataset 100% reconstruído com sucesso") or search_log(1, "Dataset encontrado localmente")
    rank2_ok = search_log(2, "Dataset 100% reconstruído com sucesso") or search_log(2, "Dataset encontrado localmente")
    
    passed = rank1_ok and rank2_ok
    
    details = []
    if not rank1_ok: details.append("Rank 1 não completou o dataset")
    if not rank2_ok: details.append("Rank 2 não completou o dataset")
    
    detail_str = ", ".join(details) if not passed else "Dataset replicado e reconstruído com sucesso em todos os nós."
    print_result("Distribuição P2P", passed, detail_str)
    return passed

# ==========================================
# 4. Teste de Balanceamento Dinâmico
# ==========================================
def test_dynamic_balancing():
    print_banner("Teste 4: Balanceamento Dinâmico")
    # Configura o Rank 1 como lento
    code, stdout, stderr = run_mpi(
        "balanceamento", 
        num_nodes=3, 
        env_overrides={"DISTRIFOLD_SLOW_WORKER": "1"},
        timeout=30
    )
    
    # Procura nos logs do líder quem fez os folds
    leader_logs = get_node_logs(0)
    assignments = {1: 0, 2: 0}
    
    for line in leader_logs:
        match = re.search(r"Atribuiu Fold \d+ para o Worker (\d)", line)
        if match:
            worker = int(match.group(1))
            if worker in assignments:
                assignments[worker] += 1
                
    passed = assignments[2] > assignments[1]
    
    detail = f"Worker 1 (Lento): {assignments[1]} folds | Worker 2 (Rápido): {assignments[2]} folds"
    print_result("Balanceamento Dinâmico", passed, detail)
    
    # Plotagem do Gráfico
    try:
        workers = ['Worker 1 (Lento)', 'Worker 2 (Rápido)']
        counts = [assignments[1], assignments[2]]
        
        plt.figure(figsize=(6, 4))
        bars = plt.bar(workers, counts, color=['#e74c3c', '#2ecc71'])
        plt.title('Distribuição Dinâmica de Folds por Capacidade de Processamento')
        plt.ylabel('Quantidade de Folds Processados')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, str(yval), ha='center', va='bottom', fontweight='bold')
            
        chart_path = os.path.join(TESTS_DIR, "balanceamento_dinamico.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"[Visual] Gráfico de performance salvo em: {chart_path}")
    except Exception as e:
        print(f"[Visual] Erro ao gerar gráfico: {e}")
        
    return passed

# ==========================================
# 5. Teste de Falha de Trabalhador
# ==========================================
def test_worker_failure():
    print_banner("Teste 5: Falha de Trabalhador")
    code, stdout, stderr = run_mpi("falha_worker", num_nodes=3, timeout=30)
    
    # Assertions
    timeout_detected = search_log(0, "detectou timeout do Nó 1")
    training_finished = search_log(0, "Acurácia Média")
    
    passed = timeout_detected and training_finished
    
    details = []
    if not timeout_detected: details.append("Líder não detectou timeout do worker 1")
    if not training_finished: details.append("Treinamento não concluiu com sucesso")
    
    detail_str = ", ".join(details) if not passed else "Queda detectada e fold reatribuído. Treinamento concluído."
    print_result("Falha de Trabalhador", passed, detail_str)
    return passed

# ==========================================
# 6. Teste de Falha do Líder
# ==========================================
def test_leader_failure():
    print_banner("Teste 6: Falha do Líder")
    code, stdout, stderr = run_mpi("falha_leader", num_nodes=3, timeout=30)
    
    # Assertions
    new_leader_ok = search_log(1, "Novo líder eleito")
    recovered_context = search_log(1, "Retomando a partir da epoch")
    final_output = search_log(1, "Acurácia Média")
    
    passed = new_leader_ok and recovered_context and final_output
    
    details = []
    if not new_leader_ok: details.append("Novo líder não foi eleito")
    if not recovered_context: details.append("Estado da execução não foi restaurado")
    if not final_output: details.append("Treinamento não terminou sob o novo líder")
    
    detail_str = ", ".join(details) if not passed else "Novo líder eleito, recuperou contexto e completou treinamento."
    print_result("Falha do Líder", passed, detail_str)
    return passed

# ==========================================
# 7. Teste de Sincronização de Estado
# ==========================================
def test_state_sync():
    print_banner("Teste 7: Sincronização de Estado")
    code, stdout, stderr = run_mpi("sincronizacao", num_nodes=3, timeout=20)
    
    # Assertions
    rank1_sync = search_log(1, "sincronizou contexto para epoch")
    rank2_sync = search_log(2, "sincronizou contexto para epoch")
    
    passed = rank1_sync and rank2_sync
    
    details = []
    if not rank1_sync: details.append("Rank 1 não sincronizou épocas")
    if not rank2_sync: details.append("Rank 2 não sincronizou épocas")
    
    detail_str = ", ".join(details) if not passed else "Trabalhadores mantiveram suas réplicas de estado sincronizadas."
    print_result("Sincronização de Estado", passed, detail_str)
    return passed

# ==========================================
# 8. Teste de Recuperação de Nós
# ==========================================
def test_node_recovery():
    print_banner("Teste 8: Recuperação de Nós")
    code, stdout, stderr = run_mpi("recuperacao", num_nodes=3, timeout=30)
    
    # Assertions
    dropped = search_log(0, "detectou timeout do Nó 1")
    recovering = search_log(1, "Voltando de queda")
    synced = search_log(1, "Sincronizei contexto após retorno") or search_log(1, "sincronizou contexto")
    
    passed = dropped and recovering and synced
    
    details = []
    if not dropped: details.append("Líder não detectou queda do Rank 1")
    if not recovering: details.append("Rank 1 não iniciou processo de retorno")
    if not synced: details.append("Rank 1 não sincronizou contexto após retorno")
    
    detail_str = ", ".join(details) if not passed else "Nó caiu, voltou e recuperou o estado atual do líder com sucesso."
    print_result("Recuperação de Nós", passed, detail_str)
    return passed

# ==========================================
# 9. Teste de Escalabilidade
# ==========================================
def test_scalability():
    print_banner("Teste 9: Escalabilidade")
    
    # Execução 1: 2 processos (1 lider, 1 worker)
    t_start = time.time()
    run_mpi("escalabilidade", num_nodes=2, timeout=25)
    t2 = time.time() - t_start
    
    # Execução 2: 3 processos (1 lider, 2 workers)
    t_start = time.time()
    run_mpi("escalabilidade", num_nodes=3, timeout=25)
    t3 = time.time() - t_start
    
    passed = t3 < t2
    detail = f"Tempo com 2 Nós: {t2:.2f}s | Tempo com 3 Nós: {t3:.2f}s"
    print_result("Escalabilidade", passed, detail)
    return passed

# ==========================================
# Executor Principal
# ==========================================
def run_all_tests():
    tests = [
        test_leader_election,
        test_node_eligibility,
        test_p2p_distribution,
        test_dynamic_balancing,
        test_worker_failure,
        test_leader_failure,
        test_state_sync,
        test_node_recovery,
        test_scalability
    ]
    
    print("\n" + "=" * 60)
    print(" INICIANDO A SUÍTE DE TESTES DISTRIBUÍDOS ".center(60, "#"))
    print("=" * 60)
    
    results = []
    for test in tests:
        try:
            res = test()
            results.append(res)
        except Exception as e:
            print(f"\033[91m[ERRO CRÍTICO] Falha de execução no teste {test.__name__}: {e}\033[0m")
            results.append(False)
            
    passed_count = sum(1 for r in results if r)
    total_count = len(results)
    
    print_banner("Resumo Final dos Testes")
    print(f"Total de Cenários: {total_count}")
    print(f"Aprovados       : {passed_count}")
    print(f"Reprovados      : {total_count - passed_count}")
    
    if passed_count == total_count:
        print("\n\033[92m[SUCESSO] Todos os testes passaram com êxito!\033[0m")
    else:
        print("\n\033[91m[FALHA] Alguns cenários falharam.\033[0m")

if __name__ == "__main__":
    run_all_tests()
