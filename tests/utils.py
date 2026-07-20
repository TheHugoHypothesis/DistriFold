import os
import shutil
import subprocess
import time
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
LOCALS_DIR = os.path.join(PROJECT_ROOT, "src", "Locals")

def clear_locals():
    """Limpa a pasta Locals para simular nós novos zerados.
    Mantém apenas o arquivo NPZ no Rank 0 para iniciar o seed.
    """
    if os.path.exists(LOCALS_DIR):
        for item in os.listdir(LOCALS_DIR):
            item_path = os.path.join(LOCALS_DIR, item)
            if os.path.isdir(item_path) and item != "Rank 0":
                try:
                    shutil.rmtree(item_path)
                except Exception as e:
                    print(f"[Cleanup] Erro ao deletar {item_path}: {e}")
            elif os.path.isdir(item_path) and item == "Rank 0":
                for file_item in os.listdir(item_path):
                    if not file_item.endswith(".npz"):
                        file_path = os.path.join(item_path, file_item)
                        try:
                            if os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                            else:
                                os.remove(file_path)
                        except Exception:
                            pass

def run_mpi(scenario_name, num_nodes=3, env_overrides=None, timeout=25):
    """Executa o script test_MPI_start.py com o cenário de testes especificado."""
    clear_locals()
    
    env = os.environ.copy()
    env["DISTRIFOLD_TEST"] = scenario_name
    env["PYTHONUNBUFFERED"] = "1"
    
    if env_overrides:
        for k, v in env_overrides.items():
            env[k] = str(v)
            
    mpi_cmd = ["mpiexec", "-n", str(num_nodes), sys.executable, "-B", "tests/test_MPI_start.py"]
    
    print(f"\n[Test Runner] Executando: {' '.join(mpi_cmd)} (Cenário: {scenario_name})")
    
    proc = subprocess.Popen(
        mpi_cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        print(f"[Test Runner] Timeout de {timeout}s expirado. Encerrando processos MPI...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        stdout, stderr = proc.communicate()
        return -1, stdout, stderr

def get_node_logs(rank):
    """Lê as linhas do log correspondente ao nó do rank especificado."""
    log_path = os.path.join(LOCALS_DIR, f"Rank {rank}", f"{rank}.txt")
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return f.readlines()
    except Exception as e:
        print(f"[Test Runner] Erro ao ler log do Rank {rank}: {e}")
        return []

def search_log(rank, keyword):
    """Retorna True se a palavra-chave for encontrada no log do nó de determinado rank."""
    logs = get_node_logs(rank)
    for line in logs:
        if keyword in line:
            return True
    return False
