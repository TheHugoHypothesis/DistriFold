import sys
import os
import time

# Adiciona o diretório 'src' ao path do Python para importações corretas
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import node_context
import worker

# ==========================================
# 1. Monkey-Patching de NodeContext.__init__
# ==========================================
original_node_context_init = node_context.NodeContext.__init__

def patched_node_context_init(self, rank, size, teste=None):
    test_scenario = os.getenv("DISTRIFOLD_TEST")
    custom_teste = None
    
    if test_scenario in ["eleicao", "falha_leader"]:
        # O líder (rank 0) morre após 4s. Os demais trabalhadores continuam ativos.
        if rank == 0:
            custom_teste = {'time_working': 4, 'time_timeout': 100}
        else:
            custom_teste = {'time_working': 0, 'time_timeout': 0}
    elif test_scenario == "elegibilidade":
        # O líder morre após 4s. Rank 1 é elegível. Rank 2 é forçado a ser inelegível (sem dataset).
        if rank == 0:
            custom_teste = {'time_working': 4, 'time_timeout': 100}
        else:
            custom_teste = {'time_working': 0, 'time_timeout': 0}
    elif test_scenario == "falha_worker":
        # Líder e Rank 2 continuam ativos. Rank 1 morre temporariamente após 3s e volta após 10s.
        if rank == 1:
            custom_teste = {'time_working': 3, 'time_timeout': 10}
        else:
            custom_teste = {'time_working': 0, 'time_timeout': 0}
    elif test_scenario == "recuperacao":
        # Líder continua ativo. Rank 1 morre após 3s e retorna após 5s.
        if rank == 1:
            custom_teste = {'time_working': 3, 'time_timeout': 5}
        else:
            custom_teste = {'time_working': 0, 'time_timeout': 0}
    else:
        # Sem queda simulada
        custom_teste = teste

    original_node_context_init(self, rank, size, teste=custom_teste)

node_context.NodeContext.__init__ = patched_node_context_init

# =========================================================
# 2. Monkey-Patching de NodeContext.has_dataset_completed
# =========================================================
# Definimos como propriedade para interceptar leituras dinâmicas
@property
def has_dataset_completed_prop(self):
    no_dataset_rank = os.getenv("DISTRIFOLD_NO_DATASET_WORKER")
    if no_dataset_rank is not None and str(self.rank) == no_dataset_rank:
        return False
    return self._has_dataset_completed

@has_dataset_completed_prop.setter
def has_dataset_completed_prop(self, val):
    self._has_dataset_completed = val

node_context.NodeContext.has_dataset_completed = has_dataset_completed_prop

# ========================================================
# 3. Monkey-Patching de worker.train_fold_from_arrays
# ========================================================
original_train_fold_from_arrays = worker.train_fold_from_arrays

def patched_train_fold_from_arrays(X, y, train_idx, test_idx, config, fold_id=None):
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    slow_rank = os.getenv("DISTRIFOLD_SLOW_WORKER")
    if slow_rank is not None and str(rank) == slow_rank:
        print(f"[TEST MONKEYPATCH] Rank {rank} simulando lentidão (dormindo por 2s antes do treino)...")
        time.sleep(2.0)
    return original_train_fold_from_arrays(X, y, train_idx, test_idx, config, fold_id)

worker.train_fold_from_arrays = patched_train_fold_from_arrays

# ==========================================
# 4. Importa e executa o MPI_start original
# ==========================================
import MPI_start

if __name__ == "__main__":
    MPI_start.main()
