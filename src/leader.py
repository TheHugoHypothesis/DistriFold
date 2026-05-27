import time
import numpy as np
from sklearn.model_selection import KFold
from node_context import NodeContext
from communication.network import MPIConnector
from communication.torrent import TorrentEngine
from communication.communication_tags import *
class LeaderWork:
    def __init__(self, context: NodeContext, connector: MPIConnector):
        self.context = context
        self.connector = connector
        self.torrent = TorrentEngine(context, connector)
        self.num_folds = 5
        self.results = {}

    def _carregar_dataset(self):
        
        #Carrega o dataset Breast Cancer da sklearn para testar inicialmente
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = data.data
        y = data.target
        print(f"[Líder] Dataset Breast Cancer carregado com sucesso. Formato: {X.shape}")


        return X, y
    
    #Comm só chama isso se ele for lider
    def run(self):
        #Aguarda a eleição do líder inicial
        while self.context.leader_rank is None:
            time.sleep(0.1)

        #Distribui o dataset apenas se for a liderança inicial (epoch == 0)
        with self.context.lock:
            is_initial_leader = (self.context.leader_context["epoch"] == 0)

        if is_initial_leader:
            print(f"[Líder Rank {self.context.rank}] Iniciando Fase de Torrenting P2P...")
            X, y = self._carregar_dataset()
            self.torrent.distribute_as_leader(X, y)
            print("[Líder] Distribuição de dados via P2P concluída.")
            
            # Inicializa a fila de folds no contexto
            with self.context.lock:
                ctx = self.context.leader_context
                ctx["pending_folds"] = list(range(self.num_folds))
                ctx["epoch"] = 1
                self.context.context_dirty = True

        #Lider que não é primeiro
        else:
            with self.context.lock:
                ctx = self.context.leader_context
                print(f"[Novo Líder Recuperado Rank {self.context.rank}] Retomando a partir da epoch {ctx['epoch']}...")
                
                #Recupera tarefas ativas que não foram concluídas
                for w_rank, fold_id in list(ctx["active_assignments"].items()):
                    if fold_id not in ctx["completed_folds"]:
                        ctx["pending_folds"].append(fold_id)
                ctx["active_assignments"] = {}
                ctx["epoch"] += 1
                self.context.context_dirty = True

        print(f"[Líder {self.context.rank}] Iniciando agendamento dinâmico dos {self.num_folds} Folds...")

        #loop principal
        while True:
            with self.context.lock:
                completed_count = len(self.context.leader_context["completed_folds"])
            
            if completed_count >= self.num_folds:
                break

            if self.context.stop_event.is_set():
                break

            #Atribui folds livres a workers livres
            with self.context.lock:
                ctx = self.context.leader_context
                pending_folds = ctx["pending_folds"]
                active_assignments = ctx["active_assignments"]
                completed_folds = ctx["completed_folds"]
                
                for worker_rank in range(self.context.size):
                    if worker_rank == self.context.leader_rank:
                        #TODO Aqui colocar para o lider trabalhar localmente também
                        continue
                    
                    if worker_rank not in active_assignments and len(pending_folds) > 0:
                        fold_id = pending_folds.pop(0)
                        active_assignments[worker_rank] = fold_id
                        ctx["epoch"] += 1
                        self.context.context_dirty = True
                        
                        # Envia a tarefa (apenas o fold_id)
                        self.connector.send(fold_id, dest=worker_rank, tag=TAG_TASK)
                        print(f"[Líder {self.context.rank}] Atribuiu Fold {fold_id} para o Worker {worker_rank}")



            #Coleta resultados (tb não-bloqueante)
            with self.context.lock:
                ctx = self.context.leader_context
                active_assignments = list(ctx["active_assignments"].keys())

            for worker_rank in active_assignments:
                res = self.connector.check_message(source=worker_rank, tag=TAG_RESULT)
                if res:
                    fold_id = res["fold_id"]
                    with self.context.lock:
                        ctx = self.context.leader_context
                        ctx["completed_folds"][fold_id] = res["metrics"]
                        if worker_rank in ctx["active_assignments"]:
                            del ctx["active_assignments"][worker_rank]
                        ctx["epoch"] += 1
                        self.context.context_dirty = True
                    print(f"[Líder {self.context.rank}] Sucesso! Recebido resultado do Fold {fold_id} do Worker {worker_rank}")

            time.sleep(0.1)



        with self.context.lock:
            completed_folds = dict(self.context.leader_context["completed_folds"])

        # Calcula as métricas médias e desvios padrão
        accuracies = [met["accuracy"] for met in completed_folds.values()]
        f1s = [met["f1"] for met in completed_folds.values()]
        losses = [met["loss"] for met in completed_folds.values()]

        


        
        #Saída final
        print('Final:')
        for f, met in sorted(completed_folds.items()):
            print(f"Fold {f}: Acurácia = {met['accuracy']:.4f} | F1-Score = {met['f1']:.4f} | Loss = {met['loss']:.4f}")
    
        print(f"Acurácia Média : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"F1-Score Médio : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"Loss Média     : {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        

        #mdnda workers encerrarem
        print("[Líder] Enviando sinal de encerramento para os workers...")
        for worker_rank in range(self.context.size):
            if worker_rank != self.context.leader_rank:
                self.connector.send(-1, dest=worker_rank, tag=TAG_TASK)

        #mata o lider
        self.context.stop_event.set()