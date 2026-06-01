import time
from logger import print_to_node as print
import numpy as np
from sklearn.model_selection import KFold
from node_context import NodeContext
from communication.network import MPIConnector
from communication.torrent import TorrentEngine
from communication.communication_tags import *
import os


class LeaderWork:
    def __init__(self, context: NodeContext, connector: MPIConnector, dataset_id, MLP_Config, FOLD_Config, comm_service=None):
        self.context = context
        self.connector = connector
        self.comm_service = comm_service
        self.dataset_id = dataset_id
        self.MLP_Config = MLP_Config
        self.FOLD_Config = FOLD_Config

        self.torrent = TorrentEngine(context, connector)
        self.num_folds = FOLD_Config['n_splits']
        self.results = {}


    
    #Comm só chama isso se ele for lider
    def run(self):
        #Aguarda a eleição do líder inicial
        time.sleep(0.1)
        if self.context.leader_rank != self.context.rank or self.context.recovering:
            return


        #Distribui o dataset apenas se for a liderança inicial (epoch == 0)
        with self.context.lock:
            is_initial_leader = (self.context.leader_context["epoch"] == 0)



        print(f"[Líder Rank {self.context.rank}] Iniciando Fase de Torrenting P2P...")
        X, y = self._carregar_dataset()
        
        self.torrent.distribute_as_leader(X, y, self.dataset_id, self.num_folds)
        print("[Líder] Distribuição de dados via P2P concluída.")


        if is_initial_leader:
 
            
            #Inicializa a fila de folds no contexto
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
        while not self.context.stop_event.is_set():            
            if self.context.recovering:
                print(f'Voltando de queda, desativando Thread Lider')
                return

            if self.context.leader_rank != self.context.rank:
                print(f"[Nó {self.context.rank}] Não é mais líder, encerrando papel de Líder.")
                return
            
            

            # if not self.context._node_esta_ativo():
            #     time.sleep(0.1)
            #     continue


            end_time = time.time() + 0.6
            now = time.time()
            while time.time() < end_time:
                for source in range(self.context.size):
                    if source == self.context.leader_rank:
                        continue



                    msg = self.comm_service.Poll(source=source, tag=TAG_ACK)
                    if msg is not None:
                        print(f"Líder recebeu ACK de {source}")
                        with self.context.lock:
                            self.context.setAlive(source)
                            self.context.leader_context["last_ack"][source] = time.time()


                    msg_ready = self.comm_service.Poll(source=source, tag=TAG_NODE_READY)
                    if msg_ready is not None:
                        print(f"Líder recebeu ACK-Pronto do Nó {source}")
                        with self.context.lock:
                            self.context.setAlive(source)
                            self.context.setReady(source)
                            self.context.leader_context["last_ack"][source] = time.time()


                time.sleep(0.05)

                now = time.time()
                for source in range(self.context.size):
                    if source == self.context.leader_rank:
                        continue
                    with self.context.lock:
                        last_ack = self.context.leader_context["last_ack"].get(source)
                        is_alive = source in self.context.leader_context["alive_nodes"]
                    if is_alive and (last_ack is None or (now - last_ack) > self.comm_service.timeout_seconds):
                        print(f"Líder detectou timeout do Nó {source} (sem ACK recente)!")
                        with self.context.lock:
                            self.context.setDead(source)



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
                
                for worker_rank in ctx["ready_nodes"]:
                    if worker_rank == self.context.leader_rank:
                        #TODO Aqui colocar para o lider trabalhar localmente também
                        continue
                    
                    if worker_rank not in active_assignments and len(pending_folds) > 0:
                        fold_id = pending_folds.pop(0)
                        active_assignments[worker_rank] = fold_id
                        ctx["epoch"] += 1
                        self.context.context_dirty = True
                        

                        config = {'MLP': self.MLP_Config,
                                'FOLD': self.FOLD_Config
                        }

                        task = {
                            "fold_id": fold_id,
                            "config": config
                        }

                        # Envia a tarefa (apenas o fold_id)
                        if self.comm_service:
                            print(f'pending_folds: {pending_folds}')
                            self.comm_service.enqueue("leader", dest=worker_rank, tag=TAG_TASK, payload=task) #leader sempre envia config e o id do fold pra ele.
                        
                        print(f"[Líder {self.context.rank}] Atribuiu Fold {fold_id} para o Worker {worker_rank}")



            #Coleta resultados (tb não-bloqueante)
            for worker_rank in range(self.context.size):
                if worker_rank == self.context.leader_rank:
                    continue

                
    
                if self.comm_service:
                    res = self.comm_service.Poll(source=worker_rank, tag=TAG_RESULT)
                if res:
                    fold_id = res["fold_id"]
                    with self.context.lock:
                        ctx = self.context.leader_context
                        if fold_id in ctx["completed_folds"]:
                            continue

                        ctx["completed_folds"][fold_id] = res["metrics"]

                        if worker_rank in ctx["active_assignments"]:
                            del ctx["active_assignments"][worker_rank]
                        if fold_id in ctx["pending_folds"]:
                            ctx["pending_folds"].remove(fold_id)


                        ctx["epoch"] += 1
                        self.context.context_dirty = True
                    print(f"[Líder {self.context.rank}] Sucesso! Recebido resultado do Fold {fold_id} do Worker {worker_rank}")

            time.sleep(0.1)



        #Saída

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


                task = {
                    "fold_id": -1,
                    "config": None
                }


                if self.comm_service:
                    self.comm_service.enqueue("leader", dest=worker_rank, tag=TAG_TASK, payload=task)

        #mata o lider
        self.context.stop_event.set()



    def _carregar_dataset(self):
        
        #Carrega o dataset Breast Cancer da sklearn para testar inicialmente        
        src_dir = os.path.dirname(os.path.abspath(__file__))
        local_dir = os.path.join(src_dir, "Locals", f"Rank {self.context.rank}")
        file_path = os.path.join(local_dir, f"{self.dataset_id}.npz")

        

        if os.path.exists(file_path):
            print(f"[Worker {self.context.rank}] Dataset '{self.dataset_id}' encontrado localmente! Carregando de {file_path}")
            data = np.load(file_path, allow_pickle=True)
            self.has_dataset_complete = True
            X = data["X"]
            y = data["y"]
            return X, y

        print(file_path)
        raise FileNotFoundError(f"Dataset não encontrado: {file_path}")