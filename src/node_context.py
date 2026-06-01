import threading
import time



#Facilitar gereciamento global dos Nós
class NodeContext:
    def __init__(self, rank, size, teste=None):
        self.rank = rank
        self.size = size
        self.leader_rank = None
        self.old_leader = None
        self.last_heartbeat = time.time()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.ready_to_work = False
        self.teste = teste
        self.start_time = time.time()
        self.last_internet_tick = time.time()
        self.recovering = False

        if self.rank == 0:
            self.has_dataset_completed = True
        else: self.has_dataset_completed = False

        self.needs_full_sync = False
        

        #Informação compartilhada pra se o líder cair
        self.context_dirty = False

        self.leader_context = {
            "epoch": 0,
            "pending_folds": [],
            "last_heartbeat":[],
            "last_ack": {},
            "alive_nodes": list(range(size)), #Considera que inicia com todo mundo vivo
            "ready_nodes": [],
            "active_assignments": {},  #Mapear quem está fazendo o que {worker_rank: fold_id}
            "completed_folds": {}      #Mapear os resultados {fold_id: metrics_dict}
        }


    
    #Adiciona o nó na lista de mortos
    def setDead(self, rank):
        if rank in self.leader_context["alive_nodes"]:
            self.leader_context["alive_nodes"].remove(rank)
        if rank in self.leader_context["ready_nodes"]:
            self.leader_context["ready_nodes"].remove(rank)

        lost_fold = self.leader_context["active_assignments"].pop(rank, None)
        if lost_fold is not None:
            self.leader_context["pending_folds"].append(lost_fold)
        self.context_dirty = True

    #Remove o nó da lista de mortos
    def setAlive(self, rank):
        if rank not in self.leader_context["alive_nodes"]:
            self.leader_context["alive_nodes"].append(rank)
            self.context_dirty = True


    def setReady(self, rank):
        if rank not in self.leader_context["ready_nodes"]:
            self.leader_context["ready_nodes"].append(rank)
            self.context_dirty = True


    #Simula janela de quedas do nó
    def _node_esta_ativo(self):
        if not self.teste:
            return True

        working = self.teste.get("time_working", 0)
        timeout = self.teste.get("time_timeout", 0)
        cycle = working + timeout

        if cycle <= 0:
            return True

        elapsed = time.time() - self.start_time
        return (elapsed % cycle) < working
    
