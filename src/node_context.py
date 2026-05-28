import threading
import time



#Facilitar gereciamento global dos Nós
class NodeContext:
    def __init__(self, rank, size):
        self.rank = rank
        self.size = size
        self.leader_rank = None
        self.last_heartbeat = time.time()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
        #Informação compartilhada pra se o líder cair
        self.context_dirty = False
        self.leader_context = {
            "epoch": 0,
            "pending_folds": [],
            "last_heartbeat":[],
            "alive_nodes": list(range(size)), #Considera que inicia com todo mundo vivo
            "ready_nodes": [],
            "active_assignments": {},  #Mapear quem está fazendo o que {worker_rank: fold_id}
            "completed_folds": {}      #Mapear os resultados {fold_id: metrics_dict}
        }


    
    #Adiciona o nó na lista de mortos
    def setDead(self, rank):
        if rank not in self.leader_context["dead_nodes"]:
            self.leader_context["alive_nodes"].remove(rank)
            lost_fold = self.leader_context["active_assignments"].pop(rank, None)
            self.leader_context["pending_folds"].append(lost_fold) if lost_fold is not None else None
            self.context_dirty = True

    #Remove o nó da lista de mortos
    def setAlive(self, rank):
        if rank not in self.leader_context["alive_nodes"]:
            self.leader_context["alive_nodes"].append(rank)
            self.context_dirty = True

    #Checa se o nó está vivo
    def isAlive(self, rank):
        return rank in self.leader_context["alive_nodes"]


    def setReady(self, rank):
        if rank not in self.leader_context["ready_nodes"]:
            self.leader_context["ready_nodes"].append(rank)
            self.context_dirty = True