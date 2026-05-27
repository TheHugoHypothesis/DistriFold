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
            "active_assignments": {},  #Mapear quem está fazendo o que {worker_rank: fold_id}
            "completed_folds": {}      #Mapear os resultados {fold_id: metrics_dict}
        }