import threading
import time

class NodeContext:
    def __init__(self, rank, size):
        self.rank = rank
        self.size = size
        self.leader_rank = None
        self.last_heartbeat = time.time()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
        # Replicação de Estado (Redundância do Líder)
        self.context_dirty = False
        self.leader_context = {
            "epoch": 0,
            "pending_folds": [],
            "active_assignments": {},  # {worker_rank: fold_id}
            "completed_folds": {}      # {fold_id: metrics_dict}
        }