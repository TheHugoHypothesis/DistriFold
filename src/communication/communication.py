import time
from communication.communication_tags import *
from node_context import NodeContext
from communication.network import MPIConnector

class CommunicationService:
    def __init__(self, context: NodeContext, connector: MPIConnector, on_leader_lost_callback):
        self.context = context
        self.connector = connector
        self.on_leader_lost = on_leader_lost_callback
        self.heartbeat_interval = 1.0
        self.timeout_seconds = 3.0
    def run(self):
        # A eleição inicial é disparada via callback
        self.on_leader_lost()
        while not self.context.stop_event.is_set():
            self.tick()
    def tick(self):
        if self.context.rank == self.context.leader_rank:
            self._leader_send()
            self._leader_collect()
            time.sleep(self.heartbeat_interval)
        else:
            self._worker_receive()
            self._worker_check_timeout()
    def _leader_send(self):
        # 1. Envia batimentos cardíacos
        for dest in range(self.context.size):
            if dest == self.context.leader_rank:
                continue
            self.connector.send("PING", dest=dest, tag=TAG_HELLO)
            
        # 2. Replicação de contexto redundante se o estado mudou (dirty)
        with self.context.lock:
            dirty = self.context.context_dirty
            payload = dict(self.context.leader_context)
            
        if dirty:
            print(f"[Líder {self.context.rank}] Redundância: Sincronizando contexto (epoch {payload['epoch']}) com os backups...")
            for dest in range(self.context.size):
                if dest == self.context.leader_rank:
                    continue
                self.connector.send(payload, dest=dest, tag=TAG_STATE_SYNC)
            with self.context.lock:
                self.context.context_dirty = False

    def _leader_collect(self):
        end_time = time.time() + 0.5
        while time.time() < end_time:
            if self.context.stop_event.is_set():
                break
            for source in range(self.context.size):
                if source == self.context.leader_rank:
                    continue
                msg = self.connector.check_message(source=source, tag=TAG_ACK)
                if msg:
                    print(f"Líder recebeu ACK de {source}")
            time.sleep(0.05)

    def _worker_receive(self):
        if self.context.leader_rank is None:
            self.on_leader_lost()
            return
            
        # A. Verifica batimentos cardíacos
        msg = self.connector.check_message(source=self.context.leader_rank, tag=TAG_HELLO)
        if msg:
            print(f"Worker {self.context.rank} recebeu PING do líder")
            self.connector.send("ACK", dest=self.context.leader_rank, tag=TAG_ACK)
            with self.context.lock:
                self.context.last_heartbeat = time.time()
                
        # B. Recebe sincronização de contexto redundante (Active Backup)
        state_msg = self.connector.check_message(source=self.context.leader_rank, tag=TAG_STATE_SYNC)
        if state_msg:
            with self.context.lock:
                if state_msg["epoch"] > self.context.leader_context["epoch"]:
                    self.context.leader_context = state_msg
                    print(f"Worker {self.context.rank} sincronizou contexto para epoch {state_msg['epoch']}")

    def _worker_check_timeout(self):
        with self.context.lock:
            elapsed = time.time() - self.context.last_heartbeat
        if elapsed > self.timeout_seconds:
            print(f"Worker {self.context.rank} detectou timeout do líder!")
            self.on_leader_lost() # Dispara a eleição sem acoplamento direto!
        else:
            time.sleep(0.1)