import time
from logger import print_to_node as print
from .communication_tags import *
from node_context import NodeContext
from .network import MPIConnector

#comunicação padrão
class CommunicationService:
    def __init__(self, context: NodeContext, connector: MPIConnector, on_leader_lost_callback):
        self.context = context
        self.connector = connector
        self.on_leader_lost = on_leader_lost_callback
        self.heartbeat_interval = 1.0
        self.timeout_seconds = 3.0

    def run(self):
        #fizemos eleição inicial é disparada via callback
        self.on_leader_lost()
        while not self.context.stop_event.is_set():
            self.tick()



    #ticka todas as funções de comunoicação
    def tick(self):
        if self.context.rank == self.context.leader_rank:
            self._leader_send()
            self._leader_collect()
            time.sleep(self.heartbeat_interval)
        else:
            self._worker_receive()
            self._worker_check_timeout()


    #Envio em BROADCAST pra todo mundo, nos dois tipos de mensagem
    def _leader_send(self):
        #Heartbeat
        for dest in range(self.context.size):
            if dest == self.context.leader_rank:
                continue
            self.connector.send("PING", dest=dest, tag=TAG_HELLO)
            
        #Clonar o contexto, se mudou
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


    #Só recebe os ACK do heartbeat dos workers
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


        #Recebe Heartbeat do Lider e já envia ACK
        msg = self.connector.check_message(source=self.context.leader_rank, tag=TAG_HELLO)
        if msg:
            print(f"Worker {self.context.rank} recebeu PING do líder")
            self.connector.send("ACK", dest=self.context.leader_rank, tag=TAG_ACK)
            with self.context.lock:
                self.context.last_heartbeat = time.time()
                
        #Recebe contexto do Lider
        state_msg = self.connector.check_message(source=self.context.leader_rank, tag=TAG_STATE_SYNC)
        if state_msg:
            with self.context.lock:
                if state_msg["epoch"] > self.context.leader_context["epoch"]:
                    self.context.leader_context = state_msg
                    print(f"Worker {self.context.rank} sincronizou contexto para epoch {state_msg['epoch']}")


    #Função só pra chegar se lider caiu. (sem precisar usar o receive)
    def _worker_check_timeout(self):
        with self.context.lock:
            elapsed = time.time() - self.context.last_heartbeat
        if elapsed > self.timeout_seconds:
            print(f"Worker {self.context.rank} detectou timeout do líder!")
            self.on_leader_lost()
        else:
            time.sleep(0.1)