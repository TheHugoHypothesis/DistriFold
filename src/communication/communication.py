import time
import threading
from logger import info as print, debug, warn
from .communication_tags import *
from node_context import NodeContext
from mpi4py import MPI
from .network import MPIConnector

#comunicação padrão
class CommunicationService:
    def __init__(self, context: NodeContext, connector: MPIConnector, role_changer):
        self.context = context
        self.connector = connector
        self.role_changer = role_changer
        self.heartbeat_interval = 1.0
        self.timeout_seconds = 3.0
        self.in_election = False
        self.queue_lock = threading.Lock()
        self.outbox = {"leader": [], "worker": []}
        self.inbox = []
        self.leader_tag = "LEADER"
        self._isend_list = []


    #Simula janela de quedas do nó
    def run(self):
        # eleição inicial realista
        self.start_election()
        self.context.last_internet_tick = time.time()


        while not self.context.stop_event.is_set():
            self.tick()

    # ticka todas as funções de comunicação
    def tick(self):
        now = time.time()
        if not self.context._node_esta_ativo():
            return #Se o nó tiver em modo queda, ele para de tickar comunicações
            

        #Dados que vem de "fora"
        #RETORNO DE QUEDA
        if (now - self.context.last_internet_tick > 1):

            time.sleep(3)
            self.collect() #Limpa tudo que chegou, pois estava sem conexão

            with self.context.lock:
                if not self.context.recovering:
                    warn(f"[Nó {self.context.rank}] Voltando de queda...")
                    self.context.recovering = True
                    self.context.old_leader = self.context.leader_rank
                    self.context.leader_rank = None
                    self.in_election = False
                    self.role_changer()


                    with self.queue_lock:
                        self.outbox = {"leader": [], "worker": []}
                        self.inbox = []        
        #parte do código para mudar.

        #sempre checa se alguem perguntou por lider e resonde
        #TODO Jogar isso em outro lugar
        for source in range(self.context.size):
            if source == self.context.rank:
                continue

            msg_leader_query = self.Poll(source=source, tag=TAG_LEADER_QUERY)
            if msg_leader_query is not None:
                with self.context.lock:
                    leader_rank = self.context.leader_rank
                if leader_rank is not None:
                    self.connector.isend(leader_rank, dest=source, tag=TAG_LEADER_ANNOUNCE)


            msg_context_req = self.Poll(source=source, tag=TAG_CONTEXT_REQ)
            if msg_context_req is not None:
                if self.context.rank == self.context.leader_rank:
                    with self.context.lock:
                        ctx_payload = dict(self.context.leader_context)
                    self.connector.isend(ctx_payload, dest=source, tag=TAG_STATE_SYNC)


        #sempre checa se alguém pediu eleição pra poder participar, isso tem prioridade
        for source in range(self.context.size):
            if source == self.context.rank:
                continue

            msg_start = self.Poll(source=source, tag=TAG_ELECTION_START)

            if msg_start:
                with self.context.lock:
                    ctx_payload = dict(self.context.leader_context)
                self.connector.isend(ctx_payload, dest=source, tag=TAG_ELECTION_CONTEXT)
                self.connector.isend(self.context.rank, dest=source, tag=TAG_ELECTION_RANK)
                debug(f"[Nó {self.context.rank}] Alguém pediu eleição")
                self.start_election()
                #return


        #-----




        #Envio de Leader
        if self.context.rank == self.context.leader_rank:
            self._leader_send()
            time.sleep(self.heartbeat_interval)
        else:
            self._worker_send()


        #Envio de todo Resto
        self._flush_outbox()

        self.collect()

        with self.context.lock:
            self.context.last_internet_tick = time.time()

        time.sleep(0.05)






    # realiza eleição com contagem de tempo coletando os ranks que aparecerem
    def start_election(self):
        with self.context.lock:
            if self.in_election:
                return
            self.in_election = True
            self.context.old_leader = self.context.leader_rank
            self.context.leader_rank = None


        print(f"[Nó {self.context.rank}] Iniciando eleição...")
        for dest in range(self.context.size):
            if dest == self.context.rank:
                continue
                
            self.connector.isend("ELECTION_START", dest=dest, tag=TAG_ELECTION_START)
            
        #inicia contagem de tempo
        if self.context.has_dataset_completed:
            lowest = self.context.rank
        else: lowest = None
        contexts = []


        
        with self.context.lock:
            contexts.append(dict(self.context.leader_context))
        

            
        end_time = time.time() + 2.5 #segundos que dura a eleição
        while time.time() < end_time:
            if self.context.stop_event.is_set():
                break

            # garante que a fila interna seja alimentada mesmo sem o loop principal
            self.collect()

            # coleta respostas e start de outros
            for source in range(self.context.size):
                if source == self.context.rank:
                    continue

                # se alguém pediu eleição no meio, respondemos com nosso número, caso temos o dataset
                msg_start = self.Poll(source=source, tag=TAG_ELECTION_START)

                if msg_start:
                    if not self.context.has_dataset_completed:
                        warn(f"[Nó {self.context.rank}] Não tenho dataset, não vou me candidatar")

                    else:
                        with self.context.lock:
                            ctx_payload = dict(self.context.leader_context)
                        self.connector.isend(ctx_payload, dest=source, tag=TAG_ELECTION_CONTEXT)
                        self.connector.isend(self.context.rank, dest=source, tag=TAG_ELECTION_RANK)

                
                #coleta rank do outro
                rank = self.Poll(source=source, tag=TAG_ELECTION_RANK)
                if lowest is None:
                    lowest = rank

                if rank is not None:
                    if rank < lowest:
                        lowest = rank

                ctx_msg = self.Poll(source=source, tag=TAG_ELECTION_CONTEXT)
                if ctx_msg is not None:
                    contexts.append(ctx_msg)

            time.sleep(0.05)

        if lowest is None:
            warn(f"[Nó {self.context.rank}] Eleição falhou: nenhum candidato elegível.")
            with self.context.lock:
                self.in_election = False
            return
        
        if lowest == self.context.rank:
            print(f"[Nó {self.context.rank}] Fim da contagem de tempo. Sou o menor ativo. Novo líder eleito!")

            

            with self.context.lock:
                if contexts:
                    self.context.leader_context = max(contexts, key=lambda c: c.get("epoch", 0))
                    self.context.context_dirty = True


            for dest in range(self.context.size):
                if dest != self.context.rank:
                    self.connector.isend(self.context.rank, dest=dest, tag=TAG_LEADER_ANNOUNCE)


            

        else:
            print(f"[Nó {self.context.rank}] Fim da contagem de tempo. Menor é o {lowest}. Novo líder eleito!")
        

        with self.context.lock:
            self.context.leader_rank = lowest
            self.context.last_heartbeat = time.time()
            if contexts:
                self.context.leader_context = max(contexts, key=lambda c: c.get("epoch", 0))


        with self.context.lock:
            self.in_election = False
        self.role_changer()
  


    def _worker_send(self):
        if self.context.recovering:
            return

        with self.context.lock:
            leader_rank = self.context.leader_rank

        if leader_rank is None:
            return

        msg = self.Poll(source=leader_rank, tag=TAG_HELLO)
        if msg:
            debug(f"Worker {self.context.rank} recebeu PING do líder")

            with self.context.lock:
                is_ready = self.context.ready_to_work
            if is_ready:
                self.enqueue("worker", dest=self.leader_tag, tag=TAG_NODE_READY, payload="ACK")
            else:
                self.enqueue("worker", dest=self.leader_tag, tag=TAG_ACK, payload="ACK")

            with self.context.lock:
                self.context.last_heartbeat = time.time()

        state_msg = self.Poll(source=leader_rank, tag=TAG_STATE_SYNC)
        if state_msg:
            with self.context.lock:
                if state_msg["epoch"] > self.context.leader_context["epoch"]:
                    self.context.leader_context = state_msg
                    print(f"Worker {self.context.rank} sincronizou contexto para epoch {state_msg['epoch']}")


        with self.context.lock:
            has_leader = self.context.leader_rank is not None
            elapsed = time.time() - self.context.last_heartbeat
        if has_leader and elapsed > self.timeout_seconds:
            warn(f"Worker {self.context.rank} detectou timeout do líder!")
            self.start_election()
            time.sleep(0.1)



    # Envio em BROADCAST pra todo mundo, nos dois tipos de mensagem
    def _leader_send(self):
        #Heartbeat
        for dest in range(self.context.size):
            if dest == self.context.leader_rank:
                continue
            self.connector.isend("PING", dest=dest, tag=TAG_HELLO)
            
            
        # Clonar o contexto, se mudou
        with self.context.lock:
            dirty = self.context.context_dirty
            payload = dict(self.context.leader_context)
            
        if dirty:
            print(f"[Líder {self.context.rank}] Redundância: Sincronizando contexto (epoch {payload['epoch']}) com os backups...")
            for dest in range(self.context.size):
                if dest == self.context.leader_rank:
                    continue

                self.connector.isend(payload, dest=dest, tag=TAG_STATE_SYNC) 

            with self.context.lock:
                self.context.context_dirty = False






    def _flush_outbox(self):
        for role in list(self.outbox.keys()):
            with self.queue_lock:
                pending = self.outbox[role]
                self.outbox[role] = []

            remaining = []
            for entry in pending:
                dest = self._resolve_dest(entry["dest"])
                if dest is None:
                    remaining.append(entry)
                    continue
                request = self.connector.isend(entry["data"], dest=dest, tag=entry["tag"])
                self._isend_list.append(request)

                #Dibrlando garbager collector que matava a request
                alive = []
                for req in self._isend_list:
                    if not req.Test():
                        alive.append(req)
                self._isend_list = alive


            if remaining:
                with self.queue_lock:
                    self.outbox[role].extend(remaining)



    def _resolve_dest(self, dest):
        if dest == self.leader_tag:
            return self.context.leader_rank
        return dest


    #Fila de envio do role
    def enqueue(self, role, dest, tag, payload):
        with self.queue_lock:
            self.outbox[role].append({"dest": dest, "tag": tag, "data": payload})




    #Polling da inbox do role
    def Poll(self, source=None, tag=None):
        with self.queue_lock:
            for i, msg in enumerate(self.inbox):

                if source is not None and msg["source"] != source:
                    continue

                if tag is not None and msg["tag"] != tag:
                    continue

                return self.inbox.pop(i)["data"]
        return None




    def collect(self):
        for source in range(self.context.size):
            while True:
                msg = self.connector.check_message_all(source=source, tag=MPI.ANY_TAG)
                if msg is None:
                    break
                
                with self.queue_lock:
                    self.inbox.append({
                        "source": msg["source"],
                        "tag": msg["tag"],
                        "data": msg["data"]
                    })



