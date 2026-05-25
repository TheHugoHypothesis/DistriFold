import time
import numpy as np
from node_context import NodeContext
from communication.network import MPIConnector
from communication.communication_tags import *
class TorrentEngine:
    def __init__(self, context: NodeContext, connector: MPIConnector):
        self.context = context
        self.connector = connector
        self.chunks = {}  # Guarda {chunk_id: (X_chunk, y_chunk)}
        self.have = []    # Vetor de booleanos: o que eu já possuo localmente
        self.peer_haves = {}  # {rank: have_list}

    def distribute_as_leader(self, X, y):
        """Método rodado apenas pelo Líder para preparar e espalhar os pedaços iniciais"""
        size = self.context.size
        self.have = [True] * size  # O líder já tem todos os pedaços
        self.peer_haves = {i: [False] * size for i in range(size)}
        self.peer_haves[self.context.rank] = self.have
        
        # Divide as linhas do dataset igualmente entre os nós (incluindo o líder)
        X_splits = np.array_split(X, size, axis=0)
        y_splits = np.array_split(y, size, axis=0)
        # Guarda os pedaços locais do líder
        for i in range(size):
            self.chunks[i] = (X_splits[i], y_splits[i])
        print(f"[Torrent Líder] Dataset particionado em {size} chunks.")
        # 1. Envia Metadados para todos
        meta = {"total_chunks": size}
        for dest in range(size):
            if dest != self.context.rank:
                self.connector.send(meta, dest=dest, tag=TAG_TORRENT_META)
        # 2. Seeding Inicial (Envia exatamente 1 chunk único para cada worker)
        for dest in range(size):
            if dest != self.context.rank:
                chunk_payload = (X_splits[dest], y_splits[dest])
                self.connector.send(chunk_payload, dest=dest, tag=TAG_TORRENT_SEED)
                print(f"[Torrent Líder] Enviado Chunk Inicial {dest} para Worker {dest}")
        # 3. Age como Seed respondendo a pedidos dos workers até que todos terminem
        self._run_swarm_loop()

    def download_as_worker(self):
        """Método rodado pelos Workers para baixar os chunks e subir no enxame P2P"""
        size = self.context.size
        self.have = [False] * size
        self.peer_haves = {i: [False] * size for i in range(size)}
        self.peer_haves[self.context.leader_rank] = [True] * size
        
        # 1. Aguarda os metadados do Líder
        meta = None
        while meta is None:
            meta = self.connector.check_message(source=self.context.leader_rank, tag=TAG_TORRENT_META)
            time.sleep(0.05)
        
        total_chunks = meta["total_chunks"]
        # 2. Aguarda seu Chunk de Semente Inicial vindo do líder
        initial_payload = None
        while initial_payload is None:
            initial_payload = self.connector.check_message(source=self.context.leader_rank, tag=TAG_TORRENT_SEED)
            time.sleep(0.05)
        # Adiciona o próprio pedaço inicial ao inventário
        self.chunks[self.context.rank] = initial_payload
        self.have[self.context.rank] = True
        self.peer_haves[self.context.rank] = self.have
        print(f"[Worker {self.context.rank}] Recebi meu chunk inicial {self.context.rank} do Líder.")
        # 3. Entra no enxame de trocas P2P
        self._run_swarm_loop()
        # 4. Concatena os pedaços e reconstrói o dataset original
        X_complete = np.vstack([self.chunks[i][0] for i in range(total_chunks)])
        y_complete = np.concatenate([self.chunks[i][1] for i in range(total_chunks)])
        print(f"[Worker {self.context.rank}] Dataset 100% reconstruído com sucesso! Formato: {X_complete.shape}")
        
        return X_complete, y_complete

    def _run_swarm_loop(self):
        """Loop de trocas P2P compartilhado por Líder e Workers"""
        size = self.context.size
        last_broadcast = 0
        self.active_requests = []  # Guarda requisições não-bloqueantes para evitar garbage collection prematuro
        
        print(f"[Nó {self.context.rank}] Entrou no loop P2P Swarm. Inventário: {self.have}")
        
        # O loop roda enquanto existir qualquer nó que não tenha completado seu download
        while not self._all_nodes_completed():
            if self.context.stop_event.is_set():
                break
            now = time.time()
            # A cada 0.5s, divulga seu inventário (HAVE) para o cluster
            if now - last_broadcast > 0.5:
                # print(f"[Nó {self.context.rank}] Divulgando meu inventário: {self.have}") # Reduz verbosidade
                for dest in range(size):
                    if dest != self.context.rank:
                        self.connector.send(self.have, dest=dest, tag=TAG_TORRENT_HAVE)
                last_broadcast = now
            # Processa mensagens recebidas
            for source in range(size):
                if source == self.context.rank:
                    continue
                # A. Recebeu atualizações de inventário (HAVE) de outro nó
                peer_have = self.connector.check_message(source=source, tag=TAG_TORRENT_HAVE)
                if peer_have:
                    print(f"[Nó {self.context.rank}] Recebi HAVE do Nó {source}: {peer_have}")
                    self.peer_haves[source] = peer_have
                    # Se o outro nó tem um chunk que eu preciso, solicita!
                    for chunk_id, exists in enumerate(peer_have):
                        if exists and not self.have[chunk_id]:
                            print(f"[Nó {self.context.rank}] Solicitando Chunk {chunk_id} do Nó {source}")
                            self.connector.send(chunk_id, dest=source, tag=TAG_TORRENT_REQ)
                # B. Recebeu solicitação (REQUEST) de um chunk seu
                requested_chunk = self.connector.check_message(source=source, tag=TAG_TORRENT_REQ)
                if requested_chunk is not None:
                    print(f"[Nó {self.context.rank}] Recebi REQ de Chunk {requested_chunk} do Nó {source}")
                    if self.have[requested_chunk]:
                        payload = (requested_chunk, self.chunks[requested_chunk])
                        # IMPORTANTE: Como os fatiamentos do dataset são grandes (> Eager Limit de ~8-16KB),
                        # o envio DEVE ser não-bloqueante (isend) para evitar deadlocks de buffer do MPI!
                        req = self.connector.isend(payload, dest=source, tag=TAG_TORRENT_PIECE)
                        self.active_requests.append(req)
                # C. Recebeu um pedaço físico solicitado (PIECE)
                piece = self.connector.check_message(source=source, tag=TAG_TORRENT_PIECE)
                if piece:
                    chunk_id, chunk_data = piece
                    self.chunks[chunk_id] = chunk_data
                    self.have[chunk_id] = True
                    self.peer_haves[self.context.rank] = self.have  # Atualiza o meu registro na tabela
                    print(f"[Nó {self.context.rank}] Recebi e salvei Chunk {chunk_id} do Nó {source}! Novo HAVE: {self.have}")
            
            # Limpa requisições enviadas concluídas para evitar vazamento de recursos
            self.active_requests = [req for req in self.active_requests if not req.Test()]
            time.sleep(0.05)

    def _all_nodes_completed(self):
        """Verifica se todos os nós registraram que têm todos os chunks"""
        return all(all(h) for h in self.peer_haves.values())