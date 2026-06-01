import time
from logger import print_to_node as print
import numpy as np
from node_context import NodeContext
from .network import MPIConnector
from .communication_tags import *
import threading
import os

#comunicação só de espalhar o dataset (pra organizar)
class TorrentEngine:
    def __init__(self, context: NodeContext, connector: MPIConnector):
        self.context = context
        self.connector = connector

        self.chunks = {}        #Guarda {chunk_id: (X_chunk, y_chunk)}
        self.have = []          #Vetor de booleanos: o que eu já possuo localmente
        self.peer_haves = {}    #{rank: have_list}
        self.torrent_thread = threading.Thread(target=self._run_swarm_loop, daemon=True)


    #Só lider faz a divisão dos primeiros pedaços
    def distribute_as_leader(self, X, y, dataset_id, folds):
        while not self.context._node_esta_ativo():
            time.sleep(0.1)

        size = self.context.size
        self.have = [True] * folds  # O líder já tem todos os pedaços
        self.peer_haves = {i: [False] * folds for i in range(size)}
        self.peer_haves[self.context.rank] = self.have
        self.total_chunks = folds
        self.dataset_id = dataset_id

        #Divide as linhas do dataset igualmente entre os nós (incluindo o líder)
        X_splits = np.array_split(X, folds, axis=0)
        y_splits = np.array_split(y, folds, axis=0)
        
        
        #Guarda os pedaços locais do líder
        for i in range(self.total_chunks):
            self.chunks[i] = (X_splits[i], y_splits[i])
        print(f"[Torrent Líder] Dataset particionado em {self.total_chunks} chunks.")
        
        


        #Passa a agir como seed normalmente
        if not self.torrent_thread.is_alive():
            self.torrent_thread.start()

    
    #Baixar os primeiros dataset do lider
    def download_as_worker(self):

        size = self.context.size

        
        # pede os metadados do lider para ver o ID
        print(f"[Worker {self.context.rank}] Pedindo metadados ao líder...")



        #Pega metadado do lider
        meta = None
        while meta is None:
            time.sleep(0.5)

            with self.context.lock:
                leader = self.context.leader_rank

            if not self.context._node_esta_ativo() or leader is None:
                continue

            meta = self.connector.check_message(source=leader, tag=TAG_TORRENT_META)
            if meta is None:

                # reenvia pedido se demorar pra responder
                self.connector.isend(self.context.rank, dest=leader, tag=TAG_TORRENT_META_REQ)
            else:
                print(f"[Worker {self.context.rank}] Recebi metadados do Líder")
                break



        self.total_chunks = meta["total_chunks"]
        self.have = [False] * self.total_chunks
        self.peer_haves = {i: [False] * self.total_chunks for i in range(size)} #inicializa tudo como falso
        self.peer_haves[self.context.leader_rank] = [True] * self.total_chunks  #exceto o lider que é tudo true


        #Caso de erros apenas
        dataset_id = meta.get("dataset_id")

        
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_dir = os.path.join(src_dir, "Locals", f"Rank {self.context.rank}")
        file_path = os.path.join(local_dir, f"{dataset_id}.npz")


        #Caso ele já tenha o dataset baixado:
        if os.path.exists(file_path):
            print(f"[Worker {self.context.rank}] Dataset '{dataset_id}' encontrado localmente! Carregando de {file_path}")
            data = np.load(file_path, allow_pickle=True)
            X_complete = data["X"]
            y_complete = data["y"]

            #reconstrói chunks locais para poder agir como seed do torrent
            X_splits = np.array_split(X_complete, self.total_chunks, axis=0)
            y_splits = np.array_split(y_complete, self.total_chunks, axis=0)
            for i in range(self.total_chunks):
                self.chunks[i] = (X_splits[i], y_splits[i])

            self.have = [True] * self.total_chunks
            self.peer_haves[self.context.rank] = self.have

            #inicia swarm loop como seed
            

            
            if not self.torrent_thread.is_alive():
                self.torrent_thread.start()

            # avisa o líder que está pronto
            if not self.context._node_esta_ativo():
                time.sleep(0.1)
            else:
                self.connector.isend(self.context.rank, dest=self.context.leader_rank, tag=TAG_NODE_READY)
            print(f"[Worker {self.context.rank}] Dataset carregado localmente e pronto para o treino!")
            return X_complete, y_complete



        #Solicita um chunk inicial ao líder
        chunk_id = self.context.rank
        if chunk_id >= self.total_chunks:
            chunk_id = self.context.rank % self.total_chunks

        initial_payload = None
        last_req_time = 0
        while initial_payload is None:
            if not self.context._node_esta_ativo():
                time.sleep(0.1)
                continue

            now = time.time()
            if now - last_req_time > 0.5:
                self.connector.isend(chunk_id, dest=self.context.leader_rank, tag=TAG_TORRENT_REQ)
                last_req_time = now

            initial_payload = self.connector.check_message(source=self.context.leader_rank, tag=TAG_TORRENT_PIECE)
            time.sleep(0.05)

        if isinstance(initial_payload, tuple) and len(initial_payload) == 2 and isinstance(initial_payload[0], int):
            chunk_id, chunk_data = initial_payload
        else:
            chunk_data = initial_payload

        self.chunks[chunk_id] = chunk_data
        self.have[chunk_id] = True
        self.peer_haves[self.context.rank] = self.have
        print(f"[Worker {self.context.rank}] Recebi meu chunk inicial {chunk_id} do Líder.")

        #entra no loop de ficar trocando, mesmo se completar
        
        if not self.torrent_thread.is_alive():
            self.torrent_thread.start()

        while not all(self.have):
            time.sleep(0.1)

        #Concatena os pedaços e reconstrói o df
        X_complete = np.vstack([self.chunks[i][0] for i in range(self.total_chunks)])
        y_complete = np.concatenate([self.chunks[i][1] for i in range(self.total_chunks)])
        
        #salva na pasta local
        os.makedirs(local_dir, exist_ok=True)
        np.savez(file_path, X=X_complete, y=y_complete)
        print(f"[Worker {self.context.rank}] Dataset '{dataset_id}' salvo com sucesso em {file_path}")


        #Avisar o lider que terminou
        print(f"[Worker {self.context.rank}] Dataset 100% reconstruído com sucesso! Formato: {X_complete.shape}")
        
        return X_complete, y_complete





    #loop de trocas comparilhadas
    def _run_swarm_loop(self):
        size = self.context.size
        last_broadcast = 0
        self.active_requests = []  #Guarda os pedido de dados que pedi
        self.dataset_id = None
        
        
        print(f"[Nó {self.context.rank}] Entrou no loop P2P Swarm. Inventário: {self.have}")
        


        #Loop do swarm
        while not self.context.stop_event.is_set():
            if not self.context._node_esta_ativo():
                time.sleep(0.1)
                continue
            now = time.time()


            # A cada 0.5s, divulga seu inventário (HAVE) para o cluster
            if now - last_broadcast > 0.5:
                # print(f"[Nó {self.context.rank}] Divulgando meu inventário: {self.have}") 
                for dest in range(size):
                    if dest != self.context.rank:
                        self.connector.isend(self.have, dest=dest, tag=TAG_TORRENT_HAVE)
                last_broadcast = now


            #Processa mensagens recebidas
            for source in range(size):
                if source == self.context.rank:
                    continue

                #Processa pedido de metadados do torrent (apenas se for líder)
                meta_req = self.connector.check_message(source=source, tag=TAG_TORRENT_META_REQ)
                if meta_req is not None:
                    
                    if self.context.rank == self.context.leader_rank and self.dataset_id is not None:
                        print(f"[Líder] Enviando metadados respondendo ao pedido do Nó {source}")
                        meta = {"total_chunks": self.total_chunks, "dataset_id": self.dataset_id}
                        self.connector.isend(meta, dest=source, tag=TAG_TORRENT_META)


                #Atualizar inventário de outro nó
                peer_have = self.connector.check_message(source=source, tag=TAG_TORRENT_HAVE)
                if peer_have:

                    #TODO descomentar isso
                    #print(f"[Nó {self.context.rank}] Recebi HAVE do Nó {source}: {peer_have}")

                    self.peer_haves[source] = peer_have

                    #Solicita caso apareceu chunk que precisa
                    for chunk_id, exists in enumerate(peer_have):
                        if exists and not self.have[chunk_id]:
                            print(f"[Nó {self.context.rank}] Solicitando Chunk {chunk_id} do Nó {source}")
                            self.connector.isend(chunk_id, dest=source, tag=TAG_TORRENT_REQ)


                #Request de um Chunk meu
                requested_chunk = self.connector.check_message(source=source, tag=TAG_TORRENT_REQ)
                if requested_chunk is not None:
                    print(f"[Nó {self.context.rank}] Recebi REQ de Chunk {requested_chunk} do Nó {source}")
                    if self.have[requested_chunk]:
                        payload = (requested_chunk, self.chunks[requested_chunk])


                        #aparaentemtne para dados maiores que 16kb o MPI usa protoloco Rendezvous, e ai o SEND trava de fato, por isso o ISEND
                        req = self.connector.isend(payload, dest=source, tag=TAG_TORRENT_PIECE)
                        self.active_requests.append(req)


                #Recebeu pedaço solicitado
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


    #Função para verficiar se todo mundo tem o dataset completo
    def _all_nodes_completed(self):
        return all(all(h) for h in self.peer_haves.values())