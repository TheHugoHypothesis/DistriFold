import time
from logger import print_to_node as print
from sklearn.model_selection import KFold
from node_context import NodeContext
from communication.network import MPIConnector
from communication.torrent import TorrentEngine
from communication.communication_tags import *
from MLP import train_fold_from_arrays





class WorkerWork:
    def __init__(self, context: NodeContext, connector: MPIConnector, comm_service=None):
        self.context = context
        self.connector = connector
        self.comm_service = comm_service
        self.torrent = TorrentEngine(context, connector)
        self.next_query_rank = (context.rank + 1) % context.size
        self.last_query_time = 0
    
    def run(self):
        

        

        #Se tiver voltando ele não faz eleição
        if not self.context.recovering and self.context.leader_rank is None:
            print(f"[Worker {self.context.rank}] Iniciando e disparando eleição para descobrir líder...")
            self.comm_service.start_election()


        #Aguarda a eleição do líder inicial antes de iniciar qualquer trabalho
        #TODO ver onde realocar isso se de fato for necessário
        # while self.context.leader_rank is None:
        #     time.sleep(0.1)



        #TODO clocar lider para processar também, agora ele só faz um ou outro
        if self.context.rank == self.context.leader_rank:
            print(f"[Nó {self.context.rank}] Sou o Líder, ignorando papel de Worker.")
            return

        
        print('passei aqui ==========')
        
        #Tratamento de reotrno
        while self.context.recovering:
            time.sleep(0.1)
            self.tratar_retorno()



        #Faz o download ou carrega localmente as fatias do dataset
        print(f"[Worker {self.context.rank}] Iniciando verificação do Dataset...")
        X, y = self.torrent.download_as_worker()
        


        self.context.ready_to_work = True
        print(f"[Worker {self.context.rank}] Pronto e aguardando ordens de Folds do Líder.")

        
        print('passei aqui ==========2')
        
        
        


        #Loop principal
        while not self.context.stop_event.is_set():
            time.sleep(0.1)

            if self.context.leader_rank == self.context.rank:
                print(f"[Nó {self.context.rank}] Virou líder, encerrando papel de Worker.")
                return
            
            # if not self.context._node_esta_ativo():
            #     time.sleep(0.1)
            #     continue

    

            #Escuta ordens do liders
            with self.context.lock:
                if self.context.leader_rank is None:
                    continue

            msg = None
            if self.comm_service:
                msg = self.comm_service.Poll(source=self.context.leader_rank, tag=TAG_TASK)
            
            if msg is None:
                continue


            fold_id = msg['fold_id']


            if fold_id is not None:
                if fold_id == -1:
                    print(f"[Worker {self.context.rank}] Sinal de encerramento recebido do Líder. Desconectando...")
                    self.context.stop_event.set()
                    break
                
                print(f"[Worker {self.context.rank}] Treinando Fold {fold_id}...")
                
                
                config = msg['config']
                fold_config = config['FOLD']
                kf = KFold(**fold_config)
                splits = list(kf.split(X))

                # Pega os índices do fold de forma determinística localmente
                #print(fold_id)
                #print(fold_config)

                train_idx, test_idx = splits[fold_id]
                config_MLP = config['MLP']
                
                
                #Treina localmente usando a classe do MPL
                res = train_fold_from_arrays(X, y, train_idx, test_idx, config_MLP, fold_id=fold_id)
                
 
                
                # print(f"[DEBUG RESULT] Rank {self.context.rank} preparando TAG_RESULT para líder={self.context.leader_rank} | tipo={type(res).__name__} | chaves={list(res.keys()) if isinstance(res, dict) else 'N/A'} | fold_id={res.get('fold_id') if isinstance(res, dict) else 'N/A'}")
                self.comm_service.enqueue("worker", dest=self.comm_service.leader_tag, tag=TAG_RESULT, payload=res)
                # with self.comm_service.queue_lock:
                #     print(self.comm_service.outbox['worker'])

                # print(f"[DEBUG RESULT] Rank {self.context.rank} enfileirou TAG_RESULT para líder={self.context.leader_rank}")


                print(f"[Worker {self.context.rank}] Concluiu e enviou métricas do Fold {fold_id}")
            




    #TRATAMENTO DE RETORNO
    def tratar_retorno(self):
        print('Tentando Retornar, perguntando lider atual para os vizinhos')
        now = time.time()
        if now - self.last_query_time > 0.5:
            if self.next_query_rank == self.context.rank:
                self.next_query_rank = (self.next_query_rank + 1) % self.context.size
            self.comm_service.enqueue("worker", dest=self.next_query_rank, tag=TAG_LEADER_QUERY, payload="WHO")
            self.last_query_time = now
            self.next_query_rank = (self.next_query_rank + 1) % self.context.size

        leader_reply = self.comm_service.Poll(tag=TAG_LEADER_ANNOUNCE)
        if leader_reply is not None:
            with self.context.lock:
                self.context.leader_rank = leader_reply
                self.context.last_heartbeat = time.time()
                print(f'Descobri líder, ele é {leader_reply}')


        if self.context.leader_rank == self.context.rank and self.context.rank == 0:
            print(f'Lider sou eu devido falta do dataset, usando meu contexto')
            self.context.recovering = False
            self.context.last_heartbeat = time.time()
            return


        if self.context.leader_rank is not None:
            self.comm_service.enqueue("worker", dest=self.context.leader_rank, tag=TAG_CONTEXT_REQ, payload="CTX")
            ctx_msg = self.comm_service.Poll(source=self.context.leader_rank, tag=TAG_STATE_SYNC)
            

            if ctx_msg:
                with self.context.lock:
                    self.context.leader_context = ctx_msg
                    self.context.recovering = False
                    self.context.last_heartbeat = time.time()

                print(f"Worker {self.context.rank} sincronizou contexto após retorno")

            else: 
                print('sem retorno')

                