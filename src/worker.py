import time
from logger import print_to_node as print
from sklearn.model_selection import KFold
from node_context import NodeContext
from communication.network import MPIConnector
from communication.torrent import TorrentEngine
from communication.communication_tags import *
from MLP import train_fold_from_arrays
class WorkerWork:
    def __init__(self, context: NodeContext, connector: MPIConnector):
        self.context = context
        self.connector = connector
        self.torrent = TorrentEngine(context, connector)
    def run(self):


        #Aguarda a eleição do líder inicial antes de iniciar qualquer trabalho
        while self.context.leader_rank is None:
            time.sleep(0.1)

        #Por enquanto o Líder não deve agir como worker para evitar conflitos de processamento e deadlocks
        #TODO clocar lider para processar também
        if self.context.rank == self.context.leader_rank:
            print(f"[Nó {self.context.rank}] Sou o Líder, ignorando papel de Worker.")
            return

        print(f"[Worker {self.context.rank}] Iniciando download P2P do Dataset...")
        
        #Faz o download de todas as fatias do dataset pelo p2p
        X, y = self.torrent.download_as_worker()

        #Inicializa o KFold TODO CONFERIR SE ESTÁ IDENTICO AO LIDER
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(X))
        print(f"[Worker {self.context.rank}] Pronto e aguardando ordens de Folds do Líder.")
        
        while not self.context.stop_event.is_set():
            #Escuta ordens do lider
            fold_id = self.connector.check_message(source=self.context.leader_rank, tag=TAG_TASK)
            if fold_id is not None:
                if fold_id == -1:
                    print(f"[Worker {self.context.rank}] Sinal de encerramento recebido do Líder. Desconectando...")
                    self.context.stop_event.set()
                    break
                
                print(f"[Worker {self.context.rank}] Treinando Fold {fold_id}...")
                


                # Pega os índices do fold de forma determinística localmente
                train_idx, test_idx = splits[fold_id]
                config = {
                    "h1": 64, "h2": 16, "lr": 0.005,
                    "epochs": 100, "batch_size": 32
                }
                
                
                #Treina localmente usando a classe do MPL
                res = train_fold_from_arrays(X, y, train_idx, test_idx, config, fold_id=fold_id)
                
                #Envia só as métricas finais e pesos de volta
                self.connector.send(res, dest=self.context.leader_rank, tag=TAG_RESULT)
                print(f"[Worker {self.context.rank}] Concluiu e enviou métricas do Fold {fold_id}")
            time.sleep(0.1)
