from mpi4py import MPI
from logger import print_to_node as print

class MPIConnector:
    def __init__(self, comm):
        self.comm = comm

    #esperar receber
    def send(self, data, dest, tag):
        self.comm.send(data, dest=dest, tag=tag)

    #joga pra fila
    def isend(self, data, dest, tag):
        return self.comm.isend(data, dest=dest, tag=tag)
    

    #Função utilitaria só pra pegar mensagem sem travar
    def check_message(self, source, tag):
        if source is None:
            return None
        if self.comm.iprobe(source=source, tag=tag):
            
            try:
                return self.comm.recv(source=source, tag=tag)
            except Exception as e:
                try:
                    rank = self.comm.Get_rank()
                except Exception:
                    rank = -1
                print(f"[MPI Debug] Rank {rank} falhou ao receber de source={source}, tag={tag}: {type(e).__name__}: {e}")
                raise
        return None
    



    def check_message_all(self, source, tag):
        if source is None:
            source = MPI.ANY_SOURCE
        if tag is None:
            tag = MPI.ANY_TAG

        status = MPI.Status()
        if self.comm.iprobe(source=source, tag=tag, status=status):
            real_source = status.Get_source()
            real_tag = status.Get_tag()
            try:
                data = self.comm.recv(
                    source=real_source,
                    tag=real_tag,
                    status=status
                )
                return {
                    "source": real_source,
                    "tag": real_tag,
                    "data": data
                }
            except Exception as e:
                try:
                    rank = self.comm.Get_rank()
                except Exception:
                    rank = -1
                print(f"[MPI Debug] Rank {rank} falhou ao coletar de source={real_source}, tag={real_tag}: {type(e).__name__}: {e}")
                raise

        return None