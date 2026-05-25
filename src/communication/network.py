class MPIConnector:
    def __init__(self, comm):
        self.comm = comm

    def send(self, data, dest, tag):
        self.comm.send(data, dest=dest, tag=tag)

    def isend(self, data, dest, tag):
        return self.comm.isend(data, dest=dest, tag=tag)
    
    def check_message(self, source, tag):
        """Verifica se há mensagem de forma não-bloqueante e retorna se houver"""
        if source is None:
            return None
        if self.comm.iprobe(source=source, tag=tag):
            return self.comm.recv(source=source, tag=tag)
        return None