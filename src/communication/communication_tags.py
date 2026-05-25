#Heartbeats
TAG_HELLO = 1
TAG_ACK = 2

# Tarefas de Fold
TAG_TASK = 3
TAG_RESULT = 4
TAG_STATE_SYNC = 5

#Torrenting P2P
TAG_TORRENT_META = 10   # Envio de metadados do torrent (ex: formato, quantidade de chunks)
TAG_TORRENT_SEED = 11   # Envio do pedaço inicial para cada nó (Seeding inicial)
TAG_TORRENT_HAVE = 12   # Envio do vetor de inventário (quais pedaços eu tenho)
TAG_TORRENT_REQ = 13    # Solicitação de um pedaço específico
TAG_TORRENT_PIECE = 14  # Envio do pedaço físico solicitado