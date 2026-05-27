import os
import sys
import datetime
from mpi4py import MPI


PRINT_TO_CONSOLE = True

_original_print = print


#Mudamos o print porque ele só tinha saída no final. Então exportamos para um TXT para ficar mais organizado
def print_to_node(*args, **kwargs):
    try:
        if not MPI.Is_initialized():
            rank = 0
        else:
            rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        rank = 0

    
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    output_dir = os.path.join(project_root, "src/Output")
    os.makedirs(output_dir, exist_ok=True)

    
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(arg) for arg in args)
    
    
    log_line = f"[{time_str}] {message}{end}"
    filename = f"{rank}.txt"
    file_path = os.path.join(output_dir, filename)

    
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        _original_print(f"[Erro no Logger do Nó {rank}]: Não foi possível gravar no arquivo. Detalhes: {e}", file=sys.stderr)

    
    if PRINT_TO_CONSOLE:
        _original_print(*args, **kwargs)
