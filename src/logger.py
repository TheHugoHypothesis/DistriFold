import os
import sys
import datetime
from mpi4py import MPI


# ==========================================
# Níveis de Log
# ==========================================
DEBUG = 0
INFO = 1
WARN = 2
ERROR = 3

_LEVEL_NAMES = {DEBUG: "DEBUG", INFO: "INFO", WARN: "WARN", ERROR: "ERROR"}
_CURRENT_LEVEL = INFO  # Padrão: mostra INFO e acima

PRINT_TO_CONSOLE = True

_original_print = print


def set_level(level):
    """Configura o nível mínimo de log. Mensagens abaixo são descartadas."""
    global _CURRENT_LEVEL
    _CURRENT_LEVEL = level


def _get_rank():
    try:
        if not MPI.Is_initialized():
            return 0
        return MPI.COMM_WORLD.Get_rank()
    except Exception:
        return 0


def _log(level, *args, **kwargs):
    """Motor central de logging com suporte a níveis."""
    if level < _CURRENT_LEVEL:
        return

    rank = _get_rank()
    level_name = _LEVEL_NAMES.get(level, "INFO")

    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    output_dir = os.path.join(project_root, f"src/Locals/Rank {rank}")
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(arg) for arg in args)

    log_line = f"[{time_str}] [{level_name}] {message}{end}"
    filename = f"{rank}.txt"
    file_path = os.path.join(output_dir, filename)

    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        _original_print(f"[Erro no Logger do Nó {rank}]: {e}", file=sys.stderr)

    if PRINT_TO_CONSOLE:
        _original_print(*args, **kwargs)


# ==========================================
# Funções públicas de log
# ==========================================
def debug(*args, **kwargs):
    """Log de nível DEBUG — detalhes internos, polling, heartbeats."""
    _log(DEBUG, *args, **kwargs)

def info(*args, **kwargs):
    """Log de nível INFO — operações normais importantes."""
    _log(INFO, *args, **kwargs)

def warn(*args, **kwargs):
    """Log de nível WARN — falhas detectadas, timeouts, recuperações."""
    _log(WARN, *args, **kwargs)

def error(*args, **kwargs):
    """Log de nível ERROR — erros que impedem operação normal."""
    _log(ERROR, *args, **kwargs)


# Backward compatibility: print_to_node mapeia para info
print_to_node = info
