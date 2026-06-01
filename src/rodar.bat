@echo off
call conda activate Distrifold
python.exe src/clear_locals.py
mpiexec -n 2 python.exe -B src/MPI_start.py