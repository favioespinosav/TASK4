all:
        nvcc -O3 -std=c++11 -I/opt/ibm/spectrum_mpi/include -I/opt/ibm/nvcc -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm main.cu -o cuda
        bsub -n $(1) -gpu "num=2"  -o stdout4.txt -e error.txt OMP_NUM_THREADS=1 mpiexec ./cuda $(2)  $(3)
