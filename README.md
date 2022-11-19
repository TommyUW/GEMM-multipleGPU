# GEMM-with-multipleGPU
This project can multiply dense matrices with multiple GPUs. Matrix-matrix multiplication has been a very popular problem in Parallel Programming. Usually, people design algorithm with one GPU. 
However, the performance can be increased a lot if we use multiple GPUs. 
In this program, I first divide matix A into two sub-matrices with the programming archetecture MPI. Then I send these two sub-matrices and two copies of matrix B into two GPUs for calculation.
My running time increased 30% comparing with program which runs on single GPU.
