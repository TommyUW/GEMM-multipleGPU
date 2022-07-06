#include"cuda_runtime.h"
#include<iostream>
#include<stdlib.h>

using namespace std;

__global__ void matrixMul(float *a,float *b,float *c,int size)
{
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int column = blockIdx.x*blockDim.x+threadIdx.x;
	int sum=0;
	for(int i=0;i<size;i++)
	{
		sum+=a[row*size+i]*b[i*size+column];
	}
	c[row*size+column]=sum;
}

void multiply(float *h_a,float *h_b,float *h_c,int row_per_proc,int n,int id,float *gpu_time)
{
	
	cout<<endl<<endl;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float *d_a,*d_b,*d_r;
	cudaSetDevice(id);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,id);
	cout<<"Device["<<id<<"] "<<deviceProp.name<<endl;
	cudaEventRecord(start,0);
	cudaMalloc(&d_a,row_per_proc*n*sizeof(float));
        cudaMalloc(&d_b,n*n*sizeof(float));
	cudaMalloc(&d_r,row_per_proc*n*sizeof(float));
	
	cudaMemcpy(d_a,h_a,row_per_proc*n*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,n*n*sizeof(float),cudaMemcpyHostToDevice);
	
	cout<<endl<<endl;
	int threads_per_block =16;
	dim3 block_size(threads_per_block,threads_per_block);
	dim3 grid_size(n/block_size.x,row_per_proc/block_size.y);

	matrixMul<<<grid_size,block_size>>>(d_a,d_b,d_r,n);
	cudaEventRecord(stop,0);
	cudaMemcpy(h_c,d_r,row_per_proc*n*sizeof(float),cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_r);
	cudaEventElapsedTime(gpu_time,start,stop);
}
