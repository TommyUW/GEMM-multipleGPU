#include<iostream>
#include<stdlib.h>
#include"multiply.h"
#include "mpi.h"
#include<sys/time.h>
using namespace std;
#define MASTER_TO_SLAVE_TAG 1
#define SLAVE_TO_MASTER_TAG 4
#define N 256
int low_bound;
int upper_bound;
int portion;
float a[N*N];
float b[N*N];
float r[N*N];
MPI_Status status;
MPI_Request request;
void InitializeMatrix()
{
	for(int i=0;i<N;i++)
		for(int j=0;j<N;j++)
		{
			a[i*N+j]=j+1;
			b[i*N+j]=j+1;
		}
}


void Check()
{
	float r_seq[N*N];
	for(int i=0;i<N;i++)
                for(int j=0;j<N;j++)
			r_seq[i*N+j]=0;
	for(int i=0;i<N;i++)
                for(int j=0;j<N;j++)
                        for(int k=0;k<N;k++)
                                r_seq[i*N+j]+=a[i*N+k]*b[k*N+j];
	int flag=1;
	for(int i=0;i<N*N;i++)
		if(r[i]!=r_seq[i])
		{	flag=0;
			cout<<"Error"<<endl;
			break;
		}
	if(flag==1)
		cout<<"Programme successfully!"<<endl;

}

int main(int argc,char *argv[])
{
	int rank,num_of_process;
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&num_of_process);
	int row_per_proc=N/num_of_process;
	cout<<"rank: "<<rank<<endl;
	
	float gpu_time;
	if(rank==0)
	{
		InitializeMatrix();
	}
	struct timeval t1,t2;
	gettimeofday(&t1,NULL);
	float *ap=(float*)malloc(row_per_proc*N*sizeof(float));
	float *rp=(float*)malloc(row_per_proc*N*sizeof(float));

	MPI_Scatter(a,row_per_proc*N,MPI_FLOAT,ap,row_per_proc*N,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(&b,N*N,MPI_FLOAT,0,MPI_COMM_WORLD);


	multiply(ap,b,rp,row_per_proc,N,rank,&gpu_time);
	
	MPI_Gather(rp,row_per_proc*N,MPI_FLOAT,r,row_per_proc*N,MPI_FLOAT,0,MPI_COMM_WORLD);
	free(ap);
        free(rp);
    
	gettimeofday(&t2,NULL);
	double time = (t2.tv_sec-t1.tv_sec)*1000.0+(t2.tv_usec-t1.tv_usec)/1000.0;	
	cout<<"time: "<<time<<endl;
	if(rank==0)
	{
		Check();
	}
	//MPI_Reduce(&cpu_time,&sum,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	//MPI_Reduce(&gpu_time,&sum,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}



