#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>

#define N 4096
#define ITERATIONS 10
#define BLOCK_SIZE 16
using namespace std;

__global__ void sgemm(float *A, float *B, float *C,
                      int n, float a, float b) 
  {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.0f;

    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) 
    {
        int A_row = row;
        int A_col = t * BLOCK_SIZE + threadIdx.x;
        int B_row = t * BLOCK_SIZE + threadIdx.y;
        int B_col = col;

        if (A_row < n && A_col < n)
            As[threadIdx.y][threadIdx.x] = A[A_row * n + A_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (B_row < n && B_col < n)
            Bs[threadIdx.y][threadIdx.x] = B[B_row * n + B_col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)  
        {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) 
    {
        float c_old = C[row * n + col];
        C[row * n + col] = a * acc + b * c_old;
    }
}


void compare(float* res1, float* res2, int n){
  int fail=0;
  for(int i=0; i<n; i++){
    float a,b;
    if(res1[i]<0)
      a=res1[i]*(-1);
    else 
      a=res1[i];
    if(res2[i]<0)
      b=res2[i]*(-1);
    else 
      b=res2[i];
    if((a<0.01)&&(b<0.01)){
      continue;
    }
    if(i<10)
      printf("i=%d %lf %lf\n",i,a,b);
    float diff=(a-b)/(a+0.000001);
    if(diff<0)
      diff=diff*(-1);
    if(diff>0.0005)
      fail++;
  }
  printf("Number of errors: %d\n", fail);
}

double timestamp(){
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

int main(){
  float* A         = (float*)malloc(sizeof(float) * N * N);
  float* B         = (float*)malloc(sizeof(float) * N * N);
  float* C_cpu     = (float*)malloc(sizeof(float) * N * N);
  float* C_gpu_final = (float*)malloc(sizeof(float) * N * N);
  //float A[N][N], B[N][N], C_cpu[N][N], C_gpu_final[N][N];
  float a=0.5, b=0.3;
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      A[i*N+j]=(float)rand()/(float)(RAND_MAX/a);
      B[i*N+j]=(float)rand()/(float)(RAND_MAX/a);
      C_cpu[i*N+j]=0;
      C_gpu_final[i*N+j]=0;
    }
  }

  for(int j=0; j<N; j++){
    for(int i=0; i<N; i++){
      C_cpu[i*N+j]+=b*C_cpu[i*N+j];
      for(int k=0; k<N; k++){
        C_cpu[i*N+j] += a*A[i*N+k]*B[k*N+j];
      }
    }
  }

  float *A_gpu;
  float *B_gpu;
  float *C_gpu;
  cudaMalloc((void **)&A_gpu, sizeof(float)*N*N);
  cudaMalloc((void **)&B_gpu, sizeof(float)*N*N);
  cudaMalloc((void **)&C_gpu, sizeof(float)*N*N);
  cudaMemcpy(A_gpu, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C_gpu_final, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((size_t)ceil( ((float)N) / ((float)block.x) ), (size_t)ceil( ((float)N) / ((float)block.y)) );

  sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, a, b);
  cudaDeviceSynchronize();
  cudaMemcpy(C_gpu_final, C_gpu, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
  compare(C_cpu, C_gpu_final, N*N);

  double time1=timestamp();
  for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){

    sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, a, b);

  }
  cudaDeviceSynchronize();
  double time2=timestamp();

  double time = (time2-time1)/ITERATIONS;
  double gn = (double)N/1000;
  double gflops = (double)2*gn*gn*gn;
  double gflopsPerSecond = (double)gflops/time;
  double GB = (double)(gn)*gn*4/1000;
  double GBpS = (double)GB/time;
  printf("GFLOPS/s=%lf\n",gflopsPerSecond );
  printf("GB/s=%lf\n",GBpS);
  printf("GFLOPS=%lf\n",gflops);
  printf("GB=%lf\n",GB);
  printf("time(s)=%lf\n",time);
  printf("\n");

  free(A);
  free(B);
  free(C_cpu);
  free(C_gpu_final);
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  return 0;
}
