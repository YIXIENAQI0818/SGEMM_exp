#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>

#include <mma.h>
#include <cuda_fp16.h>

using namespace std;
using namespace nvcuda;

#define N 4096
#define ITERATIONS 10

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

/* ------------------ 工具函数（与你原来一致） ------------------ */

void compare(float* res1, float* res2, int n){
  int fail=0;
  for(int i=0; i<n; i++){
    float a,b;
    a = fabs(res1[i]);
    b = fabs(res2[i]);
    if((a<0.01)&&(b<0.01)) continue;
    if(i < 10)
      printf("i=%d %lf %lf\n",i,a,b);
    float diff=fabs((a-b)/(a+1e-6));
    if(diff>0.0005) fail++;
  }
  printf("Number of errors: %d\n", fail);
}

double timestamp(){
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

/* ------------------ float -> half 转换 ------------------ */

__global__ void float_to_half(const float* in, half* out, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * n;
  if(idx < total){
    out[idx] = __float2half(in[idx]);
  }
}

/* ------------------ WMMA Tensor Core SGEMM ------------------ */
/*
 * 一个 block = 一个 warp
 * 一个 warp 负责一个 16x16 的 C tile
 * C = a*A*B + b*C
 */
__global__ void sgemm_wmma(const half* A,
                           const half* B,
                           float* C,
                           int n,
                           float a,
                           float b)
{
  // blockIdx 对应 tile 坐标
  int tileRow = blockIdx.y;
  int tileCol = blockIdx.x;

  // 一个 block 只有一个 warp
  if(threadIdx.x >= 32) return;

  // WMMA fragments
  wmma::fragment<wmma::matrix_a,
                 WMMA_M, WMMA_N, WMMA_K,
                 half, wmma::row_major> a_frag;

  wmma::fragment<wmma::matrix_b,
                 WMMA_M, WMMA_N, WMMA_K,
                 half, wmma::row_major> b_frag;

  wmma::fragment<wmma::accumulator,
                 WMMA_M, WMMA_N, WMMA_K,
                 float> acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // K 维分块
  for(int k = 0; k < n; k += WMMA_K){
    const half* A_tile = A + (tileRow * WMMA_M) * n + k;
    const half* B_tile = B + k * n + (tileCol * WMMA_N);

    wmma::load_matrix_sync(a_frag, A_tile, n);
    wmma::load_matrix_sync(b_frag, B_tile, n);

    // Tensor Core MMA
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // 处理 alpha / beta
  float* C_tile = C + (tileRow * WMMA_M) * n + (tileCol * WMMA_N);

  wmma::fragment<wmma::accumulator,
                 WMMA_M, WMMA_N, WMMA_K,
                 float> c_old;

  wmma::load_matrix_sync(c_old, C_tile, n, wmma::mem_row_major);

  #pragma unroll
  for(int i=0;i<c_old.num_elements;i++){
    c_old.x[i] = a * acc_frag.x[i] + b * c_old.x[i];
  }

  wmma::store_matrix_sync(C_tile, c_old, n, wmma::mem_row_major);
}

/* ------------------ main ------------------ */

int main(){

  float* A         = (float*)malloc(sizeof(float) * N * N);
  float* B         = (float*)malloc(sizeof(float) * N * N);
  float* C_cpu     = (float*)malloc(sizeof(float) * N * N);
  float* C_gpu_final = (float*)malloc(sizeof(float) * N * N);

  float a = 0.5f, b = 0.3f;

  // 初始化
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      A[i*N+j] = (float)rand()/(float)(RAND_MAX/a);
      B[i*N+j] = (float)rand()/(float)(RAND_MAX/a);
      C_cpu[i*N+j] = 0.0f;
      C_gpu_final[i*N+j] = 0.0f;
    }
  }

  // CPU reference
  for(int j=0;j<N;j++){
    for(int i=0;i<N;i++){
      float acc = b * C_cpu[i*N+j];
      for(int k=0;k<N;k++){
        acc += a * A[i*N+k] * B[k*N+j];
      }
      C_cpu[i*N+j] = acc;
    }
  }

  // GPU 内存
  float *A_gpu, *B_gpu, *C_gpu;
  half  *A_half_gpu, *B_half_gpu;

  cudaMalloc((void**)&A_gpu, sizeof(float)*N*N);
  cudaMalloc((void**)&B_gpu, sizeof(float)*N*N);
  cudaMalloc((void**)&C_gpu, sizeof(float)*N*N);
  cudaMalloc((void**)&A_half_gpu, sizeof(half)*N*N);
  cudaMalloc((void**)&B_half_gpu, sizeof(half)*N*N);

  cudaMemcpy(A_gpu, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C_gpu_final, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  // float -> half
  int total = N*N;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  float_to_half<<<blocks,threads>>>(A_gpu, A_half_gpu, N);
  float_to_half<<<blocks,threads>>>(B_gpu, B_half_gpu, N);
  cudaDeviceSynchronize();

  // WMMA kernel launch
  dim3 block(32,1,1);                 // 一个 warp
  dim3 grid(N/16, N/16, 1);            // 一个 tile 对应一个 block

  // 正确性验证
  sgemm_wmma<<<grid,block>>>(A_half_gpu, B_half_gpu, C_gpu, N, a, b);
  cudaDeviceSynchronize();
  cudaMemcpy(C_gpu_final, C_gpu, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
  compare(C_cpu, C_gpu_final, N*N);

  // 性能测试
  double t1 = timestamp();
  for(int it=0; it<ITERATIONS; it++){
    sgemm_wmma<<<grid,block>>>(A_half_gpu, B_half_gpu, C_gpu, N, a, b);
  }
  cudaDeviceSynchronize();
  double t2 = timestamp();

  double time = (t2-t1)/ITERATIONS;
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

  // 清理
  free(A);
  free(B);
  free(C_cpu);
  free(C_gpu_final);
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  cudaFree(A_half_gpu);
  cudaFree(B_half_gpu);

  return 0;
}
