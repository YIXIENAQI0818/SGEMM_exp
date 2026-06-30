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

// 一个 block 计算 32x32 的 C 子块（4个 16x16 tile）
#define BM 64
#define BN 64
#define BK 16

void compare(float* res1, float* res2, int n){
  int fail=0;
  for(int i=0; i<n; i++){
    float a,b;
    a = fabs(res1[i]);
    b = fabs(res2[i]);
    if((a<0.01)&&(b<0.01)) continue;
    if(i < 10)
      printf("i=%d %lf %lf\n",i,(double)a,(double)b);
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

__global__ void float_to_half(const float* in, half* out, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * n;
  if(idx < total){
    out[idx] = __float2half(in[idx]);
  }
}

/*
 * WMMA + Shared Memory
 * A,B: half row-major
 * C: float row-major
 * C = a*A*B + b*C
 *
 * blockDim.x = 128 (4 warps)
 * grid: ((n+31)/32, (n+31)/32)
 * shared: As[32x16], Bs[16x32]
 */
__global__ void sgemm_wmma_shared(const half* A,
                                  const half* B,
                                  float* C,
                                  int n,
                                  float a,
                                  float b)
{
  int blockRow = blockIdx.y * BM;
  int blockCol = blockIdx.x * BN;

  int tid    = threadIdx.x;
  int warpId = tid >> 5;  // 0..3
//   int lane   = tid & 31;

  // shared memory：先放 A(32x16)，再放 B(16x32)
  extern __shared__ half shmem[];
  half* As = shmem;                 // BM*BK
  half* Bs = shmem + (BM * BK);     // BK*BN

  // 4个warp对应2x2个16x16 tile
  int warpRow = warpId >> 2;        // 0,0,1,1
  int warpCol = warpId & 3;         // 0,1,0,1
  int tileRow = blockRow + warpRow * WMMA_M;
  int tileCol = blockCol + warpCol * WMMA_N;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  for(int k0 = 0; k0 < n; k0 += BK){

    // cooperative load A: 32x16 = 512 half
    for(int idx = tid; idx < BM*BK; idx += blockDim.x){
      int r = idx / BK;     // 0..31
      int c = idx % BK;     // 0..15
      int gr = blockRow + r;
      int gc = k0 + c;
      half v = __float2half(0.0f);
      if(gr < n && gc < n) v = A[gr * n + gc];
      As[r * BK + c] = v;
    }

    // cooperative load B: 16x32 = 512 half
    for(int idx = tid; idx < BK*BN; idx += blockDim.x){
      int r = idx / BN;     // 0..15
      int c = idx % BN;     // 0..31
      int gr = k0 + r;
      int gc = blockCol + c;
      half v = __float2half(0.0f);
      if(gr < n && gc < n) v = B[gr * n + gc];
      Bs[r * BN + c] = v;
    }

    __syncthreads();

    // 每个 warp 从 shared 加载自己的 16x16 A/B tile
    const half* A_tile = As + (warpRow * WMMA_M) * BK;   // stride BK=16
    const half* B_tile = Bs + (warpCol * WMMA_N);        // stride BN=32

    wmma::load_matrix_sync(a_frag, A_tile, BK);
    wmma::load_matrix_sync(b_frag, B_tile, BN);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    __syncthreads();
  }

  // alpha/beta 融合并写回
  if(tileRow < n && tileCol < n){
    float* C_tile = C + tileRow * n + tileCol;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old;
    wmma::load_matrix_sync(c_old, C_tile, n, wmma::mem_row_major);

    #pragma unroll
    for(int i=0;i<c_old.num_elements;i++){
      c_old.x[i] = a * acc_frag.x[i] + b * c_old.x[i];
    }

    wmma::store_matrix_sync(C_tile, c_old, n, wmma::mem_row_major);
  }
}

int main(){

  float* A           = (float*)malloc(sizeof(float) * (size_t)N * N);
  float* B           = (float*)malloc(sizeof(float) * (size_t)N * N);
  float* C_cpu       = (float*)malloc(sizeof(float) * (size_t)N * N);
  float* C_gpu_final = (float*)malloc(sizeof(float) * (size_t)N * N);

  float a = 0.5f, b = 0.3f;

  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      A[(size_t)i*N+j] = (float)rand()/(float)(RAND_MAX/a);
      B[(size_t)i*N+j] = (float)rand()/(float)(RAND_MAX/a);
      C_cpu[(size_t)i*N+j] = 0.0f;
      C_gpu_final[(size_t)i*N+j] = 0.0f;
    }
  }

  // CPU reference（全量，按你的要求不做 CPUcheck）
  for(int j=0;j<N;j++){
    for(int i=0;i<N;i++){
      float acc = b * C_cpu[(size_t)i*N+j];
      for(int k=0;k<N;k++){
        acc += a * A[(size_t)i*N+k] * B[(size_t)k*N+j];
      }
      C_cpu[(size_t)i*N+j] = acc;
    }
  }

  // GPU 内存
  float *A_gpu, *B_gpu, *C_gpu;
  half  *A_half_gpu, *B_half_gpu;

  cudaMalloc((void**)&A_gpu, sizeof(float)*(size_t)N*N);
  cudaMalloc((void**)&B_gpu, sizeof(float)*(size_t)N*N);
  cudaMalloc((void**)&C_gpu, sizeof(float)*(size_t)N*N);
  cudaMalloc((void**)&A_half_gpu, sizeof(half)*(size_t)N*N);
  cudaMalloc((void**)&B_half_gpu, sizeof(half)*(size_t)N*N);

  cudaMemcpy(A_gpu, A, sizeof(float)*(size_t)N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(float)*(size_t)N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C_gpu_final, sizeof(float)*(size_t)N*N, cudaMemcpyHostToDevice);

  // float -> half
  int total = N*N;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  float_to_half<<<blocks,threads>>>(A_gpu, A_half_gpu, N);
  float_to_half<<<blocks,threads>>>(B_gpu, B_half_gpu, N);
  cudaDeviceSynchronize();

  // kernel launch
  dim3 blockDim(512,1,1);   // 16 warps
  dim3 gridDim((N + BN - 1)/BN, (N + BM - 1)/BM, 1);
  size_t shmemBytes = (BM*BK + BK*BN) * sizeof(half);

  // 正确性验证
  sgemm_wmma_shared<<<gridDim,blockDim,shmemBytes>>>(A_half_gpu, B_half_gpu, C_gpu, N, a, b);
  cudaDeviceSynchronize();
  cudaMemcpy(C_gpu_final, C_gpu, sizeof(float)*(size_t)N*N, cudaMemcpyDeviceToHost);
  compare(C_cpu, C_gpu_final, N*N);

  // 性能测试
  double t1 = timestamp();
  for(int it=0; it<ITERATIONS; it++){
    sgemm_wmma_shared<<<gridDim,blockDim,shmemBytes>>>(A_half_gpu, B_half_gpu, C_gpu, N, a, b);
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
