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

// block computes 32x32 C tile = 4 warps
#define BM 32
#define BN 32
#define BK 16

// ---- utilities (keep your style) ----
void compare(float* res1, float* res2, int n){
  int fail=0;
  for(int i=0; i<n; i++){
    float a = fabs(res1[i]);
    float b = fabs(res2[i]);
    if((a<0.01f)&&(b<0.01f)) continue;
    if(i < 10) printf("i=%d %lf %lf\n", i, (double)a, (double)b);
    float diff = fabs((a-b)/(a+1e-6f));
    if(diff>0.0005f) fail++;
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

// ---------------- cp.async helpers (sm80+) ----------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr, bool pred){
  if(!pred) return;  // 需要 pred 时就直接跳过（避免 PTX predicate 写法的坑）
  unsigned int smem = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit(){
  asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all(){
  asm volatile("cp.async.wait_group 0;\n" ::);
}
#else
__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr, bool pred){
  if(pred){
    int4 v = *reinterpret_cast<const int4*>(gmem_ptr);
    *reinterpret_cast<int4*>(smem_ptr) = v;
  }
}
__device__ __forceinline__ void cp_async_commit(){}
__device__ __forceinline__ void cp_async_wait_all(){}
#endif


/*
 * WMMA + Shared + Double Buffer Pipeline
 * - 4 warps/block (128 threads)
 * - block covers C(32x32)
 * - shared buffers: As[2][32x16], Bs[2][16x32]
 */
__global__ void sgemm_wmma_pipe(const half* A,
                               const half* B,
                               float* C,
                               int n,
                               float alpha,
                               float beta)
{
  int blockRow = blockIdx.y * BM;
  int blockCol = blockIdx.x * BN;

  int tid    = threadIdx.x;
  int warpId = tid >> 5;  // 0..3

  // 2-stage shared buffers
  extern __shared__ half shmem[];
  half* As0 = shmem;
  half* Bs0 = As0 + (BM*BK);
  half* As1 = Bs0 + (BK*BN);
  half* Bs1 = As1 + (BM*BK);

  // warp -> tile mapping (2x2)
  int warpRow = warpId >> 1;   // 0,0,1,1
  int warpCol = warpId & 1;    // 0,1,0,1
  int tileRow = blockRow + warpRow * WMMA_M;
  int tileCol = blockCol + warpCol * WMMA_N;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  // number of K tiles
  int numKTiles = n / BK; // assumes n multiple of 16

  // ---- stage 0: preload kTile=0 into (As0,Bs0) ----
  {
    // load A_sub (32x16) and B_sub (16x32) using cp.async (16 bytes each)
    // total bytes: A_sub=32*16*2=1024B, B_sub=16*32*2=1024B
    // 16B per op -> 64 ops each
    // we map each thread to one 16B chunk across both A and B
    int chunksA = (BM*BK*2) / 16; // bytes / 16 = 64
    int chunksB = (BK*BN*2) / 16; // 64

    // A chunks
    for(int c = tid; c < chunksA; c += blockDim.x){
      // each chunk is 16B = 8 half
      int halfIndex = (c * 16) / 2;               // starting half index in As
      int r = halfIndex / BK;                     // row in [0..31]
      int k = halfIndex % BK;                     // col in [0..15], always aligned to 8-half boundary
      const half* gptr = A + (blockRow + r) * n + (0 + k);
      half* sptr = As0 + r * BK + k;
      cp_async_16B(sptr, gptr, true);
    }

    // B chunks
    for(int c = tid; c < chunksB; c += blockDim.x){
      int halfIndex = (c * 16) / 2;               // starting half index in Bs
      int r = halfIndex / BN;                     // row in [0..15]
      int ccol = halfIndex % BN;                  // col in [0..31], aligned
      const half* gptr = B + (0 + r) * n + (blockCol + ccol);
      half* sptr = Bs0 + r * BN + ccol;
      cp_async_16B(sptr, gptr, true);
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
  }

  int stage = 0;

  // ---- main loop over K tiles ----
  for(int kt = 0; kt < numKTiles; ++kt){
    // prefetch next tile (kt+1) into the other stage (double buffer)
    if(kt + 1 < numKTiles){
      half* AsNext = (stage == 0) ? As1 : As0;
      half* BsNext = (stage == 0) ? Bs1 : Bs0;
      int kNext = (kt + 1) * BK;

      int chunksA = (BM*BK*2) / 16; // 64
      int chunksB = (BK*BN*2) / 16; // 64

      for(int c = tid; c < chunksA; c += blockDim.x){
        int halfIndex = (c * 16) / 2;
        int r = halfIndex / BK;
        int k = halfIndex % BK;
        const half* gptr = A + (blockRow + r) * n + (kNext + k);
        half* sptr = AsNext + r * BK + k;
        cp_async_16B(sptr, gptr, true);
      }

      for(int c = tid; c < chunksB; c += blockDim.x){
        int halfIndex = (c * 16) / 2;
        int r = halfIndex / BN;
        int ccol = halfIndex % BN;
        const half* gptr = B + (kNext + r) * n + (blockCol + ccol);
        half* sptr = BsNext + r * BN + ccol;
        cp_async_16B(sptr, gptr, true);
      }

      cp_async_commit();
    }

    // compute current tile using current stage
    half* AsCur = (stage == 0) ? As0 : As1;
    half* BsCur = (stage == 0) ? Bs0 : Bs1;

    const half* A_tile = AsCur + (warpRow * WMMA_M) * BK; // stride BK
    const half* B_tile = BsCur + (warpCol * WMMA_N);      // stride BN

    wmma::load_matrix_sync(a_frag, A_tile, BK);
    wmma::load_matrix_sync(b_frag, B_tile, BN);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // wait next prefetch before switching stage
    if(kt + 1 < numKTiles){
      cp_async_wait_all();
      __syncthreads();
      stage ^= 1;
    }
  }

  // write back C tile with alpha/beta
  if(tileRow < n && tileCol < n){
    float* C_tile = C + tileRow * n + tileCol;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old;
    wmma::load_matrix_sync(c_old, C_tile, n, wmma::mem_row_major);

    #pragma unroll
    for(int i=0;i<c_old.num_elements;i++){
      c_old.x[i] = alpha * acc_frag.x[i] + beta * c_old.x[i];
    }

    wmma::store_matrix_sync(C_tile, c_old, n, wmma::mem_row_major);
  }
}

int main(){

  float* A           = (float*)malloc(sizeof(float) * (size_t)N * N);
  float* B           = (float*)malloc(sizeof(float) * (size_t)N * N);
  float* C_cpu       = (float*)malloc(sizeof(float) * (size_t)N * N);
  float* C_gpu_final = (float*)malloc(sizeof(float) * (size_t)N * N);

  float alpha = 0.5f, beta = 0.3f;

  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      A[(size_t)i*N+j] = (float)rand()/(float)(RAND_MAX/alpha);
      B[(size_t)i*N+j] = (float)rand()/(float)(RAND_MAX/alpha);
      C_cpu[(size_t)i*N+j] = 0.0f;
      C_gpu_final[(size_t)i*N+j] = 0.0f;
    }
  }

  // CPU reference (full, as you requested)
  for(int j=0;j<N;j++){
    for(int i=0;i<N;i++){
      float acc = beta * C_cpu[(size_t)i*N+j];
      for(int k=0;k<N;k++){
        acc += alpha * A[(size_t)i*N+k] * B[(size_t)k*N+j];
      }
      C_cpu[(size_t)i*N+j] = acc;
    }
  }

  // GPU memory
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

  // launch
  dim3 blockDim(128,1,1);  // 4 warps
  dim3 gridDim((N + BN - 1)/BN, (N + BM - 1)/BM, 1);

  // shared bytes for 2-stage buffers:
  // As0 (BM*BK) + Bs0 (BK*BN) + As1 (BM*BK) + Bs1 (BK*BN)
  size_t shmemBytes = 2 * (BM*BK + BK*BN) * sizeof(half);

  // correctness
  sgemm_wmma_pipe<<<gridDim,blockDim,shmemBytes>>>(A_half_gpu, B_half_gpu, C_gpu, N, alpha, beta);
  cudaDeviceSynchronize();
  cudaMemcpy(C_gpu_final, C_gpu, sizeof(float)*(size_t)N*N, cudaMemcpyDeviceToHost);
  compare(C_cpu, C_gpu_final, N*N);

  // timing
  double t1 = timestamp();
  for(int it=0; it<ITERATIONS; it++){
    sgemm_wmma_pipe<<<gridDim,blockDim,shmemBytes>>>(A_half_gpu, B_half_gpu, C_gpu, N, alpha, beta);
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
