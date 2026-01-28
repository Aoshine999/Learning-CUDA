#include <vector>
#include <cuda_fp16.h>


/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length   
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
                        
}





template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int B_BLOCK_X = 32, int C_BLOCK_X = 16, int C_WARP_X = 8, int C_WARP_Y = 4>
__global__ void float4GEMMnoBC(float* A, float* B, float* C, const int M, const int K,
                               const int N) {
  __shared__ float As[Bk][Bm];  // 存储转置后的 tileA
  __shared__ float Bs[Bk][Bn];  // 存储 tileB

  // 计算 block 负责的 tileC 左上角元素的行列坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // 当前 thread 的编号（默认为一维 block 配置）
  int tid = threadIdx.x;

  /*------ tileA ------*/
  // 写入 A tile 时，block 中 thread 排布尺寸为 (A_BLOCK_X, blockSize / A_BLOCK_X) = (8, 32)
  constexpr int A_BLOCK_Y = blockSize / A_BLOCK_X;

  // 对于 tid 号线程，其位于 blockA 中的行列坐标为 (tid / A_BLOCK_X, tid % A_BLOCK_X)
  int A_THREAD_Y = tid / A_BLOCK_X;
  int A_THREAD_X = tid % A_BLOCK_X;

  /*------ tileB ------*/
  // 写入 B tile 时，block 中 thread 排布尺寸为 (B_BLOCK_X, blockSize / B_BLOCK_X) = (32, 8)
  constexpr int B_BLOCK_Y = blockSize / B_BLOCK_X;

  // 对于 tid 号线程，其位于 blockB 中的行列坐标为 (tid / B_BLOCK_X, tid % B_BLOCK_X)
  int B_THREAD_Y = tid / B_BLOCK_X;
  int B_THREAD_X = tid % B_BLOCK_X;

  /*------ tileC ------*/
  constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X;

  // 按 8*4 排列 warp

  // 计算当前 thread 属于哪个 warp，第几个 lane
  int warpId = tid / WARP_SIZE;
  int laneId = tid % WARP_SIZE;
  // 计算总共有几行几列 warp
  constexpr int C_WARP_DIM_X = C_BLOCK_X / C_WARP_X;

  // 计算当前 thread 所在 warp 在 block 中的 x, y 坐标
  int warpX = warpId % C_WARP_DIM_X;
  int warpY = warpId / C_WARP_DIM_X;

  // 计算当前 thread 在 warp 中的 x, y 坐标
  int laneY = laneId / C_WARP_X;
  int laneX = laneId % C_WARP_X;

  // 当前 thread 在 blockC 中的行列坐标为 (warpY * C_WARP_Y + laneY, warpX * C_WARP_X + laneX)
  int C_THREAD_Y = warpY * C_WARP_Y + laneY;
  int C_THREAD_X = warpX * C_WARP_X + laneX;

  // 每个 thread 负责 Tm * Tn 个元素计算
  constexpr int Tm = Bm / C_BLOCK_Y;
  constexpr int Tn = Bn / C_BLOCK_X;
  float Ct[Tm][Tn] = {0.0};

  // 存储 A 中列向量和 B 中行向量
  float regA[Tm] = {0.0f};
  float regB[Tn] = {0.0f};

  // K- Loop
  for (int k = 0; k < K; k += Bk) {
    /* ------ 读取 global memory，存入 shared memory ------ */
    // 使用跨步循环，行方向的 stride 为 A_BLOCK_Y, 列方向的 stride 为 A_BLOCK_X
#pragma unroll
    for (int i = A_THREAD_Y; i < Bm; i += A_BLOCK_Y) {
      int r = r0 + i;
#pragma unroll
      for (int j = A_THREAD_X; j < Bk; j += A_BLOCK_X) {
        int c = k + j;
        As[j][i ^ (4 * j)] = (r < M && c < K) ? A[r * K + c] : 0.f;  // 转置
      }
    }

    // 使用跨步循环，行方向的 stride 为 B_BLOCK_Y, 列方向的 stride 为 B_BLOCK_X
#pragma unroll
    for (int i = B_THREAD_Y; i < Bk; i += B_BLOCK_Y) {
      int r = k + i;
#pragma unroll
      for (int j = B_THREAD_X; j < Bn; j += B_BLOCK_X) {
        int c = c0 + j;

        Bs[i][j] = (r < K && c < N) ? B[r * N + c] : 0.f;
      }
    }

    __syncthreads();

    /* ------ 计算 tileA * tileB ------ */
    // p-Loop，先循环 k 维度，按向量外积的方式计算
#pragma unroll
    for (int p = 0; p < Bk; ++p) {
      // 向量化访存，存储 A 中列向量到 regA
#pragma unroll
      for (int i = 0; i < Tm / 4; ++i) {
        int r = (C_THREAD_Y + i * C_BLOCK_Y) * 4;
        FLOAT4(regA[i * 4]) = FLOAT4(As[p][r ^ (4 * p)]);
      }

      // 向量化访存，存储 B 中行向量到 regB
#pragma unroll
      for (int j = 0; j < Tn / 4; ++j) {
        int c = (C_THREAD_X + j * C_BLOCK_X) * 4;
        FLOAT4(regB[j * 4]) = FLOAT4(Bs[p][c]);
      }

      // 计算 regA 与 regB 的向量外积
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
#pragma unroll
        for (int j = 0; j < Tn; ++j) { Ct[i][j] += regA[i] * regB[j]; }
      }
    }

    __syncthreads();
  }

  // 将 Ct 写入 C
#pragma unroll
  for (int i = 0; i < Tm; ++i) {
    int r = r0 + 4 * C_THREAD_Y + i / 4 * 4 * C_BLOCK_Y + i % 4;
#pragma unroll
    for (int j = 0; j < Tn; ++j) {
      int c = c0 + 4 * C_THREAD_X + j / 4 * 4 * C_BLOCK_X + j % 4;

      if (r < M && c < N) { C[r * N + c] = Ct[i][j]; }
    }
  }
}


template<int Bm, int Bn>
__global__ void transposeSharedSwizzling(float* A, float* B, const int M, const int N) {
  __shared__ float tile[Bm][Bn];

  /* -------- 读取阶段 -------- */
  // (r0, c0) 表示 tile 内左上角元素在 matrixA 中的坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // thread y 方向负责：矩阵 A 的行，shared memory 的行
  // thread x 方向负责：矩阵 A 的列，shared memory 的列
  // shared memory 中的元素 tile[y][x ^ y] = A[r0 + y, c0 + x]
#pragma unroll
  for (int y = threadIdx.y; y < Bm; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
    int r = r0 + y;
    if (r >= M) break;

#pragma unroll
    for (int x = threadIdx.x; x < Bn; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
      int c = c0 + x;
      if (c < N) {
        tile[y][x ^ y] = A[r * N + c];  // 将 A[r0 + y, c0 + x] 写入 tile[y][x ^ y]
      }
    }
  }

  __syncthreads();  // 同步线程块

/* -------- 写入阶段 -------- */
// (c0, r0) 表示 tile 内左上角元素在 matrixB 中的坐标
// thread y 方向负责：矩阵 B 的行，shared memory 的列
// thread x 方向负责：矩阵 B 的列，shared memory 的行
// shared memory 中的元素 tile[x][x ^ y] = B[c0 + y, r0 + x]
#pragma unroll
  for (int y = threadIdx.y; y < Bn; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
    int c = c0 + y;
    if (c >= N) break;

#pragma unroll
    for (int x = threadIdx.x; x < Bm; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
      int r = r0 + x;
      if (r < M) { B[c * M + r] = tile[x][x ^ y]; }  // 将 tile[x][x ^ y] 写入 B[c0 + y, r0 + x]
    }
  }
}


