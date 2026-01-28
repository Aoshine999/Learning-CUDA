#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
#define WARP_SIZE 32
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T>
__global__ void traceKernel(const T* d_input, T* d_result, size_t rows, size_t cols, size_t min_dim) {
  size_t step = cols + 1;
  __shared__ T partialSum[32];

  size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
  size_t warpId = threadIdx.x / WARP_SIZE;
  size_t laneId = threadIdx.x % WARP_SIZE;
  size_t numWarps = blockDim.x / WARP_SIZE;

  // 1.加载数据
  T val = 0;
  if(global_id < min_dim) {
    val = d_input[global_id * step];
  }

  // 2. Warp-level reduction
  val = warpReduceSum(val);


  // 3. 每个 warp 的第一个线程写入共享内存
  if (laneId == 0) {
    partialSum[warpId] = val;
  }
  __syncthreads();

  // 4. Block-level reduction
  if (warpId == 0) {
    val = (laneId < numWarps) ? partialSum[laneId] : 0;
    val = warpReduceSum(val);
    if (laneId == 0) {
      atomicAdd(d_result, val);
    }
  }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  size_t min_dim = (rows < cols) ? rows : cols;
  size_t size = rows * cols * sizeof(T);

  // 1. 分配 GPU 内存
  T* d_input;
  T* d_result;
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_result, sizeof(T));

  // 2. 将数据从 Host 拷贝到 Device
  cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

  // 初始结果设为 0
  T h_zero = 0;
  cudaMemcpy(d_result, &h_zero, sizeof(T), cudaMemcpyHostToDevice);
  

  // 3. 调用CUDA kernel
  int blockSize = 1024;
  int numBlocks = (min_dim + blockSize - 1) / blockSize;
  traceKernel<T><<<numBlocks, blockSize>>>(d_input, d_result, rows, cols, min_dim);

  
  // 4. 将结果从 Device 拷回 Host
  T h_result;
  cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);

  // 5. 释放 GPU 内存
  cudaFree(d_input);
  cudaFree(d_result);

  return h_result;
}

#define MAX_HEAD_DIM 128 // 最大头维度
#define BR 64  // Block Row size (针对 Q 的分块)
#define BC 64  // Block Col size (针对 K, V 的分块)

template <typename T>
__global__ void flash_attn_kernel(
    const T* __restrict__ Q, 
    const T* __restrict__ K, 
    const T* __restrict__ V, 
    T* __restrict__ O,
    const int batch_size, 
    const int tgt_len, 
    const int src_len, 
    const int q_heads, 
    const int kv_heads, 
    const int head_dim, 
    const float softmax_scale,
    const bool is_causal
) {
  // 1. 确定当前线程处理的 Block 坐标
  // Grid 维度: x=tgt_len/BR, y=q_heads, z=batch_size
  const int tx = threadIdx.x; // 线程在 Block 内的索引
  const int q_block_idx = blockIdx.x; // 针对 Q 的分块索引
  const int head_idx = blockIdx.y; // Query 头索引
  const int batch_idx = blockIdx.z; // Batch 索引

  // 2. GQA 索引映射
  // 多个 Q head 共享同一个 KV head。
  const int kv_head_idx = head_idx / (q_heads / kv_heads);

  // 3. 内存偏移量计算 [batch, tgt_len, head, dim]
  const int stride_q_b = tgt_len * q_heads * head_dim;
  const int stride_q_s = q_heads * head_dim;
  const int stride_q_h = head_dim;

  // [batch, src_len, head, dim]
  const int stride_k_b = src_len * kv_heads * head_dim;
  const int stride_k_s = kv_heads * head_dim;
  const int stride_k_h = head_dim;


  // 当前 Batch 和 Head 的偏移值
  const int q_offset_base = batch_idx * stride_q_b + head_idx * stride_q_h;
  const int k_offset_base = batch_idx * stride_k_b + kv_head_idx * stride_k_h;
  const int v_offset_base = batch_idx * stride_k_b + kv_head_idx * stride_k_h; // V 和 K 维度一致
  const int o_offset_base = batch_idx * stride_q_b + head_idx * stride_q_h;

  // 4. 定义共享内存
  extern __shared__ float sram[]; // 动态分配共享内存
  float *s_Q = sram; // BR x head_dim
  float *s_K = sram + BR * head_dim;
  float *s_V = sram + BR * head_dim + BC * head_dim; // BC x head_dim


  // 
  //5. 初始化 Online Softmax 累加器
  float m_i = -INFINITY;
  float l_i = 0.0f;
  float acc_o[MAX_HEAD_DIM] = {0.0f}; // 用于存储输出结果的累加值
  
  //6. 加载数据块到共享内存

  const int q_start = q_block_idx * BR;
  const int q_len = (q_start + BR > tgt_len) ? (tgt_len - q_start) : BR;


  // TODO : 加载数据需要进行优化
  // for(int i = 0; i < BR; i++){
  //   for(int d = 0; d < head_dim; d++){
  //     if(tx == 0){
  //       s_Q[i * head_dim + d] = (float)Q[q_offset_base + (q_start + i) * stride_q_s + d];
  //     }
  //   }
  // }

  // 3. 加载 Q Tile (协作加载)
  // 关键修正：利用所有线程加载数据
  // TODO 有点别扭
  for (int i = tx; i < BR * head_dim; i += blockDim.x) {
      int r = i / head_dim;
      int c = i % head_dim;
      // 边界检查填充 0
      if (r < q_len) {
          s_Q[i] = (float)Q[q_offset_base + (q_start + r) * stride_q_s + c];
      } else {
          s_Q[i] = 0.0f; 
      }
  }

  __syncthreads();

  // 7. 遍历 K,V的所有Tiles
  for(int j_block = 0; j_block < (src_len + BC - 1) / BC; j_block++){
    const int k_start = j_block * BC;
    const int k_len = (k_start + BC > src_len) ? (src_len - k_start) : BC;

    //TODO: 加载 K,V 需要进行优化
    // 加载 K,V 到共享内存
    // for(int i = 0; i < BC; i++){
    //   for(int d = 0; d < head_dim; d++){
    //     if(k_start + i < src_len && tx == 0){
    //       s_K[i * head_dim + d] = (float)K[k_offset_base + (k_start + i) * stride_k_s + d];
    //       s_V[i * head_dim + d] = (float)V[v_offset_base + (k_start + i) * stride_k_s + d];
    //     }
    //   }
    // }

    for(int i = tx; i < BC * head_dim; i += blockDim.x){
      int r = i / head_dim;
      int c = i % head_dim;
      // 边界检查填充 0
      if (r < k_len) {
          s_K[i] = (float)K[k_offset_base + (k_start + r) * stride_k_s + c];
          s_V[i] = (float)V[v_offset_base + (k_start + r) * stride_k_s + c];
      } else {
          s_K[i] = 0.0f; 
          s_V[i] = 0.0f; 
      }
    }

    __syncthreads();

    // 计算 Attention
    if(tx < q_len){
      int global_q_idx = q_start + tx;
      
      // 这里的逻辑现在是安全的，只影响当前线程的寄存器
      float scores[BC]; // 寄存器数组
      float row_m_curr = -INFINITY;

      // Q * K^T
      for(int j = 0; j < k_len; j++){
        int global_k_idx = k_start + j;

        if(is_causal && global_k_idx > global_q_idx){
          scores[j] = -INFINITY;
          continue;
        }

        float dot = 0.0f;
        for(int d = 0; d < head_dim; d++){
          dot += s_Q[tx * head_dim + d] * s_K[j * head_dim + d];
        }
        scores[j] = dot * softmax_scale;
        
        if (scores[j] > row_m_curr) row_m_curr = scores[j];
      }

      // Online Softmax Logic
      float m_new = fmaxf(m_i, row_m_curr);
      float row_l_new = 0.0f;
      float p_exp[BC];

      for(int j=0; j<k_len; ++j) {
        if (scores[j] == -INFINITY) {
            p_exp[j] = 0.0f;
        } else {
            // 减去新的最大值保证数值稳定性
            p_exp[j] = expf(scores[j] - m_new);
        }
        row_l_new += p_exp[j];
      }

      // Rescale previous accumulator
      float alpha = expf(m_i - m_new);
      l_i = l_i * alpha + row_l_new;
      
      // Update O += P * V
      for (int d = 0; d < head_dim; ++d) {
        acc_o[d] *= alpha;
        for (int j = 0; j < k_len; ++j) {
            acc_o[d] += p_exp[j] * s_V[j * head_dim + d];
        }
      }

      m_i = m_new;
    }

    // 计算 Attention Scores
    // for(int i = 0; i < BR; i++){
    //   int global_q_idx = q_start + i;
    //   if(global_q_idx >= tgt_len) continue;

    //   // 计算当前Row 与当前 K block的点积
    //   float scores[BC];
    //   float row_m_prev = m_i;
    //   float row_m_curr = -INFINITY;

    //   for(int j = 0; j < BC; j++){
    //     int global_k_idx = k_start + j;

    //     //Casual Masking: 如果是因果且 k > q, 则掩盖
    //     if(is_causal && global_k_idx > global_q_idx){
    //       scores[j] = -INFINITY;
    //       continue;
    //     }
    //     if (global_k_idx >= src_len) {
    //       scores[j] = -INFINITY;
    //       continue;
    //     }

    //     float dot = 0.0f;
    //     for(int d = 0; d < head_dim; d++){
    //       dot += s_Q[i * head_dim + d] * s_K[j * head_dim + d];
    //     }
    //     scores[j] = dot * softmax_scale;
    //     // 更新当前行的最大值
    //     if(scores[j] > row_m_curr) row_m_curr = scores[j];
    //   }

    //   // 计算 Online Softmax
    //   float m_new = fmaxf(m_i, row_m_curr);
    //   // 计算 P_ij (未归一化的概率)
    //   float p_exp[BC];
    //   float row_l_new = 0.0f;
    //   for(int j=0; j<BC; ++j) {
    //       if (scores[j] == -INFINITY) p_exp[j] = 0.0f;
    //       else p_exp[j] = expf(scores[j] - m_new);
    //       row_l_new += p_exp[j];
    //   }

    //   // 修正之前的 l_i 和 acc_o
    //   float alpha = expf(m_i - m_new); // 缩放因子
    //   l_i = l_i * alpha + row_l_new;
      
    //   // 8.3 更新 Accumulator O += P * V
    //   for (int d = 0; d < head_dim; ++d) {
    //       acc_o[d] *= alpha; // 缩放旧值
    //       for (int j = 0; j < BC; ++j) {
    //           acc_o[d] += p_exp[j] * s_V[j * head_dim + d];
    //       }
    //   }

    //   m_i = m_new; // 更新全局最大值

    // }
    
    __syncthreads();
  }

  // 5. 写入 Global Memory
  // 关键修正：每个线程写回自己那一行
  if (tx < q_len) {
    for (int d = 0; d < head_dim; ++d) {
        float res = (l_i > 0.0f) ? (acc_o[d] / l_i) : 0.0f;
        O[o_offset_base + (q_start + tx) * stride_q_s + d] = (T)res;
    }
  }

  // 9. 最终归一化并写回 Global Memory
  // O = acc_o / l_i
  // 注意：这里需要将 acc_o 写入到全局内存正确的位置
  // 实际并行代码中，每个线程负责写一部分 d
  //TODO 
  // for (int i = 0; i < BR; ++i) {
  //   if (tx == 0 && (q_start + i) < tgt_len) {
  //     for (int d = 0; d < head_dim; ++d) {
  //         // 这是一个极其简化的写入，实际上 O 的计算也需要在 loop 内完成
  //         // 这里仅展示逻辑：最终除以分母 l_i
  //         O[o_offset_base + (q_start + i) * stride_q_s + d] = (T)(acc_o[d] / l_i); 
  //     }
  //   }
  // }

}



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

// 定义分块大小 (Tile Size)
// 在实际优化中，这些值需要根据 head_dim 和共享内存大小进行调整

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function

  size_t size_q = h_q.size() * sizeof(T);
  size_t size_k = h_k.size() * sizeof(T);
  size_t size_v = h_v.size() * sizeof(T);
  size_t size_o = h_o.size() * sizeof(T);

  T *d_q, *d_k, *d_v, *d_o;
  cudaMalloc(&d_q, size_q);
  cudaMalloc(&d_k, size_k);
  cudaMalloc(&d_v, size_v);
  cudaMalloc(&d_o, size_o);


  cudaMemcpy(d_q, h_q.data(), size_q, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), size_k, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), size_v, cudaMemcpyHostToDevice);
  
  // 4. 配置 Kernel 启动参数
  // Grid: [Target_Seq_Len / Block_Row, Query_Heads, Batch_Size]
  dim3 grid((target_seq_len + BR - 1) / BR, query_heads, batch_size);
  // Block: 实际上通常使用 128 或 256 个线程来协作处理一个 Tile
  dim3 block(128);

  size_t sram_size = (BR * head_dim + BC * head_dim * 2) * sizeof(float);
  float softmax_scale = 1.0f / sqrtf((float)head_dim);

  std::cout << "Launching Kernel with Grid: " << grid.x << ", " << grid.y << ", " << grid.z << std::endl;

  flash_attn_kernel<T><<<grid, block, sram_size>>>(
      d_q, d_k, d_v, d_o,
      batch_size,
      target_seq_len,
      src_seq_len,
      query_heads,
      kv_heads,
      head_dim,
      softmax_scale,
      is_causal
  );

  // 检查 Kernel 错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  h_o.resize(h_q.size());
  cudaMemcpy(h_o.data(), d_o, size_o, cudaMemcpyDeviceToHost); 
  
  // 释放内存
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);  

}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
