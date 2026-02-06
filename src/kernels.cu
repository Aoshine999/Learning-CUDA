#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"


__global__ void traceKernel(const float* d_input, float* d_result, size_t rows, size_t cols, size_t min_dim);

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
);

template <typename T>
__global__ void flash_attn_kernel1(
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
);





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

#define MAX_HEAD_DIM 128
#define BR 16
#define BC 16

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
  // 1. 坐标设置
  const int tx = threadIdx.x; 
  const int q_block_idx = blockIdx.x; 
  const int head_idx = blockIdx.y; 
  const int batch_idx = blockIdx.z; 

  // 2. GQA 映射
  const int kv_head_idx = head_idx / (q_heads / kv_heads);

  // 3. 内存偏移量 (使用 size_t 防止大张量溢出)
  const size_t stride_q_b = (size_t)tgt_len * q_heads * head_dim;
  const size_t stride_q_s = (size_t)q_heads * head_dim;
  const size_t stride_q_h = (size_t)head_dim;

  const size_t stride_k_b = (size_t)src_len * kv_heads * head_dim;
  const size_t stride_k_s = (size_t)kv_heads * head_dim;
  const size_t stride_k_h = (size_t)head_dim; 

  const size_t q_offset_base = batch_idx * stride_q_b + head_idx * stride_q_h;
  const size_t k_offset_base = batch_idx * stride_k_b + kv_head_idx * stride_k_h;
  const size_t v_offset_base = batch_idx * stride_k_b + kv_head_idx * stride_k_h;
  const size_t o_offset_base = batch_idx * stride_q_b + head_idx * stride_q_h;

  // 4. 共享内存
  extern __shared__ float sram[];
  float *s_Q = sram; 
  float *s_K = sram + BR * head_dim;
  float *s_V = sram + BR * head_dim + BC * head_dim; 

  // 5. 寄存器状态变量 - 全部使用 double 减少累积误差
  double m_i = -INFINITY;
  double l_i = 0.0;
  double acc_o[MAX_HEAD_DIM];

  // Kahan 补偿变量 - 全部使用 double
  double l_c = 0.0;
  double acc_c[MAX_HEAD_DIM];

  for (int d = 0; d < head_dim; ++d) {
      acc_o[d] = 0.0;
      acc_c[d] = 0.0;
  }

  // 6. 加载 Q Block
  const int q_start = q_block_idx * BR;
  const int q_len = (q_start + BR > tgt_len) ? (tgt_len - q_start) : BR;

  for (int i = tx; i < BR * head_dim; i += blockDim.x) {
      int r = i / head_dim;
      int c = i % head_dim;
      if (r < q_len) {
          s_Q[i] = (float)Q[q_offset_base + (q_start + r) * stride_q_s + c];
      } else {
          s_Q[i] = 0.0f; 
      }
  }

  __syncthreads();

  // 7. 遍历所有 KV Blocks
  const int num_k_blocks = (src_len + BC - 1) / BC;
  
  for (int j_block = 0; j_block < num_k_blocks; j_block++) {
    const int k_start = j_block * BC;
    const int k_len = (k_start + BC > src_len) ? (src_len - k_start) : BC;

    for (int i = tx; i < BC * head_dim; i += blockDim.x) {
      int r = i / head_dim;
      int c = i % head_dim;
      if (r < k_len) {
          s_K[i] = (float)K[k_offset_base + (k_start + r) * stride_k_s + c];
          s_V[i] = (float)V[v_offset_base + (k_start + r) * stride_k_s + c];
      } else {
          s_K[i] = 0.0f; 
          s_V[i] = 0.0f; 
      }
    }

    __syncthreads();

    if (tx < q_len) {
      const int global_q_idx = q_start + tx;
      
      float scores[BC];
      float row_m_curr = -INFINITY;
      bool has_valid = false;

      // Q * K^T
      for (int j = 0; j < k_len; j++) {
        const int global_k_idx = k_start + j;

        if (is_causal && global_k_idx > global_q_idx) {
          scores[j] = -INFINITY;
        } else {
          float dot = 0.0f;
          for (int d = 0; d < head_dim; d++) {
            dot += s_Q[tx * head_dim + d] * s_K[j * head_dim + d];
          }
          scores[j] = dot * softmax_scale;
          has_valid = true;
          
          if (scores[j] > row_m_curr) {
            row_m_curr = scores[j];
          }
        }
      }

      for (int j = k_len; j < BC; j++) {
        scores[j] = -INFINITY;
      }

      if (has_valid) {
        const float m_new_float = fmaxf((float)m_i, row_m_curr);
        const double m_new = m_new_float;  // 转换为 double 进行后续计算

        float row_l_new = 0.0f;
        float p_exp[BC];

        // 使用 float 计算 exp，配合 FMA 减少误差
        for (int j = 0; j < k_len; ++j) {
          if (isinf(scores[j]) && scores[j] < 0.0f) {
            p_exp[j] = 0.0f;
          } else {
            p_exp[j] = expf(scores[j] - m_new_float);
          }
        }

        // 使用 FMA 进行 pairwise 求和（更精确）
        int j;
        for (j = 0; j + 1 < k_len; j += 2) {
          row_l_new = __fmaf_rn(p_exp[j], 1.0f, row_l_new);
          row_l_new = __fmaf_rn(p_exp[j + 1], 1.0f, row_l_new);
        }
        if (j < k_len) {
          row_l_new = __fmaf_rn(p_exp[j], 1.0f, row_l_new);
        }

        // 计算 alpha（使用 double 精度）
        double alpha = 0.0;
        if (m_i > -INFINITY) {
          double diff = m_i - m_new;
          // 当 diff 很小时，使用泰勒展开 exp(x) ≈ 1 + x + x²/2
          if (fabs(diff) < 1e-4) {
            alpha = 1.0 + diff + 0.5 * diff * diff;
          } else {
            alpha = exp(diff);
          }
        }

        // ============================================================
        // 正确的 Kahan 求和实现：l_i = l_i * alpha + row_l_new
        // 使用 double 精度进行累加
        // ============================================================
        {
          // 步骤1: 缩放旧值和补偿
          l_i *= alpha;
          l_c *= alpha;

          // 步骤2: 使用 Kahan 求和加上新值（row_l_new 转为 double）
          double row_l_new_d = (double)row_l_new;
          double y = row_l_new_d - l_c;
          double t = l_i + y;
          l_c = (t - l_i) - y;
          l_i = t;
        }

        // ============================================================
        // 使用 double 精度进行 acc_o 的 Kahan 求和
        // ============================================================
        for (int d = 0; d < head_dim; ++d) {
          // 步骤1: 缩放旧值和补偿（使用 double）
          acc_o[d] *= alpha;
          acc_c[d] *= alpha;

          // 步骤2: 使用 FMA 进行 pairwise 求和（float 计算）
          float pv_sum = 0.0f;
          int j;
          for (j = 0; j + 1 < k_len; j += 2) {
            pv_sum = __fmaf_rn(p_exp[j], s_V[j * head_dim + d], pv_sum);
            pv_sum = __fmaf_rn(p_exp[j + 1], s_V[(j + 1) * head_dim + d], pv_sum);
          }
          if (j < k_len) {
            pv_sum = __fmaf_rn(p_exp[j], s_V[j * head_dim + d], pv_sum);
          }

          // 步骤3: 转换为 double 进行 Kahan 求和
          double pv_sum_d = (double)pv_sum;
          double y = pv_sum_d - acc_c[d];
          double t = acc_o[d] + y;
          acc_c[d] = (t - acc_o[d]) - y;
          acc_o[d] = t;
        }

        m_i = m_new;
      }
    }
    
    __syncthreads();
  }

  // 8. 最终归一化并写回
  if (tx < q_len) {
    for (int d = 0; d < head_dim; ++d) {
      float res = (l_i > 1e-6f) ? (acc_o[d] / l_i) : 0.0f; 
      O[o_offset_base + (q_start + tx) * stride_q_s + d] = (T)res;
    }
  }
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

  flash_attn_kernel1<T><<<grid, block, sram_size>>>(
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

template <typename T>
__global__ void flash_attn_kernel1(
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


  // 3. 加载 Q Tile (协作加载)
  // 关键修正：利用所有线程加载数据
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

  int num_k_blocks = (src_len + BC - 1) / BC;

  // 7. 遍历 K,V的所有Tiles
  for(int j_block = 0; j_block < num_k_blocks; j_block++){
    const int k_start = j_block * BC;
    const int k_len = (k_start + BC > src_len) ? (src_len - k_start) : BC;

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

      for(int j = 0; j < k_len; ++j) {
        if (scores[j] == -INFINITY) {
            p_exp[j] = 0.0f;
        } else {
            // 减去新的最大值保证数值稳定性
            p_exp[j] = expf(scores[j] - m_new);
        }
        row_l_new += p_exp[j];
      }

      // Rescale previous accumulator
      float alpha = 0.0f;

      if (m_new > -INFINITY) {
          alpha = expf(m_i - m_new);
      }


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
