#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include <cuda_fp16.h>
// #include <cuda/atomic>


/*
youtube：https://www.youtube.com/watch?v=xBEh66V9gZo&t=3s
整体结构概述
代码实现了一个具有两层（输入层 → 隐藏层 → 输出层）的全连接神经网络，具备前向传播（forward propagation）、后向传播（backpropagation）、权重更新以及性能评估的功能。基于 cuda_runtime 技术进行加速，支持大规模数据处理。
网络结构与参数:
输入层：784个节点（对应28x28的MNIST图像）。
隐藏层：4096个节点，使用ReLU激活函数。
输出层：10个节点（对应0-9的10个类别），使用Softmax输出概率。

核心步骤包括：
数据加载与预处理：从文件中读取训练和测试数据，并执行初始化操作。
前向传播：通过输入层、隐藏层和输出层计算预测值。
反向传播：计算梯度，并根据损失函数优化模型参数。
权重更新：使用梯度下降法更新网络权重和偏置。
性能评估：基于测试集评估模型性能（准确率）。
*/


/*
训练参数：
批量大小（BATCH_SIZE）：32。
训练轮数（EPOCHS）：20。
学习率（LEARNING_RATE）：0.05。
训练集大小（TRAIN_SIZE）：10,000，训练集中的样本数量。
测试集大小（TEST_SIZE）：1,000。
输入维度大小（INPUT_SIZE）：MNIST 数据集的图像尺寸为 28x28 像素，展平后每张图像的大小为784
HIDDEN_SIZE：即神经网络隐藏层中的神经元（或节点）数量4096
OUTPUT_SIZE： 输出层大小，即网络最终输出的维度10（0-9数字）
*/
#define INPUT_SIZE 784
#define HIDDEN_SIZE 4096
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 32
#define EPOCHS 20
#define LEARNING_RATE 0.05

// weights和bias存储权重和偏置的显存指针
// grad_weights和grad_bias存储梯度值的显存指针
typedef struct {
    float* weights1;
    float* weights2;
    float* bias1;
    float* bias2;
    float* grad_weights1;
    float* grad_weights2;
    float* grad_bias1;
    float* grad_bias2;
} NeuralNetwork;


// Modify the CUDA_CHECK macro to print more information
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// load batched img data：从二进制文件中加载图像
// 每次读取指定数量（size）的元素，并将其存储到内存data中
void load_data(const char* filename, float* data, int size) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        // 如果数据加载失败（例如文件丢失或读取元素数量不匹配），会触发错误消息并退出
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// load batch labels:加载对应的标签存到对应内存labels中
void load_labels(const char* filename, int* labels, int size) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// 初始化处理
#if 1
// kaiming init func for weights
// 权重初始化:使用 Kaiming 初始化（He 初始化） 方法，权重值根据层大小进行归一化，避免梯度消失或爆炸,特别适用于使用 ReLU（Rectified Linear Unit）激活函数的网络
// Kaiming 初始化的目标是保持每一层的输出方差与输入方差相同
// scale = sqrt(2.0 / size)，从均匀分布中随机生成权重
// Kaiming 初始化通常使用以下公式： w~N(0, 2/Nin),Nin为该层输入节点数
void initialize_weights(float* weights, int size) {
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

// basic init for biases
// 偏置初始值为零（bias[i] = 0.0f），以确保初始状态的对称性
void initialize_bias(float* bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}
#endif


// 核心CUDA核函数
#ifndef core_kernel
#define core_kernel
/*
输入层 → 隐藏层：
矩阵乘法：matmul_a_b_kernel（X @ W1）
float* A, float* B, float* C: 这些是指向浮点数数组的指针，分别表示输入矩阵 A、B 和输出矩阵 C
int m, int n, int k: 这些是矩阵的维度，其中：
m: 矩阵 A 的行数。
n: 矩阵 A 的列数（也是矩阵 B 的行数）。
k: 矩阵 B 的列数
结果C[row * k + col] = sum;对应的也是输出矩阵某个具体位置row * k + col数值
*/
__global__ void matmul_a_b_kernel(float* A, float* B, float* C, int m, int n, int k) {
    // blockIdx: 表示当前线程块的索引。
    // blockDim: 表示每个线程块中的线程数。
    // threadIdx : 表示当前线程在其线程块中的索引
    // row 和 col 分别确定当前线程负责计算的输出矩阵 C 中的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 这行代码确保当前线程的行和列索引在矩阵 C 的有效范围内，避免访问越界
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) { // 循环遍历 A 的列和 B 的行
            sum += A[row * n + i] * B[i * k + col];  // 计算矩阵 A 中第 row 行，第 i 列的元素 * 计算矩阵 B 中第 i 行，第 col 列的元素
        }

        // 将计算得到的结果存储到输出矩阵 C 的对应位置
        C[row * k + col] = sum;
    }
}

/*
计算隐藏层的梯度：
矩阵转置乘法：CUDA kernel for matrix multiplication (A @ B.T)（grad_output @ W2.T）
float* A, float* B, float* C: 这些是指向浮点数数组的指针，分别表示输入矩阵 A、B 和输出矩阵 C
int m, int n, int k: 这些是矩阵的维度，其中：
m: 矩阵 A 的行数。
n: 矩阵 A 的列数（也是矩阵 B 的列数）。
k: 矩阵 B 的行数
*/
__global__ void matmul_a_bt_kernel(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[col * n + i];
        }
        C[row * k + col] = sum;
    }
}

// 矩阵转置乘法：CUDA kernel for matrix multiplication (A.T @ B)
// 更新权重梯度（如 hidden.T @ grad_output 更新 W2）
__global__ void matmul_at_b_kernel(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < m; ++i) {
            sum += A[i * n + row] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}


// 激活函数：将激活后的隐藏层输出的梯度置为0或1，然后与后续梯度相乘
// CUDA kernel for ReLU activation
__global__ void relu_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]); // 返回>=0
    }
}


// 添加偏置: 
// CUDA kernel for bias addition
// int size: 表示每个样本的大小（特征数量）
__global__ void bias_add_kernel(float* x, float* bias, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // idx 确定了当前线程在整个数据中的唯一索引, 从0开始
    int b = idx / size;  // 计算当前线程对应的批次索引
    int i = idx % size;  // 计算当前线程对应的特征索引

    if (b < batch_size && i < size) {
        // 将偏置向量 bias 中第 i 个元素加到输入数据 x 中对应的元素。这里的 x[idx] 是输入数据中第 idx 个元素
        x[idx] += bias[i];  
    }
}

// 输出层：应用Softmax：softmax_kernel（输出概率）
// CUDA kernel for softmax
// float* x: 指向输入数据的指针，通常是一个二维数组（以一维数组的形式存储）
// int size: 表示每个样本的大小（特征数量），一般来说等于OUTPUT_SIZE，即数字0-9类别的logits输出
__global__ void softmax_kernel(float* x, int batch_size, int size) {
    int b = blockIdx.x;  // 当前线程块的索引，这里直接用作批次索引 b, 因为线程块只有一个线程（对应一个批次）
    if (b < batch_size) {  // 确保当前线程的批次索引 b 在有效范围内
        // 初始化 max_val 为当前批次的第一个元素
        float max_val = x[b * size];
        for (int i = 1; i < size; ++i) {  // 遍历当前批次的所有元素，找到最大值 max_val。使用 fmaxf 函数来比较并更新最大值
            max_val = fmaxf(max_val, x[b * size + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = expf(x[b * size + i] - max_val); // 遍历当前批次的所有元素，计算每个元素减去最大值后的指数，并更新到 x 中
            sum += x[b * size + i];  // 同时计算所有指数的和 sum
        }

        // 计算 softmax 值,将每个元素除以总和 sum，得到 softmax 值
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f); // 使用 fmaxf 确保结果不小于 1e-7，避免数值不稳定（例如，防止出现 0 的概率）
        }
    }
}

// 梯度裁剪：代码中已实现clip_gradients_kernel，但未在反向传播中调用
// 通过将梯度限制在一个指定的范围内（[-max_norm, max_norm]），可以提高模型的训练稳定性, 防止梯度爆炸
__global__ void clip_gradients_kernel(float* gradients, int size, float max_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = gradients[idx];
        if (grad > max_norm) {
            gradients[idx] = max_norm;
        }
        else if (grad < -max_norm) {
            gradients[idx] = -max_norm;
        }
    }
}

#endif


#ifndef core_kernel_forward
#define core_kernel_forward
/*
Modified forward function using CUDA kernels:
NeuralNetwork* nn: 指向神经网络结构的指针，包含权重和偏置。
float* d_input: 指向输入数据的指针，存储在 GPU 上。
float* d_hidden: 指向隐藏层输出的指针，存储在 GPU 上。
float* d_output: 指向最终输出的指针，存储在 GPU 上。
int batch_size: 当前批次的大小。
*/
void forward(NeuralNetwork* nn, float* d_input, float* d_hidden, float* d_output, int batch_size) {
    // dim3 是 CUDA 中用于定义三维网格和块的一个数据结构。它可以表示一个三维向量，通常用于指定网格或块的大小,x、y 和 z，分别代表三个维度的大小
    // block_size.z含义: 线程块在 z 维度上的线程数量。通常在大多数应用中，z 维度不被使用，因此默认为 1。
    // 1024 threads/blocks: block_size(32, 32): 每个线程块包含 32x32 个线程，总共 1024 个线程
    dim3 block_size(32, 32);
    // just enough blocks + threads for our naive matmul kernel: 计算所需的网格大小，以便为矩阵乘法分配足够的块和线程
    // grid_size.x 和 grid_size.y，分别对应于矩阵乘法中的行和列
    // HIDDEN_SIZE + block_size.x - 1: 这个表达式确保即使 HIDDEN_SIZE 不是 block_size.x 的整数倍，也能计算出所需的块数。通过加上 block_size.x - 1，可以避免在整数除法中丢失余数
    // / block_size.x: 计算所需的块数grid_size.x
    // batch_size + block_size.y 表示维度y需要的线程块
    // 所需的总体块数=block_size.x * block_size.y
    dim3 grid_size((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);

    // 输入 → 隐藏层：
    // Input to Hidden (X @ W1): d_hidden = d_input(batchsize, input_size) 与权重矩阵 nn->weights1(input_size, hidden_size) 相乘
    // <<<>>>这部分语法用于在 CUDA 中启动一个核函数。它指定了网格（grid）和线程块（block）的配置
    matmul_a_b_kernel <<< grid_size, block_size >>> (d_input, nn->weights1, d_hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Add bias1 (one bias term for each neuron (multiple weights)): 每个输出单元加上对应的偏置值 bias2
    // 网格所需线程块数量：(batch_size * HIDDEN_SIZE + 255) / 256
    // 256： 每个像素单独一个线程
    bias_add_kernel <<<(batch_size * HIDDEN_SIZE + 255) / 256, 256 >>> (d_hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Apply ReLU: 过ReLU激活函数： 每个像素单独一个线程
    relu_kernel << <(batch_size * HIDDEN_SIZE + 255) / 256, 256 >> > (d_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Hidden to Output (Hidden @ W2): 第二个输出层，输入为第一个隐藏层输入d_hidden
    // x维度大小变成OUTPUT_SIZE
    // grid_size.y这儿应该是不变的
    grid_size.x = (OUTPUT_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    matmul_a_b_kernel <<< grid_size, block_size >>> (d_hidden, nn->weights2, d_output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Add bias2 (also one bias term per neuron)，x维度结果的每个元素对应每个像素都要加bias
    bias_add_kernel <<< (batch_size * OUTPUT_SIZE + 255) / 256, 256 >>> (d_output, nn->bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Apply softmax，输出每个线程块只要一个线程（为啥一个线程块只有一个线程？）
    // 在执行核函数时，线程块内的线程可以并行执行，线程之间可以共享内存，但线程块之间是相互独立的。
    // GPU动态调度线程块到各个 SM 上执行
    // 同时计算所有指数的和 sum, 因此线程无法再拆开并行，需要batch_size每个特征向量都做softmax概率分布用于分类
    softmax_kernel <<< batch_size, 1 >>> (d_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // cudaDeviceSynchronize 会阻塞主机程序，直到所有在设备上启动的 CUDA 核函数（kernel）和内存拷贝操作完成。这意味着在调用 cudaDeviceSynchronize 之后，主机代码会等待 GPU 上的所有任务完成后再继续执行
    CUDA_CHECK(cudaDeviceSynchronize());
}

#endif

// 核心CUDA核函数:前向传播函数
#ifndef core_kernel_backward
#define core_kernel_backward
/* Modify cross_entropy_loss to work with batches (w/out softmax because we already do this in the forward pass)
计算交叉熵损失，用于衡量预测值（output）和真实标签（labels）之间的差异，用于衡量两个概率分布之间的差异，通常用于分类任务
output: 一个指向浮点数数组的指针，包含模型的输出概率（通常是经过 softmax 处理的）。
labels: 一个指向整数数组的指针，包含每个样本的真实标签（类别索引）。
batch_size: 一个整数，表示批次中的样本数量。
Yi为0或1的计算结果，下面的CE只能用在二分类。
*/
float cross_entropy_loss(float* output, int* labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        // 对每个样本，计算对应标签的概率对数值 log(fmaxf(output[label], 1e-7f))
        // b * OUTPUT_SIZE 是当前样本在输出数组中的起始索引，labels[b] 是当前样本的真实标签（类别索引）
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f)); // logf 函数计算当前概率的自然对数，结果为负值，因此使用 -= 来累计损失
    }
    return total_loss / batch_size; // 返回平均损失值
}

// Add this CUDA kernel to zero out gradients
// gradients清零
__global__ void zero_grad_kernel(float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

/* CUDA kernel for computing output gradients： 计算每个输出单元的输出层梯度
根据交叉熵损失和Softmax的导数公式，梯度为 output - one_hot(labels) (预测概率减去真实标签的one-hot编码)
float* grad_output: 指向浮点数数组的指针，用于存储计算出的输出梯度。
float* output: 指向浮点数数组的指针，包含模型的输出概率（通常是经过 Softmax 处理的）。
int* labels: 指向整数数组的指针，包含每个样本的真实标签（类别索引）。
int batch_size: 一个整数，表示批次中的样本数量。
每个样本单独计算输出层的梯度
此核函数通过并行处理每个样本，首先将模型的输出概率赋值给梯度输出，然后根据真实标签调整相应的梯度值。最终，grad_output 中存储的就是每个样本在每个类别上的梯度，用于后续的反向传播和权重更新。
对于真实标签对应的类别，值为 输出概率 - 1.0。
对于其他类别，值为 输出概率。真实标签对应的类别的梯度值小于 0，这表明模型在该类别的预测概率低于真实标签的期望（即期望为 1），需要通过反向传播来调整权重，以提高该类别的预测概率。
对于其他类别，梯度值保持为输出概率，这样可以确保模型在更新权重时，模型对这些类别的预测信心，且在更新时也会考虑这些信心。
梯度的符号（正或负）决定了权重是增加还是减少，而梯度的大小（由输出概率决定）则影响了更新的幅度
*/
__global__ void compute_output_gradients_kernel(float* grad_output, float* output, int* labels, int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程处理的样本索引 b
    if (b < batch_size) {  
        for (int i = 0; i < OUTPUT_SIZE; ++i) {  // 遍历输出的每个类别, OUTPUT_SIZE=10, 将当前样本的输出概率复制到 grad_output 中。这是计算梯度的第一步。其他位置为1不是应该乘以0归零吗？
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        // 对于当前样本的真实标签，将对应的梯度值减去 1。
        // 在 one-hot 编码中，真实标签的位置为 1，其余位置为 0，因此减去 1 表示将真实标签的影响引入梯度计算
        // grad_output次时存的就是第b个样本的在每个类别上的梯度值,梯度反映了损失函数相对于模型参数的变化率
        // 最后参数bias更新stepsize=grad_output[i] * learningrate(学习率)，newbias = oldbias - stepsize
        // 其他nn->weights2的梯度等于grad_output[i]*Yi（Yi=d_hidden[i],隐藏层的输出元素）
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
    }
}

// CUDA kernel for updating gradients
// 更新bias1和bias2， 可以针对每个样本并行更新
/*
grad_weights: 存储权重梯度的数组。
grad_bias: 存储偏置梯度的数组。
grad_layer: 当前层的梯度。
prev_layer: 前一层的输出。
batch_size: 当前批次的样本数量。
prev_size: 前一层的节点数量。
curr_size: 当前层的节点数量。
*/
__global__ void update_gradients_kernel(float* grad_weights, float* grad_bias, float* grad_layer, float* prev_layer, int batch_size, int prev_size, int curr_size) {
    int i = blockIdx.y;  // 节点行
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 节点列

    if (i < curr_size && j < prev_size) {
        float grad_w_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad_w_sum += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
        }
        atomicAdd(&grad_weights[i * prev_size + j], grad_w_sum);  // atomicAdds累加，性能问题。在多线程环境下对同一个内存地址进行写操作

        if (j == 0) {
            float grad_b_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                grad_b_sum += grad_layer[b * curr_size + i];
            }
            atomicAdd(&grad_bias[i], grad_b_sum);
        }
    }
}

#endif