#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/*
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
__global__ void softmax_kernel(float* x, int batch_size, int size) {
    int b = blockIdx.x;  // 当前线程块的索引，这里直接用作批次索引 b
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

#endif

// 核心CUDA核函数:前向传播函数
#ifndef core_kernel_backward
#define core_kernel_backward


#endif