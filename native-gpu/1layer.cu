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


// Modified initialize function to allocate memory for gradients
// NeuralNetwork初始化函数
void initialize_neural_network(NeuralNetwork* nn) {
    CUDA_CHECK(cudaMalloc(&nn->weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * sizeof(float)));

    // Allocate temporary host memory
    float* h_weights1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float* h_weights2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float* h_bias1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float* h_bias2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    // Initialize weights and biases on the host
    initialize_weights(h_weights1, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(h_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    initialize_bias(h_bias1, HIDDEN_SIZE);
    initialize_bias(h_bias2, OUTPUT_SIZE);

    // Copy initialized values to device
    CUDA_CHECK(cudaMemcpy(nn->weights1, h_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->weights2, h_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Free temporary host memory
    free(h_weights1);
    free(h_weights2);
    free(h_bias1);
    free(h_bias2);
}

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
// 更新权重梯度（如 hidden.T @ grad_output 更新 W2）:两个矩阵的乘积
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
    // 1024 threads/blocks: block_size(32, 32): 每个线程块包含 32x32 个线程，总共 1024 个线程，每个线程对应图像一个像素
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

    // Apply ReLU: 过ReLU激活函数： 每个像素点单独一个线程
    // GPU 的线程块大小通常为 256 或 512，以匹配 GPU流式多处理器（SM）的线程调度能力。
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
    int b = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程处理的样本索引 b，blockIdx.x表示块大小
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
// 更新weights和bias的梯度， 可以针对每个样本并行更新
// 参考图CrossEntropy.png:输出层grad_output = grad_output[b * OUTPUT_SIZE + labels[b]] - 1 = output[b * OUTPUT_SIZE + i] - 1 = p - 1
// 一般dim3 block_size(32, 32);即块大小为32*32，即每个样本像素特征值
/*
grad_weights: 存储当前层权重weights梯度的数组。
grad_bias: 存储当前层偏置biases梯度的数组。
grad_layer: 当前层的梯度 结果是p*weight。
prev_layer: 前一层的输出 结果是hidden^i。
batch_size: 当前批次的样本数量。
prev_size: 前一层的节点数量。（分别为inputsize=1024（32*32）和HIDDEN_SIZE）,决定了gridsize大小，表示每个样本的特征向量长度
curr_size: 当前层的节点数量。（分别为HIDDEN_SIZE=4096和output_size=10）,和prev_size一起决定了gridsize大小，表示每个样本经过layer处理后的特征向量长度
申请的grid大小为：grid_size((prev_size + block_size.x - 1) / block_size.x, (curr_size + block_size.y - 1) / block_size.y); 由于是全连接层，因此需要线程数为prev_size / block_size * curr_size / block_size
*/
__global__ void update_gradients_kernel(float* grad_weights, float* grad_bias, float* grad_layer, float* prev_layer, int batch_size, int prev_size, int curr_size) {
    int i = blockIdx.y;  // 代表当前层的节点索引（行），由块的 y 坐标决定。由当前层的大小（curr_size）决定0-9或者4095
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 节点列,j 代表前一层的节点索引（列），由块的 x 坐标和线程在块内的索引决定。逐个像素大小值计算

    if (i < curr_size && j < prev_size) {
        float grad_w_sum = 0.0f;
        // **计算权重梯度**：对于每个权重矩阵的元素`(i, j)`，累加所有批次中`grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j]`的和，并通过`atomicAdd`累加到`grad_weights[i * prev_size + j]`。
        for (int b = 0; b < batch_size; ++b) {
            grad_w_sum += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];  // 参考图CrossEntropy3.png : 等于 p * y^i （yi=hidden^i * weight^i）
        }
        atomicAdd(&grad_weights[i * prev_size + j], grad_w_sum);  // atomicAdds累加，性能问题。在多线程环境下对同一个显存地址进行写操作

        //  仅在处理前一层的第一个节点时计算偏置的梯度。这样可以确保每个节点只在一个线程中处理偏置的更新，避免重复计算
        // bias的累加梯度和prev_layer输出无关，都是梯度累加,而是对一个批次中的所有样本进行累加，每次 j == 0 都会启动偏置梯度累加。
        // 4096个节点每个bias都在第一个节点计算到后面bias中
        if (j == 0) {
            float grad_b_sum = 0.0f;
            // **计算偏置梯度**：当`j == 0`时，累加所有批次中`grad_layer[b * curr_size + i]`的和到`grad_bias[i]`。
            for (int b = 0; b < batch_size; ++b) {
                grad_b_sum += grad_layer[b * curr_size + i];  // dCE / dBias = grad_output[b * OUTPUT_SIZE + labels[b]] (compute_output_gradients_kernel中计算的输出和bias的梯度p-1)
            }
            atomicAdd(&grad_bias[i], grad_b_sum);  // atomicAdds累加, 梯度会共享，在多线程环境下对同一个显存地址进行写操作
        }
    }
}

// reLU的梯度函数（计算ReLU的导数），drelu_kernel将激活后的隐藏层输出的梯度置为0或1（>0是y=x）
__global__ void drelu_kernel(float* x, float* d_ReLU_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 样本的每个特征值都做relu梯度
    if (idx < size) {
        d_ReLU_out[idx] = x[idx] > 0.0f ? 1.0f : 0.0f;
    }
}

// Element-wise multiplication of d_dX2 and d_grad_hidden
// 用于在 GPU 上对两个数组进行逐元素相乘的操作
// 一般以下矩阵大小一样
// float* grad1: 指向第一个浮点数组的指针（将要被修改的数组）。
// float* grad2: 指向第二个浮点数组的指针（用于乘法的数组）。
// int size : 数组的大小，用于控制循环的边界，隐藏层中等于batch_size * HIDDEN_SIZE。
__global__ void multiply_gradients_kernel(float* grad1, float* grad2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad1[idx] *= grad2[idx];
    }
}

/* Modified backward function using CUDA kernels
shape rotating is on par with the visual example (excalidraw diagram) in the mnist-cuda git repo (also found in "assets")
初始化梯度为零: 使用 CUDA 核心函数将所有梯度初始化为零。
计算输出层的梯度: 计算损失相对于输出层的梯度。
更新权重和偏置的梯度: 根据计算得到的梯度更新权重和偏置。
计算中间梯度: 计算损失相对于隐藏层的输入的梯度。
释放内存: 释放在 GPU 上分配的内存。
函数参数:
NeuralNetwork* nn: 指向神经网络结构的指针，包含权重和偏置的梯度。
float* d_input: 输入层的激活值。
float* d_hidden: 隐藏层的激活值。
float* d_output: 输出层的激活值。
int* d_labels: 目标标签。
int batch_size: 当前批次的样本数量。
*/
void backward(NeuralNetwork* nn, float* d_input, float* d_hidden, float* d_output, int* d_labels, int batch_size) {
    // Initialize gradients to zero using CUDA kernel: 所有权重的梯度初始化为零。 shape为HIDDEN_SIZE * INPUT_SIZE
    zero_grad_kernel << <(HIDDEN_SIZE * INPUT_SIZE + 256 - 1) / 256, 256 >> > (nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    zero_grad_kernel << <(OUTPUT_SIZE * HIDDEN_SIZE + 256 - 1) / 256, 256 >> > (nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    zero_grad_kernel << <(HIDDEN_SIZE + 256 - 1) / 256, 256 >> > (nn->grad_bias1, HIDDEN_SIZE);  // grad_bias1向量长度为4096 * 1
    CUDA_CHECK(cudaGetLastError());

    zero_grad_kernel <<< (OUTPUT_SIZE + 256 - 1) / 256, 256 >>> (nn->grad_bias2, OUTPUT_SIZE);  // grad_bias2向量长度为10 * 1
    CUDA_CHECK(cudaGetLastError());

    // 1，Compute gradients for output layer：计算输出层梯度
    float* d_grad_output;
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * OUTPUT_SIZE * sizeof(float))); // 输出每个样本的每个分类偏导数计算梯度
    // 根据交叉熵损失和Softmax的导数公式，梯度为 output - one_hot(labels)。
    compute_output_gradients_kernel <<< (batch_size + 255) / 256, 256 >>> (d_grad_output, d_output, d_labels, batch_size);
    CUDA_CHECK(cudaGetLastError());

    // 2, Update gradients for weights2 (W2.grad = grad_output.T @ hidden): 更新权重梯度，使用 matmul_at_b_kernel 核心函数更新第二层权重的梯度
    // 根据链式法则，输出层权重梯度应为隐藏层输出的转置与输出梯度的乘积。p*weight
    dim3 block_size(32, 32);
    dim3 grid_size((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (OUTPUT_SIZE + block_size.y - 1) / block_size.y);
    // nn->grad_weights2 = d_hidden @ d_grad_output
    matmul_at_b_kernel <<< grid_size, block_size >>> (d_hidden, d_grad_output, nn->grad_weights2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Update gradients for bias2和weight2： 更新了nn->grad_weights2和nn->grad_bias2
    // 结果累加nn->grad_weights2 = sum(d_grad_output @ d_hidden)，使用 update_gradients_kernel 核心函数更新偏置的梯度（bias2）
    update_gradients_kernel <<< grid_size, block_size >>> (nn->grad_weights2, nn->grad_bias2, d_grad_output, d_hidden, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // 3，链式法则传播到隐藏层, 求隐藏层的梯度： Compute dX2 (gradient of loss w.r.t. input of second layer)： 计算 dX2（损失相对于第二层输入的梯度）
    float* d_dX2; // 把第二层当输出层的整个梯度，d_dX2即为d 第二层的d_grad_output
    CUDA_CHECK(cudaMalloc(&d_dX2, batch_size * HIDDEN_SIZE * sizeof(float)));
    grid_size.x = (HIDDEN_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    // grad_output @ W2.T， 其中d_grad_output为batch_size * OUTPUT_SIZE 的输出层梯度p-1
    matmul_a_bt_kernel << <grid_size, block_size >> > (d_grad_output, nn->weights2, d_dX2, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    //  Compute d_ReLU_out (element-wise multiplication with ReLU derivative)：
    // ReLU 导数的逐元素乘法得到隐藏层激活函数前的整体CE梯度值
    float* d_grad_hidden;
    CUDA_CHECK(cudaMalloc(&d_grad_hidden, batch_size * HIDDEN_SIZE * sizeof(float)));
    // 计算 ReLU 的导数 d ReLu / d d_hidden的值,d_grad_hidden为0,1矩阵
    drelu_kernel << <(batch_size * HIDDEN_SIZE + 255) / 256, 256 >> > (d_hidden, d_grad_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());
    // d_dX2 = d_grad_hidden与 dX2 逐元素相乘，element-wise
    multiply_gradients_kernel << <(batch_size * HIDDEN_SIZE + 255) / 256, 256 >> > (d_dX2, d_grad_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // 4, Update gradients for weights1 (W1.grad = d_ReLU_out.T @ input): 更新第一层的权重和偏置的梯度, 参考第二层更新方法。
    grid_size.x = (INPUT_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (HIDDEN_SIZE + block_size.y - 1) / block_size.y;
    matmul_at_b_kernel << <grid_size, block_size >> > (d_input, d_dX2, nn->grad_weights1, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Update gradients for bias1和weight1：更新第一层权重和偏置的梯度值（stepsize）
    update_gradients_kernel << <grid_size, block_size >> > (nn->grad_weights1, nn->grad_bias1, d_dX2, d_input, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // 5， Free allocated memory释放
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_dX2));
    CUDA_CHECK(cudaFree(d_grad_hidden));

    // 确保所有 CUDA 操作完成，避免在主机代码中出现不一致的状态, 在 CUDA 编程中用于同步设备（GPU）和主机（CPU）之间的执行
    CUDA_CHECK(cudaDeviceSynchronize());
}

#endif


// 根据梯度更新响应权重参数
#ifndef update_weigtht
#define update_weigtht
// gradient descent step： 通过梯度下降和学习率更新参数，每个值单独更新
__global__ void update_weights_kernel(float* weights, float* grad_weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
    }
}

/*
更新权重操作：将计算得到的梯度用于更新权重和偏置。
更新公式：weights = weights - LEARNING_RATE * gradients
*/
void update_weights(NeuralNetwork* nn) {
    int block_size = 256;
    int grid_size;

    // Update weights1
    grid_size = (HIDDEN_SIZE * INPUT_SIZE + block_size - 1) / block_size;
    update_weights_kernel << <grid_size, block_size >> > (nn->weights1, nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Update weights2
    grid_size = (OUTPUT_SIZE * HIDDEN_SIZE + block_size - 1) / block_size;
    update_weights_kernel << <grid_size, block_size >> > (nn->weights2, nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Update bias1
    grid_size = (HIDDEN_SIZE + block_size - 1) / block_size;
    update_weights_kernel << <grid_size, block_size >> > (nn->bias1, nn->grad_bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Update bias2
    grid_size = (OUTPUT_SIZE + block_size - 1) / block_size;
    update_weights_kernel << <grid_size, block_size >> > (nn->bias2, nn->grad_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}
#endif


// 训练和评估
#ifndef train_evaluate
#define train_evaluate
/* 评估函数evaluate_accuracy则是在测试集上计算模型的准确率，分批次处理数据，避免显存不足
分批次处理测试集，统计预测正确的样本比例：
前向传播计算测试样本的输出，将测试集按小批量分割，分批次评估每个样本的预测值，
比较每个样本的预测类别和真实标签，统计准确率。
 Modify evaluate_accuracy to handle larger datasets by processing in batches: 评估函数
 意义：测试准确率直接反映了模型在未见数据上的表现，能够验证模型的泛化能力。
*/
float evaluate_accuracy(NeuralNetwork* nn, float* d_X_test, int* d_y_test, float* d_hidden, float* d_output, int total_size) {
    int num_batches = (total_size + BATCH_SIZE - 1) / BATCH_SIZE;
    int total_correct = 0;
    int total_processed = 0;

    // 跑num_batches个批次
    for (int batch = 0; batch < num_batches; batch++) {  // total_size按BATCH_SIZE长度切割
        int current_batch_size = (batch == num_batches - 1) ?
            (total_size - batch * BATCH_SIZE) : BATCH_SIZE;

        if (current_batch_size <= 0) break;

        // 调用 forward() 函数计算模型在测试集上的输出。
        forward(nn, &d_X_test[batch * BATCH_SIZE * INPUT_SIZE],
            d_hidden, d_output, current_batch_size);

        float* h_output = (float*)malloc(current_batch_size * OUTPUT_SIZE * sizeof(float));
        int* h_y_test = (int*)malloc(current_batch_size * sizeof(int));

        // 将 GPU 上的预测结果和测试标签拷贝回 CPU 进行对比
        CUDA_CHECK(cudaMemcpy(h_output, d_output,
            current_batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y_test, &d_y_test[batch * BATCH_SIZE],  // 只拷贝数据的第batch个BATCH_SIZE长度的数据
            current_batch_size * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < current_batch_size; i++) {
            int predicted = 0;
            // 取最大概率值
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (h_output[i * OUTPUT_SIZE + j] > h_output[i * OUTPUT_SIZE + predicted]) {
                    predicted = j;
                }
            }

            // 逐个样本对比预测值与真实值，累积计算正确率
            if (predicted == h_y_test[i]) {
                total_correct++;
            }
        }

        total_processed += current_batch_size;
        free(h_output);
        free(h_y_test);
    }

    // 输出百分比
    return 100.0f * total_correct / total_processed;
}

/*
* 函数 train() 定义了一个完整的训练过程，包括数据加载、训练轮数（epoch）循环、前向传播、反向传播、权重更新以及测试准确率的评估:
按批次进行前向传播、损失计算、反向传播和参数更新
每100个批次或首个批次后，随机选取测试集的一个批次计算准确率：
主要流程：
数据加载到 GPU → 梯度清零 → 前向传播 → 损失计算 → 反向传播 → 权重更新 → 性能评估。
每个 epoch 都遍历完整训练集，分为若干小批量训练。

关键优化：
使用 CUDA 并行化矩阵运算和激活操作，加速神经网络的前向和反向传播。
通过小批量训练平衡内存占用与计算效率。

性能监控：
通过随机测试小批量和全量测试实时评估模型的准确率。
*/ 
void train(NeuralNetwork* nn, float* X_train, int* y_train, float* X_test, int* y_test) {
    float* d_X_train, * d_X_test, * d_hidden, * d_output; // 隐藏层和输出额外分配内存
    int* d_y_train, * d_y_test;

    // Allocate memory for training and test data
    // 使用 cudaMalloc 为训练集、测试集、隐藏层输出和输出层分配 GPU 内存,分配的内存大小与数据规模和模型结构相关
    CUDA_CHECK(cudaMalloc(&d_X_train, TRAIN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X_test, TEST_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_train, TRAIN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y_test, TEST_SIZE * sizeof(int)));

    // Copy data to GPU
    // 实际使用 cudaMemcpy 将主机（CPU）上的训练数据和标签拷贝到设备（GPU）上，以便并行处理
    CUDA_CHECK(cudaMemcpy(d_X_train, X_train, TRAIN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X_test, X_test, TEST_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_train, y_train, TRAIN_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_test, y_test, TEST_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;

        // Zero out gradients at the beginning of each epoch
        // 梯度清零：每个 epoch 的开始会清除上一轮中累积的梯度。
        zero_grad_kernel << <(HIDDEN_SIZE * INPUT_SIZE + 256 - 1) / 256, 256 >> > (nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
        zero_grad_kernel << <(OUTPUT_SIZE * HIDDEN_SIZE + 256 - 1) / 256, 256 >> > (nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
        zero_grad_kernel << <(HIDDEN_SIZE + 256 - 1) / 256, 256 >> > (nn->grad_bias1, HIDDEN_SIZE);
        zero_grad_kernel << <(OUTPUT_SIZE + 256 - 1) / 256, 256 >> > (nn->grad_bias2, OUTPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());  // 完成同步，这样做可以确保每次参数更新只基于当前批次的梯度值

        for (int batch = 0; batch < num_batches; batch++) {  // 批次循环（Mini-batch Training）
            int start_idx = batch * BATCH_SIZE;

            // d_X_train为训练数据集，输出结果存储在 d_output 中，隐藏层存储在d_hidden中
            forward(nn, &d_X_train[start_idx * INPUT_SIZE], d_hidden, d_output, BATCH_SIZE);

            // 将 GPU 上的输出结果拷贝到主机（CPU）
            float* h_output = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            // 计算计算当前批次的交叉熵损失，总损失累积到 total_loss 中
            // 注意这里cpu计算的total_loss只做打印输出并没其他作用
            float loss = cross_entropy_loss(h_output, &y_train[start_idx], BATCH_SIZE);
            total_loss += loss;

            free(h_output);  // 释放掉申请的cpu内存.

            // 调用 backward() 函数计算梯度
            backward(nn, &d_X_train[start_idx * INPUT_SIZE], d_hidden, d_output, &d_y_train[start_idx], BATCH_SIZE);
            // 调用 update_weights() 函数更新权重和偏置
            update_weights(nn);

            // 随机测试样本评估:每训练 100 个小批量后，随机抽取测试集中的一个批次样本评估模型准确率
            if ((batch + 1) % 100 == 0 || (epoch == 0 && batch == 0)) {
                // Use random batch from test set for accuracy reporting
                int test_start_idx = rand() % (TEST_SIZE - BATCH_SIZE);  // 测试随机开始位置， - BATCH_SIZE保证不越界
                float test_accuracy = evaluate_accuracy(nn,
                    &d_X_test[test_start_idx * INPUT_SIZE],  //d_X_test开始位置：每个元素长度为INPUT_SIZE
                    &d_y_test[test_start_idx],  // d_y_test开始位置，每个元素长度为1
                    d_hidden, d_output, BATCH_SIZE);

                // 当前的 epoch、批次索引、损失值和测试准确率会被打印到终端，方便观察模型训练动态。
                printf("Epoch %d/%d, Iter %d/%d, Loss: %.4f, Test Accuracy: %.2f%%\n",
                    epoch + 1, EPOCHS, batch + 1, num_batches,
                    total_loss / (batch + 1), test_accuracy);
            }
        }

        // Evaluate on entire test set at end of epoch
        // 每个 epoch 完成后，使用完整的测试集评估模型的整体性能,  测试集上的准确率是对模型最终表现的衡量
        float test_accuracy = evaluate_accuracy(nn, d_X_test, d_y_test, d_hidden, d_output, TEST_SIZE);
        printf("Epoch %d/%d completed, Loss: %.4f, Test Accuracy: %.2f%%\n",
            epoch + 1, EPOCHS, total_loss / num_batches, test_accuracy);
    }

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_X_train));
    CUDA_CHECK(cudaFree(d_X_test));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_y_train));
    CUDA_CHECK(cudaFree(d_y_test));
}
#endif


int main() {
    // srand(time(NULL));: 初始化随机数生成器，使用当前时间作为种子。这确保每次运行程序时生成的随机数序列不同。
    srand(time(NULL));

    NeuralNetwork nn;
    initialize_neural_network(&nn);

    /*动态内存分配: 为训练数据和测试数据分配内存。
    X_train: 存储训练样本的特征（输入）。
    y_train : 存储训练样本的标签（输出）。
    X_test : 存储测试样本的特征。
    y_test : 存储测试样本的标签。*/
    float* X_train = (float*)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int* y_train = (int*)malloc(TRAIN_SIZE * sizeof(int));
    float* X_test = (float*)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int* y_test = (int*)malloc(TEST_SIZE * sizeof(int));

    // 分别读取数据和标签
    load_data("D://AI//mycuda//CudaRuntime1//mnist_data//mnist_data//X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("D://AI//mycuda//CudaRuntime1//mnist_data//mnist_data//y_train.bin", y_train, TRAIN_SIZE);
    load_data("D://AI//mycuda//CudaRuntime1//mnist_data//mnist_data//X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("D://AI//mycuda//CudaRuntime1//mnist_data//mnist_data//y_test.bin", y_test, TEST_SIZE);

    // print first image in the terminal
    // 将第一张图像以字符形式打印到终端
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (X_train[0 * INPUT_SIZE + i * 28 + j] > 0.0f) {
                printf("X");
            }
            else {
                printf(" ");
            }
        }
        printf("\n");
    }

    printf("First 10 training labels: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", y_train[i]);
    }
    printf("\n");

    // Start timing，CLOCK_MONOTONIC 用于获取程序运行的时间间隔，确保计算的时间不受系统时间变化的影响
    //struct timespec start, end;
    //clock_gettime(CLOCK_MONOTONIC, &start);
    time_t start, end;;
    time(&start);

    // 调用 train 函数进行神经网络的训练，传入训练数据和测试数据
    train(&nn, X_train, y_train, X_test, y_test);

    // End timing
    // clock_gettime(CLOCK_MONOTONIC, &end);
    time(&end);

    // Calculate duration in seconds with milliseconds
    double training_time = end - start;

    printf("\nTotal training time: %.2f sec\n", training_time);

    // 释放所有内存
    CUDA_CHECK(cudaFree(nn.weights1));
    CUDA_CHECK(cudaFree(nn.weights2));
    CUDA_CHECK(cudaFree(nn.bias1));
    CUDA_CHECK(cudaFree(nn.bias2));
    CUDA_CHECK(cudaFree(nn.grad_weights1));
    CUDA_CHECK(cudaFree(nn.grad_weights2));
    CUDA_CHECK(cudaFree(nn.grad_bias1));
    CUDA_CHECK(cudaFree(nn.grad_bias2));
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    // 错误检查: 检查是否有 CUDA 错误，如果有，则输出错误信息并返回 1 表示程序失败
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
