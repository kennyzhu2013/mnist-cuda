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
youtube��https://www.youtube.com/watch?v=xBEh66V9gZo&t=3s
����ṹ����
����ʵ����һ���������㣨����� �� ���ز� �� ����㣩��ȫ���������磬�߱�ǰ�򴫲���forward propagation�������򴫲���backpropagation����Ȩ�ظ����Լ����������Ĺ��ܡ����� cuda_runtime �������м��٣�֧�ִ��ģ���ݴ���
����ṹ�����:
����㣺784���ڵ㣨��Ӧ28x28��MNISTͼ�񣩡�
���ز㣺4096���ڵ㣬ʹ��ReLU�������
����㣺10���ڵ㣨��Ӧ0-9��10����𣩣�ʹ��Softmax������ʡ�

���Ĳ��������
���ݼ�����Ԥ�������ļ��ж�ȡѵ���Ͳ������ݣ���ִ�г�ʼ��������
ǰ�򴫲���ͨ������㡢���ز����������Ԥ��ֵ��
���򴫲��������ݶȣ���������ʧ�����Ż�ģ�Ͳ�����
Ȩ�ظ��£�ʹ���ݶ��½�����������Ȩ�غ�ƫ�á�
�������������ڲ��Լ�����ģ�����ܣ�׼ȷ�ʣ���
*/


/*
ѵ��������
������С��BATCH_SIZE����32��
ѵ��������EPOCHS����20��
ѧϰ�ʣ�LEARNING_RATE����0.05��
ѵ������С��TRAIN_SIZE����10,000��ѵ�����е�����������
���Լ���С��TEST_SIZE����1,000��
����ά�ȴ�С��INPUT_SIZE����MNIST ���ݼ���ͼ��ߴ�Ϊ 28x28 ���أ�չƽ��ÿ��ͼ��Ĵ�СΪ784
HIDDEN_SIZE�������������ز��е���Ԫ����ڵ㣩����4096
OUTPUT_SIZE�� ������С�����������������ά��10��0-9���֣�
*/
#define INPUT_SIZE 784
#define HIDDEN_SIZE 4096
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 32
#define EPOCHS 20
#define LEARNING_RATE 0.05

// weights��bias�洢Ȩ�غ�ƫ�õ��Դ�ָ��
// grad_weights��grad_bias�洢�ݶ�ֵ���Դ�ָ��
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


// load batched img data���Ӷ������ļ��м���ͼ��
// ÿ�ζ�ȡָ��������size����Ԫ�أ�������洢���ڴ�data��
void load_data(const char* filename, float* data, int size) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        // ������ݼ���ʧ�ܣ������ļ���ʧ���ȡԪ��������ƥ�䣩���ᴥ��������Ϣ���˳�
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// load batch labels:���ض�Ӧ�ı�ǩ�浽��Ӧ�ڴ�labels��
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

// ��ʼ������
#if 1
// kaiming init func for weights
// Ȩ�س�ʼ��:ʹ�� Kaiming ��ʼ����He ��ʼ���� ������Ȩ��ֵ���ݲ��С���й�һ���������ݶ���ʧ��ը,�ر�������ʹ�� ReLU��Rectified Linear Unit�������������
// Kaiming ��ʼ����Ŀ���Ǳ���ÿһ���������������뷽����ͬ
// scale = sqrt(2.0 / size)���Ӿ��ȷֲ����������Ȩ��
// Kaiming ��ʼ��ͨ��ʹ�����¹�ʽ�� w~N(0, 2/Nin),NinΪ�ò�����ڵ���
void initialize_weights(float* weights, int size) {
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

// basic init for biases
// ƫ�ó�ʼֵΪ�㣨bias[i] = 0.0f������ȷ����ʼ״̬�ĶԳ���
void initialize_bias(float* bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}
#endif


// ����CUDA�˺���
#ifndef core_kernel
#define core_kernel
/*
����� �� ���ز㣺
����˷���matmul_a_b_kernel��X @ W1��
float* A, float* B, float* C: ��Щ��ָ�򸡵��������ָ�룬�ֱ��ʾ������� A��B ��������� C
int m, int n, int k: ��Щ�Ǿ����ά�ȣ����У�
m: ���� A ��������
n: ���� A ��������Ҳ�Ǿ��� B ����������
k: ���� B ������
���C[row * k + col] = sum;��Ӧ��Ҳ���������ĳ������λ��row * k + col��ֵ
*/
__global__ void matmul_a_b_kernel(float* A, float* B, float* C, int m, int n, int k) {
    // blockIdx: ��ʾ��ǰ�߳̿��������
    // blockDim: ��ʾÿ���߳̿��е��߳�����
    // threadIdx : ��ʾ��ǰ�߳������߳̿��е�����
    // row �� col �ֱ�ȷ����ǰ�̸߳�������������� C �е��к���
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // ���д���ȷ����ǰ�̵߳��к��������ھ��� C ����Ч��Χ�ڣ��������Խ��
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) { // ѭ������ A ���к� B ����
            sum += A[row * n + i] * B[i * k + col];  // ������� A �е� row �У��� i �е�Ԫ�� * ������� B �е� i �У��� col �е�Ԫ��
        }

        // ������õ��Ľ���洢��������� C �Ķ�Ӧλ��
        C[row * k + col] = sum;
    }
}

/*
�������ز���ݶȣ�
����ת�ó˷���CUDA kernel for matrix multiplication (A @ B.T)��grad_output @ W2.T��
float* A, float* B, float* C: ��Щ��ָ�򸡵��������ָ�룬�ֱ��ʾ������� A��B ��������� C
int m, int n, int k: ��Щ�Ǿ����ά�ȣ����У�
m: ���� A ��������
n: ���� A ��������Ҳ�Ǿ��� B ����������
k: ���� B ������
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

// ����ת�ó˷���CUDA kernel for matrix multiplication (A.T @ B)
// ����Ȩ���ݶȣ��� hidden.T @ grad_output ���� W2��
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


// ������������������ز�������ݶ���Ϊ0��1��Ȼ��������ݶ����
// CUDA kernel for ReLU activation
__global__ void relu_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]); // ����>=0
    }
}


// ���ƫ��: 
// CUDA kernel for bias addition
// int size: ��ʾÿ�������Ĵ�С������������
__global__ void bias_add_kernel(float* x, float* bias, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // idx ȷ���˵�ǰ�߳������������е�Ψһ����, ��0��ʼ
    int b = idx / size;  // ���㵱ǰ�̶߳�Ӧ����������
    int i = idx % size;  // ���㵱ǰ�̶߳�Ӧ����������

    if (b < batch_size && i < size) {
        // ��ƫ������ bias �е� i ��Ԫ�ؼӵ��������� x �ж�Ӧ��Ԫ�ء������ x[idx] �����������е� idx ��Ԫ��
        x[idx] += bias[i];  
    }
}

// ����㣺Ӧ��Softmax��softmax_kernel��������ʣ�
// CUDA kernel for softmax
// float* x: ָ���������ݵ�ָ�룬ͨ����һ����ά���飨��һά�������ʽ�洢��
// int size: ��ʾÿ�������Ĵ�С��������������һ����˵����OUTPUT_SIZE��������0-9����logits���
__global__ void softmax_kernel(float* x, int batch_size, int size) {
    int b = blockIdx.x;  // ��ǰ�߳̿������������ֱ�������������� b, ��Ϊ�߳̿�ֻ��һ���̣߳���Ӧһ�����Σ�
    if (b < batch_size) {  // ȷ����ǰ�̵߳��������� b ����Ч��Χ��
        // ��ʼ�� max_val Ϊ��ǰ���εĵ�һ��Ԫ��
        float max_val = x[b * size];
        for (int i = 1; i < size; ++i) {  // ������ǰ���ε�����Ԫ�أ��ҵ����ֵ max_val��ʹ�� fmaxf �������Ƚϲ��������ֵ
            max_val = fmaxf(max_val, x[b * size + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = expf(x[b * size + i] - max_val); // ������ǰ���ε�����Ԫ�أ�����ÿ��Ԫ�ؼ�ȥ���ֵ���ָ���������µ� x ��
            sum += x[b * size + i];  // ͬʱ��������ָ���ĺ� sum
        }

        // ���� softmax ֵ,��ÿ��Ԫ�س����ܺ� sum���õ� softmax ֵ
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f); // ʹ�� fmaxf ȷ�������С�� 1e-7��������ֵ���ȶ������磬��ֹ���� 0 �ĸ��ʣ�
        }
    }
}

// �ݶȲü�����������ʵ��clip_gradients_kernel����δ�ڷ��򴫲��е���
// ͨ�����ݶ�������һ��ָ���ķ�Χ�ڣ�[-max_norm, max_norm]�����������ģ�͵�ѵ���ȶ���, ��ֹ�ݶȱ�ը
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
NeuralNetwork* nn: ָ��������ṹ��ָ�룬����Ȩ�غ�ƫ�á�
float* d_input: ָ���������ݵ�ָ�룬�洢�� GPU �ϡ�
float* d_hidden: ָ�����ز������ָ�룬�洢�� GPU �ϡ�
float* d_output: ָ�����������ָ�룬�洢�� GPU �ϡ�
int batch_size: ��ǰ���εĴ�С��
*/
void forward(NeuralNetwork* nn, float* d_input, float* d_hidden, float* d_output, int batch_size) {
    // dim3 �� CUDA �����ڶ�����ά����Ϳ��һ�����ݽṹ�������Ա�ʾһ����ά������ͨ������ָ��������Ĵ�С,x��y �� z���ֱ��������ά�ȵĴ�С
    // block_size.z����: �߳̿��� z ά���ϵ��߳�������ͨ���ڴ����Ӧ���У�z ά�Ȳ���ʹ�ã����Ĭ��Ϊ 1��
    // 1024 threads/blocks: block_size(32, 32): ÿ���߳̿���� 32x32 ���̣߳��ܹ� 1024 ���߳�
    dim3 block_size(32, 32);
    // just enough blocks + threads for our naive matmul kernel: ��������������С���Ա�Ϊ����˷������㹻�Ŀ���߳�
    // grid_size.x �� grid_size.y���ֱ��Ӧ�ھ���˷��е��к���
    // HIDDEN_SIZE + block_size.x - 1: ������ʽȷ����ʹ HIDDEN_SIZE ���� block_size.x ����������Ҳ�ܼ��������Ŀ�����ͨ������ block_size.x - 1�����Ա��������������ж�ʧ����
    // / block_size.x: ��������Ŀ���grid_size.x
    // batch_size + block_size.y ��ʾά��y��Ҫ���߳̿�
    // ������������=block_size.x * block_size.y
    dim3 grid_size((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);

    // ���� �� ���ز㣺
    // Input to Hidden (X @ W1): d_hidden = d_input(batchsize, input_size) ��Ȩ�ؾ��� nn->weights1(input_size, hidden_size) ���
    // <<<>>>�ⲿ���﷨������ CUDA ������һ���˺�������ָ��������grid�����߳̿飨block��������
    matmul_a_b_kernel <<< grid_size, block_size >>> (d_input, nn->weights1, d_hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Add bias1 (one bias term for each neuron (multiple weights)): ÿ�������Ԫ���϶�Ӧ��ƫ��ֵ bias2
    // ���������߳̿�������(batch_size * HIDDEN_SIZE + 255) / 256
    // 256�� ÿ�����ص���һ���߳�
    bias_add_kernel <<<(batch_size * HIDDEN_SIZE + 255) / 256, 256 >>> (d_hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Apply ReLU: ��ReLU������� ÿ�����ص���һ���߳�
    relu_kernel << <(batch_size * HIDDEN_SIZE + 255) / 256, 256 >> > (d_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Hidden to Output (Hidden @ W2): �ڶ�������㣬����Ϊ��һ�����ز�����d_hidden
    // xά�ȴ�С���OUTPUT_SIZE
    // grid_size.y���Ӧ���ǲ����
    grid_size.x = (OUTPUT_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    matmul_a_b_kernel <<< grid_size, block_size >>> (d_hidden, nn->weights2, d_output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Add bias2 (also one bias term per neuron)��xά�Ƚ����ÿ��Ԫ�ض�Ӧÿ�����ض�Ҫ��bias
    bias_add_kernel <<< (batch_size * OUTPUT_SIZE + 255) / 256, 256 >>> (d_output, nn->bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Apply softmax�����ÿ���߳̿�ֻҪһ���̣߳�Ϊɶһ���߳̿�ֻ��һ���̣߳���
    // ��ִ�к˺���ʱ���߳̿��ڵ��߳̿��Բ���ִ�У��߳�֮����Թ����ڴ棬���߳̿�֮�����໥�����ġ�
    // GPU��̬�����߳̿鵽���� SM ��ִ��
    // ͬʱ��������ָ���ĺ� sum, ����߳��޷��ٲ𿪲��У���Ҫbatch_sizeÿ��������������softmax���ʷֲ����ڷ���
    softmax_kernel <<< batch_size, 1 >>> (d_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // cudaDeviceSynchronize ��������������ֱ���������豸�������� CUDA �˺�����kernel�����ڴ濽��������ɡ�����ζ���ڵ��� cudaDeviceSynchronize ֮�����������ȴ� GPU �ϵ�����������ɺ��ټ���ִ��
    CUDA_CHECK(cudaDeviceSynchronize());
}

#endif

// ����CUDA�˺���:ǰ�򴫲�����
#ifndef core_kernel_backward
#define core_kernel_backward
/* Modify cross_entropy_loss to work with batches (w/out softmax because we already do this in the forward pass)
���㽻������ʧ�����ں���Ԥ��ֵ��output������ʵ��ǩ��labels��֮��Ĳ��죬���ں����������ʷֲ�֮��Ĳ��죬ͨ�����ڷ�������
output: һ��ָ�򸡵��������ָ�룬����ģ�͵�������ʣ�ͨ���Ǿ��� softmax ����ģ���
labels: һ��ָ�����������ָ�룬����ÿ����������ʵ��ǩ�������������
batch_size: һ����������ʾ�����е�����������
YiΪ0��1�ļ������������CEֻ�����ڶ����ࡣ
*/
float cross_entropy_loss(float* output, int* labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        // ��ÿ�������������Ӧ��ǩ�ĸ��ʶ���ֵ log(fmaxf(output[label], 1e-7f))
        // b * OUTPUT_SIZE �ǵ�ǰ��������������е���ʼ������labels[b] �ǵ�ǰ��������ʵ��ǩ�����������
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f)); // logf �������㵱ǰ���ʵ���Ȼ���������Ϊ��ֵ�����ʹ�� -= ���ۼ���ʧ
    }
    return total_loss / batch_size; // ����ƽ����ʧֵ
}

// Add this CUDA kernel to zero out gradients
// gradients����
__global__ void zero_grad_kernel(float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

/* CUDA kernel for computing output gradients�� ����ÿ�������Ԫ��������ݶ�
���ݽ�������ʧ��Softmax�ĵ�����ʽ���ݶ�Ϊ output - one_hot(labels) (Ԥ����ʼ�ȥ��ʵ��ǩ��one-hot����)
float* grad_output: ָ�򸡵��������ָ�룬���ڴ洢�����������ݶȡ�
float* output: ָ�򸡵��������ָ�룬����ģ�͵�������ʣ�ͨ���Ǿ��� Softmax ����ģ���
int* labels: ָ�����������ָ�룬����ÿ����������ʵ��ǩ�������������
int batch_size: һ����������ʾ�����е�����������
ÿ���������������������ݶ�
�˺˺���ͨ�����д���ÿ�����������Ƚ�ģ�͵�������ʸ�ֵ���ݶ������Ȼ�������ʵ��ǩ������Ӧ���ݶ�ֵ�����գ�grad_output �д洢�ľ���ÿ��������ÿ������ϵ��ݶȣ����ں����ķ��򴫲���Ȩ�ظ��¡�
������ʵ��ǩ��Ӧ�����ֵΪ ������� - 1.0��
�����������ֵΪ ������ʡ���ʵ��ǩ��Ӧ�������ݶ�ֵС�� 0�������ģ���ڸ�����Ԥ����ʵ�����ʵ��ǩ��������������Ϊ 1������Ҫͨ�����򴫲�������Ȩ�أ�����߸�����Ԥ����ʡ�
������������ݶ�ֵ����Ϊ������ʣ���������ȷ��ģ���ڸ���Ȩ��ʱ��ģ�Ͷ���Щ����Ԥ�����ģ����ڸ���ʱҲ�ῼ����Щ���ġ�
�ݶȵķ��ţ����򸺣�������Ȩ�������ӻ��Ǽ��٣����ݶȵĴ�С����������ʾ�������Ӱ���˸��µķ���
*/
__global__ void compute_output_gradients_kernel(float* grad_output, float* output, int* labels, int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;  // ���㵱ǰ�̴߳������������ b
    if (b < batch_size) {  
        for (int i = 0; i < OUTPUT_SIZE; ++i) {  // ���������ÿ�����, OUTPUT_SIZE=10, ����ǰ������������ʸ��Ƶ� grad_output �С����Ǽ����ݶȵĵ�һ��������λ��Ϊ1����Ӧ�ó���0������
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        // ���ڵ�ǰ��������ʵ��ǩ������Ӧ���ݶ�ֵ��ȥ 1��
        // �� one-hot �����У���ʵ��ǩ��λ��Ϊ 1������λ��Ϊ 0����˼�ȥ 1 ��ʾ����ʵ��ǩ��Ӱ�������ݶȼ���
        // grad_output��ʱ��ľ��ǵ�b����������ÿ������ϵ��ݶ�ֵ,�ݶȷ�ӳ����ʧ���������ģ�Ͳ����ı仯��
        // ������bias����stepsize=grad_output[i] * learningrate(ѧϰ��)��newbias = oldbias - stepsize
        // ����nn->weights2���ݶȵ���grad_output[i]*Yi��Yi=d_hidden[i],���ز�����Ԫ�أ�
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
    }
}

// CUDA kernel for updating gradients
// ����bias1��bias2�� �������ÿ���������и���
/*
grad_weights: �洢Ȩ���ݶȵ����顣
grad_bias: �洢ƫ���ݶȵ����顣
grad_layer: ��ǰ����ݶȡ�
prev_layer: ǰһ��������
batch_size: ��ǰ���ε�����������
prev_size: ǰһ��Ľڵ�������
curr_size: ��ǰ��Ľڵ�������
*/
__global__ void update_gradients_kernel(float* grad_weights, float* grad_bias, float* grad_layer, float* prev_layer, int batch_size, int prev_size, int curr_size) {
    int i = blockIdx.y;  // �ڵ���
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // �ڵ���

    if (i < curr_size && j < prev_size) {
        float grad_w_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad_w_sum += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
        }
        atomicAdd(&grad_weights[i * prev_size + j], grad_w_sum);  // atomicAdds�ۼӣ��������⡣�ڶ��̻߳����¶�ͬһ���ڴ��ַ����д����

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