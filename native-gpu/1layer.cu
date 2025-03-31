#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/*
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
__global__ void softmax_kernel(float* x, int batch_size, int size) {
    int b = blockIdx.x;  // ��ǰ�߳̿������������ֱ�������������� b
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

#endif

// ����CUDA�˺���:ǰ�򴫲�����
#ifndef core_kernel_backward
#define core_kernel_backward


#endif