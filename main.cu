// Made with CLion Educational License

#include <cmath>
#include <chrono>
#include <iostream>

__device__ __host__ void swap(float &x, float &y){
    float temp = y;
    y = x;
    x = temp;
}

// CPU Bubble Sort
__host__ void bubble_sort(int n, float *x){
    // Set optimizing variables
    bool next_loop = true;
    int k = 0;

    // Main loop
    while(next_loop){
        next_loop = false;
        for (int j = 0; j < n - k - 1 ; j ++){
            // Do the comparison and swap
            if(x[j] > x[j + 1]){
                swap(x[j], x[j + 1]);
                next_loop = true;
            }
        }
        k++;
    }
}

// GPU Bubble Sort
// ODD-EVEN Sort
__global__ void bubble_sort(int n, float *x, bool parity) {
    // Get current index (only even)
    int i = 2 * blockDim.x *blockIdx.x + threadIdx.x * 2;
    if(i < n){
        // Check if we are checking in front or behind or (i + 1)
        if(parity){
            // Check whether we are inside of array
            if(i + 1 < n){
                if(x[i] > x[i + 1]){
                    swap(x[i], x[i + 1]);
                }
            }
        }else{
            // Check whether we are inside of array
            if(i - 1 >= 0){
                if(x[i] < x[i - 1]){
                    swap(x[i], x[i - 1]);
                }
            }
        }
    }
}

// CPU Merge Sort
__host__ void merge(float *x, int l, int m, int r){
    // Calculate temporary arrays length
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary arrays
    auto R = new float[n2]; // Left and Right
    auto L = new float[n1];

    // Copy array to subarrays
    for (int i = 0; i < n1; ++i)
        L[i] = x[l + i];
    for (int i = 0; i < n2; ++i)
        R[i] = x[(m + 1) + i];

    // Init indices of arrays
    int i = 0; // L
    int j = 0; // R
    int p = l; // *x

    // Choose smaller value from subarrays
    while(i < n1 && j < n2)
        x[p++] = L[i] < R[j] ? L[i++] : R[j++];

    // Copy remaining elements
    while(i < n1)
        x[p++] = L[i++];

    while(j < n2)
        x[p++] = R[j++];

    delete[] R;
    delete[] L;
}

__host__ void merge_sort(float *x, int l, int r){
    // Check if it is more than 1 element in the array
    if(r > l) {
        // Get middle of the array
        int m = (l + r) / 2;

        // Divide recursively on half arrays
        merge_sort(x, l, m);
        merge_sort(x, m + 1, r);

        // Merge arrays together
        merge(x, l, m, r);
    }
}

// GPU Bitonic Sort
__device__ void compare_and_swap(int n, float *x, int i, int j){
    // Check whether values are in good order. If they are not -> swap
    if(i < n && j < n)
        if(x[i] > x[j]) swap(x[i], x[j]);
}

__global__ void compare(int n, float *x, int size, int current_size){
    // Get current comparison id
    int i = (blockIdx.x*blockDim.x + threadIdx.x);
    // Check if the comparison lays in math.ceil(n / 2)
    if(i < (n + 1) / 2){
        // Divide comparisons into blocks
        int block = i / current_size;

        // Calculate direction of the group
        int block_dir = i / size;

        // Calculate offset in the group
        int num_in_block = i % current_size;
        int pivot, comparator;

        // Check direction of comparison and calculate indecies
        if(block_dir % 2 == 0) {
            pivot = 2 * (block * current_size) + num_in_block; // Number of element in x
            comparator = pivot + current_size;
        }else{
            pivot = 2 * ((block + 1) * current_size ) - num_in_block - 1; // Number of element in x
            comparator = pivot - current_size;
        }
        // Compare and swap right indices
        compare_and_swap(n, x, pivot, comparator);
    }

}

__host__ void bitonic_sort(int n, float *x){
    int current_size;
    for (int size=1; size <= n / 2; size *= 2)
    {
        for (current_size = size; current_size >= 1; current_size /= 2){
            compare<<<std::ceil((float) (n / 2) / 1024.0f), 1024>>>(n, x, size, current_size);
        }
    }
}


// CPU Quick sort

__host__ int partition (float* x, int low, int high)
{
    float pivot = x[high]; // pivot
    int i = low; // Index of smaller element and indicates the right position of pivot found so far

    for (int j = low; j <= high - 1; j++)
        // If current element is smaller than the pivot
        if (x[j] < pivot)
            swap(x[i++], x[j]);
        
    swap(x[i], x[high]);
    return i;
}

__host__ void quick_sort(float *x, int i, int j){
    if(i < j){
        int pivot = partition(x, i, j);
        quick_sort(x, i, pivot - 1);
        quick_sort(x, pivot + 1, j);
    }
}


int main(){
    // Initialize data
    int order_of_magnitude;
    std::cout << "Enter order of magnitude to test: ";
    std::cin >> order_of_magnitude;
    std::cout << std::endl << "----------------" << std::endl << std::endl;
    if(order_of_magnitude > 26) std::cout << "WARNING" << std::endl << "Order of magnitude lowered to 26 due to the performance issues" << std::endl << std::endl;
    order_of_magnitude = std::min(26, order_of_magnitude);
    int N = (1 << order_of_magnitude); // 2^n
    int next_power = pow(2, ceil(log(N)/log(2)));
    float *gpu_bubble, *cuda_gpu_bubble, *cpu_bubble, *gpu_bitonic, *cuda_gpu_bitonic, *cpu_merge, *cpu_quick;

    // Allocate memory on CPU
    gpu_bubble  = (float*)malloc(N * sizeof(float));
    cpu_bubble  = (float*)malloc(N * sizeof(float));
    gpu_bitonic = (float*)malloc(next_power * sizeof(float)); // Need to ensure that date amount is 2^n
    cpu_merge   = (float*)malloc(N * sizeof(float));
    cpu_quick   = (float*)malloc(N * sizeof(float));

    // Allocate memory on GPU
    cudaMalloc(&cuda_gpu_bubble, N * sizeof(float));
    cudaMalloc(&cuda_gpu_bitonic , next_power * sizeof(float)); // Need to ensure that date amount is 2^n

    // Choose pseudo-random numbers
    for (int i = 0; i < next_power; i ++) {
        if(i < N) {
            gpu_bubble[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            cpu_bubble[i] = gpu_bubble[i];
            gpu_bitonic[i] = gpu_bubble[i];
            cpu_merge[i] = gpu_bubble[i];
            cpu_quick[i] = gpu_bubble[i];
        }else{
            gpu_bitonic[i] = - INFINITY;
        }
    }

    // Bubble Sort GPU
    auto cuda_bubble_begin = std::chrono::steady_clock::now();
    if(N <= (1 << 14)) {
        cudaMemcpy(cuda_gpu_bubble, gpu_bubble, N * sizeof(float), cudaMemcpyHostToDevice);
        for (int i = 0; i < N; i++) {
            bubble_sort<<<ceil((float)N / 2048), 1024>>>(N, cuda_gpu_bubble, i % 2 == 0);
        }
        cudaMemcpy(gpu_bubble, cuda_gpu_bubble, N * sizeof(float), cudaMemcpyDeviceToHost);
    }else{
        std::cout << "WARNING!" << std::endl << "GPU bubble sort disabled due to it's low performance" << std::endl << std::endl;
    }
    auto cuda_bubble_end = std::chrono::steady_clock::now();


    // Bubble Sort CPU
    auto cpu_bubble_begin = std::chrono::steady_clock::now();
    if(N <= (1 << 14)) {
        bubble_sort(N, cpu_bubble);
    }else{
        std::cout << "WARNING!" << std::endl << "CPU bubble sort disabled due to it's low performance" << std::endl;
        std::cout << std::endl << "----------------" << std::endl << std::endl;
    }
    auto cpu_bubble_end = std::chrono::steady_clock::now();


    // Bitonic Sort GPU
    auto gpu_merge_start = std::chrono::steady_clock::now();
    cudaMemcpy(cuda_gpu_bitonic, gpu_bitonic, next_power * sizeof(float), cudaMemcpyHostToDevice);
    bitonic_sort(next_power, cuda_gpu_bitonic);
    cudaMemcpy(gpu_bitonic, cuda_gpu_bitonic, next_power * sizeof(float), cudaMemcpyDeviceToHost);
    auto gpu_merge_end = std::chrono::steady_clock::now();


    // Merge Sort CPU
    auto cpu_merge_start = std::chrono::steady_clock::now();
        merge_sort(cpu_merge, 0, N - 1);
    auto cpu_merge_end = std::chrono::steady_clock::now();


    // Quick Sort CPU
    auto cpu_quick_start = std::chrono::steady_clock::now();
    quick_sort(cpu_quick, 0, N - 1);
    auto cpu_quick_end = std::chrono::steady_clock::now();


    // Set correctness flag
    bool gpu_bubble_correct = true;
    bool cpu_bubble_correct = true;
    bool cpu_merge_correct = true;
    bool gpu_bitonic_correct = true;
    bool cpu_quick_correct = true;


    // Check sorts correctness
    for (int i = 0; i < next_power - 1 ; i ++) {
        if(i < N - 1) {
            if (gpu_bubble[i] > gpu_bubble[i + 1]) gpu_bubble_correct = false;
            if (cpu_bubble[i] > cpu_bubble[i + 1]) cpu_bubble_correct = false;
            if (cpu_merge[i] > cpu_merge[i + 1]) cpu_merge_correct = false;
            if(cpu_quick[i]  > cpu_quick [i + 1]) cpu_quick_correct  = false;
        }
        if (gpu_bitonic[i] > gpu_bitonic[i + 1]) gpu_bitonic_correct = false;

    }

    // Display number of elements
    std::cout << "Sorting algorithms for: " << N << " elements" << std::endl << std::endl;

    // Display correctness test
    std::cout << "GPU Bubble Sort correctness  : "<< gpu_bubble_correct << std::endl;
    std::cout << "CPU Bubble Sort correctness  : "<< cpu_bubble_correct << std::endl;
    std::cout << "GPU Bitonic Sort correctness : " << gpu_bitonic_correct << std::endl;
    std::cout << "CPU Merge Sort correctness   : " << cpu_merge_correct  << std::endl;
    std::cout << "CPU Quick Sort correctness   : " << cpu_quick_correct  << std::endl;

    // Make space
    std::cout << std::endl << "----------------" << std::endl << std::endl;

    // Display times time
    std::cout << "GPU Bubble Sort time  = " << std::chrono::duration_cast<std::chrono::microseconds>(cuda_bubble_end - cuda_bubble_begin).count() << " µs" << std::endl;
    std::cout << "CPU Bubble Sort time  = " << std::chrono::duration_cast<std::chrono::microseconds>(cpu_bubble_end - cpu_bubble_begin).count() << " µs" << std::endl;
    std::cout << "GPU Bitonic Sort time = " << std::chrono::duration_cast<std::chrono::microseconds>(gpu_merge_end - gpu_merge_start).count() << " µs" << std::endl;
    std::cout << "CPU Merge Sort time   = " << std::chrono::duration_cast<std::chrono::microseconds>(cpu_merge_end - cpu_merge_start).count() << " µs" << std::endl;
    std::cout << "CPU Quick Sort time   = " << std::chrono::duration_cast<std::chrono::microseconds>(cpu_quick_end - cpu_quick_start).count() << " µs" << std::endl;

    cudaFree(cuda_gpu_bubble);
    cudaFree(cuda_gpu_bitonic);


}