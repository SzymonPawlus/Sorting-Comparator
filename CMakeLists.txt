cmake_minimum_required(VERSION 3.19)
project(cuda_test_2 CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_test_2 main.cu)

set_target_properties(
        cuda_test_2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)