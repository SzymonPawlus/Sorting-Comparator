cmake_minimum_required(VERSION 3.19)
project(sorting_comparator CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(sorting_comparator main.cu)

set_target_properties(
        sorting_comparator
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)