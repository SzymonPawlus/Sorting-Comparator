# Sorting Algorithms Comparison

## Bubble Sort
- ! DISABLED ABOVE 2^15 ELEMENTS !
- Average Performance: O(n^2)
- Space Complexity:  : O(1)
- https://en.wikipedia.org/wiki/Bubble_sort

## Parallel Bubble Sort
- ! DISABLED ABOVE 2^15 ELEMENTS !
- Average Performance: O(n^2), O(n) - full parallelism
- Space Complexity:  : O(1)
- https://www.alanzucconi.com/2017/12/13/gpu-sorting-1/

## Parallel Bitonic (Merge) Sort
- Average Performance: O(n * log^2(n)), O(log^2(n)) - full parallelism
- Space Complexity   : O(1)
- Works properly for 2^n elements
- https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting

## Merge Sort
- Average Performance: O(n * log(n))
- Space Complexity   : O(n)
- https://www.geeksforgeeks.org/merge-sort/

## Quick Sort
- Average Performance   : O(n * log(n))
- Worst Case Performance: O(n ^ 2)
- Space Complexity      : O(1)
- https://www.youtube.com/watch?v=LKiaoV86iJo

# Tests:

# TEST 1
- Intel i7-10510U
- Nvidia GeForce GTX 1650 Max-Q
- Order of magnitude: 20
- Elements: 1048576

-> GPU Bitonic sort : 20 ms

-> CPU Merge sort   : 198 ms

-> CPU Quick sort   : 158 ms

# TEST 2
- Intel i7-10510U
- Nvidia GeForce GTX 1650 Max-Q
- Order of magnitude: 25
- Elements: 33554432

-> GPU Bitonic sort : 840 ms

-> CPU Merge sort   : 7783 ms

-> CPU Quick sort   : 6537 ms

# TEST 3
- Intel i7-10510U
- Nvidia GeForce GTX 1650 Max-Q
- Order of magnitude: 14
- Elements: 16384

-> GPU Bubble sort  : 42
ms

-> CPU Bubble sort  : 682 ms

-> GPU Bitonic sort : < 1 ms

-> CPU Merge sort   : 2 ms

-> CPU Quick sort   : 1  ms

# TEST 4
- Intel i5-9300H
- Nvidia GeForce GTX 1050 Ti
- Order of magnitude: 20
- Elements: 1048576

-> GPU Bitonic sort : 31 ms

-> CPU Merge sort   : 350 ms

-> CPU Quick sort   : 280 ms

# TEST 5
- Intel i5-9300H
- Nvidia GeForce GTX 1050 Ti
- Order of magnitude: 25
- Elements: 33554432

-> GPU Bitonic sort : 1445 ms

-> CPU Merge sort   : 13721 ms

-> CPU Quick sort   : 11305 ms

# TEST 6
- Intel i5-9300H
- Nvidia GeForce GTX 1050 Ti
- Order of magnitude: 14
- Elements: 16384
  
-> GPU Bubble sort  : 40 ms

-> CPU Bubble sort  : 1230 ms

-> GPU Bitonic sort : < 1 ms

-> CPU Merge sort   : 4 ms

-> CPU Quick sort   : 3 ms

# TEST 7
- Intel i7-5820K
- Nvidia GeForce GTX 970 
- Order of magnitude: 20
- Elements: 1048576

-> GPU Bitonic sort : 14 ms

-> CPU Merge sort   : 254 ms

-> CPU Quick sort   : 95 ms

# TEST 8
- Intel i7-5820K
- Nvidia GeForce GTX 970
- Order of magnitude: 25
- Elements: 33554432

-> GPU Bitonic sort : 519 ms

-> CPU Merge sort   : 8201 ms

-> CPU Quick sort   : 14748 ms

# TEST 9
- Intel i7-5820K
- Nvidia GeForce GTX 970
- Order of magnitude: 14
- Elements: 16384

-> GPU Bubble sort  : 70 ms

-> CPU Bubble sort  : 445 ms

-> GPU Bitonic sort : < 1 ms

-> CPU Merge sort   : 3 ms

-> CPU Quick sort   : 1  ms

