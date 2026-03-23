# WIP - A simple C library to work with machine learning
---

The goal is to provide header-only library that you can use that anywhere.

Note: RAII is implemented using `__attribute__((cleanup))`, which only works in Clang and GCC. Manual memory management is still available through `FreeArray(array)`, `FreeArrayPointer(array_pointer)`, and `Array(T, size)`.

## Todo(s)
  - ~Update the functions models to use destination buffers, e.g., `func(src1, src2, dest)`. This enforces stricter memory management and reduces leaks.~
  - Add support for hardware offloading (GPU) or handling very large arrays in chuncks.
  - ~Add documentation/comments for all functions~ (PARTIAL DONE)
  - Implement a simple neural network to benchmark performance.
