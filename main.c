#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define TYPE_LIST \
  X(int8_t, i8, "%d") \
  X(int16_t, i16, "%d") \
  X(int32_t, i32, "%d") \
  X(int64_t, i64, "%ld") \
  X(uint8_t, u8, "%u") \
  X(uint16_t, u16, "%u") \
  X(uint32_t, u32, "%u") \
  X(uint64_t, u64, "%lu") \
  X(float, f32, "%f") \
  X(double, f64, "%lf")

// define each type alias
#define X(real_type, alias, printformat) typedef real_type alias;
TYPE_LIST
#undef X

#define DeclareArray(T) typedef struct { \
  T * data; \
  size_t size; \
} T##Array;


#define Array(T, sz) (T##Array) {.data = (T*)malloc(sizeof(T) * (sz)), .size=(sz)}

#define FreeArrayPointer(array) \
  free(array->data); \
  free(array);

#define DefineZeros(T) \
  void zeros_##T(T##Array array) { \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < array.size; i++) array.data[i] = 0; \
  }

#define DefineOnes(T) \
  void ones_##T(T##Array array) { \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < array.size; i++) array.data[i] = 1; \
  }

#define DefineRand(T) \
  void rand_##T(T##Array array, u32 seed, u64 max_value) { \
    _Pragma("omp parallel private(seed)") \
    { \
    u32 my_seed = seed * omp_get_thread_num(); \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < array.size; i++) array.data[i] = rand_r(&my_seed) % max_value; \
    } \
  }

#define DefineSum(T) \
  T##Array * sum_##T(T##Array array1, T##Array array2) { \
    if (array1.size != array2.size) { \
      printf("The arrays have different sizes (%ld != %ld), returning error! ", array1.size, array2.size); \
      return NULL; \
    } \
    T##Array * ret = malloc(sizeof(T##Array) * 1); \
    *ret = Array(T, array1.size); \
    _Pragma("omp parallel for") \
      for (size_t i = 0; i < array1.size; i++) { \
        ret->data[i] = array1.data[i] + array2.data[i]; \
      } \
    return ret; \
  }

#define DefineScalarMUL(T) \
  void scalar_mul_##T(T##Array array, u64 scalar) { \
    _Pragma("omp parallel for") \
      for (size_t i = 0; i < array.size; i++) { \
        array.data[i] = array.data[i] * (T) scalar; \
      } \
  }

#define DefinePrint(T, format) \
  void print_##T##Array(T##Array array, size_t limit) { \
    printf("{"); \
    for (size_t i = 0; i < limit; i++) { \
        printf(format, array.data[i]); \
        if (i != array.size - 1) \
        printf(", "); \
    } \
    printf("}\n"); \
  }


#define X(real_type, alias, format) \
  DeclareArray(alias) \
  DefineZeros(alias) \
  DefineRand(alias) \
  DefineOnes(alias) \
  DefineSum(alias) \
  DefineScalarMUL(alias) \
  DefinePrint(alias, format)
TYPE_LIST
#undef X


int main() {
  
  i8Array array = Array(i8, 100);
  u64 scalar = 2;

  if (array.data != NULL) {
    rand_i8(array, 42, 50);
    print_i8Array(array, array.size);
    scalar_mul_i8(array, scalar);
    print_i8Array(array, array.size);
  }

  return 0;
}
