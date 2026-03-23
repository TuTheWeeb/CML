#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define TYPE_LIST(O) \
  O(int8_t, i8, "%d") \
  O(int16_t, i16, "%d") \
  O(int32_t, i32, "%d") \
  O(int64_t, i64, "%ld") \
  O(uint8_t, u8, "%u") \
  O(uint16_t, u16, "%u") \
  O(uint32_t, u32, "%u") \
  O(uint64_t, u64, "%lu") \
  O(float, f32, "%f") \
  O(double, f64, "%lf")

// define each type alias
#define X(real_type, alias, printformat) typedef real_type alias;
TYPE_LIST(X)

// Declare all types of Array
#define DeclareArray(T) typedef struct { \
  T * restrict data; \
  size_t size; \
} T##Array;

// Automatically alocates memory
#define Array(T, sz) (T##Array) {.data = (T*)malloc(sizeof(T) * (sz)), .size=(sz)}

// For manual use
#define FreeArray(array) \
    free(array.data);

// For manual use
#define FreeArrayPointer(array) \
  FreeArray(*array); \
  free(array);

// Defines all zeros functions
#define DefineZeros(T) \
  void zeros_##T(T##Array array) { \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < array.size; i++) array.data[i] = 0; \
  }

// Defines all ones functions
#define DefineOnes(T) \
  void ones_##T(T##Array array) { \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < array.size; i++) array.data[i] = 1; \
  }

// Defines all rands functions
#define DefineRand(T) \
  void rand_##T(T##Array array, u32 seed, u64 max_value) { \
    _Pragma("omp parallel") \
    { \
    u32 my_seed = seed * (omp_get_thread_num() + 1); \
    _Pragma("omp for") \
    for (size_t i = 0; i < array.size; i++) array.data[i] = rand_r(&my_seed) % max_value; \
    } \
  }

// Defines all sum functions
#define DefineSum(T) \
  T##Array sum_##T(T##Array array1, T##Array array2) { \
    if (array1.size != array2.size) { \
      printf("The arrays have different sizes (%ld != %ld), returning error!\n", array1.size, array2.size); \
      return (T##Array) {.data=NULL, .size=0}; \
    } \
    T##Array ret = Array(T, array1.size); \
    _Pragma("omp parallel for") \
      for (size_t i = 0; i < array1.size; i++) { \
        ret.data[i] = array1.data[i] + array2.data[i]; \
      } \
    return ret; \
  }

// Defines all scalar functions
#define DefineScalarMUL(T) \
  void scalar_mul_##T(T##Array array, u64 scalar) { \
    _Pragma("omp parallel for") \
      for (size_t i = 0; i < array.size; i++) { \
        array.data[i] = array.data[i] * (T) scalar; \
      } \
  }

// Defines all dot product funcions
#define DefineDotProduct(T) \
    T dot_product_##T(T##Array array1, T##Array array2) { \
      if (array1.size != array2.size) {\
        printf("The arrays have different sizes (%ld != %ld), returning error!\n", array1.size, array2.size); \
        return (T) 0; \
      } \
      T acum = 0; \
      _Pragma("omp parallel for reduction(+:acum)") \
      for (size_t i = 0; i < array1.size; i++) {\
        acum += array1.data[i] * array2.data[i]; \
      } \
      return acum; \
    }

// Defines all map functions
#define DefineMap(T) \
    T##Array map_##T##Array(T##Array array, T (*func)(T)) { \
      if (array.data == NULL) { \
        printf("Array is empty\n"); \
        return (T##Array) {.data=NULL, .size=0}; \
      } \
      T##Array ret = Array(T, array.size); \
      _Pragma("omp parallel for") \
      for (size_t i = 0; i < array.size; i++) {\
        ret.data[i] = func(array.data[i]); \
      } \
      return ret; \
    }

// Defines all print functions
#define DefinePrint(T, format) \
  void print_##T##Array(T##Array array, size_t limit) { \
    if (limit >= array.size) { \
      limit = array.size; \
    } \
    printf("{"); \
    for (size_t i = 0; i < limit; i++) { \
        printf(format, array.data[i]); \
        if (i != limit - 1) \
        printf(", "); \
    } \
    printf("}\n"); \
  }

// Define all array clean functions
#define DefineArrayFree(T) \
    void free_##T##Array(T##Array * array) {\
      if (array->data != NULL) free(array->data); \
      array->data = NULL; \
    }

// Expand each macro
#define GENERATE_ALL(real_type, alias, format) \
  DeclareArray(alias) \
  DefineZeros(alias) \
  DefineRand(alias) \
  DefineOnes(alias) \
  DefineSum(alias) \
  DefineScalarMUL(alias) \
  DefineDotProduct(alias) \
  DefineMap(alias) \
  DefineArrayFree(alias) \
  DefinePrint(alias, format)
TYPE_LIST(GENERATE_ALL)

// Creates DISPATCHES for all functions
#define DISPATCH_ZEROS(real_type, alias, format) alias##Array: zeros_##alias,
#define DISPATCH_ONES(real_type, alias, format) alias##Array: ones_##alias,
#define DISPATCH_RAND(real_type, alias, format) alias##Array: rand_##alias,
#define DISPATCH_SUM(real_type, alias, format) alias##Array: sum_##alias,
#define DISPATCH_PRINT(real_type, alias, format) alias##Array: print_##alias##Array,
#define DISPATCH_SCALAR_MUL(real_type, alias, format) alias##Array: scalar_mul_##alias,
#define DISPATCH_DOT(real_type, alias, format) alias##Array: dot_product_##alias,
#define DISPATCH_MAP(real_type, alias, format) alias##Array: map_##alias##Array,

// Use the DISPATCHES to create generics
#define cml_zeros(arr) _Generic((arr), TYPE_LIST(DISPATCH_ZEROS) default: NULL)(arr)
#define cml_ones(arr) _Generic((arr), TYPE_LIST(DISPATCH_ONES) default: NULL)(arr)
#define cml_rand(arr, s, max) _Generic((arr), TYPE_LIST(DISPATCH_RAND) default: NULL)(arr, s, max)
#define cml_sum(arr1, arr2) _Generic((arr1), TYPE_LIST(DISPATCH_SUM) default: NULL)(arr1, arr2)
#define cml_print_n(arr, limit) _Generic((arr), TYPE_LIST(DISPATCH_PRINT) default: NULL)(arr, limit)
#define cml_print(arr) _Generic((arr), TYPE_LIST(DISPATCH_PRINT) default: NULL)(arr, arr.size)
#define cml_scalar_mul(arr, scalar) _Generic((arr), TYPE_LIST(DISPATCH_SCALAR_MUL) default: NULL)(arr, scalar)
#define cml_dot(arr1, arr2) _Generic((arr1), TYPE_LIST(DISPATCH_DOT) default: NULL)(arr1, arr2)
#define cml_map(arr, func) _Generic((arr), TYPE_LIST(DISPATCH_MAP) default: NULL)(arr, func)

// GNU only
#define ArrayInit(name, T, size) \
  __attribute__((cleanup(free_##T##Array))) T##Array name = Array(T, size);

int main() {
  size_t size = 10;
  ArrayInit(array, i8, size);
  ArrayInit(array1, i8, size);

  cml_rand(array, 42, 256);
  cml_rand(array1, 41, 256);
  cml_print(array);
  cml_print(array1);
  i8 res = cml_dot(array, array1);
  printf("res: %d\n", res);

  cml_print(array);
  cml_print(array1);

  return 0;
}
