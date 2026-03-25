#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <string.h>

#define CML_CROP 1000

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
//T * restrict data; // removed for testing
#define DeclareArray(T) typedef struct { \
  T * data; \
  size_t size; \
} T##Array;

// Automatically alocates memory
#define Array(T, sz) (T##Array) {.data = (T*)malloc(sizeof(T) * (sz)), .size=(sz)}

// For manual use
#define FreeArray(array) \
    free(array.data);

// For manual use
#define FreeArrayPointer(array_pointer) \
  FreeArray(*array); \
  free(array);

// Defines all zeros functions
#define DefineZeros(T) \
  void zeros_##T(T##Array array) { \
    if (array.size < CML_CROP) {\
      for (size_t i = 0; i < array.size; i++) array.data[i] = 0; \
      return;\
    }\
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < array.size; i++) array.data[i] = 0; \
  }

// Defines all ones functions
#define DefineOnes(T) \
  void ones_##T(T##Array array) { \
    if (array.size < CML_CROP) {\
      for (size_t i = 0; i < array.size; i++) array.data[i] = 1; \
      return;\
    }\
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < array.size; i++) array.data[i] = 1; \
  }

// Defines all rands functions
#define DefineRand(T) \
  void rand_##T(T##Array array, u32 seed, T max_value) { \
    if (array.size < CML_CROP) {\
      for (size_t i = 0; i < array.size; i++) { \
        array.data[i] = _Generic((T)0, \
            f32:  (((f32)rand_r(&seed) / (f32)RAND_MAX) * 2.0f * (f32)max_value) - (f32)max_value, \
            f64: (((f64)rand_r(&seed) / (f64)RAND_MAX) * 2.0f * (f64)max_value) - (f64)max_value, \
            default: (T)(rand_r(&seed) % (u64) max_value) \
          ); \
      }\
      return;\
    }\
    _Pragma("omp parallel") \
    { \
    u32 my_seed = seed * (omp_get_thread_num() + 1); \
    _Pragma("omp for") \
    for (size_t i = 0; i < array.size; i++) { \
      array.data[i] = _Generic((T)0, \
          f32:  (((f32)rand_r(&my_seed) / (f32)RAND_MAX) * 2.0f * (f32)max_value) - (f32)max_value, \
          f64: (((f64)rand_r(&my_seed) / (f64)RAND_MAX) * 2.0f * (f64)max_value) - (f64)max_value, \
          default: (T)(rand_r(&my_seed) % (u64) max_value) \
        ); \
      } \
    } \
  }

// Defines all sum functions
#define DefineSum(T) \
  void sum_##T(T##Array array1, T##Array array2, T##Array dest) { \
    if (array1.size != array2.size) { \
      printf("The arrays have different sizes (%ld != %ld), returning error!\n", array1.size, array2.size); \
      return; \
    } \
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < array1.size; i++) { \
      dest.data[i] = array1.data[i] + array2.data[i]; \
    } \
  }

// Defines all sub functions
#define DefineSub(T) \
  void sub_##T(T##Array array1, T##Array array2, T##Array dest) { \
    if (array1.size != array2.size) { \
      printf("The arrays have different sizes (%ld != %ld), returning error!\n", array1.size, array2.size); \
      return; \
    } \
    if (array1.size < CML_CROP) {\
      for (size_t i = 0; i < array1.size; i++) { \
        dest.data[i] = array1.data[i] - array2.data[i]; \
      } \
      return;\
    }\
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < array1.size; i++) { \
      dest.data[i] = array1.data[i] - array2.data[i]; \
    } \
  }

// Defines all scalar functions
#define DefineScalarMUL(T) \
  void scalar_mul_##T(T##Array array, T scalar, T##Array dest) { \
    if (array.size < CML_CROP) {\
      for (size_t i = 0; i < array.size; i++) { \
        dest.data[i] = array.data[i] * (T) scalar; \
      } \
      return;\
    }\
    _Pragma("omp parallel for simd") \
      for (size_t i = 0; i < array.size; i++) { \
        dest.data[i] = array.data[i] * (T) scalar; \
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
      /*Jumps if the size is less than CML_CROP*/\
      if (array1.size < CML_CROP) {\
        for (size_t i = 0; i < array1.size; i++) {\
          acum += array1.data[i] * array2.data[i]; \
        } \
        return acum;\
      }\
      _Pragma("omp parallel for simd reduction(+:acum)") \
      for (size_t i = 0; i < array1.size; i++) {\
        acum += array1.data[i] * array2.data[i]; \
      } \
      return acum; \
    }

// Defines all map functions
#define DefineMap(T) \
    void map_##T##Array(T##Array array, T (*func)(T), T##Array dest) { \
      if (array.data == NULL) { \
        printf("Array is empty\n"); \
        return; \
      } \
      if (array.size < CML_CROP) {\
       for (size_t i = 0; i < array.size; i++) {\
         dest.data[i] = func(array.data[i]); \
       } \
       return;\
      }\
      _Pragma("omp parallel for simd") \
      for (size_t i = 0; i < array.size; i++) {\
        dest.data[i] = func(array.data[i]); \
      } \
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

#define DefinePrintWrapper(T, format) \
    void print_wrapper_##T##Array(T##Array array) {\
      print_##T##Array(array, array.size);\
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
  DefinePrint(alias, format) \
  DefinePrintWrapper(alias, format) \
  DefineSub(alias)
TYPE_LIST(GENERATE_ALL)
#undef GENERATE_ALL

// Declare all types of Matrix
#define DeclareMatrix(T) typedef struct { \
  T##Array * data; \
  T * allocator;\
  size_t rs; \
  size_t cs; \
} T##Matrix;

#define AcessMatrix(matrix, rows, cols) matrix.data[rows][cols]

// For manual use
#define FreeMatrix(matrix) \
    for (size_t i = 0; i < matrix.rs; i++) { \
      free(matrix[i].data); \
    } \
    free(matrix.data);

// For manual use
#define FreeMatrixPointer(matrix_pointer) \
  FreeArray(*matrix_pointer); \
  free(matrix_pointer);

// Defines all zeros functions
#define DefineZerosMatrix(T) \
  void zeros_##T##Matrix(T##Matrix matrix) { \
    if (matrix.rs * matrix.cs < CML_CROP) {\
      for (size_t i = 0; i < matrix.rs; i++) { \
        for (size_t j = 0; j < matrix.cs; j++) \
          matrix.data[i].data[j] = 0;\
      } \
      return; \
    }\
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < matrix.rs; i++) { \
      for (size_t j = 0; j < matrix.cs; j++) \
        matrix.data[i].data[j] = 0;\
    } \
  }

// Defines all ones functions
#define DefineOnesMatrix(T) \
  void ones_##T##Matrix(T##Matrix matrix) { \
    if (matrix.rs * matrix.cs < CML_CROP) {\
      for (size_t i = 0; i < matrix.rs; i++) { \
        for (size_t j = 0; j < matrix.cs; j++) \
          matrix.data[i].data[j] = 1;\
      } \
      return;\
    }\
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < matrix.rs; i++) { \
      for (size_t j = 0; j < matrix.cs; j++) \
        matrix.data[i].data[j] = 1;\
    } \
  }

// Defines all rands functions
#define DefineRandMatrix(T) \
  void rand_##T##Matrix(T##Matrix matrix, u32 seed, T max_value) { \
    for (size_t i = 0; i < matrix.rs; i++) { \
      rand_##T(matrix.data[i], seed, max_value); \
    } \
  }

// Defines all sum functions, assuming broadcasting sum when sizes are different
#define DefineSumMatrix(T) \
  void sum_##T##Matrix(T##Matrix matrix1, T##Matrix matrix2, T##Matrix dest) { \
    if (matrix1.rs != matrix2.rs || matrix1.cs != matrix2.cs) { \
      _Pragma("omp parallel for simd") \
      for (size_t i = 0; i < matrix1.rs; i++) {\
        size_t m2_i = i % matrix2.rs; /*makes sure the size is respected*/ \
        for (size_t j = 0; j < matrix1.cs; j++) {\
          size_t m2_j = j % matrix2.cs; /*makes sure the size is respected*/ \
          dest.data[i].data[j] = matrix1.data[i].data[j] + matrix2.data[m2_i].data[m2_j]; \
        }\
      }\
      return; \
    } \
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < matrix1.rs; i++) { \
      for (size_t j = 0; j < matrix1.cs; j++) \
        dest.data[i].data[j] = matrix1.data[i].data[j] + matrix2.data[i].data[j]; \
    } \
  }

#define DefineSumAxis0Matrix(T) \
  void sum_axis0_##T##Matrix(T##Matrix matrix, T##Matrix dest) { \
    if (dest.rs != 1 || dest.cs != matrix.cs) { \
        printf("Dest must be 1x%ld\n", matrix.cs); return; \
    } \
    for (size_t j = 0; j < matrix.cs; j++) { \
      T sum = 0; \
      for (size_t i = 0; i < matrix.rs; i++) { \
        sum += matrix.data[i].data[j]; \
      } \
      dest.data[0].data[j] = sum; \
    } \
  }

#define DefineSubMatrix(T)\
    void sub_##T##Matrix(T##Matrix matrix1, T##Matrix matrix2, T##Matrix dest) { \
    if (matrix1.rs != matrix2.rs || matrix1.cs != matrix2.cs) { \
      _Pragma("omp parallel for simd") \
      for (size_t i = 0; i < matrix1.rs; i++) {\
        size_t m2_i = i % matrix2.rs; /*makes sure the size is respected*/ \
        for (size_t j = 0; j < matrix1.cs; j++) {\
          size_t m2_j = j % matrix2.cs; /*makes sure the size is respected*/ \
          dest.data[i].data[j] = matrix1.data[i].data[j] - matrix2.data[m2_i].data[m2_j]; \
        }\
      }\
      return; \
    } \
    _Pragma("omp parallel for simd") \
    for (size_t i = 0; i < matrix1.rs; i++) { \
      for (size_t j = 0; j < matrix1.cs; j++) \
        dest.data[i].data[j] = matrix1.data[i].data[j] - matrix2.data[i].data[j]; \
    } \
  }

// Defines all scalar functions
#define DefineScalarMULMatrix(T) \
  void scalar_mul_##T##Matrix(T##Matrix matrix , T scalar, T##Matrix dest) { \
      if (matrix.rs * matrix.cs < CML_CROP) {\
        for (size_t i = 0; i < matrix.rs; i++) { \
          for (size_t j = 0; j < matrix.cs; j++) \
            dest.data[i].data[j] = matrix.data[i].data[j] * scalar; \
        } \
        return;\
      }\
      _Pragma("omp parallel for simd") \
      for (size_t i = 0; i < matrix.rs; i++) { \
        for (size_t j = 0; j < matrix.cs; j++) \
          dest.data[i].data[j] = matrix.data[i].data[j] * scalar; \
      } \
  }

// Defines all matrix mul functions 
#define DefineMultiplicationMatrix(T) \
    void mul_matrix_##T##Matrix(T##Matrix matrix1, T##Matrix matrix2, T##Matrix dest) { \
      if (matrix1.cs != matrix2.rs || dest.rs != matrix1.rs || dest.cs != matrix2.cs) { \
        printf("MatMul dimension mismatch!\n"); return; \
      } \
      _Pragma("omp parallel for simd") \
      for (size_t i = 0; i < matrix1.rs; i++) { \
        /* Initialize the destination row to 0 first */ \
        for (size_t j = 0; j < matrix2.cs; j++) dest.data[i].data[j] = 0; \
        /* The i, k, j loop ordering for Cache Locality */ \
        for (size_t k = 0; k < matrix1.cs; k++) { \
          T r = matrix1.data[i].data[k]; \
          for (size_t j = 0; j < matrix2.cs; j++) { \
            dest.data[i].data[j] += r * matrix2.data[k].data[j]; \
          } \
        } \
      } \
    }

#define DefineMatrixArrayMul(T) \
    void mul_matrix_array_##T(T##Matrix matrix, T##Array array, T##Array dest) {\
      if (matrix.cs != array.size) {\
        printf("matrix and array of incompatible sizes! (%ld x %ld) and (%ld)", matrix.rs, matrix.cs, array.size); \
        return; \
      }\
      if (matrix.cs * matrix.rs < CML_CROP) {\
        for (size_t i = 0; i < matrix.rs; i++) { \
          for (size_t j = 0; j < matrix.cs; j++) { \
            dest.data[i] += matrix.data[i].data[j]*array.data[j];\
        } \
      } \
        return;\
      }\
      _Pragma("omp parallel for simd") \
      for (size_t i = 0; i < matrix.rs; i++) {\
        for (size_t j = 0; j < matrix.cs; j++) {\
          dest.data[i] += matrix.data[i].data[j]*array.data[j];\
        }\
      }\
    }\

#define DefineMatrixTranspose(T) \
    void matrix_transpose_##T(T##Matrix matrix, T##Matrix dest) { \
      if (matrix.cs != dest.rs || matrix.rs != dest.cs) {\
        printf("matrix1(%ld x %ld) cant transpose to dest (%ld x %ld)\n", matrix.rs, matrix.cs, dest.rs, dest.cs); \
        return;\
      }\
      if (matrix.cs * matrix.rs < CML_CROP) {\
        for (size_t i = 0; i < matrix.rs; i++) { \
          for (size_t j = 0; j < matrix.cs; j++) { \
            dest.data[j].data[i] = matrix.data[i].data[j]; \
        } \
      } \
        return;\
      }\
      _Pragma("omp parallel for simd")\
      for (size_t i = 0; i < matrix.rs; i++) { \
        for (size_t j = 0; j < matrix.cs; j++) { \
          dest.data[j].data[i] = matrix.data[i].data[j]; \
        } \
      } \
    }

// Defines all map functions
#define DefineMapMatrix(T) \
    void map_##T##Matrix(T##Matrix matrix, T (*func)(T), T##Matrix dest) { \
      if (matrix.data == NULL) { \
        printf("Matrix is empty\n"); \
        return; \
      } \
      for (size_t i = 0; i < matrix.rs; i++) {\
        map_##T##Array(matrix.data[i], func, dest.data[i]); \
      } \
    }

// Defines all print functions
#define DefinePrintMatrix(T) \
  void print_##T##Matrix(T##Matrix matrix, size_t w, size_t h) { \
    if (w >= matrix.rs) { \
      w = matrix.rs; \
    } \
    if (h >= matrix.cs) { \
      h = matrix.cs; \
    } \
    printf("{\n"); \
    for (size_t i = 0; i < w; i++) {\
      printf(" "); \
      print_##T##Array(matrix.data[i], h); \
    }\
    printf("}\n"); \
  }

// Define all array clean functions
#define DefineMatrixFree(T) \
    void free_##T##Matrix(T##Matrix * matrix) {\
      if (matrix->data != NULL) {\
        free(matrix->data);\
        free(matrix->allocator);\
        matrix->data = NULL; \
        matrix->allocator = NULL; \
      } \
    }

#define DefinePrintWrapperMatrix(T) \
    void print_wrapper_##T##Matrix(T##Matrix matrix) {\
      print_##T##Matrix(matrix, matrix.rs, matrix.cs);\
    }

// Expand each macro
#define GENERATE_ALL(real_type, alias, format) \
  DeclareMatrix(alias) \
  DefineZerosMatrix(alias) \
  DefineRandMatrix(alias) \
  DefineOnesMatrix(alias) \
  DefineSumMatrix(alias) \
  DefineScalarMULMatrix(alias) \
  DefineMapMatrix(alias) \
  DefineMatrixFree(alias) \
  DefinePrintMatrix(alias) \
  DefineMultiplicationMatrix(alias) \
  DefinePrintWrapperMatrix(alias) \
  DefineMatrixArrayMul(alias) \
  DefineMatrixTranspose(alias) \
  DefineSubMatrix(alias) \
  DefineSumAxis0Matrix(alias)
TYPE_LIST(GENERATE_ALL)
#undef GENERATE_ALL

// Creates DISPATCHES for all functions
#define DISPATCH_ZEROS(real_type, alias, format) alias##Array: zeros_##alias,
#define DISPATCH_ONES(real_type, alias, format) alias##Array: ones_##alias,
#define DISPATCH_RAND(real_type, alias, format) alias##Array: rand_##alias,
#define DISPATCH_SUM(real_type, alias, format) alias##Array: sum_##alias,
#define DISPATCH_PRINT_WRAPPER(real_type, alias, format) alias##Array: print_wrapper_##alias##Array,
#define DISPATCH_PRINT(real_type, alias, format) alias##Array: print_##alias##Array,
#define DISPATCH_SCALAR_MUL(real_type, alias, format) alias##Array: scalar_mul_##alias,
#define DISPATCH_DOT(real_type, alias, format) alias##Array: dot_product_##alias,
#define DISPATCH_MAP(real_type, alias, format) alias##Array: map_##alias##Array,
#define DISPATCH_SUB_ARRAY(real_type, alias, format) alias##Array: sub_##alias,

#define DISPATCH_ZEROS_MATRIX(real_type, alias, format) alias##Matrix: zeros_##alias##Matrix,
#define DISPATCH_ONES_MATRIX(real_type, alias, format) alias##Matrix: ones_##alias##Matrix,
#define DISPATCH_RAND_MATRIX(real_type, alias, format) alias##Matrix: rand_##alias##Matrix,
#define DISPATCH_SUM_MATRIX(real_type, alias, format) alias##Matrix: sum_##alias##Matrix,
#define DISPATCH_PRINT_WRAPPER_MATRIX(real_type, alias, format) alias##Matrix: print_wrapper_##alias##Matrix,
#define DISPATCH_PRINT_MATRIX(real_type, alias, format) alias##Matrix: print_##alias##Matrix,
#define DISPATCH_SCALAR_MUL_MATRIX(real_type, alias, format) alias##Matrix: scalar_mul_##alias##Matrix,
#define DISPATCH_MUL_MATRIX(real_type, alias, format) alias##Matrix: mul_matrix_##alias##Matrix,
#define DISPATCH_MAP_MATRIX(real_type, alias, format) alias##Matrix: map_##alias##Matrix,
#define DISPATCH_MATRIX_ARRAY_MUL(real_type, alias, format) alias##Matrix: mul_matrix_array_##alias,
#define DISPATCH_MATRIX_TRANSPOSE(real_type, alias, format) alias##Matrix: matrix_transpose_##alias,
#define DISPATCH_SUB_MATRIX(real_type, alias, format) alias##Matrix: sub_##alias##Matrix,
#define DISPATCH_SUM_AXIS0_MATRIX(real_type, alias, format) alias##Matrix: sum_axis0_##alias##Matrix,

// Macro for each variant of functions
#define cml_zeros(obj) _Generic((obj), \
    TYPE_LIST(DISPATCH_ZEROS) \
    TYPE_LIST(DISPATCH_ZEROS_MATRIX) \
    default: NULL \
    )(obj)
#define cml_ones(obj) _Generic((obj), \
    TYPE_LIST(DISPATCH_ONES) \
    TYPE_LIST(DISPATCH_ONES_MATRIX) \
    default: NULL \
    )(obj)
#define cml_rand(obj, s, max) _Generic((obj), \
    TYPE_LIST(DISPATCH_RAND) \
    TYPE_LIST(DISPATCH_RAND_MATRIX) \
    default: NULL \
    )(obj, s, max)
#define cml_sum(obj1, obj2, dest) _Generic((obj1), \
    TYPE_LIST(DISPATCH_SUM) \
    TYPE_LIST(DISPATCH_SUM_MATRIX) \
    default: NULL \
    )(obj1, obj2, dest)
#define cml_sum_axis0(matrix, dest) _Generic((matrix), \
    TYPE_LIST(DISPATCH_SUM_AXIS0_MATRIX) \
    default: NULL \
    )(matrix, dest)
#define cml_sub(obj1, obj2, dest) _Generic((obj1), \
    TYPE_LIST(DISPATCH_SUB_MATRIX) \
    TYPE_LIST(DISPATCH_SUB_ARRAY) \
    default: NULL \
    )(obj1, obj2, dest)
#define cml_print_n(obj, limit) _Generic((obj), \
    TYPE_LIST(DISPATCH_PRINT) \
    default: NULL \
    )(obj, limit)
#define cml_print_matrix_n(obj, w, h) _Generic((obj), \
    TYPE_LIST(DISPATCH_PRINT_MATRIX) \
    default: NULL \
    )(obj, w, h)
#define cml_print(obj) _Generic((obj), \
    TYPE_LIST(DISPATCH_PRINT_WRAPPER) \
    TYPE_LIST(DISPATCH_PRINT_WRAPPER_MATRIX) \
    default: NULL \
    )(obj)
#define cml_scalar_mul(obj, scalar, dest) _Generic((obj), \
    TYPE_LIST(DISPATCH_SCALAR_MUL) \
    TYPE_LIST(DISPATCH_SCALAR_MUL_MATRIX) \
    default: NULL \
    )(obj, scalar, dest)
#define cml_dot(obj1, obj2) _Generic((obj1), \
    TYPE_LIST(DISPATCH_DOT) \
    default: NULL)(obj1, obj2)

#define cml_mul(matrix, obj, obj_dest) _Generic((matrix), \
    TYPE_LIST(DISPATCH_MUL_MATRIX) \
    default: NULL \
    )(matrix, obj, obj_dest)

#define cml_mul_array(matrix, obj, obj_dest) _Generic((matrix), \
    TYPE_LIST(DISPATCH_MATRIX_ARRAY_MUL) \
    default: NULL \
    )(matrix, obj, obj_dest)

#define cml_transpose(matrix, dest) _Generic((matrix), \
    TYPE_LIST(DISPATCH_MATRIX_TRANSPOSE) \
    default: NULL \
    )(matrix, dest)

#define cml_map(obj, func, dest) _Generic((obj), \
    TYPE_LIST(DISPATCH_MAP) \
    TYPE_LIST(DISPATCH_MAP_MATRIX) \
    default: NULL \
    )(obj, func, dest)

// GNU only, using __VA_ARGS__ to start the Array with stack values
// if something go wrong its because the size of the array that you are trying to use as input is less than the size that you informed 
#define ArrayInit(name, T, size, ...) \
  __attribute__((cleanup(free_##T##Array))) T##Array name = Array(T, size); \
  __VA_OPT__(\
      memcpy(name.data, (T[]){__VA_ARGS__}, size*sizeof(T)) \
      ) 

#define ARG_2(a, b, ...) b
#define IS_EMPTY(...) ARG_2(dummy __VA_OPT__(,) 0, 1, ~)
#define CONCAT_HELPER(a, b) a##b
#define CONCAT(a, b) CONCAT_HELPER(a, b)

// Garante que vai fazer o Loop uma unica vez
#define MATRIX_INIT_1(name, T, rows, cols) \
    for (size_t i = 0; i < rows*cols; i+=cols) {\
      name.data[i/cols].data = name.allocator + i;\
      name.data[i/cols].size = cols;\
    }\

#define MATRIX_INIT_0(name, T, rows, cols, ...) \
    { \
      T _tmp_buff[rows*cols] = {__VA_ARGS__}; \
      for (size_t i = 0; i < rows*cols; i+=cols) { \
        name.data[i/cols].data = name.allocator + i;\
        name.data[i/cols].size = cols;\
        memcpy(name.data[i/cols].data, _tmp_buff + i, cols*sizeof(T)); \
      } \
    }

// GNU only, using __VA_ARGS__ to start the Array with stack values
// if something go wrong its because the size of the matrix that you are trying to use as input is less than the size that you informed
#define MatrixInit(name, T, rows, cols, ...) \
  __attribute__((cleanup(free_##T##Matrix))) T##Matrix name; \
  {\
    name = (T##Matrix) {\
      .data=(T##Array*) malloc(sizeof(T##Array) * rows), \
      .allocator=malloc(sizeof(T) * rows * cols), \
      .rs=rows,\
      .cs=cols \
    }; \
    CONCAT(MATRIX_INIT_, IS_EMPTY(__VA_ARGS__))(name, T, rows, cols __VA_OPT__(,) __VA_ARGS__) \
  }
