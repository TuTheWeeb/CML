#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <string.h>

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
  void rand_##T(T##Array array, u32 seed, T max_value) { \
    _Pragma("omp parallel") \
    { \
    u32 my_seed = seed * (omp_get_thread_num() + 1); \
    _Pragma("omp for") \
    for (size_t i = 0; i < array.size; i++) { \
      array.data[i] = _Generic((T)0, \
          f32:  ((f32)rand_r(&my_seed) / (f32)RAND_MAX) * (f32)max_value, \
          f64: ((f64)rand_r(&my_seed) / (f64)RAND_MAX) * (f64)max_value, \
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
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < array1.size; i++) { \
      dest.data[i] = array1.data[i] + array2.data[i]; \
    } \
  }

// Defines all scalar functions
#define DefineScalarMUL(T) \
  void scalar_mul_##T(T##Array array, T scalar, T##Array dest) { \
    _Pragma("omp parallel for") \
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
      _Pragma("omp parallel for reduction(+:acum)") \
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
      _Pragma("omp parallel for") \
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
  DefinePrintWrapper(alias, format)
TYPE_LIST(GENERATE_ALL)
#undef GENERATE_ALL





// Declare all types of Matrix
#define DeclareMatrix(T) typedef struct { \
  T##Array * data; \
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
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < matrix.rs; i++) { \
      zeros_##T(matrix.data[i]); \
    } \
  }

// Defines all ones functions
#define DefineOnesMatrix(T) \
  void ones_##T##Matrix(T##Matrix matrix) { \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < matrix.rs; i++) { \
      ones_##T(matrix.data[i]); \
    } \
  }

// Defines all rands functions
#define DefineRandMatrix(T) \
  void rand_##T##Matrix(T##Matrix matrix, u32 seed, T max_value) { \
    _Pragma("omp parallel for") \
    for (size_t i = 0; i < matrix.rs; i++) { \
      rand_##T(matrix.data[i], seed, max_value); \
    } \
  }

// Defines all sum functions
#define DefineSumMatrix(T) \
  void sum_##T##Matrix(T##Matrix matrix1, T##Matrix matrix2, T##Matrix dest) { \
    if (matrix1.rs != matrix2.rs && matrix1.cs != matrix2.cs) { \
      printf("Both matrix have different dimensions (rs1: %ld != rs2: %ld && cs1: %ld != cs2: %ld), returning error!\n", matrix1.rs, matrix2.rs, matrix1.cs, matrix2.cs); \
      return; \
    } \
    for (size_t i = 0; i < matrix1.rs; i++) { \
      sum_##T(matrix1.data[i], matrix2.data[i], dest.data[i]); \
    } \
  }

// Defines all scalar functions
#define DefineScalarMULMatrix(T) \
  void scalar_mul_##T##Matrix(T##Matrix matrix , T scalar, T##Matrix dest) { \
      for (size_t i = 0; i < matrix.rs; i++) { \
        scalar_mul_##T(matrix.data[i], scalar, dest.data[i]); \
      } \
  }

// Defines all dot product funcions
#define DefineDotProductMatrix(T) \
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
        for (size_t i = 0; i < matrix->rs; i++) { \
          FreeArray(matrix->data[i]); \
        } \
        free(matrix->data); \
        matrix->data = NULL; \
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
  DefinePrintWrapperMatrix(alias)
TYPE_LIST(GENERATE_ALL)

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

// Use the DISPATCHES to create generics
//#define cml_zeros(arr) _Generic((arr), TYPE_LIST(DISPATCH_ZEROS) default: NULL)(arr)
//#define cml_ones(arr) _Generic((arr), TYPE_LIST(DISPATCH_ONES) default: NULL)(arr)
//#define cml_rand(arr, s, max) _Generic((arr), TYPE_LIST(DISPATCH_RAND) default: NULL)(arr, s, max)
//#define cml_sum(arr1, arr2, dest) _Generic((arr1), TYPE_LIST(DISPATCH_SUM) default: NULL)(arr1, arr2, dest)
//#define cml_print_n(arr, limit) _Generic((arr), TYPE_LIST(DISPATCH_PRINT) default: NULL)(arr, limit)
//#define cml_print(arr) _Generic((arr), TYPE_LIST(DISPATCH_PRINT) default: NULL)(arr, arr.size)
//#define cml_scalar_mul(arr, scalar, dest) _Generic((arr), TYPE_LIST(DISPATCH_SCALAR_MUL) default: NULL)(arr, scalar, dest)
//#define cml_dot(arr1, arr2) _Generic((arr1), TYPE_LIST(DISPATCH_DOT) default: NULL)(arr1, arr2)
//#define cml_map(arr, func, dest) _Generic((arr), TYPE_LIST(DISPATCH_MAP) default: NULL)(arr, func, dest)

// Creates DISPATCHES for all functions
#define DISPATCH_ZEROS_MATRIX(real_type, alias, format) alias##Matrix: zeros_##alias(x),
#define DISPATCH_ONES_MATRIX(real_type, alias, format) alias##Matrix: ones_##alias(x),
#define DISPATCH_RAND_MATRIX(real_type, alias, format) alias##Matrix: rand_##alias,
#define DISPATCH_SUM_MATRIX(real_type, alias, format) alias##Matrix: sum_##alias,
#define DISPATCH_PRINT_WRAPPER_MATRIX(real_type, alias, format) alias##Matrix: print_wrapper_##alias##Matrix,
#define DISPATCH_PRINT_MATRIX(real_type, alias, format) alias##Matrix: print_##alias##Matrix,
#define DISPATCH_SCALAR_MUL_MATRIX(real_type, alias, format) alias##Matrix: scalar_mul_##alias,
#define DISPATCH_DOT_MATRIX(real_type, alias, format) alias##Matrix: dot_product_##alias,
#define DISPATCH_MAP_MATRIX(real_type, alias, format) alias##Matrix: map_##alias##Matrix,

// Use the DISPATCHES to create generics
//#define cml_zeros_matrix(matrix) _Generic((matrix), TYPE_LIST(DISPATCH_ZEROS_MATRIX) default: NULL)(matrix)
//#define cml_ones_matrix(matrix) _Generic((matrix), TYPE_LIST(DISPATCH_ONES_MATRIX) default: NULL)(matrix)
//#define cml_rand_matrix(matrix, s, max) _Generic((matrix), TYPE_LIST(DISPATCH_RAND_MATRIX) default: NULL)(matrix, s, max)
//#define cml_sum_matrix(matrix1, matrix2, dest) _Generic((matrix1), TYPE_LIST(DISPATCH_SUM_MATRIX) default: NULL)(matrix1, matrix2, dest)
//#define cml_print_n_matrix(matrix, limit) _Generic((matrix), TYPE_LIST(DISPATCH_PRINT_MATRIX) default: NULL)(matrix, limit)
//#define cml_print_matrix(matrix) _Generic((matrix), TYPE_LIST(DISPATCH_PRINT_MATRIX) default: NULL)(matrix, matrix.size)
//#define cml_scalar_mul_matrix(matrix, scalar, dest) _Generic((matrix), TYPE_LIST(DISPATCH_SCALAR_MUL_MATRIX) default: NULL)(matrix, scalar, dest)
//#define cml_dot_matrix(matrix1, matrix2) _Generic((matrix1), TYPE_LIST(DISPATCH_DOT_MATRIX) default: NULL)(matrix1, matrix2)
//#define cml_map_matrix(matrix, func, dest) _Generic((matrix), TYPE_LIST(DISPATCH_MAP_MATRIX) default: NULL)(matrix, func, dest)

#define cml_zeros(obj) _Generic((obj), \
    TYPE_LIST(DISPATCH_ZEROS) \
    TYPE_LIST(DISPATCH_ZEROS_MATRIX) \
    default: NULL)(obj)
#define cml_ones(obj) _Generic((obj), \
    TYPE_LIST(DISPATCH_ONES) \
    TYPE_LIST(DISPATCH_ONES_MATRIX) \
    default: NULL)(obj)
#define cml_rand(obj, s, max) _Generic((obj), \
    TYPE_LIST(DISPATCH_RAND) \
    TYPE_LIST(DISPATCH_RAND_MATRIX) \
    default: NULL)(obj, s, max)
#define cml_sum(obj1, obj2, dest) _Generic((obj1), \
    TYPE_LIST(DISPATCH_SUM) \
    TYPE_LIST(DISPATCH_SUM_MATRIX) \
    default: NULL)(obj1, obj2, dest)
#define cml_print_n(obj, limit) _Generic((obj), \
    TYPE_LIST(DISPATCH_PRINT) \
    default: NULL)(obj, limit)
#define cml_print_matrix_n(obj, w, h) _Generic((obj), \
    TYPE_LIST(DISPATCH_PRINT_MATRIX) \
    default: NULL)(obj, w, h)
#define cml_print(obj) _Generic((obj), \
    TYPE_LIST(DISPATCH_PRINT_WRAPPER) \
    TYPE_LIST(DISPATCH_PRINT_WRAPPER_MATRIX) \
    default: NULL)(obj)
#define cml_scalar_mul(obj, scalar, dest) _Generic((obj), \
    TYPE_LIST(DISPATCH_SCALAR_MUL) \
    TYPE_LIST(DISPATCH_SCALAR_MUL_MATRIX) \
    default: NULL)(obj, scalar, dest)
#define cml_dot(obj1, obj2) _Generic((obj1), \
    TYPE_LIST(DISPATCH_DOT) \
    TYPE_LIST(DISPATCH_DOT_MATRIX) \
    default: NULL)(obj1, obj2)
#define cml_map(obj, func, dest) _Generic((obj), \
    TYPE_LIST(DISPATCH_MAP) \
    TYPE_LIST(DISPATCH_MAP_MATRIX) \
    default: NULL)(obj, func, dest)

// GNU only, using __VA_ARGS__ to start the Array with stack values
// if something go wrong its because the size of the array that you are trying to use as input is less than the size that you informed 
#define ArrayInit(name, T, size, ...) \
  __attribute__((cleanup(free_##T##Array))) T##Array name = Array(T, size); \
  __VA_OPT__(\
      memcpy(name.data, (T[]){__VA_ARGS__}, size*sizeof(T)) \
      ) 

#define ARG_2(a, b, ...) b
#define IS_EMPTY(...) ARG_2(dummy __VA_OPT__(,) \
    for (size_t i = 0; i < name.rs; i++) { \
      name.data[i] = Array(T, cols); \
    }, \
    { \
      T _tmp_buff[rows][cols] = {__VA_ARGS__}; \
      for (size_t i = 0; i < name.rs; i++) { \
        name.data[i] = Array(T, cols); \
        memcpy(name.data[i].data, _tmp_buff[i], cols*sizeof(T)); \
      } \
    },\
    ~)
#define CONCAT_HELPER(a, b) a##b
#define CONCAT(a, b) CONCAT_HELPER(a, b)

// GNU only, using __VA_ARGS__ to start the Array with stack values
// if something go wrong its because the size of the matrix that you are trying to use as input is less than the size that you informed
#define MatrixInit(name, T, rows, cols, ...) \
  __attribute__((cleanup(free_##T##Matrix))) T##Matrix name = (T##Matrix) {\
      .data = (T##Array*) malloc(sizeof(T##Array) * (rows)),\
      .rs=rows,\
      .cs=cols \
  }; \
  for (size_t i = 0; i < name.rs; i++) { \
    name.data[i] = Array(T, cols); \
  } \
  __VA_OPT__({ \
        T _tmp_buff[rows][cols] = {__VA_ARGS__}; \
        for (size_t i = 0; i < name.rs; i++) { \
          name.data[i] = Array(T, cols); \
          memcpy(name.data[i].data, _tmp_buff[i], cols*sizeof(T)); \
        } \
      })

int main() {
  MatrixInit(matrix, f32, 2, 2, {1.0, 3.0}, {2.0, 2.0});
  MatrixInit(matrix1, f32, 2, 2, {2.0, 3.0}, {3.0, 5.0});
  cml_print(matrix);
  return 0;
}
