#define CML_CROP 10000
#include "cml.h"

// Activation Functions for cml_map
f64 relu(f64 x) { return x > 0 ? x : 0; }
f64 relu_derivative(f64 x) { return x > 0 ? 1.0f : 0.0f; }
// A helper for element-wise multiplication (Hadamard product)
f64 multiply_elements(f64 a, f64 b) { return a * b; }

int main(int argc, char **argv) {
    size_t hidden_size = 1000;
    size_t out_size = 1;
    #define n_samples 5
    f64 learning_rate = 0.001f;
    u32 seed = 42;
    size_t epochs = 100000;
    f64 input = 8.0f;
    f64 loss = 1.0f;

    if (argc > 1) epochs = atoll(argv[1]);
    if (argc > 2) learning_rate = strtod(argv[2], NULL);
    if (argc > 3) input = strtod(argv[3], NULL);

    // 1. The Data
    MatrixInit(X, f64, n_samples, 1, -2.0f, -1.0f, 0.0f, 1.f, 2.f);
    MatrixInit(Y, f64, n_samples, 1, -4.0f, -2.0f, 0.0f, 2.f, 4.f);

    // 2. Weights and Biases (Pre-allocating)
    MatrixInit(W1, f64, 1, hidden_size);
    MatrixInit(b1, f64, 1, hidden_size);
    MatrixInit(W2, f64, hidden_size, out_size);
    MatrixInit(b2, f64, 1, out_size);

    // Initialize Weights randomly, Biases to 0
    cml_rand(W1, seed, 0.1f);
    cml_zeros(b1);
    cml_rand(W2, seed, 0.1f);
    cml_zeros(b2);

    // --- Pre-allocate ALL intermediate matrices for the loop ---
    MatrixInit(Z1, f64, n_samples, hidden_size);
    MatrixInit(A1, f64, n_samples, hidden_size);
    MatrixInit(Z2, f64, n_samples, out_size);
    MatrixInit(A2, f64, n_samples, out_size);
    
    // Gradients
    MatrixInit(dZ2, f64, n_samples, out_size);
    MatrixInit(dW2, f64, hidden_size, out_size);
    MatrixInit(db2, f64, 1, out_size);
    
    MatrixInit(dA1, f64, n_samples, hidden_size);
    MatrixInit(dZ1, f64, n_samples, hidden_size);
    MatrixInit(dW1, f64, 1, hidden_size);
    MatrixInit(db1, f64, 1, hidden_size);
    
    // Transposes & Math helpers
    MatrixInit(A1_T, f64, hidden_size, n_samples);
    MatrixInit(W2_T, f64, out_size, hidden_size);
    MatrixInit(X_T, f64, 1, n_samples);
    MatrixInit(relu_deriv_Z1, f64, n_samples, hidden_size);

    // 4. The Training Loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // --- FORWARD PASS ---
        // Z1 = np.dot(X, W1) + b1
        cml_mul(X, W1, Z1);
        cml_sum(Z1, b1, Z1); // Broadcasting bias!
        
        // A1 = relu(Z1)
        cml_map(Z1, relu, A1);

        // Z2 = np.dot(A1, W2) + b2
        cml_mul(A1, W2, Z2);
        cml_sum(Z2, b2, Z2); // Broadcasting bias!
        
        // A2 = Z2
        CopyMatrix(A2, Z2);
        //for(size_t i=0; i<n_samples*out_size; i++) A2.allocator[i] = Z2.allocator[i];

        // loss = np.mean((A2 - Y) ** 2)
        MatrixInit(dest, f64, n_samples, out_size);
        cml_sub(A2, Y, dest);
        loss = cml_mean(dest);

        // --- BACKWARD PASS ---
        // dZ2 = 2 * (A2 - Y) / n_samples
        cml_sub(A2, Y, dZ2);
        cml_scalar_mul(dZ2, 2.0f / n_samples, dZ2);

        // dW2 = np.dot(A1.T, dZ2)
        cml_transpose(A1, A1_T);
        cml_mul(A1_T, dZ2, dW2);

        // db2 = np.sum(dZ2, axis=0)
        cml_sum_axis0(dZ2, db2);

        // dA1 = np.dot(dZ2, W2.T)
        cml_transpose(W2, W2_T);
        cml_mul(dZ2, W2_T, dA1);

        // dZ1 = dA1 * relu_derivative(Z1) (Element-wise)
        cml_map(Z1, relu_derivative, relu_deriv_Z1);
        for(size_t i=0; i<n_samples*hidden_size; i++) {
             dZ1.allocator[i] = dA1.allocator[i] * relu_deriv_Z1.allocator[i];
        }

        // dW1 = np.dot(X.T, dZ1)
        cml_transpose(X, X_T);
        cml_mul(X_T, dZ1, dW1);

        // db1 = np.sum(dZ1, axis=0)
        cml_sum_axis0(dZ1, db1);

        // --- UPDATE WEIGHTS ---
        // W1 -= learning_rate * dW1 (We multiply by -LR, then add)
        cml_scalar_mul(dW1, -learning_rate, dW1);
        cml_sum(W1, dW1, W1);

        cml_scalar_mul(db1, -learning_rate, db1);
        cml_sum(b1, db1, b1);

        cml_scalar_mul(dW2, -learning_rate, dW2);
        cml_sum(W2, dW2, W2);

        cml_scalar_mul(db2, -learning_rate, db2);
        cml_sum(b2, db2, b2);
    }

    MatrixInit(test, f64, 1, 1, input);
    MatrixInit(predicition, f64, 1, 1, 0);
    MatrixInit(Z1_test, f64, 1, hidden_size);
    MatrixInit(A1_test, f64, 1, hidden_size);
    
    // Add an intermediate 1x1 matrix for the layer 2 calculation
    MatrixInit(Z2_test, f64, 1, out_size); 

    // Z1_test = np.dot(test_input, W1) + b1
    cml_mul(test, W1, Z1_test); // Order fixed: test * W1
    cml_sum(Z1_test, b1, Z1_test);
    
    // A1_test = relu(Z1_test)
    cml_map(Z1_test, relu, A1_test); // Input fixed to Z1_test
    
    // Z2_test = np.dot(A1_test, W2)
    cml_mul(A1_test, W2, Z2_test); // Destination is now a 1x1 matrix
    
    // prediction = Z2_test + b2
    cml_sum(Z2_test, b2, predicition); 
    
    printf("\nAI Predicts:\n");
    cml_print(predicition);
    printf("loss: %lf\n", loss);
    printf("Training Complete!\n");
    printf("Final predictions (A2):\n");
    cml_print(A2);

    return 0;
}
