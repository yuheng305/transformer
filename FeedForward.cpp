#include "FeedForward.h"
#include <cmath>
#include <algorithm>

// template<size_t M, size_t N, size_t P>
// Matrix<M, N> Linear(const Matrix<M, P> &input, const Matrix<P, N> &W, double b) {   // output = input * W + b
//     // TODO: optimize matrix multiplication inside
//     Matrix<M, N> output;
//     for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
//         output[i][j] = b;
//         for (int k = 0; k < P; k++) output[i][j] += input[i][k] * W[k][j];
//     }
//     return output;
// }

// template<size_t N, size_t M>
// Tensor3D<N, M, d_model> FeedForward(const Tensor3D<N, M, d_model> &input, const Matrix<d_model, d_ff> &W1, Array<N> b1,
//                                                                           const Matrix<d_ff, d_model> &W2, Array<N> b2,
//                                                                           bool use_bias) {
//     Tensor3D<N, M, d_model> output;
//     if (!use_bias) {
//         fill(b1.begin(), b1.end(), 0);
//         fill(b2.begin(), b2.end(), 0);
//     }

//     for (int i = 0; i < N; i++) {
//         Matrix<M, d_model> h = Linear(input[i], W1, b1[i]);
//         for (int j = 0; j < M; j++) for (int k = 0; k < d_model; k++) h[j][k] = std::max(0.0, h[j][k]);
//         output[i] = Linear(h, W2, b2[i]);
//     }
//     return output;
// }

#include <vector>
#include <algorithm>
#include <iostream>
#include <random>

using Matrix = std::vector<std::vector<double>>;

Matrix createMatrix(size_t rows, size_t cols, double init_value = 0.0)
{
    return Matrix(rows, std::vector<double>(cols, init_value));
}

Matrix createRandomMatrix(size_t rows, size_t cols, double mean = 0.0, double stddev = 1.0)
{
    Matrix mat(rows, std::vector<double>(cols));
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, stddev);

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            mat[i][j] = distribution(generator);
        }
    }
    return mat;
}

std::vector<double> createRandomVector(size_t size, double mean = 0.0, double stddev = 1.0)
{
    std::vector<double> vec(size);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, stddev);

    for (size_t i = 0; i < size; ++i)
    {
        vec[i] = distribution(generator);
    }
    return vec;
}

// output = input * W + b
Matrix Linear(const Matrix &input, const Matrix &W, double b)
{
    size_t M = input.size();    // input.row
    size_t P = input[0].size(); // input.col= W.row
    size_t N = W[0].size();     // W.col

    Matrix output = createMatrix(M, N, b);

    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            for (size_t k = 0; k < P; ++k)
            {
                output[i][j] += input[i][k] * W[k][j];
            }
        }
    }
    return output;
}

std::vector<Matrix> FeedForward(const std::vector<Matrix> &input,
                                const Matrix &W1, const std::vector<double> &b1,
                                const Matrix &W2, const std::vector<double> &b2,
                                bool use_bias)
{
    size_t N = input.size();
    size_t M = input[0].size();
    size_t d_ff = W1[0].size();
    size_t d_model = W2[0].size();

    std::vector<Matrix> output(N, createMatrix(M, d_model));

    std::vector<double> b1_adj = use_bias ? b1 : std::vector<double>(M, 0.0);
    std::vector<double> b2_adj = use_bias ? b2 : std::vector<double>(M, 0.0);

    for (size_t i = 0; i < N; ++i)
    {
        Matrix h = Linear(input[i], W1, b1_adj[i]);

        for (size_t j = 0; j < M; ++j)
        {
            for (size_t k = 0; k < d_ff; ++k)
            {
                h[j][k] = std::max(0.0, h[j][k]);
            }
        }

        output[i] = Linear(h, W2, b2_adj[i]);
    }

    return output;
}