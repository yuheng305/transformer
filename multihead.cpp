#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

using namespace std;

using Matrix = std::vector<std::vector<double>>;

// const int EMBEDDING_DIM = 512;
// const int MAX_POSITION = 1000;
// const int NUM_HEADS = 8;
// const int HEAD_DIM = EMBEDDING_DIM / NUM_HEADS;

const int EMBEDDING_DIM = 200;
const int MAX_POSITION = 1000;
const int NUM_HEADS = 8;
const int HEAD_DIM = EMBEDDING_DIM / NUM_HEADS;
class HEAD
{
public:
    int rows, cols;
    vector<vector<double>> data;

    HEAD(int r, int c) : rows(r), cols(c), data(r, vector<double>(c, 0)) {}

    HEAD operator+(const HEAD &other) const
    {
        if (rows != other.rows || cols != other.cols)
        {
            throw invalid_argument("HEAD dimensions do not match for addition.");
        }
        HEAD result(rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    HEAD operator*(const HEAD &other) const
    {
        if (cols != other.rows)
        {
            throw invalid_argument("HEAD dimensions do not match for multiplication.");
        }
        HEAD result(rows, other.cols);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < other.cols; ++j)
            {
                for (int k = 0; k < cols; ++k)
                {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    Matrix toMatrix() const
    {
        Matrix mat(rows, vector<double>(cols));
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                mat[i][j] = static_cast<double>(data[i][j]);
            }
        }
        return mat;
    }
    void print() const
    {
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                cout << data[i][j] << " ";
            }
            cout << endl;
        }
    }
};

void randomizeHEAD(HEAD &matrix)
{
    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            matrix.data[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
}

// MultiHead Attention generation
void generate_multihead_qkv(const HEAD &input_embedding, vector<HEAD> &Q_heads, vector<HEAD> &K_heads, vector<HEAD> &V_heads)
{
    vector<HEAD> W_Q_heads(NUM_HEADS, HEAD(EMBEDDING_DIM, HEAD_DIM));
    vector<HEAD> W_K_heads(NUM_HEADS, HEAD(EMBEDDING_DIM, HEAD_DIM));
    vector<HEAD> W_V_heads(NUM_HEADS, HEAD(EMBEDDING_DIM, HEAD_DIM));

    for (int h = 0; h < NUM_HEADS; ++h)
    {
        randomizeHEAD(W_Q_heads[h]);
        randomizeHEAD(W_K_heads[h]);
        randomizeHEAD(W_V_heads[h]);

        Q_heads[h] = input_embedding * W_Q_heads[h];
        K_heads[h] = input_embedding * W_K_heads[h];
        V_heads[h] = input_embedding * W_V_heads[h];
    }
}

// // Concatenate heads and project output
// HEAD concatenate_and_project(const vector<HEAD> &heads)
// {
//     int sequence_length = heads[0].rows;
//     HEAD output(sequence_length, EMBEDDING_DIM);
//     for (int i = 0; i < sequence_length; ++i)
//     {
//         int col_offset = 0;
//         for (int h = 0; h < NUM_HEADS; ++h)
//         {
//             for (int j = 0; j < HEAD_DIM; ++j)
//             {
//                 output.data[i][col_offset + j] = heads[h].data[i][j];
//             }
//             col_offset += HEAD_DIM;
//         }
//     }
//     // Optional projection step if desired
//     // HEAD W_O(EMBEDDING_DIM, EMBEDDING_DIM);
//     // randomizeHEAD(W_O);
//     // output = output * W_O;
//     return output;
// }

// int main() {
//     srand(static_cast<unsigned int>(time(0))); // Seed for random number generation

//     // Example input embedding matrix with sequence length of 10
//     HEAD input_embedding(10, EMBEDDING_DIM);
//     randomizeHEAD(input_embedding);

//     // Vectors to store the resulting Q, K, V heads
//     vector<HEAD> Q_heads(NUM_HEADS, HEAD(10, HEAD_DIM));
//     vector<HEAD> K_heads(NUM_HEADS, HEAD(10, HEAD_DIM));
//     vector<HEAD> V_heads(NUM_HEADS, HEAD(10, HEAD_DIM));

//     // Generate Q, K, V heads for multi-head attention
//     generate_multihead_qkv(input_embedding, Q_heads, K_heads, V_heads);

//     // Concatenate heads and project
//     HEAD multihead_output = concatenate_and_project(V_heads);

//     // Example output to check dimensions
//     cout << "Multi-head attention output with dimensions: " << multihead_output.rows << "x" << multihead_output.cols << endl;
//     multihead_output.print();
//     return 0;
// }
