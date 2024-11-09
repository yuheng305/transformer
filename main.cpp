#include "Input_embedding_glove.cpp"
#include "multihead.cpp"
#include "part2.cpp"
#include "FeedForward.cpp"
#include "AddnNorm.cpp"
#include <iostream>

using namespace std;
void EncoderLayer(const Matrix &input_embedding)
{
    int num_tokens = input_embedding.size();
    // Initialize vectors to store the resulting Q, K, V heads
    vector<HEAD> Q_heads(NUM_HEADS, HEAD(num_tokens, HEAD_DIM));
    vector<HEAD> K_heads(NUM_HEADS, HEAD(num_tokens, HEAD_DIM));
    vector<HEAD> V_heads(NUM_HEADS, HEAD(num_tokens, HEAD_DIM));

    // Generate Q, K, V heads for multi-head attention
    HEAD input_head = HEAD(input_embedding.size(), input_embedding[0].size());
    input_head.data = input_embedding;
    generate_multihead_qkv(input_head, Q_heads, K_heads, V_heads);

    // Calculating scaled attention for each head
    vector<Matrix> attention_scores(NUM_HEADS);
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        attention_scores[h] = scaledAttention(Q_heads[h].toMatrix(), K_heads[h].toMatrix());
        softmax(attention_scores[h]); // Apply softmax to attention scores
    }

    // apply attention weights to value heads
    vector<Matrix> outputs(NUM_HEADS);
    for (int h = 0; h < NUM_HEADS; ++h)
    {
        outputs[h] = applyAttentionWeights(attention_scores[h], V_heads[h].toMatrix());
    }

    // Concatenate heads and project
    Matrix multihead_output = concatenate_matrices(outputs); // Concatenate outputs from all heads
    // cout << "Multi-head attention output with dimensions: " << multihead_output.size() << "x" << multihead_output[0].size() << endl;
    // printMatrix(multihead_output); // Assuming you have a printMatrix function to display the output

    // cout << "---------------------" << endl;
    Matrix orig_matrix = input_embedding;
    AddnNorm(multihead_output, orig_matrix);

    size_t d_model = 200;
    size_t d_ff = 200;
    size_t M = 1; // Batch size
    size_t N = 1; // Number of input tensors

    // Create random matrices and biases
    Matrix W1 = createRandomMatrix(d_model, d_ff);
    Matrix W2 = createRandomMatrix(d_ff, d_model);
    Array b1 = createRandomVector(N);
    Array b2 = createRandomVector(N);

    // Create random input tensor
    // std::vector<Matrix> input(N, createRandomMatrix(M, d_model));

    // Get FeedForward output
    auto feedforward_output = FeedForward(multihead_output, W1, b1, W2, b2, true);

    // Apply Add and Norm
    AddnNorm(feedforward_output, multihead_output);
    cout << "After processing Encoder Layer" << endl;
    printMatrix(feedforward_output);
    // return feedforward_output;
}

int main()
{
    unordered_map<string, vector<float>> embeddings;
    load_embeddings("glove.100.200d.txt", embeddings);

    string sentence;
    cout << "Input a sentence: ";
    getline(cin, sentence);

    vector<string> words = split_sentence(sentence);
    size_t num_tokens = words.size();

    HEAD input_embedding(num_tokens, EMBEDDING_DIM);
    for (size_t i = 0; i < num_tokens; ++i)
    {
        vector<float> embedding = get_embedding(words[i], embeddings);
        if (embedding.size() == EMBEDDING_DIM)
        {
            for (size_t j = 0; j < EMBEDDING_DIM; ++j)
            {
                input_embedding.data[i][j] = embedding[j];
            }
        }

        vector<float> position_vector = load_vector_position(i, EMBEDDING_DIM);
        for (size_t j = 0; j < EMBEDDING_DIM; ++j)
        {
            input_embedding.data[i][j] += position_vector[j];
        }
    }

    Matrix current_input = input_embedding.toMatrix();
    for (int i = 0; i < 6; ++i)
    {
        cout << "Processing Encoder Layer " << i + 1 << endl;
        EncoderLayer(current_input);
    }

    cout << "Final encoder output (after 6 layers):" << endl;
    printMatrix(current_input);

    return 0;
}
