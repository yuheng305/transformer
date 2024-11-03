#include "Input_embedding_glove.cpp"
#include "multihead.cpp"
#include "part2.cpp"
#include "FeedForward.cpp"
#include "AddnNorm.cpp"
#include <bits/stdc++.h>

using namespace std;

int main()
{
    unordered_map<string, vector<float>> embeddings;
    load_embeddings("glove.100.200d.txt", embeddings); // Load file

    // Input sentence
    string sentence;
    cout << "Input a sentence: ";
    getline(cin, sentence);

    // Words to vec
    vector<string> words = split_sentence(sentence);
    size_t num_tokens = words.size();

    // Initialize input_embedding with the appropriate dimensions
    HEAD input_embedding(num_tokens, EMBEDDING_DIM);

    // Populate input_embedding with embeddings for each word
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
    }

    // Initialize vectors to store the resulting Q, K, V heads
    vector<HEAD> Q_heads(NUM_HEADS, HEAD(num_tokens, HEAD_DIM));
    vector<HEAD> K_heads(NUM_HEADS, HEAD(num_tokens, HEAD_DIM));
    vector<HEAD> V_heads(NUM_HEADS, HEAD(num_tokens, HEAD_DIM));

    // Generate Q, K, V heads for multi-head attention
    generate_multihead_qkv(input_embedding, Q_heads, K_heads, V_heads);

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
    cout << "Multi-head attention output with dimensions: " << multihead_output.size() << "x" << multihead_output[0].size() << endl;
    printMatrix(multihead_output); // Assuming you have a printMatrix function to display the output

    // add and norm multihead_output
    //  FeedForward part
    size_t d_model = 200;
    size_t d_ff = 200;
    size_t M = 1;

    Matrix W1 = createRandomMatrix(d_model, d_ff);
    Matrix W2 = createRandomMatrix(d_ff, d_model);
    vector<double> b1 = createRandomVector(M);
    vector<double> b2 = createRandomVector(M);

    vector<Matrix> input(1, createRandomMatrix(M, d_model));

    // FeedForward output
    auto feedforward_output = FeedForward(input, W1, b1, W2, b2, true);

    // Apply Add and Norm
    for (size_t i = 0; i < feedforward_output.size(); ++i)
    {
        AddnNorm(feedforward_output[i], input[i]); // Apply Add and Norm with default gamma and beta
    }

    cout << "\n FeedForward with Add and Norm output\n";
    for (const auto &mat : feedforward_output)
    {
        for (const auto &row : mat)
        {
            for (const auto &val : row)
            {
                cout << val << " ";
            }
            cout << endl;
        }
        cout << "-------" << endl;
    }

    return 0;
}
