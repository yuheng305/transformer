#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>

using namespace std;

template <typename T>
void AddnNorm(vector<vector<T>> &target, vector<vector<T>> &orig, double gamma = 1, double beta = 0){
    // Residual neural network: He et al. (2016)
    // Layer normalization: Ba et al. (2016)
    // this function performs the LayerNorm(x + Sublayer(x)) operation directly on target
    // T type must support element-wise addition and scalar addition/multiplication
    // target and orig are assumed to be vectors of size N x 512

    int sz = target.size();

    for (int i = 0; i < sz; ++i){
        double mu = 0, sigma = 0;

        // this is VERY vectorizable
        for (int j = 0; j < 512; ++j){
            target[i][j] += orig[i][j]; // residual connections
            mu += target[i][j]; // E[X]
            sigma += target[i][j]*target[i][j]; // E[X^2]
        }

        mu /= 512;
        sigma = sqrt(sigma/512 - mu*mu + 1e-10); 
        // sigma^2(X) = Var(X) = E[X^2] - E^2[X]

        // same goes for this one
        for (int j = 0; j < 512; ++j)
            target[i][j] = gamma/sigma*(target[i][j] - mu) + beta;
        
        // also try this (slightly) optimized version
        // mu /= 512;
        // sigma = gamma / sqrt(sigma/512 - mu*mu + 1e-10); 
        // mu = beta - sigma*mu;
        //
        // for (int j = 0; j < 512; ++j)
        //     target[i][j] = sigma*target[i][j] + mu;
    }

    return;
}