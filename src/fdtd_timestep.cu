#include <chrono>
#include <math.h>
#include "solver_matriz.h"
#include "solver_func.h"

__global__
void timestepE(const Tensor E, const Tensor H, const Tensor J, const Params params) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int thread_i = index; thread_i < E.Hx.length * E.Hx.size_1; thread_i += stride) {
        int i, j;
        converte_index(E.Hx, thread_i, i, j);

        if (i > 0 && j > 0 && i < E.Hx.size_0 - 1 && j < E.Hx.size_1 - 1) {
            double Ec = params.Ec; 
            double Jc = params.Jc; 

            //updatePML()

            double value;

            value = Get(E.Hx, i, j);

            value += Ec * (Get(H.Hz, i, j) - Get(H.Hz, i, j - 1));
            value -= Ec * (Get(H.Ey, i, j) - Get(H.Ey, i, j));

            Set(E.Hx, i, j, value);

            value = Get(E.Ey, i, j);

            value += Ec * (Get(H.Hx, i, j) - Get(H.Hx, i, j));
            value -= Ec * (Get(H.Hz, i, j) - Get(H.Hz, i - 1, j));

            Set(E.Ey, i, j, value);

            value = Get(E.Hz, i, j);

            value += Ec * (Get(H.Ey, i, j) - Get(H.Ey, i - 1, j));
            value -= Ec * (Get(H.Hx, i, j) - Get(H.Hx, i, j - 1));

            Set(E.Hz, i, j, value);

            // Update da fonte
            value = Get(E.Hx, params.source_i, params.source_j);
            value -= Jc * Get(J.Hx, params.source_i, params.source_j) * exp(-params.t / params.tau) * cos(params.omega * params.t);
            Set(E.Hx, params.source_i, params.source_j, value);

            value = Get(E.Ey, params.source_i, params.source_j);
            value -= Jc * Get(J.Ey, params.source_i, params.source_j) * exp(-params.t / params.tau) * cos(params.omega * params.t);
            Set(E.Ey, params.source_i, params.source_j, value);

            value = Get(E.Hz, params.source_i, params.source_j);
            value -= Jc * Get(J.Hz, params.source_i, params.source_j) * exp(-params.t / params.tau) * cos(params.omega * params.t);
            Set(E.Hz, params.source_i, params.source_j, value);
        }
    }
}

__global__
void timestepH(const Tensor E, Tensor H, const Tensor J, const Params params) {
    int index = blockIdx.Hx * blockDim.Hx + threadIdx.Hx;
    int stride = blockDim.Hx * gridDim.Hx;

    for (int thread_i = index; thread_i < E.Hx.length * E.Hx.size_1; thread_i += stride) {
        int i, j;
        converte_index(E.Hx, thread_i, i, j);

        if (i > 0 && j > 0 && i < E.Hx.size_0 - 1 && j < E.Hx.size_1 - 1) {
            double Hc = params.Hc;
            double value;

            value = Get(H.Hx, i, j);

            value += Hc * (Get(E.Ey, i, j + 1) - Get(E.Ey, i, j));
            value -= Hc * (Get(E.Hz, i + 1, j) - Get(E.Hz, i, j));

            Set(H.Hx, i, j, value);

            value = Get(H.Ey, i, j);

            value += Hc * (Get(E.Hz, i + 1, j) - Get(E.Hz, i, j));
            value -= Hc * (Get(E.Hx, i, j + 1) - Get(E.Hx, i, j));

            Set(H.Ey, i, j, value);

            value = Get(H.Hz, i, j);

            value += Hc * (Get(E.Hx, i, j + 1) - Get(E.Hx, i, j));
            value -= Hc * (Get(E.Ey, i + 1, j) - Get(E.Ey, i, j));
            
            Set(H.Hz, i, j, value);
        }
    }
}