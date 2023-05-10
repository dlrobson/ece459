#include <cmath>

#define SIZE 1022

#define twoD(x,y) ((x)*(SIZE+2)+(y))
#define fourD(x1,y1,x2,y2) ((((x)*(SIZE+2)+(y))*2+(x2))*2+(y2))

extern "C"  __global__ void add(const float *P, const float *A, float *sums) {
    int y = threadIdx.x;
    if (y == 0 || y > SIZE) {
        return;
    }

    for (int x = 1; x < SIZE + 1; x++){
        sums[twoD(x, y)] = P[twoD(x, y)] + A[fourD(x, y, 1, 0)] * P[twoD(x, y - 1)]
                                        + A[fourD(x, y, 0, 1)] * P[twoD(x - 1, y)]
                                        + A[fourD(x, y, 1, 2)] * P[twoD(x, y + 1)]
                                        + A[fourD(x, y, 2, 1)] * P[twoD(x + 1, y)];
    }
}

// you may find it helpful to have out as a parameter for debugging purposes, but you don't need it in the final version
// you are also free to drop N if you want.
extern "C" __global__ void find_max_index(/*float *out, */int *out_idx, float *sums) {
    int row = threadIdx.x;
    int max_i = -1;
    for (int i = row * (SIZE + 2) + 1; i < (row + 1) * (SIZE + 2) - 1; i++) {
        if (sums[max_i] < sums[i]) {
            max_i = i;
        }
    }
    out_idx[row] = max_i;
}
