#include "api.hh"
#include "xtensor/core/xoperation.hpp"
#include <xtensor/generators/xrandom.hpp>
#include <stdfloat>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << "N (N x N 8bw kernel)" << std::endl;
        std::cerr << "       " << argv[0] << "N M (N x M 8bw kernel)" << std::endl;
        return 1;
    }
    int N = std::atoi(argv[1]);
    int M = (argc >= 3) ? std::atoi(argv[2]) : N;
    xt::random::seed(0);
    auto kernel = xt::random::randint({N, M}, -128, 128);
    auto kernel_f = xt::cast<std::float32_t>(kernel);
    float t0 = clock();
    volatile auto solution =
        solve(kernel_f, "wmc", "auto", -1, -2, {}, {}, -1, -1, false);
    float t1 = clock();
    std::cout << "Time: " << (t1 - t0) / CLOCKS_PER_SEC << "s" << std::endl;
    return 0;
}
