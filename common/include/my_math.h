#ifndef MY_MATH_H
#define MY_MATH_H

#include <cstdint>
#include <vector>
#include <stdexcept>

#define ReLU(x) ((x) > 0 ? (x) : 0)
#define GeLU(x) (0.5f * (x) * (1 + tanh(sqrt(2.0f / M_PI) * ((x) + 0.044715f * pow((x), 3)))))

void compute_matmul_info(std::vector<uint32_t> a_shape, std::vector<uint32_t> b_shape, std::vector<uint32_t> c_shape,
                         uint32_t &row, uint32_t &common_dim, uint32_t &col, uint32_t &broadcast_size,
                         uint32_t &common_size, bool &is_b_broadcast);

#endif // MY_MATH_H
