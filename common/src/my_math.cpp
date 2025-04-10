#include "my_math.h"

void compute_matmul_info(std::vector<uint32_t> a_shape, std::vector<uint32_t> b_shape, std::vector<uint32_t> c_shape,
                         uint32_t &row, uint32_t &common_dim, uint32_t &col, uint32_t &broadcast_size,
                         uint32_t &common_size, bool &is_b_broadcast)
{
    const uint32_t a_shape_size = a_shape.size();
    const uint32_t b_shape_size = b_shape.size();
    const uint32_t c_shape_size = c_shape.size();

    row = c_shape[c_shape_size - 2];
    common_dim = a_shape[a_shape_size - 1];
    col = c_shape[c_shape_size - 1];
    broadcast_size = 1;
    common_size = 1;
    is_b_broadcast = false;
    bool flag = false;

    uint32_t max_shape_size = std::max(a_shape_size, b_shape_size);

    std::vector<uint32_t> a_aligned_shape(max_shape_size, 1);
    std::vector<uint32_t> b_aligned_shape(max_shape_size, 1);

    // 填充a和b的对齐后的形状
    for (int i = 0; i < a_shape_size; i++)
    {
        a_aligned_shape[max_shape_size - a_shape_size + i] = a_shape[i];
    }
    for (int i = 0; i < b_shape_size; i++)
    {
        b_aligned_shape[max_shape_size - b_shape_size + i] = b_shape[i];
    }

    for (int i = 0; i < max_shape_size - 2; i++)
    {
        // 对应维度的大小
        uint32_t a_dim_size = a_aligned_shape[max_shape_size - 3 - i];
        uint32_t b_dim_size = b_aligned_shape[max_shape_size - 3 - i];

        if (a_dim_size == b_dim_size)
        {
            // 如果对应维度大小相等，直接相乘
            if (flag)
            {
                broadcast_size *= a_dim_size;
            }
            else
            {
                common_size *= a_dim_size;
            }
        }
        else if (a_dim_size == 1 || b_dim_size == 1)
        {
            // 如果其中一个矩阵在该维度为1，表示可以进行广播
            broadcast_size *= std::max(a_dim_size, b_dim_size);
            common_size *= std::min(a_dim_size, b_dim_size);

            if (b_dim_size == 1)
            {
                // 如果b在该维度为1，表示b需要广播
                is_b_broadcast = true;
            }
            flag = true;
        }
        else
        {
            // 如果维度不匹配且不能广播，无法进行操作
            // 这里需要报错或者其他逻辑处理
            throw std::invalid_argument("Matrices cannot be broadcasted");
        }
    }
}