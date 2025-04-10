#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <omp.h>
#include <cstdint>
#include <cassert>
#include <vector>
#include <cmath>
#include <filesystem>
#include "globals.h"
#include "my_assert.h"
#include "my_math.h"

template <typename T>
class Tensor
{
    bool isFreed = false;

public:
    T *data = nullptr;
    T *d_data = nullptr;
    std::vector<uint32_t> shape;
    bool isOwner = true;

    bool wasCopiedToCpu = false;

    bool wasFreed() { return isFreed; }

    Tensor() {};

    Tensor(const Tensor<T> &other) : shape(other.shape), isOwner(other.isOwner),
                                     wasCopiedToCpu(other.wasCopiedToCpu), isFreed(other.wasFreed())
    {
        if (other.data != nullptr)
        {
            data = new T[other.size()];
            std::copy(other.data, other.data + other.size(), data);
        }

        if (other.d_data != nullptr)
        {
            d_data = new T[other.size()];
            std::copy(other.d_data, other.d_data + other.size(), d_data);
        }
    }

    T value_at(const std::vector<uint32_t> &idx) const
    {
        always_assert(idx.size() == this->shape.size());
        uint32_t offset = 0;
        for (uint32_t i = 0; i < idx.size(); i++)
        {
            always_assert(idx[i] < this->shape[i]);
            uint32_t stride = 1;
            for (uint32_t j = i + 1; j < idx.size(); j++)
            {
                stride *= this->shape[j];
            }
            offset += idx[i] * stride;
        }
        return this->data[offset];
    }

    void allocate(const std::vector<uint32_t> &s)
    {
        always_assert(isOwner);
        this->shape = s;
        if (this->size() > 0)
        {
            this->data = new T[this->size()];
            isFreed = false;
        }
        else
        {
            this->data = nullptr;
            isFreed = true;
        }
    }

    void free()
    {
        always_assert(isOwner);
        if (isFreed)
        {
            return;
        }
        if (this->size() == 0)
        {
            return;
        }
        delete[] data;
        this->shape = {};
        isFreed = true;
    }

    void resize(const std::vector<uint32_t> &s)
    {
        always_assert(isOwner);
        if (s.size() == this->shape.size())
        {
            bool allSameDims = true;
            for (uint32_t i = 0; i < s.size(); i++)
            {
                if (s[i] != this->shape[i])
                {
                    allSameDims = false;
                    break;
                }
            }
            if (allSameDims)
            {
                return;
            }
        }
        free();
        allocate(s);
    }

    void reshape(const std::vector<uint32_t> &new_shape)
    {
        uint32_t new_size = 1;
        for (uint32_t i = 0; i < new_shape.size(); i++)
        {
            new_size *= new_shape[i];
        }
        always_assert(new_size == size());
        shape = new_shape;
    }

    Tensor(const std::vector<uint32_t> &s)
    {
        allocate(s);
    }

    Tensor(std::initializer_list<uint32_t> s)
    {
        allocate(s);
    }

    Tensor(T *data, T *d_data, const std::vector<uint32_t> &s)
    {
        this->data = data;
        this->d_data = d_data;
        this->shape = s;
        this->isOwner = false;
    }

    Tensor(T *data, const std::vector<uint32_t> &s)
    {
        this->data = data;
        this->d_data = nullptr;
        this->shape = s;
        this->isOwner = false;
    }

    ~Tensor()
    {
        // printf("Freeing tensor=%d, %lx, %lu\n", isOwner, data, size());
        if (isOwner)
            free();
    }

    uint32_t size() const
    {
        if (this->shape.size() == 0)
        {
            return 0;
        }
        uint32_t s = 1;
        for (auto d : this->shape)
        {
            s *= d;
        }
        return s;
    }

    bool is_same_shape(const Tensor<T> &other) const
    {
        if (this->shape.size() != other.shape.size())
        {
            return false;
        }
        for (uint32_t i = 0; i < this->shape.size(); i++)
        {
            if (this->shape[i] != other.shape[i])
            {
                return false;
            }
        }
        return true;
    }

    void assert_same_shape(const Tensor<T> &other)
    {
        always_assert(this->shape.size() == other.shape.size());
#pragma omp parallel for
        for (uint32_t i = 0; i < this->shape.size(); i++)
        {
            always_assert(this->shape[i] == other.shape[i]);
        }
    }

    bool is_zero() const
    {
        bool flag = true;
#pragma omp parallel for
        for (uint32_t i = 0; i < size(); i++)
        {
            if (data[i] != 0)
            {
                flag = false;
            }
        }
        return flag;
    }

    void assert_zero()
    {
        always_assert(is_zero());
    }

    void copy(const Tensor<T> &other)
    {
        assert_same_shape(other);
        // memcpy(data, other.data, size() * sizeof(T));
#pragma omp parallel for
        for (uint32_t i = 0; i < size(); ++i)
        {
            data[i] = other.data[i];
        }
    }

    Tensor<T> clone()
    {
        Tensor<T> t(this->shape);
        t.copy(*this);
        return t;
    }

    void fill(T x)
    {
#pragma omp parallel for
        for (uint32_t i = 0; i < size(); i++)
        {
            data[i] = x;
        }
    }

    void zero()
    {
        fill(0);
    }

    void ones()
    {
        fill(1);
    }

    void rand()
    {
#pragma omp parallel for
        for (uint32_t i = 0; i < size(); i++)
        {
            data[i] = std::rand();
        }
    }

    void rand(T min, T max)
    {
#pragma omp parallel for
        for (uint32_t i = 0; i < size(); i++)
        {
            data[i] = std::rand() % (T)(max - min + 1) + min;
        }
    }

    void print()
    {
        std::cout << "Tensor: (";
        for (uint32_t i = 0; i < size() - 1; i++)
        {
            std::cout << data[i] << ", ";
        }
        std::cout << data[size() - 1] << ")" << std::endl;
    }

    void printshape()
    {
        std::cout << "(";
        for (int i = 0; i < this->shape.size(); i++)
        {
            std::cout << this->shape[i] << ", ";
        }
        std::cout << ")" << std::endl;
    }

    T multidir_broadcast_value(const std::vector<uint32_t> &broadcast_shape, const std::vector<uint32_t> &idx) const
    {
        always_assert(broadcast_shape.size() >= this->shape.size());
        always_assert(broadcast_shape.size() == idx.size());
        int num_broadcast_dims = broadcast_shape.size() - this->shape.size();
        std::vector<uint32_t> new_idx;
        for (uint32_t i = 0; i < this->shape.size(); i++)
        {
            always_assert(this->shape[i] == 1 || this->shape[i] == broadcast_shape[i + num_broadcast_dims]);
            if (this->shape[i] == 1)
            {
                new_idx.push_back(0);
            }
            else
            {
                always_assert(idx[i + num_broadcast_dims] < this->shape[i]);
                new_idx.push_back(idx[i + num_broadcast_dims]);
            }
        }
        return this->value_at(new_idx);
    }

    Tensor<T> view(uint32_t i)
    {
        always_assert(i < shape[0]);
        uint32_t newsize = size() / shape[0];
        auto newshape = shape;
        newshape.erase(newshape.begin());
        return Tensor<T>(data + i * newsize, newshape);
    }

    Tensor<T> &operator=(const Tensor<T> &other)
    {
        if (this != &other)
        {
            delete[] data;
            allocate(other.shape);
            std::copy(other.data, other.data + size(), data);
        }
        return *this;
    }

    static void add(Tensor<T> &result, const Tensor<T> &a, const Tensor<T> &b)
    {
        result.assert_same_shape(a);
        result.assert_same_shape(b);
#pragma omp parallel for
        for (uint32_t i = 0; i < result.size(); i++)
        {
            result.data[i] = a.data[i] + b.data[i];
        }
    }

    static void add(Tensor<T> &result, const Tensor<T> &a, const T &b)
    {
        result.assert_same_shape(a);

#pragma omp parallel for
        for (uint32_t i = 0; i < result.size(); i++)
        {
            result.data[i] = a.data[i] + b;
        }
    }

    static void minus(Tensor<T> &result, const Tensor<T> &a, const Tensor<T> &b)
    {
        result.assert_same_shape(a);
        result.assert_same_shape(b);

#pragma omp parallel for
        for (uint32_t i = 0; i < result.size(); i++)
        {
            result.data[i] = a.data[i] - b.data[i];
        }
    }

    static void minus(Tensor<T> &result, const Tensor<T> &a, const T &b)
    {
        result.assert_same_shape(a);

#pragma omp parallel for
        for (uint32_t i = 0; i < result.size(); i++)
        {
            result.data[i] = a.data[i] - b;
        }
    }

    static void minus(Tensor<T> &result, const T &a, const Tensor<T> &b)
    {
        result.assert_same_shape(b);

#pragma omp parallel for
        for (uint32_t i = 0; i < result.size(); i++)
        {
            result.data[i] = a - b.data[i];
        }
    }

    static void mul(Tensor<T> &result, const Tensor<T> &a, const Tensor<T> &b)
    {
        result.assert_same_shape(a);
        result.assert_same_shape(b);

#pragma omp parallel for
        for (uint32_t i = 0; i < result.size(); i++)
        {
            result.data[i] = a.data[i] * b.data[i];
        }
    }

    static void mul(Tensor<T> &result, const Tensor<T> &a, const T &b)
    {
        result.assert_same_shape(a);

#pragma omp parallel for
        for (uint32_t i = 0; i < result.size(); i++)
        {
            result.data[i] = a.data[i] * b;
        }
    }

    static void matmul_2d(Tensor<T> &result, const Tensor<T> &a, const Tensor<T> &b)
    {
        always_assert(a.shape.size() == 2);
        always_assert(b.shape.size() == 2);
        always_assert(a.shape[1] == b.shape[0]);
        always_assert(result.shape[0] == a.shape[0]);
        always_assert(result.shape[1] == b.shape[1]);

#pragma omp parallel for
        for (uint32_t i = 0; i < a.shape[0]; i++)
        {
            for (uint32_t j = 0; j < b.shape[1]; j++)
            {
                T sum = 0;
                for (uint32_t k = 0; k < a.shape[1]; k++)
                {
                    sum += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
                }
                result.data[i * result.shape[1] + j] = sum;
            }
        }
    }

    static void matmul(Tensor<T> &result, const Tensor<T> &a, const Tensor<T> &b, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast)
    {
        uint32_t index, index_a, index_b;
        if (is_b_broadcast)
        {
            for (uint32_t i = 0; i < broadcast_size; i++)
            {
                for (uint32_t j = 0; j < common_size; j++)
                {
#pragma omp parallel for
                    for (uint32_t k = 0; k < row; k++)
                    {
                        for (uint32_t l = 0; l < col; l++)
                        {
                            index = i * common_size * row * col + j * row * col + k * col + l;
                            index_a = i * common_size * row * common_dim + j * row * common_dim + k * common_dim;
                            index_b = j * common_dim * col + l;
                            result.data[index] = 0;
                            for (uint32_t m = 0; m < common_dim; m++)
                            {
                                result.data[index] += a.data[index_a + m] * b.data[index_b + m * col];
                            }
                        }
                    }
                }
            }
        }
        else
        {
            for (uint32_t i = 0; i < broadcast_size; i++)
            {
                for (uint32_t j = 0; j < common_size; j++)
                {
#pragma omp parallel for
                    for (uint32_t k = 0; k < row; k++)
                    {
                        for (uint32_t l = 0; l < col; l++)
                        {
                            uint32_t index = 0;
                            uint32_t index_a = 0;
                            uint32_t index_b = 0;

                            index = i * common_size * row * col + j * row * col + k * col + l;
                            index_a = j * row * common_dim + k * common_dim;
                            index_b = i * common_size * common_dim * col + j * common_dim * col + l;
                            result.data[index] = 0;
                            for (uint32_t m = 0; m < common_dim; m++)
                            {
                                result.data[index] += a.data[index_a + m] * b.data[index_b + m * col];
                            }
                        }
                    }
                }
            }
        }
    }

    static void matmul(Tensor<T> &result, const Tensor<T> &a, const Tensor<T> &b)
    {
        if (a.shape.size() == 2 && b.shape.size() == 2)
        {
            matmul_2d(result, a, b);
        }
        else
        {
            const uint32_t a_shape_size = a.shape.size();
            const uint32_t b_shape_size = b.shape.size();
            const uint32_t c_shape_size = result.shape.size();
            always_assert(a_shape_size >= 2 && b_shape_size >= 2 && c_shape_size > 2);
            always_assert(a.shape[a_shape_size - 1] == b.shape[b_shape_size - 2]);
            always_assert(result.shape[c_shape_size - 2] == a.shape[a_shape_size - 2]);
            always_assert(result.shape[c_shape_size - 1] == b.shape[b_shape_size - 1]);

            uint32_t row, common_dim, col, broadcast_size, common_size;
            bool is_b_broadcast;

            compute_matmul_info(a.shape, b.shape, result.shape, row, common_dim, col, broadcast_size, common_size,
                                is_b_broadcast);

            matmul(result, a, b, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);
        }
    }

    static void div(Tensor<T> &self, const T &b)
    {
#pragma omp parallel for
        for (uint32_t i = 0; i < self.size(); i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                self.data[i] = (T)((double)((int32_t)self.data[i]) / (double)((int32_t)b));
            }
            else
            {
                self.data[i] = (T)((double)((int64_t)self.data[i]) / (double)((int64_t)b));
            }
        }
    }

    static void div(Tensor<T> &result, const Tensor<T> &a, const T &b)
    {
        result.assert_same_shape(a);

        // TODO: support 64-bit
#pragma omp parallel for
        for (uint32_t i = 0; i < result.size(); i++)
        {
            result.data[i] = (T)((double)((int32_t)a.data[i]) / (double)((int32_t)b));
        }
    }

    Tensor<T> transpose()
    {
        // only transpose the last two dimensions
        always_assert(shape.size() >= 2);
        std::vector<uint32_t> new_shape = shape;
        std::swap(new_shape[new_shape.size() - 1], new_shape[new_shape.size() - 2]);
        Tensor<T> res(new_shape);

        uint32_t freeze_size = 1;
        uint32_t transpose_size = shape[shape.size() - 1] * shape[shape.size() - 2];
        for (int i = 0; i < shape.size() - 2; i++)
        {
            freeze_size *= shape[i];
        }

        for (int i = 0; i < freeze_size; i++)
        {
            for (int j = 0; j < shape[shape.size() - 1]; j++)
            {
                for (int k = 0; k < shape[shape.size() - 2]; k++)
                {
                    res.data[i * transpose_size + j * shape[shape.size() - 2] + k] =
                        data[i * transpose_size + k * shape[shape.size() - 1] + j];
                }
            }
        }
        return res;
    }

    Tensor<T> permute(const std::vector<uint32_t> &permute_dims)
    {
        always_assert(permute_dims.size() == shape.size() && permute_dims.size() > 1);
        int shape_size = shape.size();
        for (int i = 0; i < shape_size; i++)
        {
            always_assert(permute_dims[i] < shape_size);
        }

        std::vector<uint32_t> new_shape = shape;
        for (uint32_t i = 0; i < shape_size; i++)
        {
            new_shape[i] = shape[permute_dims[i]];
        }

        std::vector<uint32_t> idx_base(shape_size, 1);
        for (int i = shape_size - 2; i >= 0; i--)
        {
            idx_base[i] = idx_base[i + 1] * shape[i + 1];
        }

        std::vector<uint32_t> new_idx_base = idx_base;
        for (uint32_t i = 0; i < shape_size; i++)
        {
            new_idx_base[i] = idx_base[permute_dims[i]];
        }

        Tensor<T> res(new_shape);
        uint32_t idx = 0;
        for (uint32_t i = 0; i < size(); i++)
        {
            idx = 0;
            uint32_t temp = i;
            for (int j = shape_size - 1; j >= 0; j--)
            {
                idx += (temp % new_shape[j]) * new_idx_base[j];
                temp /= new_shape[j];
            }
            res.data[i] = data[idx];
        }
        return res;
    }
};

#endif // TENSOR_H