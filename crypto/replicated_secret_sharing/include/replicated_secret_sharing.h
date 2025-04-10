#pragma once
#include "globals.h"
#include "tensor.h"

template <typename T>
class RSSTensor
{
public:
    Tensor<T> first;
    Tensor<T> second;
    std::vector<uint32_t> shape;

    static constexpr auto float_scale_bit = kFloat_Precision<T>;

    RSSTensor() {};

    RSSTensor(const RSSTensor &other)
    {
        shape = other.shape;
        first = other.first;
        second = other.second;
    }

    RSSTensor(std::vector<uint32_t> shape)
    {
        first.allocate(shape);
        second.allocate(shape);
        this->shape = shape;
    }

    RSSTensor(const Tensor<T> &first, const Tensor<T> &second)
    {
        always_assert(first.is_same_shape(second));
        this->shape = first.shape;
        this->first.allocate(shape);
        this->second.allocate(shape);
        this->first.copy(first);
        this->second.copy(second);
    }

    void reshape(const std::vector<uint32_t> &shape)
    {
        first.reshape(shape);
        second.reshape(shape);
        this->shape = shape;
    }

    void allocate(const std::vector<uint32_t> &shape)
    {
        this->shape = shape;
        first.allocate(shape);
        second.allocate(shape);
    }

    void free()
    {
        this->shape = {};
        this->first.free();
        this->second.free();
    }

    void rand()
    {
        first.rand();
        second.rand();
    }

    void zeros()
    {
        first.zero();
        second.zero();
    }

    void fill(T x)
    {
        first.fill(x);
        second.fill(x);
    }

    void print()
    {
        std::cout << "ReplicatedSecretSharing(\n";
        std::cout << "first: ";
        first.print();
        std::cout << "second: ";
        second.print();
        std::cout << ")\n";
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

    uint32_t size() const
    {
        return first.size();
    }

    bool is_same_shape(const RSSTensor<T> &other)
    {
        return first.assert_same_shape(other.first);
    }

    bool is_same_shape(const Tensor<T> &other)
    {
        return first.assert_same_shape(other);
    }

    void assert_same_shape(const RSSTensor<T> &other)
    {
        first.assert_same_shape(other.first);
    }

    void assert_same_shape(const Tensor<T> &other)
    {
        first.assert_same_shape(other);
    }

    RSSTensor<T> &operator=(const RSSTensor<T> &other)
    {
        if (this != &other)
        {
            shape = other.shape;
            first = other.first;
            second = other.second;
        }
        return *this;
    }

    RSSTensor<T> transpose()
    {
        // only transpose the last two dimensions
        always_assert(shape.size() >= 2);
        std::vector<uint32_t> new_shape = shape;
        std::swap(new_shape[new_shape.size() - 1], new_shape[new_shape.size() - 2]);
        RSSTensor<T> res(new_shape);

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
                    res.first.data[i * transpose_size + j * shape[shape.size() - 2] + k] =
                        first.data[i * transpose_size + k * shape[shape.size() - 1] + j];
                    res.second.data[i * transpose_size + j * shape[shape.size() - 2] + k] =
                        second.data[i * transpose_size + k * shape[shape.size() - 1] + j];
                }
            }
        }
        return res;
    }

    RSSTensor<T> permute(const std::vector<uint32_t> &permute_dims)
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

        RSSTensor<T> res(new_shape);
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
            res.first.data[i] = first.data[idx];
            res.second.data[i] = second.data[idx];
        }
        return res;
    }
};
