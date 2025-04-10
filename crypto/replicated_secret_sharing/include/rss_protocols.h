#pragma once

#include <fstream>
#include <cmath>
#include "globals.h"
#include "params.h"
#include "rss_protocols.h"
#include "ass_protocols.h"
#include "party3pc.h"

namespace rss_protocols
{
    namespace debug
    {
        template <typename T>
        void openPrintReal(RSSTensor<T> &x);

        template <typename T>
        void openPrintReal(RSSTensor<T> &x, int index);

        template <typename T>
        void openPrintReal(RSSTensor<T> &x, std::vector<int> index);

        template <typename T>
        void openPrintReal(RSSTensor<T> &x, int start, int end);

        template <typename T>
        void printRealToFile(RSSTensor<T> &x, const std::string &file_name);
    }

    namespace utils
    {
        template <typename T>
        void RSSMatMul(RSSTensor<T> &x, RSSTensor<T> &y, Tensor<T> &res, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast);

        template <typename T>
        void RSSMatMul(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast);

        template <typename T>
        void RSSMatMul(Tensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast);

        template <typename T>
        void privateCompare(Tensor<T> &x_with_pre, Tensor<T> &x_with_next, Tensor<T> &res_with_pre,
                            Tensor<T> &res_with_next, Parameters<T> &parameter);

        template <typename T>
        void getk(RSSTensor<T> &x, RSSTensor<T> &k, Parameters<T> &parameter, bool malicious = false);

        template <typename T>
        void gelu_same_scale(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);
    }

    template <typename T>
    void restore(RSSTensor<T> &x, Tensor<T> &res, bool malicious = false);

    template <typename T>
    void reconstruct_to(int target_id, RSSTensor<T> &x, Tensor<T> &res = NULL, bool malicious = false);

    template <typename T>
    void share(Tensor<T> &x, RSSTensor<T> &res);

    template <typename T>
    void recv_shares_from(int source_id, RSSTensor<T> &res);

    template <typename T>
    void coin(std::vector<uint32_t> shape, RSSTensor<T> &res);

    template <typename T>
    void reshare(Tensor<T> &x, RSSTensor<T> &res);

    template <typename T>
    RSSTensor<T> reshare(Tensor<T> &x);

    template <typename T>
    void add(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void add(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void add(RSSTensor<T> &x, T y, RSSTensor<T> &res);

    template <typename T>
    void sub(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void sub(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void sub(RSSTensor<T> &x, T y, RSSTensor<T> &res);

    template <typename T>
    void sub(T x, RSSTensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void mulConst(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void mulConst(RSSTensor<T> &x, T y, RSSTensor<T> &res);

    template <typename T>
    void mulConstAddBias(RSSTensor<T> &x, T y, T bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstAddBias(RSSTensor<T> &x, T y, RSSTensor<T> &bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstAddBias(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstSubBias(RSSTensor<T> &x, T y, T bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstSubBias(RSSTensor<T> &x, T y, RSSTensor<T> &bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstSubBias(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &bias, RSSTensor<T> &res);

    template <typename T>
    void mul(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, bool needTrunc = false, bool malicious = false);

    template <typename T>
    void mul(RSSTensor<T> &x, std::pair<T, T> &y, RSSTensor<T> &res);

    template <typename T>
    void square(RSSTensor<T> &x, RSSTensor<T> &res, bool needTrunc = false, bool malicious = false);

    template <typename T>
    void matMul(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, bool needTrunc = false, bool malicious = false);

    template <typename T>
    void truncate(RSSTensor<T> &x, RSSTensor<T> &res, size_t scale, bool malicious = false);

    template <typename T>
    void truncate(RSSTensor<T> &x, size_t scale, bool malicious = false);

    template <typename T>
    void truncate(RSSTensor<T> &x, bool malicious = false);

    template <typename T>
    void checkZero(RSSTensor<T> &x);

    template <typename T>
    void macCheck(const RSSTensor<T> &x, const RSSTensor<T> &mx, const std::pair<T, T> &mac_key);

    template <typename T>
    void macCheck(RSSTensor<T> &x, RSSTensor<T> &mx, RSSTensor<T> &mac_key);

    template <typename T>
    void macCheckSimulate(uint32_t size);

    template <typename T>
    void pc_msb(RSSTensor<T> &x, Tensor<T> &res_with_pre, Tensor<T> &res_with_next,
                Parameters<T> &parameter, const uint32_t size, bool malicious = false);

    template <typename T>
    void nonNegative(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat = true, bool malicious = false);

    template <typename T>
    void greaterEqual(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat = true, bool malicious = false);

    template <typename T>
    void select(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious = false);

    template <typename T>
    void select(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, uint32_t y_num, Parameters<T> &parameter, bool malicious = false);

    template <typename T>
    void lut(RSSTensor<T> &x, RSSTensor<T> &res, LUT_Param<T> &parameter, bool malicious = false);

    template <typename T>
    void lut(RSSTensor<T> &x, RSSTensor<T> &res1, RSSTensor<T> &res2, LUT_Param<T> &parameter1, LUT_Param<T> &parameter2, bool malicious = false);

    template <typename T>
    void inv(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void rsqrt(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void gelu(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void max_last_dim(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void neg_exp(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &Parameters, bool malicious = false);

    template <typename T>
    void softmax_forward(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious = false);

    template <typename T, typename U>
    void downcast(RSSTensor<T> &x, RSSTensor<U> &res);

    template <typename U, typename T>
    void upcast(RSSTensor<U> &x, RSSTensor<T> &res, int party_id, bool malicious = false);
}

template <typename T>
void rss_protocols::debug::openPrintReal(RSSTensor<T> &x)
{
    openPrintReal(x, 0, x.size());
}

template <typename T>
void rss_protocols::debug::openPrintReal(RSSTensor<T> &x, int index)
{
    openPrintReal(x, index, index + 1);
}

template <typename T>
void rss_protocols::debug::openPrintReal(RSSTensor<T> &x, std::vector<int> index)
{
    Tensor<T> real_res(x.shape);
    restore(x, real_res);

    if (Party3PC::getInstance().party_id == 0)
    {
        for (int i : index)
        {
            always_assert(i < x.size());
            std::cout << (float)(long)real_res.data[i] / (1 << x.float_scale_bit) << ", ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void rss_protocols::debug::openPrintReal(RSSTensor<T> &x, int start, int end)
{
    always_assert(start < end);
    Tensor<T> real_res(x.shape);
    restore(x, real_res);

    if (Party3PC::getInstance().party_id == 0)
    {
        for (int i = start; i < end; i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                std::cout << (float)(int)real_res.data[i] / (1 << x.float_scale_bit) << ", ";
            }
            else
            {
                std::cout << (double)(long)real_res.data[i] / (1 << x.float_scale_bit) << ", ";
            }
        }
        std::cout << std::endl;
    }
}

template <typename T>
void rss_protocols::debug::printRealToFile(RSSTensor<T> &x, const std::string &file_name)
{
    Tensor<T> real_res(x.shape);
    restore(x, real_res);

    if (Party3PC::getInstance().party_id == 0)
    {
        std::ofstream outFile;
        outFile.open(file_name);
        outFile << "[";
        for (int i = 0; i < x.size(); i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                outFile << ((float)(int)real_res.data[i] / (1 << x.float_scale_bit)) << ", ";
            }
            else
            {
                outFile << ((double)(long)real_res.data[i] / (1 << x.float_scale_bit)) << ", ";
            }
        }
        outFile << "]" << std::endl;
    }
}

template <typename T>
void rss_protocols::utils::RSSMatMul(RSSTensor<T> &x, RSSTensor<T> &y, Tensor<T> &res, const uint32_t broadcast_size,
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

                        res.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.data[index] += x.first.data[index_a + m] * y.first.data[index_b + m * col] + x.second.data[index_a + m] * y.first.data[index_b + m * col] + x.first.data[index_a + m] * y.second.data[index_b + m * col];
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
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = j * row * common_dim + k * common_dim;
                        index_b = i * common_size * common_dim * col + j * common_dim * col + l;

                        res.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.data[index] += x.first.data[index_a + m] * y.first.data[index_b + m * col] + x.second.data[index_a + m] * y.first.data[index_b + m * col] + x.first.data[index_a + m] * y.second.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void rss_protocols::utils::RSSMatMul(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res, const uint32_t broadcast_size,
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

                        res.first.data[index] = 0;
                        res.second.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.first.data[index] += x.first.data[index_a + m] * y.data[index_b + m * col];
                            res.second.data[index] += x.second.data[index_a + m] * y.data[index_b + m * col];
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
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = j * row * common_dim + k * common_dim;
                        index_b = i * common_size * common_dim * col + j * common_dim * col + l;

                        res.first.data[index] = 0;
                        res.second.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.first.data[index] += x.first.data[index_a + m] * y.data[index_b + m * col];
                            res.second.data[index] += x.second.data[index_a + m] * y.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void rss_protocols::utils::RSSMatMul(Tensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, const uint32_t broadcast_size,
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

                        res.first.data[index] = 0;
                        res.second.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.first.data[index] += x.data[index_a + m] * y.first.data[index_b + m * col];
                            res.second.data[index] += x.data[index_a + m] * y.second.data[index_b + m * col];
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
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = j * row * common_dim + k * common_dim;
                        index_b = i * common_size * common_dim * col + j * common_dim * col + l;

                        res.first.data[index] = 0;
                        res.second.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.first.data[index] += x.data[index_a + m] * y.first.data[index_b + m * col];
                            res.second.data[index] += x.data[index_a + m] * y.second.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void rss_protocols::restore(RSSTensor<T> &x, Tensor<T> &res, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    x.assert_same_shape(res);
    party.send_tensor_to(party.next_party_id, x.first);

    party.recv_tensor_from(party.pre_party_id, res);

    if (malicious)
    {
        party.send_tensor_to(party.pre_party_id, x.second);
        Tensor<T> tmp(x.shape);
        party.recv_tensor_from(party.next_party_id, tmp);

        Tensor<T>::minus(tmp, res, tmp);
        tmp.assert_zero();
    }

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.data[i] = res.data[i] + x.first.data[i] + x.second.data[i];
    }
}

template <typename T>
void rss_protocols::reconstruct_to(int target_id, RSSTensor<T> &x, Tensor<T> &res, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    if (target_id == party.pre_party_id)
    {
        if (malicious)
        {
            party.send_tensor_to(target_id, x.second);
        }
    }
    else if (target_id == party.next_party_id)
    {
        party.send_tensor_to(target_id, x.first);
    }
    else
    {
        Tensor<T> tmp(x.shape);
        party.recv_tensor_from(party.pre_party_id, res);

        if (malicious)
        {
            party.recv_tensor_from(party.next_party_id, tmp);

            Tensor<T>::minus(tmp, tmp, res);
            tmp.assert_zero();
        }

#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.data[i] = res.data[i] + x.first.data[i] + x.second.data[i];
        }
    }
}

template <typename T>
void rss_protocols::share(Tensor<T> &x, RSSTensor<T> &res)
{
    Party3PC &party = Party3PC::getInstance();
    res.assert_same_shape(x);
    res.rand();
    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        tmp.data[i] = x.data[i] - res.first.data[i] - res.second.data[i];
    }

    party.send_tensor_to(party.next_party_id, res.second);
    party.send_tensor_to(party.next_party_id, tmp);

    party.send_tensor_to(party.pre_party_id, tmp);
    party.send_tensor_to(party.pre_party_id, res.first);
}

template <typename T>
void rss_protocols::recv_shares_from(int source_id, RSSTensor<T> &res)
{
    Party3PC &party = Party3PC::getInstance();
    party.recv_tensor_from(source_id, res.first);
    party.recv_tensor_from(source_id, res.second);
}

template <typename T>
void rss_protocols::coin(std::vector<uint32_t> shape, RSSTensor<T> &res)
{
}

template <typename T>
void rss_protocols::reshare(Tensor<T> &x, RSSTensor<T> &res)
{
    Party3PC &party = Party3PC::getInstance();
    res.assert_same_shape(x);
    party.send_tensor_to(party.pre_party_id, x);
    party.recv_tensor_from(party.next_party_id, res.second);
    res.first.copy(x);
}

template <typename T>
RSSTensor<T> rss_protocols::reshare(Tensor<T> &x)
{
    Party3PC &party = Party3PC::getInstance();
    Tensor<T> tmp(x.shape);
    party.send_tensor_to(party.pre_party_id, x);
    party.recv_tensor_from(party.next_party_id, tmp);
    return RSSTensor<T>(x, tmp);
}

template <typename T>
void rss_protocols::add(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] + y.first.data[i];
        res.second.data[i] = x.second.data[i] + y.second.data[i];
    }
}

template <typename T>
void rss_protocols::add(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);
    res.assert_same_shape(y);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] + y.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i] + y.data[i];
        }
    }
}

template <typename T>
void rss_protocols::add(RSSTensor<T> &x, T y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] + y;
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i] + y;
        }
    }
}

template <typename T>
void rss_protocols::sub(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] - y.first.data[i];
        res.second.data[i] = x.second.data[i] - y.second.data[i];
    }
}

template <typename T>
void rss_protocols::sub(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);
    res.assert_same_shape(y);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] - y.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i] - y.data[i];
        }
    }
}

template <typename T>
void rss_protocols::sub(RSSTensor<T> &x, T y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] - y;
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i] - y;
        }
    }
}

template <typename T>
void rss_protocols::sub(T x, RSSTensor<T> &y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(y);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < y.size(); i++)
        {
            res.first.data[i] = x - y.first.data[i];
            res.second.data[i] = -y.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < y.size(); i++)
        {
            res.first.data[i] = -y.first.data[i];
            res.second.data[i] = -y.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < y.size(); i++)
        {
            res.first.data[i] = -y.first.data[i];
            res.second.data[i] = x - y.second.data[i];
        }
    }
}

template <typename T>
void rss_protocols::mulConst(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y.data[i];
        res.second.data[i] = x.second.data[i] * y.data[i];
    }
}

template <typename T>
void rss_protocols::mulConst(RSSTensor<T> &x, T y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y;
        res.second.data[i] = x.second.data[i] * y;
    }
}

template <typename T>
void rss_protocols::mulConstAddBias(RSSTensor<T> &x, T y, T bias, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y + bias;
            res.second.data[i] = x.second.data[i] * y;
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y;
            res.second.data[i] = x.second.data[i] * y;
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y;
            res.second.data[i] = x.second.data[i] * y + bias;
        }
    }
}

template <typename T>
void rss_protocols::mulConstAddBias(RSSTensor<T> &x, T y, RSSTensor<T> &bias, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y + bias.first.data[i];
        res.second.data[i] = x.second.data[i] * y + bias.second.data[i];
    }
}

template <typename T>
void rss_protocols::mulConstAddBias(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &bias, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y.data[i] + bias.first.data[i];
        res.second.data[i] = x.second.data[i] * y.data[i] + bias.second.data[i];
    }
}

template <typename T>
void rss_protocols::mulConstSubBias(RSSTensor<T> &x, T y, T bias, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y - bias;
            res.second.data[i] = x.second.data[i] * y;
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y;
            res.second.data[i] = x.second.data[i] * y;
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y;
            res.second.data[i] = x.second.data[i] * y - bias;
        }
    }
}

template <typename T>
void rss_protocols::mulConstSubBias(RSSTensor<T> &x, T y, RSSTensor<T> &bias, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y - bias.first.data[i];
        res.second.data[i] = x.second.data[i] * y - bias.second.data[i];
    }
}

template <typename T>
void rss_protocols::mulConstSubBias(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &bias, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y.data[i] - bias.first.data[i];
        res.second.data[i] = x.second.data[i] * y.data[i] - bias.second.data[i];
    }
}

template <typename T>
void rss_protocols::mul(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, bool needTrunc, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    res.assert_same_shape(x);
    res.assert_same_shape(y);
    uint32_t size = x.size();

    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        tmp.data[i] = x.first.data[i] * y.first.data[i] + x.first.data[i] * y.second.data[i] +
                      x.second.data[i] * y.first.data[i];
    }

    reshare(tmp, res);
    if (malicious)
    {
        RSSTensor<T> a(x.shape), b(y.shape), c(res.shape);
        a.zeros();
        b.zeros();
        c.zeros();

        uint32_t combined_size = 2 * size;
        RSSTensor<T> combined({combined_size});

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            combined.first.data[i] = a.first.data[i] - x.first.data[i];
            combined.second.data[i] = a.second.data[i] - x.second.data[i];
            combined.first.data[i + size] = b.first.data[i] - y.first.data[i];
            combined.second.data[i + size] = b.second.data[i] - y.second.data[i];
        }
        Tensor<T> rhoSigma({combined_size}), rho(x.shape), sigma(y.shape);
        restore(combined, rhoSigma, malicious);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            rho.data[i] = rhoSigma.data[i];
            sigma.data[i] = rhoSigma.data[i + size];
        }

        RSSTensor<T> zero_res(x.shape);

        if (party.party_id == 0)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * sigma.data[i] +
                                         rho.data[i] * b.first.data[i] - rho.data[i] * sigma.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * sigma.data[i] +
                                          rho.data[i] * b.second.data[i];
            }
        }
        else if (party.party_id == 1)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * sigma.data[i] +
                                         rho.data[i] * b.first.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * sigma.data[i] +
                                          rho.data[i] * b.second.data[i];
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * sigma.data[i] +
                                         rho.data[i] * b.first.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * sigma.data[i] +
                                          rho.data[i] * b.second.data[i] - rho.data[i] * sigma.data[i];
            }
        }

        party.send_tensor_to(party.next_party_id, zero_res.first);
        Tensor<T> tmp_recv(x.shape);
        party.recv_tensor_from(party.pre_party_id, tmp_recv);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            always_assert(zero_res.first.data[i] + zero_res.second.data[i] + tmp_recv.data[i] == 0);
        }
    }
    if (needTrunc)
    {
        truncate(res, malicious);
    }
}

template <typename T>
void rss_protocols::mul(RSSTensor<T> &x, std::pair<T, T> &y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    uint32_t size = x.size();

    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        tmp.data[i] = x.first.data[i] * y.first + x.first.data[i] * y.second +
                      x.second.data[i] * y.first;
    }

    reshare(tmp, res);
}

template <typename T>
void rss_protocols::square(RSSTensor<T> &x, RSSTensor<T> &res, bool needTrunc, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    res.assert_same_shape(x);
    uint32_t size = x.size();

    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        tmp.data[i] = x.first.data[i] * x.first.data[i] + x.first.data[i] * x.second.data[i] +
                      x.second.data[i] * x.first.data[i];
    }

    reshare(tmp, res);
    if (malicious)
    {
        RSSTensor<T> a(x.shape), c(res.shape); // c = a * a
        a.zeros();
        c.zeros();

        RSSTensor<T> rho_share({size});

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            rho_share.first.data[i] = a.first.data[i] - x.first.data[i];
            rho_share.second.data[i] = a.second.data[i] - x.second.data[i];
        }
        Tensor<T> rho({size});
        restore(rho_share, rho, malicious);

        RSSTensor<T> zero_res({size});

        if (party.party_id == 0)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * rho.data[i] * 2 - rho.data[i] * rho.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * rho.data[i] * 2;
            }
        }
        else if (party.party_id == 1)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * rho.data[i] * 2;
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * rho.data[i] * 2;
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * rho.data[i] * 2;
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * rho.data[i] * 2 -
                                          rho.data[i] * rho.data[i];
            }
        }

        party.send_tensor_to(party.next_party_id, zero_res.first);
        Tensor<T> tmp_recv({size});
        party.recv_tensor_from(party.pre_party_id, tmp_recv);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            always_assert(zero_res.first.data[i] + zero_res.second.data[i] + tmp_recv.data[i] == 0);
        }
    }
    if (needTrunc)
    {
        truncate(res, malicious);
    }
}

template <typename T>
void rss_protocols::matMul(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, bool needTrunc, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    int threads_num = omp_get_max_threads();
    omp_set_num_threads(64);

    uint32_t size = res.size(), x_size = x.size(), y_size = y.size();
    Tensor<T> tmp(res.shape);

    const uint32_t x_shape_size = x.shape.size();
    const uint32_t y_shape_size = y.shape.size();
    const uint32_t z_shape_size = res.shape.size();
    always_assert(x_shape_size >= 2 && y_shape_size >= 2 && z_shape_size >= 2);
    always_assert(x.shape[x_shape_size - 1] == y.shape[y_shape_size - 2]);
    always_assert(res.shape[z_shape_size - 2] == x.shape[x_shape_size - 2]);
    always_assert(res.shape[z_shape_size - 1] == y.shape[y_shape_size - 1]);

    uint32_t row, common_dim, col, broadcast_size, common_size;
    bool is_b_broadcast;

    compute_matmul_info(x.shape, y.shape, res.shape, row, common_dim, col, broadcast_size, common_size,
                        is_b_broadcast);

    utils::RSSMatMul(x, y, tmp, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);
    reshare(tmp, res);

    if (malicious)
    {
        RSSTensor<T> a(x.shape), b(y.shape), c(res.shape);
        a.zeros();
        b.zeros();
        c.zeros();

        uint32_t combined_size = x_size + y_size;
        RSSTensor<T> combined({combined_size});

#pragma omp parallel for
        for (int i = 0; i < x_size; i++)
        {
            combined.first.data[i] = a.first.data[i] - x.first.data[i];
            combined.second.data[i] = a.second.data[i] - x.second.data[i];
        }
#pragma omp parallel for
        for (int i = 0; i < y_size; i++)
        {
            combined.first.data[i + x_size] = b.first.data[i] - y.first.data[i];
            combined.second.data[i + x_size] = b.second.data[i] - y.second.data[i];
        }

        Tensor<T> rhoSigma({combined_size}), rho(x.shape), sigma(y.shape);
        restore(combined, rhoSigma, malicious);

#pragma omp parallel for
        for (int i = 0; i < x_size; i++)
        {
            rho.data[i] = rhoSigma.data[i];
        }
        for (int i = 0; i < y_size; i++)
        {
            sigma.data[i] = rhoSigma.data[i + x_size];
        }

        RSSTensor<T> zero_res(res.shape), af(res.shape), eb(res.shape);
        Tensor<T> ef(res.shape);

        utils::RSSMatMul(a, sigma, af, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);
        utils::RSSMatMul(rho, b, eb, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);
        Tensor<T>::matmul(ef, rho, sigma, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);

        if (party.party_id == 0)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] =
                    res.first.data[i] - c.first.data[i] + af.first.data[i] + eb.first.data[i] - ef.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + af.second.data[i] + eb.second.data[i];
            }
        }
        else if (party.party_id == 1)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + af.first.data[i] + eb.first.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + af.second.data[i] + eb.second.data[i];
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + af.first.data[i] + eb.first.data[i];
                zero_res.second.data[i] =
                    res.second.data[i] - c.second.data[i] + af.second.data[i] + eb.second.data[i] - ef.data[i];
            }
        }

        party.send_tensor_to(party.next_party_id, zero_res.first);
        Tensor<T> tmp_recv(res.shape);
        party.recv_tensor_from(party.pre_party_id, tmp_recv);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            always_assert(zero_res.first.data[i] + zero_res.second.data[i] + tmp_recv.data[i] == 0);
        }
    }
    omp_set_num_threads(threads_num);
    if (needTrunc)
    {
        truncate(res, malicious);
    }
}

template <typename T>
void rss_protocols::truncate(RSSTensor<T> &x, RSSTensor<T> &res, size_t scale, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    RSSTensor<T> r(x.shape), r_t(x.shape);
    uint32_t size = x.size();
    r.zeros();
    r_t.zeros();

    sub(x, r, x);
    Tensor<T> x_shift(x.shape);
    restore(x, x_shift, malicious);

    if (party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                res.first.data[i] =
                    r_t.first.data[i] + (T)((double)((int32_t)x_shift.data[i]) / (double)((int32_t)scale));
            }
            else
            {
                res.first.data[i] =
                    r_t.first.data[i] + (T)((double)((int64_t)x_shift.data[i]) / (double)((int64_t)scale));
            }

            res.second.data[i] = r_t.second.data[i];
        }
    }
    else if (party.party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res.first.data[i] = r_t.first.data[i];
            res.second.data[i] = r_t.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res.first.data[i] = r_t.first.data[i];
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                res.second.data[i] =
                    r_t.second.data[i] + (T)((double)((int32_t)x_shift.data[i]) / (double)((int32_t)scale));
            }
            else
            {
                res.second.data[i] =
                    r_t.second.data[i] + (T)((double)((int64_t)x_shift.data[i]) / (double)((int64_t)scale));
            }
        }
    }
}

template <typename T>
void rss_protocols::truncate(RSSTensor<T> &x, size_t scale, bool malicious)
{
    truncate(x, x, scale, malicious);
}

template <typename T>
void rss_protocols::truncate(RSSTensor<T> &x, bool malicious)
{
    truncate(x, 1 << x.float_scale_bit, malicious);
}

template <typename T>
void rss_protocols::checkZero(RSSTensor<T> &x)
{
    RSSTensor<T> r(x.shape), xr(x.shape);
    Tensor<T> xr_open(x.shape);

    r.zeros(); // it should be random
    mul(x, r, xr);
    restore(xr, xr_open, true);

    xr_open.assert_zero();
}

template <typename T>
void rss_protocols::macCheck(const RSSTensor<T> &x, const RSSTensor<T> &mx, const std::pair<T, T> &mac_key)
{
#if (IS_MALICIOUS)
    Party3PC &party = Party3PC::getInstance();
    RSSTensor<T> r({1}), mr({1});
    r.zeros(); // it should be random
    mul(r, mac_key, mr);
    RSSTensor<T> ro_share(x.shape);
    ro_share.zeros(); // it should be random
    Tensor<T> ro(x.shape);
    restore(ro_share, ro, true);
    RSSTensor<T> v({1}), w({1});
    v.first.data[0] = r.first.data[0];
    v.second.data[0] = r.second.data[0];

    w.first.data[0] = mr.first.data[0];
    w.second.data[0] = mr.second.data[0];

    for (int i = 0; i < x.size(); i++)
    {
        v.first.data[i] += x.first.data[i] * ro.data[i];
        v.second.data[i] += x.second.data[i] * ro.data[i];
        w.first.data[i] += mx.first.data[i] * ro.data[i];
        w.second.data[i] += mx.second.data[i] * ro.data[i];
    }
    Tensor<T> v_real({1});
    restore(v, v_real, true);
    RSSTensor<T> delta({1});
    mulConstSubBias(mac_key, v_real, w, delta);
    checkZero(delta);
#endif
}

template <typename T>
void rss_protocols::macCheck(RSSTensor<T> &x, RSSTensor<T> &mx, RSSTensor<T> &mac_key)
{
#if (IS_MALICIOUS)
    RSSTensor<T> r(x.shape), mr(mx.shape);
    r.zeros(); // it should be random
    mul(r, mac_key, mr);
    RSSTensor<T> ro_share(x.shape);
    ro_share.zeros(); // it should be random
    Tensor<T> ro(x.shape);
    restore(ro_share, ro, true);
    RSSTensor<T> v(x.shape), w(x.shape);
    mulConstAddBias(x, ro, r, v);
    mulConstAddBias(mx, ro, mr, w);
    Tensor<T> v_real(x.shape);
    restore(v, v_real, true);
    RSSTensor<T> delta(x.shape);
    mulConstSubBias(mac_key, v_real, w, delta);
    checkZero(delta);
#endif
}

template <typename T>
void rss_protocols::macCheckSimulate(uint32_t size)
{
#if (IS_MALICIOUS)
    if (size == 0)
    {
        return;
    }
    RSSTensor<T> x({size}), mx({size}), mac_key({size});
    x.zeros();
    mx.zeros();
    mac_key.zeros();

    macCheck(x, mx, mac_key);
    MAC_SIZE = 0;
#endif
}

template <typename T>
void rss_protocols::utils::privateCompare(Tensor<T> &x_with_pre, Tensor<T> &x_with_next, Tensor<T> &res_with_pre,
                                          Tensor<T> &res_with_next, Parameters<T> &parameter)
{
    Party3PC &party = Party3PC::getInstance();
    // r_bit_with_pre and r_bit_with_next are provided by parameter
    uint32_t size = x_with_pre.size();
    uint32_t bit_length = 8 * sizeof(T);
    uint32_t double_bit_length = 2 * bit_length;
    uint32_t long_size = size * (bit_length - 1); // The actual input is \ell - 1 bit and only need to calculate \ell - 1 bit

    Tensor<uint8_t> c_with_pre({long_size}), c_with_next({long_size});
    T w_with_pre, w_with_next, x_bit_with_pre, x_bit_with_next, w_sum_with_pre, w_sum_with_next;
    T r_with_pre_bit, r_with_next_bit;

    int index;
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        w_sum_with_pre = 0;
        w_sum_with_next = 0;

        for (int j = 1; j < bit_length; j++)
        {
            x_bit_with_pre = (x_with_pre.data[i] >> (bit_length - 1 - j)) & 1;
            x_bit_with_next = (x_with_next.data[i] >> (bit_length - 1 - j)) & 1;

            r_with_pre_bit = parameter.pc_cmp.r_with_pre_bits.data[j];
            r_with_next_bit = parameter.pc_cmp.r_with_next_bits.data[j];

            if (x_bit_with_pre == 0)
            {
                w_with_pre = r_with_pre_bit;
            }
            else
            {
                w_with_pre = -r_with_pre_bit;
            }

            if (x_bit_with_next == 0)
            {
                w_with_next = r_with_next_bit;
            }
            else
            {
                w_with_next = 1 - r_with_next_bit;
            }

            index = i * (bit_length - 1) + j - 1;
            c_with_pre.data[index] = (uint8_t)(r_with_pre_bit + w_sum_with_pre + parameter.pc_cmp.round1_r);
            c_with_next.data[index] = (uint8_t)(r_with_next_bit - x_bit_with_next + w_sum_with_next + 1 + parameter.pc_cmp.round1_r);

            w_sum_with_pre += w_with_pre;
            w_sum_with_next += w_with_next;
        }
    }
    Tensor<uint8_t> c_with_pre_tmp({long_size}), c_with_next_tmp({long_size});

    if (party.party_id == 2)
    {
        party.send_tensor_to(party.next_party_id, c_with_next);
        party.recv_tensor_from(party.next_party_id, c_with_next_tmp);

        party.send_tensor_to(party.pre_party_id, c_with_pre);
        party.recv_tensor_from(party.pre_party_id, c_with_pre_tmp);
    }
    else
    {
        party.send_tensor_to(party.pre_party_id, c_with_pre);
        party.recv_tensor_from(party.pre_party_id, c_with_pre_tmp);

        party.send_tensor_to(party.next_party_id, c_with_next);
        party.recv_tensor_from(party.next_party_id, c_with_next_tmp);
    }

#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        res_with_pre.data[i] = 0;
        res_with_next.data[i] = 0;
        for (int j = 0; j < bit_length - 1; j++)
        {
            index = i * (bit_length - 1) + j;
            res_with_pre.data[i] += parameter.pc_cmp.round1_real_table_with_pre.data[(c_with_pre.data[index] + c_with_pre_tmp.data[index]) % double_bit_length];
            res_with_next.data[i] += parameter.pc_cmp.round1_real_table_with_next.data[(c_with_next.data[index] + c_with_next_tmp.data[index]) % double_bit_length];
        }
    }
}

template <typename T>
void rss_protocols::pc_msb(RSSTensor<T> &x, Tensor<T> &res_with_pre, Tensor<T> &res_with_next,
                           Parameters<T> &parameter, const uint32_t size, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    RSSTensor<T> delta0(x.shape), delta1(x.shape), delta2(x.shape);
    Tensor<T> dt_with_pre(x.shape), dt_with_next(x.shape);

    if (party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            delta0.first.data[i] = parameter.pc_cmp.self_r1 + x.first.data[i];
            delta0.second.data[i] = parameter.pc_cmp.self_r0 + x.second.data[i];

            delta1.first.data[i] = x.first.data[i];
            delta1.second.data[i] = parameter.pc_cmp.r_with_pre + x.second.data[i];

            delta2.first.data[i] = parameter.pc_cmp.r_with_next + x.first.data[i];
            delta2.second.data[i] = x.second.data[i];
        }
    }
    else if (party.party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            delta0.first.data[i] = parameter.pc_cmp.r_with_next + x.first.data[i];
            delta0.second.data[i] = x.second.data[i];

            delta1.first.data[i] = parameter.pc_cmp.self_r1 + x.first.data[i];
            delta1.second.data[i] = parameter.pc_cmp.self_r0 + x.second.data[i];

            delta2.first.data[i] = x.first.data[i];
            delta2.second.data[i] = parameter.pc_cmp.r_with_pre + x.second.data[i];
        }
    }
    else if (party.party_id == 2)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            delta0.first.data[i] = x.first.data[i];
            delta0.second.data[i] = parameter.pc_cmp.r_with_pre + x.second.data[i];

            delta1.first.data[i] = parameter.pc_cmp.r_with_next + x.first.data[i];
            delta1.second.data[i] = x.second.data[i];

            delta2.first.data[i] = parameter.pc_cmp.self_r1 + x.first.data[i];
            delta2.second.data[i] = parameter.pc_cmp.self_r0 + x.second.data[i];
        }
    }

    reconstruct_to(0, delta1, dt_with_pre, malicious);
    reconstruct_to(0, delta2, dt_with_next, malicious);

    reconstruct_to(1, delta2, dt_with_pre, malicious);
    reconstruct_to(1, delta0, dt_with_next, malicious);

    reconstruct_to(2, delta0, dt_with_pre, malicious);
    reconstruct_to(2, delta1, dt_with_next, malicious);

    int bit_len_sub_1 = sizeof(T) * 8 - 1;
    utils::privateCompare(dt_with_pre, dt_with_next, res_with_pre, res_with_next, parameter);

    for (int i = 0; i < size; i++)
    {
        res_with_pre.data[i] = (res_with_pre.data[i] + parameter.pc_cmp.r_with_pre_bits.data[0]) % 2;
        res_with_next.data[i] = (res_with_next.data[i] + parameter.pc_cmp.r_with_next_bits.data[0] + ((dt_with_next.data[i] >> bit_len_sub_1) & 1)) % 2;
    }

    if (party.party_id == 2)
    {
        ass_protocols::b2a(res_with_next, *party.peer_with_next, 1);
        ass_protocols::b2a(res_with_pre, *party.peer_with_pre, 0);
    }
    else
    {
        ass_protocols::b2a(res_with_pre, *party.peer_with_pre, 0);
        ass_protocols::b2a(res_with_next, *party.peer_with_next, 1);
    }
}

template <typename T>
void rss_protocols::nonNegative(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    uint32_t size = x.size();
    Tensor<T> res_with_pre(x.shape), res_with_next(x.shape);
    Tensor<T> mac_res_with_pre(x.shape), mac_res_with_next(x.shape);

    pc_msb(x, res_with_pre, res_with_next, parameter, size, malicious);

    if (malicious)
    {
        std::pair<T, T> one_share, mac_one_share;
        mac_one_share = std::make_pair(0, 0);
        if (party.party_id == 0)
        {
            one_share = std::make_pair(1, 0);
        }
        else if (party.party_id == 1)
        {
            one_share = std::make_pair(0, 0);
        }
        else
        {
            one_share = std::make_pair(0, 1);
        }

        Tensor<T> res_33(res.shape);
        Tensor<T> mac_res_33(res.shape);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res_33.data[i] = one_share.first * (0 - res_with_pre.data[i]) + one_share.second * (1 - res_with_next.data[i]);
            mac_res_33.data[i] = mac_one_share.first * (0 - res_with_pre.data[i]) + mac_one_share.second * (1 - res_with_next.data[i]);
        }

        reshare(res_33, res);

        RSSTensor<T> mac_res(res.shape);
        reshare(mac_res_33, mac_res);

        // mac check
#if (LATER_CHECK)
        // party.mac_buffer.add_value(res, mac_res, mac_key);
        MAC_SIZE += res.size();
#else
        std::pair<T, T> mac_key = std::make_pair(0, 0);
        macCheck(res, mac_res, mac_key);
#endif
    }
    else
    {
        std::pair<T, T> one_share;
        if (party.party_id == 0)
        {
            one_share = std::make_pair(1, 0);
        }
        else if (party.party_id == 1)
        {
            one_share = std::make_pair(0, 0);
        }
        else
        {
            one_share = std::make_pair(0, 1);
        }

        Tensor<T> res_33(res.shape);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res_33.data[i] = one_share.first * (0 - res_with_pre.data[i]) + one_share.second * (1 - res_with_next.data[i]);
        }

        reshare(res_33, res);
    }

    if (isFloat)
    {
        mulConst(res, (T)(1 << x.float_scale_bit), res);
    }
}

template <typename T>
void rss_protocols::greaterEqual(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat, bool malicious)
{
    RSSTensor<T> z(x.shape);
    sub(x, y, z);
    nonNegative(z, res, parameter, isFloat, malicious);
}

template <typename T>
void rss_protocols::select(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    uint32_t size = x.size();
    Tensor<T> res_with_pre(x.shape), res_with_next(x.shape);

    pc_msb(x, res_with_pre, res_with_next, parameter, size, malicious);

    if (malicious)
    {
        std::pair<T, T> mac_key;
        mac_key = std::make_pair(0, 0);
        RSSTensor<T> mac_y(y.shape);
        mul(y, mac_key, mac_y);

        Tensor<T> res_33(res.shape);
        Tensor<T> mac_res_33(res.shape);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res_33.data[i] = y.first.data[i] * (0 - res_with_pre.data[i]) + y.second.data[i] * (1 - res_with_next.data[i]);
            mac_res_33.data[i] = mac_y.first.data[i] * (0 - res_with_pre.data[i]) + mac_y.second.data[i] * (1 - res_with_next.data[i]);
        }

        reshare(res_33, res);

        RSSTensor<T> mac_res(res.shape);
        reshare(mac_res_33, mac_res);

        // mac check
#if (LATER_CHECK)
        // party.mac_buffer.add_value(res, mac_res, mac_key);
        MAC_SIZE += res.size();
#else
        macCheck(res, mac_res, mac_key);
#endif
    }
    else
    {
        Tensor<T> res_33(res.shape);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res_33.data[i] = y.first.data[i] * (0 - res_with_pre.data[i]) + y.second.data[i] * (1 - res_with_next.data[i]);
        }

        reshare(res_33, res);
    }
}

template <typename T>
void rss_protocols::select(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, uint32_t y_num, Parameters<T> &parameter, bool malicious)
{
    uint32_t size = x.size();
    Tensor<T> res_with_pre(x.shape), res_with_next(x.shape);

    pc_msb(x, res_with_pre, res_with_next, parameter, size, malicious);

    if (malicious)
    {
        std::pair<T, T> mac_key;
        mac_key = std::make_pair(0, 0);
        RSSTensor<T> mac_y(y.shape);

        Tensor<T> res_33(res.shape);
        Tensor<T> mac_res_33(res.shape);

        int index;
        mul(y, mac_key, mac_y);
        for (int i = 0; i < y_num; i++)
        {
#pragma omp parallel for
            for (int j = 0; j < size; j++)
            {
                index = i * size + j;
                res_33.data[index] = y.first.data[index] * (0 - res_with_pre.data[j]) + y.second.data[index] * (1 - res_with_next.data[j]);
                mac_res_33.data[index] = mac_y.first.data[index] * (0 - res_with_pre.data[j]) + mac_y.second.data[index] * (1 - res_with_next.data[j]);
            }
        }

        reshare(res_33, res);

        RSSTensor<T> mac_res(res.shape);
        reshare(mac_res_33, mac_res);

        // mac check
#if (LATER_CHECK)
        // party.mac_buffer.add_value(res, mac_res, mac_key);
        MAC_SIZE += res.size();
#else
        macCheck(res, mac_res, mac_key);
#endif
    }
    else
    {
        Tensor<T> res_33(res.shape);
        int index;
        for (int i = 0; i < y_num; i++)
        {
#pragma omp parallel for
            for (int j = 0; j < size; j++)
            {
                index = i * size + j;
                res_33.data[index] = y.first.data[index] * (0 - res_with_pre.data[j]) + y.second.data[index] * (1 - res_with_next.data[j]);
            }
        }

        reshare(res_33, res);
    }
}

template <typename T>
void rss_protocols::lut(RSSTensor<T> &x, RSSTensor<T> &res, LUT_Param<T> &parameter, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    uint32_t size = x.size(), table_size = parameter.table_size;
    RSSTensor<T> delta0(x.shape), delta1(x.shape), delta2(x.shape);
    Tensor<T> dt_with_pre(x.shape), dt_with_next(x.shape);

    if (party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            delta0.first.data[i] = parameter.self_r1 - x.first.data[i];
            delta0.second.data[i] = parameter.self_r0 - x.second.data[i];

            delta1.first.data[i] = 0 - x.first.data[i];
            delta1.second.data[i] = parameter.r_with_pre - x.second.data[i];

            delta2.first.data[i] = parameter.r_with_next - x.first.data[i];
            delta2.second.data[i] = 0 - x.second.data[i];
        }
    }
    else if (party.party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            delta0.first.data[i] = parameter.r_with_next - x.first.data[i];
            delta0.second.data[i] = 0 - x.second.data[i];

            delta1.first.data[i] = parameter.self_r1 - x.first.data[i];
            delta1.second.data[i] = parameter.self_r0 - x.second.data[i];

            delta2.first.data[i] = 0 - x.first.data[i];
            delta2.second.data[i] = parameter.r_with_pre - x.second.data[i];
        }
    }
    else if (party.party_id == 2)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            delta0.first.data[i] = 0 - x.first.data[i];
            delta0.second.data[i] = parameter.r_with_pre - x.second.data[i];

            delta1.first.data[i] = parameter.r_with_next - x.first.data[i];
            delta1.second.data[i] = 0 - x.second.data[i];

            delta2.first.data[i] = parameter.self_r1 - x.first.data[i];
            delta2.second.data[i] = parameter.self_r0 - x.second.data[i];
        }
    }

    reconstruct_to(0, delta1, dt_with_pre, malicious);
    reconstruct_to(0, delta2, dt_with_next, malicious);

    reconstruct_to(1, delta2, dt_with_pre, malicious);
    reconstruct_to(1, delta0, dt_with_next, malicious);

    reconstruct_to(2, delta0, dt_with_pre, malicious);
    reconstruct_to(2, delta1, dt_with_next, malicious);

    if (malicious)
    {
        std::pair<T, T> one_share, mac_one_share;
        mac_one_share = std::make_pair(0, 0);
        if (party.party_id == 0)
        {
            one_share = std::make_pair(1, 0);
        }
        else if (party.party_id == 1)
        {
            one_share = std::make_pair(0, 0);
        }
        else
        {
            one_share = std::make_pair(0, 1);
        }

        Tensor<T> res_33(res.shape);
        Tensor<T> mac_res_33(res.shape);

        uint32_t idx_pre;
        uint32_t idx_next;

        T res_with_pre, res_with_next;

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res_with_pre = 0;
            res_with_next = 0;
            for (int j = 0; j < table_size; j++)
            {
                idx_pre = (j + dt_with_pre.data[i] + table_size) % table_size;
                idx_next = (j + dt_with_next.data[i] + table_size) % table_size;

                res_with_pre += parameter.onehot_table_with_pre.data[idx_pre] * parameter.table.data[j];
                res_with_next += parameter.onehot_table_with_next.data[idx_pre] * parameter.table.data[j];
            }
            res_33.data[i] = one_share.first * res_with_pre + one_share.second * res_with_next;
            mac_res_33.data[i] = mac_one_share.first * res_with_pre + mac_one_share.second * res_with_next;
        }

        reshare(res_33, res);

        RSSTensor<T> mac_res(res.shape);
        reshare(mac_res_33, mac_res);

        // mac check
#if (LATER_CHECK)
        // party.mac_buffer.add_value(res, mac_res, mac_key);
        MAC_SIZE += res.size();
#else
        std::pair<T, T> mac_key = std::make_pair(0, 0);
        macCheck(res, mac_res, mac_key);
#endif
    }
    else
    {
        std::pair<T, T> one_share;
        if (party.party_id == 0)
        {
            one_share = std::make_pair(1, 0);
        }
        else if (party.party_id == 1)
        {
            one_share = std::make_pair(0, 0);
        }
        else
        {
            one_share = std::make_pair(0, 1);
        }

        Tensor<T> res_33(res.shape);

        uint32_t idx_pre;
        uint32_t idx_next;

        T res_with_pre, res_with_next;

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res_with_pre = 0;
            res_with_next = 0;
            for (int j = 0; j < table_size; j++)
            {
                idx_pre = (j + dt_with_pre.data[i] + table_size) % table_size;
                idx_next = (j + dt_with_next.data[i] + table_size) % table_size;

                res_with_pre += parameter.onehot_table_with_pre.data[idx_pre] * parameter.table.data[j];
                res_with_next += parameter.onehot_table_with_next.data[idx_pre] * parameter.table.data[j];
            }
            res_33.data[i] = one_share.first * res_with_pre + one_share.second * res_with_next;
        }

        reshare(res_33, res);
    }
}

template <typename T>
void rss_protocols::lut(RSSTensor<T> &x, RSSTensor<T> &res1, RSSTensor<T> &res2, LUT_Param<T> &parameter1, LUT_Param<T> &parameter2, bool malicious)
{
    // TODO: extend to multi-table
    Party3PC &party = Party3PC::getInstance();
    uint32_t size = x.size(), double_size = 2 * size, table1_size = parameter1.table_size, table2_size = parameter2.table_size;
    RSSTensor<T> delta0({double_size}), delta1({double_size}), delta2({double_size});
    Tensor<T> dt_with_pre({double_size}), dt_with_next({double_size});

    if (party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            // 1
            delta0.first.data[i] = parameter1.self_r1 - x.first.data[i];
            delta0.second.data[i] = parameter1.self_r0 - x.second.data[i];

            delta1.first.data[i] = 0 - x.first.data[i];
            delta1.second.data[i] = parameter1.r_with_pre - x.second.data[i];

            delta2.first.data[i] = parameter1.r_with_next - x.first.data[i];
            delta2.second.data[i] = 0 - x.second.data[i];

            // 2
            delta0.first.data[i + size] = parameter2.self_r1 - x.first.data[i];
            delta0.second.data[i + size] = parameter2.self_r0 - x.second.data[i];

            delta1.first.data[i + size] = 0 - x.first.data[i];
            delta1.second.data[i + size] = parameter2.r_with_pre - x.second.data[i];

            delta2.first.data[i + size] = parameter2.r_with_next - x.first.data[i];
            delta2.second.data[i + size] = 0 - x.second.data[i];
        }
    }
    else if (party.party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            // 1
            delta0.first.data[i] = parameter1.r_with_next - x.first.data[i];
            delta0.second.data[i] = 0 - x.second.data[i];

            delta1.first.data[i] = parameter1.self_r1 - x.first.data[i];
            delta1.second.data[i] = parameter1.self_r0 - x.second.data[i];

            delta2.first.data[i] = 0 - x.first.data[i];
            delta2.second.data[i] = parameter1.r_with_pre - x.second.data[i];

            // 2
            delta0.first.data[i + size] = parameter2.r_with_next - x.first.data[i];
            delta0.second.data[i + size] = 0 - x.second.data[i];

            delta1.first.data[i + size] = parameter2.self_r1 - x.first.data[i];
            delta1.second.data[i + size] = parameter2.self_r0 - x.second.data[i];

            delta2.first.data[i + size] = 0 - x.first.data[i];
            delta2.second.data[i + size] = parameter2.r_with_pre - x.second.data[i];
        }
    }
    else if (party.party_id == 2)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            // 1
            delta0.first.data[i] = 0 - x.first.data[i];
            delta0.second.data[i] = parameter1.r_with_pre - x.second.data[i];

            delta1.first.data[i] = parameter1.r_with_next - x.first.data[i];
            delta1.second.data[i] = 0 - x.second.data[i];

            delta2.first.data[i] = parameter1.self_r1 - x.first.data[i];
            delta2.second.data[i] = parameter1.self_r0 - x.second.data[i];

            // 2
            delta0.first.data[i + size] = 0 - x.first.data[i];
            delta0.second.data[i + size] = parameter2.r_with_pre - x.second.data[i];

            delta1.first.data[i + size] = parameter2.r_with_next - x.first.data[i];
            delta1.second.data[i + size] = 0 - x.second.data[i];

            delta2.first.data[i + size] = parameter2.self_r1 - x.first.data[i];
            delta2.second.data[i + size] = parameter2.self_r0 - x.second.data[i];
        }
    }

    reconstruct_to(0, delta1, dt_with_pre, malicious);
    reconstruct_to(0, delta2, dt_with_next, malicious);

    reconstruct_to(1, delta2, dt_with_pre, malicious);
    reconstruct_to(1, delta0, dt_with_next, malicious);

    reconstruct_to(2, delta0, dt_with_pre, malicious);
    reconstruct_to(2, delta1, dt_with_next, malicious);

    if (malicious)
    {
        std::pair<T, T> one_share, mac_one_share;
        mac_one_share = std::make_pair(0, 0);
        if (party.party_id == 0)
        {
            one_share = std::make_pair(1, 0);
        }
        else if (party.party_id == 1)
        {
            one_share = std::make_pair(0, 0);
        }
        else
        {
            one_share = std::make_pair(0, 1);
        }

        Tensor<T> res_33({double_size});
        Tensor<T> mac_res_33({double_size});

        uint32_t idx_pre;
        uint32_t idx_next;

        T res_with_pre, res_with_next;

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            // 1
            res_with_pre = 0;
            res_with_next = 0;
            for (int j = 0; j < table1_size; j++)
            {
                idx_pre = (j + dt_with_pre.data[i] + table1_size) % table1_size;
                idx_next = (j + dt_with_next.data[i] + table1_size) % table1_size;

                res_with_pre += parameter1.onehot_table_with_pre.data[idx_pre] * parameter1.table.data[j];
                res_with_next += parameter1.onehot_table_with_next.data[idx_pre] * parameter1.table.data[j];
            }
            res_33.data[i] = one_share.first * res_with_pre + one_share.second * res_with_next;
            mac_res_33.data[i] = mac_one_share.first * res_with_pre + mac_one_share.second * res_with_next;

            // 2
            res_with_pre = 0;
            res_with_next = 0;
            for (int j = 0; j < table2_size; j++)
            {
                idx_pre = (j + dt_with_pre.data[i + size] + table2_size) % table2_size;
                idx_next = (j + dt_with_next.data[i + size] + table2_size) % table2_size;

                res_with_pre += parameter2.onehot_table_with_pre.data[idx_pre] * parameter2.table.data[j];
                res_with_next += parameter2.onehot_table_with_next.data[idx_pre] * parameter2.table.data[j];
            }
            res_33.data[i + size] = one_share.first * res_with_pre + one_share.second * res_with_next;
            mac_res_33.data[i + size] = mac_one_share.first * res_with_pre + mac_one_share.second * res_with_next;
        }

        RSSTensor<T> res({double_size}), mac_res({double_size});
        reshare(res_33, res);
        reshare(mac_res_33, mac_res);

        for (int i = 0; i < size; i++)
        {
            res1.first.data[i] = res.first.data[i];
            res1.second.data[i] = res.second.data[i];

            res2.first.data[i] = res.first.data[i + size];
            res2.second.data[i] = res.second.data[i + size];
        }

        // mac check
#if (LATER_CHECK)
        // party.mac_buffer.add_value(res, mac_res, mac_key);
        MAC_SIZE += res.size();
#else
        std::pair<T, T> mac_key = std::make_pair(0, 0);
        macCheck(res, mac_res, mac_key);
#endif
    }
    else
    {
        std::pair<T, T> one_share;
        if (party.party_id == 0)
        {
            one_share = std::make_pair(1, 0);
        }
        else if (party.party_id == 1)
        {
            one_share = std::make_pair(0, 0);
        }
        else
        {
            one_share = std::make_pair(0, 1);
        }

        Tensor<T> res_33({double_size});

        uint32_t idx_pre;
        uint32_t idx_next;

        T res_with_pre, res_with_next;

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            // 1
            res_with_pre = 0;
            res_with_next = 0;
            for (int j = 0; j < table1_size; j++)
            {
                idx_pre = (j + dt_with_pre.data[i] + table1_size) % table1_size;
                idx_next = (j + dt_with_next.data[i] + table1_size) % table1_size;

                res_with_pre += parameter1.onehot_table_with_pre.data[idx_pre] * parameter1.table.data[j];
                res_with_next += parameter1.onehot_table_with_next.data[idx_pre] * parameter1.table.data[j];
            }
            res_33.data[i] = one_share.first * res_with_pre + one_share.second * res_with_next;

            // 2
            res_with_pre = 0;
            res_with_next = 0;
            for (int j = 0; j < table2_size; j++)
            {
                idx_pre = (j + dt_with_pre.data[i + size] + table2_size) % table2_size;
                idx_next = (j + dt_with_next.data[i + size] + table2_size) % table2_size;

                res_with_pre += parameter2.onehot_table_with_pre.data[idx_pre] * parameter2.table.data[j];
                res_with_next += parameter2.onehot_table_with_next.data[idx_pre] * parameter2.table.data[j];
            }
            res_33.data[i + size] = one_share.first * res_with_pre + one_share.second * res_with_next;
        }

        RSSTensor<T> res({double_size});
        reshare(res_33, res);

        for (int i = 0; i < size; i++)
        {
            res1.first.data[i] = res.first.data[i];
            res1.second.data[i] = res.second.data[i];

            res2.first.data[i] = res.first.data[i + size];
            res2.second.data[i] = res.second.data[i + size];
        }
    }
}

template <typename T>
void rss_protocols::utils::getk(RSSTensor<T> &x, RSSTensor<T> &k, Parameters<T> &parameters, bool malicious)
{
    // $b ^ k \le x < b ^ {k + 1}$ find k and calculate k + 1
    Party3PC &party = Party3PC::getInstance();
    uint32_t size = x.size();
    uint32_t nexpb_size = (int)(log(pow(2, 2 * x.float_scale_bit)) / log(SCALE_BASE));
    RSSTensor<T> delta({size, nexpb_size}); // x - b ^ i for i from 1 to max size

    if (party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < nexpb_size; j++)
            {
                delta.first.data[i * nexpb_size + j] = x.first.data[i] - (T)(pow(SCALE_BASE, j + 1));
                delta.second.data[i * nexpb_size + j] = x.second.data[i];
            }
        }
    }
    else if (party.party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < nexpb_size; j++)
            {
                delta.first.data[i * nexpb_size + j] = x.first.data[i];
                delta.second.data[i * nexpb_size + j] = x.second.data[i];
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < nexpb_size; j++)
            {
                delta.first.data[i * nexpb_size + j] = x.first.data[i];
                delta.second.data[i * nexpb_size + j] = x.second.data[i] - (T)(pow(SCALE_BASE, j + 1));
            }
        }
    }
    nonNegative(delta, delta, parameters, false, malicious);

    // calculate k + 1
    if (party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            k.first.data[i] = 1;
            k.second.data[i] = 0;
            for (int j = 0; j < nexpb_size; j++)
            {
                k.first.data[i] += delta.first.data[i * nexpb_size + j];
                k.second.data[i] += delta.second.data[i * nexpb_size + j];
            }
        }
    }
    else if (party.party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            k.first.data[i] = 0;
            k.second.data[i] = 0;
            for (int j = 0; j < nexpb_size; j++)
            {
                k.first.data[i] += delta.first.data[i * nexpb_size + j];
                k.second.data[i] += delta.second.data[i * nexpb_size + j];
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            k.first.data[i] = 0;
            k.second.data[i] = 1;
            for (int j = 0; j < nexpb_size; j++)
            {
                k.first.data[i] += delta.first.data[i * nexpb_size + j];
                k.second.data[i] += delta.second.data[i * nexpb_size + j];
            }
        }
    }
}

template <typename T>
void rss_protocols::inv(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    RSSTensor<T> k(x.shape);
    utils::getk(x, k, parameters, malicious);

    // calculate b^(-(k+1))
    lut(k, k, parameters.nexpb_param, malicious);

    RSSTensor<T> b(x.shape);
    mul(x, k, b, true, malicious); // b  = x * b^(-(k+1))
    sub(b, (T)((1.0 / SCALE_BASE) * (1 << x.float_scale_bit)), b);
    lut(b, b, parameters.inv_param, malicious);
    mul(k, b, res, true, malicious);
}

template <typename T>
void rss_protocols::rsqrt(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    RSSTensor<T> k(x.shape);
    utils::getk(x, k, parameters, malicious);

    // calculate b^(-(k+1)) and b^(-1/2(k+1))
    RSSTensor<T> sqrt_nexpbk(k.shape), nexpbk(k.shape);
    lut(k, sqrt_nexpbk, nexpbk, parameters.sqrt_nexpb_param, parameters.nexpb_param, malicious);

    RSSTensor<T> b(x.shape);
    mul(x, nexpbk, b, true, malicious); // b  = x * b^(-(k+1))
    sub(b, (T)((1.0 / SCALE_BASE) * (1 << x.float_scale_bit)), b);
    lut(b, b, parameters.rsqrt_param, malicious);
    mul(sqrt_nexpbk, b, res, true, malicious);
}

template <typename T>
void rss_protocols::utils::gelu_same_scale(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    T table_size = parameters.gelu_param.table_size;
    uint32_t size = x.size();
    RSSTensor<T> y(x.shape);

    RSSTensor<T> dx(x.shape); // dx = relu(x)
    rss_protocols::select(x, x, dx, parameters, malicious);

    RSSTensor<T> abs_x(x.shape), sizeSubAbs(x.shape);
    rss_protocols::mulConstSubBias(dx, (T)2, x, abs_x); // calculate abs_x = dx * 2 - x

    rss_protocols::sub(table_size, abs_x, sizeSubAbs);
    RSSTensor<T> ia(x.shape);
    rss_protocols::select(sizeSubAbs, abs_x, ia, parameters, malicious);
    rss_protocols::add(ia, (T)(table_size - 1), ia);
    RSSTensor<T> c(x.shape);
    rss_protocols::lut(ia, c, parameters.gelu_param, malicious);
    rss_protocols::sub(dx, c, res);
}

template <typename T>
void rss_protocols::gelu(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
    if (x.float_scale_bit == GELU_TABLE_PRECISION)
    {
        utils::gelu_same_scale(x, res, parameters, malicious);
    }
    else
    {
        T table_size = parameters.gelu_param.table_size;
        uint32_t size = x.size();
        RSSTensor<T> y(x.shape);

        truncate(x, y, 1 << (x.float_scale_bit - GELU_TABLE_PRECISION), malicious);

        RSSTensor<T> x_and_y({2, size});
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            x_and_y.first.data[i] = x.first.data[i];
            x_and_y.second.data[i] = x.second.data[i];
            x_and_y.first.data[size + i] = y.first.data[i];
            x_and_y.second.data[size + i] = y.second.data[i];
        }

        RSSTensor<T> dx_and_dy({2, size}); // dx = relu(x), dy = relu(y)
        select(y, x_and_y, dx_and_dy, 2, parameters, malicious);
        RSSTensor<T> dx(x.shape), dy(x.shape);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            dx.first.data[i] = dx_and_dy.first.data[i];
            dx.second.data[i] = dx_and_dy.second.data[i];
            dy.first.data[i] = dx_and_dy.first.data[size + i];
            dy.second.data[i] = dx_and_dy.second.data[size + i];
        }

        RSSTensor<T> abs_y(x.shape), sizeSubAbs(x.shape);
        mulConstSubBias(dy, (T)2, y, abs_y); // calculate abs_y = dy * 2 - y

        sub(table_size, abs_y, sizeSubAbs);
        RSSTensor<T> ia(x.shape);
        select(sizeSubAbs, abs_y, ia, parameters, malicious);
        add(ia, (T)(table_size - 1), ia);
        RSSTensor<T> c(x.shape);
        lut(ia, c, parameters.gelu_param, malicious);
        sub(dx, c, res);
    }
}

template <typename T>
void rss_protocols::max_last_dim(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    uint32_t last_dim_size = x.shape.back();
    RSSTensor<T> tmp(x), even, odd, delta;
    uint32_t freeze_size = res.size();
    std::vector<uint32_t> new_shape = res.shape;
    uint32_t half_size;

    int index0, index1;

    while (last_dim_size > 1)
    {
        half_size = (last_dim_size + 1) / 2;
        new_shape.push_back(half_size);

        even.allocate(new_shape);
        odd.allocate(new_shape);
        delta.allocate(new_shape);

        for (uint32_t i = 0; i < freeze_size; ++i)
        {
            for (uint32_t j = 0; j < half_size - 1; ++j)
            {
                index0 = i * half_size + j;
                index1 = i * last_dim_size + 2 * j;
                even.first.data[index0] = tmp.first.data[index1];
                even.second.data[index0] = tmp.second.data[index1];

                odd.first.data[index0] = tmp.first.data[index1 + 1];
                odd.second.data[index0] = tmp.second.data[index1 + 1];
            }
            index0 = i * half_size + half_size - 1;
            index1 = i * last_dim_size + 2 * (half_size - 1);
            even.first.data[index0] = tmp.first.data[index1];
            even.second.data[index0] = tmp.second.data[index1];

            odd.first.data[index0] = tmp.first.data[i * last_dim_size + last_dim_size - 1];
            odd.second.data[index0] = tmp.second.data[i * last_dim_size + last_dim_size - 1];
        }

        sub(even, odd, delta);
        select(delta, delta, delta, parameter, malicious);
        add(odd, delta, even);
        tmp = even;
        even.free();
        odd.free();
        last_dim_size = half_size;
        new_shape = res.shape;
    }
    res = tmp;
}

template <typename T>
void rss_protocols::neg_exp(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    T ln2 = (T)(int)floor(log(2) * (1 << x.float_scale_bit));

    uint32_t size = x.size();
    RSSTensor<T> z(x.shape), p(x.shape), p2(x.shape), neg_exp2_z(x.shape), exp_p(x.shape);

    mulConst(x, (T)(-1), z);
    truncate(z, ln2, malicious);                                // z = -x / ln2
    mulConstAddBias(z, ln2, x, p);                              // p = z * ln2 + x
    add(p, (T)(int)floor(1.353 * (1 << x.float_scale_bit)), p); // p + 1.353
    square(p, p2, true, malicious);                             // (p + 1.353) ^ 2
    mulConst(p2, (T)(int)floor(0.3585 * (1 << x.float_scale_bit)), exp_p);
    truncate(exp_p, malicious);
    add(exp_p, (T)(int)floor(0.344 * (1 << x.float_scale_bit)), exp_p); // 0.3585 * (p + 1.353) ^ 2 + 0.344

    // clip z by choose minimum one between z and scale
    RSSTensor<T> z_minus_scale(z.shape), scale_minus_z(z.shape), min_s_z(z.shape);
    sub((T)x.float_scale_bit, z, scale_minus_z);
    sub(z, (T)x.float_scale_bit, z_minus_scale);
    select(z_minus_scale, scale_minus_z, min_s_z, parameter, malicious);

    if (Party3PC::getInstance().party_id == 0)
    {
        for (int i = 0; i < size; i++)
        {
            z.first.data[i] += min_s_z.first.data[i] + (T)x.float_scale_bit;
            z.second.data[i] += min_s_z.second.data[i];
        }
    }
    else if (Party3PC::getInstance().party_id == 1)
    {
        for (int i = 0; i < size; i++)
        {
            z.first.data[i] += min_s_z.first.data[i];
            z.second.data[i] += min_s_z.second.data[i];
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            z.first.data[i] += min_s_z.first.data[i];
            z.second.data[i] += min_s_z.second.data[i] + (T)x.float_scale_bit;
        }
    }

    lut(z, neg_exp2_z, parameter.nexp2_param, malicious); // 2^(-k)
    mul(exp_p, neg_exp2_z, res, true, malicious);
}

template <typename T>
void rss_protocols::softmax_forward(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    /* only support dim = -1*/
    std::vector<uint32_t> sum_shape = x.shape;
    uint32_t dim_size = sum_shape.back();
    sum_shape.pop_back();

    RSSTensor<T> x_max(sum_shape), delta(x.shape);
    uint32_t common_size = x_max.size();
    max_last_dim(x, x_max, parameter, malicious);
    int index;
#pragma omp parallel for
    for (int i = 0; i < common_size; i++)
    {
        for (int j = 0; j < dim_size; j++)
        {
            index = i * dim_size + j;
            delta.first.data[index] = x.first.data[index] - x_max.first.data[i];
            delta.second.data[index] = x.second.data[index] - x_max.second.data[i];
        }
    }

    RSSTensor<T> exp_x(x.shape);
    neg_exp(delta, exp_x, parameter, malicious);
    RSSTensor<T> sum(sum_shape);

#pragma omp parallel for
    for (int i = 0; i < common_size; i++)
    {
        sum.first.data[i] = 0;
        sum.second.data[i] = 0;
        for (int j = 0; j < dim_size; j++)
        {
            sum.first.data[i] += exp_x.first.data[i * dim_size + j];
            sum.second.data[i] += exp_x.second.data[i * dim_size + j];
        }
    }

    inv(sum, sum, parameter, malicious);
    RSSTensor<T> broadcast_sum(x.shape);
#pragma omp parallel
    for (int i = 0; i < common_size; i++)
    {
        for (int j = 0; j < dim_size; j++)
        {
            broadcast_sum.first.data[i * dim_size + j] = sum.first.data[i];
            broadcast_sum.second.data[i * dim_size + j] = sum.second.data[i];
        }
    }
    mul(exp_x, broadcast_sum, res, true, malicious);
}

template <typename T, typename U>
void rss_protocols::downcast(RSSTensor<T> &x, RSSTensor<U> &res)
{
    int bit_len = sizeof(U) * 8;
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = (x.first.data[i] >> (x.float_scale_bit - res.float_scale_bit)) % ((uint64_t)1 << bit_len);
        res.second.data[i] = (x.second.data[i] >> (x.float_scale_bit - res.float_scale_bit)) % ((uint64_t)1 << bit_len);
    }
}

template <typename U, typename T>
void rss_protocols::upcast(RSSTensor<U> &x, RSSTensor<T> &res, int party_id, bool malicious)
{
    RSSTensor<U> r(x.shape);
    RSSTensor<T> r_upper(x.shape);
    RSSTensor<T> s(x.shape); // s is the most significant bit of r_upper
    r.zeros();
    r_upper.zeros();
    s.zeros();
    uint32_t size = x.size();
    int down_bit_len = sizeof(U) * 8;

    U bias = 1 << (down_bit_len - 2);
    uint64_t down_ring_max = 1ULL << down_bit_len;
    uint64_t scale_delta = 1 << (res.float_scale_bit - x.float_scale_bit);
    uint64_t w_first, w_second;
    uint64_t is_x_hat_non_neg;
    RSSTensor<U> x_hat(x.shape);
    Tensor<U> x_hat_open(x.shape);

    if (party_id == 0)
    {
        for (int i = 0; i < size; i++)
        {
            x_hat.first.data[i] = x.first.data[i] + r.first.data[i] + bias;
            x_hat.second.data[i] = x.second.data[i] + r.second.data[i];
        }

        rss_protocols::restore(x_hat, x_hat_open, malicious);

        for (int i = 0; i < size; i++)
        {
            is_x_hat_non_neg = (1 - (x_hat_open.data[i] >> (down_bit_len - 1)));
            w_first = s.first.data[i] * is_x_hat_non_neg;
            w_second = s.second.data[i] * is_x_hat_non_neg;

            res.first.data[i] = (x_hat_open.data[i] - r_upper.first.data[i] + w_first * down_ring_max - bias) * scale_delta;
            res.second.data[i] = (-r_upper.first.data[i] + w_second * down_ring_max) * scale_delta;
        }
    }
    else if (party_id == 1)
    {
        for (int i = 0; i < size; i++)
        {
            x_hat.first.data[i] = x.first.data[i] + r.first.data[i];
            x_hat.second.data[i] = x.second.data[i] + r.second.data[i];
        }

        rss_protocols::restore(x_hat, x_hat_open, malicious);

        for (int i = 0; i < size; i++)
        {
            is_x_hat_non_neg = (1 - (x_hat_open.data[i] >> (down_bit_len - 1)));
            w_first = s.first.data[i] * is_x_hat_non_neg;
            w_second = s.second.data[i] * is_x_hat_non_neg;

            res.first.data[i] = (-r_upper.first.data[i] + w_first * down_ring_max) * scale_delta;
            res.second.data[i] = (-r_upper.first.data[i] + w_second * down_ring_max) * scale_delta;
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            x_hat.first.data[i] = x.first.data[i] + r.first.data[i];
            x_hat.second.data[i] = x.second.data[i] + r.second.data[i] + bias;
        }

        rss_protocols::restore(x_hat, x_hat_open, malicious);

        for (int i = 0; i < size; i++)
        {
            is_x_hat_non_neg = (1 - (x_hat_open.data[i] >> (down_bit_len - 1)));
            w_first = s.first.data[i] * is_x_hat_non_neg;
            w_second = s.second.data[i] * is_x_hat_non_neg;

            res.first.data[i] = (-r_upper.first.data[i] + w_first * down_ring_max) * scale_delta;
            res.second.data[i] = (x_hat_open.data[i] - r_upper.first.data[i] + w_second * down_ring_max - bias) * scale_delta;
        }
    }
}
