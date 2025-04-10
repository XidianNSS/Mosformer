//
// Created by nssx on 24-12-28.
//

#ifndef LAYERS_H
#define LAYERS_H

#include <functional>
#include <unordered_map>
#include <utility>
#include <mutex>

#include "llmConfig.h"
#include "rss_protocols.h"
#include "params.h"

namespace utils
{
    template <typename T>
    void mask(RSSTensor<T> &attention_probs)
    {
        uint32_t common_size = attention_probs.shape[0] * attention_probs.shape[1];
        uint32_t mask_size = attention_probs.shape[2] * attention_probs.shape[3];
        for (int i = 0; i < common_size; i++)
        {
            for (int j = 0; j < attention_probs.shape[2]; j++)
            {
                for (int k = 0; k < attention_probs.shape[3]; k++)
                {
                    if (j < k)
                    {
                        attention_probs.first.data[i * mask_size + j * attention_probs.shape[3] + k] = 0;
                        attention_probs.second.data[i * mask_size + j * attention_probs.shape[3] + k] = 0;
                    }
                }
            }
        }
    }

    template <typename T>
    void masked_softmax(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious = false);

    template <typename T>
    RSSTensor<T> img2col_pool(RSSTensor<T> &x, const int kernel_size, const int stride = 1, const int pad = 0);

    template <typename T>
    RSSTensor<T> img2col_conv(RSSTensor<T> &x, const int kernel_size, const int stride = 1, const int pad = 0);
}

template <typename T, typename U = T, typename V = U>
class Layers
{
private:
    std::once_flag param_flag;

public:
    std::string name;
    std::unordered_map<std::string, RSSTensor<T>> parameters;
    bool need_parameters = false;

    Layers() = default;

    Layers(std::string name) : name(std::move(name))
    {
    }

    Layers(const Layers<T, U, V> &other) : name(other.name), parameters(other.parameters), need_parameters(other.need_parameters) {}

    virtual ~Layers() = default;

    virtual RSSTensor<T> forward(RSSTensor<T> &x) { return x; }

    virtual RSSTensor<T> forward(RSSTensor<T> &x, Parameters<U> &parameters) { return x; }

    virtual RSSTensor<T> forward(RSSTensor<T> &x, Parameters<U> &parameters_u, Parameters<V> &parameters_v) { return x; }

    virtual RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters_t, Parameters<U> &parameters_u, Parameters<V> &parameters_v) { return x; }

    virtual void init_param() { return; }

    void load_param()
    {
        std::call_once(this->param_flag, [&]()
                       {
            if (need_parameters)
            {
                Loader<T>& loader = Party3PC::getInstance().get_loader<T>();
                if (!loader.is_buffer_empty)
                {
                    this->parameters = loader.get_data();
                }
                need_parameters = false;
                init_param();
            } });
    }

    RSSTensor<T> operator()(RSSTensor<T> &x)
    {
        load_param();
        return forward(x);
    }

    RSSTensor<T> operator()(RSSTensor<T> &x, Parameters<U> &parameters)
    {
        load_param();
        return forward(x, parameters);
    }

    RSSTensor<T> operator()(RSSTensor<T> &x, Parameters<U> &parameters_u, Parameters<V> &parameters_v)
    {
        load_param();
        return forward(x, parameters_u, parameters_v);
    }

    RSSTensor<T> operator()(RSSTensor<T> &x, Parameters<T> &parameters_t, Parameters<U> &parameters_u, Parameters<V> &parameters_v)
    {
        load_param();
        return forward(x, parameters_t, parameters_u, parameters_v);
    }
};

template <typename T>
class SecConv : public Layers<T>
{
public:
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;

    RSSTensor<T> weight;
    RSSTensor<T> bias;

    SecConv(uint32_t in_channels, uint32_t out_channels, uint32_t kernel_size) : Layers<T>("Conv"),
                                                                                 in_channels(in_channels),
                                                                                 out_channels(out_channels), kernel_size(kernel_size)
    {
        this->stride = 1;
        this->padding = 0;

        weight.allocate({out_channels, in_channels, kernel_size, kernel_size});
        bias.allocate({out_channels});
        weight.zeros();
        bias.zeros();

        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;

        this->need_parameters = true;
    }

    SecConv(uint32_t in_channels, uint32_t out_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding) : Layers<T>("Conv"),
                                                                                                                    in_channels(in_channels),
                                                                                                                    out_channels(out_channels),
                                                                                                                    kernel_size(kernel_size),
                                                                                                                    stride(stride), padding(padding)
    {
        weight.allocate({out_channels, in_channels, kernel_size, kernel_size});
        bias.allocate({out_channels});
        weight.zeros();
        bias.zeros();

        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;

        this->need_parameters = true;
    }

    SecConv(RSSTensor<T> &weight, RSSTensor<T> &bias) : Layers<T>("Conv"), weight(weight), bias(bias)
    {
        in_channels = weight.shape[1];
        out_channels = weight.shape[0];
        kernel_size = weight.shape[2];
        stride = 1;
        padding = 0;

        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;
    }

    void set_weight(RSSTensor<T> &weight)
    {
        always_assert(this->in_channels == weight.shape[1]);
        always_assert(this->out_channels == weight.shape[0]);
        always_assert(this->kernel_size == weight.shape[2]);
        this->weight = weight;
        this->parameters["weight"] = weight;
    }

    void set_bias(RSSTensor<T> &bias)
    {
        always_assert(this->out_channels == bias.shape[0]);
        this->bias = bias;
        this->parameters["bias"] = bias;
    }

    ~SecConv() override = default;

    void init_param()
    {
        this->parameters["weight"].reshape({this->out_channels, this->in_channels * this->kernel_size * this->kernel_size});
        weight = this->parameters["weight"].transpose();
        bias = this->parameters["bias"];
    }

    std::vector<uint32_t> get_out_shape(std::vector<uint32_t> in_shape)
    {
        uint32_t batch = in_shape[0], in_height = in_shape[2], in_width = in_shape[3];
        uint32_t out_height = (in_height - this->kernel_size + 2 * this->padding) / this->stride + 1;
        uint32_t out_width = (in_width - this->kernel_size + 2 * this->padding) / this->stride + 1;

        return {batch, this->out_channels, out_height, out_width};
    }

    RSSTensor<T> forward(RSSTensor<T> &x) override
    {
        std::vector<uint32_t> out_shape = get_out_shape(x.shape);

        RSSTensor<T> x_col = utils::img2col_conv(x, this->kernel_size, this->stride, this->padding);
        RSSTensor<T> output({x_col.shape[0], x_col.shape[1], this->out_channels});

        rss_protocols::matMul(x_col, weight, output, true, IS_MALICIOUS);
        uint32_t broadcast_size = x_col.shape[0] * x_col.shape[1];

        for (int i = 0; i < broadcast_size; i++)
        {
            for (int j = 0; j < this->out_channels; j++)
            {
                output.first.data[i * this->out_channels + j] += bias.first.data[j];
                output.second.data[i * this->out_channels + j] += bias.second.data[j];
            }
        }
        output = output.transpose();
        output.reshape(out_shape);
        return output;
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        return forward(x);
    }
};

template <typename T>
class SecMaxPool : public Layers<T>
{
public:
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;

    SecMaxPool(uint32_t kernel_size, uint32_t stride = 1, uint32_t padding = 0) : Layers<T>("MaxPool"), kernel_size(kernel_size), stride(stride), padding(padding)
    {
    }

    std::vector<uint32_t> get_out_shape(std::vector<uint32_t> in_shape)
    {
        uint32_t batch = in_shape[0], in_channel = in_shape[1], in_height = in_shape[2], in_width = in_shape[3];
        uint32_t out_height = (in_height - this->kernel_size + 2 * this->padding) / this->stride + 1;
        uint32_t out_width = (in_width - this->kernel_size + 2 * this->padding) / this->stride + 1;

        return {batch, in_channel, out_height, out_width};
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        std::vector<uint32_t> out_shape = get_out_shape(x.shape);
        RSSTensor<T> x_col = utils::img2col_pool(x, this->kernel_size, this->stride, this->padding);
        std::vector<uint32_t> col_max_shape = x_col.shape;
        col_max_shape.pop_back();

        RSSTensor<T> res(col_max_shape);

        rss_protocols::max_last_dim(x_col, res, parameters, IS_MALICIOUS);
        res.reshape(out_shape);
        return res;
    }
};

template <typename T>
class SecAvgPool : public Layers<T>
{
public:
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;

    SecAvgPool(uint32_t kernel_size, uint32_t stride = 1, uint32_t padding = 0) : Layers<T>("AvgPool"), kernel_size(kernel_size), stride(stride), padding(padding)
    {
    }

    std::vector<uint32_t> get_out_shape(std::vector<uint32_t> in_shape)
    {
        uint32_t batch = in_shape[0], in_channel = in_shape[1], in_height = in_shape[2], in_width = in_shape[3];
        uint32_t out_height = (in_height - this->kernel_size + 2 * this->padding) / this->stride + 1;
        uint32_t out_width = (in_width - this->kernel_size + 2 * this->padding) / this->stride + 1;

        return {batch, in_channel, out_height, out_width};
    }

    RSSTensor<T> forward(RSSTensor<T> &x) override
    {
        std::vector<uint32_t> out_shape = get_out_shape(x.shape);
        RSSTensor<T> x_col = utils::img2col_pool(x, this->kernel_size, this->stride, this->padding);
        std::vector<uint32_t> res_shape = x_col.shape;
        uint32_t kernel_shape = res_shape.back();
        res_shape.pop_back();

        RSSTensor<T> res(res_shape);
        uint32_t res_size = res.size();
        for (int i = 0; i < res_size; i++)
        {
            res.first.data[i] = 0;
            res.second.data[i] = 0;
            for (int j = 0; j < kernel_shape; j++)
            {
                res.first.data[i] += x_col.first.data[i * kernel_shape + j];
                res.second.data[i] += x_col.second.data[i * kernel_shape + j];
            }
        }
        rss_protocols::truncate(res, kernel_shape, IS_MALICIOUS);

        return res;
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        return forward(x);
    }
};

template <typename T>
class SecAdaptiveAvgPool : public Layers<T>
{
public:
    uint32_t out_size;

    SecAdaptiveAvgPool(uint32_t out_size) : Layers<T>("AdaptiveAvgPool"), out_size(out_size)
    {
    }

    std::vector<uint32_t> get_out_shape(std::vector<uint32_t> in_shape, uint32_t kernel_size, uint32_t stride, uint32_t padding)
    {
        uint32_t batch = in_shape[0], in_channel = in_shape[1], in_height = in_shape[2], in_width = in_shape[3];
        uint32_t out_height = (in_height - kernel_size + 2 * padding) / stride + 1;
        uint32_t out_width = (in_width - kernel_size + 2 * padding) / stride + 1;

        return {batch, in_channel, out_height, out_width};
    }

    RSSTensor<T> forward(RSSTensor<T> &x) override
    {
        uint32_t in_size = x.shape.back();
        uint32_t stride = floor((float)(in_size) / (float)(this->out_size));
        uint32_t kernel_size = in_size - (this->out_size - 1) * stride;

        std::vector<uint32_t> out_shape = get_out_shape(x.shape, kernel_size, stride, 0);
        RSSTensor<T> x_col = utils::img2col_pool(x, kernel_size, stride, 0);
        std::vector<uint32_t> res_shape = x_col.shape;
        uint32_t kernel_shape = res_shape.back();
        res_shape.pop_back();

        RSSTensor<T> res(res_shape);
        uint32_t res_size = res.size();
        for (int i = 0; i < res_size; i++)
        {
            res.first.data[i] = 0;
            res.second.data[i] = 0;
            for (int j = 0; j < kernel_shape; j++)
            {
                res.first.data[i] += x_col.first.data[i * kernel_shape + j];
                res.second.data[i] += x_col.second.data[i * kernel_shape + j];
            }
        }
        rss_protocols::truncate(res, kernel_shape, IS_MALICIOUS);

        return res;
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        return forward(x);
    }
};

template <typename T>
class SecBatchNorm : public Layers<T>
{
public:
    uint32_t num_features;
    RSSTensor<T> weight;
    RSSTensor<T> bias;

    SecBatchNorm(uint32_t num_features) : Layers<T>("BatchNorm"), num_features(num_features)
    {
        weight.allocate({num_features});
        bias.allocate({num_features});
        weight.zeros();
        bias.zeros();
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;

        this->need_parameters = true;
    }

    void set_weight(RSSTensor<T> &weight)
    {
        always_assert(this->num_features == weight.shape[0]);
        this->weight = weight;
        this->parameters["weight"] = weight;
    }

    void set_bias(RSSTensor<T> &bias)
    {
        always_assert(this->num_features == bias.shape[0]);
        this->bias = bias;
        this->parameters["bias"] = bias;
    }

    SecBatchNorm(RSSTensor<T> &weight, RSSTensor<T> &bias) : Layers<T>("BatchNorm")
    {
        num_features = weight.shape[0];
        weight.allocate({num_features});
        bias.allocate({num_features});
        this->weight = weight;
        this->bias = bias;
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;
    }

    void init_param()
    {
        weight = this->parameters["weight"];
        bias = this->parameters["bias"];
    }

    RSSTensor<T> forward(RSSTensor<T> &x) override
    {
        RSSTensor<T> res(x.shape);

        uint32_t batch = x.shape[0], channel = x.shape[1], each_shape = x.shape[2] * x.shape[3];
        always_assert(channel == this->num_features);
        RSSTensor<T> gamma(x.shape);

        int index;
        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < channel; j++)
            {
                for (int k = 0; k < each_shape; k++)
                {
                    index = i * channel * each_shape + j * each_shape + k;
                    gamma.first.data[index] = this->weight.first.data[j];
                    gamma.second.data[index] = this->weight.second.data[j];
                }
            }
        }

        RSSTensor<T> x_weighted(x.shape);
        rss_protocols::mul(x, gamma, x_weighted, true, IS_MALICIOUS); // calculate gamma * x

        // calculate gamma * x + beta
        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < channel; j++)
            {
                for (int k = 0; k < each_shape; k++)
                {
                    index = i * channel * each_shape + j * each_shape + k;
                    res.first.data[index] = x_weighted.first.data[index] + this->bias.first.data[j];
                    res.second.data[index] = x_weighted.second.data[index] + this->bias.second.data[j];
                }
            }
        }

        return res;
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        return forward(x);
    }
};

template <typename T>
class SecLinear : public Layers<T>
{
public:
    uint32_t in_features;
    uint32_t out_features;
    RSSTensor<T> weight;
    RSSTensor<T> bias;

    SecLinear(uint32_t in_features, uint32_t out_features) : Layers<T>("Linear"),
                                                             in_features(in_features),
                                                             out_features(out_features)
    {
        weight.allocate({out_features, in_features});
        bias.allocate({out_features});
        weight.zeros();
        bias.zeros();
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;

        this->need_parameters = true;
    }

    SecLinear(RSSTensor<T> &weight, RSSTensor<T> &bias) : Layers<T>("Linear")
    {
        in_features = weight.shape[1];
        out_features = weight.shape[0];
        weight.allocate({out_features, in_features});
        bias.allocate({out_features});
        this->weight = weight;
        this->bias = bias;
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;
    }

    void set_weight(RSSTensor<T> &weight)
    {
        always_assert(this->in_features == weight.shape[1]);
        always_assert(this->out_features == weight.shape[0]);
        this->weight = weight;
        this->parameters["weight"] = weight;
    }

    void set_bias(RSSTensor<T> &bias)
    {
        always_assert(this->out_features == bias.shape[0]);
        this->bias = bias;
        this->parameters["bias"] = bias;
    }

    void init_param()
    {
        weight = this->parameters["weight"].transpose();
        bias = this->parameters["bias"];
    }

    RSSTensor<T> forward(RSSTensor<T> &x) override
    {
        std::vector<uint32_t> res_shape;
        uint32_t broadcast_size = 1;
        for (int i = 0; i < x.shape.size() - 1; i++)
        {
            res_shape.push_back(x.shape[i]);
            broadcast_size *= x.shape[i];
        }
        res_shape.push_back(this->weight.shape[1]);
        RSSTensor<T> res(res_shape);
        rss_protocols::matMul(x, weight, res, true, IS_MALICIOUS);

        for (int i = 0; i < broadcast_size; i++)
        {
            for (int j = 0; j < this->out_features; j++)
            {
                res.first.data[i * this->out_features + j] += this->bias.first.data[j];
                res.second.data[i * this->out_features + j] += this->bias.second.data[j];
            }
        }
        return res;
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        return forward(x);
    }
};

template <typename T>
class SecReLU : public Layers<T>
{
public:
    SecReLU() : Layers<T>("ReLU")
    {
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> res(x.shape);
        rss_protocols::select(x, x, res, parameters, IS_MALICIOUS);
        return res;
    }
};

template <typename T>
class SecGELU : public Layers<T>
{
public:
    SecGELU() : Layers<T>("GELU")
    {
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> res(x.shape);
        rss_protocols::gelu(x, res, parameters, IS_MALICIOUS);
        return res;
    }
};

template <typename T>
class SecSoftmax : public Layers<T>
{
public:
    bool masked;

    SecSoftmax(bool masked = false) : Layers<T>("Softmax"), masked(masked)
    {
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> res(x.shape);
        if (masked)
            utils::masked_softmax(x, res, parameters, IS_MALICIOUS);
        else
            rss_protocols::softmax_forward(x, res, parameters, IS_MALICIOUS);
        return res;
    }
};

template <typename T>
class SecLayerNorm : public Layers<T>
{
public:
    uint32_t normalized_shape;
    RSSTensor<T> weight;
    RSSTensor<T> bias;

    SecLayerNorm(uint32_t normalized_shape) : Layers<T>("LayerNorm"),
                                              normalized_shape(normalized_shape)
    {
        weight.allocate({normalized_shape});
        bias.allocate({normalized_shape});
        weight.zeros();
        bias.zeros();
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;

        this->need_parameters = true;
    }

    void set_weight(RSSTensor<T> &weight)
    {
        always_assert(this->normalized_shape == weight.shape[0]);
        this->weight = weight;
        this->parameters["weight"] = weight;
    }

    void set_bias(RSSTensor<T> &bias)
    {
        always_assert(this->normalized_shape == bias.shape[0]);
        this->bias = bias;
        this->parameters["bias"] = bias;
    }

    SecLayerNorm(RSSTensor<T> &weight, RSSTensor<T> &bias) : Layers<T>(
                                                                 "LayerNorm")
    {
        normalized_shape = weight.shape[0];
        weight.allocate({normalized_shape});
        bias.allocate({normalized_shape});
        this->weight = weight;
        this->bias = bias;
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;
    }

    void init_param()
    {
        weight = this->parameters["weight"];
        bias = this->parameters["bias"];
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> res(x.shape);

        std::vector<uint32_t> sum_shape = x.shape;
        uint32_t dim_size = sum_shape.back();
        always_assert(dim_size == this->normalized_shape); // can only normalize the last dimension
        sum_shape.pop_back();

        RSSTensor<T> x_sum(sum_shape);
        const uint32_t sum_size = x_sum.size();
#pragma omp parallel for
        for (int i = 0; i < sum_size; i++)
        {
            x_sum.first.data[i] = 0;
            x_sum.second.data[i] = 0;
            for (int j = 0; j < dim_size; j++)
            {
                x_sum.first.data[i] += x.first.data[i * dim_size + j];
                x_sum.second.data[i] += x.second.data[i * dim_size + j];
            }
        }

        rss_protocols::truncate(x_sum, normalized_shape, IS_MALICIOUS); // calculate mean

        RSSTensor<T> z(x.shape);
#pragma omp parallel for
        for (int i = 0; i < sum_size; i++)
        {
            for (int j = 0; j < dim_size; j++)
            {
                z.first.data[i * dim_size + j] = x.first.data[i * dim_size + j] - x_sum.first.data[i];
                z.second.data[i * dim_size + j] = x.second.data[i * dim_size + j] - x_sum.second.data[i];
            }
        }

        RSSTensor<T> z_square(x.shape), square_sum(sum_shape);
        rss_protocols::square(z, z_square, true, IS_MALICIOUS); // calculate z^2

#pragma omp parallel for
        for (int i = 0; i < sum_size; i++)
        {
            square_sum.first.data[i] = 0;
            square_sum.second.data[i] = 0;
            for (int j = 0; j < dim_size; j++)
            {
                square_sum.first.data[i] += z_square.first.data[i * dim_size + j];
                square_sum.second.data[i] += z_square.second.data[i * dim_size + j];
            }
        }

        rss_protocols::truncate(square_sum, normalized_shape, IS_MALICIOUS); // calculate variance

        RSSTensor<T> rsqrt_var(sum_shape), broadcast_rsqrt_var(x.shape), gamma(x.shape);
        rss_protocols::rsqrt(square_sum, rsqrt_var, parameters, IS_MALICIOUS); // calculate 1 / sqrt(var)

#pragma omp parallel for
        for (int i = 0; i < sum_size; i++)
        {
            for (int j = 0; j < dim_size; j++)
            {
                broadcast_rsqrt_var.first.data[i * dim_size + j] = rsqrt_var.first.data[i];
                broadcast_rsqrt_var.second.data[i * dim_size + j] = rsqrt_var.second.data[i];

                gamma.first.data[i * dim_size + j] = this->weight.first.data[j];
                gamma.second.data[i * dim_size + j] = this->weight.second.data[j];
            }
        }

        RSSTensor<T> z_norm(x.shape), z_norm_weighted(x.shape);
        rss_protocols::mul(z, broadcast_rsqrt_var, z_norm, true, IS_MALICIOUS); // calculate z_norm
        rss_protocols::mul(z_norm, gamma, z_norm_weighted, true, IS_MALICIOUS); // calculate gamma * z_norm

        // calculate gamma * z_norm + beta
#pragma omp parallel for
        for (int i = 0; i < sum_size; i++)
        {
            for (int j = 0; j < dim_size; j++)
            {
                res.first.data[i * dim_size + j] = z_norm_weighted.first.data[i * dim_size + j] + this->bias.first.data[j];
                res.second.data[i * dim_size + j] = z_norm_weighted.second.data[i * dim_size + j] + this->bias.second.data[j];
            }
        }

        return res;
    }
};

template <typename T>
class SecSelfAttention : public Layers<T>
{
public:
    uint32_t num_heads;
    uint32_t head_size;
    uint32_t all_head_size;
    Layers<T> *query, *key, *value, *softmax;

    SecSelfAttention(TransformerConfig &config, const bool masked = false) : Layers<T>("SelfAttention"),
                                                                             num_heads(config.num_attention_heads),
                                                                             head_size(
                                                                                 config.hidden_size / config.num_attention_heads),
                                                                             all_head_size(num_heads * head_size)
    {
        query = new SecLinear<T>(config.hidden_size, all_head_size);
        key = new SecLinear<T>(config.hidden_size, all_head_size);
        value = new SecLinear<T>(config.hidden_size, all_head_size);
        softmax = new SecSoftmax<T>(masked);
    }

    SecSelfAttention(TransformerConfig &config, std::vector<Layers<T>> layers, const bool masked = false) : Layers<T>("SelfAttention"),
                                                                                                            num_heads(config.num_attention_heads),
                                                                                                            head_size(config.hidden_size / config.num_attention_heads),
                                                                                                            all_head_size(num_heads * head_size)
    {
        query = &layers[0];
        key = &layers[1];
        value = &layers[2];
        softmax = &layers[3];
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> mixed_query = (*this->query)(x);
        RSSTensor<T> mixed_key = (*this->key)(x);
        RSSTensor<T> mixed_value = (*this->value)(x);
        std::vector<uint32_t> new_shape = x.shape;
        new_shape.pop_back();
        new_shape.push_back(this->num_heads);
        new_shape.push_back(this->head_size);

        mixed_query.reshape(new_shape);
        mixed_key.reshape(new_shape);
        mixed_value.reshape(new_shape);

        RSSTensor<T> query_layer = mixed_query.permute({0, 2, 1, 3});
        RSSTensor<T> key_layer = mixed_key.permute({0, 2, 3, 1});
        RSSTensor<T> value_layer = mixed_value.permute({0, 2, 1, 3});

        RSSTensor<T> attention_scores(
            {query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], key_layer.shape[3]});
        rss_protocols::matMul(query_layer, key_layer, attention_scores, true, IS_MALICIOUS);
        rss_protocols::truncate(attention_scores, (uint32_t)sqrt((double)this->head_size), IS_MALICIOUS);

        RSSTensor<T> attention_probs = (*this->softmax)(attention_scores, parameters);

        RSSTensor<T> context_layer({attention_probs.shape[0], attention_probs.shape[1], attention_probs.shape[2],
                                    value_layer.shape[3]});
        rss_protocols::matMul(attention_probs, value_layer, context_layer, true, IS_MALICIOUS);

        RSSTensor<T> res = context_layer.permute({0, 2, 1, 3});
        std::vector<uint32_t> res_shape = res.shape;
        res_shape.pop_back();
        res_shape.pop_back();
        res_shape.push_back(this->all_head_size);
        res.reshape(res_shape);

        return res;
    }
};

template <typename T>
class SecAttention : public Layers<T>
{
public:
    Layers<T> *self, *dense;

    SecAttention(TransformerConfig &config, const bool masked = false) : Layers<T>(
                                                                             "Attention")
    {
        self = new SecSelfAttention<T>(config, masked);
        dense = new SecLinear<T>(config.hidden_size, config.hidden_size);
    }

    SecAttention(TransformerConfig &config, std::vector<Layers<T>> layers) : Layers<T>(
                                                                                 "Attention")
    {
        self = &layers[0];
        dense = &layers[1];
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> self_outputs = (*this->self)(x, parameters);
        return (*this->dense)(self_outputs, parameters);
    }
};

template <typename T>
class SecGPT2Attention : public Layers<T>
{
public:
    uint32_t num_heads;
    uint32_t head_size;
    uint32_t all_head_size;
    Layers<T> *c_attn, *softmax, *c_proj;

    SecGPT2Attention(TransformerConfig &config) : Layers<T>("GPT2Attention"),
                                                  num_heads(config.num_attention_heads),
                                                  head_size(
                                                      config.hidden_size / config.num_attention_heads),
                                                  all_head_size(num_heads * head_size)
    {
        c_attn = new SecLinear<T>(config.hidden_size, 3 * all_head_size);
        softmax = new SecSoftmax<T>(true);
        c_proj = new SecLinear<T>(config.hidden_size, all_head_size);
    }

    SecGPT2Attention(TransformerConfig &config, std::vector<Layers<T>> layers) : Layers<T>("GPT2Attention"),
                                                                                 num_heads(config.num_attention_heads),
                                                                                 head_size(config.hidden_size / config.num_attention_heads),
                                                                                 all_head_size(num_heads * head_size)
    {
        c_attn = &layers[0];
        softmax = &layers[1];
        c_proj = &layers[2];
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> qkv = (*this->c_attn)(x);
        RSSTensor<T> mixed_query(x.shape), mixed_key(x.shape), mixed_value(x.shape);

        uint32_t size = x.size(), last_dim = x.shape[x.shape.size() - 1];
        uint32_t other_dim = size / last_dim;
        // split, dim = -1
        uint32_t qkv_last_dim = qkv.shape[qkv.shape.size() - 1];
        uint32_t index, qkv_index;
        for (int i = 0; i < other_dim; i++)
        {
            for (int j = 0; j < last_dim; j++)
            {
                index = i * last_dim + j;
                qkv_index = i * qkv_last_dim + j;

                mixed_query.first.data[index] = qkv.first.data[qkv_index];
                mixed_query.second.data[index] = qkv.second.data[qkv_index];

                mixed_key.first.data[index] = qkv.first.data[qkv_index + last_dim];
                mixed_key.second.data[index] = qkv.second.data[qkv_index + last_dim];

                mixed_value.first.data[index] = qkv.first.data[qkv_index + last_dim * 2];
                mixed_value.second.data[index] = qkv.second.data[qkv_index + last_dim * 2];
            }
        }

        std::vector<uint32_t> new_shape = x.shape;
        new_shape.pop_back();
        new_shape.push_back(this->num_heads);
        new_shape.push_back(this->head_size);

        mixed_query.reshape(new_shape);
        mixed_key.reshape(new_shape);
        mixed_value.reshape(new_shape);

        RSSTensor<T> query_layer = mixed_query.permute({0, 2, 1, 3});
        RSSTensor<T> key_layer = mixed_key.permute({0, 2, 3, 1});
        RSSTensor<T> value_layer = mixed_value.permute({0, 2, 1, 3});

        RSSTensor<T> attention_scores(
            {query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], key_layer.shape[3]});
        rss_protocols::matMul(query_layer, key_layer, attention_scores, true, IS_MALICIOUS);
        rss_protocols::truncate(attention_scores, (uint32_t)sqrt((double)this->head_size), IS_MALICIOUS);

        RSSTensor<T> attention_probs = (*this->softmax)(attention_scores, parameters);

        RSSTensor<T> context_layer({attention_probs.shape[0], attention_probs.shape[1], attention_probs.shape[2],
                                    value_layer.shape[3]});
        rss_protocols::matMul(attention_probs, value_layer, context_layer, true, IS_MALICIOUS);

        RSSTensor<T> attn_output = context_layer.permute({0, 2, 1, 3});
        std::vector<uint32_t> output_shape = attn_output.shape;
        output_shape.pop_back();
        output_shape.pop_back();
        output_shape.push_back(this->all_head_size);
        attn_output.reshape(output_shape);

        return (*this->c_proj)(attn_output);
    }
};

template <typename T>
class SecFFN : public Layers<T>
{
public:
    Layers<T> *dense0, *act_fn, *dense1;

    SecFFN(TransformerConfig &config) : Layers<T>("FFN")
    {
        dense0 = new SecLinear<T>(config.hidden_size, config.intermediate_size);
        if (config.isReLUAct)
        {
            act_fn = new SecReLU<T>();
        }
        else
        {
            act_fn = new SecGELU<T>();
        }
        dense1 = new SecLinear<T>(config.intermediate_size, config.hidden_size);
    }

    SecFFN(TransformerConfig &config,
           std::vector<Layers<T>> layers) : Layers<T>("FFN")
    {
        dense0 = &layers[0];
        act_fn = &layers[1];
        dense1 = &layers[2];
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> hidden_states = (*this->dense0)(x);
        hidden_states = (*this->act_fn)(hidden_states, parameters);
        return (*this->dense1)(hidden_states);
    }
};

template <typename T>
class SecEncoderLayer : public Layers<T>
{
public:
    Layers<T> *attention, *LayerNorm1, *ffn, *LayerNorm2;

    SecEncoderLayer(TransformerConfig &config) : Layers<T>("EncoderLayer")
    {
        attention = new SecAttention<T>(config);
        LayerNorm1 = new SecLayerNorm<T>(config.hidden_size);
        ffn = new SecFFN<T>(config);
        LayerNorm2 = new SecLayerNorm<T>(config.hidden_size);
    }

    SecEncoderLayer(TransformerConfig &config,
                    std::vector<Layers<T>> layers) : Layers<T>("EncoderLayer")
    {
        attention = &layers[0];
        LayerNorm1 = &layers[1];
        ffn = &layers[2];
        LayerNorm2 = &layers[3];
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> self_attention_output = (*this->attention)(x, parameters);
        rss_protocols::add(x, self_attention_output, self_attention_output);
        RSSTensor<T> attention_output = (*this->LayerNorm1)(self_attention_output, parameters);

        RSSTensor<T> ffn_output = (*this->ffn)(attention_output, parameters);
        rss_protocols::add(attention_output, ffn_output, ffn_output);
        return (*this->LayerNorm2)(ffn_output, parameters);
    }
};

template <typename T>
class SecDecoderLayer : public Layers<T>
{
public:
    Layers<T> *masked_attention, *LayerNorm0, *attention, *LayerNorm1, *ffn, *LayerNorm2;

    SecDecoderLayer(TransformerConfig &config) : Layers<T>("DecoderLayer")
    {
        masked_attention = new SecAttention<T>(config, true);
        LayerNorm0 = new SecLayerNorm<T>(config.hidden_size);
        attention = new SecAttention<T>(config);
        LayerNorm1 = new SecLayerNorm<T>(config.hidden_size);
        ffn = new SecFFN<T>(config);
        LayerNorm2 = new SecLayerNorm<T>(config.hidden_size);
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        // TODO: input from previous layers
        RSSTensor<T> masked_attention_output = (*this->masked_attention)(x, parameters);
        rss_protocols::add(x, masked_attention_output, masked_attention_output);
        RSSTensor<T> masked_attention_output_norm = (*this->LayerNorm0)(masked_attention_output, parameters);

        RSSTensor<T> attention_output = (*this->attention)(masked_attention_output_norm, parameters);
        rss_protocols::add(masked_attention_output_norm, attention_output, attention_output);
        RSSTensor<T> attention_output_norm = (*this->LayerNorm1)(attention_output, parameters);

        RSSTensor<T> ffn_output = (*this->ffn)(attention_output_norm, parameters);
        rss_protocols::add(attention_output_norm, ffn_output, ffn_output);
        return (*this->LayerNorm2)(ffn_output, parameters);
    }
};

template <typename T>
class SecGPT2Layer : public Layers<T>
{
public:
    Layers<T> *LayerNorm0, *masked_attention, *LayerNorm1, *ffn;

    SecGPT2Layer(TransformerConfig &config) : Layers<T>("GPT2Layer")
    {
        LayerNorm0 = new SecLayerNorm<T>(config.hidden_size);
        masked_attention = new SecGPT2Attention<T>(config);
        LayerNorm1 = new SecLayerNorm<T>(config.hidden_size);
        ffn = new SecFFN<T>(config);
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> ln0_out = (*this->LayerNorm0)(x, parameters);
        RSSTensor<T> attn_out = (*this->masked_attention)(ln0_out, parameters);
        rss_protocols::add(x, attn_out, attn_out);
        RSSTensor<T> ln1_out = (*this->LayerNorm1)(attn_out, parameters);
        RSSTensor<T> ffn_out = (*this->ffn)(ln1_out, parameters);
        rss_protocols::add(attn_out, ffn_out, ffn_out);
        return ffn_out;
    }
};

template <typename T>
class SecEncoder : public Layers<T>
{
public:
    TransformerConfig config;
    std::vector<SecEncoderLayer<T>> layers;

    SecEncoder(TransformerConfig &config) : Layers<T>("Encoder"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
    }

    SecEncoder(TransformerConfig &config,
               std::vector<Layers<T>> all_layers) : Layers<T>("Encoder"),
                                                    config(config)
    {
        always_assert(config.num_layers == all_layers.size());
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config, all_layers[i]);
        }
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters);
        }
        return hidden_states;
    }
};

template <typename T>
class SecDecoder : public Layers<T>
{
public:
    TransformerConfig config;
    std::vector<SecDecoderLayer<T>> layers;

    SecDecoder(TransformerConfig &config) : Layers<T>("Decoder"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters);
        }
        return hidden_states;
    }
};

template <typename T>
class SecGPT2 : public Layers<T>
{
public:
    TransformerConfig config;
    std::vector<SecGPT2Layer<T>> layers;
    Layers<T> *ln_f;

    SecGPT2(TransformerConfig &config) : Layers<T>("GPT2"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
        ln_f = new SecLayerNorm<T>(config.hidden_size);
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters);
        }
        return (*this->ln_f)(hidden_states, parameters);
    }
};

template <typename T>
class SecAlexNet : public Layers<T>
{
    // AlexNet for CIFAR 10, the same arichtecture as Falcon
public:
    Layers<T> *conv1, *pool1, *relu1, *conv2, *pool2, *relu2, *conv3, *relu3, *conv4, *relu4, *conv5, *relu5, *fc1, *relu6, *fc2, *relu7, *fc3, *relu8;

    SecAlexNet() : Layers<T>("AlexNet")
    {
        conv1 = new SecConv<T>(3, 96, 11, 4, 9);
        pool1 = new SecMaxPool<T>(3, 2);
        relu1 = new SecReLU<T>();

        conv2 = new SecConv<T>(96, 256, 5, 1, 1);
        pool2 = new SecMaxPool<T>(3, 2);
        relu2 = new SecReLU<T>();

        conv3 = new SecConv<T>(256, 384, 3, 1, 1);
        relu3 = new SecReLU<T>();

        conv4 = new SecConv<T>(384, 384, 3, 1, 1);
        relu4 = new SecReLU<T>();

        conv5 = new SecConv<T>(384, 256, 3, 1, 1);
        relu5 = new SecReLU<T>();

        fc1 = new SecLinear<T>(256, 256);
        relu6 = new SecReLU<T>();
        
        fc2 = new SecLinear<T>(256, 256);
        relu7 = new SecReLU<T>();

        fc3 = new SecLinear<T>(256, 10);
        relu8 = new SecReLU<T>();
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> res(x.shape);
        res = x;
        res = (*this->conv1)(res, parameters);
        res = (*this->pool1)(res, parameters);
        res = (*this->relu1)(res, parameters);

        res = (*this->conv2)(res, parameters);
        res = (*this->pool2)(res, parameters);
        res = (*this->relu2)(res, parameters);

        res = (*this->conv3)(res, parameters);
        res = (*this->relu3)(res, parameters);

        res = (*this->conv4)(res, parameters);
        res = (*this->relu4)(res, parameters);

        res = (*this->conv5)(res, parameters);
        res = (*this->relu5)(res, parameters);

        res.reshape({res.size() / (256), 256});

        res = (*this->fc1)(res, parameters);
        res = (*this->relu6)(res, parameters);
        res = (*this->fc2)(res, parameters);
        res = (*this->relu7)(res, parameters);
        res = (*this->fc3)(res, parameters);
        res = (*this->relu8)(res, parameters);
        return res;
    }
};

template <typename T>
class ShortCut : public Layers<T>
{
public:
    Layers<T> *conv, *bn;

    ShortCut(uint32_t in_channels, uint32_t out_channels, uint32_t expansion, uint32_t stride) : Layers<T>("Shortcut")
    {
        conv = new SecConv<T>(in_channels, out_channels * expansion, 1, stride, 0);
        bn = new SecBatchNorm<T>(out_channels * expansion);
    }

    RSSTensor<T> forward(RSSTensor<T> &x) override
    {
        RSSTensor<T> res = (*this->conv)(x);
        return (*this->bn)(res);
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        return forward(x);
    }
};

template <typename T>
class BottleNeck : public Layers<T>
{
public:
    Layers<T> *conv1, *bn1, *relu1, *conv2, *bn2, *relu2, *conv3, *bn3, *shortcut, *relu;
    uint32_t expansion = 4;
    
    BottleNeck(uint32_t in_channels, uint32_t out_channels, uint32_t stride = 1) : Layers<T>("BottleNeck")
    {
        conv1 = new SecConv<T>(in_channels, out_channels, 1);
        bn1 = new SecBatchNorm<T>(out_channels);
        relu1 = new SecReLU<T>();
        conv2 = new SecConv<T>(out_channels, out_channels, 3, stride, 1);
        bn2 = new SecBatchNorm<T>(out_channels);
        relu2 = new SecReLU<T>();
        conv3 = new SecConv<T>(out_channels, out_channels * expansion, 1);
        bn3 = new SecBatchNorm<T>(out_channels * expansion);

        relu = new SecReLU<T>();

        if (stride != 1 || in_channels != out_channels * expansion)
        {
            shortcut = new ShortCut<T>(in_channels, out_channels, expansion, stride);
        }
        else
        {
            shortcut = new Layers<T>();
        }
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> x_copy(x.shape);
        x_copy = x;
        RSSTensor<T> residual_res = (*this->conv1)(x);
        residual_res = (*this->bn1)(residual_res);
        residual_res = (*this->relu1)(residual_res, parameters);

        residual_res = (*this->conv2)(residual_res);
        residual_res = (*this->bn2)(residual_res);
        residual_res = (*this->relu2)(residual_res, parameters);

        residual_res = (*this->conv3)(residual_res);
        residual_res = (*this->bn3)(residual_res);

        RSSTensor<T> shortcut_res = (*this->shortcut)(x_copy);
        rss_protocols::add(residual_res, shortcut_res, residual_res);

        return (*this->relu1)(residual_res, parameters);
    }
};

template <typename T>
class SecResNet50 : public Layers<T>
{
public:
    Layers<T> *conv, *bn, *relu, *pool;
    std::vector<BottleNeck<T>> conv2_x, conv3_x, conv4_x, conv5_x;
    Layers<T> *avg_pool, *fc;
    uint32_t in_channels = 64;
    uint32_t expansion = 4;

    std::vector<BottleNeck<T>> make_layer(uint32_t out_channels, uint32_t num_blocks, uint32_t stride)
    {
        std::vector<BottleNeck<T>> layers;

        for(int i = 0; i < num_blocks; i++)
        {
            if (i != 0)
            {
                stride = 1;
            }
            layers.emplace_back(this->in_channels, out_channels, stride);
            this->in_channels = out_channels * expansion;
        }

        return layers;
    }

    SecResNet50(uint32_t num_classes = 100) : Layers<T>("resnet")
    {
        conv = new SecConv<T>(3, 64, 7, 2, 3);
        bn = new SecBatchNorm<T>(64);
        relu = new SecReLU<T>();
        pool = new SecMaxPool<T>(3, 2, 1);

        conv2_x = make_layer(64, 3, 1);
        conv3_x = make_layer(128, 4, 2);
        conv4_x = make_layer(256, 6, 2);
        conv5_x = make_layer(512, 3, 2);

        avg_pool = new SecAdaptiveAvgPool<T>(1);
        fc = new SecLinear<T>(512 * expansion, num_classes);
    }

    RSSTensor<T> forward(RSSTensor<T> &x, Parameters<T> &parameters) override
    {
        RSSTensor<T> output = (*this->conv)(x);
        output = (*this->bn)(output);
        output = (*this->relu)(output, parameters);
        output = (*this->pool)(output, parameters);

        for(int i = 0; i < conv2_x.size(); i++)
        {
            output = this->conv2_x[i](output, parameters);
        }

        for(int i = 0; i < conv3_x.size(); i++)
        {
            output = this->conv3_x[i](output, parameters);
        }

        for(int i = 0; i < conv4_x.size(); i++)
        {
            output = this->conv4_x[i](output, parameters);
        }

        for(int i = 0; i < conv5_x.size(); i++)
        {
            output = this->conv5_x[i](output, parameters);
        }

        output = (*this->avg_pool)(output);
        output.reshape({output.shape[0], output.size() / output.shape[0]});
        return (*this->fc)(output);
    }

};

class SecMixedLayerNorm : public Layers<uint64_t, uint32_t>
{
public:
    uint32_t normalized_shape;
    RSSTensor<uint64_t> weight;
    RSSTensor<uint64_t> bias;

    SecMixedLayerNorm(uint32_t normalized_shape) : Layers<uint64_t, uint32_t>("LayerNorm"),
                                                   normalized_shape(normalized_shape)
    {
        weight.allocate({normalized_shape});
        bias.allocate({normalized_shape});
        weight.zeros();
        bias.zeros();
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;

        this->need_parameters = true;
    }

    SecMixedLayerNorm(RSSTensor<uint64_t> &weight, RSSTensor<uint64_t> &bias) : Layers<uint64_t, uint32_t>("LayerNorm")
    {
        normalized_shape = weight.shape[0];
        weight.allocate({normalized_shape});
        bias.allocate({normalized_shape});
        this->weight = weight;
        this->bias = bias;
        this->parameters["weight"] = weight;
        this->parameters["bias"] = bias;
    }

    void set_weight(RSSTensor<uint64_t> &weight)
    {
        always_assert(this->normalized_shape == weight.shape[0]);
        this->weight = weight;
        this->parameters["weight"] = weight;
    }

    void set_bias(RSSTensor<uint64_t> &bias)
    {
        always_assert(this->normalized_shape == bias.shape[0]);
        this->bias = bias;
        this->parameters["bias"] = bias;
    }

    void init_param()
    {
        weight = this->parameters["weight"];
        bias = this->parameters["bias"];
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters) override
    {
        RSSTensor<uint64_t> res(x.shape);

        std::vector<uint32_t> sum_shape = x.shape;
        uint32_t dim_size = sum_shape.back();
        always_assert(dim_size == this->normalized_shape); // can only normalize the last dimension
        sum_shape.pop_back();

        RSSTensor<uint64_t> x_sum(sum_shape);
        const uint32_t sum_size = x_sum.size();
#pragma omp parallel for
        for (int i = 0; i < sum_size; i++)
        {
            x_sum.first.data[i] = 0;
            x_sum.second.data[i] = 0;
            for (int j = 0; j < dim_size; j++)
            {
                x_sum.first.data[i] += x.first.data[i * dim_size + j];
                x_sum.second.data[i] += x.second.data[i * dim_size + j];
            }
        }

        rss_protocols::truncate(x_sum, normalized_shape, IS_MALICIOUS); // calculate mean

        RSSTensor<uint64_t> z(x.shape);
#pragma omp parallel for
        for (int i = 0; i < sum_size; i++)
        {
            for (int j = 0; j < dim_size; j++)
            {
                z.first.data[i * dim_size + j] = x.first.data[i * dim_size + j] - x_sum.first.data[i];
                z.second.data[i * dim_size + j] = x.second.data[i * dim_size + j] - x_sum.second.data[i];
            }
        }

        RSSTensor<uint64_t> z_square(x.shape), square_sum(sum_shape);
        rss_protocols::square(z, z_square, true, IS_MALICIOUS); // calculate z^2

#pragma omp parallel for
        for (int i = 0; i < sum_size; i++)
        {
            square_sum.first.data[i] = 0;
            square_sum.second.data[i] = 0;
            for (int j = 0; j < dim_size; j++)
            {
                square_sum.first.data[i] += z_square.first.data[i * dim_size + j];
                square_sum.second.data[i] += z_square.second.data[i * dim_size + j];
            }
        }

        rss_protocols::truncate(square_sum, normalized_shape, IS_MALICIOUS); // calculate variance

        RSSTensor<uint64_t> rsqrt_var(sum_shape), broadcast_rsqrt_var(x.shape), gamma(x.shape);
        RSSTensor<uint32_t> square_sum_32(square_sum.shape), rsqrt_var_32(sum_shape);

        // quatization
        rss_protocols::downcast(square_sum, square_sum_32);
        rss_protocols::rsqrt(square_sum_32, rsqrt_var_32, parameters, IS_MALICIOUS); // calculate 1 / sqrt(var)
        rss_protocols::upcast(rsqrt_var_32, rsqrt_var, Party3PC::getInstance().party_id, IS_MALICIOUS);

#pragma omp parallel for
        for (int i = 0; i < sum_size; i++)
        {
            for (int j = 0; j < dim_size; j++)
            {
                broadcast_rsqrt_var.first.data[i * dim_size + j] = rsqrt_var.first.data[i];
                broadcast_rsqrt_var.second.data[i * dim_size + j] = rsqrt_var.second.data[i];

                gamma.first.data[i * dim_size + j] = this->weight.first.data[j];
                gamma.second.data[i * dim_size + j] = this->weight.second.data[j];
            }
        }

        RSSTensor<uint64_t> z_norm(x.shape), z_norm_weighted(x.shape);
        rss_protocols::mul(z, broadcast_rsqrt_var, z_norm, true, IS_MALICIOUS); // calculate z_norm
        rss_protocols::mul(z_norm, gamma, z_norm_weighted, true, IS_MALICIOUS); // calculate gamma * z_norm

        // calculate gamma * z_norm + beta
#pragma omp parallel for
        for (int i = 0; i < sum_size; i++)
        {
            for (int j = 0; j < dim_size; j++)
            {
                res.first.data[i * dim_size + j] = z_norm_weighted.first.data[i * dim_size + j] + this->bias.first.data[j];
                res.second.data[i * dim_size + j] = z_norm_weighted.second.data[i * dim_size + j] + this->bias.second.data[j];
            }
        }

        return res;
    }
};

class SecMixedSelfAttention : public Layers<uint64_t, uint32_t>
{
public:
    uint32_t num_heads;
    uint32_t head_size;
    uint32_t all_head_size;
    Layers<uint64_t> *query, *key, *value;
    Layers<uint32_t> *softmax;

    SecMixedSelfAttention(TransformerConfig &config, const bool masked = false) : Layers<uint64_t, uint32_t>("SelfAttention"),
                                                                                  num_heads(config.num_attention_heads),
                                                                                  head_size(
                                                                                      config.hidden_size / config.num_attention_heads),
                                                                                  all_head_size(num_heads * head_size)
    {
        query = new SecLinear<uint64_t>(config.hidden_size, all_head_size);
        key = new SecLinear<uint64_t>(config.hidden_size, all_head_size);
        value = new SecLinear<uint64_t>(config.hidden_size, all_head_size);
        softmax = new SecSoftmax<uint32_t>(masked);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters) override
    {
        RSSTensor<uint64_t> mixed_query = (*this->query)(x);
        RSSTensor<uint64_t> mixed_key = (*this->key)(x);
        RSSTensor<uint64_t> mixed_value = (*this->value)(x);

        std::vector<uint32_t> new_shape = x.shape;
        new_shape.pop_back();
        new_shape.push_back(this->num_heads);
        new_shape.push_back(this->head_size);

        mixed_query.reshape(new_shape);
        mixed_key.reshape(new_shape);
        mixed_value.reshape(new_shape);

        RSSTensor<uint64_t> query_layer = mixed_query.permute({0, 2, 1, 3});
        RSSTensor<uint64_t> key_layer = mixed_key.permute({0, 2, 3, 1});
        RSSTensor<uint64_t> value_layer = mixed_value.permute({0, 2, 1, 3});

        RSSTensor<uint64_t> attention_scores(
            {query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], key_layer.shape[3]});
        rss_protocols::matMul(query_layer, key_layer, attention_scores, true, IS_MALICIOUS);
        rss_protocols::truncate(attention_scores, (uint32_t)sqrt((double)this->head_size), IS_MALICIOUS);

        RSSTensor<uint32_t> attention_scores_32(attention_scores.shape);
        rss_protocols::downcast(attention_scores, attention_scores_32);
        RSSTensor<uint32_t> attention_probs_32 = (*this->softmax)(attention_scores_32, parameters);

        RSSTensor<uint64_t> attention_probs(attention_probs_32.shape);
        rss_protocols::upcast(attention_probs_32, attention_probs, Party3PC::getInstance().party_id, IS_MALICIOUS);

        RSSTensor<uint64_t> context_layer({attention_probs.shape[0], attention_probs.shape[1], attention_probs.shape[2],
                                           value_layer.shape[3]});
        rss_protocols::matMul(attention_probs, value_layer, context_layer, true, IS_MALICIOUS);

        RSSTensor<uint64_t> res = context_layer.permute({0, 2, 1, 3});
        std::vector<uint32_t> res_shape = res.shape;
        res_shape.pop_back();
        res_shape.pop_back();
        res_shape.push_back(this->all_head_size);
        res.reshape(res_shape);

        return res;
    }
};

class SecMixedAttention : public Layers<uint64_t, uint32_t>
{
public:
    Layers<uint64_t, uint32_t> *self;
    Layers<uint64_t> *dense;

    SecMixedAttention(TransformerConfig &config, const bool masked = false) : Layers<uint64_t, uint32_t>(
                                                                                  "Attention")
    {
        self = new SecMixedSelfAttention(config, masked);
        dense = new SecLinear<uint64_t>(config.hidden_size, config.hidden_size);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters) override
    {
        RSSTensor<uint64_t> self_outputs = (*this->self)(x, parameters);
        return (*this->dense)(self_outputs);
    }
};

class SecMixedGPT2Attention : public Layers<uint64_t, uint32_t>
{
public:
    uint32_t num_heads;
    uint32_t head_size;
    uint32_t all_head_size;
    Layers<uint64_t> *c_attn, *c_proj;
    Layers<uint32_t> *softmax;

    SecMixedGPT2Attention(TransformerConfig &config) : Layers<uint64_t, uint32_t>("GPT2Attention"),
                                                       num_heads(config.num_attention_heads),
                                                       head_size(config.hidden_size / config.num_attention_heads),
                                                       all_head_size(num_heads * head_size)
    {
        c_attn = new SecLinear<uint64_t>(config.hidden_size, 3 * all_head_size);
        softmax = new SecSoftmax<uint32_t>(true);
        c_proj = new SecLinear<uint64_t>(config.hidden_size, all_head_size);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters) override
    {
        RSSTensor<uint64_t> qkv = (*this->c_attn)(x);
        RSSTensor<uint64_t> mixed_query(x.shape), mixed_key(x.shape), mixed_value(x.shape);

        uint32_t size = x.size(), last_dim = x.shape[x.shape.size() - 1];
        uint32_t other_dim = size / last_dim;
        // split, dim = -1
        uint32_t qkv_last_dim = qkv.shape[qkv.shape.size() - 1];
        uint32_t index, qkv_index;
        for (int i = 0; i < other_dim; i++)
        {
            for (int j = 0; j < last_dim; j++)
            {
                index = i * last_dim + j;
                qkv_index = i * qkv_last_dim + j;

                mixed_query.first.data[index] = qkv.first.data[qkv_index];
                mixed_query.second.data[index] = qkv.second.data[qkv_index];

                mixed_key.first.data[index] = qkv.first.data[qkv_index + last_dim];
                mixed_key.second.data[index] = qkv.second.data[qkv_index + last_dim];

                mixed_value.first.data[index] = qkv.first.data[qkv_index + last_dim * 2];
                mixed_value.second.data[index] = qkv.second.data[qkv_index + last_dim * 2];
            }
        }

        std::vector<uint32_t> new_shape = x.shape;
        new_shape.pop_back();
        new_shape.push_back(this->num_heads);
        new_shape.push_back(this->head_size);

        mixed_query.reshape(new_shape);
        mixed_key.reshape(new_shape);
        mixed_value.reshape(new_shape);

        RSSTensor<uint64_t> query_layer = mixed_query.permute({0, 2, 1, 3});
        RSSTensor<uint64_t> key_layer = mixed_key.permute({0, 2, 3, 1});
        RSSTensor<uint64_t> value_layer = mixed_value.permute({0, 2, 1, 3});

        RSSTensor<uint64_t> attention_scores(
            {query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], key_layer.shape[3]});
        rss_protocols::matMul(query_layer, key_layer, attention_scores, true, IS_MALICIOUS);
        rss_protocols::truncate(attention_scores, (uint32_t)sqrt((double)this->head_size), IS_MALICIOUS);

        RSSTensor<uint32_t> attention_scores_32(attention_scores.shape);
        rss_protocols::downcast(attention_scores, attention_scores_32);
        RSSTensor<uint32_t> attention_probs_32 = (*this->softmax)(attention_scores_32, parameters);

        RSSTensor<uint64_t> attention_probs(attention_probs_32.shape);
        rss_protocols::upcast(attention_probs_32, attention_probs, Party3PC::getInstance().party_id, IS_MALICIOUS);

        RSSTensor<uint64_t> context_layer({attention_probs.shape[0], attention_probs.shape[1], attention_probs.shape[2],
                                           value_layer.shape[3]});
        rss_protocols::matMul(attention_probs, value_layer, context_layer, true, IS_MALICIOUS);

        RSSTensor<uint64_t> attn_output = context_layer.permute({0, 2, 1, 3});
        std::vector<uint32_t> output_shape = attn_output.shape;
        output_shape.pop_back();
        output_shape.pop_back();
        output_shape.push_back(this->all_head_size);
        attn_output.reshape(output_shape);

        return (*this->c_proj)(attn_output);
    }
};

class SecMixedFFN : public Layers<uint64_t, uint16_t>
{
public:
    Layers<uint64_t> *dense0, *dense1;
    Layers<uint16_t> *act_fn;

    SecMixedFFN(TransformerConfig &config) : Layers<uint64_t, uint16_t>("FFN")
    {
        dense0 = new SecLinear<uint64_t>(config.hidden_size, config.intermediate_size);
        if (config.isReLUAct)
        {
            act_fn = new SecReLU<uint16_t>();
        }
        else
        {
            act_fn = new SecGELU<uint16_t>();
        }
        dense1 = new SecLinear<uint64_t>(config.intermediate_size, config.hidden_size);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint16_t> &parameters) override
    {
        RSSTensor<uint64_t> hidden_states = (*this->dense0)(x);
        RSSTensor<uint16_t> hidden_states_down(hidden_states.shape);
        rss_protocols::downcast(hidden_states, hidden_states_down);
        RSSTensor<uint16_t> ffn_output = (*this->act_fn)(hidden_states_down, parameters);
        rss_protocols::upcast(ffn_output, hidden_states, Party3PC::getInstance().party_id, IS_MALICIOUS);
        return (*this->dense1)(hidden_states);
    }
};

class SecMixedEncoderLayer : public Layers<uint64_t, uint32_t, uint16_t>
{
public:
    Layers<uint64_t, uint32_t> *attention, *LayerNorm1, *LayerNorm2;
    Layers<uint64_t, uint16_t> *ffn;

    SecMixedEncoderLayer(TransformerConfig &config) : Layers<uint64_t, uint32_t, uint16_t>("EncoderLayer")
    {
        attention = new SecMixedAttention(config);
        LayerNorm1 = new SecMixedLayerNorm(config.hidden_size);
        ffn = new SecMixedFFN(config);
        LayerNorm2 = new SecMixedLayerNorm(config.hidden_size);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v) override
    {
        RSSTensor<uint64_t> self_attention_output = (*this->attention)(x, parameters_u);
        rss_protocols::add(x, self_attention_output, self_attention_output);
        RSSTensor<uint64_t> attention_output = (*this->LayerNorm1)(self_attention_output, parameters_u);

        RSSTensor<uint64_t> ffn_output = (*this->ffn)(attention_output, parameters_v);

        rss_protocols::add(attention_output, ffn_output, ffn_output);
        return (*this->LayerNorm2)(ffn_output, parameters_u);
    }
};

class SecMixedDecoderLayer : public Layers<uint64_t, uint32_t, uint16_t>
{
public:
    Layers<uint64_t, uint32_t> *masked_attention, *LayerNorm0, *attention, *LayerNorm1, *LayerNorm2;
    Layers<uint64_t, uint16_t> *ffn;

    SecMixedDecoderLayer(TransformerConfig &config) : Layers<uint64_t, uint32_t, uint16_t>("DecoderLayer")
    {
        masked_attention = new SecMixedAttention(config, true);
        LayerNorm0 = new SecMixedLayerNorm(config.hidden_size);
        attention = new SecMixedAttention(config);
        LayerNorm1 = new SecMixedLayerNorm(config.hidden_size);
        ffn = new SecMixedFFN(config);
        LayerNorm2 = new SecMixedLayerNorm(config.hidden_size);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v) override
    {
        // TODO: input from previous layers
        RSSTensor<uint64_t> masked_attention_output = (*this->masked_attention)(x, parameters_u);
        rss_protocols::add(x, masked_attention_output, masked_attention_output);
        RSSTensor<uint64_t> masked_attention_output_norm = (*this->LayerNorm0)(masked_attention_output, parameters_u);

        RSSTensor<uint64_t> attention_output = (*this->attention)(masked_attention_output_norm, parameters_u);
        rss_protocols::add(masked_attention_output_norm, attention_output, attention_output);
        RSSTensor<uint64_t> attention_output_norm = (*this->LayerNorm1)(attention_output, parameters_u);

        RSSTensor<uint64_t> ffn_output = (*this->ffn)(attention_output_norm, parameters_v);
        rss_protocols::add(attention_output_norm, ffn_output, ffn_output);
        return (*this->LayerNorm2)(ffn_output, parameters_u);
    }
};

class SecMixedGPT2Layer : public Layers<uint64_t, uint32_t, uint16_t>
{
public:
    Layers<uint64_t> *LayerNorm0, *LayerNorm1;
    Layers<uint64_t, uint32_t> *masked_attention;
    Layers<uint64_t, uint16_t> *ffn;

    SecMixedGPT2Layer(TransformerConfig &config) : Layers<uint64_t, uint32_t, uint16_t>("GPT2Layer")
    {
        LayerNorm0 = new SecLayerNorm<uint64_t>(config.hidden_size);
        masked_attention = new SecMixedGPT2Attention(config);
        LayerNorm1 = new SecLayerNorm<uint64_t>(config.hidden_size);
        ffn = new SecMixedFFN(config);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint64_t> &parameters_t, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v) override
    {
        RSSTensor<uint64_t> ln0_out = (*this->LayerNorm0)(x, parameters_t);
        RSSTensor<uint64_t> attn_out = (*this->masked_attention)(ln0_out, parameters_u);
        rss_protocols::add(x, attn_out, attn_out);
        RSSTensor<uint64_t> ln1_out = (*this->LayerNorm1)(attn_out, parameters_t);
        RSSTensor<uint64_t> ffn_out = (*this->ffn)(ln1_out, parameters_v);
        rss_protocols::add(attn_out, ffn_out, ffn_out);
        return ffn_out;
    }
};

class SecMixedEncoder : public Layers<uint64_t, uint32_t, uint16_t>
{
public:
    TransformerConfig config;
    std::vector<SecMixedEncoderLayer> layers;

    SecMixedEncoder(TransformerConfig &config) : Layers<uint64_t, uint32_t, uint16_t>("Encoder"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v) override
    {
        RSSTensor<uint64_t> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters_u, parameters_v);
        }
        return hidden_states;
    }
};

class SecMixedDecoder : public Layers<uint64_t, uint32_t, uint16_t>
{
public:
    TransformerConfig config;
    std::vector<SecMixedDecoderLayer> layers;

    SecMixedDecoder(TransformerConfig &config) : Layers<uint64_t, uint32_t, uint16_t>("Decoder"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v) override
    {
        RSSTensor<uint64_t> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters_u, parameters_v);
        }
        return hidden_states;
    }
};

class SecMixedGPT2 : public Layers<uint64_t, uint32_t, uint16_t>
{
public:
    TransformerConfig config;
    std::vector<SecMixedGPT2Layer> layers;
    Layers<uint64_t, uint32_t> *ln_f;

    SecMixedGPT2(TransformerConfig &config) : Layers<uint64_t, uint32_t, uint16_t>("GPT2"), config(config)
    {
        for (int i = 0; i < config.num_layers; i++)
        {
            layers.emplace_back(config);
        }
        ln_f = new SecMixedLayerNorm(config.hidden_size);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint64_t> &parameters_t, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v) override
    {
        RSSTensor<uint64_t> hidden_states(x.shape);
        hidden_states = x;
        for (int i = 0; i < this->config.num_layers; i++)
        {
            hidden_states = this->layers[i](hidden_states, parameters_t, parameters_u, parameters_v);
        }
        return (*this->ln_f)(hidden_states, parameters_u);
    }
};

class SecBertClassifier : public Layers<uint64_t, uint32_t, uint16_t>
{
public:
    TransformerConfig config;
    Layers<uint64_t, uint32_t, uint16_t> *bert;
    Layers<uint64_t> *classifier;

    SecBertClassifier(TransformerConfig &config, int num_labels = 2) : Layers<uint64_t, uint32_t, uint16_t>("BertClassifier"), config(config)
    {
        bert = new SecMixedEncoder(config);
        classifier = new SecLinear<uint64_t>(config.hidden_size, num_labels);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v) override
    {
        RSSTensor<uint64_t> bert_outputs(x.shape);
        bert_outputs = (*this->bert)(x, parameters_u, parameters_v);

        uint32_t batch_size = bert_outputs.shape[0];
        uint32_t seq_len = bert_outputs.shape[1];
        uint32_t hidden_size = config.hidden_size;
        uint32_t one_value_size = seq_len * hidden_size;
        RSSTensor<uint64_t> outputs({batch_size, hidden_size});
        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < hidden_size; j++)
            {
                outputs.first.data[i * hidden_size + j] = bert_outputs.first.data[i * one_value_size + j];
                outputs.second.data[i * hidden_size + j] = bert_outputs.second.data[i * one_value_size + j];
            }
        }
        return (*this->classifier)(outputs);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint64_t> &parameters_t, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v) override
    {
        return forward(x, parameters_u, parameters_v);
    }
};

class SecBertClassifier64 : public Layers<uint64_t>
{
public:
    TransformerConfig config;
    Layers<uint64_t> *bert;
    Layers<uint64_t> *classifier;

    SecBertClassifier64(TransformerConfig &config, int num_labels = 2) : Layers<uint64_t>("BertClassifier"), config(config)
    {
        bert = new SecEncoder<uint64_t>(config);
        classifier = new SecLinear<uint64_t>(config.hidden_size, num_labels);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        RSSTensor<uint64_t> bert_outputs(x.shape);
        bert_outputs = (*this->bert)(x, parameters);

        uint32_t batch_size = bert_outputs.shape[0];
        uint32_t seq_len = bert_outputs.shape[1];
        uint32_t hidden_size = config.hidden_size;
        uint32_t one_value_size = seq_len * hidden_size;
        RSSTensor<uint64_t> outputs({batch_size, hidden_size});
        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < hidden_size; j++)
            {
                outputs.first.data[i * hidden_size + j] = bert_outputs.first.data[i * one_value_size + j];
                outputs.second.data[i * hidden_size + j] = bert_outputs.second.data[i * one_value_size + j];
            }
        }
        return (*this->classifier)(outputs);
    }
};

class SecGPT2LMHead : public Layers<uint64_t, uint32_t, uint16_t>
{
public:
    TransformerConfig config;
    Layers<uint64_t, uint32_t, uint16_t> *gpt2;
    Layers<uint64_t> *lm_head;

    SecGPT2LMHead(TransformerConfig &config, int vocab_size = 50257) : Layers<uint64_t, uint32_t, uint16_t>("SecGPT2LMHead"), config(config)
    {
        gpt2 = new SecMixedGPT2(config);
        lm_head = new SecLinear<uint64_t>(config.hidden_size, vocab_size);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint64_t> &parameters_t, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v) override
    {
        RSSTensor<uint64_t> outputs(x.shape);
        outputs = (*this->gpt2)(x, parameters_t, parameters_u, parameters_v);
        return (*this->lm_head)(outputs);
    }
};

class SecGPT2LMHead64 : public Layers<uint64_t>
{
public:
    TransformerConfig config;
    Layers<uint64_t> *gpt2;
    Layers<uint64_t> *lm_head;

    SecGPT2LMHead64(TransformerConfig &config, int vocab_size = 50257) : Layers<uint64_t>("SecGPT2LMHead"), config(config)
    {
        gpt2 = new SecGPT2<uint64_t>(config);
        lm_head = new SecLinear<uint64_t>(config.hidden_size, vocab_size);
    }

    RSSTensor<uint64_t> forward(RSSTensor<uint64_t> &x, Parameters<uint64_t> &parameters) override
    {
        RSSTensor<uint64_t> outputs(x.shape);
        outputs = (*this->gpt2)(x, parameters);
        return (*this->lm_head)(outputs);
    }
};

template <typename T>
void utils::masked_softmax(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    always_assert(x.shape[x.shape.size() - 1] == x.shape[x.shape.size() - 2]);

    std::vector<uint32_t> sum_shape = x.shape;
    uint32_t dim_size = sum_shape.back();
    sum_shape.pop_back();
    uint32_t second_dim = sum_shape.back();
    uint32_t mask_size = dim_size * second_dim;
    uint32_t other_dim = x.size() / mask_size;

    // mask: use the first value to replace triu
    int index;
    for (int i = 0; i < other_dim; i++)
    {
        for (int j = 0; j < second_dim; j++)
        {
            for (int k = 0; k < dim_size; k++)
            {
                if (j < k)
                {
                    index = i * mask_size + j * dim_size;
                    x.first.data[index + k] = x.first.data[index];
                    x.second.data[index + k] = x.second.data[index];
                }
            }
        }
    }

    RSSTensor<T> x_max(sum_shape), delta(x.shape);
    uint32_t common_size = x_max.size();
    rss_protocols::max_last_dim(x, x_max, parameter, malicious);

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
    rss_protocols::neg_exp(delta, exp_x, parameter, malicious);
    RSSTensor<T> sum(sum_shape);

    // only need to sum the tril
#pragma omp parallel for
    for (int i = 0; i < other_dim; i++)
    {
        for (int j = 0; j < second_dim; j++)
        {
            index = i * second_dim + j;

            sum.first.data[index] = 0;
            sum.second.data[index] = 0;

            for (int k = 0; k < dim_size; k++)
            {
                if (j >= k)
                {
                    sum.first.data[index] += exp_x.first.data[index * dim_size + k];
                    sum.second.data[index] += exp_x.second.data[index * dim_size + k];
                }
            }
        }
    }

    rss_protocols::inv(sum, sum, parameter, malicious);
    RSSTensor<T> broadcast_sum(x.shape);
#pragma omp parallel
    for (int i = 0; i < sum.size(); i++)
    {
        for (int j = 0; j < dim_size; j++)
        {
            broadcast_sum.first.data[i * dim_size + j] = sum.first.data[i];
            broadcast_sum.second.data[i * dim_size + j] = sum.second.data[i];
        }
    }
    rss_protocols::mul(exp_x, broadcast_sum, res, true, malicious);
    mask(res);
}

template <typename T>
RSSTensor<T> utils::img2col_pool(RSSTensor<T> &x, const int kernel_size, const int stride, const int pad)
{
    always_assert(x.shape.size() == 4);
    uint32_t batch = x.shape[0], channel = x.shape[1], height_origin = x.shape[2], width_origin = x.shape[3];

    const uint32_t height_padded = height_origin + 2 * pad;
    const uint32_t width_padded = width_origin + 2 * pad;

    const uint32_t height_out = (height_padded - kernel_size) / stride + 1;
    const uint32_t width_out = (width_padded - kernel_size) / stride + 1;

    const uint32_t out_size = height_out * width_out;
    const uint32_t k_square = kernel_size * kernel_size;
    RSSTensor<T> output({batch, channel, out_size, k_square});

    const uint32_t origin_shape = height_origin * width_origin;
    const uint32_t channel_origin_shape = channel * origin_shape;
    const uint32_t out_shape = out_size * k_square;
    const uint32_t channel_out_shape = channel * out_shape;

    for (int b = 0; b < batch; ++b)
    {
        const uint32_t b_offset = b * channel_origin_shape;
        const uint32_t b_out_offset = b * channel_out_shape;

        for (int kh = 0; kh < kernel_size; ++kh)
        {
            for (int kw = 0; kw < kernel_size; ++kw)
            {
                const uint32_t kernel_element = kh * kernel_size + kw;

                for (int c = 0; c < channel; ++c)
                {
                    const uint32_t c_input_offset = b_offset + c * origin_shape;
                    const uint32_t c_output_offset = b_out_offset + c * out_shape + kernel_element;

                    for (int h = 0; h < height_out; ++h)
                    {
                        const int h_start = h * stride - pad + kh;
                        const uint32_t h_out_base = h * width_out;
                        const bool h_valid = (h_start >= 0) && (h_start < height_origin);

                        for (int w = 0; w < width_out; ++w)
                        {
                            const int w_start = w * stride - pad + kw;

                            const uint32_t output_idx = c_output_offset + (h_out_base + w) * k_square;

                            if (h_valid && (w_start >= 0) && (w_start < width_origin))
                            {
                                const uint32_t input_idx = c_input_offset + h_start * width_origin + w_start;
                                output.first.data[output_idx] = x.first.data[input_idx];
                                output.second.data[output_idx] = x.second.data[input_idx];
                            }
                            else
                            {
                                output.first.data[output_idx] = T(0);
                                output.second.data[output_idx] = T(0);
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}

template <typename T>
RSSTensor<T> utils::img2col_conv(RSSTensor<T> &x, const int kernel_size, const int stride, const int pad)
{
    always_assert(x.shape.size() == 4);
    const uint32_t batch = x.shape[0];
    const uint32_t channel = x.shape[1];
    const uint32_t height_origin = x.shape[2];
    const uint32_t width_origin = x.shape[3];

    const uint32_t height_padded = height_origin + 2 * pad;
    const uint32_t width_padded = width_origin + 2 * pad;

    const uint32_t height_out = (height_padded - kernel_size) / stride + 1;
    const uint32_t width_out = (width_padded - kernel_size) / stride + 1;

    const uint32_t out_size = height_out * width_out;
    const uint32_t k_square = kernel_size * kernel_size;

    RSSTensor<T> output({batch, out_size, channel * k_square});

    // precompute steps
    const uint32_t batch_stride = out_size * channel * k_square;
    const uint32_t out_dim_stride = channel * k_square;

#pragma omp parallel for collapse(3) schedule(static)
    for (int b = 0; b < batch; ++b)
    {
        for (int h = 0; h < height_out; ++h)
        {
            for (int w = 0; w < width_out; ++w)
            {
                const uint32_t pos = h * width_out + w;
                const uint32_t pos_offset = b * batch_stride + pos * out_dim_stride;

                for (int kh = 0; kh < kernel_size; ++kh)
                {
                    const int h_start = h * stride - pad + kh;
                    const bool h_valid = (h_start >= 0) && (h_start < height_origin);

                    for (int kw = 0; kw < kernel_size; ++kw)
                    {
                        const int w_start = w * stride - pad + kw;
                        const uint32_t kernel_element = kh * kernel_size + kw;

                        for (int c = 0; c < channel; ++c)
                        {
                            // [b][pos][c*k_square + kernel_element]
                            const uint32_t output_idx = pos_offset + c * k_square + kernel_element;

                            const uint32_t input_base = b * channel * height_origin * width_origin + c * height_origin * width_origin;

                            if (h_valid && (w_start >= 0) && (w_start < width_origin))
                            {
                                const uint32_t input_idx = input_base + h_start * width_origin + w_start;
                                output.first.data[output_idx] = x.first.data[input_idx];
                                output.second.data[output_idx] = x.second.data[input_idx];
                            }
                            else
                            {
                                output.first.data[output_idx] = T(0);
                                output.second.data[output_idx] = T(0);
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}

#endif // LAYERS_H
