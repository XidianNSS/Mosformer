#include <functional>
#include "party3pc.h"
#include "rss_protocols.h"
#include "tensor.h"
#include "params.h"
#include "test_base.h"
#include "globals.h"

using namespace rss_protocols;

static uint32_t num = 100;
static bool print_res = false;
static int loop = print_res ? 1 : 10;
static int warm_up = print_res ? 0 : 5;
static std::vector<uint32_t> x_shape = {num, num};
static std::vector<uint32_t> y_shape = {num, num};
static std::vector<uint32_t> z_shape = {num, num};

template <typename T>
void add_test(RSSTensor<T> &, RSSTensor<T> &, RSSTensor<T> &);

template <typename T>
void mul_test(RSSTensor<T> &, RSSTensor<T> &, RSSTensor<T> &);

template <typename T>
void matmul_test(RSSTensor<T> &, RSSTensor<T> &, RSSTensor<T> &);

template <typename T>
void ge_test(RSSTensor<T> &, RSSTensor<T> &, RSSTensor<T> &, Parameters<T> &);

template <typename T>
void inv_test(RSSTensor<T> &, RSSTensor<T> &, Parameters<T> &);

template <typename T>
void rsqrt_test(RSSTensor<T> &, RSSTensor<T> &, Parameters<T> &);

template <typename T>
void softmax_test(RSSTensor<T> &, RSSTensor<T> &, Parameters<T> &);

template <typename T>
void drelu_test(RSSTensor<T> &, RSSTensor<T> &, Parameters<T> &);

template <typename T>
void relu_test(RSSTensor<T> &, RSSTensor<T> &, Parameters<T> &);

template <typename T>
void gelu_test(RSSTensor<T> &, RSSTensor<T> &, Parameters<T> &);

template <typename T>
void lut_test(RSSTensor<T> &, RSSTensor<T> &);

template <typename T>
void nexp_test(RSSTensor<T> &, RSSTensor<T> &, Parameters<T> &);

template <typename T>
void max_test(RSSTensor<T> &, RSSTensor<T> &, Parameters<T> &);

template <typename T>
void convert_test(RSSTensor<T> &, int party_id);

template <typename T>
static void share_data(RSSTensor<T> &x_share, RSSTensor<T> &y_share)
{
    static constexpr auto scale_bit = kFloat_Precision<T>;

    // init x and y
    Tensor<T> x(x_shape);
    x.rand(-5, 5);

    if (print_res)
        x.print();
    Tensor<T>::mul(x, x, 1 << scale_bit);

    srand(time(0));
    Tensor<T> y(y_shape);
    y.rand(0, 5);

    if (print_res)
        y.print();
    Tensor<T>::mul(y, y, 1 << scale_bit);

    // share x and y to party 1 and 2
    share(x, x_share);
    share(y, y_share);
}

template <typename T>
static void recv_share(RSSTensor<T> &x_share, RSSTensor<T> &y_share)
{
    recv_shares_from(0, x_share);
    recv_shares_from(0, y_share);
}

void RssTest::main_test(int party_id)
{
    omp_set_num_threads(1);
    RSSTensor<ringType> x_share(x_shape), y_share(y_shape);
    if (party_id == 0)
    {
        share_data(x_share, y_share);
    }
    else
    {
        recv_share(x_share, y_share);
    }

    Parameters<ringType> parameters;
    parameters.init_all();

    RSSTensor<ringType> z_share(z_shape);

    // test addition
    test_func<Party3PC>(add_test<ringType>, "addition", loop, warm_up, x_share, y_share, z_share);
    open_print(z_share, party_id, print_res);

    // test multiplication
    test_func<Party3PC>(mul_test<ringType>, "multiplication", loop, warm_up, x_share, y_share, z_share);
    open_print(z_share, party_id, print_res);

    // test matMul
    test_func<Party3PC>(matmul_test<ringType>, "matmul", loop, warm_up, x_share, y_share, z_share);
    open_print(z_share, party_id, print_res);

    // test greaterEqual
    test_func<Party3PC>(ge_test<ringType>, "greater equal", loop, warm_up, x_share, y_share, z_share, parameters);
    open_print(z_share, party_id, print_res);

    // test inverse
    test_func<Party3PC>(inv_test<ringType>, "inverse", loop, warm_up, x_share, z_share, parameters);
    open_print(z_share, party_id, print_res);

    // test rsqrt
    test_func<Party3PC>(rsqrt_test<ringType>, "rsqrt", loop, warm_up, x_share, z_share, parameters);
    open_print(z_share, party_id, print_res);

    // test softmax
    test_func<Party3PC>(softmax_test<ringType>, "softmax", loop, warm_up, x_share, z_share, parameters);
    open_print(z_share, party_id, print_res);

    // test relu
    test_func<Party3PC>(relu_test<ringType>, "relu", loop, warm_up, x_share, z_share, parameters);
    open_print(z_share, party_id, print_res);

    // test gelu
    test_func<Party3PC>(gelu_test<ringType>, "gelu", loop, warm_up, x_share, z_share, parameters);
    open_print(z_share, party_id, print_res);

    // test nexp
    test_func<Party3PC>(nexp_test<ringType>, "nexp", loop, warm_up, x_share, z_share, parameters);
    open_print(z_share, party_id, print_res);

    // test max
    RSSTensor<ringType> max_share({num});
    test_func<Party3PC>(max_test<ringType>, "max", loop, warm_up, x_share, max_share, parameters);
    open_print(max_share, party_id, print_res);

    // test type conversion
    test_func<Party3PC>(convert_test<ringType>, "type conversion", loop, warm_up, x_share, party_id);
}

void RssBenchmarkTest::main_test(int party_id)
{
    omp_set_num_threads(1);
    for (uint32_t num = 100; num < 1000000; num *= 10)
    {
        RSSTensor<ringType> x_share({num});
        if (party_id == 0)
        {
            Tensor<ringType> x({num});
            x.rand(-5, 5);
            Tensor<ringType>::mul(x, x, 1 << kFloat_Precision<ringType>);
            share(x, x_share);
        }
        else
        {
            recv_shares_from(0, x_share);
        }

        Parameters<ringType> parameters;
        parameters.init_all();

        RSSTensor<ringType> z_share({num});

        if (party_id == 0)
        {
            std::cout << "********************** num = " << num << " **********************" << std::endl;
        }
        // test drelu
        test_func<Party3PC>(drelu_test<ringType>, "drelu", loop, warm_up, x_share, x_share, parameters);

        // test relu
        test_func<Party3PC>(relu_test<ringType>, "relu", loop, warm_up, x_share, z_share, parameters);

        // test gelu
        test_func<Party3PC>(gelu_test<ringType>, "gelu", loop, warm_up, x_share, z_share, parameters);

        // test inverse
        test_func<Party3PC>(inv_test<ringType>, "inverse", loop, warm_up, x_share, z_share, parameters);

        // test rsqrt
        test_func<Party3PC>(rsqrt_test<ringType>, "rsqrt", loop, warm_up, x_share, z_share, parameters);

        // test lut
        test_func<Party3PC>(lut_test<ringType>, "lut", loop, warm_up, x_share, z_share);
    }
}

template <typename T>
void add_test(RSSTensor<T> &x_share, RSSTensor<T> &y_share, RSSTensor<T> &z_share)
{
    add(x_share, y_share, z_share);
}

template <typename T>
void mul_test(RSSTensor<T> &x_share, RSSTensor<T> &y_share, RSSTensor<T> &z_share)
{
    mul(x_share, y_share, z_share, true, IS_MALICIOUS);
}

template <typename T>
void matmul_test(RSSTensor<T> &x_share, RSSTensor<T> &y_share, RSSTensor<T> &z_share)
{
    matMul(x_share, y_share, z_share, true, IS_MALICIOUS);
}

template <typename T>
void ge_test(RSSTensor<T> &x_share, RSSTensor<T> &y_share, RSSTensor<T> &z_share, Parameters<T> &param)
{
    greaterEqual(x_share, y_share, z_share, param, true, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void inv_test(RSSTensor<T> &x_share, RSSTensor<T> &z_share, Parameters<T> &param)
{
    inv(x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void rsqrt_test(RSSTensor<T> &x_share, RSSTensor<T> &z_share, Parameters<T> &param)
{
    rsqrt(x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void softmax_test(RSSTensor<T> &x_share, RSSTensor<T> &z_share, Parameters<T> &param)
{
    softmax_forward(x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void drelu_test(RSSTensor<T> &x_share, RSSTensor<T> &z_share, Parameters<T> &param)
{
    nonNegative(x_share, z_share, param, true, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void relu_test(RSSTensor<T> &x_share, RSSTensor<T> &z_share, Parameters<T> &param)
{
    select(x_share, x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void gelu_test(RSSTensor<T> &x_share, RSSTensor<T> &z_share, Parameters<T> &param)
{
    gelu(x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void lut_test(RSSTensor<T> &x_share, RSSTensor<T> &z_share)
{
    uint32_t table_size = 1 << 12;
    LUT_Param<T> lut_param;
    lut_param.init(table_size);
    for (int i = 0; i < table_size; i++)
    {
        lut_param.table.data[i] = 0;
    }
    lut(x_share, z_share, lut_param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void nexp_test(RSSTensor<T> &x_share, RSSTensor<T> &z_share, Parameters<T> &param)
{
    neg_exp(x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void max_test(RSSTensor<T> &x_share, RSSTensor<T> &z_share, Parameters<T> &param)
{
    max_last_dim(x_share, z_share, param, IS_MALICIOUS);
    macCheckSimulate<T>(MAC_SIZE);
}

template <typename T>
void convert_test(RSSTensor<T> &x_share, int party_id)
{
    RSSTensor<uint64_t> upper_x(x_share.shape);
    RSSTensor<T> down_x(x_share.shape);

    upcast(x_share, upper_x, party_id, IS_MALICIOUS);
    open_print(upper_x, party_id, print_res);

    downcast(upper_x, down_x);
    open_print(down_x, party_id, print_res);
}