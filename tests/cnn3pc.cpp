#include "party3pc.h"
#include "rss_protocols.h"
#include "layers.h"
#include "llmConfig.h"
#include "params.h"
#include "test_base.h"

static bool print_res = false;
static int loop = print_res ? 1 : 5;
static int warmup_rounds = print_res ? 0 : 3;

static void alexnet_test(int party_id);

static void resnet50_test(int party_id);

void CNN3pcTest::main_test(int party_id)
{
    alexnet_test(party_id);

    resnet50_test(party_id);
}

template <typename T>
static void share_data(RSSTensor<T> &x_share)
{
    static constexpr auto scale_bit = kFloat_Precision<T>;
    Tensor<T> x(x_share.shape);
    x.rand(-5, 5);

    Tensor<T>::mul(x, x, 1 << scale_bit);

    rss_protocols::share(x, x_share);
}

template <typename T>
static void recv_share(RSSTensor<T> &x_share)
{
    rss_protocols::recv_shares_from(0, x_share);
}

template <typename T>
static void nn_forward(RSSTensor<T> &x, Parameters<T> &parameters, Layers<T> &layer)
{
    layer(x, parameters);
    rss_protocols::macCheckSimulate<T>(MAC_SIZE);
}

void alexnet_test(int party_id)
{
    uint32_t batch_size = 128, channel = 3, height = 33, width = 33;  // shape of padded CIFAR 10
    Party3PC::getInstance().load_param<ringType>();
    Layers<ringType> *net = new SecAlexNet<ringType>();

    std::vector<uint32_t> x_shape = {batch_size, channel, height, width};

    omp_set_num_threads(1);
    RSSTensor<ringType> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<ringType> parameters;
    parameters.init_all();

    test_func<Party3PC>(nn_forward<ringType>, "AlexNet", loop, warmup_rounds, x_share, parameters, (*net));
}

void resnet50_test(int party_id)
{
    uint32_t batch_size = 1, channel = 3, height = 224, width = 224;  // shape of imageNet
    Party3PC::getInstance().load_param<ringType>();
    Layers<ringType> *net = new SecResNet50<ringType>();

    std::vector<uint32_t> x_shape = {batch_size, channel, height, width};

    omp_set_num_threads(1);
    RSSTensor<ringType> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<ringType> parameters;
    parameters.init_all();

    test_func<Party3PC>(nn_forward<ringType>, "ResNet50", loop, warmup_rounds, x_share, parameters, (*net));
}
