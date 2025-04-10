#include "party3pc.h"
#include "rss_protocols.h"
#include "layers.h"
#include "llmConfig.h"
#include "params.h"
#include "test_base.h"

static uint32_t batch_size = 1;
static bool print_res = false;
static int loop = print_res ? 1 : 3;
static int warmup_rounds = print_res ? 0 : 1;

static void tf_test(int party_id);

static void convert_test_sim(int party_id);

static void tf_test_64(int party_id);

static void bert_test(int party_id);

static void gpt2_test(int party_id);

static void gpt2_xl_test(int party_id);

void LLM3pcTest::main_test(int party_id)
{
    tf_test(party_id);
    convert_test_sim(party_id);
    tf_test_64(party_id);
    bert_test(party_id);
    gpt2_test(party_id);
    // gpt2_xl_test(party_id);
}

static void share_data(RSSTensor<uint64_t> &x_share)
{
    Tensor<uint64_t> x(x_share.shape);
    x.rand(-5, 5);

    Tensor<uint64_t>::mul(x, x, 1 << FLOAT_PRECISION_64);

    rss_protocols::share(x, x_share);
}

static void recv_share(RSSTensor<uint64_t> &x_share)
{
    rss_protocols::recv_shares_from(0, x_share);
}

static void mixed_ffn_forward(RSSTensor<uint64_t> &x, Parameters<uint16_t> &parameters, Layers<uint64_t, uint16_t> &layer)
{
    layer(x, parameters);
}

static void mixed_layer_forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters, Layers<uint64_t, uint32_t> &layer)
{
    layer(x, parameters);
}

static void mixed_llm_forward(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v, Layers<uint64_t, uint32_t, uint16_t> &layer)
{
    layer(x, parameters_u, parameters_v);
    rss_protocols::macCheckSimulate<uint32_t>(MAC_SIZE);
}

static void mixed_llm_forward_ln(RSSTensor<uint64_t> &x, Parameters<uint32_t> &parameters_u, Parameters<uint16_t> &parameters_v,
                                 Layers<uint64_t, uint32_t, uint16_t> &layer, Layers<uint64_t, uint32_t> &layer_norm)
{
    layer_norm(x, parameters_u);
    layer(x, parameters_u, parameters_v);
    rss_protocols::macCheckSimulate<uint32_t>(MAC_SIZE);
}

static void mixed_gpt2_forward(RSSTensor<uint64_t> &x, Parameters<uint64_t> &parameters_t, Parameters<uint32_t> &parameters_u,
                               Parameters<uint16_t> &parameters_v, Layers<uint64_t, uint32_t, uint16_t> &layer)
{
    layer(x, parameters_t, parameters_u, parameters_v);
    rss_protocols::macCheckSimulate<uint32_t>(MAC_SIZE);
}

static void layer_forward(RSSTensor<uint64_t> &x, Parameters<uint64_t> &parameters, Layers<uint64_t> &layer)
{
    layer(x, parameters);
}

static void llm_forward(RSSTensor<uint64_t> &x, Parameters<uint64_t> &parameters, Layers<uint64_t> &layer)
{
    layer(x, parameters);
    rss_protocols::macCheckSimulate<uint64_t>(MAC_SIZE);
}

void tf_test(int party_id)
{
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig tf_config("transformer", 64);
    Layers<uint64_t, uint32_t> *attention = new SecMixedAttention(tf_config);
    Layers<uint64_t, uint32_t> *masked_attention = new SecMixedAttention(tf_config, true);
    Layers<uint64_t, uint16_t> *ffn = new SecMixedFFN(tf_config);
    Layers<uint64_t, uint32_t> *layer_norm = new SecMixedLayerNorm(tf_config.hidden_size);
    Layers<uint64_t, uint32_t, uint16_t> *encoder = new SecMixedEncoder(tf_config);
    Layers<uint64_t, uint32_t, uint16_t> *decoder = new SecMixedDecoder(tf_config);

    std::vector<uint32_t> x_shape = {batch_size, tf_config.seq_len, tf_config.hidden_size};

    omp_set_num_threads(1);
    RSSTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint32_t> parameters_u;
    parameters_u.init_all();
    Parameters<uint16_t> parameters_v;
    parameters_v.init_pc_cmp();
    parameters_v.init_gelu();

    test_func<Party3PC>(mixed_layer_forward, "attention", loop, warmup_rounds, x_share, parameters_u, (*attention));
    test_func<Party3PC>(mixed_layer_forward, "masked attention", loop, warmup_rounds, x_share, parameters_u, (*masked_attention));
    test_func<Party3PC>(mixed_ffn_forward, "ffn", loop, warmup_rounds, x_share, parameters_v, (*ffn));
    test_func<Party3PC>(mixed_layer_forward, "layernorm", loop, warmup_rounds, x_share, parameters_u, (*layer_norm));
    rss_protocols::macCheckSimulate<uint32_t>(MAC_SIZE);
    test_func<Party3PC>(mixed_llm_forward, "encoder", loop, warmup_rounds, x_share, parameters_u, parameters_v, (*encoder));
    test_func<Party3PC>(mixed_llm_forward, "decoder", loop, warmup_rounds, x_share, parameters_u, parameters_v, (*decoder));
}

void convert_encoder(int party_id, RSSTensor<uint64_t> &attn, RSSTensor<uint64_t> &ffn, RSSTensor<uint64_t> &ln)
{
    // encoder: 6 layers, 1 attn, 1 ffn, 2 layerNorm
    for (int i = 0; i < 6; i++)
    {
        RSSTensor<uint32_t> attn_down(attn.shape);
        rss_protocols::downcast(attn, attn_down);
        rss_protocols::upcast(attn_down, attn, party_id, IS_MALICIOUS);

        RSSTensor<uint32_t> ln_down1(ln.shape);
        rss_protocols::downcast(ln, ln_down1);
        rss_protocols::upcast(ln_down1, ln, party_id, IS_MALICIOUS);

        RSSTensor<uint16_t> ffn_down(ffn.shape);
        rss_protocols::downcast(ffn, ffn_down);
        rss_protocols::upcast(ffn_down, ffn, party_id, IS_MALICIOUS);

        RSSTensor<uint32_t> ln_down2(ln.shape);
        rss_protocols::downcast(ln, ln_down2);
        rss_protocols::upcast(ln_down2, ln, party_id, IS_MALICIOUS);
    }
}

void convert_decoder(int party_id, RSSTensor<uint64_t> &attn, RSSTensor<uint64_t> &ffn, RSSTensor<uint64_t> &ln)
{
    // decoder: 6 layers, 1 attn, 1 masked attn, 1 ffn, 3 layerNorm
    for (int i = 0; i < 6; i++)
    {
        RSSTensor<uint32_t> m_attn_down(attn.shape);
        rss_protocols::downcast(attn, m_attn_down);
        rss_protocols::upcast(m_attn_down, attn, party_id, IS_MALICIOUS);

        RSSTensor<uint32_t> ln_down0(ln.shape);
        rss_protocols::downcast(ln, ln_down0);
        rss_protocols::upcast(ln_down0, ln, party_id, IS_MALICIOUS);

        RSSTensor<uint32_t> attn_down(attn.shape);
        rss_protocols::downcast(attn, attn_down);
        rss_protocols::upcast(attn_down, attn, party_id, IS_MALICIOUS);

        RSSTensor<uint32_t> ln_down1(ln.shape);
        rss_protocols::downcast(ln, ln_down1);
        rss_protocols::upcast(ln_down1, ln, party_id, IS_MALICIOUS);

        RSSTensor<uint16_t> ffn_down(ffn.shape);
        rss_protocols::downcast(ffn, ffn_down);
        rss_protocols::upcast(ffn_down, ffn, party_id, IS_MALICIOUS);

        RSSTensor<uint32_t> ln_down2(ln.shape);
        rss_protocols::downcast(ln, ln_down2);
        rss_protocols::upcast(ln_down2, ln, party_id, IS_MALICIOUS);
    }
}

void convert_test_sim(int party_id)
{
    // total conversion cost of transformer(simulate test)
    RSSTensor<uint64_t> attn_up({1, 8, 64, 64}); // batch size, num_heads, input size, each head size
    RSSTensor<uint64_t> ffn_up({1, 64, 2048});   // batch size, input size, intermediate_size
    RSSTensor<uint64_t> ln_up({1, 64});          // batch size, input size

    attn_up.fill(1 * (1 << kFloat_Precision<uint64_t>));
    ffn_up.fill(1 * (1 << kFloat_Precision<uint64_t>));
    ln_up.fill(1 * (1 << kFloat_Precision<uint64_t>));

    test_func<Party3PC>(convert_encoder, "conversion in encoder", loop, warmup_rounds, party_id, attn_up, ffn_up, ln_up);
    test_func<Party3PC>(convert_decoder, "conversion in decoder", loop, warmup_rounds, party_id, attn_up, ffn_up, ln_up);
}

void tf_test_64(int party_id)
{
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig tf_config("transformer", 64);
    Layers<uint64_t> *attention = new SecAttention<uint64_t>(tf_config);
    Layers<uint64_t> *masked_attention = new SecAttention<uint64_t>(tf_config, true);
    Layers<uint64_t> *ffn = new SecFFN<uint64_t>(tf_config);
    Layers<uint64_t> *layer_norm = new SecLayerNorm<uint64_t>(tf_config.hidden_size);
    Layers<uint64_t> *encoder = new SecEncoder<uint64_t>(tf_config);
    Layers<uint64_t> *decoder = new SecDecoder<uint64_t>(tf_config);

    std::vector<uint32_t> x_shape = {batch_size, tf_config.seq_len, tf_config.hidden_size};

    omp_set_num_threads(1);
    RSSTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters;
    parameters.init_all();

    test_func<Party3PC>(layer_forward, "attention 64", loop, warmup_rounds, x_share, parameters, (*attention));
    test_func<Party3PC>(layer_forward, "masked attention 64", loop, warmup_rounds, x_share, parameters, (*masked_attention));
    test_func<Party3PC>(layer_forward, "ffn 64", loop, warmup_rounds, x_share, parameters, (*ffn));
    test_func<Party3PC>(layer_forward, "layernorm 64", loop, warmup_rounds, x_share, parameters, (*layer_norm));
    rss_protocols::macCheckSimulate<uint64_t>(MAC_SIZE);
    test_func<Party3PC>(llm_forward, "encoder 64", loop, warmup_rounds, x_share, parameters, (*encoder));
    test_func<Party3PC>(llm_forward, "decoder 64", loop, warmup_rounds, x_share, parameters, (*decoder));
}

void bert_test(int party_id)
{
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig bert_config("bert-base", 128);
    Layers<uint64_t, uint32_t, uint16_t> *bert = new SecMixedEncoder(bert_config);

    std::vector<uint32_t> x_shape = {batch_size, bert_config.seq_len, bert_config.hidden_size};

    omp_set_num_threads(64);
    RSSTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint32_t> parameters_u;
    parameters_u.init_all();
    Parameters<uint16_t> parameters_v;
    parameters_v.init_pc_cmp();
    parameters_v.init_gelu();

    Layers<uint64_t, uint32_t> *layer_norm = new SecMixedLayerNorm(bert_config.hidden_size);
    test_func<Party3PC>(mixed_llm_forward_ln, "bert-base", loop, warmup_rounds, x_share, parameters_u, parameters_v, (*bert), (*layer_norm));
}

void gpt2_test(int party_id)
{
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig gpt2_config("gpt2", 64);
    Layers<uint64_t, uint32_t, uint16_t> *gpt2 = new SecMixedGPT2(gpt2_config);

    std::vector<uint32_t> x_shape = {batch_size, gpt2_config.seq_len, gpt2_config.hidden_size};

    omp_set_num_threads(1);
    RSSTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters_t;
    parameters_t.init_all();
    Parameters<uint32_t> parameters_u;
    parameters_u.init_all();
    Parameters<uint16_t> parameters_v;
    parameters_v.init_pc_cmp();
    parameters_v.init_gelu();

    test_func<Party3PC>(mixed_gpt2_forward, "gpt2", loop, warmup_rounds, x_share, parameters_t, parameters_u, parameters_v, (*gpt2));
}

void gpt2_xl_test(int party_id)
{
    Party3PC::getInstance().load_param<uint64_t>();
    TransformerConfig gpt2xl_config("gpt2-XL", 128);
    Layers<uint64_t, uint32_t, uint16_t> *gpt2 = new SecMixedGPT2(gpt2xl_config);

    std::vector<uint32_t> x_shape = {batch_size, gpt2xl_config.seq_len, gpt2xl_config.hidden_size};

    omp_set_num_threads(1);
    RSSTensor<uint64_t> x_share(x_shape);
    if (party_id == 0)
    {
        share_data(x_share);
    }
    else
    {
        recv_share(x_share);
    }

    Parameters<uint64_t> parameters_t;
    parameters_t.init_all();
    Parameters<uint32_t> parameters_u;
    parameters_u.init_all();
    Parameters<uint16_t> parameters_v;
    parameters_v.init_pc_cmp();
    parameters_v.init_gelu();

    test_func<Party3PC>(mixed_gpt2_forward, "gpt2-xl", loop, warmup_rounds, x_share, parameters_t, parameters_u, parameters_v, (*gpt2));
}