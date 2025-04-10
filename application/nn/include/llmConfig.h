#ifndef LLMCONFIG_H
#define LLMCONFIG_H
#include <cstdint>
#include <stdexcept>
#include <string>

class TransformerConfig
{
public:
    uint32_t seq_len, hidden_size, intermediate_size, num_attention_heads, num_layers;
    bool isReLUAct;

    TransformerConfig() = default;

    TransformerConfig(uint32_t seq_len, uint32_t hidden_size, uint32_t intermediate_size, uint32_t num_attention_heads,
                      uint32_t num_layers, bool isReLUAct) : seq_len(seq_len), hidden_size(hidden_size),
                                                             intermediate_size(intermediate_size),
                                                             num_attention_heads(num_attention_heads),
                                                             num_layers(num_layers), isReLUAct(isReLUAct)
    {
    }

    TransformerConfig(const std::string &model_type, const int seq_len) : seq_len(seq_len)
    {
        if (model_type == "bert-base" || model_type == "gpt2")
        {
            hidden_size = 768;
            num_attention_heads = 12;
            num_layers = 12;
            isReLUAct = false;
        }
        else if (model_type == "bert-large")
        {
            hidden_size = 1024;
            num_attention_heads = 16;
            num_layers = 24;
            isReLUAct = false;
        }
        else if (model_type == "gpt2-XL")
        {
            hidden_size = 1600;
            num_attention_heads = 25;
            num_layers = 48;
            isReLUAct = false;
        }
        else if (model_type == "transformer")
        {
            hidden_size = 512;
            num_attention_heads = 8;
            num_layers = 6;
            isReLUAct = true;
        }
        else
        {
            throw std::invalid_argument("Invalid model type");
        }
        intermediate_size = hidden_size * 4;
    }
};

#endif // LLMCONFIG_H
