#include <fstream>
#include <unordered_set>
#include "party3pc.h"
#include "rss_protocols.h"
#include "layers.h"
#include "llmConfig.h"
#include "params.h"
#include "test_base.h"

static std::unordered_set<std::string> bert_dataset = {"RTE", "QNLI", "STS-B"};
static std::unordered_set<std::string> gpt2_dataset = {"Wiki"};

static void llm_acc(int party_id, std::string dataset_name, bool is_mixed = true);

void LLMAccTest::main_test(int party_id)
{
    llm_acc(party_id, "RTE");
    llm_acc(party_id, "QNLI");
    llm_acc(party_id, "STS-B");
    llm_acc(party_id, "Wiki");

    llm_acc(party_id, "RTE", false);
    llm_acc(party_id, "QNLI", false);
    llm_acc(party_id, "STS-B", false);
    llm_acc(party_id, "Wiki", false);
}

static void llm_acc(int party_id, std::string dataset_name, bool is_mixed)
{
    bool is_bert = true;
    Layers<uint64_t> *llm_64;
    Layers<uint64_t, uint32_t, uint16_t> *llm;
    TransformerConfig *config;
    std::string param_base_path = "../log/model_shares/";
    int num_lables = dataset_name == "STS-B" ? 1 : 2;
    if (bert_dataset.find(dataset_name) != bert_dataset.end())
    {
        std::string param_path = param_base_path + "Bert_base_" + dataset_name + "_" + std::to_string(party_id) + ".npz";
        Party3PC::getInstance().load_param<uint64_t>(param_path);
        config = new TransformerConfig("bert-base", 128);
        if (is_mixed)
            llm = new SecBertClassifier(*config, num_lables);
        else
            llm_64 = new SecBertClassifier64(*config, num_lables);
    }
    else if (gpt2_dataset.find(dataset_name) != gpt2_dataset.end())
    {
        is_bert = false;
        std::string param_path = param_base_path + "GPT2_" + dataset_name + "_" + std::to_string(party_id) + ".npz";
        Party3PC::getInstance().load_param<uint64_t>(param_path);
        config = new TransformerConfig("gpt2", 64);
        if (is_mixed)
            llm = new SecMixedGPT2(*config);
        else
            llm_64 = new SecGPT2<uint64_t>(*config);
    }
    else
    {
        std::cerr << "Unsupported dataset" << std::endl;
        return;
    }

    Parameters<uint64_t> parameters_t;
    parameters_t.init_all();

    Parameters<uint32_t> parameters_u;
    parameters_u.init_all();

    Parameters<uint16_t> parameters_v;
    parameters_v.init_pc_cmp();
    parameters_v.init_gelu();

    omp_set_num_threads(64);
    std::string data_base_path = "../log/data_shares/";
    std::string data_path = data_base_path + dataset_name + "_" + std::to_string(party_id) + ".npz";

    // load data and inference
    cnpy::npz_t npz_data = cnpy::npz_load(data_path);
    auto it = npz_data.begin();

    std::string res_path = is_mixed ? "../log/" + dataset_name + "_res.log" : "../log/" + dataset_name + "_res_64.log";
    std::ofstream outFile;
    if (party_id == 0)
        outFile.open(res_path);

    while (it != npz_data.end())
    {
        std::string name1 = it->first;
        cnpy::NpyArray &array1 = it->second;
        ++it;
        std::string name2 = it->first;
        cnpy::NpyArray &array2 = it->second;

        std::vector<size_t> shape = array1.shape;
        size_t num_elements = 1;
        for (size_t dim : shape)
        {
            num_elements *= dim;
        }
        uint64_t *data1 = new uint64_t[num_elements];
        std::copy(array1.data<uint64_t>(), array1.data<uint64_t>() + num_elements, data1);

        uint64_t *data2 = new uint64_t[num_elements];
        std::copy(array2.data<uint64_t>(), array2.data<uint64_t>() + num_elements, data2);

        std::vector<uint32_t> shapeu32(array1.shape.begin(), array1.shape.end());
        RSSTensor<uint64_t> x_share(Tensor<uint64_t>(data1, shapeu32), Tensor<uint64_t>(data2, shapeu32));

        RSSTensor<uint64_t> res;
        if (is_mixed)
            res = (*llm)(x_share, parameters_t, parameters_u, parameters_v);
        else
            res = (*llm_64)(x_share, parameters_t);

        Tensor<uint64_t> real_res(res.shape);
        rss_protocols::restore(res, real_res);
        if (party_id == 0)
        {
            outFile << "[";
            if (is_bert)
            {
                if (num_lables == 1)
                {
                    for (int i = 0; i < real_res.size(); i++)
                        outFile << ((float)(long)real_res.data[i] / (1 << x_share.float_scale_bit)) << ", ";
                }
                else
                {
                    for (int i = 0; i < real_res.shape[0]; i++)
                    {
                        float prob_0 = (float)(long)real_res.data[i] / (1 << x_share.float_scale_bit);
                        float prob_1 = (float)(long)real_res.data[i + 1] / (1 << x_share.float_scale_bit);

                        int classified_res = prob_0 > prob_1 ? 0 : 1;
                        outFile << classified_res << ", ";
                    }
                }
            }
            else
            {
                for (int i = 0; i < real_res.size(); i++)
                {
                    outFile << ((float)(long)real_res.data[i] / (1 << x_share.float_scale_bit)) << ", ";
                }
            }
            outFile << "]" << std::endl;
        }
        ++it;
    }
    if (party_id == 0)
    {
        outFile.close();
        std::cout << "The inference results of " << dataset_name << " have been save to " << res_path << std::endl;
    }
}