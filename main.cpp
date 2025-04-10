#include <iostream>
#include <cstring>
#include <functional>
#include <map>
#include "tests/test_base.h"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <test_name> " << "<party_id>" << std::endl;
        return 1;
    }

    std::map<std::string, std::function<Test *()>> test_map = {
        {"rss", []()
         { return new RssTest(); }},
        {"bench", []()
         { return new RssBenchmarkTest(); }},
        {"cnn3pc", []()
         { return new CNN3pcTest(); }},
        {"llm3pc", []()
         { return new LLM3pcTest(); }},
        {"llmacc", []()
         { return new LLMAccTest(); }},
    };

    std::string test_name = argv[1];
    int party_id = std::stoi(argv[2]);

    if (test_map.find(test_name) != test_map.end())
    {
        Test *test = test_map[test_name]();
        test->run(party_id);
        delete test;
    }
    else
    {
        std::cerr << "Unknown test: " << test_name << std::endl;
        return 1;
    }

    return 0;
}