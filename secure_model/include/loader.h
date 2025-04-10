//
// Created by lqx on 25-1-13.
//

#ifndef LOADER_H
#define LOADER_H

#pragma once

#include <deque>
#include <unordered_map>
#include "cnpy/cnpy.h"
#include "replicated_secret_sharing.h"

template <typename T>
class Loader
{
private:
    std::deque<std::unordered_map<std::string, RSSTensor<T>>> buffer_stack;

public:
    bool is_buffer_empty = true;

    Loader() = default;

    Loader(const std::string &path)
    {
        cnpy::npz_t npz_data = cnpy::npz_load(path);
        auto it = npz_data.begin();
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
            T *data1 = new T[num_elements];
            std::copy(array1.data<T>(), array1.data<T>() + num_elements, data1);

            T *data2 = new T[num_elements];
            std::copy(array2.data<T>(), array2.data<T>() + num_elements, data2);

            std::vector<uint32_t> shapeu32(array1.shape.begin(), array1.shape.end());
            RSSTensor<T> data = RSSTensor<T>(Tensor<T>(data1, shapeu32), Tensor<T>(data2, shapeu32));

            if (name1.back() == 't')
            {
                buffer_stack.push_back({{"weight", data}});
            }
            else if (name1.back() == 's')
            {
                buffer_stack.back().insert({"bias", data});
            }
            ++it;
        }
        is_buffer_empty = false;
    }

    std::unordered_map<std::string, RSSTensor<T>> get_data()
    {
        always_assert(is_buffer_empty == false);
        std::unordered_map<std::string, RSSTensor<T>> data = buffer_stack.front();
        buffer_stack.pop_front();
        return data;
    }
};

#endif // LOADER_H
