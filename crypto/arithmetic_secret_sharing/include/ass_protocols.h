#pragma once

#include "tensor.h"
#include "comms.h"

namespace ass_protocols
{
    template <typename T>
    void b2a(Tensor<T> &x, Peer &peer, int party_id)
    {
        uint32_t size = x.size();
        Tensor<T> r(x.shape);
        r.zero();

        Tensor<uint8_t> x_shift(x.shape), tmp(x.shape);
        for (int i = 0; i < size; i++)
        {
            x_shift.data[i] = (uint8_t)((x.data[i] + r.data[i]) % 2);
        }
        peer.send_batched_input(x_shift.data, size, 1);
        peer.recv_batched_input(tmp.data, size, 1);

        if (party_id == 0)
        {
            for (int i = 0; i < size; i++)
            {
                x.data[i] = r.data[i] - r.data[i] * (x_shift.data[i] ^ tmp.data[i]) * 2;
            }
        }
        else
        {
            T x_shift_restore;
            for (int i = 0; i < size; i++)
            {
                x_shift_restore = x_shift.data[i] ^ tmp.data[i];
                x.data[i] = x_shift_restore + r.data[i] - r.data[i] * x_shift_restore * 2;
            }
        }
    }

    /**
     * r > x
     * @tparam T
     * @param r_bits shared value of bits of r
     * @param x_shift plaintext value
     * @param res
     * @param peer
     * @param party_id
     */
    template <typename T>
    void privateCompare(Tensor<T> &r_bits, Tensor<T> &x_shift, Tensor<T> &res, Peer &peer, int party_id)
    {
        uint32_t size = x_shift.size();
        uint32_t long_size = r_bits.size();
        uint32_t bit_length = 8 * sizeof(T);
        uint32_t double_bit_length = 2 * bit_length;

        Tensor<T> c({long_size});
        T w, x_shift_bit, w_sum; // w = r_bits ^ x_bits

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {

            w_sum = 0;

            for (int j = 0; j < bit_length; j++)
            {
                x_shift_bit = (x_shift.data[i] >> (bit_length - 1 - j)) & 1;

                if (party_id == 0)
                {
                    if (x_shift_bit == 0)
                    {
                        w = r_bits.data[i * bit_length + j];
                    }
                    else
                    {
                        w = -r_bits.data[i * bit_length + j];
                    }

                    c.data[i * bit_length + j] = r_bits.data[i * bit_length + j] + w_sum;
                }
                else
                {
                    if (x_shift_bit == 0)
                    {
                        w = r_bits.data[i * bit_length + j];
                    }
                    else
                    {
                        w = 1 - r_bits.data[i * bit_length + j];
                    }

                    c.data[i * bit_length + j] = r_bits.data[i * bit_length + j] - x_shift_bit + w_sum + 1;
                }
                w_sum += w;
            }
        }

        Tensor<T> round1_r({long_size}), round2_r({size}), c_shift({long_size}), tmp({long_size});
        round1_r.zero();
        round2_r.zero();

        // round 1: judge whether c is 0
        Tensor<T>::add(c_shift, round1_r, c);

        peer.send_batched_input(c_shift.data, long_size, bit_length);
        peer.recv_batched_input(tmp.data, long_size, bit_length);

#pragma omp parallel for
        for (int i = 0; i < long_size; i++)
        {
            c_shift.data[i] = (c_shift.data[i] + tmp.data[i]) % double_bit_length;
        }

        Tensor<T> round1_real_table({double_bit_length});
        round1_real_table.zero();
        if (party_id == 0)
        {
            round1_real_table.data[0] = 1;
        }

        Tensor<T> c_sum({size});

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            c_sum.data[i] = 0;
            for (int j = 0; j < bit_length; j++)
            {
                c_sum.data[i] += round1_real_table.data[c_shift.data[i * bit_length + j]];
            }
        }

        Tensor<T> c_sum_shift({size}), tmp2({size});
        // round 2: judge whether c_sum is 0
        Tensor<T>::add(c_sum_shift, round2_r, c_sum);
        peer.send_batched_input(c_sum_shift.data, size, bit_length);
        peer.recv_batched_input(tmp2.data, size, bit_length);

        Tensor<T> round2_real_table({double_bit_length});
        round2_real_table.zero();
        if (party_id == 0)
        {
            round2_real_table.data[0] = 1;
        }

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res.data[i] = round2_real_table.data[(c_sum_shift.data[i] + tmp2.data[i]) % double_bit_length];
        }
    }
}
