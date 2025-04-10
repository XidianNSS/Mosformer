#ifndef GLOBAL_H
#define GLOBAL_H

#include <string>
#include <cstdint>
#include <climits>

#define FLOAT_PRECISION_16 6
#define FLOAT_PRECISION_32 12
#define FLOAT_PRECISION_64 16
#define GELU_TABLE_PRECISION 6
#define SCALE_BASE 256
#define LOCAL_TEST true
#define LATER_CHECK true
#define IS_MALICIOUS true

typedef uint32_t ringType;
extern uint32_t MAC_SIZE;

const int kBitSize = (sizeof(ringType) * CHAR_BIT);
const ringType kLargestNeg = ((ringType)1 << (kBitSize - 1));
const ringType kMinusOne = (ringType)-1;

#if (LOCAL_TEST)
const std::string kP0_IP = "127.0.0.1";
const std::string kP1_IP = "127.0.0.1";
const std::string kP2_IP = "127.0.0.1";
#else
const std::string kP0_IP = "192.168.1.228";
const std::string kP1_IP = "192.168.1.227";
const std::string kP2_IP = "192.168.1.229";
#endif

const int kP0_Port = 18080;
const int kP1_Port = 18081;
const int kP2_Port = 18082;

template <typename T>
constexpr int kFloat_Precision = []{
    if constexpr (std::is_same_v<T, uint64_t>) {
        return FLOAT_PRECISION_64;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return FLOAT_PRECISION_32;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return FLOAT_PRECISION_16;
    } else {
        return 1;
    }
}();

double diff(timespec start, timespec end);

double convert_bytes(uint64_t bytes);

std::string convert_bytes_to_string(uint64_t bytes);

#endif // GLOBAL_H
