#include "globals.h"

uint32_t MAC_SIZE = 0;

double diff(timespec start, timespec end)
{
    timespec temp;

    if ((end.tv_nsec - start.tv_nsec) < 0)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp.tv_sec + (double)temp.tv_nsec / 1E9;
}

double convert_bytes(uint64_t bytes)
{
    if (bytes < 1024)
        return bytes;
    else if (bytes < 1024 * 1024)
        return bytes / 1024.0;
    else if (bytes < 1024 * 1024 * 1024)
        return bytes / (1024.0 * 1024.0);
    else
        return bytes / (1024.0 * 1024.0 * 1024.0);
}

std::string convert_bytes_to_string(uint64_t bytes)
{
    if (bytes < 1024)
        return std::to_string(bytes) + " B";
    else if (bytes < 1024 * 1024)
        return std::to_string(bytes / 1024.0) + " KB";
    else if (bytes < 1024 * 1024 * 1024)
        return std::to_string(bytes / (1024.0 * 1024.0)) + " MB";
    else
        return std::to_string(bytes / (1024.0 * 1024.0 * 1024.0)) + " GB";
}