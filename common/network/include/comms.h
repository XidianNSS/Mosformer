#ifndef COMMS_H
#define COMMS_H

#include <string>
#include <iostream>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>

typedef enum BufType
{
    BUF_FILE,
    BUF_SOCKET,
    BUF_MEM
} BufType;

class KeyBuf
{
public:
    uint64_t bytesSent = 0;
    uint64_t bytesReceived = 0;

    uint64_t roundsSent = 0;
    uint64_t roundsReceived = 0;

    BufType t;
    virtual void sync() {}
    virtual void read(char *buf, int bytes) = 0;
    virtual char *read(int bytes) = 0;
    virtual void write(char *buf, int bytes) = 0;
    virtual void close() = 0;
    bool isMem() { return t == BUF_MEM; }
};

class SocketBuf : public KeyBuf
{
public:
    int sendsocket, recvsocket;

    SocketBuf(std::string ip, int port, bool onlyRecv);
    SocketBuf(int sendsocket, int recvsocket) : sendsocket(sendsocket), recvsocket(recvsocket)
    {
        this->t = BUF_SOCKET;
    }
    void sync();
    void read(char *buf, int bytes);
    char *read(int bytes);
    void write(char *buf, int bytes);
    void close();
};

class Peer
{
public:
    KeyBuf *keyBuf;

    Peer(std::string ip, int port)
    {
        keyBuf = new SocketBuf(ip, port, false);
    }

    Peer(int sendsocket, int recvsocket)
    {
        keyBuf = new SocketBuf(sendsocket, recvsocket);
    }

    inline uint64_t bytesSent()
    {
        return keyBuf->bytesSent;
    }

    inline uint64_t bytesReceived()
    {
        return keyBuf->bytesReceived;
    }

    inline uint64_t roundsSent()
    {
        return keyBuf->roundsSent;
    }

    inline uint64_t roundsReceived()
    {
        return keyBuf->roundsReceived;
    }

    void inline zeroBytesSent()
    {
        keyBuf->bytesSent = 0;
    }

    void inline zeroBytesReceived()
    {
        keyBuf->bytesReceived = 0;
    }

    void sync();

    void close();

    void send(char *g, int size, int bw);

    void recv(char *g, int size, int bw);

    template<typename T>
    void send_input(const T &g)
    {
        char* buf = (char*)(&g);
        this->send(buf, 1, sizeof(T) * 8);
    }

    template<typename T>
    T recv_input()
    {
        char buf[sizeof(T)];
        this->keyBuf->read(buf, sizeof(T));
        T g = *(T*)buf;
        return g;
    }

    template<typename T>
    void send_batched_input(T *g, int size, int bw)
    {
        this->keyBuf->write((char*)g, size * sizeof(T));
    }

    template<typename T>
    void recv_batched_input(T *g, int size, int bw)
    {
        this->keyBuf->read((char*)g, sizeof(T) * size);
    }
};

Peer *waitForPeer(int port);

#endif // COMMS_H