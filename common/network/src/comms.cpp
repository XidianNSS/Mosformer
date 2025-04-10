#include <chrono>
#include "comms.h"
#include "my_assert.h"

SocketBuf::SocketBuf(std::string ip, int port, bool onlyRecv = false)
{
    this->t = BUF_SOCKET;
    std::cerr << "trying to connect with server..." << std::endl;
    {
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = inet_addr(ip.c_str());
        while (1)
        {
            recvsocket = socket(AF_INET, SOCK_STREAM, 0);
            if (recvsocket < 0)
            {
                perror("socket");
                exit(1);
            }
            if (connect(recvsocket, (struct sockaddr*)&addr, sizeof(addr)) == 0)
            {
                break;
            }
            ::close(recvsocket);
            usleep(1000);
        }
        const int one = 1;
        setsockopt(recvsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    sleep(1);
    if (!onlyRecv)
    {
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port + 3);
        addr.sin_addr.s_addr = inet_addr(ip.c_str());
        while (1)
        {
            sendsocket = socket(AF_INET, SOCK_STREAM, 0);
            if (sendsocket < 0)
            {
                perror("socket");
                exit(1);
            }
            if (connect(sendsocket, (struct sockaddr*)&addr, sizeof(addr)) == 0)
            {
                break;
            }
            ::close(sendsocket);
            usleep(1000);
        }
        const int one = 1;
        setsockopt(sendsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    std::cerr << "connected" << std::endl;
}

void SocketBuf::sync()
{
    char buf[1] = {1};
    send(sendsocket, buf, 1, 0);
    recv(recvsocket, buf, 1, MSG_WAITALL);
    bytesReceived += 1;
    bytesSent += 1;

    roundsSent += 1;
    roundsReceived += 1;

    always_assert(buf[0] == 1);
}

void SocketBuf::read(char* buf, int bytes)
{
    int recv_bytes = recv(recvsocket, (char*)buf, bytes, MSG_WAITALL);
    if (recv_bytes < 0)
    {
        perror("recv failed");
        printf("recv failed with error: %s\n", strerror(errno));
    }
    always_assert(bytes == recv_bytes);
    bytesReceived += bytes;
    roundsReceived += 1;
}

char* SocketBuf::read(int bytes)
{
    char* tmpBuf = new char[bytes];
    always_assert(bytes == recv(recvsocket, (char *)tmpBuf, bytes, MSG_WAITALL));
    bytesReceived += bytes;
    roundsReceived += 1;
    return tmpBuf;
}

void SocketBuf::write(char* buf, int bytes)
{
    always_assert(bytes == send(sendsocket, buf, bytes, 0));
    bytesSent += bytes;
    roundsSent += 1;
}

void SocketBuf::close()
{
    ::close(sendsocket);
    ::close(recvsocket);
}

void Peer::close()
{
    keyBuf->close();
    std::cout << "closed" << std::endl;
}

Peer* waitForPeer(int port)
{
    int sendsocket, recvsocket;
    std::cerr << "waiting for connection from client..." << std::endl;
    {
        struct sockaddr_in dest;
        struct sockaddr_in serv;
        socklen_t socksize = sizeof(struct sockaddr_in);
        memset(&serv, 0, sizeof(serv));
        serv.sin_family = AF_INET;
        serv.sin_addr.s_addr = htonl(INADDR_ANY); /* set our address to any interface */
        serv.sin_port = htons(port); /* set the server port number */
        int mysocket = socket(AF_INET, SOCK_STREAM, 0);
        int reuse = 1;
        setsockopt(mysocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse,
                   sizeof(reuse));
        if (::bind(mysocket, (struct sockaddr*)&serv, sizeof(struct sockaddr)) < 0)
        {
            perror("error: bind");
            exit(1);
        }
        if (listen(mysocket, 1) < 0)
        {
            perror("error: listen");
            exit(1);
        }
        sendsocket = accept(mysocket, (struct sockaddr*)&dest, &socksize);
        const int one = 1;
        setsockopt(sendsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        close(mysocket);
    }

    {
        struct sockaddr_in dest;
        struct sockaddr_in serv;
        socklen_t socksize = sizeof(struct sockaddr_in);
        memset(&serv, 0, sizeof(serv));
        serv.sin_family = AF_INET;
        serv.sin_addr.s_addr = htonl(INADDR_ANY); /* set our address to any interface */
        serv.sin_port = htons(port + 3); /* set the server port number */
        int mysocket = socket(AF_INET, SOCK_STREAM, 0);
        int reuse = 1;
        setsockopt(mysocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse,
                   sizeof(reuse));
        if (::bind(mysocket, (struct sockaddr*)&serv, sizeof(struct sockaddr)) < 0)
        {
            perror("error: bind");
            exit(1);
        }
        if (listen(mysocket, 1) < 0)
        {
            perror("error: listen");
            exit(1);
        }
        recvsocket = accept(mysocket, (struct sockaddr*)&dest, &socksize);
        const int one = 1;
        setsockopt(recvsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        close(mysocket);
    }

    std::cerr << "connected" << std::endl;
    return new Peer(sendsocket, recvsocket);
}

void Peer::send(char* g, int size, int bw)
{
    this->keyBuf->write(g, size * bw);
}

void Peer::recv(char* g, int size, int bw)
{
    this->keyBuf->read(g, size * bw);
}

void Peer::sync()
{
    this->keyBuf->sync();
}
