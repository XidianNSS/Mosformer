#include <iostream>
#include <cstring>
#include "party3pc.h"

Party3PC &Party3PC::getInstance()
{
    static Party3PC instance;
    return instance;
}

void Party3PC::connect(int party_id, int port, std::string next_ip, int next_port)
{
    always_assert(party_id == 0 || party_id == 1 || party_id == 2);
    this->party_id = party_id;
    if (party_id == 0 || party_id == 1)
    {
        this->peer_with_next = new Peer(next_ip, next_port);
        this->peer_with_pre = waitForPeer(port);
    }
    else
    {
        this->peer_with_pre = waitForPeer(port);
        this->peer_with_next = new Peer(next_ip, next_port);
    }
    this->next_party_id = (party_id + 1) % 3;
    this->pre_party_id = (party_id + 2) % 3;
}

void Party3PC::sync()
{
    if (party_id == 2)
    {
        this->peer_with_next->sync();
        this->peer_with_pre->sync();
    }
    else
    {
        this->peer_with_pre->sync();
        this->peer_with_next->sync();
    }
}

void Party3PC::close()
{
    this->peer_with_pre->close();
    this->peer_with_next->close();
}

void Party3PC::semi_honest()
{
    this->is_malicious_ = false;
}

void Party3PC::malicious()
{
    this->is_malicious_ = true;
}

bool Party3PC::is_malicious()
{
    return this->is_malicious_;
}

uint64_t Party3PC::bytes_sent()
{
    return this->peer_with_next->bytesSent() + this->peer_with_pre->bytesSent();
}

uint64_t Party3PC::bytes_received()
{
    return this->peer_with_next->bytesReceived() + this->peer_with_pre->bytesReceived();
}

uint64_t Party3PC::rounds_sent()
{
    return this->peer_with_next->roundsSent() + this->peer_with_pre->roundsSent();
}

uint64_t Party3PC::rounds_received()
{
    return this->peer_with_next->roundsReceived() + this->peer_with_pre->roundsReceived();
}

// template <typename T>
// void Party3PC<T>::mac_check()
// {
//     RSSTensor<T> value({static_cast<uint32_t>(mac_buffer.x.size())}), mac_value({
//                      static_cast<uint32_t>(mac_buffer.x.size())
//                  }), key(
//                      {static_cast<uint32_t>(mac_buffer.x.size())});

//     for (int i = 0; i < mac_buffer.x.size(); i++)
//     {
//         value.first.data[i] = mac_buffer.x[i].first;
//         value.second.data[i] = mac_buffer.x[i].second;
//         mac_value.first.data[i] = mac_buffer.mx[i].first;
//         mac_value.second.data[i] = mac_buffer.mx[i].second;
//         key.first.data[i] = mac_buffer.mac_key[i].first;
//         key.second.data[i] = mac_buffer.mac_key[i].second;
//     }

//     RSSProtocols::macCheck(value, mac_value, key, *this);

//     mac_buffer.x = std::vector<std::pair<T, T>>();
//     mac_buffer.mx = std::vector<std::pair<T, T>>();
//     mac_buffer.mac_key = std::vector<std::pair<T, T>>();
// }
