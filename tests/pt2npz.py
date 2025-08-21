import sys
import heapq
from collections import OrderedDict

import numpy as np
import torch

from NssMPC import RingTensor, ReplicatedSecretSharing

model_base_path = "../log/model_save/"
data_base_path = "../log/data_save/"


def dict_order_numbers():
    heap = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    while heap:
        current = heapq.heappop(heap)
        if len(current) == 6:
            yield int(current)
        elif len(current) < 6:
            for digit in '0123456789':
                next_str = current + digit
                heapq.heappush(heap, next_str)


itern = dict_order_numbers()


def share_model(model, dataset):
    model_params = torch.load(model_base_path + f"{model}_{dataset}.pt")

    model_params = OrderedDict({key: value for key, value in model_params.items()})  # need to delete some unused params

    shared_param_dict_list = [OrderedDict() for _ in range(3)]

    for name, param in model_params.items():
        ring_param = RingTensor.convert_to_ring(param)
        shares = ReplicatedSecretSharing.share(ring_param)
        for param_dict, share in zip(shared_param_dict_list, shares):
            i = next(itern)
            param_dict[str(i) + '0' + name[-1]] = share.item.tensor[0].cpu().numpy()
            param_dict[str(i) + '1' + name[-1]] = share.item.tensor[1].cpu().numpy()

    for i in range(3):
        np.savez(f"../log/model_shares/{model}_{dataset}_{i}.npz", **shared_param_dict_list[i])

    print("finish sharing model")


def share_data(dataset):
    data = torch.load(data_base_path + f"{dataset}.pt")  # need to preprocess the data before sharing

    shared_param_dict_list = [OrderedDict() for _ in range(3)]
    for i in range(data.shape[0]):
        data_i = data[i].unsqueeze(0)
        ring_param = RingTensor.convert_to_ring(data_i)
        shares = ReplicatedSecretSharing.share(ring_param)
        for param_dict, share in zip(shared_param_dict_list, shares):
            j = next(itern)
            param_dict[str(j) + '0'] = share.item.tensor[0].cpu().numpy()
            param_dict[str(j) + '1'] = share.item.tensor[1].cpu().numpy()

    for i in range(3):
        np.savez(f"../log/data_shares/{dataset}_{i}.npz", **shared_param_dict_list[i])

    print("finish sharing data")


def main():
    tasks = [
        ("Bert_base", "RTE"),
        ("Bert_base", "QNLI"),
        ("Bert_base", "STS-B"),
        ("GPT2", "Wiki")
    ]

    def show_help():
        print("\nUsage:")
        print("  python script.py [model] [dataset]")
        print("\nDescription:")
        print("  Run share_model and share_data for either all tasks (default)")
        print("  or for a specific <model, dataset> pair.")
        print("\nAvailable tasks:")
        for model, dataset in tasks:
            print(f"  {model:10s} {dataset}")
        print("\nOptions:")
        print("  -h, --help    Show this help message and exit\n")

    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        show_help()
        sys.exit(0)

    if len(sys.argv) == 3:
        # 用户指定模型和数据集
        model, dataset = sys.argv[1], sys.argv[2]
        share_model(model, dataset)
        share_data(dataset)
    elif len(sys.argv) == 1:
        # 默认运行所有任务
        for model, dataset in tasks:
            share_model(model, dataset)
            share_data(dataset)
    else:
        print("Invalid arguments. Use -h or --help for usage information.")
        sys.exit(1)

if __name__ == "__main__":
    main()


