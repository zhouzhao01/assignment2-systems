from cs336_basics.model import BasicsTransformerLM

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import timeit
import yaml

def get_model(
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,  
) -> BasicsTransformerLM:
    
    model = BasicsTransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta
    )

    return model

def cal_loss(x:Tensor, y:Tensor):
    batch_size, length, vocab_size = x.shape

    x = x.view(batch_size * length, vocab_size)
    y = y.flatten()

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(x, y)

    return loss

def generate_data(
        batch_size: int,
        length: int,
        vocab_size: int
) -> Tensor:
    
    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, length)
    )

    label = F.one_hot(x, num_classes=vocab_size)

    return x, label

def timer(
        model: nn.Module,
        warmup_steps: int,
        all_steps: int,
        if_backward: bool,

        batch_size: int,
        length: int,
        vocab_size: int,

        device
):  
    
    model.to(device)

    if warmup_steps >= 0:
        for w in range(warmup_steps):
            data, label = generate_data(batch_size, length, vocab_size)
            data.to(device)

            rlt = model(data)
            loss = cal_loss(rlt, data)
            loss.backward()

    time = timeit.timeit()
    time_forward, time_backward = 0.0, 0.0
    for step in range(all_steps):
        data, label = generate_data(batch_size, length, vocab_size)
        data.to(device)

        rlt = model(data)
        time_forward = time_forward + timeit.timeit() - time
        time = timeit.timeit() 

        loss = cal_loss(rlt, data)
        loss.backward()
        time_backward = time_backward + timeit.timeit() - time
        time = timeit.timeit() 

    
    time_forward_avg = time_forward / all_steps
    time_backward_avg = time_backward / all_steps
    return time_forward_avg, time_backward_avg


if __name__ == "__main__":
    model_size_config_path = "configs/model_sizes.yaml"
    with open(model_size_config_path) as f:
        model_size_config = yaml.safe_load(f)

    model_size = "small"
    if model_size not in list(model_size_config.keys()):
        raise ValueError(f"Not a valid model size predefined name: {model_size}")

    d_model = model_size_config[model_size]["d_model"]
    d_ff = model_size_config[model_size]["d_ff"]
    num_layers = model_size_config[model_size]["num_layers"]
    num_heads = model_size_config[model_size]["num_heads"]
    rope_theta = 10000
    vocab_size = 10000
    context_length = 512

    model = get_model(
        vocab_size = vocab_size,
        context_length = context_length,
        d_model = d_model,
        d_ff = d_ff,
        num_layers = num_layers,
        num_heads = num_heads,
        rope_theta = rope_theta
    )

    warmup_steps = 5
    all_steps = 10
    if_backward = True

    batch_size = 4
    length = context_length

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    print(f"Test on {device}")
    
    time_forward_avg, time_backward_avg = timer(
        model=model,
        warmup_steps=warmup_steps,
        all_steps=all_steps,
        if_backward=if_backward,
        batch_size=batch_size,
        length=length,
        vocab_size=vocab_size,
        device = device,
    )

    print(f"time_forward_avg: {time_forward_avg}")
    print(f"time_backward_avg: {time_backward_avg}")
