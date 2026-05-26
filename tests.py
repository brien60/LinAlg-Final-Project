import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms as T
from datasets import load_dataset
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchsummary import summary

from time import time

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
model.eval()

ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


total_params = sum(param.numel() for param in model.parameters())
print(f"total parameters pre-compression: {total_params}")

# vanilla SVD
CRs = [1.0, 1.0, 1.0, 0.6] # parameter reduction: 22.91%, acc drop: 2.64%
CRs = [0.68, 0.68, 0.68, 0.68] # parameter reduction: 22.02%, acc drop: 10.51%
CRs = [0.6, 1.0, 1.0, 1.0] # parameter reduction: 0.43%, acc drop: 6.34%

for name, module in model.named_modules():
    if (isinstance(module, nn.Conv2d) and (not "downsample" in name) and (not name == "conv1")):
        parts = name.split(".")
        # print(name, module.weight.shape)

        layer_num = int(parts[0][-1])
        compression_ratio = CRs[layer_num-1]

        if (compression_ratio == 1.0): continue  

        W = module.weight
        f, c, k_h, k_w = W.shape
        j = int(compression_ratio * f)

        W = torch.reshape(W, (f, c*k_h*k_w))

        U, S, V_h = torch.linalg.svd(W, full_matrices=False)

        U_j = U[:, :j]
        S_j = S[:j]
        V_j = V_h[:j, :]

        E = 100 * sum(s*s for s in S_j) / sum(s*s for s in S)
        print(f"Name: {name} | Rank: {j}/{f} | Energy: {E:.2f}")

        conv_V = nn.Conv2d(
            in_channels=c, out_channels=j,
            kernel_size=(k_h, k_w),
            stride=module.stride,
            padding=module.padding,
            bias=False
        )
        conv_U = nn.Conv2d(in_channels=j, out_channels=f, kernel_size=1, bias=False)
        
        with torch.no_grad():
            conv_V.weight.copy_(torch.reshape(torch.diag(S_j) @ V_j, (j, c, k_h, k_w)))
            conv_U.weight.copy_(torch.reshape(U_j, (f, j, 1, 1)))

        W_approx = nn.Sequential(conv_V, conv_U)
        # print("Before", module)
        # print("After", W_approx)

        parent = model
        for part in parts[: -1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], W_approx)
        
        # W_approx = U_j @ torch.diag(S_j) @ V_j
        # print(torch.dist(W, W_approx).item())
        # W_approx = torch.reshape(W_approx, (f, c, k_h, k_w))

total_params = sum(param.numel() for param in model.parameters())
print(f"total parameters post-compression: {total_params}")

print(f"Parameter reduction: {100*(11689512 - total_params) / 11689512:.2f}%")


total = 0
correct = 0

t = 0
total_time = 0
avg_time = 0

with torch.inference_mode():
    for sample in tqdm(ds, total=50000, unit="img"):
        img = sample["image"]
        label = sample["label"]
        
        input = transform(img).unsqueeze(0)
        t = time()
        pred = model(input).argmax(dim=1).item()
        total_time += time() - t

        correct += (label == pred)

        if (total % 500 == 0) and total != 0:
            avg_time += total_time
            print(f"{total}/50000 | acc: {100*correct/total :.2f} | time: {total_time :.2f}s | avg time: {1000*avg_time / total:.2f}ms/img")
            total_time = 0

        total += 1


acc = 100*correct/total 
print(f"Final acc: {acc:.2f}")


print(f"Accuracy drop: {69.7 - acc:.2f}%")