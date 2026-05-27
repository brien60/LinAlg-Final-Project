import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms as T
from datasets import load_dataset
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchinfo import summary

from time import time

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("No Compression")
summary(model, input_size=(1, 3, 224, 224))

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
original_num_params = total_params
print(f"total parameters pre-compression: {original_num_params}")


def weight_svd(W: torch.Tensor, j: float) -> tuple[torch.Tensor, torch.Tensor]:
    f, c, k_h, k_w = W.shape

    W = torch.reshape(W, (f, c*k_h*k_w))
    U, S, V_h = torch.linalg.svd(W, full_matrices=False)

    U_j = U[:, :j]
    S_j = S[:j]
    V_j = V_h[:j, :]

        
    # E = 100 * sum(s*s for s in S_j) / sum(s*s for s in S)
    # print(f"Name: {name} | Rank: {j}/{f} | Energy: {E:.2f}")

    V = torch.reshape(torch.diag(S_j) @ V_j, (j, c, k_h, k_w))
    U = torch.reshape(U_j, (f, j, 1, 1))

    return U, V

def replace_conv(model, name, new_conv):
    parts = name.split(".")
    parent = model
    for part in parts[: -1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_conv)

def evaluate(model):

    total = 0
    correct = 0

    t = 0
    total_time = 0
    avg_time = 0

    model.eval()
    with torch.inference_mode():
        for sample in tqdm(ds, total=50000, unit="img"):
            img = sample["image"]
            label = sample["label"]
            
            input = transform(img).unsqueeze(0)
            input = input.to(device)
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



# vanilla SVD
print("\n\nVanilla SVD")

CRs = [0.68, 0.68, 0.68, 0.68] # parameter reduction: 22.02%, acc drop: 10.51%
CRs = [0.6, 1.0, 1.0, 1.0] # parameter reduction: 0.43%, acc drop: 6.34%
CRs = [1.0, 1.0, 1.0, 0.6] # parameter reduction: 22.91%, acc drop: 2.64%


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

        conv_V = nn.Conv2d(
            in_channels=c, out_channels=j,
            kernel_size=(k_h, k_w),
            stride=module.stride,
            padding=module.padding,
            bias=False
        )
        conv_U = nn.Conv2d(in_channels=j, out_channels=f, kernel_size=1, bias=False)

        U, V = weight_svd(W, j)

        with torch.no_grad():
            conv_V.weight.copy_(V)
            conv_U.weight.copy_(U)


        W_approx = nn.Sequential(conv_V, conv_U).to(device)
        # print("Before", module)
        # print("After", W_approx)

        replace_conv(model, name, W_approx)
        
        # W_approx = U_j @ torch.diag(S_j) @ V_j
        # print(torch.dist(W, W_approx).item())
        # W_approx = torch.reshape(W_approx, (f, c, k_h, k_w))

summary(model, input_size=(1, 3, 224, 224))

total_params = sum(param.numel() for param in model.parameters())
print(f"total parameters post-compression (vanilla): {total_params}")
print(f"Parameter reduction (vanilla): {100*(original_num_params - total_params) / original_num_params:.2f}%")

evaluate(model=model)



# channel slicing
print("\n\nChannel Slicing")
model = models.resnet18(weights=weights).to(device) # reset model
 
class ChannelSlicedSVD(nn.Module):
    def __init__(self, j: float, k: float, module: nn.Module) -> None:
        super().__init__()
        W = module.weight
        f, c, k_h, k_w = W.shape
        
        c_i = c // k
        channels = [c_i] * k
        channels[k-1] += c - c_i*k

        self.channels = channels
        self.conv_V = nn.ModuleList()
        self.conv_U = nn.Conv2d(in_channels=k*j, out_channels=f, kernel_size=1, bias=False)

        U_is = []

        start = 0
        for i in range(k):
            conv_V_i = nn.Conv2d(
                in_channels=channels[i], out_channels=j,
                kernel_size=(k_h, k_w),
                stride=module.stride,
                padding=module.padding,
                bias=False
            )
            
            W_i = W[:, start:start+channels[i], :, :]
            U_i, V_i = weight_svd(W_i, j)
            # print(f"V_{i}", V_i.shape)
            with torch.no_grad():
                conv_V_i.weight.copy_(V_i)
            
            start+=channels[i]
            self.conv_V.append(conv_V_i)
            U_is.append(U_i)

        U = torch.cat(U_is, dim=1)
        with torch.no_grad(): self.conv_U.weight.copy_(U)
        # print(U.shape)



    def forward(self, x) -> torch.Tensor:
        feature_maps = []

        start = 0
        for i in range(len(self.conv_V)):
           x_i = x[:, start:start+self.channels[i], :, :]
           V_i = self.conv_V[i]
           feature_maps.append(V_i(x_i))
           start+=self.channels[i]

        # for fm in feature_maps: print(fm.shape)
        feature_maps = torch.cat(feature_maps, dim=1)
        # print(feature_maps.shape)
        approx_feature_maps = self.conv_U(feature_maps)

        return approx_feature_maps
    
k = 4
CRs = [1.0, 1.0, 1.0, 0.44] # 23.83% 0.88%
# js = [64, 128, 256, 320]

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
        # j = js[layer_num-1]
        if (j == f): continue

        W_approx = ChannelSlicedSVD(j, k, module).to(device)
        replace_conv(model, name, W_approx)

summary(model, input_size=(1, 3, 224, 224))

total_params = sum(param.numel() for param in model.parameters())
print(f"total parameters post-compression (sliced): {total_params}")
print(f"Parameter reduction (sliced): {100*(original_num_params - total_params) / original_num_params:.2f}%")

evaluate(model=model)