#!/usr/bin/env python
# coding: utf-8

import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from tqdm.auto import tqdm
from tqdm.contrib.itertools import product



class NoisyCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with optional noise adjustment using a transition matrix.
    
    If a transition matrix is provided, the model's predicted probabilities are 
    adjusted by multiplying with the transition matrix before computing the loss. 
    This adjustment accounts for label noise, improving model robustness to noisy labels.
    """
    def __init__(self, transition: Optional[torch.Tensor] = None):
        super(NoisyCrossEntropyLoss, self).__init__()
        self.register_buffer("transition", transition)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo().eps
        prob = F.softmax(x, dim=1)
        if self.transition is not None:
            prob = torch.matmul(prob, self.transition)
        loss = F.nll_loss(torch.log(prob + eps), y)
        return loss


def get_available_device():
    """
    Returns the best available device for computation.
    
    Checks for CUDA and MPS (Metal Performance Shaders) availability 
    and returns the corresponding device. Defaults to CPU if neither 
    is available.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
    
    
def reproducible(seed=3407):
    """
    Sets the seed for reproducibility in PyTorch, NumPy, and Python's random module.
    
    Disables CuDNN benchmark mode to ensure consistent results across runs.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    

def load_dataset(name, data_dir=Path("data")):
    """
    Loads data from a .npz file, converts it to PyTorch tensors, and prepares datasets.

    - Loads the training and test data from the specified file using NumPy.
    - Converts the data to PyTorch tensors with appropriate data types.
    - Adds a channel dimension if the data is 3D and rearranges dimensions to match 
      the (N, C, H, W) format commonly used for PyTorch models.
    - Returns the prepared training and test datasets, along with the dataset name.
    """
    data = np.load(data_dir / f"{name}.npz")
    X_train, y_train, X_test, y_test = map(
        data.get, ["X_tr", "S_tr", "X_ts", "Y_ts"])
    data.close()
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)
    if len(X_train.shape) == 3:
        X_train = X_train.unsqueeze(3)
        X_test = X_test.unsqueeze(3)
    X_train = X_train.permute(0, 3, 1, 2)
    X_test = X_test.permute(0, 3, 1, 2)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    return train_dataset, test_dataset, name
    

def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device
                    ):
    model.to(device)
    model.train()
    loss_fn.to(device)
    loss_fn.train()

    running_loss = 0.
    running_acc = 0.

    for data in dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
        running_acc += correct / len(labels)

    return running_loss / len(dataloader), running_acc / len(dataloader)


def test_one_epoch(model: nn.Module,
                   dataloader: DataLoader,
                   loss_fn: nn.Module,
                   device: torch.device
                   ):
    model.to(device)
    model.eval()
    loss_fn.to(device)
    loss_fn.eval()

    running_loss = 0.
    running_acc = 0.

    with torch.inference_mode():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)    
            loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            running_acc += correct / len(labels)

    return running_loss / len(dataloader), running_acc / len(dataloader)


def train(model: nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          test_dataloader: DataLoader,
          loss_fn: nn.Module = nn.CrossEntropyLoss(),
          noisy_loss_fn: Optional[nn.Module] = None,
          optimizer: Optional[torch.optim.Optimizer] = None,
          device: torch.device = get_available_device(),
          n_epoch: int = 1_000,
          n_print: int = 10
          ):
    if noisy_loss_fn is None:
        noisy_loss_fn = loss_fn
    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=5e-5)
        
    model.to(device)
    best_val_acc = 0.
    best_state = model.state_dict()
    for epoch in tqdm(range(1, n_epoch + 1),
                      desc="Train",
                      unit="epoch",
                      leave=False):
        avg_train_loss, avg_train_acc = train_one_epoch(
            model,
            train_dataloader,
            loss_fn=noisy_loss_fn,
            optimizer=optimizer,
            device=device
        )
        avg_val_loss, avg_val_acc = test_one_epoch(
            model,
            val_dataloader,
            loss_fn=noisy_loss_fn,
            device=device
        )

    avg_test_loss, avg_test_acc = test_one_epoch(
        model,
        test_dataloader,
        loss_fn=loss_fn,
        device=device
    )
    return avg_test_loss, avg_test_acc
    

def estimate_transition(model: nn.Module,
                        dataloader: DataLoader,
                        device: torch.device = get_available_device()):
    """
    Estimates a class transition matrix using a trained model and data.
    
    Returns:
        anchors (torch.Tensor): Estimated class transition matrix.
    """
    model.to(device)
    model.eval()

    prob = []
    with torch.inference_mode():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)    
            prob.append(torch.softmax(outputs, dim=1))
    prob = torch.concat(prob)
    n_classes = prob.size(1)
    
    anchors = torch.zeros((n_classes,) * 2).to(device)
    for i in range(anchors.size(0)):
        anchors[i, :] = prob[torch.argmax(prob[:, i])]
    return anchors


def create_cnn(datasets):
    from custom_cnn import CustomCNN, params
    numel = datasets[0][0][0].numel()
    name = datasets[2]
    kwargs = params[name].copy()
    n_epoch = kwargs.pop("num_epochs")
    model = CustomCNN(**kwargs)
    return n_epoch, model


def create_vit(datasets):
    from custom_vit import CustomViT
    shape = datasets[0][0][0].shape
    n_epoch = 10
    model = CustomViT(
        chw=shape,
        no_h_patches=4,
        no_w_patches=4,
        no_classes=4,
        dim=32,
        depth=4,
        no_heads=2,
        mlp_hidden_dim=64
    )
    return n_epoch, model
    

fashion_3 = load_dataset("FashionMNIST0.3")
fashion_6 = load_dataset("FashionMNIST0.6")
cifar = load_dataset("CIFAR10")

fashion_3_transition = torch.tensor([
    [0.7, 0.3, 0.0, 0.0],
    [0.0, 0.7, 0.3, 0.0],
    [0.0, 0.0, 0.7, 0.3],
    [0.3, 0.0, 0.0, 0.7]
])
fashion_6_transition = torch.full((4, 4), 0.2).fill_diagonal_(0.4)

exp_dir = Path("experiments")
exp_dir.mkdir(exist_ok=True)

setup = [
    [create_cnn, create_vit],
    [
        [fashion_3, fashion_3_transition],
        [fashion_6, fashion_6_transition],
        *product(
            [
                fashion_3,
                fashion_6,
                cifar
            ],
            [True, False]
        )
    ]
]

results = []
for model_fn, (datasets, transition) in product(*setup, desc="Experiments"):
    reproducible()

    train_val_dataset, test_dataset, dataset_name = datasets
    train_dataset, val_dataset = random_split(train_val_dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128)
    test_dataloader = DataLoader(test_dataset, batch_size=128)
    
    for run in tqdm(range(10), desc="Runs", leave=False):
        # estimate the transition if necessary
        estimate = None
        if transition is False:
            loss_fn = nn.CrossEntropyLoss()
            noisy_loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = NoisyCrossEntropyLoss()
            if transition is True:
                aux_n_epoch, aux_model = model_fn(datasets)
                loss, acc = train(
                    aux_model,
                    train_dataloader,
                    val_dataloader,
                    test_dataloader,
                    n_epoch=aux_n_epoch
                )
                train_val_dataloader = DataLoader(
                    train_val_dataset,
                    batch_size=128,
                    shuffle=True
                )
                estimate = estimate_transition(aux_model, train_val_dataloader)
                noisy_loss_fn = NoisyCrossEntropyLoss(estimate)
            else:
                noisy_loss_fn = NoisyCrossEntropyLoss(transition)
    
        n_epoch, model = model_fn(datasets)
        loss, acc = train(
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss_fn=loss_fn,
            noisy_loss_fn=noisy_loss_fn,
            n_epoch=n_epoch
        )
    
        results.append({
            "Model": model_fn.__name__.removeprefix("create_"),
            "Datasets": datasets[2],
            "Transition": {False: "None", True: "Estimate"}.get(transition, "Truth"),
            "Run": run + 1,
            "Loss": loss,
            "Acc": acc,
            "Estimate": None if estimate is None else estimate.tolist()
        })
    
now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
pd.DataFrame(results).to_csv(exp_dir / f"{now_str}.csv")
