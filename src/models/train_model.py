import argparse
import sys

import torch
from torch import optim
from torch import nn

from model import MyAwesomeModel

import matplotlib.pyplot as plt


def main():
    model = MyAwesomeModel()
    train_set = torch.load('data/processed/train.pt')
    train_losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    print("Training day and night")
    for epoch in range(10):
        running_loss = 0
        model.train()
        n_total_steps = len(train_set)
        for i, (images, labels) in enumerate(train_set):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        train_losses.append(running_loss/n_total_steps)
    torch.save(model, 'models/model.pth')
    plt.plot(train_losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig("reports/figures/training.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    print("Training day and night")
    main()
