import argparse
import sys

import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim

from data import mnist


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001)
        parser.add_argument('--epochs', default=10)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        train_losses = []
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            running_loss = 0
            n_total_steps = len(train_set)
            for i, (images, labels) in enumerate(train_set):
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_losses.append(running_loss/n_total_steps)
        torch.save(model, 'model.pth')
        plt.plot(train_losses)
        plt.show()

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        _, test_set = mnist()
        accuracy = 0
        for i,  (images, labels) in enumerate(test_set):
            output = model.forward(images)
            ps = torch.exp(output)
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).sum()
        print(f"Accuracy: {accuracy*100/len(test_set.dataset)}%")


if __name__ == '__main__':
    TrainOREvaluate()
