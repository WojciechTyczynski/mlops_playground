import torch
from torch import nn
from model import MyAwesomeModel

def main():
    model = torch.load('models/model.pth')
    test_set = torch.load('data/processed/train.pt')
    dataiter = iter(test_set)
    criterion = nn.NLLLoss()
    accuracy = 0
    for i,  (images, labels) in enumerate(test_set):
        output = model.forward(images)
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).sum()
    print(f"Accuracy: {accuracy/len(test_set.dataset)}%")

if __name__ == '__main__':
    main()