import torch
import torch.nn
from torch import optim
from dataset import SantaNoSanta
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from lenet import LeNet
from vgg_net import VGGNet
from tqdm import tqdm
from utils import count_parameters


# hyper parameters
lr = 0.001
batch_size = 224
image_size = 224
class_num = 2
input_channel = 3
epochs_num = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = SantaNoSanta(dog_root='dataset/train/dogs/',
                         cat_root='dataset/train/cats/',
                         size=image_size,
                         transform=transforms.ToTensor())

test_set = SantaNoSanta(dog_root='dataset/test/dogs/',
                        cat_root='dataset/test/cats/',
                        size=image_size,
                        transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

model = VGGNet(3, 1).float()
model.to(device=device)

count_parameters(model=model)

loss_function = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs_num):
    losses = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for batch_idx, (data, label) in loop:
        data = data.to(device).float()
        label = label.to(device).float().reshape(-1, 1)

        predict = model(data).float()
        loss = loss_function(predict, label).float()
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f'epoch[{epoch}/{epochs_num}]')
        loop.set_postfix(loss=loss.item(), acc=torch.rand(1).item())


def check_acc(loader, network):
    correct = 0
    sample = 0
    network.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float().reshape(-1, 1)

            scorse = network(x).float()
            _, prediction = scorse.max(1)
            correct += (prediction == y).sum()
            sample += prediction.size(0)
            print(f'got {correct}/{sample} with acc {round(float(correct) / float(sample) * 100, 2)}')


print('check acc on test set')
check_acc(test_loader, model)
