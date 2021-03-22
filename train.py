import torch
import multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import MyDataset
from model import Resnet
import os

# Check if runtime uses GPU
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train():
    my_model = Resnet(kernel_size=3, filters=64, inChannels=3, input_shape=(3, 240, 320), conv_nonlinearity='relu', num_class=25)
    my_model = my_model.to(device)
    if os.path.exists('my_model.pt'):
        my_model.load_state_dict(torch.load('my_model.pt'))
        print('Load my_model.pt')
    batch_size = 32
    num_epoch = 100
    num_classes = 25
    learning_rate = 8e-4

    train_set = MyDataset(is_train=True, num_cat=num_classes)
    validation_set = MyDataset(is_train=False, num_cat=num_classes)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=True,
                                                    pin_memory=True)

    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    loss_func = torch.nn.NLLLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10,
                                  threshold=2e-1, verbose=True, min_lr=1e-5)
    bestTestAccuracy = 0

    print('Start training')
    train_size = len(train_loader.dataset)
    test_size = len(validation_loader.dataset)
    for epoch in range(num_epoch):
        total = 0
        correct = 0
        my_model.train()
        for i, data in enumerate(train_loader, 0):
            labels = data['label'].to(device)
            img = data['img'].to(device).float()
            prediction = my_model(img)

            loss = loss_func(prediction, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            _, predicted = torch.max(prediction, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f'Train | Epoch {epoch}/{num_epoch}, Batch {i}/{int(train_size/batch_size)} '
                  f' Loss: {loss.clone().item():.3f} LR: {get_lr(optimizer):.6f}'
                  f' Acc: {(100 * correct / total):.3f}')

        total = 0
        correct = 0
        my_model.eval()
        for i, data in enumerate(validation_loader, 0):
            labels = data['label'].to(device)
            img = data['img'].to(device).float()
            prediction = my_model(img)

            _, predicted = torch.max(prediction, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print(f'Test | Epoch {epoch}/{num_epoch}, Batch {i}/{int(test_size/batch_size)} '
                  f' Loss: {loss.clone().item():.3f} LR: {get_lr(optimizer):.6f}'
                  f' Acc: {(100 * correct / total):.3f} Best-so-far: {100*bestTestAccuracy:.5f}')

        if (correct/total) > bestTestAccuracy:
            bestTestAccuracy = correct/total
            print(f'Update best test: {100*bestTestAccuracy:.5f}')
            torch.save(my_model.state_dict(), f"my_model_{str(round(100*bestTestAccuracy,2)).replace('.', '_')}.pt")

        scheduler.step(bestTestAccuracy)




train()