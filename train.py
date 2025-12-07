import wandb
from customnet import CustomNet
from utils.transform import transform
from torch import nn
import torch
from torchvision.datasets import ImageFolder
import os

# initialize wandb
wandb.init(project="tiny-imagenet-customnet", config={
    "epochs": 10,
    "batch_size": 32,
    "lr": 0.001,
    "momentum": 0.9,
})

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

    # log to wandb
    wandb.log({
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "epoch": epoch
    })


# dataset
tiny_imagenet_dataset_train = ImageFolder(
    root='dataset/tiny-imagenet/tiny-imagenet-200/train',
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    tiny_imagenet_dataset_train,
    batch_size=wandb.config.batch_size,
    shuffle=True,
    num_workers=2
)

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=wandb.config.lr,
    momentum=wandb.config.momentum
)

num_epochs = wandb.config.epochs

# training loop
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)

# save model
os.makedirs("models", exist_ok=True)
save_path = "models/customnet_final.pth"
torch.save(model.state_dict(), save_path)
print(f"\nModel saved correctly at: {save_path}")

wandb.save(save_path)
wandb.finish()
