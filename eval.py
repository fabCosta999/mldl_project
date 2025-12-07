import wandb
import torch
from torch import nn
from customnet import CustomNet
from utils.transform import transform
from torchvision.datasets import ImageFolder

wandb.init(project="tiny-imagenet-customnet", name="evaluation")

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')

    wandb.log({
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    return val_accuracy


# dataset
tiny_imagenet_dataset_val = ImageFolder(
    root='tiny-imagenet/tiny-imagenet-200/val',
    transform=transform
)

val_loader = torch.utils.data.DataLoader(
    tiny_imagenet_dataset_val,
    batch_size=32,
    shuffle=False
)

model = CustomNet()
model.load_state_dict(torch.load("models/customnet_final.pth"))
model = model.cuda()
model.eval()

criterion = nn.CrossEntropyLoss()
val_accuracy = validate(model, val_loader, criterion)

print("model accuracy:", val_accuracy)

wandb.finish()
