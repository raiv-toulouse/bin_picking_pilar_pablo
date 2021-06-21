import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

transform = transforms.Compose([
            transforms.CenterCrop(size=224),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
trainset = datasets.ImageFolder("/home/student1/Pictures", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
# writer.add_graph(model, images)
writer.close()