import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
])

train = datasets.ImageFolder(root=r".\Train",transform=transform)
test = datasets.ImageFolder(root=r".\Test",transform=transform)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(16*61*61, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out = self.pool(self.relu(self.conv1(x)))
        out = self.pool(self.relu(self.conv2(out)))
        out = out.view(out.size(0),-1)
        out = self.relu(self.l1(out))
        out = self.relu(self.l2(out))
        return self.sigmoid(self.l3(out))

def train_the_model(model,train,test):
    batch_size=32
    num_epochs=100
    learning_rate = 0.001
    train_ds = DataLoader(train,batch_size=batch_size,shuffle=True)
    test_ds = DataLoader(test,batch_size=batch_size,shuffle=True)
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    best_accuracy = 0
    min_loss = np.inf
    best_weights = None
    for epochs in range(num_epochs):
        model.train()
        total_loss = 0
        for (image,label) in train_ds:
            image,label = image.to(device), label.to(device)
            out = model(image)
            label = label.view(-1,1).to(dtype=torch.float32)
            loss = loss_fn(out, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f'Epoch: {epochs+1}/{num_epochs} | Loss: {total_loss/len(train_ds):.3f}')

        with torch.no_grad():
            model.eval()
            correct =0 
            total_loss = 0
            total = 0
            for (image,label) in test_ds:
                image, label = image.to(device), label.to(device)
                out = model(image)
                label = label.view(-1,1).to(dtype=torch.float32)
                loss = loss_fn(out,label)
                total_loss += loss.item() * image.size(0)
                predicted = (out > 0.5).float()
                total = label.size(0)
                correct += (predicted == label).sum().item()
            accuracy = 100 * (correct/total)
            best_accuracy = max(best_accuracy,accuracy)
            avg_loss = total_loss/len(test_ds)
            if avg_loss < min_loss:
                min_loss = avg_loss
                best_weights = model.state_dict()
    model.load_state_dict(best_weights)
    return f'Training complete! Best Loss: {min_loss:.3f} | Best Accuracy: {best_accuracy:.3f}'

model = NeuralNetwork().to(device)
train_the_model(model,train,test)
torch.save(model.state_dict(), "deepfake_model.pth")