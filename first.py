import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor
import datetime

torch.backends.cudnn.benchmark = True

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.resnet(x)
        return x


if __name__ == '__main__':

    batch_size = 32
    # generate fake data
    test_data = datasets.FakeData(size=batch_size*1000, image_size=(3,224,224), num_classes=1000, transform=ToTensor())


    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8, pin_memory=True, prefetch_factor=32, persistent_workers=True)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        print(f'size: {size}, num_batches: {num_batches}')
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                
                model(X)
                # pred = model(X)
        #         test_loss += loss_fn(pred, y).item()
        #         correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # test_loss /= num_batches
        # correct /= size
        # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        begin_time = datetime.datetime.now()
        print(f"begin test {t+1}: {begin_time}\n")
        test(test_dataloader, model, loss_fn)
        end_time = datetime.datetime.now()
        print(f"end test {t+1}: {end_time}, cost {end_time - begin_time}s total time\n")
    print("Done!")

