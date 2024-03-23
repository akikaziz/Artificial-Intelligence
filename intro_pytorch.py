import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training =True):

    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if training:
        dataset = datasets.FashionMNIST('./data', train = True, download = True, transform = custom_transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    else:
        dataset = datasets.FashionMNIST('./data', train = False, transform = custom_transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle = False)

    return loader

def build_model():
    
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    return model

def train_model(model, train_loader, criterion, T):

    criterion = nn.CrossEntropyLoss()

    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    
    for epoch in range(T):

        epoch_loss = 0.0
        correct_count = 0

        for images, labels in train_loader:

            opt.zero_grad()

            out = model(images)
            loss = criterion(out, labels)

            loss.backward()
            opt.step()

            epoch_loss += images.size(0)*loss.item()

            max_val,max_ind = torch.max(out.data, 1)

            correct_count += (max_ind == labels).sum().item()

        mean_loss = epoch_loss/len(train_loader.dataset)
        accuracy = correct_count*100.0 / len(train_loader.dataset)

        print(f'Train Epoch: {epoch} Accuracy: {correct_count}/{len(train_loader.dataset)}({accuracy:.2f}%) Loss: {mean_loss:.3f}')

def evaluate_model(model, test_loader, criterion, show_loss=True):

    model.eval()

    epoch_loss = 0.0
    correct_count = 0

    with torch.no_grad():

        for data, labels in test_loader:

            out = model(data)
            loss = criterion(out, labels)
            
            epoch_loss += data.size(0)*loss.item()

            max_val,max_ind = torch.max(out.data, 1)

            correct_count += (max_ind == labels).sum().item()

    mean_loss = epoch_loss/len(test_loader.dataset)
    accuracy = correct_count*100 / len(test_loader.dataset)

    if show_loss:
        print(f'Average loss: {mean_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')

def predict_label(model, test_images, index):

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    logits = model(test_images)

    prob = F.softmax(logits, dim = 1)

    top3_prob,top3 = torch.topk(prob, 3)

    for i in range(3):
        print(f'{class_names[top3[0][i]]}: {prob[0][top3[0][i]] * 100:.2f}%')

def main():

    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    
    test_loader = get_data_loader(False)
    # print(type(test_loader))
    # print(test_loader.dataset)

    model = build_model()
    print(model)

    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, criterion, 5)

    evaluate_model(model, test_loader, criterion, show_loss = False)
    evaluate_model(model, test_loader, criterion, show_loss = True)

    test_images = test_loader.dataset.data.float()
    predict_label(model, test_images, 1)

if __name__=="__main__":
    main()
