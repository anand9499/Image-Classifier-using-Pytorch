import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse


def load_data(location="./flowers"):
    data_dir = location
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])}

    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])}

    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)}
    trainloader = dataloaders['train']
    validloader = dataloaders['valid']
    testloader = dataloaders['test']

    return trainloader, validloader, testloader


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def buildnn(model_type, device, hidden_units=1024, dropout=0.2, lr=0.001):
    if model_type == 'vgg19':
        model = models.vgg19(pretrained=True)

    elif model_type == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Please select a either vgg19 or densenet121")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(dropout)),
        ('fc2', nn.Linear(4096, hidden_units)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(dropout)),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    criteria = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    if torch.cuda.is_available() and device == 'gpu':
        model.to('cuda')
    return model, criteria, optimizer


def model_train(model, learning_rate, criterion, optimizer, tloader, vloader, print_every=40, epochs=9, device='gpu'):
    model.train()
    tloader, vloader, te_loader = load_data("./flowers")
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0

        for ii, (inputs, labels) in enumerate(tloader):
            steps += 1

            if torch.cuda.is_available() and device == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            criterion = nn.NLLLoss()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss, validation_accuracy = check_accuracy_and_loss(model, criterion, vloader)
                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "Training Loss: {:.4f}".format(running_loss / print_every),
                      "Validation Loss : {:.4f}".format(validation_loss),
                      "Validation Accuracy:{:.4f}".format(validation_accuracy))


def check_accuracy_and_loss(model, criteria, validloader):
    model.eval()
    correct = 0
    total = 0
    valid_loss = 0
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            valid_loss = criteria(outputs, labels).item()
            valid_accuracy = correct / total
        return valid_loss, valid_accuracy


def save_checkpoint(path='classifier.pth', model_type='vgg19', hidden_units=1024, dropout=0.5, lr=0.001, epochs=9):
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    torch.save({'model_type': model_type,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
               'classifier.pth')


def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)

    if chpt['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Sorry base architecture not recognized")

    model.class_to_idx = chpt['class_to_idx']

    # Create the classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(4096, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    # Put the classifier on the pretrained network
    model.classifier = classifier

    model.load_state_dict(chpt['state_dict'])

    return model


def process_image(image):
    from PIL import Image
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))
    # Normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))

    return img


def predict(image_path, model, topk=5):
    if torch.cuda.is_available() and power == 'gpu':
        model.to('cuda')

    model.eval()

    # Process image
    img = process_image(image_path)

    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)

    # Probabilities
    output = model.forward(model_input)
    probs = torch.exp(output)

    # Top probabilities
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]

    return top_probs, top_labels, top_flowers
