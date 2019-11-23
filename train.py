import torch
from torch import nn
from torchvision import datasets, transforms, models
import argparse
from collections import OrderedDict
import json

# Train a new network on a data set with train.py
#   Basic usage: python train.py data_directory
#   Prints out training loss, validation loss, and validation accuracy as the network trains

def main():
    args = setCommandArgs()
    model, model_name = build_model(args)
    train_model(args, model)
    save_model(args, model, model_name)

def train_model(command_args, model):
    trainloader, validloader, testloader = get_loaders(command_args)

    device = torch.device("cuda" if command_args.gpu else "cpu")
    print("using device: {}.  learning rate: {}".format(device, command_args.learning_rate))

    model.to(device);
    criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=command_args.learning_rate)

    epochs = command_args.epochs
    running_loss = 0
    print_every = 25

    for epoch in range(epochs):
        steps = 0
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                running_test_loss = 0
                accuracy = 0
                model.eval()  # set to eval mode to not use dropout
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        logps = model.forward(inputs)
                        test_batch_loss = criterion(logps, labels)

                        running_test_loss += test_batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs} ({steps}/{len(trainloader)}): "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss:  {running_test_loss/len(testloader):.3f}.. "
                      f"Test accy:  {100*accuracy/len(testloader):.3f}%")

                running_loss = 0
                model.train()  # set back to train mode

def save_model(command_args, model, model_name):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    checkpoint = {'input_size': [3, 224, 224],
                  'output_size': 103,
                  'features': model.features,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'cat_to_name': cat_to_name,
                  'arch': command_args.arch}

    checkpoint_filename = f"checkpoint_{model_name}.pth"

    save_path = f"{command_args.save_dir}/{checkpoint_filename}"

    torch.save(checkpoint, save_path)

    print(f"model checkpoint saved: {save_path}")


def build_model(args):
    model, model_name = get_pretrained_model(args.arch)
    print(f"using arch:  {model_name}")

    if(isinstance(model.classifier, torch.nn.modules.container.Sequential)):
        for i in range(len(model.classifier)):
            if isinstance(model.classifier[i], torch.nn.modules.linear.Linear):
                input_features = model.classifier[i].in_features
                break
    else:
        input_features = model.classifier.in_features

    for param in model.parameters():
        param.requires_grad = False

    # create a new layer to perform classifications.  The dataset has 102 labels
    classifier = nn.Sequential(OrderedDict([
        ('drop1', nn.Dropout(0.2)),
        ('fc1', nn.Linear(input_features, 512)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(0.2)),
        ('fc2', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))]))

    # replace last layer with new trainable one
    model.classifier = classifier
    print(type(model))

    return model, model_name

def get_pretrained_model(arch):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model_name = 'vgg16'
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        model_name = 'alexnet'
    else:
        model = models.densenet121(pretrained=True)
        print(f"arch {arch} not found.  Using default model.");
        model_name = 'densenet121'
    return model, model_name


def get_loaders(command_args):
    data_dir = command_args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    testloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64)

    return trainloader, validloader, testloader

def setCommandArgs():
    parser = argparse.ArgumentParser(
        description='Options for training a model'
    )

    parser.add_argument('data_dir', help='directory to train/test/validation data')
    parser.add_argument('--save_dir', action="store", dest="save_dir", required=True)
    parser.add_argument('--arch', action="store", dest="arch", default="vgg13")
    parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.01, type=float)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", default=512, type=int)
    parser.add_argument('--epochs', action="store", dest="epochs", default=20, type=int)
    parser.add_argument('--gpu', action="store_true", dest="gpu", default=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
