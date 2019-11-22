import torch
from torch import nn
from torchvision import datasets, transforms, models
import argparse
from collections import OrderedDict

# Train a new network on a data set with train.py
#   Basic usage: python train.py data_directory
#   Prints out training loss, validation loss, and validation accuracy as the network trains

def main():
    args = setCommandArgs()
    print(args.save_dir)
    save_dir = args.save_dir
    build_model(args)

def build_model(args):
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        print(f"using arch vgg16")
    elif args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        print(f"using arch alexnet")
    else:
        model = models.densenet121(pretrained=True)
        print(f"arch {args.arch} not found.  Using default model densenet121");

    if(isinstance(model.classifier, torch.nn.modules.container.Sequential)):
        for i in range(len(model.classifier)):
            if isinstance(model.classifier[i], torch.nn.modules.linear.Linear):
                input_features = model.classifier[i].in_features
                break
    else:
        input_features = model.classifier.in_features

    #print(f"input features: {input_features}")

    for param in model.parameters():
        param.requires_grad = False

    # create a new layer to perform classifications.  The dataset has 102 labels
    classifier = nn.Sequential(OrderedDict([
        ('drop1', nn.Dropout(0.25)),
        ('fc1', nn.Linear(input_features, 512)),
        ('relu', nn.ReLU()),
        ('drop2', nn.Dropout(0.25)),
        ('fc2', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))]))

    # replace last layer with new trainable one
    model.classifier = classifier

    return model



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
    # python train.py
    #    data_dir --save_dir save_directory
    #    Choose architecture: python train.py data_dir --arch "vgg13"
    #    Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    #    Use GPU for training: python train.py data_dir --gpu
    parser = argparse.ArgumentParser(
        description='Options for training a model',
    )


    parser.add_argument('data_dir', help='directory to train/test/validation data')
    parser.add_argument('--save_dir', action="store", dest="save_dir", required=True)
    parser.add_argument('--arch', action="store", dest="arch", default="vgg13")
    parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.01, type=float)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", default=512, type=int)
    parser.add_argument('--epochs', action="store", dest="epochs", default=20, type=int)
    parser.add_argument('--gpu', action="store_true", dest="gpu", default="False")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
