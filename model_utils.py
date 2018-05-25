import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

OUTPUT_SIZE = 102
PRINT_EVERY = 40

def get_loaders(data_dir):
    """
    Return dataloaders for training, validation and teting datasets.
    """
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid/'
    test_dir = data_dir + '/test/'
    
    data_transforms = {
        'training' : transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])]),

        'validation' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]),

        'testing' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    }
    
    image_datasets = {
        'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
        'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }

    dataloaders = {
        'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=False),
        'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True)
    }
    
    class_to_idx = image_datasets['training'].class_to_idx
    return dataloaders, class_to_idx


def get_model(arch, hidden_units):
    '''
    Load pretrained model
    '''
    if arch == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == 'densenet201':
        model = models.densenet201(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = model.fc.in_features
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=True)
        input_size = model.fc.in_features
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        input_size = model.fc.in_features
    else:
        raise Exception("Unknown model")

    for param in model.parameters():
        param.requires_grad = False

    output_size = OUTPUT_SIZE

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    if 'vgg' in arch:
        model.classifier = classifier
    elif 'densenet' in arch:
        model.classifier = classifier
    elif 'resnet' in arch:
        model.fc = classifier

    return model

def build_model(arch, hidden_units, learning_rate):
    '''
    Build the pretrained model
    '''
    model = get_model(arch, hidden_units)
    print("Pretrained model retrieved.")
    
    # Set the parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)
    optimizer.zero_grad()
    criterion = nn.NLLLoss()
    return model, optimizer, criterion
    
def train_model(model, epochs, criterion, optimizer, train_loader,
                val_loader, use_gpu):
    '''
    Train the model
    '''
    print("Beginning model training")
    model.train()
    print_every = PRINT_EVERY
    steps = 0

    for epoch in range(epochs):
        running_loss = 0

        # Get inputs and labels
        for inputs, labels in iter(train_loader):
            steps += 1

            # Move to GPU
            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)

            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if steps % print_every == 0:
                val_loss, val_accuracy = validate(model, criterion, val_loader, use_gpu)
                model.train()
                print("Epoch: {}/{} ".format(epoch+1, epochs),
                      "Train Loss: {:.3f} ".format(running_loss/print_every),
                      "Val. Loss: {:.3f} ".format(val_loss),
                      "Val. Acc.: {:.3f}".format(val_accuracy))
                running_loss = 0

def validate(model, criterion, data_loader, use_gpu):
    ''' 
    Validate the model
    '''
    print("Validating the model")
    model.eval()
    accuracy = 0
    test_loss = 0
    for inputs, labels in iter(data_loader):
        if use_gpu:
            inputs = Variable(inputs.float().cuda(), volatile=True)
            labels = Variable(labels.long().cuda(), volatile=True)
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels, volatile=True)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).data[0]
        ps = torch.exp(output).data
        measure = (labels.data == ps.max(1)[1])
        accuracy += measure.type_as(torch.FloatTensor()).mean()

    return test_loss/len(data_loader), accuracy/len(data_loader)

    
