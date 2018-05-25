import os
import argparse
import torch
from torchvision import datasets, transforms
import model_utils

def get_command_line_args():
    parser = argparse.ArgumentParser()
    #-----Required Arguments----------
    parser.add_argument('data_dir', type=str,
                        help='Directory of flower images')
    
    #-----Optional Arguments----------
    parser.add_argument('--gpu', dest='gpu',
                        action='store_true', help='Train with GPU')
    parser.set_defaults(gpu=False)
    
    architectures = {'densenet121',
                     'densenet161',
                     'densenet201',
                     'resnet18',
                     'resnet34',
                     'resnet50',
                     'vgg13_bn',
                     'vgg16_bn',
                     'vgg19_bn',
                   }
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save checkpoints')
    
    parser.add_argument('--arch', dest='arch', default='densenet121', action='store',
                        choices=architectures,
                        help='Architecture to use')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Model learning rate')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of epochs to train')
    
    return parser.parse_args()


def save(arch, learning_rate, hidden_units, epochs, save_path, model, optimizer):
    ''' 
    Save the checkpoint
    '''
    state = {
        'arch': arch,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(state, save_path)

def main():
    
    # Get Command Line Arguments
    args = get_command_line_args()
    use_gpu = torch.cuda.is_available() and args.gpu
    print("Data directory: {}".format(args.data_dir))
    if use_gpu:
        print("Training on GPU.")
    else:
        print("Training on CPU.")
    print("Architecture: {}".format(args.arch))
    if args.save_dir:
        print("Checkpoint save directory: {}".format(args.save_dir))
    print("Learning rate: {}".format(args.learning_rate))
    print("Hidden units: {}".format(args.hidden_units))
    print("Epochs: {}".format(args.epochs))
    
    # Get data loaders
    dataloaders, class_to_idx = model_utils.get_loaders(args.data_dir)
    for key, value in dataloaders.items():
        print("{} data loader retrieved".format(key))
    
    # Build the model
    model, optimizer, criterion = model_utils.build_model(args.arch, args.hidden_units, args.learning_rate)
    model.class_to_idx = class_to_idx
    
    # Check if GPU availiable and move
    if use_gpu:
        print("GPU is availaible. Moving Tensors.")
        model.cuda()
        criterion.cuda()
    
    # Train the model
    model_utils.train_model(model, args.epochs, criterion, optimizer,
                       dataloaders['training'], dataloaders['validation'], use_gpu)
    
    # Save the checkpoint
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_path = args.save_dir + '/' + args.arch + '_checkpoint.pth'
    else:
        save_path = args.arch + '_checkpoint.pth'
    print("Will save checkpoint to {}".format(save_path))

    save(args.arch, args.learning_rate, args.hidden_units, args.epochs, save_path, model, optimizer)
    print("Checkpoint saved")

    # Validate the accuracy
    test_loss, accuracy = model_utils.validate(model, criterion, dataloaders['testing'], use_gpu)
    print("Test Loss: {:.3f}".format(test_loss))
    print("Test Acc.: {:.3f}".format(accuracy))
          
if __name__ == "__main__":
    main()