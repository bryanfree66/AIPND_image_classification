import torch
from torchvision import models
from torch.autograd import Variable
import model_utils
from PIL import Image
import numpy as np


def load_checkpoint(checkpoint):
    '''
    Load the checkpoint file and build model
    '''
    state = torch.load(checkpoint)
    
    arch = state['arch']
    lr = float(state['learning_rate'])
    hidden_units = int(state['hidden_units'])
    
    model, optimizer, criterion = \
        model_utils.build_model(arch, hidden_units, lr)

    model.class_to_idx = state['class_to_idx']
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    return model

def process_image(image):
    '''
    Scales and crops the image
    '''
    size = 224
    # TODO: Process a PIL image for use in a PyTorch model
    width, height = image.size
    
    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)
        
    resized_image = image.resize((width, height))
        
    x0 = (width - size) / 2
    y0 = (height - size) / 2
    x1 = x0 + size
    y1 = y0 + size
    cropped_image = image.crop((x0, y0, x1, y1))
    np_image = np.array(cropped_image) / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])     
    np_image_array = (np_image - mean) / std
    np_image_array = np_image.transpose((2, 0, 1))
    
    return np_image_array

def predict(input_path, model, use_gpu, results_to_show, top_k):
    ''' 
    Predict classes for the image
    '''
    model.eval()
    image = Image.open(input_path)
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array)

    # Set GPU
    if use_gpu:
        var_inputs = Variable(tensor.float().cuda(), volatile=True)
    else:
        var_inputs = Variable(tensor, volatile=True).float()

    var_inputs = var_inputs.unsqueeze(0)
    output = model.forward(var_inputs)
    ps = torch.exp(output).data.topk(top_k)
    probs = ps[0].cpu() if use_gpu else ps[0]
    classes = ps[1].cpu() if use_gpu else ps[1]
    class_to_idx_inverted = {
        model.class_to_idx[k]: k for k in model.class_to_idx}
    classes_list = list()
    for label in classes.numpy()[0]:
        classes_list.append(class_to_idx_inverted[label])
    return probs.numpy()[0], classes_list