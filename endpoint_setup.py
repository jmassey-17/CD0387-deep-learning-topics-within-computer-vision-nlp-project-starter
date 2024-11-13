import subprocess
import logging
import json
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

# Install smdebug if not present
subprocess.call(['pip', 'install', 'smdebug'])

import smdebug

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def net(num_classes):
    """
    Initializes and returns a pretrained model with modified output layer.
    """
    model = models.resnet50(pretrained=True)
    
    # Freeze parameters in the pretrained model
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Update the final fully connected layer 
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256), 
        nn.ReLU(),  
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)
    )

    return model

def model_fn(model_dir):
    """
    Load the model from the directory specified by SageMaker.
    """
    logger.info(f'Loading model from {model_dir}')
    model = net(133)  
    
    model_path = os.path.join(model_dir, 'model.pt')
    try:
        # Load TorchScript model
        model = torch.jit.load(model_path, map_location=device)
        logger.info('Model loaded as TorchScript successfully.')
    except Exception as e:
        logger.error(f'Error loading TorchScript model: {e}')
        raise

    model.to(device).eval()
    return model

def input_fn(request_body, request_content_type):
    """
    Process the input data to the model.
    """
    logger.info('Starting input processing...')
    try:
        if request_content_type == 'application/json':
            data = json.loads(request_body)
            data = torch.tensor(data, dtype=torch.float32, device=device)
        elif request_content_type in ['image/jpeg', 'image/jpg']:
            data = Image.open(io.BytesIO(request_body))
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
        logger.info('Input processing complete.')
        return data
    except Exception as e:
        logger.error(f'Error in input_fn: {e}')
        raise

def predict_fn(input_object, model):
    """
    Perform inference on the input data using the model.
    """
    logger.info('Starting prediction...')
    try:
        prediction_transformation = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_object = prediction_transformation(input_object).unsqueeze(0).to(device)
        logger.info(f"Input shape: {input_object.shape}, dtype: {input_object.dtype}")
        
        with torch.no_grad():
            prediction = model(input_object)
        logger.info('Prediction complete. Output shape: %s', prediction.shape)
        return prediction
    except Exception as e:
        logger.error(f'Error in predict_fn: {e}')
        raise
