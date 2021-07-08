import torch, torchvision
import os
from models.unet import UNet

model_key = ['resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'vgg16',
             'fasterrcnn', 'unet']

def get_model(key):
    if key in model_key:
        modellist = {
            'resnet18': 'torchvision.models.resnet18(pretrained=True)',
            'resnet34': 'torchvision.models.resnet34(pretrained=True)',
            'resnet50': 'torchvision.models.resnet50(pretrained=True)',
            'resnet101': 'torchvision.models.resnet101(pretrained=True)',
            'resnet152': 'torchvision.models.resnet152(pretrained=True)',

            'vgg16': 'torchvision.models.vgg16(pretrained=True)',

            'fasterrcnn': 'torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)',

            'unet': 'UNet(n_channels=3, n_classes=1)'
        }
    else:
        raise ValueError(r"Don't support this model")

    model = eval(modellist[key])

    if key == 'unet':
        root_dir = os.getcwd()
        checkpoint = os.path.join(root_dir, 'models', 'unet', 'unet_carvana_scale1_epoch5.pth')
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cuda')))


    return model
