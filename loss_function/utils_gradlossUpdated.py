import torch
import torch.nn as nn
import numpy as np

def img_gradient(img):
    # Get the number of channels from the input image
    channels = img.size(1)
    
    # # Define Sobel filters for x and y gradients
    # a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    # b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Consistent Gradient Operators
    a = np.array([[-0.112737, 0.000000, 0.112737],[-0.274526, 0.000000, 0.274526],[-0.112737,0.000000,0.112737]])
    b = np.transpose(a)
    
    # Create convolutional layer for the x gradient
    conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False, groups=channels)
    a = torch.from_numpy(a).float().unsqueeze(0)  # Shape [1, 3, 3]
    a = torch.cat([a] * channels).unsqueeze(1)    # Shape [channels, 1, 3, 3]
    conv1.weight = nn.Parameter(a, requires_grad=False)
    conv1 = conv1.to(img.device)
    G_x = conv1(img)

    # Create convolutional layer for the y gradient
    conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False, groups=channels)
    b = torch.from_numpy(b).float().unsqueeze(0)  # Shape [1, 3, 3]
    b = torch.cat([b] * channels).unsqueeze(1)    # Shape [channels, 1, 3, 3]
    conv2.weight = nn.Parameter(b, requires_grad=False)
    conv2 = conv2.to(img.device)
    G_y = conv2(img)

    return G_x, G_y

class Gradient_Loss(nn.Module):
    def __init__(self, Loss_criterion=nn.L1Loss()):
        super(Gradient_Loss, self).__init__()
        self.Loss_criterion = Loss_criterion

    def forward(self, imgClear, imgDenoised):
        clear_x, clear_y = img_gradient(imgClear)
        # clear_G = torch.sqrt(torch.pow(clear_x, 2) + torch.pow(clear_y, 2))
        pred_x, pred_y = img_gradient(imgDenoised)
        # pred_G = torch.sqrt(torch.pow(pred_x, 2) + torch.pow(pred_y, 2))
        gradient_loss = self.Loss_criterion(pred_x, clear_x) + self.Loss_criterion(pred_y, clear_y)
        return gradient_loss