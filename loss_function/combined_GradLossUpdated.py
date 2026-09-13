import torch
import torch.nn as nn
from utils_gradlossUpdated import Gradient_Loss
from fastmri.losses import SSIMLoss 

class DualStreamLoss(nn.Module):
    def __init__(self, lambda_1=0.4, lambda_2=0.1, lambda_3=0.6):
        super(DualStreamLoss, self).__init__()
        self.L2_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.Grad_loss = Gradient_Loss()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

    def forward(self, ground_truth, predicted):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ssim_lossfunc = self.ssim_loss.to(device)
        loss_1 = self.L2_loss(ground_truth, predicted)
        loss_2 = self.Grad_loss(ground_truth, predicted)
        data_range = (ground_truth.max() - ground_truth.min()).view(1, 1, 1, 1).to(device)
        loss_3 = ssim_lossfunc(predicted, ground_truth, data_range=data_range)
        total_loss = self.lambda_1*loss_1 + self.lambda_2*loss_2 + self.lambda_3*loss_3
        return total_loss