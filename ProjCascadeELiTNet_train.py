import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import fastmri
from torchsummary import summary
from torch.nn import functional as F
from common.args import Args
from data import transforms as T
from fastmri.data import transforms as T1
from model import Model
import scipy.io as sio
from fastmri.data import subsample
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from common.subsample import create_mask_for_mask_type
from combined_GradLossUpdated import DualStreamLoss
from fastmri.losses import SSIMLoss
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchmetrics.image import StructuralSimilarityIndexMeasure
# from ViT_utils import SSIM, PSNR
from data.mri_data import SliceData
from torch.utils.data import DataLoader
from ELiTNet_Model import ELiTNetModel
from unet_model import UnetModel
from dclayer.dclayer import updated_DClayer



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(f"Using {torch.cuda.device_count()} GPUs!")

#%% Projection Based Cascaded ELiTNet

# # ELiTNet_Cascade_Model
# class ELiTNet_Cascade_Model(nn.Module):
#     def __init__(self, num_chans, num_pools, drop_prob):
#         super().__init__()

#         self.in_chans = num_chans
#         self.num_pools = num_pools
#         self.drop_prob = drop_prob

#         self.elitnet = ELiTNetModel(
#             in_c=1, 
#             n_classes=1, 
#             layers=[8, 16, 32, 64], 
#             conv_bridge=True, 
#             shortcut=True
#         )

#         # self.unet_model = UnetModel(in_chans=5, 
#         #                             out_chans=1, 
#         #                             chans=num_chans, 
#         #                             num_pool_layers=num_pools, 
#         #                             drop_prob=drop_prob).cuda()
        
#         self.elitnet_model = ELiTNetModel(
#             in_c=5, 
#             n_classes=1, 
#             layers=[8, 16, 32, 64], 
#             conv_bridge=True, 
#             shortcut=True
#         )#.to(device)

#     def forward(self, input):
#         output = self.elitnet(input.unsqueeze(1)).squeeze(1)
#         output = torch.add(input, output)
#         return output
    
#     def last_layer(self, input):
#         output = self.elitnet_model(input).squeeze(1)
#         return output

#     def updatedcascad(self, input, mean, std, masked_kspace, inv_mask):
#         elitnet1 = self.forward(input)
#         outUDC1, int_output1 = updated_DClayer(elitnet1, masked_kspace, inv_mask, mean, std)

#         elitnet2 = self.forward(outUDC1)
#         outUDC2, int_output2 = updated_DClayer(elitnet2, masked_kspace, inv_mask, mean, std)

#         elitnet3 = self.forward(outUDC2)
#         outUDC3, int_output3 = updated_DClayer(elitnet3, masked_kspace, inv_mask, mean, std)

#         elitnet4 = self.forward(outUDC3)
#         outUDC4, int_output4 = updated_DClayer(elitnet4, masked_kspace, inv_mask, mean, std)

#         concat = torch.cat([outUDC4.unsqueeze(1), int_output1.unsqueeze(1),
#                             int_output2.unsqueeze(1), int_output3.unsqueeze(1),
#                             int_output4.unsqueeze(1)], 1)

#         elitnet5 = self.last_layer(concat)
#         output = torch.add(outUDC4, elitnet5)
#         outUDC5, _ = updated_DClayer(output, masked_kspace, inv_mask, mean, std)
#         return outUDC5


# ELiTNet_Cascade_Model
class ELiTNet_Cascade_Model(nn.Module):
    def __init__(self, num_chans, num_pools, drop_prob):
        super().__init__()

        self.in_chans = num_chans
        self.num_pools = num_pools
        self.drop_prob = drop_prob

        self.elitnet = UnetModel(
            in_chans=1, 
            out_chans=1, 
            chans=num_chans, 
            num_pool_layers=num_pools, 
            drop_prob=drop_prob
        )

        # self.unet_model = UnetModel(in_chans=5, 
        #                             out_chans=1, 
        #                             chans=num_chans, 
        #                             num_pool_layers=num_pools, 
        #                             drop_prob=drop_prob).cuda()
        
        self.elitnet_model = ELiTNetModel(
            in_c=5, 
            n_classes=1, 
            layers=[16, 32, 64, 128], 
            conv_bridge=True, 
            shortcut=True
        )#.to(device)

    def elitenet_forward(self, input):
        output = self.elitnet(input.unsqueeze(1)).squeeze(1)
        output = torch.add(input, output)
        return output
    
    def last_layer(self, input):
        output = self.elitnet_model(input).squeeze(1)
        return output

    # def forward(self, input, mean, std, masked_kspace, inv_mask):
    def forward(self, input, masked_kspace, inv_mask):
        input, mean, std = T.normalize_instance(input, eps=1e-11)
        # max_val = input.max()
        # max_val = max_val.view(-1, 1, 1)
        # input = input/(max_val + 1e-11)
        # input = torch.clamp(input, min=-6, max=6)

        elitnet1 = self.elitenet_forward(input)
        outUDC1, int_output1 = updated_DClayer(elitnet1, masked_kspace, inv_mask, mean, std)

        elitnet2 = self.elitenet_forward(outUDC1)
        outUDC2, int_output2 = updated_DClayer(elitnet2, masked_kspace, inv_mask, mean, std)

        elitnet3 = self.elitenet_forward(outUDC2)
        outUDC3, int_output3 = updated_DClayer(elitnet3, masked_kspace, inv_mask, mean, std)

        elitnet4 = self.elitenet_forward(outUDC3)
        outUDC4, int_output4 = updated_DClayer(elitnet4, masked_kspace, inv_mask, mean, std)

        concat = torch.cat([outUDC4.unsqueeze(1), int_output1.unsqueeze(1),
                            int_output2.unsqueeze(1), int_output3.unsqueeze(1),
                            int_output4.unsqueeze(1)], 1)

        elitnet5 = self.last_layer(concat)
        output = torch.add(outUDC4, elitnet5)
        outUDC5, _ = updated_DClayer(output, masked_kspace, inv_mask, mean, std)
        return outUDC5

#%% DataLoader and DataTransform
class DataTransform:
    """
    Data Transformer for training ELiTNet models.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice): #
        # if self.which_challenge == 'multicoil':
        #     image = fastmri.rss(target)
        image = T.to_tensor(target.astype(complex))
        k_space = T.fft2c_new(image)
        
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask,_ = T1.apply_mask(k_space, self.mask_func, seed)
        inv_mask = (1 - mask)
 
        zf = T.ifft2c_new(masked_kspace)
        dirty = T.complex_abs(zf).type(torch.FloatTensor)

        target = T.to_tensor(target.astype(np.float32))

        return dirty, target, fname, slice, masked_kspace,inv_mask, attrs['max']
    

def create_data_loaders(train_data_paths, val_data_path, mask_func, batch_size, resolution, challenge, sample_rate):
    if challenge not in ('singlecoil', 'multicoil'):
        raise ValueError(f"Challenge should be either 'singlecoil' or 'multicoil', got {challenge}")

    train_datasets = [
        SliceData(
            root=path,
            transform=DataTransform(resolution, challenge, mask_func, use_seed=True),
            challenge=challenge,
            sample_rate=sample_rate,
        )
        for path in train_data_paths
    ]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    val_dataset = SliceData(
        root=val_data_path,
        transform=DataTransform(resolution, challenge, mask_func, use_seed=True),
        challenge=challenge,
        sample_rate=sample_rate,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,drop_last=True)

    return train_loader, val_loader

#%%Train and Validation Epoch

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    start_epoch = time.perf_counter()
    # ssim_loss = SSIMLoss().to(device)
    # criterion=nn.L1Loss()
    loss_fn = fastmri.SSIMLoss().to(device)
    for batch in train_loader:
        dirty, target, _, _, masked_kspace, inv_mask, max_val = batch
        _, mean, std = T.normalize_instance(dirty, eps=1e-11)
        # dirty = torch.clamp(dirty, min=-6, max=6)
        target = T.normalize(target, mean, std, eps=1e-11)
        # target = torch.clamp(target, min=-6, max=6)
        max_val = max_val.to(device)
        dirty = dirty.to(device)
        target = target.to(device)
        data_range = target.max().view(-1, 1, 1, 1) - target.min().view(-1, 1, 1, 1)  # Ensure correct shape

        output = model(dirty, masked_kspace.to(device), inv_mask.to(device))
        target = target.unsqueeze(1)
        output = output.unsqueeze(1)

        # print('target max; ', target.max())
        # print('output max: ', output.max())
        # print('target min: ', target.min())
        # print('output min: ', output.min())

        loss = loss_fn(output, target, data_range=data_range)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
   
    # Calculate epoch duration
    epoch_duration = time.perf_counter() - start_epoch
    return total_loss / len(train_loader), epoch_duration

def validate_epoch(model, val_loader, device):
    model.eval()
    # criterion=nn.L1Loss()
    total_loss = 0
    start_epoch = time.perf_counter()
    avg_ssim = 0  # To track average SSIM over the validation dataset
    loss_fn = fastmri.SSIMLoss().to(device)

    with torch.no_grad():
        for batch in val_loader:
            dirty, target, _, _, masked_kspace, inv_mask, max_val = batch
            _, mean, std = T.normalize_instance(dirty, eps=1e-11)
            # dirty = torch.clamp(dirty, min=-6, max=6)
            target = T.normalize(target, mean, std, eps=1e-11)
            # target = torch.clamp(target, min=-6, max=6)
            max_val = max_val.to(device)
            dirty = dirty.to(device)
            target = target.to(device)
            data_range = target.max().view(-1, 1, 1, 1) - target.min().view(-1, 1, 1, 1) 
            # output = model(dirty, masked_kspace.to(device), inv_mask.to(device), torch.tensor([mean]).to(device), torch.tensor([std]).to(device))
            output = model(dirty, masked_kspace.to(device), inv_mask.to(device))
            target = target.unsqueeze(1)
            output = output.unsqueeze(1)
            loss = loss_fn(output, target, data_range=data_range)

            # loss=criterion(output,target)

            # Calculate SSIM separately for metric tracking
            output_np = output.squeeze().cpu().numpy()
            img_gt_np = target.squeeze().cpu().numpy()
            ssim_val = ssim(output_np, img_gt_np, data_range=img_gt_np.max()-img_gt_np.min()).item()
            
            # # Iterate over the batch
            # ssim_vals = []
            # for i in range(output_np.shape[0]):  # Loop through the batch
            #     # Ensure the win_size is appropriate for the given image dimensions
            #     min_dim = min(output_np[i].shape[-2:])  # Find the smallest dimension (H, W) for this image
            #     win_size = min(7, min_dim)  # Set win_size to 7 or smaller, ensuring it's odd
                
            #     if win_size % 2 == 0:  # Ensure win_size is odd
            #         win_size -= 1
                
            #     # Compute SSIM for this image
            #     ssim_val = ssim(output_np[i], img_gt_np[i], data_range=img_gt_np[i].max() - img_gt_np[i].min(), win_size=win_size)
            #     ssim_vals.append(ssim_val)

            # # Average SSIM across the batch
            mean_ssim = np.mean(ssim_val)
            avg_ssim += mean_ssim

            total_loss += loss.item()

    # Calculate epoch duration
    epoch_duration = time.perf_counter() - start_epoch
    avg_ssim /= len(val_loader)
    return total_loss / len(val_loader), avg_ssim, epoch_duration


# %% Main Function

if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # print(f"Using {torch.cuda.device_count()} GPUs!")

    hparams = {
        'lr': 0.001,
        'batch_size': 8,
        'max_epochs': 100,
        'lr_step_size': 80,
        'lr_gamma': 0.1,
                        }


    # Paths and Data Loader
    train_data_paths = [
        "/mnt/sdb/susant/Knee_Multicoil_train_batch0/multicoil_train",
        "/mnt/sdb/susant/Knee_Multicoil_train_batch1/multicoil_train"
    ]


    # print(f"Using {num_gpus} GPUs. Each GPU will process a batch of size {batch_size_per_gpu}.")

    val_data_path = "/mnt/sdb/susant/knee_multicoil_val_subset"
    mask_func = subsample.RandomMaskFunc(center_fractions=[0.04], accelerations=[8])
    train_loader, val_loader = create_data_loaders(train_data_paths, val_data_path, mask_func, 
                                                batch_size=8, 
                                                resolution=320, challenge='multicoil', sample_rate=1.0)
    
    # Model, Optimizer, and Scheduler
    model = ELiTNet_Cascade_Model(num_chans=32, num_pools=4, drop_prob=0.0)#.to(device)
    model = torch.nn.DataParallel(model)  # Then enable DataParallel
    model=model.to(device)
    
    optimizer = RMSprop(model.parameters(), lr=hparams['lr'], weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=hparams['lr_step_size'], gamma=hparams['lr_gamma'])


    # Save Model in a directory
    save_dir = "/mnt/sdb/susant/ProjBased_CascadeUNet/ProjCascadeUNet_savedModels"
    os.makedirs(save_dir, exist_ok=True)
    # Define a log file for losses
    log_file = os.path.join(save_dir, "PBCasELiTNet16_Acc8_trainlog.txt")
    with open(log_file, "w") as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\tVal_SSIM\tLR\n")  # Header for the log file
    
    # Variables to track the best model and early stopping
    best_ssim = 0  # Initialize best NMSE as infinity
    # Lists to store the losses
    val_ssim = []

    # Training Loop
    for epoch in range(hparams['max_epochs']):
        train_loss, train_time = train_epoch(model, train_loader, optimizer, device)
        val_loss, avg_val_ssim, val_time = validate_epoch(model, val_loader, device)
        val_ssim.append(avg_val_ssim)
        scheduler.step()

        # Log losses to the file
        with open(log_file, "a") as f:
            f.write(f"{epoch+1}\t{train_loss:.6f}\t{val_loss:.6f}\t{avg_val_ssim:.6f}\t{scheduler.get_last_lr()[0]:.7f}\n")

        # Print epoch results
        print(f"Epoch {epoch+1}/{hparams['max_epochs']}, "
              f"Train Loss: {train_loss:.4f}, Train Time: {train_time:.2f}s, "
              f"Val Loss: {val_loss:.4f}, Val SSIM: {avg_val_ssim:.6f}, Val Time: {val_time:.2f}s, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save the model with best Val SSIM
        if avg_val_ssim > best_ssim:
            best_ssim = avg_val_ssim
            best_model_path = os.path.join(save_dir, "CascadeELiTNet16_Acc8_best_l1b8_new.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with SSIM: {best_ssim:.4f} at {best_model_path}")
        
        # Save after 20 epochs
        if (epoch + 1) % 20 == 0:
            model_save_path = os.path.join(save_dir, f"CascadeELiTNet16_Acc8_epoch{epoch+1}_l1b8_new.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch+1} to {model_save_path}")

        # Save the last epoch
        if (epoch+1) == hparams['max_epochs']:
            save_path = os.path.join(save_dir, f"CascadeELiTNet16_Acc8_epoch{epoch+1}_l1b8_new.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")



#%% PBCasELitNet8 Reaumed 


# # Resuming the run model again

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(device)
#     print(f"Using {torch.cuda.device_count()} GPUs!")

#     hparams = {
#         'lr': 0.001,
#         'batch_size': 8,
#         'max_epochs': 120,
#         'lr_step_size': 20,
#         'lr_gamma': 0.1,
#                         }


#     # Paths and Data Loader
#     train_data_paths = [
#         "/mnt/sdb/susant/Knee_Multicoil_train_batch0/multicoil_train",
#         "/mnt/sdb/susant/Knee_Multicoil_train_batch1/multicoil_train"
#     ]

#     val_data_path = "/mnt/sdb/susant/knee_multicoil_val_subset"
#     save_dir = "/mnt/sdb/susant/ProjBased_CascadeUNet/ProjCascadeUNet_savedModels"
#     saved_model_path = "/mnt/sdb/susant/ProjBased_CascadeUNet/ProjCascadeUNet_savedModels/Acc4_ProjCasELiTNet8_b8/CascadeELiTNet8_Acc4_best_l1b8_new.pth"

#     # Hyperparameters
#     batch_size = 8
#     num_epochs = 120
#     resume_epoch = 41
#     learning_rate = 0.001
#     save_every = 20

#     # Create save directory if it doesn't exist
#     os.makedirs(save_dir, exist_ok=True)

#     # Dataloader
#     mask_func = subsample.RandomMaskFunc(center_fractions=[0.08], accelerations=[4])
#     train_loader, val_loader = create_data_loaders(train_data_paths, val_data_path, mask_func, 
#                                                 batch_size=8, 
#                                                 resolution=320, challenge='multicoil', sample_rate=1.0)
    
#     # Model, Optimizer, and Scheduler
#     model = ELiTNet_Cascade_Model(num_chans=32, num_pools=4, drop_prob=0.0)
#     model = nn.DataParallel(model, device_ids=[0, 1])
#     model.to(device)
#     optimizer = RMSprop(model.parameters(), lr=hparams['lr'], weight_decay=0.0)
#     scheduler = StepLR(optimizer, step_size=hparams['lr_step_size'], gamma=hparams['lr_gamma'])

#      # Load saved state
#     if os.path.exists(saved_model_path):
#         # Load the state dictionary
#         state_dict = torch.load(saved_model_path)
        
#         # Add 'module.' prefix to all keys if the model is wrapped in DataParallel
#         if isinstance(model, nn.DataParallel):
#             state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        
#         # Load the modified state dictionary into the model
#         model.load_state_dict(state_dict)
#         print(f"Loaded model weights from {saved_model_path}")
#     else:
#         raise FileNotFoundError(f"Saved model not found at {saved_model_path}")

#     # Load saved state
#     if os.path.exists(saved_model_path):
#         model.load_state_dict(torch.load(saved_model_path))
#         print(f"Loaded model weights from {saved_model_path}")
#     else:
#         raise FileNotFoundError(f"Saved model not found at {saved_model_path}")


#     # Logging setup
#     log_file = os.path.join(save_dir, "PBCasELitNet8_Acc4_trainlog_resumed.txt")
#     with open(log_file, "a") as f:
#         f.write("Epoch\tTrain_Loss\tVal_Loss\tVal_SSIM\tlearning_rate\n")

#     # train_losses = []
#     # val_losses = []
#     val_ssim = []
#     best_ssim = 0.813891  # Assuming you are tracking this during validation

#     # Training Loop
#     for epoch in range(hparams['max_epochs']):
#         train_loss, train_time = train_epoch(model, train_loader, optimizer, device)
#         val_loss, avg_val_ssim, val_time = validate_epoch(model, val_loader, device)
#         val_ssim.append(avg_val_ssim)
#         scheduler.step()

#         # Log losses to the file
#         with open(log_file, "a") as f:
#             f.write(f"{epoch+1}\t{train_loss:.6f}\t{val_loss:.6f}\t{avg_val_ssim:.6f}\t{scheduler.get_last_lr()[0]:.7f}\n")

#         # Print epoch results
#         print(f"Epoch {epoch+1}/{hparams['max_epochs']}, "
#               f"Train Loss: {train_loss:.4f}, Train Time: {train_time:.2f}s, "
#               f"Val Loss: {val_loss:.4f}, Val SSIM: {avg_val_ssim:.6f}, Val Time: {val_time:.2f}s, "
#               f"LR: {scheduler.get_last_lr()[0]:.6f}")

#         # Save the model with best Val SSIM
#         if avg_val_ssim > best_ssim:
#             best_ssim = avg_val_ssim
#             best_model_path = os.path.join(save_dir, "CascadeELiTNet8_Acc4_best_l1b1_resumed.pth")
#             torch.save(model.state_dict(), best_model_path)
#             print(f"Best model saved with SSIM: {best_ssim:.4f} at {best_model_path}")
        
#         # Save after 20 epochs
#         if (epoch + 1) % 20 == 0:
#             model_save_path = os.path.join(save_dir, f"CascadeELiTNet8_Acc4_epoch{epoch+1}_l1b1_resumed.pth")
#             torch.save(model.state_dict(), model_save_path)
#             print(f"Model saved at epoch {epoch+1} to {model_save_path}")

#         # Save the last epoch
#         if (epoch+1) == hparams['max_epochs']:
#             save_path = os.path.join(save_dir, f"CascadeELiTNet8_Acc4_epoch{epoch+1}_l1b1_resumed.pth")
#             torch.save(model.state_dict(), save_path)
#             print(f"Model saved at {save_path}")
