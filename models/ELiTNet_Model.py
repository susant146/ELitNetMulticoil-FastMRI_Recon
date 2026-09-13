# import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- ELiTNet -------------------
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True):
        '''
        pool_mode can be False (no pooling) or True ('maxpool')
        '''
        super(ConvBlock, self).__init__()
        if shortcut==True: self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c))
        else: self.shortcut=False
        pad = (k_sz - 1) // 2

        block = []
        if pool: self.pool = nn.MaxPool2d(kernel_size=2)
        else: self.pool = False

        block.append(nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        block.append(nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        self.block = nn.Sequential(*block)
    def forward(self, x):
        if self.pool: x = self.pool(x)
        out = self.block(x)
        if self.shortcut: return out + self.shortcut(x)
        else: return out
        
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation = [1,2,3]):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        #self.relu = nn.ReLU(inplace=True)
        self.mish = nn.GELU()
        self.conv1 = nn.Conv2d(in_c, out_c // 4 , 1)
        self.bn2 = nn.BatchNorm2d(out_c // 4)
        #self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c // 4, out_c // 4, 3, padding='same')
        # self.conv3 = nn.Conv2d(out_c * 4, out_c, 3, padding='same', dilation=dilation[1])
        # self.conv4 = nn.Conv2d(out_c * 4, out_c, 3, padding='same', dilation=dilation[2])
        
        
        #self.conv2 = nn.Conv2d(output_channels/4, output_channels/4, 3, stride, padding = 1, bias = False)
        self.dropout = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm2d(out_c // 4)
        #self.relu = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(out_c // 4, out_c, 1, 1, bias = False)
        self.conv6 = nn.Conv2d(in_c, out_c , 1, 1, padding='same', bias = False)
        
    def forward(self, x):
        #residual = x
        #out = self.bn1(x)
        #out = self.mish(out)
        out = self.conv1(x)
        #out = self.bn2(out)
        out = self.mish(out)
        out = self.conv2(out)
        # out2 = self.conv3(out)
        # out3 = self.conv4(out)
        # out = torch.add(torch.add(out1,out2),out3)
        #out =  self.conv2(out)+ self.conv3(out)+ self.conv4(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = self.mish(out)
        out = self.conv5(out)
        
       
        #if (self.input_channels != self.output_channels) or (self.stride !=1 ):
        residual = self.conv6(x)
        out += residual
        return out


class AttConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True, attention=False):
        '''
        pool_mode can be False (no pooling) or True ('maxpool')
        '''
        super(AttConvBlock, self).__init__()
        if shortcut==True: self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c))
        else: self.shortcut=False
        pad = (k_sz - 1) // 2

        if pool: self.pool = nn.MaxPool2d(kernel_size=2)
        else: self.pool = False

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        if attention==True:
            self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            #self.softmax1_blocks = DiResBlock(in_c, out_c, dilation= [1,2,4])
            self.softmax1_blocks = nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding='same', dilation= 2)

            self.skip1_connection_residual_block = nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same')

            self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            #self.softmax2_blocks = DiResBlock(out_c, out_c, dilation= [2,4,8])
            self.softmax2_blocks = nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same', dilation= 4)

            self.skip2_connection_residual_block = nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same')

            self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.softmax3_blocks = nn.Sequential(
                nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same', dilation= 8),
                nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same', dilation= 8)
            )

            self.interpolation3 = nn.UpsamplingBilinear2d(scale_factor=2)

            #self.softmax4_blocks = DiResBlock(out_c, out_c, dilation= [2,4,8])
            self.softmax4_blocks = nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same', dilation= 4)

            self.interpolation2 = nn.UpsamplingBilinear2d(scale_factor=2)

            #self.softmax5_blocks = DiResBlock(out_c, out_c, dilation= [1,2,4])
            self.softmax5_blocks = nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same', dilation= 2)

            self.interpolation1 = nn.UpsamplingBilinear2d(scale_factor=2)

            self.softmax6_blocks = nn.Sequential(
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c , kernel_size = 1, stride = 1, bias = False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c , kernel_size = 1, stride = 1, bias = False),
                nn.Sigmoid()      # Commented Here ......20.30.25
            )

            self.last_blocks = nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same')
        
    def forward(self, x, attention = False):
        if self.pool: x = self.pool(x)
        out_trunk = self.conv(x)
        if attention==True:
            out_mpool1 = self.mpool1(x)
            out_softmax1 = self.softmax1_blocks(out_mpool1)
            out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
            out_mpool2 = self.mpool2(out_softmax1)
            out_softmax2 = self.softmax2_blocks(out_mpool2)
            #print(out_softmax2.data.shape)
            out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
            out_mpool3 = self.mpool3(out_softmax2)
            out_softmax3 = self.softmax3_blocks(out_mpool3)
            #
            out_interp3 = self.interpolation3(out_softmax3)
            #print(out_skip2_connection.data.shape)
            #print(out_interp3.data.shape)
            out = torch.add(out_interp3, out_skip2_connection)
            out_softmax4 = self.softmax4_blocks(out)
            out_interp2 = self.interpolation2(out_softmax4)
            out = torch.add(out_interp2, out_skip1_connection)
            out_softmax5 = self.softmax5_blocks(out)
            out_interp1 = self.interpolation1(out_softmax5)
            out_softmax6 = self.softmax6_blocks(out_interp1)
            #print(out_softmax6.shape)
            #print(out_trunk.shape)
            out = torch.multiply((1 + out_softmax6), out_trunk)
            out = self.last_blocks(out)
        else:
            out = out_trunk
        if self.shortcut: return out + self.shortcut(x)
        else: return out
        
        
class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, up_mode='transp_conv'):
        super(UpsampleBlock, self).__init__()
        block = []
        if up_mode == 'transp_conv':
            block.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
        elif up_mode == 'up_conv':
            block.append(nn.UpsamplingBilinear2d(scale_factor=2))
            block.append(nn.Conv2d(in_c, out_c, kernel_size=1))
        else:
            raise Exception('Upsampling mode not supported')

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out
    
class DoubleAttBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut= True, attention = True):
        super(DoubleAttBlock, self).__init__()
        

        self.block1 = AttConvBlock(in_c, in_c, k_sz=k_sz,
                              shortcut=shortcut, pool=False, attention=attention)
        self.block2 = AttConvBlock(in_c, out_c, k_sz=k_sz,
                              shortcut=shortcut, pool=True, attention=attention)

    def forward(self, x):
        out = self.block1(x, attention = True)
        out = self.block2(out, attention = True)
        return out

class ConvBridgeBlock(torch.nn.Module):
    def __init__(self, channels, k_sz=3):
        super(ConvBridgeBlock, self).__init__()
        # pad = (k_sz - 1) // 2
        # block=[]

        # block.append(nn.Conv2d(channels, channels, kernel_size=k_sz, padding=pad))
        # block.append(nn.ReLU())
        # block.append(nn.BatchNorm2d(channels))

        # self.block = nn.Sequential(*block)
        self.block = ResBlock(channels, channels)

    def forward(self, x):
        out = self.block(x)
        return out

class UpConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, up_mode='up_conv', conv_bridge=False, shortcut=False):
        super(UpConvBlock, self).__init__()
        self.conv_bridge = conv_bridge

        self.up_layer = UpsampleBlock(in_c, out_c, up_mode=up_mode)
        self.conv_layer1 = AttConvBlock(out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False, attention= True)
        self.conv_layer2 = AttConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False, attention= True)
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)

    def forward(self, x, skip):
        up = self.up_layer(x)
        up = self.conv_layer1(up, attention = True)
        #skip = torch.multiply((1 + up), skip)
        if self.conv_bridge:
            skip = self.conv_bridge_layer(skip)
            skip = torch.multiply((1 + up), skip)
            out = torch.cat([up, skip], dim=1) 
        else:
            skip = torch.multiply((1 + up), skip)
            out = torch.cat([up, skip], dim=1)
        out = self.conv_layer2(out, attention = True)
        return out

class ELiTNetModel(nn.Module):
    def __init__(self, in_c, n_classes, layers, k_sz=3, up_mode='up_conv', conv_bridge=True, shortcut=True):
        super(ELiTNetModel, self).__init__()
        self.n_classes = n_classes
        self.first = ConvBlock(in_c=in_c, out_c=layers[0], k_sz=k_sz,
                               shortcut=shortcut, pool=False)

        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            block = DoubleAttBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz,
                              shortcut=shortcut, attention=True)
            self.down_path.append(block)

        self.up_path = nn.ModuleList()
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            block = UpConvBlock(in_c=reversed_layers[i], out_c=reversed_layers[i + 1], k_sz=k_sz,
                                up_mode=up_mode, conv_bridge=conv_bridge, shortcut=shortcut)
            self.up_path.append(block)

        # init, shamelessly lifted from torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.final = nn.Conv2d(layers[0], n_classes, kernel_size=1)

    def forward(self, x):
        x = self.first(x)
        down_activations = []
        for i, down in enumerate(self.down_path):
            down_activations.append(x)
            x = down(x)
        down_activations.reverse()
        for i, up in enumerate(self.up_path):
            x = up(x, down_activations[i])
        return self.final(x)  
    

