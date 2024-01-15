import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np
import sys
#sys.path.append('../../')
sys.path.append('/home/um202173632/suchangsheng/projects/DN-DETR')
from models.DN_DAB_DETR.attention import MultiheadAttention

class enhance_net_nopool(nn.Module):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        #self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
            
        self.init_Conv = nn.Conv2d(3,32,3,1,1,bias=True)


		
    def forward(self, x):
        #x = self.relu(self.init_Conv(x))

        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

        

        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        return x_r
        # r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


        # x = x + r1*(torch.pow(x,2)-x)
        # x = x + r2*(torch.pow(x,2)-x)
        # x = x + r3*(torch.pow(x,2)-x)
        # enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
        # x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
        # x = x + r6*(torch.pow(x,2)-x)	
        # x = x + r7*(torch.pow(x,2)-x)
        # enhance_image = x + r8*(torch.pow(x,2)-x)
        # r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
        # return enhance_image_1,enhance_image,r


class MultiScaleRepresentation(nn.Module):
    def __init__(self):
        super(MultiScaleRepresentation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=4,padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2,padding=1)
        #self.upsample_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        #self.upsample_4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_8 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.fen_I = enhance_net_nopool()
        self.fen_Iq = enhance_net_nopool()
        self.fen_Io = enhance_net_nopool()

        self.self_attn = nn.MultiheadAttention(24, 8, dropout=0)
        self.conv1_post = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=7, stride=4,padding=1)
        self.conv2_post = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=2,padding=1)

        self.conv_final = nn.Conv2d(in_channels=24, out_channels=3, kernel_size=3, stride=1,padding=1)

    def forward(self, x):
        # Input x is a low-light RGB image I ∈ R^(H×W×C)
        # Applying the first convolution to generate Iq ∈ R^(H/4 × W/4 × 3)
        bs0,c0,h0,w0 = x.shape
        I = x
        Iq = self.conv1(x)
        # Applying the second convolution on Iq to generate Io ∈ R^(H/8 × W/8 × 3)
        Io = self.conv2(Iq)

        print("I.shape",I.shape,"Iq",Iq.shape,"Io",Io.shape)

        F1 = self.fen_I(I)
        Fq = self.fen_Iq(Iq)
        Fo = self.fen_Io(Io)
        print("F1.shape",F1.shape,"Fq",Fq.shape,"Fo",Fo.shape)

        Q = self.conv2_post(self.conv1_post(F1))
        K = self.conv2_post(Fq)
        print("Q.shape",Q.shape,"K",K.shape)
        bs,c,h,w = Q.shape
        Q = Q.flatten(2).permute(2,0,1)
        
        K = K.flatten(2).permute(2,0,1)
        
        out = self.self_attn(Q,K,K)[0]
        out= out.permute(1,2,0).view(bs,-1,h,w)
        print("out",out.shape)
        #enhanced = self.upsample_8(out) + self.upsample_8(Fo)
        enhanced = F.interpolate(out, size=[h0, w0], mode='bilinear', align_corners=False)
        enhanced += F.interpolate(Fo, size=[h0, w0], mode='bilinear', align_corners=False)
        
        return self.conv_final(enhanced) + x
# Usage
# Assuming input_image is your low-light RGB image tensor of shape (H, W, C)
# Reshape it to a tensor of shape (1, C, H, W) to fit into the Conv2d module
input_image = torch.randn(1, 3, 256, 256)  # Replace torch.randn with your actual input tensor

# Create an instance of MultiScaleRepresentation
model = MultiScaleRepresentation()

# Pass the input through the model
output = model(input_image)
print(output.shape)
