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
from torchsummary import summary
from torchstat import stat
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
import einops

class Mlp(nn.Module):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x, y):
        #print("x",x.shape)
        query = self.to_query(x)
        #print("query",query.shape)
        key = self.to_key(y)
        #print("key", key.shape)

        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        #print("query",query.shape)
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD
        #print("key", key.shape)

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        #print("query_weight", query_weight.shape)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        #print("A", A.shape)
        #print("A * query", A.shape, query.shape, (A * query).shape)
        G = torch.sum(A * query, dim=1) # BxD
        #print("G", G.shape)

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD
        #print("G repeat", G.shape)

        out = self.Proj(G * key) + query #BxNxD
        #print("out", out.shape)

        out = self.final(out) # BxNxD
        #print("out", out.shape)

        return out


class SwiftFormerLocalRepresentation(nn.Module):
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


class SwiftFormerEncoder(nn.Module):
    """
    SwiftFormer Encoder Block for SwiftFormer. It consists of (1) Local representation module, (2) EfficientAdditiveAttention, and (3) MLP block.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=False, layer_scale_init_value=1e-5):

        super().__init__()

        self.local_representation = SwiftFormerLocalRepresentation(dim=dim, kernel_size=3, drop_path=0.,
                                                                   use_layer_scale=True)
        self.attn = EfficientAdditiveAttnetion(in_dims=dim, token_dim=dim, num_heads=1)
        self.linear = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x, y):
        x = self.local_representation(x)
        y = self.local_representation(y)
        B, C, H, W = x.shape
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1 * self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C), y.permute(0, 2, 3, 1).reshape(B, H * W, C)).reshape(B, H, W, C).permute(
                    0, 3, 1, 2))
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))

        else:
            x = x + self.drop_path(
                self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C), y.permute(0, 2, 3, 1).reshape(B, H * W, C)).reshape(B, H, W, C).permute(0, 3, 1, 2))
            x = x + self.drop_path(self.linear(x))
        return x

class enhance_net_nopool(nn.Module):

    def __init__(self,in_c=3, out_c=24):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(in_c,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,out_c,3,1,1,bias=True) 

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

        self.self_attn = nn.MultiheadAttention(24, 1, dropout=0)
        self.conv1_post = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=7, stride=4,padding=1)
        self.conv2_post = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=2,padding=1)

        self.conv_final = nn.Conv2d(in_channels=24, out_channels=3, kernel_size=3, stride=1,padding=1)

        self.swiftFormerEncoder = SwiftFormerEncoder(24)

    def forward(self, x):
        # Input x is a low-light RGB image I ∈ R^(H×W×C)
        # Applying the first convolution to generate Iq ∈ R^(H/4 × W/4 × 3)
        bs0,c0,h0,w0 = x.shape
        I = x
        Iq = self.conv1(x)
        # Applying the second convolution on Iq to generate Io ∈ R^(H/8 × W/8 × 3)
        Io = self.conv2(Iq)

        #print("I.shape",I.shape,"Iq",Iq.shape,"Io",Io.shape)

        F1 = self.fen_I(I)
        Fq = self.fen_Iq(Iq)
        Fo = self.fen_Io(Io)
        #print("F1.shape",F1.shape,"Fq",Fq.shape,"Fo",Fo.shape)

        Q = self.conv2_post(self.conv1_post(F1))
        K = self.conv2_post(Fq)
        #print("Q.shape",Q.shape,"K",K.shape)
        bs,c,h,w = Q.shape
        #Q = Q.flatten(2).permute(2,0,1)
        
        #K = K.flatten(2).permute(2,0,1)
        
        #out = self.self_attn(Q,K,K)[0]
        #out= out.permute(1,2,0).view(bs,-1,h,w)
        out = self.swiftFormerEncoder(Q,K)
        #print("out",out.shape)
        #enhanced = self.upsample_8(out) + self.upsample_8(Fo)
        enhanced = F.interpolate(out, size=[h0, w0], mode='bilinear', align_corners=False)
        enhanced += F.interpolate(Fo, size=[h0, w0], mode='bilinear', align_corners=False)

        # out = Q
        # out= out.permute(1,2,0).view(bs,-1,h,w)
        # enhanced = F.interpolate(out, size=[h0, w0], mode='bilinear', align_corners=False)
        
        return self.conv_final(enhanced) + x
# Usage
# Assuming input_image is your low-light RGB image tensor of shape (H, W, C)
# Reshape it to a tensor of shape (1, C, H, W) to fit into the Conv2d module
input_image = torch.randn(1, 3, 756, 1256).to('cuda')  # Replace torch.randn with your actual input tensor

# Create an instance of MultiScaleRepresentation
model = MultiScaleRepresentation().to('cuda')

# Pass the input through the model
output = model(input_image)
print("output",output.shape)

print(torch.cuda.memory_allocated()/ 1024**3)
# model.to('cpu')
# stat(model,(3, 756, 1256))
