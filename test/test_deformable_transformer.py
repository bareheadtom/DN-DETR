import unittest
import torch
import sys
#sys.path.append('../../')
sys.path.append('/home/um202173632/suchangsheng/projects/DN-DETR')
import torch.nn as nn
from models.DN_DAB_DETR.dn_components import prepare_for_dn

from models.DN_DAB_DETR.dn_components2 import prepare_for_dn as prepare_for_cdn
device = 'cpu'
enc_output = nn.Linear(256, 256).to(device)
enc_output_norm = nn.LayerNorm(256).to(device)

def gen_encoder_output_proposals( memory, memory_padding_mask, spatial_shapes):
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        t = memory_padding_mask[:, _cur:(_cur + H_ * W_)]
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        t1 = ~mask_flatten_[:, :, 0, 0]
        t2 = ~mask_flatten_[:, 0, :, 0]
        valid_H = torch.sum(t1, 1)
        valid_W = torch.sum(t2, 1)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    output_memory = enc_output_norm(enc_output(output_memory))
    return output_memory, output_proposals
class Testpermute(unittest.TestCase):
    def t1est_permute(self):
        memory = torch.randn(2,4,256).to(device)
        memory_padding_mask  = torch.tensor([[False, True,False,True],[False,False,True,True]],dtype=bool)
        #torch.zeros(2,4, dtype=torch.bool).to(device) 
        spatial_shapes = torch.tensor([[ 2, 2]], device=device)

        output_memory, output_proposals = gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes)
        print(output_memory.shape, output_proposals.shape,output_proposals)
    
    def t1est_topk(self):
        enc_output_class = torch.tensor([[[0.5,0.6,0.1,0.4],[0.9,0.1,0.2,0.3],[0.1,0.2,0.3,0.9]]])
        print(enc_output_class.shape)
        max_1 = enc_output_class.max(-1).values
        print("max_1",max_1)
        print("enc_output_class.max(-1)[0]",enc_output_class.max(-1)[0])
        _,topk_ind = torch.topk(enc_output_class.max(-1).values, 2, dim = 1)
        print(topk_ind)
        max_11 = enc_output_class[...,0]
        print("max_11",max_11)
        _,topk_ind1 =  torch.topk(enc_output_class[...,0],2,dim=1)
        print(topk_ind1)
        
        #_,topk_ind = torch.topk()
    def t1est_override(self):
        a = torch.tensor([[[0.5,0.6,0.1,0.4],[0.9,0.1,0.2,0.3],[0.1,0.2,0.3,0.9]],[[0.5,0.6,0.1,0.4],[0.9,0.1,0.2,0.3],[0.1,0.2,0.3,0.9]],[[0.5,0.6,0.1,0.4],[0.9,0.1,0.2,0.3],[0.1,0.2,0.3,0.9]]])
        b = torch.tensor([[[1.5,1.6,1.1,0.4],[1.9,1.1,1.2,1.3],[1.1,1.2,1.3,1.9]],[[1.5,1.6,1.1,0.4],[1.9,1.1,1.2,1.3],[1.1,1.2,1.3,1.9]]])
        print(a.shape, b.shape)
        a[:2,:,:] = b 
        print(a)
    
    def test_transpose(self):
        a = torch.tensor([[1,2,3],[4,5,6]])
        print(a.shape,a)
        a1  = a.transpose(0,1)
        print(a1.shape,a1)
        a2 = a.permute(1,0)
        print(a2.shape,a2)


if __name__ == "__main__":
    unittest.main()
