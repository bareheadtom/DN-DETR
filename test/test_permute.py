import unittest
import torch
import sys
#sys.path.append('../../')
sys.path.append('/home/um202173632/suchangsheng/projects/DN-DETR')
import torch.nn as nn
from models.DN_DAB_DETR.dn_components import prepare_for_dn

from models.DN_DAB_DETR.dn_components2 import prepare_for_dn as prepare_for_cdn

class Testpermute(unittest.TestCase):
    def test_permute(self):
        src = torch.tensor([
            [[[1,2],[3,4]],[[11,12],[13,14]],[[21,22],[23,24]]]
        ])
        print(src.shape,src)
        bs,c,h,w = src.shape
        src = src.flatten(2).permute(2,0,1)
        print(src.shape,src)
        src = src.permute(1, 2, 0).view(bs, -1, h, w)
        print(src.shape,src)

if __name__ == "__main__":
    unittest.main()
