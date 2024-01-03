import unittest
import torch
import sys
#sys.path.append('../../')
sys.path.append('/root/autodl-fs/projects/DN-DETR')
import torch.nn as nn
from models.DN_DAB_DETR.dn_components import prepare_for_dn


class TestDnComponents(unittest.TestCase):
    def test_prepare_for_cdn(self):
        print("\n************test_prepare_for_dn")
        targets = [
            {"labels": torch.tensor([1, 2, 3]).to('cuda'), "boxes": torch.tensor([[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8],[0.1,0.2,0.3,0.4]]).to('cuda')},
            {"labels": torch.tensor([1, 2]).to('cuda'), "boxes": torch.tensor([[0.1,0.2,0.3,0.4],[0.2,0.4,0.6,0.8]]).to('cuda')}
        ]
        scalar = 5 
        label_noise_scale = 0.2
        box_noise_scale = 0.4
        num_patterns = 0
        refpoint_embed = nn.Embedding(300, 4)
        embedweight = refpoint_embed.weight
        embedweight = embedweight.to('cuda')
        num_queries = 300
        num_classes = 13
        hidden_dim=256
        label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1).to('cuda')
        dn_args = (targets,scalar,label_noise_scale,box_noise_scale,num_patterns)

        out = prepare_for_dn(dn_args=dn_args,
                        embedweight= embedweight,
                        batch_size=2,
                        training=True,
                        num_queries=num_queries,
                        num_classes=num_classes,
                        hidden_dim=hidden_dim,
                        label_enc=label_enc
                        )
        input_query_label, input_query_bbox, attn_mask, mask_dict = out
        #input_query_label, input_query_bbox, attn_mask, dn_meta = out
        print("input_query_label",input_query_label.shape)
        print("input_query_bbox",input_query_bbox.shape)
        print("attn_mask",attn_mask.shape)
        print("dn_meta",mask_dict)
        # input_query_label torch.Size([315, 2, 256])
        # input_query_bbox torch.Size([315, 2, 4])
        # attn_mask torch.Size([315, 315])
        # dn_meta {'known_indice': tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3,
        #         4], device='cuda:0'), 'batch_idx': tensor([0, 0, 0, 1, 1], device='cuda:0'), 'map_known_indice': tensor([ 0,  1,  2,  0,  1,  3,  4,  5,  3,  4,  6,  7,  8,  6,  7,  9, 10, 11,
        #         9, 10, 12, 13, 14, 12, 13]), 'known_lbs_bboxes': (tensor([1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 1,
        #         2], device='cuda:0'), tensor([[0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.2000, 0.4000, 0.6000, 0.8000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.2000, 0.4000, 0.6000, 0.8000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.2000, 0.4000, 0.6000, 0.8000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.2000, 0.4000, 0.6000, 0.8000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.2000, 0.4000, 0.6000, 0.8000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.2000, 0.4000, 0.6000, 0.8000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.2000, 0.4000, 0.6000, 0.8000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.2000, 0.4000, 0.6000, 0.8000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.2000, 0.4000, 0.6000, 0.8000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.1000, 0.2000, 0.3000, 0.4000],
        #         [0.2000, 0.4000, 0.6000, 0.8000]], device='cuda:0')), 'know_idx': [tensor([[0],
        #         [1],
        #         [2]], device='cuda:0'), tensor([[0],
        #         [1]], device='cuda:0')], 'pad_size': 15}

    def t1est_prepare_for_cdn2(self):
        print("\n************test_prepare_for_cdn2")
        targets = [
            {'boxes': torch.tensor([[0.0616, 0.5895, 0.1232, 0.5361]], device='cuda:0'), 
             'labels': torch.tensor([2], device='cuda:0'), 
             }
             ]
        scalar = 5 
        label_noise_scale = 0.2
        box_noise_scale = 0.4
        num_patterns = 0
        dn_args = (targets,scalar,label_noise_scale,box_noise_scale,num_patterns)
        refpoint_embed = nn.Embedding(300, 4)
        embedweight = refpoint_embed.weight
        embedweight = embedweight.to('cuda')
        num_queries = 300
        num_classes = 13
        hidden_dim=256
        label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1).to('cuda')

        out = prepare_for_dn(dn_args=dn_args,
                        embedweight= embedweight,
                        batch_size=1,
                        training=True,
                        num_queries=num_queries,
                        num_classes=num_classes,
                        hidden_dim=hidden_dim,
                        label_enc=label_enc
                        )
        input_query_label, input_query_bbox, attn_mask, mask_dict = out
        #input_query_label, input_query_bbox, attn_mask, dn_meta = out
        print("input_query_label",input_query_label.shape)
        print("input_query_bbox",input_query_bbox.shape)
        print("attn_mask",attn_mask.shape)
        print("dn_meta",mask_dict)
        

    def test_dn_post_process(self):
        print("\n************test_dn_post_process")
        
        

if __name__ == "__main__":
    unittest.main()
