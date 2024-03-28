import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vlnbert_init import get_vlnbert_models

class VLNBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the GOAT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None) 
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(self, mode, batch):
        batch = collections.defaultdict(lambda: None, batch)
        
        if mode == 'language':         
            txt_embeds = self.vln_bert(mode, batch)
            return txt_embeds

        elif mode == 'panorama':
            if not batch['already_dropout']:
                batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'reverie_obj_img_fts' in batch:
                batch['reverie_obj_img_fts'] = self.drop_env(batch['reverie_obj_img_fts'])
            pano_embeds, pano_masks, pano_fused_embeds = self.vln_bert(mode, batch)
            return pano_embeds, pano_masks, pano_fused_embeds

        else:
            outs = self.vln_bert(mode, batch)
            return outs

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()