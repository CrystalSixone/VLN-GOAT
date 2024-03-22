# GOAT
import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaLayer

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad
from typing import Optional

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
)
logger = logging.getLogger(__name__)


from .Bert_backbone import *

class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [RobertaLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_embeds, txt_masks):
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        return txt_embeds

class LanguageEncoderDo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [RobertaLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False
        
        # For intervention
        if self.config.do_back_txt or self.config.do_front_txt:
            self.z_txt_linear = nn.Linear(config.hidden_size,config.hidden_size)
            self.z_direct_linear = nn.Linear(config.hidden_size,config.hidden_size)
            self.z_landm_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.z_concat_layernorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.z_direct_ln = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.z_landm_ln = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if self.config.do_back_txt_type == 'type_2':
                self.z_direc_cross_attn = RobertaAttention(config)
                self.z_landm_cross_attn = RobertaAttention(config)
                self.txt_self_attn = RobertaAttention(config)
                self.instr_aug_linear = nn.Linear(config.hidden_size,1)
                self.instr_ori_linear = nn.Linear(config.hidden_size,1)
                self.instr_sigmoid = nn.Sigmoid()
                self.concat_linear = nn.Linear(config.hidden_size*3, config.hidden_size)
                
        # Frontdoor Intervention
        if config.do_front_txt:
            self.z_front_cross_attn = RobertaAttention(config)
            self.z_front_linear = nn.Linear(config.hidden_size,config.hidden_size)
            self.z_front_ln = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, txt_embeds, txt_masks, z_direc_embeds=None, z_direc_pzs=None, z_landm_embeds=None, z_landm_pzs=None,
                front_txt_embeds=None):
        extended_txt_masks = extend_neg_masks(txt_masks)
        # BERT
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]

        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        
        # Causal Intervention
        if self.config.do_back_txt or self.config.do_front_txt:
            if self.config.do_back_txt_type == 'type_1':
                if self.config.do_back_txt:
                    p_z_direct = z_direc_embeds * z_direc_pzs.to(torch.float32) 
                    sum_z_direct = torch.sum(p_z_direct,1).unsqueeze(1)  
                    p_z_landm = z_landm_embeds * z_landm_pzs.to(torch.float32) 
                    sum_z_landm = torch.sum(p_z_landm,1).unsqueeze(1) 
                    txt_embeds = self.z_txt_linear(txt_embeds) + self.z_direct_linear(sum_z_direct) + self.z_landm_linear(sum_z_landm)
                if self.config.do_front_txt and front_txt_embeds is not None:
                    # Frontdoor Intervention
                    z_front_embeds = self.z_front_cross_attn(txt_embeds, 
                                                            encoder_hidden_states=front_txt_embeds)[0]
                    z_front_embeds = self.z_front_ln(self.z_front_linear(z_front_embeds))
                    txt_embeds = txt_embeds + z_front_embeds
                txt_embeds = self.z_concat_layernorm(txt_embeds)
            elif self.config.do_back_txt_type == 'type_2':
                if self.config.do_back_txt:
                    z_direc_embeds = self.z_direc_cross_attn(txt_embeds, 
                                                            encoder_hidden_states=z_direc_embeds)[0] 
                    z_direc_embeds = self.z_direct_ln(self.z_direct_linear(z_direc_embeds))
                    if z_landm_embeds is not None:
                        z_landm_embeds = self.z_landm_cross_attn(txt_embeds, 
                                                                encoder_hidden_states=z_landm_embeds)[0]
                        z_landm_embeds = self.z_landm_ln(self.z_landm_linear(z_landm_embeds))
                
                if self.config.do_front_txt and front_txt_embeds is not None:
                    # Frontdoor Intervention
                    z_front_embeds = self.z_front_cross_attn(txt_embeds, 
                                                            encoder_hidden_states=front_txt_embeds)[0] 
                    z_front_embeds = self.z_front_ln(self.z_front_linear(z_front_embeds))
                
                if self.config.do_add_method == 'door':
                    if self.config.do_back_txt:
                        instr_aug_embeds = z_direc_embeds
                        if z_landm_embeds is not None:
                            instr_aug_embeds = instr_aug_embeds + z_landm_embeds
                        if front_txt_embeds is not None:
                            instr_aug_embeds = instr_aug_embeds + z_front_embeds
                    elif self.config.do_front_txt and front_txt_embeds is not None:
                        instr_aug_embeds = z_front_embeds
                        
                    aug_linear_weight = self.instr_aug_linear(instr_aug_embeds)
                    ori_linear_weight = self.instr_ori_linear(txt_embeds)
                    aug_weight = self.instr_sigmoid(aug_linear_weight+ori_linear_weight)
                    txt_embeds = torch.mul(aug_weight,instr_aug_embeds) + torch.mul((1-aug_weight),txt_embeds)  
                elif self.config.do_add_method == 'add':
                    if self.config.do_back_txt:
                        txt_embeds = txt_embeds + z_direc_embeds + z_landm_embeds
                    if self.config.do_front_txt and front_txt_embeds is not None:
                        txt_embeds = txt_embeds + z_front_embeds
                elif self.config.do_add_method == 'concat':
                    concat_txt_embeds = torch.cat((txt_embeds, z_direc_embeds, z_landm_embeds),-1)
                    txt_embeds = self.concat_linear(concat_txt_embeds)
                
                txt_embeds = self.z_concat_layernorm(txt_embeds)
                    
        return txt_embeds

class CausalImageEmbeddings(nn.Module):
    ''' Causal learning
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config

        ''' For interventional image
        '''
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size+3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        
        if config.name not in ['REVERIE','SOON']:
            self.img_self_attn = BertAttention(config)
            self.img_self_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
            
        # Back-adjustment
        self.do_back_img = config.do_back_img
        if self.do_back_img:
            self.do_img_before_linear = nn.Linear(config.image_feat_size, config.hidden_size)
            self.do_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.do_img_attn = BertAttention(config)
            self.do_img_after_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.img_after_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.do_img_concat_layernorm = BertLayerNorm(config.hidden_size, eps=1e-12)

            if self.config.do_back_img_type == 'type_2':
                if self.config.do_add_method == 'door':
                    self.sigmoid = nn.Sigmoid()
                elif self.config.do_add_method == 'concat':
                    self.do_concat_img_linear = nn.Linear(config.hidden_size*2, config.hidden_size)
        
        img_obj_config = copy.copy(config)
        img_obj_config.num_top_layer = config.num_pano_layers
        self.img_obj_attn = CrossmodalEncoder(img_obj_config)
        
        '''For reverie'''
        if self.config.name == 'REVERIE' or self.config.name == 'SOON':
            if self.config.use_obj_name:
                self.obj_name_linear = nn.Embedding(config.obj_name_vocab_size, config.hidden_size)
            self.obj_reverie_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_reverie_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.nav_type_embedding = nn.Embedding(3, config.hidden_size)
            self.pano_encoder = create_transformer_encoder(
                        config, config.num_pano_layers, norm=True
                    )
        else:
            self.nav_type_embedding = nn.Embedding(2, config.hidden_size)
            
        '''For global map aggregation
        '''
        if config.adaptive_pano_fusion: 
            self.adaptive_pano_attn = nn.Linear(config.hidden_size,1) # 768 -> 1
            self.adaptive_pano_act = ACT2FN[config.hidden_act]
            self.adaptive_softmax = nn.Softmax(dim=1)

        # 0: objects, 1: navigable
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self, traj_view_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, type_embed_layer, 
        traj_reverie_obj_fts=None, traj_reverie_obj_lens=None,
        z_img_features=None, z_img_pzs=None
    ):
        ''' Image & Object encoding
        '''
        view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))
            
        if self.config.name not in ['REVERIE', 'SOON']:
            view_img_embeds = view_img_embeds + self.loc_layer_norm(self.loc_linear(traj_loc_fts))
        
        img_masks = gen_seq_masks(traj_vp_view_lens)
        extended_img_masks = extend_neg_masks(img_masks)

        if z_img_features is not None:
            # Do intervention
            if self.config.do_back_img_type == 'type_1':
                z_img_embeds = self.do_img_layer_norm(self.do_img_before_linear(z_img_features))
                p_z_img = z_img_embeds * z_img_pzs.to(torch.float32)
                sum_z_img = torch.sum(p_z_img,1).unsqueeze(1) #[bs,1,dim]
                view_img_embeds = self.img_after_linear(view_img_embeds) + self.do_img_after_linear(sum_z_img)
                view_img_embeds = self.do_img_concat_layernorm(view_img_embeds)
            
            elif self.config.do_back_img_type == 'type_2':
                z_img_embeds = self.do_img_layer_norm(self.do_img_before_linear(z_img_features))
                z_img_embeds = self.do_img_attn(view_img_embeds,encoder_hidden_states=z_img_embeds)[0]

                if self.config.do_add_method == 'door':
                    ori_img_embeds = self.img_after_linear(view_img_embeds)
                    aug_z_img_embeds = self.do_img_after_linear(z_img_embeds)
                    aug_img_weight = self.sigmoid(ori_img_embeds+aug_z_img_embeds)
                    view_img_embeds = torch.mul(aug_img_weight, view_img_embeds) + torch.mul((1-aug_img_weight),z_img_embeds)
                elif self.config.do_add_method == 'add':
                    view_img_embeds = view_img_embeds + z_img_embeds
                elif self.config.do_add_method == 'concat':
                    concat_img_embeds = torch.cat((view_img_embeds, z_img_embeds),-1)
                    view_img_embeds = self.do_concat_img_linear(concat_img_embeds)

                view_img_embeds = self.do_img_concat_layernorm(view_img_embeds)
        
        if self.config.name not in ['REVERIE', 'SOON']:
            img_embeds = self.dropout(view_img_embeds)
            view_img_embeds = self.img_self_encoder(
                img_embeds, src_key_padding_mask=img_masks.logical_not()
            )

        '''For REVERIE'''
        if traj_reverie_obj_fts is not None:
            reverie_obj_img_embeds = self.obj_reverie_layer_norm(
                self.obj_reverie_linear(traj_reverie_obj_fts))

            img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                    view_img_embeds, reverie_obj_img_embeds, traj_vp_view_lens, traj_reverie_obj_lens
                ):
                if obj_len > 0:
                    img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    img_embeds.append(view_embed[:view_len])
            img_embeds = pad_tensors_wgrad(img_embeds)
            traj_vp_view_lens = traj_vp_view_lens + traj_reverie_obj_lens

            fused_img_reverie_embeds =  img_embeds +\
                    self.nav_type_embedding(traj_nav_types) +\
                    self.loc_layer_norm(self.loc_linear(traj_loc_fts))
                
            img_masks = gen_seq_masks(traj_vp_view_lens)
            extended_img_masks = extend_neg_masks(img_masks)
            img_reverie_embeds = self.dropout(fused_img_reverie_embeds)
            view_img_embeds = self.pano_encoder(
                img_reverie_embeds, src_key_padding_mask=img_masks.logical_not()
            )
        
        split_traj_embeds = torch.split(view_img_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_view_lens, traj_step_lens, 0)

        if self.config.adaptive_pano_fusion:
            traj_ori_embeds = view_img_embeds.clone()
            traj_fused_weight = self.adaptive_pano_attn(traj_ori_embeds) 
            traj_fused_weight_act = torch.tanh(traj_fused_weight) 
            traj_fused_weight_act = self.adaptive_softmax(traj_fused_weight_act)
            traj_fused_embeded_update = torch.mul(traj_ori_embeds,traj_fused_weight_act)
            traj_fused_embeds = torch.sum(traj_fused_embeded_update,dim=1)
            split_traj_fused_embeds = torch.split(traj_fused_embeds, traj_step_lens, 0)
            return split_traj_embeds, split_traj_vp_lens, split_traj_fused_embeds
        
        return split_traj_embeds, split_traj_vp_lens, None
        
class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.encoder = CrossmodalEncoder(config)

        if config.do_front_img or config.mode == 'extract_cfp_features':
            self.tim_self_encoder = BertAttention(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        vp_lens = torch.stack([x[-1]+1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks
    
    def vp_input_embedding_mlm(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts, split_traj_obj_lens=None):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        vp_lens = torch.stack([x[-1]+1 for x in split_traj_vp_lens], 0)
        if split_traj_obj_lens is not None:
            obj_lens = torch.stack([x[-1] for x in split_traj_obj_lens], 0)
            total_lens = vp_lens + obj_lens
        else:
            total_lens = vp_lens
        vp_masks = gen_seq_masks(total_lens)
        max_vp_len = max(total_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_pos_fts = vp_pos_fts[:,:max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks

    def forward(
        self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_embeds = self.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
        return vp_embeds

    def forward_cfp(
        self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_masks = extend_neg_masks(vp_masks)
        vp_embeds = self.tim_self_encoder(vp_embeds, vp_masks)[0]
        return vp_embeds

class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)
        
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None
        
        if config.do_front_his or config.mode == 'extract_cfp_features':
            self.tim_self_encoder = BertAttention(config)

    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        split_traj_fused_embeds=None, seperate_his=False
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                if split_traj_fused_embeds is not None:
                    visited_vp_fts[traj_vpids[i][t]] = split_traj_fused_embeds[i][t]
                else:
                    visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            gmap_img_fts = []
            start_id = 2 if seperate_his else 1
            for vp in gmap_vpids[i][start_id:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        if seperate_his:
            batch_gmap_img_fts = torch.cat(
                [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), 
                torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device),
                batch_gmap_img_fts], dim=1)
        else:
            batch_gmap_img_fts = torch.cat(
                [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts], 
                dim=1
            )
            
        return batch_gmap_img_fts
    
    def gmap_input_embedding(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens,
        split_traj_fused_embeds=None, seperate_his=False
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            split_traj_fused_embeds=split_traj_fused_embeds,seperate_his=seperate_his
        )
        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        gmap_masks = gen_seq_masks(gmap_lens)
        return gmap_embeds, gmap_masks

    def forward(
        self, txt_embeds, txt_masks,
        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
        return gmap_embeds
    
    def forward_cfp(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None,
        split_traj_fused_embeds=None
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens, split_traj_fused_embeds=split_traj_fused_embeds
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_masks = extend_neg_masks(gmap_masks)
        gmap_self_embeds = self.tim_self_encoder(
            gmap_embeds, gmap_masks
        )[0]

        return gmap_self_embeds     
    
class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None, output_size=1):
        super().__init__()
        if input_size is None:
            input_size = hidden_size

        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                BertLayerNorm(hidden_size, eps=1e-12),
                                nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.net(x)

class FrontDoorEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ll_self_attn = BertAttention(config)
        self.lg_cross_attn = BertAttention(config)
        self.ln = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.config = config

        self.aug_linear = nn.Linear(config.hidden_size,1)
        self.ori_linear = nn.Linear(config.hidden_size,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, local_feats, global_feats, local_feats_masks=None):
        '''
        :local_feats: input's hidden_states
        :global_feats: KMeans's hidden_states
        '''
        if local_feats_masks is not None and len(local_feats_masks.shape) != 4:
            local_feats_masks = extend_neg_masks(local_feats_masks)
        ll_feats = self.ll_self_attn(local_feats,attention_mask=local_feats_masks)[0]
        lg_feats = self.lg_cross_attn(hidden_states=local_feats, encoder_hidden_states=global_feats)[0]
        out_feats = self.ln(ll_feats + lg_feats)

        aug_linear_weight = self.aug_linear(out_feats)
        ori_linear_weight = self.ori_linear(local_feats)
        aug_weight = self.sigmoid(aug_linear_weight+ori_linear_weight)
        out_feats = torch.mul(aug_weight, out_feats) + torch.mul((1-aug_weight), local_feats)

        return out_feats

class GlocalTextPathNavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = RobertaEmbeddings(config)
        if config.do_back_txt or config.do_front_txt:
            self.lang_encoder = LanguageEncoderDo(config)
        else:
            self.lang_encoder = LanguageEncoder(config)

        self.img_embeddings = CausalImageEmbeddings(config)
        
        self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)

        self.global_sap_head = ClsPrediction(self.config.hidden_size)
        self.local_sap_head = ClsPrediction(self.config.hidden_size)
        if config.glocal_fuse:
            self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
        else:
            self.sap_fuse_linear = None
        if self.config.obj_feat_size > 0:
            self.og_head = ClsPrediction(self.config.hidden_size)
            
        self.object_encoder = None
        self.extra_drop = nn.Dropout(0.2)

        self.gmap_pooler = BertPooler(config)
        self.vp_pooler = BertPooler(config)
        self.txt_pooler = BertPooler(config)
        self.local_his_map = nn.Linear(config.hidden_size*3, config.hidden_size)
        self.local_his_ln = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.drop_env = nn.Dropout(p=config.feat_dropout)
        
        if config.do_front_img or config.mode == 'extract_cfp_features':
            self.tim_local_head = BertPredictionHeadTransform(self.config)
            self.tim_local_attn = nn.Parameter(torch.Tensor(self.config.hidden_size,1))
            self.temperature = self.config.cfp_temperature
            self.front_local_encoder = FrontDoorEncoder(config)
            
        if config.do_front_his or config.mode == 'extract_cfp_features':
            self.tim_global_head = BertPredictionHeadTransform(self.config)
            self.tim_global_attn = nn.Parameter(torch.Tensor(self.config.hidden_size,1))
            self.temperature = self.config.cfp_temperature
            self.front_global_encoder = FrontDoorEncoder(config)
            
        if config.do_front_txt or config.mode == 'extract_cfp_features':
            self.tim_txt_head = BertPredictionHeadTransform(self.config)
            self.tim_txt_attn = nn.Parameter(torch.Tensor(self.config.hidden_size,1))
            self.temperature = self.config.cfp_temperature
            self.front_txt_encoder = FrontDoorEncoder(config)

        self.init_weights()
        
        if config.fix_lang_embedding or config.fix_local_branch:
            for k, v in self.embeddings.named_parameters():
                v.requires_grad = False
            for k, v in self.lang_encoder.named_parameters():
                v.requires_grad = False
        if config.fix_pano_embedding or config.fix_local_branch:
            for k, v in self.img_embeddings.named_parameters():
                v.requires_grad = False
        if config.fix_local_branch:
            for k, v in self.local_encoder.named_parameters():
                v.requires_grad = False
            for k, v in self.local_sap_head.named_parameters():
                v.requires_grad = False
            for k, v in self.og_head.named_parameters():
                v.requires_grad = False
    
    def forward_text(self, txt_ids, txt_masks, instr_z_direction_features=None, instr_z_direction_pzs=None, instr_z_landmark_features=None, instr_z_landmark_pzs=None, front_txt_embeds=None):
        txt_token_type_ids = torch.zeros_like(txt_ids)
        if self.config.do_back_txt or self.config.do_front_txt: # intervention
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
            txt_embeds = self.lang_encoder(txt_embeds, txt_masks,
                                           instr_z_direction_features, instr_z_direction_pzs, instr_z_landmark_features, 
                                           instr_z_landmark_pzs, front_txt_embeds
                                        )
        else:
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
            txt_embeds = self.lang_encoder(txt_embeds, txt_masks)
        return txt_embeds
    
    def instr_feature_extractor(self, txt_ids, txt_masks):
        ''' Update instruction's z-dict
        '''
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)[0]
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks)
        return txt_embeds

    def forward_panorama_do_per_step(self,view_img_fts, loc_fts, nav_types, view_lens,
            z_img_features=None, z_img_pzs=None,
            reverie_obj_fts=None, reverie_obj_lens=None,
            reverie_obj_names=None
            ):
        ''' Image encoding
        '''
        img_masks = gen_seq_masks(view_lens)
        view_img_embeds = self.img_embeddings.img_layer_norm(self.img_embeddings.img_linear(view_img_fts))
        
        if z_img_features is not None:
            # Do intervention
            if self.config.do_back_img_type == 'type_1':
                z_img_embeds = self.img_embeddings.do_img_layer_norm(self.img_embeddings.do_img_before_linear(z_img_features))
                
                p_z_img = z_img_embeds * z_img_pzs.to(torch.float32)
                sum_z_img = torch.sum(p_z_img,1).unsqueeze(1)
                view_img_embeds = self.img_embeddings.img_after_linear(view_img_embeds) + self.img_embeddings.do_img_after_linear(sum_z_img)
                view_img_embeds = self.img_embeddings.do_img_concat_layernorm(view_img_embeds)
            elif self.config.do_back_img_type == 'type_2':
                z_img_embeds = self.img_embeddings.do_img_layer_norm(self.img_embeddings.do_img_before_linear(z_img_features))
                z_img_embeds = self.img_embeddings.do_img_attn(view_img_embeds,encoder_hidden_states=z_img_embeds)[0]

                if self.config.do_add_method == 'door':
                    ori_img_embeds = self.img_embeddings.img_after_linear(view_img_embeds)
                    aug_z_img_embeds = self.img_embeddings.do_img_after_linear(z_img_embeds)
                    aug_img_weight = self.img_embeddings.sigmoid(ori_img_embeds+aug_z_img_embeds)
                    view_img_embeds = torch.mul(aug_img_weight, view_img_embeds) + torch.mul((1-aug_img_weight),z_img_embeds)
                elif self.config.do_add_method == 'add':
                    view_img_embeds = view_img_embeds + z_img_embeds
                elif self.config.do_add_method == 'concat':
                    concat_img_embeds = torch.cat((view_img_embeds, z_img_embeds),-1)
                    view_img_embeds = self.img_embeddings.do_concat_img_linear(concat_img_embeds)

                view_img_embeds = self.img_embeddings.do_img_concat_layernorm(view_img_embeds)
        
        img_masks = gen_seq_masks(view_lens)
        extended_img_masks = extend_neg_masks(img_masks)
        
        if self.config.name not in ['REVERIE', 'SOON']:
            view_img_embeds = view_img_embeds +\
                            self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts))

            view_img_embeds = self.img_embeddings.dropout(view_img_embeds)
            view_img_embeds = self.img_embeddings.img_self_encoder(
                view_img_embeds, src_key_padding_mask=img_masks.logical_not()
            )
        
        '''For REVERIE'''
        if self.config.name == 'REVERIE' or self.config.name == 'SOON':
            reverie_obj_img_embeds = self.img_embeddings.obj_reverie_linear(reverie_obj_fts) 
            if self.config.use_obj_name:
                reverie_obj_img_embeds = reverie_obj_img_embeds + self.img_embeddings.obj_name_linear(reverie_obj_names)
            
            reverie_obj_img_embeds = self.img_embeddings.obj_reverie_layer_norm(reverie_obj_img_embeds)

            img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                    view_img_embeds, reverie_obj_img_embeds, view_lens, reverie_obj_lens
                ):
                if obj_len > 0:
                    img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    img_embeds.append(view_embed[:view_len])
            img_embeds = pad_tensors_wgrad(img_embeds)
            view_lens = view_lens + reverie_obj_lens

            traj_embeds =  img_embeds +\
                    self.img_embeddings.loc_layer_norm(self.img_embeddings.loc_linear(loc_fts)) + \
                    self.img_embeddings.nav_type_embedding(nav_types)
                    
            traj_embeds = self.img_embeddings.layer_norm(traj_embeds)
            traj_embeds = self.img_embeddings.dropout(traj_embeds)
                
            img_masks = gen_seq_masks(view_lens)
            view_img_embeds = self.img_embeddings.pano_encoder(
                traj_embeds, src_key_padding_mask=img_masks.logical_not()
            )

        if self.config.adaptive_pano_fusion:
            traj_ori_embeds = view_img_embeds.clone()
            traj_fused_weight = self.img_embeddings.adaptive_pano_attn(traj_ori_embeds) 
            traj_fused_weight_act = torch.tanh(traj_fused_weight) 
            traj_fused_weight_act = self.img_embeddings.adaptive_softmax(traj_fused_weight_act)
            traj_fused_embeded_update = torch.mul(traj_ori_embeds,traj_fused_weight_act)
            traj_fused_embeds = torch.sum(traj_fused_embeded_update,dim=1)
            return view_img_embeds, img_masks, traj_fused_embeds
        else:
            return view_img_embeds, img_masks, None

    def forward_navigation_per_step(
        self, txt_embeds, txt_masks, gmap_img_embeds, gmap_step_ids, gmap_pos_fts, 
        gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids,
        vp_img_embeds, vp_pos_fts, vp_masks, vp_nav_masks, vp_obj_masks, vp_cand_vpids,
        front_vp_feats=None, front_gmap_feats=None
    ):  
        batch_size = txt_embeds.size(0)

        # global branch
        gmap_embeds = gmap_img_embeds + \
                    self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                    self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

        if self.global_encoder.sprel_linear is not None:
            graph_sprels = self.global_encoder.sprel_linear(
                gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None
        
        # Global Front Intervention
        if front_gmap_feats is not None:
            gmap_embeds = self.front_global_encoder(gmap_embeds, front_gmap_feats, gmap_masks)

        gmap_embeds = self.global_encoder.encoder(
            gmap_embeds, gmap_masks, txt_embeds, txt_masks, 
            graph_sprels=graph_sprels
        )
       
        # local branch
        vp_embeds = vp_img_embeds + self.local_encoder.vp_pos_embeddings(vp_pos_fts)

        # Local Frontdoor Intervention
        if front_vp_feats is not None:
            vp_embeds = self.front_local_encoder(vp_embeds, front_vp_feats, vp_masks)

        vp_embeds = self.local_encoder.encoder(vp_embeds, vp_masks, txt_embeds, txt_masks) 

        # navigation logits
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5

        else: # default
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)

        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))

        local_logits.masked_fill_(vp_nav_masks.logical_not(), -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)    
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(vp_cand_vpids[i]): # Process Local branch
                if j > 1: # jump over the [stop] and [MEM] token
                    if cand_vpid in visited_nodes:
                        bw_logits += local_logits[i, j]
                    else:
                        tmp[cand_vpid] = local_logits[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 1 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        # object grounding logits
        if vp_obj_masks is not None and self.config.dataset in ['reverie','soon']:
            obj_logits = self.og_head(vp_embeds).squeeze(2)
            obj_logits.masked_fill_(vp_obj_masks.logical_not(), -float('inf'))
        else:
            obj_logits = None 
        
        # represent the history information for per step.
        gmap_cls_embed = self.gmap_pooler(gmap_embeds,location=0)
        vp_cls_embeds = self.vp_pooler(vp_embeds,location=0)
        txt_cls_embeds = self.txt_pooler(txt_embeds,location=0)
        cls_embeds = self.local_his_ln(self.local_his_map(torch.cat((gmap_cls_embed,vp_cls_embeds,txt_cls_embeds),dim=-1)))

        outs = {
            'gmap_embeds': gmap_embeds,
            'vp_embeds': vp_embeds,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'fused_logits': fused_logits,
            'obj_logits': obj_logits,
            'txt_embeds': txt_embeds,
            'cls_embeds':cls_embeds
        }
        
        return outs

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked
    
    def forward(self, mode, batch, **kwargs):
        if mode == 'language':
            txt_embeds = self.forward_text(batch['txt_ids'], batch['txt_masks'], 
                batch['instr_z_direction_features'], batch['instr_z_direction_pzs'],
                batch['instr_z_landmark_features'], batch['instr_z_landmark_pzs'],
                batch['front_txt_feats']
                )
            return txt_embeds

        elif mode == 'panorama':
            pano_embeds, pano_masks, pano_fused_embeds = self.forward_panorama_do_per_step(
                batch['view_img_fts'], batch['loc_fts'],
                batch['nav_types'], batch['view_lens'],
                batch['z_img_features'], batch['z_img_pzs'],
                batch['reverie_obj_img_fts'], batch['reverie_obj_lens'], batch['reverie_obj_names']
            )
            return pano_embeds, pano_masks, pano_fused_embeds

        elif mode == 'navigation':
             return self.forward_navigation_per_step(
                batch['txt_embeds'], batch['txt_masks'], batch['gmap_img_embeds'], 
                batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_masks'],
                batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'], 
                batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_masks'],
                batch['vp_nav_masks'], batch['vp_obj_masks'], batch['vp_cand_vpids'],
                batch['front_vp_feats'], batch['front_gmap_feats']
            )

        elif mode == 'instr_zdict_update':
            txt_embeds = self.forward_text(batch['z_txt'], batch['z_txt_mask'],
                instr_z_direction_features=batch['instr_z_direction_features'], 
                instr_z_direction_pzs=batch['instr_z_direction_pzs'],
                instr_z_landmark_features=batch['instr_z_landmark_features'], 
                instr_z_landmark_pzs=batch['instr_z_landmark_pzs'],
                front_txt_embeds=batch['front_txt_feats'])
            return txt_embeds
    
        elif mode == 'extract_cfp_features':
            txt_embeds = self.forward_text(batch['txt_ids'], batch['txt_masks'])
            split_traj_embeds, split_traj_vp_lens, split_traj_fused_embeds = self.img_embeddings(
                batch['traj_view_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'],batch['traj_vp_view_lens'], None, 
                batch['traj_reverie_obj_fts'], batch['traj_reverie_obj_lens'], batch['traj_reverie_obj_locs']
            )
            gmap_embeds = self.global_encoder.forward_cfp(
                split_traj_embeds, split_traj_vp_lens, batch['traj_vpids'], batch['traj_cand_vpids'], batch['gmap_vpids'],
                batch['gmap_step_ids'], batch['gmap_pos_fts'], batch['gmap_lens'], graph_sprels=batch['gmap_pair_dists'],
                split_traj_fused_embeds=split_traj_fused_embeds
            )
            vp_embeds = self.local_encoder.forward_cfp(
                split_traj_embeds, split_traj_vp_lens, batch['vp_pos_fts']
            )

            # Use CFP heads to extract front-door features
            # Compute attention and aggragation, same as pre-training
            gmap_embeds = self.tim_global_head(gmap_embeds)
            vp_embeds = self.tim_local_head(vp_embeds)
            txt_embeds = self.tim_txt_head(txt_embeds)

            M1 = torch.tanh(gmap_embeds)
            a1 = torch.softmax(torch.matmul(M1,self.tim_global_attn),1) 
            out1 = torch.sum(gmap_embeds * a1,1) 
            gmap_outputs = torch.tanh(out1) 

            M2 = torch.tanh(vp_embeds)
            a2 = torch.softmax(torch.matmul(M2,self.tim_local_attn),1)
            out2 = torch.sum(vp_embeds * a2,1)
            vp_outputs = torch.tanh(out2)

            M3 = torch.tanh(txt_embeds)
            a3 = torch.softmax(torch.matmul(M3,self.tim_txt_attn),1) 
            out3 = torch.sum(txt_embeds * a3,1) 
            txt_outputs = torch.tanh(out3) 
                
            outputs = {
                'txt_outputs': txt_outputs,
                'vp_outputs': vp_outputs,
                'gmap_outputs': gmap_outputs
            }
            return outputs