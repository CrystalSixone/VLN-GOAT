# This is to count the size of the parameters of the model
import os,sys
root_path = os.getcwd()
current_path = os.path.join(root_path,'map_nav_src')
sys.path.append(root_path)
sys.path.append(current_path)

import torch
import torch.nn as nn
import numpy as np
from r2r.parser import parse_args
from models.model import VLNBert
from models import *
from thop import profile

def GFLOPs_count(bs, txt_lens, global_lens, local_lens, h_dim=768):
    '''Language'''
    txt_ids = torch.randint(0,2000,size=(bs,txt_lens)).cuda()
    txt_masks = torch.randint(0, 2, size=(bs, txt_lens), dtype=torch.bool).cuda()
    instr_z_direction_features = torch.rand(bs,35,h_dim).cuda()
    instr_z_direction_pzs = torch.rand(bs,35,1).cuda()
    instr_z_landmark_features = torch.rand(bs,39,h_dim).cuda()
    instr_z_landmark_pzs = torch.rand(bs,39,1).cuda()
    front_txt_embeds = torch.rand(bs,24,h_dim)
    
    mode = 'language'
    lan_input = {
        'txt_ids': txt_ids,
        'txt_masks': txt_masks,
        'instr_z_direction_features': instr_z_direction_features,
        'instr_z_direction_pzs': instr_z_direction_pzs,
        'instr_z_landmark_features': instr_z_landmark_features,
        'instr_z_landmark_pzs': instr_z_landmark_pzs,
        'front_txt_embeds': front_txt_embeds
    }

    lan_macs, lan_params = profile(vln_bert, (mode, lan_input))
    lan_gflops = lan_macs*2/(10**9)
    print('Language module GFLOPs:%.3f' %(lan_gflops))
    print('*****')

    '''panorama'''
    view_img_fts = torch.rand(8,local_lens,h_dim).cuda()
    obj_img_fts = None
    loc_fts = torch.rand(8,local_lens,7).cuda()
    nav_types = torch.randint(0,2,size=(8,local_lens)).cuda()
    view_lens = torch.ones(8,dtype=torch.int).cuda()*local_lens
    z_img_features = torch.rand(bs,50,h_dim).cuda()
    z_img_pzs = torch.rand(bs,50,1).cuda()
    
    mode = 'panorama'

    pan_input = {
        'view_img_fts': view_img_fts,
        'obj_img_fts': obj_img_fts,
        'loc_fts': loc_fts,
        'nav_types': nav_types,
        'view_lens': view_lens,
        'z_img_features': z_img_features,
        'z_img_pzs': z_img_pzs 
        }

    pan_macs, pan_params = profile(vln_bert, (mode, pan_input))
    pan_gflops = pan_macs * 2 / (10 ** 9)
    print('Panorama Module GFLOPs: %.3f' % (pan_gflops))
    print('*****')


    '''navigation'''
    txt_embeds = torch.rand(8,txt_lens,h_dim).cuda()
    txt_masks = torch.rand(8,txt_lens).cuda()
    gmap_img_embeds = torch.rand(8,global_lens,h_dim).cuda()
    gmap_step_ids = torch.randint(0, 2, size=(8, global_lens), dtype=torch.int).cuda()
    gmap_pos_fts = torch.rand(8,global_lens,7).cuda()
    gmap_masks = torch.randint(0, 2, size=(8, global_lens), dtype=torch.bool).cuda()
    gmap_pair_dists = torch.rand(8,global_lens,global_lens).cuda()
    gmap_visited_masks = torch.randint(0, 2, size=(8, global_lens), dtype=torch.bool).cuda()
    gmap_vpids = None
    vp_img_embeds = torch.rand(8,local_lens,h_dim).cuda()
    vp_pos_fts = torch.rand(8,local_lens,14).cuda()
    vp_masks = torch.randint(0, 2, size=(8, local_lens), dtype=torch.bool).cuda()
    vp_nav_masks = torch.randint(0, 2, size=(8, local_lens), dtype=torch.bool).cuda()
    vp_obj_masks = None
    vp_cand_vpids = None
    front_vp_feats = torch.rand(8,24,h_dim).cuda()
    front_gmap_feats = torch.rand(8,24,h_dim).cuda()
    
    mode = 'navigation'

    nav_input = {
        'txt_embeds': txt_embeds,
        'txt_masks': txt_masks,
        'gmap_img_embeds': gmap_img_embeds,
        'gmap_step_ids': gmap_step_ids,
        'gmap_pos_fts': gmap_pos_fts,
        'gmap_masks': gmap_masks,
        'gmap_pair_dists': gmap_pair_dists,
        'gmap_visited_masks': gmap_visited_masks,
        'gmap_vpids': gmap_vpids,
        'vp_img_embeds': vp_img_embeds,
        'vp_pos_fts': vp_pos_fts,
        'vp_masks': vp_masks,
        'vp_nav_masks': vp_nav_masks,
        'vp_obj_masks': vp_obj_masks,
        'vp_cand_vpids': vp_cand_vpids,
        'front_vp_feats': front_vp_feats,
        'front_gmap_feats': front_gmap_feats,
        "flops_count": True
    }

    nav_macs, nav_params = profile(vln_bert, (mode, nav_input))
    nav_gflops = nav_macs * 2 / (10 ** 9)
    print('Panorama Module GFLOPs: %.3f' % (nav_gflops))
    print('*****')

    print('Total GFLOPs: %.3f' % (lan_gflops + pan_gflops + nav_gflops)) 
    total_GFLOPs = lan_gflops + pan_gflops + nav_gflops
    return total_GFLOPs
    
if __name__ == '__main__':
    args = parse_args()
    
    '''Parameter count'''
    vln_bert = VLNBert(args).cuda()
    vln_bert_size = sum([x.numel() for x in vln_bert.parameters()])
    print('Total Parameters: %.3f M.'%(vln_bert_size/1e6))

    '''FLOPs count'''
    # Default definition
    bs = 8
    txt_lens = 44
    local_lens = 36
    global_lens = 6
    h_dim = 768

    total_gflops = GFLOPs_count(bs,txt_lens,global_lens,local_lens,h_dim)

    print('Total Parameters: %.3f M.'%(vln_bert_size/1e6))
