import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
import line_profiler
import gc
import csv
import base64

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.ops import pad_tensors, gen_seq_masks
from utils.data import get_angle_fts
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent
from .eval_utils import cal_dtw

from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad


class GMapNavAgent(Seq2SeqAgent):  
    def _build_model(self):
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        # buffer
        self.scanvp_cands = {}

    def _language_variable(self, obs, instr_dict, front_txt_feats):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]
        batch_size = len(obs)
        
        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True
        
        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()

        instr_z_direction_features, instr_z_direction_pzs, instr_z_landmark_features, instr_z_landmark_pzs = None, None, None, None
        if instr_dict is not None:
            instr_z_direction_features = instr_dict['instr_direction_features'].repeat(batch_size,1).reshape(batch_size,-1,768) # add batch_size in the first dimension
            instr_z_direction_pzs = instr_dict['instr_direction_pzs'].repeat(batch_size,1).reshape(batch_size,-1,1)
            instr_z_landmark_features = instr_dict['instr_landmark_features'].repeat(batch_size,1).reshape(batch_size,-1,768)
            instr_z_landmark_pzs = instr_dict['instr_landmark_pzs'].repeat(batch_size,1).reshape(batch_size,-1,1)

        return {
            'txt_ids': seq_tensor, 'txt_masks': mask,
            'instr_z_direction_features': instr_z_direction_features,
            'instr_z_direction_pzs': instr_z_direction_pzs,
            'instr_z_landmark_features': instr_z_landmark_features,
            'instr_z_landmark_pzs': instr_z_landmark_pzs,
            'front_txt_feats': front_txt_feats
        }
    
    def _get_vp_pos_fts(self, obs, traj, nav_inputs,traj_fts):
        batch_vp_pos_fts = []
        for i, ob in enumerate(obs):
            scan = ob['scan']
            start_heading = ob.get('heading', 0)
            cur_heading, cur_elevation = self.env.get_cur_angle(scan, traj[i]['path'], start_heading)
            cur_cand_pos_fts = self.env.get_gmap_pos_fts(scan, traj[i]['path'][-1][-1], nav_inputs['vp_cand_vpids'][i], cur_heading, cur_elevation)
            cur_start_pos_fts = self.env.get_gmap_pos_fts(scan, traj[i]['path'][-1][-1], traj[i]['path'][0], cur_heading, cur_elevation)
                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((len(traj_fts['traj_nav_types'][-1])+1, 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(vp_pos_fts)
        
        batch_vp_pos_fts = pad_tensors(torch.from_numpy(np.array(batch_vp_pos_fts))).cuda()
        return batch_vp_pos_fts


    def _panorama_feature_variable_do(self, obs, img_zdict=None, noise=None):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens = []
        batch_cand_vpids = []
        
        z_img_features, z_img_pzs = None, None
        batch_size = len(obs)
        
        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            obj_nav_types = []
            # cand views
            used_viewidxs = set()
            # Only use candidate images
            for j, cc in enumerate(ob['candidate']):
                if noise is None:
                    view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                else:
                    view_img_fts.append(cc['feature'][:self.args.image_feat_size]*noise)
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            if noise is None:
                view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                    in enumerate(ob['feature']) if k not in used_viewidxs])
            else:
                view_img_fts.extend([x[:self.args.image_feat_size]*noise for k, x \
                    in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts,view_box_fts], 1)
            
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()
        
        if img_zdict is not None:
            z_img_features = img_zdict['img_features'].repeat(batch_size,1).reshape(batch_size,-1,768)
            z_img_pzs = img_zdict['img_pzs'].repeat(batch_size,1).reshape(batch_size,-1,1)
        
        already_dropout = False if noise is None else True
        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts, 
            'nav_types': batch_nav_types,'view_lens': batch_view_lens, 
            'cand_vpids': batch_cand_vpids,
            'z_img_features': z_img_features, 'z_img_pzs': z_img_pzs,
            'already_dropout': already_dropout
        }

    def _nav_gmap_variable(self, obs, gmaps, last_embeds=None):
        # [stop] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []                
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes: # False
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph: # True
                # stop -> memory -> visited -> unvisited
                gmap_vpids = [None] + [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[2:]]
            cat_rec_embeds = [torch.zeros_like(gmap_img_embeds[0])] if last_embeds is None else [last_embeds[i]]
            gmap_img_embeds = torch.stack(
                    [torch.zeros_like(gmap_img_embeds[0])] + cat_rec_embeds + gmap_img_embeds, 0
                )

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(2, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_masks[:,1] = False # [MEM] token cannot be achieved
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left, 't': None
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types,obj_lens=0):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i], 
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp], 
                obs[i]['heading'], obs[i]['elevation']
            )                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens+obj_lens+1),
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
            'vp_obj_masks': None,
        }
    
    def _nav_vp_variable_mem(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types, last_embeds=None):
        batch_size = len(obs)

        # add [stop] token
        cat_stop_embeds = torch.zeros_like(pano_embeds[:, :1]) if last_embeds is None else last_embeds.unsqueeze(1)
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), cat_stop_embeds, pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i], 
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp], 
                obs[i]['heading'], obs[i]['elevation']
            )                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32) # [bs,14]
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[2:len(cur_cand_pos_fts)+2, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), torch.zeros(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        vp_masks = gen_seq_masks(view_lens+2)
        vp_cand_vpids = [[None]+[None]+x for x in cand_vpids]

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': vp_masks,
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': vp_cand_vpids,
        }

    def _teacher_action(
        self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0    # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0    # Stop if arrived 
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 1 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan], 
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][1:], 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, gmaps, obs, traj=None, jump_traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        scans, actions, headings, elevations = [], [], [], []
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if jump_traj is not None:
                    jump_traj[i]['path'].append([action])
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
            else:
                state = self.env.env.sims.getState()[i]
                action = state.location.viewpointId
                heading = state.heading
                elevation = state.elevation
            scans.append(ob['scan'])
            actions.append(action)
            headings.append(heading)
            elevations.append(elevation)
        
        self.env.env.sims.newEpisode(scans, actions, headings, elevations)

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']
    
    def pad_instr_tokens(self, instr_tokens, maxlength=20):
        if len(instr_tokens) <= 2: #assert len(raw_instr_tokens) > 2
            return None

        if len(instr_tokens) > maxlength - 2: # -2 for [CLS] and [SEP]
            instr_tokens = instr_tokens[:(maxlength-2)]

        instr_tokens = ['[CLS]'] + instr_tokens + ['[SEP]']
        num_words = len(instr_tokens)  # - 1  # include [SEP]
        instr_tokens += ['[PAD]'] * (maxlength-len(instr_tokens))

        assert len(instr_tokens) == maxlength

        return instr_tokens, num_words
    
    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()
    
    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()
    
    def accumulate_gradient(self, feedback='teacher', z_dicts={}, z_front_dict={}, **kwargs):
        self.vln_bert.train()
        self.critic.train()
        if self.args.train_alg == 'imitation':
                self.feedback = 'teacher'
                self.rollout(
                    train_ml=1., train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs
                )
        elif self.args.train_alg == 'dagger':  
            if self.args.ml_weight != 0:
                self.feedback = 'teacher'
                self.rollout(
                    train_ml=self.args.ml_weight, train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict,**kwargs
                )
            self.feedback = 'expl_sample' if self.args.expl_sample else 'sample'
            self.rollout(train_ml=1, train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs) 
        else:
            if self.args.ml_weight != 0:
                self.feedback = 'teacher'
                self.rollout(
                    train_ml=self.args.ml_weight, train_rl=False, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs
                )
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, z_dicts=z_dicts, z_front_dict=z_front_dict, **kwargs)
    
    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True, speaker=None,test=False, z_dicts={}, z_front_dict={}):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)

        # back-translation
        noise = None
        if speaker is not None:         # Trigger the self_train mode!
            noise = self.vln_bert.drop_env(torch.ones(self.feature_size).cuda())
            batch = self.env.batch.copy()
            speaker.env = self.env
            new_insts = speaker.infer_batch(featdropmask=noise)
            noise = noise.cpu().numpy() 
            for i, (datum, inst) in enumerate(zip(batch, new_insts)):
                inst = speaker.tok.shrink(inst)
                datum.pop('instruction')
                datum['instruction'] = speaker.tok.decode_sentence(inst)
                instr = datum['instruction']
                ''' BERT tokenizer '''
                datum['instr_encoding'] = self.tok(instr)['input_ids']
                
            obs = np.array(self.env.reset(batch))
            batch_size = len(obs)

        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
        } for ob in obs]

        # Do intervention
        img_zdict, instr_zdict = None, None
        if 'img_zdict' in z_dicts:
            img_zdict = z_dicts['img_zdict']
        if 'instr_zdict' in z_dicts:
            instr_zdict = z_dicts['instr_zdict']
        
        # Frontdoor intervention
        front_txt_feats, front_vp_feats, front_gmap_feats = None, None, None
        if z_front_dict is not None:
            if self.args.do_front_txt and 'txt_feats' in z_front_dict:
                front_txt_feats = []
                for ob in obs:
                    front_txt_feats.append(z_front_dict['txt_feats'])
                front_txt_feats = torch.from_numpy(np.array(front_txt_feats)).cuda()
            if self.args.do_front_img and 'vp_feats' in z_front_dict:
                front_vp_feats = []
                for ob in obs:
                    front_vp_feats.append(z_front_dict['vp_feats'])
                front_vp_feats = torch.from_numpy(np.array(front_vp_feats)).cuda()
            if self.args.do_front_his and 'gmap_feats' in z_front_dict:
                front_gmap_feats = []
                for ob in obs:
                    front_gmap_feats.append(z_front_dict['gmap_feats'])
                front_gmap_feats = torch.from_numpy(np.array(front_gmap_feats)).cuda()
        
        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs, instr_zdict, front_txt_feats)
        
        # Step-1: compute the txt embedding through network
        if test:
            with torch.no_grad():
                txt_embeds = self.vln_bert('language', language_inputs)
        else:
            txt_embeds = self.vln_bert('language', language_inputs)
    
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.    
        jump_traj = None
        last_embeds = None

        for t in range(self.args.max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable_do(obs, img_zdict, noise=noise)

            # Step-2: compute the current panoramic features through network
            if test:
                with torch.no_grad():
                    pano_embeds, pano_masks, pano_fused_embeds = self.vln_bert('panorama', pano_inputs)
            else:
                pano_embeds, pano_masks, pano_fused_embeds = self.vln_bert('panorama', pano_inputs)
            
            if not self.args.adaptive_pano_fusion:
                avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                                  torch.sum(pano_masks, 1, keepdim=True) 
            else:
                avg_pano_embeds = pano_fused_embeds

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node (current)
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps, last_embeds)
            nav_inputs.update(
                self._nav_vp_variable_mem(
                        obs, gmaps, pano_embeds, pano_inputs['cand_vpids'], 
                        pano_inputs['view_lens'], pano_inputs['nav_types'],
                        last_embeds
                    ))
            
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
            })

            # Frontdoor Intervention
            nav_inputs['front_txt_feats'] = front_txt_feats
            nav_inputs['front_vp_feats'] = front_vp_feats
            nav_inputs['front_gmap_feats'] = front_gmap_feats

            # Step-3: compute the cross-modal prediction through network
            if test:
                with torch.no_grad():
                    nav_outs = self.vln_bert('navigation', nav_inputs)
            else:
                nav_outs = self.vln_bert('navigation', nav_inputs)

            last_embeds = nav_outs['cls_embeds']

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)
            
            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    } # record the probability of stoping
                                        
            if train_ml is not None:
                # Supervised training
                nav_targets = self._teacher_action(
                    obs, nav_vpids, ended, 
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                    imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                )
                ml_loss += self.criterion(nav_logits, nav_targets)
                                              
            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)        # student forcing - argmax
                a_t = a_t.detach() 
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach() 
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample': # in training
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0 

            # Prepare environment action
            cpu_a_t = []  
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])   

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj, jump_traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    if jump_traj is not None:
                        jump_traj[i]['path'].append([stop_node])
                    if self.args.detailed_output: # False
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.loss += ml_loss
            self.logs['IL_loss'].append(ml_loss.item())

        if self.args.submit:
            # generate submit file for online test
            for i,item in enumerate(traj):
                new_paths = []
                for node in item['path']:
                    for each_sub_node in node:
                        new_paths.append([each_sub_node])
                traj[i]['path'] = new_paths

        return traj


    def update_z_dict(self, instr_data, z_dict=None, save_all_features=False):
        ''' update z_dict through the whole training set and save
        At present, we only support instruction.
        save_all_features: record all features.
        '''
        self.vln_bert.eval()
        batch_num = 64
        landmark_output_features_dict = defaultdict()
        landmark_output_features = []
        landmark_pz_list = []
        direction_output_features_dict = defaultdict()
        direction_output_features = []
        direction_pz_list = []
        landmark_dict = defaultdict(lambda:[])
        direction_dict = defaultdict(lambda:[])
        for i in range(0,len(instr_data),batch_num):
            batch_txt, batch_txt_encoding, batch_txt_landmark, batch_txt_direction = [],[],[],[]
            ori_batch = instr_data[i:i+batch_num]
            current_batch_size = len(ori_batch)
            for data in ori_batch:
                instr = data['instruction']
                instr_tok = data['instr_encoding']
                instr_id = data['instr_id']
                if len(self.instr_specific_dict[instr_id]) == 0:
                    landmarks, directions, tokens = self.word_picker.pick_action_object_words_with_index(instr,map=False)
                    self.instr_specific_dict[instr_id].append(landmarks)
                    self.instr_specific_dict[instr_id].append(directions)
                else:
                    landmarks = self.instr_specific_dict[instr_id][0]
                    directions = self.instr_specific_dict[instr_id][1]
                batch_txt.append(instr)
                batch_txt_encoding.append(instr_tok)
                batch_txt_landmark.append(landmarks)
                batch_txt_direction.append(directions)
            
            seq_lengths = [len(x) for x in batch_txt_encoding]
            seq_tensor = np.zeros((len(batch_txt), max(seq_lengths)), dtype=np.int64)
            mask = np.zeros((len(batch_txt), max(seq_lengths)), dtype=np.bool)
            for i, enc in enumerate(batch_txt_encoding):
                seq_tensor[i, :seq_lengths[i]] = enc
                mask[i, :seq_lengths[i]] = True
            seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
            mask = torch.from_numpy(mask).cuda()

            txt_inputs = {
                'z_txt': seq_tensor,
                'z_txt_mask': mask
            }
            
            if z_dict is not None:
                instr_z_dict = z_dict['instr_zdict']
                txt_inputs['instr_z_direction_features'] = instr_z_dict['instr_direction_features'].repeat(current_batch_size,1).reshape(current_batch_size,-1,768) # add batch_size in the first dimension
                txt_inputs['instr_z_direction_pzs']  = instr_z_dict['instr_direction_pzs'].repeat(current_batch_size,1).reshape(current_batch_size,-1,1)
                txt_inputs['instr_z_landmark_features'] = instr_z_dict['instr_landmark_features'].repeat(current_batch_size,1).reshape(current_batch_size,-1,768)
                txt_inputs['instr_z_landmark_pzs'] = instr_z_dict['instr_landmark_pzs'].repeat(current_batch_size,1).reshape(current_batch_size,-1,1)
            else:
                txt_inputs['instr_z_direction_features'] = None
                txt_inputs['instr_z_direction_pzs'] = None
                txt_inputs['instr_z_landmark_features'] = None
                txt_inputs['instr_z_landmark_pzs'] = None

            with torch.no_grad():
                outputs = self.vln_bert('instr_zdict_update', txt_inputs).detach().cpu()

            # Pick the targeted token embedding
            for idx, output in enumerate(outputs):
                re_text = self.tok.convert_ids_to_tokens(batch_txt_encoding[idx],skip_special_tokens=True)
                count = 0
                landmark_idx = 0
                direction_idx = 0
                landmark_list = batch_txt_landmark[idx]
                direction_list = batch_txt_direction[idx]
                for j, re_token in enumerate(re_text):
                    if re_token[0] == '#':
                        continue
                    else:
                        if landmark_idx < len(landmark_list) and count == landmark_list[landmark_idx][0]:
                            key = landmark_list[landmark_idx][1]
                            emb = np.array(output[j+1]) # output +1 since there is [CLS] token
                            landmark_dict[key].append(emb)
                            landmark_idx += 1
                        if direction_idx < len(direction_list) and count == direction_list[direction_idx][0]:
                            key = direction_list[direction_idx][1]
                            emb = np.array(output[j+1])
                            direction_dict[key].append(emb)
                            direction_idx += 1
                        count += 1

        # pz
        landmark_pz_dict = {}
        direction_pz_dict = {}
        l_total_num, d_total_num = 0, 0
        for key, value in landmark_dict.items():
            l_total_num += len(value)
        for key, value in landmark_dict.items():
            landmark_pz_dict[key] = len(value) / l_total_num
        for key, value in direction_dict.items():
            d_total_num += len(value)
        for key, value in direction_dict.items():
            direction_pz_dict[key] = len(value) / d_total_num
        
        # feature
        for key, value in landmark_dict.items():
            feature = np.mean(np.array(value),axis=0)
            landmark_output_features_dict[key] = feature
            landmark_output_features.append(feature)
            landmark_pz_list.append(landmark_pz_dict[key])

        for key, value in direction_dict.items():
            feature = np.mean(np.array(value),axis=0)
            direction_output_features_dict[key] = feature
            direction_output_features.append(feature)
            direction_pz_list.append(direction_pz_dict[key])
        
        landmark_output = torch.from_numpy(np.array(landmark_output_features)).cuda()
        direction_output = torch.from_numpy(np.array(direction_output_features)).cuda()
        direction_pz = torch.from_numpy(np.array(direction_pz_list)).cuda()
        landmark_pz = torch.from_numpy(np.array(landmark_pz_list)).cuda()
        
        self.vln_bert.train()
        if not save_all_features:
            if z_dict is not None:
                z_dict["instr_zdict"]["instr_direction_features"] = direction_output
                z_dict["instr_zdict"]["instr_landmark_features"] = landmark_output
                z_dict["instr_zdict"]["instr_direction_pzs"] = direction_pz
                z_dict["instr_zdict"]["instr_landmark_pzs"] = landmark_pz
            else:
                z_dict = {}
                z_dict["instr_zdict"] = {
                    "instr_direction_features": direction_output,
                    "instr_direction_pzs": direction_pz,
                    "instr_landmark_features": landmark_output,
                    "instr_landmark_pzs": landmark_pz
                }
         
        return z_dict, landmark_output_features_dict, direction_output_features_dict, landmark_pz_dict, direction_pz_dict

    def save_backdoor_z_dict(self, landmark_dict, direction_dict, landmark_pz_dict, direction_pz_dict):
        backdoor_dict_file = os.path.join(self.args.z_back_log_dir, f'backdoor_update_features.tsv')
        
        INSTR_TSV_FIELDNAMES = ['token_type','token','feature','pz']
        with open(backdoor_dict_file, 'wt') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = INSTR_TSV_FIELDNAMES)
            for key, value in landmark_dict.items():
                record = {
                    'feature': str(base64.b64encode(value), "utf-8"),
                    'token_type': 'landmark',
                    'pz': landmark_pz_dict[key],
                    'token': key
                }
                writer.writerow(record)
            for key, value in direction_dict.items():
                record = {
                    'feature': str(base64.b64encode(value), "utf-8"),
                    'token_type': 'direction',
                    'pz': direction_pz_dict[key],
                    'token': key
                }
                writer.writerow(record)

    def get_per_traj_feats(self, idx):
        ''' get a whole trajectory features from dataset
        '''
        item = self.env.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item['heading']
        gt_path = item['path']
        end_vp = gt_path[-1]
        end_idx = gt_path.index(end_vp)

        cur_heading, cur_elevation = self.env.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > self.args.max_action_len:
            # truncate trajectory
            gt_path = gt_path[:self.args.max_action_len] + [end_vp]
        
        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], []

        for vp in gt_path:
            view_fts = self.env.env.feat_db.get_image_feature(scan, vp)
            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.env.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_angle = self.env.all_point_rel_angles[12][v[0]]
                heading = cur_heading - view_angle[0] + v[2]
                elevation = cur_elevation - view_angle[1] + v[3]
                view_angles.append([heading, elevation])
                cand_vpids.append(k)

            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.env.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
    
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.env.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            
            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_loc_fts.append(np.concatenate([view_ang_fts, view_box_fts], 1))
            traj_nav_types.append([1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)))

            traj_cand_vpids.append(cand_vpids)
            last_vp_angles = view_angles

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.env.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.env.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))
        
        outs = {
            'instr_id': item['instr_id'],
            'txt_ids': torch.LongTensor(item['instr_encoding'][:self.args.max_instr_len]),
            
            'traj_view_img_fts': [torch.from_numpy(x[:, :self.args.image_feat_size]) for x in traj_view_img_fts],
            'traj_loc_fts': [torch.from_numpy(x) for x in traj_loc_fts],
            'traj_reverie_loc_fts': None,
            'traj_nav_types': [torch.LongTensor(x) for x in traj_nav_types],
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': torch.LongTensor(gmap_step_ids),
            'gmap_visited_masks': torch.BoolTensor(gmap_visited_masks),
            'gmap_pos_fts': torch.from_numpy(gmap_pos_fts),
            'gmap_pair_dists': torch.from_numpy(gmap_pair_dists),

            'vp_pos_fts': torch.from_numpy(vp_pos_fts),
            'vp_angles': last_vp_angles,
        }
        if 'instr' in item.keys():
            outs['instr'] = item['instr'][:self.args.max_instr_len]
        
        return outs

    def cfp_collate(self, inputs):
        batch = {
            k: [x[k] for x in inputs] for k in inputs[0].keys()
        }
        # text batches
        batch['txt_lens'] = torch.LongTensor([len(x) for x in batch['txt_ids']])
        batch['txt_ids'] = pad_sequence(batch['txt_ids'], batch_first=True, padding_value=0)
        mask = np.zeros((len(batch['txt_lens']), max(batch['txt_lens'])), dtype=np.bool)
        for i in range(len(batch['txt_lens'])):
            mask[i, :batch['txt_lens'][i]] = True
        batch['txt_masks'] = torch.from_numpy(mask)

        # trajectory batches: traj_cand_vpids, traj_vpids
        batch['traj_step_lens'] = [len(x) for x in batch['traj_view_img_fts']]
        batch['traj_vp_view_lens'] = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch['traj_view_img_fts']], [])
        )
        batch['traj_view_img_fts'] = pad_tensors(sum(batch['traj_view_img_fts'], []))
        if 'traj_obj_img_fts' in batch:
            batch['traj_vp_obj_lens'] = torch.LongTensor(
                sum([[len(y) for y in x] for x in batch['traj_obj_img_fts']], [])
            )
            batch['traj_obj_img_fts'] = pad_tensors(sum(batch['traj_obj_img_fts'], []))
            
        batch['traj_loc_fts'] = pad_tensors(sum(batch['traj_loc_fts'], []))
        batch['traj_nav_types'] = pad_sequence(sum(batch['traj_nav_types'], []), batch_first=True, padding_value=0)

        batch['traj_reverie_loc_fts'] = None

        # gmap batches: gmap_vpids
        batch['gmap_lens'] = torch.LongTensor([len(x) for x in batch['gmap_step_ids']]) # included [stop]
        batch['gmap_step_ids'] = pad_sequence(batch['gmap_step_ids'], batch_first=True, padding_value=0)
        batch['gmap_visited_masks'] = pad_sequence(batch['gmap_visited_masks'], batch_first=True, padding_value=0)
        batch['gmap_pos_fts'] = pad_tensors(batch['gmap_pos_fts'])
        max_gmap_len = max(batch['gmap_lens'])
        batch_size = len(batch['gmap_lens'])
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch['gmap_lens'][i], :batch['gmap_lens'][i]] = batch['gmap_pair_dists'][i]
        batch['gmap_pair_dists'] = gmap_pair_dists

        # vp batches: vp_angles
        batch['vp_lens'] = torch.LongTensor([len(x[-1]) for x in batch['vp_pos_fts']])  # included [stop]
        batch['vp_pos_fts'] = pad_tensors(batch['vp_pos_fts'])

        # Change to Cuda
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = batch[k].cuda()
        return batch
    
    def extract_cfp_features(self, train_data, z_dict={}, iter=0, save_file=True):
        '''Used to extract fused cfp features for the whole trajectory and instructions
        '''
        print('Extract CFP features ...')
        batch_num = 64
        txt_output_feats = []
        vp_output_feats = []
        gmap_output_feats = []
        for i in range(0,len(train_data),batch_num):
            end_num = batch_num+i if (batch_num+i) < len(train_data) else len(train_data)
            batch_data = train_data[i:end_num]
            batch_traj_data = []
            for j, item in enumerate(batch_data):
                idx = i+j
                per_traj_feat = self.get_per_traj_feats(idx)
                batch_traj_data.append(per_traj_feat)
            traj_inputs = self.cfp_collate(batch_traj_data)
            with torch.no_grad():
                outputs = self.vln_bert('extract_cfp_features', traj_inputs)
                txt_outputs = outputs['txt_outputs'].detach().cpu()
                vp_outputs = outputs['vp_outputs'].detach().cpu()
                gmap_outputs = outputs['gmap_outputs'].detach().cpu()

            txt_output_feats += txt_outputs
            vp_output_feats += vp_outputs
            gmap_output_feats += gmap_outputs
        
        assert len(txt_output_feats) == len(train_data)

        if save_file:
            backdoor_dict_file = os.path.join(self.args.log_dir, f'r2r_cfp_features.tsv')
            INSTR_TSV_FIELDNAMES = ['path_id','txt_feats','vp_feats','gmap_feats']
            with open(backdoor_dict_file, 'wt') as tsvfile:
                writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = INSTR_TSV_FIELDNAMES)
                for idx in range(len(txt_output_feats)):
                    record = {
                        'path_id': self.env.data[idx]['path_id'],
                        'txt_feats': str(base64.b64encode(np.array(txt_output_feats[idx])), "utf-8"),
                        'vp_feats': str(base64.b64encode(np.array(vp_output_feats[idx])), "utf-8"),
                        'gmap_feats': str(base64.b64encode(np.array(gmap_output_feats[idx])), "utf-8"),
                    }
                    writer.writerow(record)