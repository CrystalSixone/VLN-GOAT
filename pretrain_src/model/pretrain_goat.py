from collections import defaultdict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vilmodel_goat import BertLayerNorm, BertOnlyMLMHead, GlocalTextPathCMT, BertPredictionHeadTransform
from .ops import pad_tensors_wgrad, gen_seq_masks
from data.common import check_gpu_mem_usedRate

class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output

class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class GlocalTextPathCMTPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = GlocalTextPathCMT(config)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'mrc' in config.pretrain_tasks:
            self.image_classifier = RegionClassification(self.config.hidden_size, self.config.image_prob_size)
            if self.config.obj_prob_size > 0 and self.config.obj_prob_size != self.config.image_prob_size:
                self.obj_classifier = RegionClassification(self.config.hidden_size, self.config.obj_prob_size)
            else:
                self.obj_classifier = None
        if 'sap' in config.pretrain_tasks:
            self.global_sap_head = ClsPrediction(self.config.hidden_size)
            self.local_sap_head = ClsPrediction(self.config.hidden_size)
            if config.glocal_fuse:
                self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
            else:
                self.sap_fuse_linear = None
        if 'og' in config.pretrain_tasks:
            self.og_head = ClsPrediction(self.config.hidden_size)
        
        if 'cfp' in config.pretrain_tasks:
            self.tim_txt_head = BertPredictionHeadTransform(self.config)
            self.tim_global_head = BertPredictionHeadTransform(self.config)
            self.tim_local_head = BertPredictionHeadTransform(self.config)
            self.tim_fused_head = BertPredictionHeadTransform(self.config)

            # attention for last
            self.tim_txt_attn = nn.Parameter(torch.Tensor(self.config.hidden_size,1))
            self.tim_global_attn = nn.Parameter(torch.Tensor(self.config.hidden_size,1))
            self.tim_local_attn = nn.Parameter(torch.Tensor(self.config.hidden_size,1))
            self.tim_fused_attn = nn.Parameter(torch.Tensor(self.config.hidden_size,1))
            nn.init.uniform_(self.tim_txt_attn, -0.1, 0.1)
            nn.init.uniform_(self.tim_global_attn, -0.1, 0.1)
            nn.init.uniform_(self.tim_local_attn, -0.1, 0.1)
            nn.init.uniform_(self.tim_fused_attn, -0.1, 0.1)

            self.temperature = self.config.cfp_temperature

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                self.bert.embeddings.word_embeddings)

    def forward(self, batch, task, compute_loss=True):
        if self.config.empty_cache: # empty useless cuda cache
            used, used_rate, total = check_gpu_mem_usedRate(self.config.cuda_first_device)
            if used_rate > 0.9:
                torch.cuda.empty_cache()
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            return self.forward_mlm(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['txt_labels'], compute_loss,
                batch['traj_reverie_loc_fts'], batch['traj_reverie_obj_names'],
                batch['instr_z_landmark_features'],batch['instr_z_landmark_pzs'],
                batch['instr_z_direction_features'],batch['instr_z_direction_pzs'],
                batch['img_z_features'], batch['img_z_pzs']
            )
        elif task.startswith('mrc'):
            return self.forward_mrc(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['vp_view_mrc_masks'], batch['vp_view_probs'], 
                batch['vp_obj_mrc_masks'], batch['vp_obj_probs'], compute_loss,
                batch['traj_reverie_loc_fts'], batch['traj_reverie_obj_names'],
                batch['instr_z_landmark_features'],batch['instr_z_landmark_pzs'],
                batch['instr_z_direction_features'],batch['instr_z_direction_pzs'],
                batch['img_z_features'], batch['img_z_pzs']
            )
        elif task.startswith('sap'):
            return self.forward_sap(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['gmap_visited_masks'],
                batch['global_act_labels'], batch['local_act_labels'], compute_loss,
                batch['traj_reverie_loc_fts'], batch['traj_reverie_obj_names'],
                batch['instr_z_landmark_features'],batch['instr_z_landmark_pzs'],
                batch['instr_z_direction_features'],batch['instr_z_direction_pzs'],
                batch['img_z_features'], batch['img_z_pzs']
            )
        elif task.startswith('og'):
            return self.forward_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['obj_labels'], compute_loss,
                batch['traj_reverie_loc_fts'], batch['traj_reverie_obj_names'],
                batch['instr_z_landmark_features'],batch['instr_z_landmark_pzs'],
                batch['instr_z_direction_features'],batch['instr_z_direction_pzs'],
                batch['img_z_features'], batch['img_z_pzs']
            )
        elif task.startswith('valid_sap_og'):
            return self.forward_sap_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['gmap_visited_masks'], batch['global_act_labels'], batch['local_act_labels'], 
                batch['obj_labels'],
                batch['traj_reverie_loc_fts'], batch['traj_reverie_obj_names'],
                batch['instr_z_landmark_features'],batch['instr_z_landmark_pzs'],
                batch['instr_z_direction_features'],batch['instr_z_direction_pzs'],
                batch['img_z_features'], batch['img_z_pzs']
            )
        elif task.startswith('cfp'):
            return self.forward_cfp(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['gmap_visited_masks'],
                batch['global_act_labels'], batch['local_act_labels'], compute_loss,
                batch['traj_reverie_loc_fts'], batch['extra_heads'], batch['traj_reverie_obj_names'],
                batch['instr_z_landmark_features'],batch['instr_z_landmark_pzs'],
                batch['instr_z_direction_features'],batch['instr_z_direction_pzs'],
                batch['img_z_features'], batch['img_z_pzs']
            )
        else:
            raise ValueError('invalid task')

    def forward_mlm(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        txt_labels, compute_loss,
        traj_reverie_loc_fts=None,traj_reverie_obj_names=None,
        instr_z_landmark_features=None, instr_z_landmark_pzs=None,
        instr_z_direction_features=None, instr_z_direction_pzs=None,
        img_z_fts=None, img_z_pzs=None
    ):
        txt_embeds = self.bert.forward_mlm(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            traj_reverie_loc_fts=traj_reverie_loc_fts,traj_reverie_obj_names=traj_reverie_obj_names,
            instr_z_landmark_features=instr_z_landmark_features, instr_z_landmark_pzs=instr_z_landmark_pzs,
            instr_z_direction_features=instr_z_direction_features, instr_z_direction_pzs=instr_z_direction_pzs,
            z_img_features=img_z_fts, z_img_pzs=img_z_pzs
        )

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(
                prediction_scores, txt_labels[txt_labels != -1], reduction='none'
            )
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mrc(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        vp_view_mrc_masks, vp_view_probs, vp_obj_mrc_masks, vp_obj_probs, compute_loss=True,
        traj_reverie_loc_fts=None,traj_reverie_obj_names=None,
        instr_z_landmark_features=None, instr_z_landmark_pzs=None,
        instr_z_direction_features=None, instr_z_direction_pzs=None,
        img_z_fts=None, img_z_pzs=None
    ):
        _, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False,
            traj_reverie_loc_fts=traj_reverie_loc_fts,traj_reverie_obj_names=traj_reverie_obj_names,
            instr_z_landmark_features=instr_z_landmark_features, instr_z_landmark_pzs=instr_z_landmark_pzs,
            instr_z_direction_features=instr_z_direction_features, instr_z_direction_pzs=instr_z_direction_pzs,
            z_img_features=img_z_fts, z_img_pzs=img_z_pzs
        )
        
        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens)]
        vp_view_embeds = pad_tensors_wgrad(
            [x[1:view_len+1] for x, view_len in zip(vp_embeds, vp_view_lens)]
        )   # [stop] at 0
        # vp_view_mrc_masks = vp_view_mrc_masks[:, :vp_view_embeds.size(1)]
        
        # only compute masked regions for better efficient=cy
        view_masked_output = self._compute_masked_hidden(vp_view_embeds, vp_view_mrc_masks)
        view_prediction_soft_labels = self.image_classifier(view_masked_output)
        view_mrc_targets = self._compute_masked_hidden(vp_view_probs, vp_view_mrc_masks)

        if traj_obj_img_fts is not None:
            vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens)]
            vp_obj_embeds = pad_tensors_wgrad(
                [x[view_len+1:view_len+obj_len+1] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)]
            )
            # vp_obj_mrc_masks = vp_obj_mrc_masks[:, :vp_obj_embeds.size(1)]
            obj_masked_output = self._compute_masked_hidden(vp_obj_embeds, vp_obj_mrc_masks)
            if self.obj_classifier is None:
                obj_prediction_soft_labels = self.image_classifier(obj_masked_output)
            else:
                obj_prediction_soft_labels = self.obj_classifier(obj_masked_output)
            obj_mrc_targets = self._compute_masked_hidden(vp_obj_probs, vp_obj_mrc_masks)
        else:
            obj_prediction_soft_labels, obj_mrc_targets = None, None

        if compute_loss:
            view_prediction_soft_labels = F.log_softmax(view_prediction_soft_labels, dim=-1)
            view_mrc_loss = F.kl_div(view_prediction_soft_labels, view_mrc_targets, reduction='none').sum(dim=1)
            if obj_prediction_soft_labels is None:
                mrc_loss = view_mrc_loss
            else:
                obj_prediction_soft_labels = F.log_softmax(obj_prediction_soft_labels, dim=-1)
                obj_mrc_loss = F.kl_div(obj_prediction_soft_labels, obj_mrc_targets, reduction='none').sum(dim=1)
                mrc_loss = torch.cat([view_mrc_loss, obj_mrc_loss], 0)
            return mrc_loss
        else:
            return view_prediction_soft_labels, view_mrc_targets, obj_prediction_soft_labels, obj_mrc_targets

    def forward_sap(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        gmap_visited_masks, global_act_labels, local_act_labels, compute_loss,
        traj_reverie_loc_fts=None,traj_reverie_obj_names=None,
        instr_z_landmark_features=None, instr_z_landmark_pzs=None,
        instr_z_direction_features=None, instr_z_direction_pzs=None,
        img_z_fts=None, img_z_pzs=None
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            traj_reverie_loc_fts=traj_reverie_loc_fts,traj_reverie_obj_names=traj_reverie_obj_names,
            instr_z_landmark_features=instr_z_landmark_features, instr_z_landmark_pzs=instr_z_landmark_pzs,
            instr_z_direction_features=instr_z_direction_features, instr_z_direction_pzs=instr_z_direction_pzs,
            z_img_features=img_z_fts, z_img_pzs=img_z_pzs
        )
        
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1]!=1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1)-1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )   # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j+1]
                else:
                    tmp[cand_vpid] = local_logits[i, j+1]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        if compute_loss: # Default: True
            global_losses = F.cross_entropy(global_logits, global_act_labels, reduction='none')
            local_losses = F.cross_entropy(local_logits, local_act_labels, reduction='none')
            fused_losses = F.cross_entropy(fused_logits, global_act_labels, reduction='none')
            losses = global_losses + local_losses + fused_losses
            return losses
        else:
            return global_logits, local_logits, fused_logits, global_act_labels, local_act_labels

    def forward_og(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        obj_labels, compute_loss,
        traj_reverie_loc_fts=None,traj_reverie_obj_names=None,
        instr_z_landmark_features=None, instr_z_landmark_pzs=None,
        instr_z_direction_features=None, instr_z_direction_pzs=None,
        img_z_fts=None, img_z_pzs=None
    ):
        gmap_embeds, vp_embeds = self.bert.forward(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False,
            traj_reverie_loc_fts=traj_reverie_loc_fts,traj_reverie_obj_names=traj_reverie_obj_names,
            instr_z_landmark_features=instr_z_landmark_features, instr_z_landmark_pzs=instr_z_landmark_pzs,
            instr_z_direction_features=instr_z_direction_features, instr_z_direction_pzs=instr_z_direction_pzs,
            z_img_features=img_z_fts, z_img_pzs=img_z_pzs
        )

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1+view_len: 1+view_len+obj_len] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

        if compute_loss:
            losses = F.cross_entropy(obj_logits, obj_labels, reduction='none')
            return losses
        else:
            return obj_logits

    def forward_sap_og(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        gmap_visited_masks,
        traj_reverie_loc_fts=None,traj_reverie_obj_names=None,
        instr_z_landmark_features=None, instr_z_landmark_pzs=None,
        instr_z_direction_features=None, instr_z_direction_pzs=None,
        img_z_fts=None, img_z_pzs=None
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            traj_reverie_loc_fts=traj_reverie_loc_fts,traj_reverie_obj_names=traj_reverie_obj_names,
            instr_z_landmark_features=instr_z_landmark_features, instr_z_landmark_pzs=instr_z_landmark_pzs,
            instr_z_direction_features=instr_z_direction_features, instr_z_direction_pzs=instr_z_direction_pzs,
            z_img_features=img_z_fts, z_img_pzs=img_z_pzs
        )
        
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1]!=1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1)-1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )   # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j+1]
                else:
                    tmp[cand_vpid] = local_logits[i, j+1]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1+view_len: 1+view_len+obj_len] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))
        
        return global_logits, local_logits, fused_logits, obj_logits

    ''' Cross-modal Feature Pooling '''
    def forward_cfp(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        gmap_visited_masks, global_act_labels, local_act_labels, compute_loss,
        traj_reverie_loc_fts=None, extra_heads=False,traj_reverie_obj_names=None,
        instr_z_landmark_features=None, instr_z_landmark_pzs=None,
        instr_z_direction_features=None, instr_z_direction_pzs=None,
        img_z_fts=None, img_z_pzs=None
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds, vp_embeds, txt_embeds = self.bert.forward_cfp(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            traj_reverie_loc_fts=traj_reverie_loc_fts,
            return_txt_embeds=True,traj_reverie_obj_names=traj_reverie_obj_names,
            instr_z_landmark_features=instr_z_landmark_features, instr_z_landmark_pzs=instr_z_landmark_pzs,
            instr_z_direction_features=instr_z_direction_features, instr_z_direction_pzs=instr_z_direction_pzs,
            z_img_features=img_z_fts, z_img_pzs=img_z_pzs
        ) 
        if extra_heads:
            gmap_embeds = self.tim_global_head(gmap_embeds)
            vp_embeds = self.tim_local_head(vp_embeds)
            txt_embeds = self.tim_txt_head(txt_embeds)
        
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))
        
        ''' Use attention to flatten features'''
        M1 = torch.tanh(gmap_embeds)
        a1 = torch.softmax(torch.matmul(M1,self.tim_global_attn),1) # [bs,max_len,1]
        out1 = torch.sum(gmap_embeds * a1,1) # [bs,hd]
        gmap_outputs = torch.tanh(out1) # [bs,hd]

        M2 = torch.tanh(vp_embeds)
        a2 = torch.softmax(torch.matmul(M2,self.tim_local_attn),1) # [bs,max_len,1]
        out2 = torch.sum(vp_embeds * a2,1) # [bs,hd]
        vp_outputs = torch.tanh(out2) # [bs,hd]

        M3 = torch.tanh(txt_embeds)
        a3 = torch.softmax(torch.matmul(M3,self.tim_txt_attn),1) # [bs,max_len,1]
        out3 = torch.sum(txt_embeds * a3,1) # [bs,hd]
        txt_outputs = torch.tanh(out3) # [bs,hd]

        fused_outputs = gmap_outputs * fuse_weights + vp_outputs * (1-fuse_weights)

        if compute_loss: 
            target_sim = torch.arange(batch_size).cuda()

            gmap_txt_sim = ( gmap_outputs @ txt_outputs.T ) / self.temperature
            global_txt_losses = (F.cross_entropy(gmap_txt_sim, target_sim, reduction='none') +\
                                 F.cross_entropy(gmap_txt_sim.T, target_sim.T, reduction='none')) / 2.0

            vp_txt_sim = ( vp_outputs @ txt_outputs.T ) / self.temperature
            vp_txt_losses = (F.cross_entropy(vp_txt_sim, target_sim, reduction='none') +\
                                 F.cross_entropy(vp_txt_sim.T, target_sim.T, reduction='none')) / 2.0
            
            fused_txt_sim = ( fused_outputs @ txt_outputs.T) / self.temperature
            fused_txt_losses = (F.cross_entropy(fused_txt_sim, target_sim, reduction='none') +\
                                 F.cross_entropy(fused_txt_sim.T, target_sim.T, reduction='none')) / 2.0

            losses = global_txt_losses + vp_txt_losses + fused_txt_losses

            del target_sim
            
            return losses
        
        else:
            return gmap_outputs, vp_outputs, fused_outputs, txt_outputs
