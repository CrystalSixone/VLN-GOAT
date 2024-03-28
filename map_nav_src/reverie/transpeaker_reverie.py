import os

import torch
import torch.nn as nn
import numpy as np

import reverie.speaker_utils as utils

import models.transpeaker_model_reverie as transpeaker_model

import torch.nn.functional as F
from torch.autograd import Variable


class Speaker():
    env_actions = {
      'left': ([0.0],[-1.0], [0.0]), # left
      'right': ([0], [1], [0]), # right
      'up': ([0], [0], [1]), # up
      'down': ([0], [0],[-1]), # down
      'forward': ([1], [0], [0]), # forward
      '<end>': ([0], [0], [0]), # <end>
      '<start>': ([0], [0], [0]), # <start>
      '<ignore>': ([0], [0], [0])  # <ignore>
    }

    def __init__(self, args, env, tok):
        self.env = env
        self.args = args
        self.feature_size = args.image_feat_size
        self.tok = tok
        self.tok.finalize()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        # Initialize the model
        self.model = transpeaker_model.Transpeaker(
            feature_size=self.feature_size+args.speaker_angle_size,
            hidden_size=args.h_dim, 
            word_size=args.wemb, 
            tgt_vocab_size=self.tok.vocab_size(),
            obj_size=args.obj_feat_size
        )
        if self.args.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.lr) 
        elif self.args.optim == 'adamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=args.lr)

        # Evaluation
        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'])
        self.valid_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'],reduction='none')

    def train(self, iters):
        losses = 0
        for i in range(iters):
            self.env.reset()
        
            self.optimizer.zero_grad()                
            
            loss = self.teacher_forcing(train=True)
            losses += loss

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 20.)

            self.optimizer.step()
        
        return losses 

    def get_insts(self, wrapper=(lambda x: x)):
        # Get the caption for all the data
        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        total = self.env.size()

        noise=None
        for _ in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            insts = self.infer_batch(featdropmask=noise)

            path_ids = [str(ob['path_id'])+'_'+str(ob['gt_obj_id']) for ob in obs]

            for path_id, inst in zip(path_ids, insts):
                if path_id not in path2inst:
                    path2inst[path_id] = {}
                    path2inst[path_id]['inst'] = self.tok.shrink(inst)  # Shrink the words
                    path2inst[path_id]['loss'] = 0.0 

        return path2inst

    def valid(self, *aargs, **kwargs):
        """
        :param iters:
        :return: path2inst: path_id --> inst (the number from <bos> to <eos>)
                 loss: The XE loss
                 word_accu: per word accuracy
                 sent_accu: per sent accuracy
        """
        path2inst = self.get_insts(*aargs, **kwargs) # record the whole val-seen/val-unseen dataset

        # Calculate the teacher-forcing metrics
        self.env.reset_epoch(shuffle=True)
        N = 3    
        metrics = np.zeros(3)
        for i in range(N):
            self.env.reset()
            metrics += np.array(self.teacher_forcing(train=False))
        metrics /= N
        # metrics: [loss, word_accu, sent_accu]

        return (path2inst, *metrics)

    def make_equiv_action(self, a_t, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[i].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, 'down')
                    src_level -= 1
                while self.env.env.sims[i].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, select_candidate['idx'])

                state = self.env.env.sims[i].getState()[0]
                if traj is not None:
                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def _teacher_action(self, obs, ended, tracker=None):
        """ navigate or select an object as stop!
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        ref = np.zeros(len(obs),dtype=np.int64) 
        len_ref = np.zeros(len(obs),dtype=np.int64)
        ref_info = [[] for i in range(len(obs))] 
        objId = [[] for i in range(len(obs))] 
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
                ref[i] = self.args.ignoreid
                len_ref[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['teacher']:
                    a[i] = len(ob['candidate']) 
                    candidate_objs_ids = ob['obj_ids']
                    for k,kid in enumerate(candidate_objs_ids):
                        if str(kid) == str(ob['gt_obj_id']):
                            ref[i] = k
                            objId[i] = str(kid)
                            len_ref[i] = len(candidate_objs_ids)

                            ref_angle = ob['obj_ang_fts'][k]
                            ref_feat = ob['obj_img_fts'][k]
                            ref_pos = ob['obj_box_fts'][k]
                            ref_name = np.array([ob['obj_name'][k]])
                            ref_info[i] = np.concatenate((ref_angle,ref_feat,ref_pos,ref_name))
                            break
                else:
                    ref[i] = self.args.ignoreid
                    len_ref[i] = self.args.ignoreid
                    for k, candidate in enumerate(ob['candidate']):
                        if candidate['viewpointId'] == ob['teacher']:   # Next view point
                            a[i] = k
                            break
                    else:   # Stop here
                        assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                        a[i] = len(ob['candidate'])
        return a, ref, len_ref, ref_info, objId

    def _candidate_variable(self, obs, actions, use_angle=False):
        candidate_feat = np.zeros((len(obs), self.feature_size + self.args.speaker_angle_size), dtype=np.float32)

        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == -1:  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                c = ob['candidate'][act]
                candidate_feat[i, :] = c['speaker_feature']
        return torch.from_numpy(candidate_feat).cuda()

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), self.args.views, self.feature_size + self.args.speaker_angle_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['speaker_feature']  # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()
    
    def pad_obj_feats(self, obj_feats_list):
        """
        :param obj_feats_list:[(bs,len,dim)]
        """
        bs,_,dim = obj_feats_list[0].shape
        max_length = 1
        for item in obj_feats_list:
            len = item.shape[1]
            if len>max_length:
                max_length = len
        
        for i,_ in enumerate(obj_feats_list):
            len = obj_feats_list[i].shape[1]
            if len<max_length:
                pad_tensor = torch.zeros(bs,max_length-len,dim)
                obj_feats_list[i] = torch.cat((obj_feats_list[i],pad_tensor),dim=1).cuda()
            else:
                obj_feats_list[i] = obj_feats_list[i].cuda()

        return obj_feats_list

    def from_shortest_path(self, viewpoints=None, get_first_feat=False):
        obs = self.env._get_obs()
        ended = np.array([False] * len(obs)) # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []
        gt_objids = [[]for _ in range(len(obs))] # target objids
        gt_objids_len = [[] for _ in range(len(obs))]
        gt_objids_info = np.zeros([len(obs),776],dtype=np.float32)
        objIds = [[] for _ in range(len(obs))] 

        obj_feats, obj_angles, obj_poses,obj_ids = [],[],[],[]
        first_feat = np.zeros((len(obs), self.feature_size+self.args.speaker_angle_size), np.float32)

        for i, ob in enumerate(obs):
            first_feat[i, -self.args.speaker_angle_size:] = utils.angle_feature(ob['heading'], ob['elevation'])
            # get the heading and elevation of the first viewpoint
        first_feat = torch.from_numpy(first_feat).cuda()
        index = 0
        while not ended.all():
            index += 1
            if viewpoints is not None:
                for i, ob in enumerate(obs):
                    viewpoints[i].append(ob['viewpoint'])
            img_feats.append(self._feature_variable(obs))
            obj_feat, obj_angle, obj_pos, obj_lens,obj_id = self._object_variable(obs)
            obj_feats.append(obj_feat)
            obj_angles.append(obj_angle)
            obj_poses.append(obj_pos)
            obj_ids.append(obj_id)
            teacher_action, ref, len_ref, ref_info, objId = self._teacher_action(obs, ended) # [batch_size, ] the index of action
            for j in range(len(obs)):
                if ref[j] != self.args.ignoreid:
                    gt_objids[j] = ref[j]
                    gt_objids_len[j] = len_ref[j]
                    if len(ref_info[j])>0:
                        gt_objids_info[j] = ref_info[j]
                        objIds[j] = objId[j]

            for i, act in enumerate(teacher_action):
                if act < 0 or act == len(obs[i]['candidate']):  # Ignore or Stop
                    teacher_action[i] = -1                      # Stop Action

            can_feats.append(self._candidate_variable(obs, teacher_action))
            # already contain the relavent heading and elevation information.
            self.make_equiv_action(teacher_action, obs)
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == -1))
            obs = self.env._get_obs()
        img_feats = torch.stack(img_feats, 1).cuda()  
        can_feats = torch.stack(can_feats, 1).cuda() 
        gt_objids = torch.from_numpy(np.array(gt_objids)).cuda() 
        gt_objids_info = torch.from_numpy(np.array(gt_objids_info)).cuda() 
        gt_objids_info = gt_objids_info.to(torch.float32)
        if self.args.pred_objids != 'v3':
            gt_objids_info = None

        # note that the length of obj_feats may different! (bs,len,dim)
        max_obj_len = 1
        for item in obj_feats:
            leng = item.shape[1]
            if leng>max_obj_len:
                max_obj_len = leng
        obj_feats = torch.stack(self.pad_obj_feats(obj_feats), 1).cuda()
        obj_angles = torch.stack(self.pad_obj_feats(obj_angles), 1).cuda()
        obj_poses = torch.stack(self.pad_obj_feats(obj_poses), 1).cuda()
        obj_ids = torch.stack(self.pad_obj_feats(obj_ids),1).cuda()

        if get_first_feat:
            return (img_feats, can_feats, first_feat), length
        else:
            return (img_feats, can_feats, obj_feats, obj_angles, obj_poses,obj_ids,gt_objids,gt_objids_info), length,gt_objids_len,max_obj_len,objIds

    def gt_words(self, obs):
        """
        See "utils.Tokenizer.encode_sentence(...)" for "instr_encoding" details
        """
        seq_tensor = np.array([self.tok.encode_sentence(ob['instruction']) for ob in obs])
        try:
            return torch.from_numpy(seq_tensor).cuda()
        except Exception:
            return torch.from_numpy(seq_tensor.astype(int)).cuda()

    def _object_variable(self, obs):
        obj_lens = [max(len(ob['obj_img_fts']), 1) for ob in obs] # in case no object in a vp
        obj_feats = np.zeros((len(obs), max(obj_lens), self.args.obj_feat_size), dtype=np.float32)
        obj_angs = np.zeros((len(obs), max(obj_lens), self.args.angle_feat_size),dtype=np.float32)
        obj_poses = np.zeros((len(obs), max(obj_lens), 3), dtype=np.float32) # [w,h,a]
        obj_ids = np.zeros((len(obs),max(obj_lens),1),dtype=np.float32)
        for i, ob in enumerate(obs):
            obj_img_fts, obj_ang_fts, obj_box_fts = ob['obj_img_fts'], ob['obj_ang_fts'],ob['obj_box_fts']
            obj_name = ob['obj_name']
            if len(obj_img_fts) > 0:
                obj_feats[i, :obj_lens[i]] = obj_img_fts
                obj_poses[i, :obj_lens[i]] = obj_box_fts
                obj_angs[i, :obj_lens[i]] = obj_ang_fts
                obj_name = np.array([int(id) for id in obj_name])
                obj_name = obj_name.reshape(len(obj_name),1)
                obj_ids[i,:obj_lens[i]] = obj_name[:obj_lens[i]]

        obj_angles = torch.from_numpy(obj_angs)
        obj_feats = torch.from_numpy(obj_feats)
        obj_poses = torch.from_numpy(obj_poses)
        obj_ids = torch.from_numpy(obj_ids)
        return obj_feats, obj_angles, obj_poses, obj_lens, obj_ids

    def teacher_forcing(self, train=True, features=None, insts=None):
        if train:
            self.model.train()
        else:
            self.model.eval()

        ctx_mask = None
        obj_ctx_mask = None
        gt_objids_info = None

        # Get Image Input & Encode
        if features is not None:
            assert insts is not None
            (img_feats, can_feats), lengths = features
            batch_size = len(lengths)
        else:
            obs = self.env._get_obs() 
            batch_size = len(obs)
            (img_feats, can_feats, obj_feats, obj_angles, obj_poses,obj_ids, gt_objids,gt_objids_info), lengths, obj_len,max_obj_len,objIds = self.from_shortest_path()     
        
        if self.args.obj_type == "":
            obj_feats, obj_angles, obj_poses = None, None, None

        if not self.args.use_objids:
            obj_ids = None
        noise = False

        # Get Language Input
        if insts is None:
            insts = self.gt_words(obs)                                       # Language Feature
            # insts: [batch_size, tgt_len]

        if self.args.speaker_ctx_mask:
            ctx_mask = utils.length2mask(lengths)

        logits, attns, _, _ = self.model(can_feats, img_feats, insts, obj_feats,obj_angles,obj_poses, ctx_mask=ctx_mask,obj_ids=obj_ids,length=lengths,target_obj_info=gt_objids_info) 
        # logits:[bs,src_len,dim], attns: [bs,1,obj_len] # if pred_objids
        
        # Because the softmax_loss only allow dim-1 to be logit,
        # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
        logits = logits.permute(0, 2, 1)
        loss = self.softmax_loss(
            input  = logits[:, :, :-1],         # -1 for aligning
            target = insts[:, 1:]               # "1:" to ignore the word <BOS>
        )

        if train:   
            return loss
        else:
            # Evaluation
            _, predict = logits.max(dim=1)                                  # BATCH, LENGTH
            gt_mask = (insts != self.tok.word_to_index['<PAD>'])
            correct = (predict[:, :-1] == insts[:, 1:]) * gt_mask[:, 1:]    # Not pad and equal to gt
            correct, gt_mask = correct.type(torch.LongTensor), gt_mask.type(torch.LongTensor)
            word_accu = correct.sum().item() / gt_mask[:, 1:].sum().item()     # Exclude <BOS>
            sent_accu = (correct.sum(dim=1) == gt_mask[:, 1:].sum(dim=1)).sum().item() / batch_size  # Exclude <BOS>
            return loss.item(), word_accu, sent_accu

    def infer_batch(self, sampling=False, train=False, featdropmask=None):
        if train:
            self.model.train()
        else:
            self.model.eval()

        # Image Input for the Encoder
        obs = self.env._get_obs()
        batch_size = len(obs)

        # Get feature
        (img_feats, can_feats, obj_feats, obj_angles, obj_poses,obj_ids, gt_objids,gt_objids_info), lengths, obj_len,max_obj_len,objIds = self.from_shortest_path()      # Image Feature (from the shortest path)

        if self.args.obj_type == "":
            obj_feats, obj_angles, obj_poses = None, None, None
        
        if not self.args.use_objids:
            obj_ids = None

        if self.args.speaker_ctx_mask:
            ctx_mask = utils.length2mask(lengths)

        # This code block is only used for the featdrop.
        if featdropmask is not None:
            img_feats[..., :-self.args.speaker_angle_size] *= featdropmask 
            can_feats[..., :-self.args.speaker_angle_size] *= featdropmask

        # Encoder
        with torch.no_grad():
            enc_inputs, enc_outputs, enc_self_attns = self.model.encoder(can_feats, img_feats,obj_feats, obj_angles, obj_poses, already_dropfeat=(featdropmask is not None),obj_ids=obj_ids,length=lengths,target_obj_info=gt_objids_info)
        
        batch_size = can_feats.size()[0]

        # Decoder
        words = []
        log_probs = []
        entropies = []
        ended = np.zeros(batch_size, np.bool)
        word = np.ones(batch_size, np.int64) * self.tok.word_to_index['<BOS>']   # First word is <BOS>
        word = torch.from_numpy(word).reshape(-1, 1).cuda() # [64, 1]


        next_word = np.ones(batch_size, np.int64) * self.tok.word_to_index['<BOS>'] 
        next_word = torch.from_numpy(next_word).reshape(-1,1).cuda()

        for i in range(self.args.maxDecode):
            # Decode Step
            with torch.no_grad():
                dec_outputs, dec_self_attns, dec_enc_attns = self.model.decoder(word, enc_inputs, enc_outputs, ctx_mask=None)     
                logits = self.model.projection(dec_outputs)

            # Select the word
            logits[:,:, self.tok.word_to_index['<UNK>']] = -float("inf")          # No <UNK> in infer
            if sampling:
                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                word = m.sample()
                log_prob = m.log_prob(word)
                if train:
                    log_probs.append(log_prob)
                    entropies.append(m.entropy())
                else:
                    log_probs.append(log_prob.detach())
                    entropies.append(m.entropy().detach())
            else:
                values, prob = logits.max(dim=-1,keepdim=True)
            next_word = prob[:,-1,0] 

            # Append the word
            next_word[ended] = self.tok.word_to_index['<PAD>'] # [bs]
            word = torch.cat([word.detach(),next_word.unsqueeze(-1)],-1) 

            # End?
            ended = np.logical_or(ended, next_word.cpu().numpy() == self.tok.word_to_index['<EOS>'])
            if ended.all():
                break
        
        if train and sampling:
            return np.stack(words, 1), torch.stack(log_probs, 1), torch.stack(entropies, 1)
        else:
            return word.cpu().numpy()
    
    def train_aug(self):
        self.env.reset()
        loss = self.teacher_forcing(train=True)
        self.loss += loss
    
    def zero_grad(self):
        self.loss = 0.
        self.optimizer.zero_grad()
    
    def optim_step(self):
        self.loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 40.)
        self.optimizer.step()
    
    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("transpeaker", self.model, self.optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        print("Load the speaker's state dict from %s" % path)
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            state = model.state_dict()
            state.update(states[name]['state_dict'])
            model.load_state_dict(state,strict=False) 
            if self.args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'],strict=True)
        all_tuple = [("transpeaker", self.model, self.optimizer)]
        for param in all_tuple:
            recover_state(*param)
        print('load epoch:{}'.format(states['transpeaker']['epoch']-1))
        return states['transpeaker']['epoch'] - 1