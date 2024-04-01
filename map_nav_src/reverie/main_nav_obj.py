import os,sys
root_path = os.getcwd()
current_path = os.path.join(root_path,'map_nav_src')
sys.path.append(root_path)
sys.path.append(current_path)

import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import torch
import string
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from utils.data import ImageFeaturesDB, KMeansPicker

from reverie.agent_obj_goat import GMapObjectNavAgent
from reverie.data_utils import ObjectFeatureDB, construct_instrs, load_obj2vps
from reverie.env import ReverieObjectNavBatch
from reverie.parser import parse_args

from r2r.data_utils import LoadZdict

from reverie.transpeaker_reverie import Speaker
from reverie.spice_scorer import BleuScorer

class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        self.index_to_word = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i
            new_w2i = defaultdict(lambda: self.word_to_index['<UNK>'])
            new_w2i.update(self.word_to_index)
            self.word_to_index = new_w2i
            for key, value in self.word_to_index.items():
                self.index_to_word[value] = key
        old = self.vocab_size()
        self.add_word('<BOS>')
        assert self.vocab_size() == old+1
        print("OLD_VOCAB_SIZE", old)
        print("VOCAB_SIZE", self.vocab_size())
        print("VOACB", len(vocab))

    def finalize(self):
        """
        This is used for debug
        """
        self.word_to_index = dict(self.word_to_index)   # To avoid using mis-typing tokens

    def add_word(self, word):
        assert word not in self.word_to_index
        self.word_to_index[word] = self.vocab_size()  
        self.index_to_word[self.vocab_size()] = word

    @staticmethod
    def split_sentence(sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def vocab_size(self):
        return len(self.index_to_word)

    def encode_sentence(self, sentence, max_length=None):
        if max_length is None:
            max_length = self.encoding_length
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')

        encoding = [self.word_to_index['<BOS>']]
        for word in self.split_sentence(sentence):
            if word in self.word_to_index.keys():
                encoding.append(self.word_to_index[word])   # Default Dict
            else:
                encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])

        if len(encoding) <= 2:
            return None
        #assert len(encoding) > 2

        if len(encoding) < max_length:
            encoding += [self.word_to_index['<PAD>']] * (max_length-len(encoding))  # Padding
        elif len(encoding) > max_length:
            encoding[max_length - 1] = self.word_to_index['<EOS>']                  # Cut the length with EOS

        return np.array(encoding[:max_length])

    def decode_sentence(self, encoding, length=None):
        sentence = []
        if length is not None:
            encoding = encoding[:length]
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.index_to_word[ix])
        return " ".join(sentence)

    def shrink(self, inst):
        """
        :param inst:    The id inst
        :return:  Remove the potential <BOS> and <EOS>
                  If no <EOS> return empty list
        """
        if len(inst) == 0:
            return inst
    
        new_inst = []
        for i in range(len(inst)-1):
            if inst[i] == inst[i+1]:
                continue
            new_inst.append(inst[i])
        inst = new_inst

        end = np.argmax(np.array(inst) == self.word_to_index['<EOS>'])  
        if end == 0: 
            end = len(inst)
            
        if len(inst) > 1 and inst[0] == self.word_to_index['<BOS>']:
            start = 1
        else:
            start = 0
        return inst[start: end]

def build_dataset(args, rank=0):
    # Tokenizer for speaker
    with open(args.speaker_train_vocab) as f:
        vocab = [word.strip() for word in f.readlines()]

    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    elif args.tokenizer == 'roberta':
        # cfg_name = 'roberta-base'
        cfg_name = 'datasets/pretrained/roberta'
    else:
        cfg_name = 'bert-base-uncased'
    try:
        bert_tok = AutoTokenizer.from_pretrained(cfg_name)
    except Exception:
        cfg_name = '../' + cfg_name
        bert_tok = AutoTokenizer.from_pretrained(cfg_name)
    print('Tokenizer: {}'.format(cfg_name))

    speaker_tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)
    if 'speaker' in args.mode:
        tok = speaker_tok
    else:
        tok = bert_tok
    
    # For do-intervention
    instr_z_file = args.instr_zdict_file
    instr_zdict_file = args.backdoor_dict_file if len(args.backdoor_dict_file)>1 else instr_z_file
    
    ZdictReader = LoadZdict(args.img_zdict_file,instr_zdict_file)
    z_dicts = defaultdict(lambda:None)
    if args.do_back_img:
        img_zdict = ZdictReader.load_img_tensor()
        z_dicts['img_zdict'] = img_zdict
    if args.do_back_txt:
        instr_zdict = ZdictReader.load_instr_tensor()
        z_dicts['instr_zdict'] = instr_zdict
    
    if args.do_front_img or args.do_front_his or args.do_front_txt:
        front_feat_loader = KMeansPicker(args.front_feat_file,n_clusters=args.front_n_clusters)
    else:
        front_feat_loader = None

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    obj_db = ObjectFeatureDB(args.obj_ft_file, args.obj_feat_size, cat_file=args.cat_file)
    obj2vps = load_obj2vps(os.path.join(args.anno_dir, 'BBoxes.json'))

    # Use env_edit?
    if args.env_aug == 'env_edit':
        print('use env_edit features!!')
        envedit_feat_db = ImageFeaturesDB(args.envedit_ft_file, args.image_feat_size)
        train_feat_db = [feat_db,envedit_feat_db]
    else:
        train_feat_db = feat_db

    dataset_class = ReverieObjectNavBatch

    # Load augmented dataset
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        aug_env = dataset_class(
            train_feat_db, obj_db, aug_instr_data, args.connectivity_dir, obj2vps, 
            batch_size=args.batch_size, max_objects=args.max_objects,
            angle_feat_size=args.angle_feat_size, 
            seed=args.seed+rank, sel_data_idxs=None, name='aug', 
            multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints,tok=tok,args=args,
            scanvp_cands_file=args.scanvp_cands_file
        )
    else:
        aug_env = None
        aug_instr_data = None

    # Load the training set
    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], 
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
    )
    train_env = dataset_class(
        train_feat_db, obj_db, train_instr_data, args.connectivity_dir, obj2vps,
        batch_size=args.batch_size, max_objects=args.max_objects,
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train', 
        multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints,tok=tok,args=args,
        scanvp_cands_file=args.scanvp_cands_file
    )

    val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    if args.submit:
        val_env_names.append('test')
        
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        val_env = dataset_class(
            feat_db, obj_db, val_instr_data, args.connectivity_dir, obj2vps, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
            max_objects=None, multi_endpoints=False, multi_startpoints=False,tok=tok,args=args,
            scanvp_cands_file=args.scanvp_cands_file
        )   # evaluation using all objects
        val_envs[split] = val_env

    return train_env, val_envs, aug_env, bert_tok, speaker_tok, z_dicts, train_instr_data, front_feat_loader

def train_speaker(args, train_env, val_envs, tok, n_iters, log_every=150, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)
    '''train the speaker model'''
    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    speaker = Speaker(args,train_env,tok)
    bleu_scorer = BleuScorer()
    best_bleu = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 1232)
    best_both_bleu = 0 # rocord the best bleu both on seen and unseen environments

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        if args.scorer_pretrain: # load scorer
            start_iter = speaker.load_scorer(os.path.join(args.resume_file))
        else:
            start_iter = speaker.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration {}".format(args.resume_file, start_iter),
                record_file
            )

    if default_gpu:
        write_to_record_file(
            '\nSpeaker training starts, start iteration: %s' % str(start_iter), record_file
        )

    for idx in range(start_iter, n_iters, log_every):
        interval = min(log_every, n_iters - idx)
        iter = idx + interval

        if aug_env is None:
            # Train for log_every interval
            speaker.env = train_env
            train_loss = speaker.train(interval)   # Train interval iters
            writer.add_scalar("loss/train_loss",train_loss,idx)
        else:
            # semi-supervised for speaker
            if args.accumulate_grad:
                train_loss = 0.
                for jdx in range(interval // 2):
                    speaker.zero_grad()

                    # Train with GT data
                    speaker.env = train_env
                    speaker.train_aug()

                    speaker.env = aug_env
                    speaker.train_aug()

                    # Train with Aug data
                    speaker.optim_step()
                    train_loss += speaker.loss
                writer.add_scalar("loss/train_loss",train_loss,idx)
            else:
                jdx_length = len(range(interval // 2))
                for jdx in range(interval // 2):
                    # Train with GT data
                    speaker.env = train_env
                    # args.ml_weight = 0.2
                    loss = speaker.train(1, feedback=args.feedback)
                    train_loss += loss

                    # Train with Augmented data
                    speaker.env = aug_env
                    # args.ml_weight = 0.2
                    loss = speaker.train(1, feedback=args.feedback)
                    train_loss += loss
                writer.add_scalar("loss/train_loss",train_loss,idx)
        write_to_record_file('\nIter: %d' % idx, record_file)
        write_to_record_file(f"Train loss: {train_loss}", record_file)

        # Evaluation
        current_valseen_bleu4 = 0
        current_valunseen_bleu4 = 0
        for env_name, env in val_envs.items():
            write_to_record_file("............ Evaluating %s ............." % env_name, record_file)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid()
            write_to_record_file(f"Evaluation loss: {loss}",record_file)
            path_id = list(path2inst.keys())[0]
            write_to_record_file("Inference: %s" % tok.decode_sentence(path2inst[path_id]['inst']), record_file)
            write_to_record_file("GT: %s" % env.gt_insts[path_id], record_file)

            data = []
            # Compute BLEU
            for path_id in path2inst.keys():
                inference_sentences = tok.decode_sentence(path2inst[path_id]['inst'])
                gt_sentences = env.gt_insts[path_id]
                temp_data = {
                    "path_id":path_id,
                    "Inference": [inference_sentences],
                    "Ground Truth":gt_sentences,
                }
                data.append(temp_data)
            if env_name == 'val_seen':
                best_data_seen = data
            elif env_name == 'val_unseen':
                best_data_unseen = data
            precisions = bleu_scorer.compute_scores(data) 
            bleu_score = precisions[3] # use bleu4
            
            # save the best check-point both on seen and unseen subsets
            if env_name == 'val_seen':
                current_valseen_bleu4 = bleu_score
            elif env_name == 'val_unseen':
                current_valunseen_bleu4 = bleu_score
                if current_valunseen_bleu4 + current_valseen_bleu4 >= best_both_bleu:
                    best_both_bleu = current_valunseen_bleu4 + current_valseen_bleu4
                    write_to_record_file(
                        'Save the model with val_seen BEST env bleu %0.4f and val_unseen BEST env bleu %0.4f' % (current_valseen_bleu4,current_valunseen_bleu4),record_file
                    )
                    # saving model
                    speaker.save(idx, os.path.join(args.ckpt_dir, 'best_both_bleu.pt'))

            # Tensorboard log
            writer.add_scalar("bleu/%s" % (env_name), precisions[0], idx)
            writer.add_scalar("loss/%s" % (env_name), loss, idx)
            writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
            writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
            writer.add_scalar("bleu4/%s" % (env_name), precisions[3], idx)

            if bleu_score > best_bleu[env_name]:
                best_bleu[env_name] = bleu_score
                write_to_record_file(
                    'Save the model with %s BEST env bleu %0.4f' % (env_name, bleu_score),record_file
                )

            if loss < best_loss[env_name]:
                best_loss[env_name] = loss
                write_to_record_file('Save the model with %s BEST env loss %0.4f' % (env_name, loss),record_file)
            
            write_to_record_file(
                "Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions[:4]),record_file
                )
            if len(best_bleu)!=0:
                write_to_record_file(
                    "Best bleu: %0.4f, Best loss: %0.4f" % (best_bleu[env_name],best_loss[env_name]),record_file
                    )

def train(args, train_env, val_envs, aug_env=None, rank=-1, tok=None, speaker_tok=None,
          z_dicts=None,train_instr_data=None,front_feat_loader=None):
    '''train the navigator'''
    default_gpu = is_default_gpu(args)
    speaker = None

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = GMapObjectNavAgent
    listner = agent_class(args, train_env, rank=rank,tok=tok)
    
    # Load intervention features
    if args.z_instr_update:
        z_dicts, landmark_dict, landmark_pz_dict = listner.update_z_dict(train_instr_data,z_dicts,z_dict=z_dicts)
    
    if args.do_front_img or args.do_front_his or args.do_front_txt:
        if len(args.frontdoor_dict_file) > 0: # load specific file
            z_front_dict =  front_feat_loader.read_tim_tsv(args.frontdoor_dict_file, return_dict=True)
        else: # random sample
            z_front_dict = front_feat_loader.random_pick_front_features(args, iter=0, save_file=True)
    else:
        z_front_dict = None

    # use speaker to provide pseudo labels
    if args.use_transpeaker:
        speaker = Speaker(args,train_env,speaker_tok)
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)
        print("Load speaker model successully.")

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration {}".format(args.resume_file, start_iter),
                record_file
            )
       
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            prefix = 'submit' if args.detailed_output is False else 'detail'
            output_file = os.path.join(args.pred_dir, "%s_%s_%s.json" % (
                prefix, env_name, args.fusion))
            # if os.path.exists(output_file):
            #     continue
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results(detailed_output=args.detailed_output)
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            
            if default_gpu and env_name != 'test':
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
            
            for i,_ in enumerate(preds):
                preds[i]["predObjId"] = preds[i].pop("pred_objid")
                if preds[i]["predObjId"] is not None:
                    preds[i]["predObjId"] = int(preds[i]["predObjId"])
            
            if args.submit:
                with open(output_file,'w') as f:
                    json.dump(preds, f, sort_keys=True, indent=4, separators=(',', ': '))
                print('submit file has been saved in {}.'.format(output_file))

        if default_gpu:
            write_to_record_file(loss_str, record_file)

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {'val_unseen': {"rgs": 0., "rgspl": 0., "state":""}}

    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        is_update = False

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback, z_dicts=z_dicts,z_front_dict=z_front_dict)  # Train interval iters
        else:
            if args.accumulate_grad: # accumulateGrad
                jdx_length = len(range(interval // 2))
                for jdx in range(interval // 2):
                    listner.zero_grad()
                    listner.env = train_env

                    # Train with GT data
                    listner.accumulate_gradient(args.feedback, z_dicts=z_dicts, z_front_dict=z_front_dict)
                    listner.env = aug_env

                    # Train with Back Translation
                    listner.accumulate_gradient(args.feedback, speaker=speaker, z_dicts=z_dicts, z_front_dict=z_front_dict)
                    listner.optim_step()
                    
                    if default_gpu:
                        print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

            else:
                jdx_length = len(range(interval // 2))
                for jdx in range(interval // 2):
                    # Train with GT data
                    listner.env = train_env
                    listner.train(1, feedback=args.feedback)

                    # Train with Augmented data
                    listner.env = aug_env
                    listner.train(1, feedback=args.feedback)

                    if default_gpu:
                        print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            OG_loss = sum(listner.logs['OG_loss']) / max(len(listner.logs['OG_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/OG_loss", OG_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, OG_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, OG_loss, policy_loss, critic_loss),
                record_file
            )

        # Run validation
        loss_str = "iter {}".format(iter)
                
        for env_name, env in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None, z_dicts=z_dicts, z_front_dict=z_front_dict)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

                # select model by rgs + rgspl
                if env_name in best_val:
                    if score_summary['rgs'] + score_summary['rgspl'] >= best_val[env_name]['rgs'] + score_summary['rgspl']:
                        best_val[env_name]['rgs'] = score_summary['rgs']
                        best_val[env_name]['rgspl'] = score_summary['rgspl']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_%s.pt" % (env_name)))

                        if args.z_instr_update:
                            is_update = True
                            listner.save_backdoor_z_dict(landmark_dict, landmark_pz_dict)
                            z_dicts, landmark_dict, landmark_pz_dict = listner.update_z_dict(train_instr_data,z_dicts)
                            
                        if args.do_front_img or args.do_front_his or args.do_front_txt:
                            front_feat_loader.save_features(args, z_front_dict)
                            z_front_dict = front_feat_loader.random_pick_front_features(args, iter, save_file=True)
        
        if args.z_instr_update and iter%(args.update_iter)==0 and (not is_update):
            if args.do_back_txt:
                z_dicts, landmark_dict, landmark_pz_dict = listner.update_z_dict(train_instr_data,z_dicts)
            
            if args.do_front_img or args.do_front_his or args.do_front_txt:
                z_front_dict = front_feat_loader.random_pick_front_features()
        
        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict.pt"))

            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)


def valid(args, train_env, val_envs, rank=-1, z_dicts={}, front_feat_loader=None):
    '''valid navigator'''
    default_gpu = is_default_gpu(args)

    agent_class = GMapObjectNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
    
    if args.do_front_img or args.do_front_his or args.do_front_txt:
        if len(args.frontdoor_dict_file) > 0:
            z_front_dict =  front_feat_loader.read_tim_tsv(args.frontdoor_dict_file, return_dict=True)
        else:
            z_front_dict = front_feat_loader.random_pick_front_features()
    else:
        z_front_dict = None

    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False else 'detail'
        output_file = os.path.join(args.pred_dir, "%s_%s_%s.json" % (
            prefix, env_name, args.fusion))
        if os.path.exists(output_file):
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters, z_dicts=z_dicts, z_front_dict=z_front_dict)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)

            if args.submit:
                json.dump(
                    preds, open(output_file, 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
                print('pred file has been saved in {}.'.format(output_file))
               
def extract_cfp_features(args, train_env, rank=0, bert_tok=None, save_file=True):
    agent_class = GMapObjectNavAgent
    agent = agent_class(args, train_env, rank=rank, tok=bert_tok)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))
    
    agent.extract_cfp_features(train_env.data, save_file=save_file)


def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env, bert_tok, speaker_tok, z_dicts, train_instr_data, front_feat_loader = build_dataset(args, rank=rank)
            
    if args.mode == 'train':
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank,tok=bert_tok,speaker_tok=speaker_tok,z_dicts=z_dicts,train_instr_data=train_instr_data,front_feat_loader=front_feat_loader)
    elif args.mode == 'valid':
        valid(args, train_env, val_envs, rank=rank, z_dicts=z_dicts, front_feat_loader=front_feat_loader)
    elif args.mode == 'extract_cfp_features':
        extract_cfp_features(args, train_env, bert_tok=bert_tok)
    elif args.mode == 'speaker':
        train_speaker(args,train_env,val_envs,tok=speaker_tok,n_iters=args.iters,aug_env=aug_env,rank=rank,log_every=args.log_every)

if __name__ == '__main__':
    main()