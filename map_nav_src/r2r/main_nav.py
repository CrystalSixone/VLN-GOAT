import os,sys
root_path = os.getcwd()
current_path = os.path.join(root_path,'map_nav_src')
sys.path.append(root_path)
sys.path.append(current_path)
import json
import time
import numpy as np
import copy
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from utils.data import ImageFeaturesDB, Tokenizer, KMeansPicker
from r2r.transpeaker import Speaker
from r2r.data_utils import construct_instrs
from r2r.env import R2RNavBatch
from r2r.parser import parse_args

from r2r.agent import GMapNavAgent
from data_utils import LoadZdict


def build_dataset(args, rank=0):
    # Load vocab for speaker
    with open(args.speaker_train_vocab) as f:
        vocab = [word.strip() for word in f.readlines()]
    speaker_tok = Tokenizer(vocab=vocab, encoding_length=args.max_instr_len)
    
    try:
        bert_tok = AutoTokenizer.from_pretrained('../datasets/pretrained/roberta')        
    except Exception:
        bert_tok = AutoTokenizer.from_pretrained(os.path.join(root_path,'datasets','pretrained','roberta')) # for Debug
    
    # For do-intervention
    if args.dataset == 'rxr':
        instr_z_file = args.rxr_instr_zdict_roberta_file
    else:
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
        front_feat_file = args.rxr_front_feat_file if args.dataset=='rxr' else args.front_feat_file
        front_feat_loader = KMeansPicker(front_feat_file,n_clusters=args.front_n_clusters)
    else:
        front_feat_loader = None

    # Load features
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    
    # Use augmented features
    if args.use_aug_env:
        train_feat_db = [feat_db]
        if args.env_edit:
            envedit_feat_db = ImageFeaturesDB(args.aug_img_ft_file_envedit, args.image_feat_size)
            train_feat_db.append(envedit_feat_db)    
        if len(train_feat_db) == 1:
            train_feat_db = feat_db
    else:
        train_feat_db = feat_db

    dataset_class = R2RNavBatch

    instr_tok = bert_tok
    if args.aug is not None: # trajectory & instruction aug
        aug_feat_db = train_feat_db
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len, for_debug=args.for_debug,
            tok=instr_tok,
            is_rxr=(args.dataset=='rxr')
        )
        aug_env = dataset_class(
            aug_feat_db, aug_instr_data, args.connectivity_dir, 
            batch_size=args.batch_size, angle_feat_size=args.angle_feat_size, 
            seed=args.seed+rank, sel_data_idxs=None, name='aug', 
            args=args, scanvp_cands_file=args.scanvp_cands_file
        )
    else:
        aug_env = None

    # Load the training dataset
    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], 
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len, for_debug=args.for_debug,
        tok=instr_tok, is_rxr=(args.dataset=='rxr')
    )
    train_env = dataset_class(
        train_feat_db, train_instr_data, args.connectivity_dir,
        batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train', 
        args=args, scanvp_cands_file=args.scanvp_cands_file
    )

    val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']
    if args.dataset == 'rxr':
        val_env_names.remove('val_train_seen')
        if not args.submit:
            val_env_names.remove('val_seen')
    
    if args.submit and args.dataset != 'rxr':
        val_env_names.append('test')
        
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,for_debug=args.for_debug,
            tok=instr_tok, is_rxr=(args.dataset=='rxr')
        )
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
            args=args, scanvp_cands_file=args.scanvp_cands_file
        )  
        val_envs[split] = val_env

    return train_env, val_envs, aug_env, bert_tok, speaker_tok, z_dicts, train_instr_data, front_feat_loader


def train(args, train_env, val_envs, aug_env=None, rank=-1, bert_tok=None, speaker_tok=None, z_dicts=None,train_instr_data=None,front_feat_loader=None):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = GMapNavAgent
    listner = agent_class(args, train_env, rank=rank, tok=bert_tok)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration {}".format(args.resume_file, start_iter),
                record_file
            )
        # start_iter = 0
       
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        if default_gpu:
            write_to_record_file(loss_str, record_file)

    if args.do_front_img or args.do_front_his or args.do_front_txt:
        if len(args.frontdoor_dict_file) > 0:
            z_front_dict =  front_feat_loader.read_tim_tsv(args.frontdoor_dict_file, return_dict=True)
        else:
            z_front_dict = front_feat_loader.random_pick_front_features()
    else:
        z_front_dict = None
    
    if args.z_instr_update:
        if args.do_back_txt:
            z_dicts, landmark_dict, direction_dict, landmark_pz_dict, direction_pz_dict = listner.update_z_dict(train_instr_data, z_dict=z_dicts)

    if args.use_transpeaker:
        speaker = Speaker(args,train_env,speaker_tok)
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)
        print("Load speaker model successully.")
    else:
        speaker = None

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":"", "both": 0.}}
    if args.dataset == 'rxr':
        best_val = {'val_unseen': {"nDTW": 0., "SDTW": 0., "state":"", "both": 0.}}
    
    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        is_update = False

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback, z_dicts=z_dicts, z_front_dict=z_front_dict)  # Train interval iters
        else:
            if args.accumulate_grad: # accumulateGrad
                jdx_length = len(range(interval // (args.aug_times+1)))
                for jdx in range(interval // (args.aug_times+1)):
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
                jdx_length = len(range(interval // (args.aug_times+1)))
                for jdx in range(interval // (args.aug_times+1)):
                    # Train with GT data
                    listner.env = train_env
                    listner.train(1, feedback=args.feedback,z_dicts=z_dicts, z_front_dict=z_front_dict)

                    # Train with Augmented data
                    listner.env = aug_env
                    listner.train(args.aug_times, feedback=args.feedback, speaker=speaker, z_dicts=z_dicts, z_front_dict=z_front_dict)

                    if default_gpu:
                        print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            if args.use_lr_sch:
                LR = sum(listner.logs['lr']) / max(len(listner.logs['lr']), 1)
                writer.add_scalar("lr", LR,idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, RL_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, RL_loss, policy_loss, critic_loss),
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

                # select model by spl
                if env_name in best_val:
                    if args.dataset == 'rxr':
                        target_metric = 'nDTW'
                        target_metric_2 = 'SDTW'
                    else:
                        target_metric = 'spl'
                        target_metric_2 = 'sr'

                    if score_summary[target_metric] + score_summary[target_metric_2] >= best_val[env_name]['both']:
                        best_val[env_name][target_metric] = score_summary[target_metric]
                        best_val[env_name][target_metric_2] = score_summary[target_metric_2]
                        best_val[env_name]['both'] = score_summary[target_metric] + score_summary[target_metric_2]
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_%s.pt" % (env_name)))

                        if args.z_instr_update:
                            is_update = True
                            if args.do_back_txt:
                                listner.save_backdoor_z_dict(landmark_dict, direction_dict, landmark_pz_dict, direction_pz_dict)
                                z_dicts, landmark_dict, direction_dict, landmark_pz_dict, direction_pz_dict = listner.update_z_dict(train_instr_data, z_dicts)
                            if args.do_front_img or args.do_front_his or args.do_front_txt:
                                front_feat_loader.save_features(args, z_front_dict)
                                z_front_dict = front_feat_loader.random_pick_front_features()

        if args.z_instr_update and iter%(args.update_iter)==0 and (not is_update):
            if args.do_back_txt:
                z_dicts, landmark_dict, direction_dict, landmark_pz_dict, direction_pz_dict = listner.update_z_dict(train_instr_data,z_dicts)
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


def valid(args, train_env, val_envs, rank=-1, z_dicts=None, front_feat_loader=None):
    default_gpu = is_default_gpu(args)

    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
    
    # Load front-door dictionary
    if args.do_front_img or args.do_front_his or args.do_front_txt:
        if len(args.frontdoor_dict_file) > 0:
            z_front_dict =  front_feat_loader.read_tim_tsv(args.frontdoor_dict_file, return_dict=True)
        else:
            z_front_dict = front_feat_loader.random_pick_front_features()
    else:
        z_front_dict = None

    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False else 'detail'
        # if os.path.exists(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name))):
        #     continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters,z_dicts=z_dicts, z_front_dict=z_front_dict)
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
                    preds,
                    open(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name)), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )

def extract_cfp_features(args, train_env, z_dict, rank=0, bert_tok=None, save_file=True):
    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank, tok=bert_tok)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))
    
    agent.extract_cfp_features(train_env.data, z_dict=z_dict, save_file=save_file)

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
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank, bert_tok=bert_tok, speaker_tok=speaker_tok, z_dicts=z_dicts,train_instr_data=train_instr_data,front_feat_loader=front_feat_loader)
    elif args.mode == 'valid':
        valid(args, train_env, val_envs, rank=rank, z_dicts=z_dicts,front_feat_loader=front_feat_loader)
    elif args.mode == 'extract_cfp_features':
        extract_cfp_features(args, train_env, z_dict=z_dicts, bert_tok=bert_tok)
            
if __name__ == '__main__':
    main()
