import os
import time
from collections import defaultdict
from easydict import EasyDict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist

import torch.cuda.amp as amp  

from transformers import AutoTokenizer, PretrainedConfig
from transformers import AutoModel

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_dropout, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from optim import get_lr_sched
from optim.misc import build_optimizer

from parser import load_parser, parse_with_config, postprocess_args

from data.loader import MetaLoader, PrefetchLoader, build_dataloader
from data.dataset import ReverieTextPathData
from data.dataset import read_img_features_from_h5py, read_reverie_obj_features, read_category_file
from data.tasks import (
    MlmDataset, mlm_collate,
    MrcDataset, mrc_collate,
    SapDataset, sap_collate,
    OGDataset, og_collate,
    CfpDataset, cfp_collate)

from model.pretrain_goat import GlocalTextPathCMTPreTraining

def create_dataloaders(
    data_cfg, nav_db, tok, is_train: bool, device: torch.device, opts
):
    dataloaders = {}
    for k, task_name in enumerate(data_cfg.tasks):
        if task_name == 'mlm':
            task_dataset = MlmDataset(nav_db, tok)
            task_collate_fn = mlm_collate
        elif task_name == 'mrc':
            task_dataset = MrcDataset(nav_db, tok, opts.mrc_mask_prob)
            task_collate_fn = mrc_collate
        elif task_name == 'sap':
            task_dataset = SapDataset(nav_db, tok)
            task_collate_fn = sap_collate
        elif task_name == 'og':
            task_dataset = OGDataset(nav_db, tok)
            task_collate_fn = og_collate
        elif task_name == 'cfp':
            task_dataset = CfpDataset(nav_db, tok)
            task_collate_fn = cfp_collate
        else:
            raise ValueError(f'Undefined task {task_name}')

        LOGGER.info(f"{task_name}: {len(task_dataset)} samples loaded")

        task_loader, pre_epoch = build_dataloader(
            task_name, task_dataset, task_collate_fn, is_train, opts
        )

        if is_train:
            ratio = data_cfg.mix_ratio[k]
            dataloaders[task_name] = (task_loader, ratio, pre_epoch)
        else:
            dataloaders[task_name] = PrefetchLoader(task_loader, device)
    return dataloaders


def main(opts):
    default_gpu, n_gpu, device = set_cuda(opts)

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}'.format(
                device, n_gpu, bool(opts.local_rank != -1), opts.fp16
            )
        )
 
    seed = opts.seed
    if opts.local_rank != -1:
        seed += opts.rank
    set_random_seed(seed)

    if default_gpu:
        save_training_meta(opts)
        TB_LOGGER.create(os.path.join(opts.output_dir, 'logs'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # Model config
    model_config = PretrainedConfig.from_json_file(opts.model_config)
    model_config.pretrain_tasks = []
    for train_dataset_config in opts.train_datasets.values():
        model_config.pretrain_tasks.extend(train_dataset_config['tasks'])
    model_config.pretrain_tasks = set(model_config.pretrain_tasks)
    model_config.cuda_first_device = opts.cuda_first_device

    tokenizer = AutoTokenizer.from_pretrained(model_config.lang_bert_name)
    LOGGER.info(f'Tokenizer: {tokenizer}')

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint, map_location=lambda storage, loc: storage)
    else:
        checkpoint = {}
        if opts.init_pretrained == 'bert':
            tmp = AutoModel.from_pretrained(model_config.lang_bert_name)
            for param_name, param in tmp.named_parameters():
                checkpoint[param_name] = param
            if model_config.lang_bert_name == 'xlm-roberta-base':
                # embeddings.token_type_embeddings.weight (1 -> 2, the second is for image embedding)
                checkpoint['embeddings.token_type_embeddings.weight'] = torch.cat(
                    [checkpoint['embeddings.token_type_embeddings.weight']] * 2, 0
                )
            del tmp
        elif opts.init_pretrained == 'lxmert':
            try:
                tmp = torch.load(
                    '../datasets/pretrained/LXMERT/model_LXRT.pth', 
                    map_location=lambda storage, loc: storage
                )
            except Exception:
                tmp = torch.load(
                    'datasets/pretrained/LXMERT/model_LXRT.pth', 
                    map_location=lambda storage, loc: storage
                )
            for param_name, param in tmp.items():
                param_name = param_name.replace('module.', '')
                if 'bert.encoder.layer' in param_name:
                    param_name = param_name.replace('bert.encoder.layer', 'bert.lang_encoder.layer')
                    checkpoint[param_name] = param
                elif 'bert.encoder.x_layers' in param_name:
                    param_name1 = param_name.replace('bert.encoder.x_layers', 'bert.local_encoder.encoder.x_layers')
                    param_name2 = param_name.replace('bert.encoder.x_layers', 'bert.global_encoder.encoder.x_layers')
                    checkpoint[param_name1] = checkpoint[param_name2] = param
                elif 'cls.predictions' in param_name:
                    param_name = param_name.replace('cls.predictions', 'mlm_head.predictions')
                    checkpoint[param_name] = param
                else:
                    checkpoint[param_name] = param
            del tmp
        elif opts.init_pretrained == 'meter':
            try:
                tmp = torch.load('../datasets/pretrained/METER/meter_clip16_224_roberta_pretrain.ckpt')
            except Exception:
                tmp = torch.load('datasets/pretrained/METER/meter_clip16_224_roberta_pretrain.ckpt')
            tmp = tmp['state_dict']
            for param_name, param in tmp.items():
                if 'text_transformer.embeddings' in param_name:
                    param_name = param_name.replace('text_transformer.', 'bert.')
                    checkpoint[param_name] = param
                elif 'text_transformer.encoder' in param_name:
                    param_name = param_name.replace('text_transformer.encoder', 'bert.lang_encoder')
                    checkpoint[param_name] = param
                elif 'cross_modal_image_layers' in param_name:
                    param_name1 = param_name.replace('cross_modal_image_layers', 'bert.local_encoder.encoder.crossattention')
                    param_name2 = param_name.replace('cross_modal_image_layers', 'bert.global_encoder.encoder.crossattention')
                    checkpoint[param_name1] = checkpoint[param_name2] = param
                else:
                    checkpoint[param_name] = param
            del tmp

    data_cfg = EasyDict(opts.train_datasets['REVERIE'])
    LOGGER.info(f'   Use {opts.init_pretrained } model to initialize.')

    if 'roberta' in model_config.lang_bert_name:
        data_cfg.train_traj_files = data_cfg.train_roberta_files
        data_cfg.val_seen_traj_files = data_cfg.val_seen_roberta_files
        data_cfg.val_unseen_traj_files = data_cfg.val_unseen_roberta_files
    LOGGER.info(f'   Use {model_config.lang_bert_name} tokenizer')
    
    # img feature
    if model_config.img_feature_type == 'clip768':
        model_config.img_file_type = 'hdf5'
        model_config.image_feat_size = 768
        data_cfg.img_ft_file = data_cfg.clip768_img_ft_file
    else:
        LOGGER.info(f' Wrong image features')
        return
    
    model_config.name = data_cfg.name
    LOGGER.info(f'   Use {model_config.img_feature_type} image features')

    
    model_class = GlocalTextPathCMTPreTraining
    
    # update some training configs
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, config=model_config, state_dict=checkpoint
    )
    model.train()
    set_dropout(model, opts.dropout)
    model = wrap_model(model, device, opts.local_rank)
    del checkpoint
        
    img_ft_db = read_img_features_from_h5py(data_cfg.img_ft_file, model_config.image_feat_size)
    cat_mapping, category_number = read_category_file(data_cfg.cat_file)
    obj_ft_db = read_reverie_obj_features(data_cfg.obj_ft_file, opts.max_objects, model_config.obj_feat_size, model_config.obj_prob_size,cat_mapping,category_number)
    aug_img_db = read_img_features_from_h5py(data_cfg.aug_img_file, model_config.image_feat_size)
    # load data training set
    train_nav_db = ReverieTextPathData(
        data_cfg.train_traj_files, img_ft_db, obj_ft_db,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size, 
        angle_feat_size=model_config.angle_feat_size,
        obj_feat_size=model_config.obj_feat_size,
        obj_prob_size=model_config.obj_prob_size,
        max_txt_len=opts.max_txt_len, in_memory=True,
        cat_file=data_cfg.cat_file,
        args=model_config, tok=tokenizer,
        aug_img_db=aug_img_db
    )
    val_nav_db = ReverieTextPathData(
        data_cfg.val_seen_traj_files, img_ft_db, obj_ft_db,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size, 
        angle_feat_size=model_config.angle_feat_size,
        obj_feat_size=model_config.obj_feat_size,
        obj_prob_size=model_config.obj_prob_size,
        max_txt_len=opts.max_txt_len, in_memory=True,
        cat_file=data_cfg.cat_file,
        args=model_config, tok=tokenizer,
        aug_img_db=aug_img_db
    )
    val2_nav_db = ReverieTextPathData(
        data_cfg.val_unseen_traj_files, img_ft_db, obj_ft_db,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size, 
        angle_feat_size=model_config.angle_feat_size,
        obj_feat_size=model_config.obj_feat_size,
        obj_prob_size=model_config.obj_prob_size,
        max_txt_len=opts.max_txt_len, in_memory=True,
        cat_file=data_cfg.cat_file,
        args=model_config, tok=tokenizer,
        aug_img_db=aug_img_db
    )

    # Build data loaders
    train_dataloaders = create_dataloaders(
        data_cfg, train_nav_db, tokenizer, True, device, opts
    )
    val_dataloaders = create_dataloaders(
        data_cfg, val_nav_db, tokenizer, False, device, opts
    )
    val2_dataloaders = create_dataloaders(
        data_cfg, val2_nav_db, tokenizer, False, device, opts
    )
    meta_loader = MetaLoader(
        train_dataloaders,
        accum_steps=opts.gradient_accumulation_steps,
        distributed=opts.local_rank != -1,
        device=device
    )
    meta_loader = PrefetchLoader(meta_loader, device)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}

    if opts.fp16:
        grad_scaler = amp.GradScaler()

    global_step = 0
    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.train_batch_size if opts.local_rank == -1 else opts.train_batch_size * opts.world_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    # to compute training statistics
    task2loss = {task: RunningMeter(f'loss/{task}')
                 for task in train_dataloaders.keys()}

    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)
    grad_norm = 0

    start_time = time.time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    max_unseen_facc = 0
    for step, (name, batch) in enumerate(meta_loader):
        # forward pass
        n_examples[name] += batch['txt_ids'].size(0)
        n_in_units[name] += batch['txt_lens'].sum().item()
        task = name.split('_')[0]
        # print(name, task)
        # for k, v in batch.items():
        #     print(k, v.size())
        # continue
        if opts.fp16:
            with amp.autocast():
                loss = model(batch, task=task, compute_loss=True)
        else:
            loss = model(batch, task=task, compute_loss=True)

        n_loss_units[name] += loss.size(0)
        loss = loss.mean()  # loss is not normalized in model

        # backward pass
        if args.gradient_accumulation_steps > 1: # average loss 
            loss = loss / args.gradient_accumulation_steps

        delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
        if opts.fp16:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        task2loss[name](loss.cpu().detach().numpy().item())

        # optimizer update and logging
        if (step + 1) % opts.gradient_accumulation_steps == 0:
            global_step += 1

            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, opts)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
            # NOTE: not gathered across GPUs for efficiency
            TB_LOGGER.log_scalar_dict({ll.name: ll.val
                                       for ll in task2loss.values()
                                       if ll.val is not None})
            TB_LOGGER.step()

            # update model params
            if opts.grad_norm != -1:
                if opts.fp16:
                    grad_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opts.grad_norm
                )
                # print(step, name, grad_norm)
                # for k, v in model.named_parameters():
                #     if v.grad is not None:
                #         v = torch.norm(v).data.item()
                #         print(k, v)
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            if opts.fp16:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)

            if global_step % opts.log_steps == 0:
                # monitor training throughput
                LOGGER.info(f'==============Step {global_step}===============')
                for t in train_dataloaders.keys():
                    tot_ex = n_examples[t]
                    ex_per_sec = int(tot_ex / (time.time() - start_time))
                    tot_in = n_in_units[t]
                    in_per_sec = int(tot_in / (time.time() - start_time))
                    tot_l = n_loss_units[t]
                    l_per_sec = int(tot_l / (time.time() - start_time))
                    LOGGER.info(f'{t}: {tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar(f'perf/{t}_ex_per_s', ex_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_in_per_s', in_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_loss_per_s', l_per_sec,
                                         global_step)
                LOGGER.info('===============================================')

            if global_step % opts.valid_steps == 0:
                max_update_flag = False
                LOGGER.info(f'------Step {global_step}: start validation seen------')
                validate(model, val_dataloaders, setname='_seen', tem=model_config.cfp_temperature)
                LOGGER.info(f'------Step {global_step}: start validation unseen------')
                max_unseen_facc, max_update_flag=validate(model, val2_dataloaders, setname='_unseen',max_metrix=max_unseen_facc,\
                                                          tem=model_config.cfp_temperature)
                LOGGER.info(f'Best unseen facc: {max_unseen_facc}.')
                # validate(model, val2_dataloaders, setname='_unseen')
                if max_update_flag:
                    model_saver.save_latest(model, global_step, is_max=True)
                model_saver.save_latest(model, global_step)
        if global_step >= opts.num_train_steps:
            break
    if global_step % opts.valid_steps != 0:
        LOGGER.info(f'------Step {global_step}: start validation seen------')
        validate(model, val_dataloaders, setname='_seen', tem=model_config.cfp_temperature)
        LOGGER.info(f'------Step {global_step}: start validation unseen------')
        validate(model, val2_dataloaders, setname='_unseen', tem=model_config.cfp_temperature)
        # model_saver.save(model, global_step)   
        model_saver.save_latest(model, global_step) # only save the latest model


def validate(model, val_dataloaders, setname='', max_metrix=None, tem=None):
    model.eval()
    max_update_flag = False
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate val{setname} on {task} task")
        if task.startswith('mlm'):
            val_log = validate_mlm(model, loader)
        elif task.startswith('mrc'):
            val_log = validate_mrc(model, loader)
        elif task.startswith('sap'):
            val_log = validate_sap(model, loader)
        elif task.startswith('og'):
            val_log = validate_og(model, loader)
            if setname == '_unseen' and max_metrix is not None:
                if val_log['acc'] >= max_metrix:
                    max_metrix = val_log['acc']
                    max_update_flag = True
        elif task.startswith('cfp'):
            val_log = validate_cfp(model, loader, tem)
        else:
            raise ValueError(f'Undefined task {task}')
        val_log = {f'val{setname}_{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scalar_dict(
            {f'valid{setname}_{task}/{k}': v for k, v in val_log.items()}
        )
    model.train()
    if max_metrix is not None:
        return max_metrix, max_update_flag

@torch.no_grad()
def validate_mlm(model, val_loader):
    LOGGER.info("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='mlm', compute_loss=False)
        labels = batch['txt_labels']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
    val_loss = sum(all_gather(val_loss))
    n_correct = sum(all_gather(n_correct))
    n_word = sum(all_gather(n_word))
    tot_time = time.time()-st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log

def compute_accuracy_for_soft_targets(out, labels):
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]  # argmax
    n_correct = (outputs == labels).sum().item()
    return n_correct

@torch.no_grad()
def validate_mrc(model, val_loader):
    LOGGER.info("start running MRC validation...")
    val_loss = 0
    n_feat = 0
    st = time.time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        view_logits, view_targets, obj_logits, obj_targets = model(batch, task='mrc', compute_loss=False)
        view_logprobs = F.log_softmax(view_logits, dim=-1)
        obj_logprobs = F.log_softmax(obj_logits, dim=-1)
        loss = F.kl_div(view_logprobs, view_targets, reduction='sum') + \
               F.kl_div(obj_logprobs, obj_targets, reduction='sum')
        tot_score += compute_accuracy_for_soft_targets(view_logits, view_targets) + \
                     compute_accuracy_for_soft_targets(obj_logits, obj_targets)
        val_loss += loss.item()
        n_feat += batch['vp_view_mrc_masks'].sum().item() + batch['vp_obj_mrc_masks'].sum().item()
    val_loss = sum(all_gather(val_loss))
    tot_score = sum(all_gather(tot_score))
    n_feat = sum(all_gather(n_feat))
    tot_time = time.time()-st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log
    
@torch.no_grad()
def validate_sap(model, val_loader):
    LOGGER.info("start running SAP validation...")
    val_gloss, val_lloss, val_floss = 0, 0, 0
    n_gcorrect, n_lcorrect, n_fcorrect = 0, 0, 0
    n_data = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        global_logits, local_logits, fused_logits, global_act_labels, local_act_labels = \
            model(batch, task='sap', compute_loss=False)
        val_gloss += F.cross_entropy(global_logits, global_act_labels, reduction='sum').data.item()
        val_lloss += F.cross_entropy(local_logits, local_act_labels, reduction='sum').data.item()
        val_floss += F.cross_entropy(fused_logits, global_act_labels, reduction='sum').data.item()
        n_gcorrect += torch.sum(torch.argmax(global_logits, 1) == global_act_labels).item()
        n_lcorrect += torch.sum(torch.argmax(local_logits, 1) == local_act_labels).item()
        n_fcorrect += torch.sum(torch.argmax(fused_logits, 1) == global_act_labels).item()
        n_data += len(global_act_labels)

    n_data = sum(all_gather(n_data))
    val_gloss = sum(all_gather(val_gloss)) / n_data
    val_lloss = sum(all_gather(val_lloss)) / n_data
    val_floss = sum(all_gather(val_floss)) / n_data
    gacc = sum(all_gather(n_gcorrect)) / n_data
    lacc = sum(all_gather(n_lcorrect)) / n_data
    facc = sum(all_gather(n_fcorrect)) / n_data
    
    tot_time = time.time()-st
    val_log = {'gloss': val_gloss, 'lloss': val_lloss, 'floss': val_floss,
               'gacc': gacc, 'lacc': lacc, 'facc': facc,
               'tok_per_s': n_data/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"gacc: {gacc*100:.2f}, lacc: {lacc*100:.2f}, facc: {facc*100:.2f}")
    return val_log

@torch.no_grad()
def validate_cfp(model, val_loader, temperature):
    LOGGER.info("start running CFP validation...")
    val_gloss, val_lloss, val_floss = 0, 0, 0
    n_gcorrect, n_lcorrect, n_fcorrect = 0, 0, 0
    n_data = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        gmap_outputs, vp_outputs, fused_outputs, txt_outputs = \
            model(batch, task='cfp', compute_loss=False)

        target_sim = torch.arange(len(gmap_outputs)).cuda()
        
        gmap_txt_sim = ( gmap_outputs @ txt_outputs.T ) / temperature
        global_txt_losses = (F.cross_entropy(gmap_txt_sim, target_sim, reduction='sum') +\
                                F.cross_entropy(gmap_txt_sim.T, target_sim.T, reduction='sum')) / 2.0

        vp_txt_sim = ( vp_outputs @ txt_outputs.T ) / temperature
        vp_txt_losses = (F.cross_entropy(vp_txt_sim, target_sim, reduction='sum') +\
                                F.cross_entropy(vp_txt_sim.T, target_sim.T, reduction='sum')) / 2.0
        
        fused_txt_sim = ( fused_outputs @ txt_outputs.T) / temperature
        fused_txt_losses = (F.cross_entropy(fused_txt_sim, target_sim, reduction='sum') +\
                                F.cross_entropy(fused_txt_sim.T, target_sim.T, reduction='sum')) / 2.0
        

        val_gloss += global_txt_losses.item()
        val_lloss += vp_txt_losses.item()
        val_floss += fused_txt_losses.item()
        n_gcorrect += torch.sum(torch.argmax(gmap_txt_sim, 1) == target_sim).item()
        n_lcorrect += torch.sum(torch.argmax(vp_txt_sim, 1) == target_sim).item()
        n_fcorrect += torch.sum(torch.argmax(fused_txt_sim, 1) == target_sim).item()
        n_data += len(target_sim)

        del target_sim

    n_data = sum(all_gather(n_data))
    val_gloss = sum(all_gather(val_gloss)) / n_data
    val_lloss = sum(all_gather(val_lloss)) / n_data
    val_floss = sum(all_gather(val_floss)) / n_data
    gacc = sum(all_gather(n_gcorrect)) / n_data
    lacc = sum(all_gather(n_lcorrect)) / n_data
    facc = sum(all_gather(n_fcorrect)) / n_data
    
    tot_time = time.time()-st
    val_log = {'gloss': val_gloss, 'lloss': val_lloss, 'floss': val_floss,
               'gacc': gacc, 'lacc': lacc, 'facc': facc,
               'tok_per_s': n_data/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"gacc: {gacc*100:.2f}, lacc: {lacc*100:.2f}, facc: {facc*100:.2f}")
    return val_log

@torch.no_grad()
def validate_og(model, val_loader):
    LOGGER.info("start running Object Grounding validation...")
    val_loss = 0
    n_correct = 0
    n_data = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='og', compute_loss=False)
        labels = batch['obj_labels']
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_data += labels.numel()
    val_loss = sum(all_gather(val_loss))
    n_correct = sum(all_gather(n_correct))
    n_data = sum(all_gather(n_data))
    tot_time = time.time()-st
    val_loss /= n_data
    acc = n_correct / n_data
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_data/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log


def build_args():
    parser = load_parser()

    opts = parse_with_config(parser)
    postprocess_args(opts)

    # if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
    #     LOGGER.warning(
    #         "Output directory ({}) already exists and is not empty.".format(
    #             opts.output_dir
    #         )
    #     )

    return opts

if __name__ == '__main__':
    args = build_args()
    main(args)
