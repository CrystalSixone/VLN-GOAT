import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--dataset', type=str, default='r2r', choices=['r2r', 'rxr'])
    parser.add_argument('--output_dir', type=str, default='', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--tokenizer', choices=['bert', 'xlm','roberta'], default='bert')

    parser.add_argument('--act_visited_nodes', action='store_true', default=False)
    parser.add_argument('--fusion', default='dynamic',choices=['global', 'local', 'avg', 'dynamic'])
    parser.add_argument('--expl_sample', action='store_true', default=False)
    parser.add_argument('--expl_max_ratio', type=float, default=0.6)
    parser.add_argument('--expert_policy', default='spl', choices=['spl', 'ndtw'])

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")
    
    # General
    parser.add_argument('--iters', type=int, default=200000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--eval_first', action='store_true', default=False)
    parser.add_argument("--save_optimizer", action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=200)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')
    
    # Load the model from
    parser.add_argument("--resume_file", default=None, help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', type=str, default=None)

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)

    parser.add_argument("--features", type=str, default='clip768')

    parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
    parser.add_argument('--fix_pano_embedding', action='store_true', default=False)
    parser.add_argument('--fix_local_branch', action='store_true', default=False)

    parser.add_argument('--num_l_layers', type=int, default=6)
    parser.add_argument('--num_pano_layers', type=int, default=2)
    parser.add_argument('--num_x_layers', type=int, default=3)

    parser.add_argument('--enc_full_graph', default=True, action='store_true')
    parser.add_argument('--graph_sprels', action='store_true', default=True)

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--feat_dropout', type=float, default=0.4)
    parser.add_argument('--views', type=int, default=36)

    # Submision configuration
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=False)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='adamW',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )   
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--obj_feat_size', type=int, default=768)
    parser.add_argument("--cat_file",type=str,default='../datasets/R2R/annotations/category_mapping.tsv')
    parser.add_argument("--aug_times",type=int,default=1)
    
    # # A2C
    parser.add_argument("--gamma", default=0.9, type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total", 
        type=str, help='batch or total'
    )
    parser.add_argument('--train_alg', 
        choices=['imitation', 'dagger'], 
        default='dagger'
    )

    # Speaker
    parser.add_argument("--speaker", default=None)
    parser.add_argument('--maxDecode', type=int, default=120, help="max input instruction")
    parser.add_argument('--use_transpeaker',default=False,action='store_true')
    parser.add_argument('--use_drop',action='store_true',default=False)
    parser.add_argument('--speaker_dropout', type=float, default=0.2) # For speaker!
    parser.add_argument('--wemb',type=int,default=256)
    parser.add_argument('--hDim', dest="h_dim", type=int, default=512)
    parser.add_argument('--proj_hidden',default=1024,type=int) 
    parser.add_argument('--aemb', type=int, default=64)
    parser.add_argument("--subout", dest="sub_out", type=str, default="tanh")  # tanh, max
    parser.add_argument("--accumulateGrad", dest='accumulate_grad', action='store_const', default=False, const=True)
    parser.add_argument('--featdropout', type=float, default=0.3)
    parser.add_argument("--loadOptim",action="store_const", default=False, const=True)
    parser.add_argument("--speaker_angle_size",type=int,default=128)
    parser.add_argument('--speaker_layer_num',default=3,type=int) 
    parser.add_argument('--speaker_head_num',default=4,type=int)

    # Features
    parser.add_argument("--use_aug_env",action="store_true",default=False) # whether use additional augmented features
    parser.add_argument("--env_edit",action="store_true",default=False) # use env_Edit?

    # Adaptive Pano Fusion
    parser.add_argument("--adaptive_pano_fusion",action='store_true',default=True) 

    # Causal Learning
    parser.add_argument("--do_back_img",action='store_true',default=False) # do_back_img
    parser.add_argument("--do_back_txt",action='store_true',default=False) # do_back_txt
    parser.add_argument("--do_front_img",action='store_true',default=False) # do_front_img
    parser.add_argument("--do_front_his",action='store_true',default=False) # do_front_history
    parser.add_argument("--do_front_txt",action='store_true',default=False) # do_front_txt
    parser.add_argument("--cfp_temperature",type=float,default=1.0)
    parser.add_argument("--z_instr_update",action='store_true',default=False) # whether update the dictionary of back_txt
    parser.add_argument("--backdoor_dict_file",type=str,default='') # the saving path of the updated dictionary of back_txt
    parser.add_argument("--do_back_txt_type",type=str,default='type_2') # type-1: p_z; type_2: attention
    parser.add_argument("--do_back_img_type",type=str,default='type_1') 
    parser.add_argument("--do_add_method",type=str,default='door') # door; add
    parser.add_argument("--update_iter",type=int,default=3000)

    parser.add_argument("--front_n_clusters", type=int, default=24) # The number of KMeans clusters for front-door
    parser.add_argument("--frontdoor_dict_file", type=str, default='') # the saving path of front-door dictionaries

    # others
    parser.add_argument("--mode",type=str,required=True) # train; valid; extract_cfp_features
    parser.add_argument("--name",type=str,default='debug')
    parser.add_argument("--for_debug",action='store_true',default=False) 
    parser.add_argument("--use_lr_sch",action='store_true',default=False)
    parser.add_argument("--lr_sch",type=str,default='polynomial') # constant\constant_with_warmup\linear\polynomial\cosine\cosine_with_restarts

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir
    
    # set up features
    ft_file_map = {
        'clip768': 'CLIP-ViT-B-16-views.hdf5',
    }
    aug_ft_file_map = {
        'env_edit': 'CLIP-ViT-B-16-views-st-samefilter.hdf5'
    }
    args.img_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.features])
    args.img_type = 'hdf5'
    args.feature_size = args.image_feat_size = 768

    args.aug_img_ft_file_envedit = os.path.join(ROOTDIR, 'EnvEdit', 'hamt_features', aug_ft_file_map['env_edit'])

    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')
    args.speaker_train_vocab = os.path.join(ROOTDIR, 'R2R', 'features', 'r2r_speaker_train_vocab.txt')

    # do intervention
    args.img_zdict_file = os.path.join(ROOTDIR, 'R2R', 'features', 'image_z_dict_clip_50.tsv')
    args.img_zdict_size = 50
    args.instr_zdict_file = os.path.join(ROOTDIR, 'R2R', 'features', 'r2r_z_instr_dict.tsv')
    args.instr_zdict_size = 81
    args.rxr_instr_zdict_roberta_file = os.path.join(ROOTDIR, 'R2R', 'features', 'rxr_z_instr_dict.tsv')

    # For cfp intervention (frontdoor)
    args.front_feat_file = os.path.join(ROOTDIR, 'R2R', 'features', 'r2r_cfp_features.tsv')
    args.rxr_front_feat_file = os.path.join(ROOTDIR, 'R2R', 'features', 'rxr_cfp_features.tsv')

    args.scanvp_cands_file = os.path.join(ROOTDIR, 'R2R', 'annotations','scanvp_candview_relangles.json')

    # Build paths
    if 'train' in args.mode:
        args.output_dir = os.path.join(args.output_dir,'navigator',args.name)
    else:
        args.output_dir = os.path.join(args.output_dir,'test',args.name)

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')
    
    # Intervention
    args.z_back_log_dir = os.path.join(args.output_dir, 'logs', 'backdoor')
    args.z_front_log_dir = os.path.join(args.output_dir, 'logs', 'frontdoor')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)
    
    os.makedirs(args.z_back_log_dir, exist_ok=True)
    os.makedirs(args.z_front_log_dir, exist_ok=True)

    return args

