import torch
from transformers import AutoModel


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    elif args.tokenizer == 'roberta':
        cfg_name = 'roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(args, config=None):
    from transformers import PretrainedConfig
    from models.vilmodel_GOAT import GlocalTextPathNavCMT
    
    model_name_or_path = args.bert_ckpt_file
    new_ckpt_weights = {}
    if model_name_or_path == 'bert':
        tmp = AutoModel.from_pretrained('bert-base-uncased')
        for param_name, param in tmp.named_parameters():
            # new_ckpt_weights[param_name] = param
            if 'bert.encoder.layer' in param_name:
                param_name = param_name.replace('bert.encoder.layer', 'bert.lang_encoder.layer')
                new_ckpt_weights[param_name] = param
            else:
                new_ckpt_weights[param_name] = param
        del tmp
    elif model_name_or_path == 'meter':
        try:
            tmp = torch.load('../datasets/pretrained/METER/meter_clip16_224_roberta_pretrain.ckpt')
        except Exception:
            tmp = torch.load('datasets/pretrained/METER/meter_clip16_224_roberta_pretrain.ckpt')
        tmp = tmp['state_dict']
        for param_name, param in tmp.items():
            if 'text_transformer.embeddings' in param_name:
                param_name = param_name.replace('text_transformer.', 'bert.')
                new_ckpt_weights[param_name] = param
            elif 'text_transformer.encoder' in param_name:
                param_name = param_name.replace('text_transformer.encoder', 'bert.lang_encoder')
                new_ckpt_weights[param_name] = param
            elif 'cross_modal_image_layers' in param_name:
                param_name1 = param_name.replace('cross_modal_image_layers', 'bert.local_encoder.encoder.crossattention')
                param_name2 = param_name.replace('cross_modal_image_layers', 'bert.global_encoder.encoder.crossattention')
                new_ckpt_weights[param_name1] = new_ckpt_weights[param_name2] = param
            else:
                new_ckpt_weights[param_name] = param
        del tmp
    elif model_name_or_path is not None:
        # pretrain model (path)
        model_name = None
        ckpt_weights = torch.load(model_name_or_path)
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                k = k[7:]    
            if k.startswith('vln_bert'):
                k = 'bert' + k[8:]
            if '_head' in k or 'sap_fuse' in k:
                new_ckpt_weights['bert.' + k] = v
            elif 'tim' in k or 'temperature' in k:
                if 'self_encoder' not in k:
                    new_ckpt_weights['bert.' + k] = v
                else:
                    new_ckpt_weights[k] = v
            else:
                new_ckpt_weights[k] = v
            
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    elif args.tokenizer == 'roberta':
        # cfg_name = 'roberta-base'
        cfg_name = 'datasets/pretrained/roberta' # the local model.
    else:
        cfg_name = 'bert-base-uncased'
    try:
        vis_config = PretrainedConfig.from_pretrained(cfg_name)
    except Exception:
        cfg_name = '../' + cfg_name
        vis_config = PretrainedConfig.from_pretrained(cfg_name)

    if args.tokenizer == 'xlm':
        vis_config.type_vocab_size = 2
    elif args.tokenizer == 'roberta':
        assert vis_config.type_vocab_size == 1
    
    vis_config.dataset = args.dataset
    vis_config.mode = args.mode
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.obj_name_vocab_size = 45
    vis_config.num_l_layers = args.num_l_layers 
    vis_config.num_pano_layers = args.num_pano_layers 
    vis_config.num_x_layers = args.num_x_layers 
    vis_config.graph_sprels = args.graph_sprels 
    vis_config.glocal_fuse = args.fusion == 'dynamic'

    vis_config.fix_lang_embedding = args.fix_lang_embedding 
    vis_config.fix_pano_embedding = args.fix_pano_embedding 
    vis_config.fix_local_branch = args.fix_local_branch

    vis_config.update_lang_bert = not args.fix_lang_embedding 
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    
    vis_config.max_instr_len = args.max_instr_len
    vis_config.feat_dropout = args.feat_dropout
    vis_config.adaptive_pano_fusion = args.adaptive_pano_fusion

    ''' Causal Learning
    '''
    vis_config.do_back_img = args.do_back_img
    vis_config.do_back_txt = args.do_back_txt
    vis_config.do_front_img = args.do_front_img
    vis_config.do_front_his = args.do_front_his
    vis_config.do_front_txt = args.do_front_txt
    vis_config.cfp_temperature = args.cfp_temperature
    vis_config.do_back_txt_type = args.do_back_txt_type
    vis_config.do_back_img_type = args.do_back_img_type
    vis_config.do_add_method = args.do_add_method
    
    # METER param
    # Text Setting
    vis_config.type_vocab_size = 1
    vis_config.max_position_embeddings = 514
    vis_config.vocab_size = 50265 
    vis_config.mlm_prob = 0.15
    vis_config.draw_false_text = 0
    
    # Transformer Setting
    vis_config.num_top_layer = args.num_x_layers # cross-attention
    vis_config.input_image_embed_size = 768
    vis_config.input_text_embed_size = 768
    vis_config.hidden_size = 768
    vis_config.num_attention_heads = 12
    vis_config.num_hidden_layers = args.num_l_layers # language BERT
    vis_config.mlp_ratio = 4
    vis_config.hidden_dropout_prob = args.dropout 
    vis_config.attention_probs_dropout_prob = 0.1
    vis_config.intermediate_size = 3072 # 768 * mlp_ratio

    vis_config.name = 'R2R'
    if args.dataset == 'reverie':
        vis_config.name = 'REVERIE'
        vis_config.use_obj_name = True
        
    elif args.dataset == 'soon':
        vis_config.name ='SOON'
        vis_config.use_obj_name = False
        
    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
        
    return visual_model
