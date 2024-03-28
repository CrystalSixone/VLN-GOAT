name=reverie_goat_cfp
DATA_ROOT=../datasets

train_alg=dagger
features=clip768
ft_dim=768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

outdir=${DATA_ROOT}/REVERIE/
augdir=${DATA_ROOT}/REVERIE/annotations/REVERIE_aug_roberta_enc.json
reverie_pretrain_file=${DATA_ROOT}/REVERIE/pretrain/reverie_goat_pretrain/ckpts/model_step_best.pt

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta
      --mode extract_cfp_features
      --name ${name}

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 6
      --num_x_layers 3
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 80
      --max_objects 20

      --batch_size 12

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2
      "

# train
CUDA_VISIBLE_DEVICES='0' python -u reverie/main_nav_obj.py $flag  \
      --bert_ckpt_file ${reverie_pretrain_file}