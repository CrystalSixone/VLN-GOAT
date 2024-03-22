name=goat_rxr_cfp
DATA_ROOT=../datasets

train_alg=dagger
features=clip768
ft_dim=768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

outdir=${DATA_ROOT}/RxR/
augdir=${DATA_ROOT}/RxR/annotations/RxR_aug_roberta_enc.json
rxr_pretrain_file=${DATA_ROOT}/RxR/pretrain/goat_rxr_pretrain/ckpts/model_step_best.pt

flag="--root_dir ${DATA_ROOT}
      --dataset rxr
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta
      --name ${name}   
      --mode extract_cfp_features

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 6
      --num_x_layers 3
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 8

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   
      "

# train
CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
      --bert_ckpt_file ${rxr_pretrain_file}