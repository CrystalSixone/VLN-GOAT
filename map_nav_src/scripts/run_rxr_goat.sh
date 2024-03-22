name=goat_rxr
DATA_ROOT=../datasets

train_alg=dagger
ft_dim=768
features=clip768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

outdir=${DATA_ROOT}/RxR/
aug_file=${DATA_ROOT}/R2R/annotations/prevalent_aug_train_enc.json
rxr_pretrain_file=${DATA_ROOT}/RxR/pretrain/goat_rxr_pretrain/ckpts/model_step_best_262500.pt

flag="--root_dir ${DATA_ROOT}
      --dataset rxr
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta   
      --name ${name}   
      --mode train

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy ndtw
      --train_alg ${train_alg}
      
      --num_l_layers 6
      --num_x_layers 3
      --num_pano_layers 2
      
      --max_action_len 28
      --max_instr_len 250

      --batch_size 5
      --lr 2e-5
      --iters 100000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.5
      --dropout 0.1

      --do_back_txt
      --do_back_img
      --do_back_txt_type type_2
      --do_back_imgobj_type type_1
      --do_add_method door
      --z_instr_update

      --do_front_img
      --do_front_his
      --do_front_txt
      "

# train
CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
      --bert_ckpt_file ${rxr_pretrain_file}