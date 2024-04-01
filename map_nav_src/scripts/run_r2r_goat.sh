name=goat_r2r
DATA_ROOT=../datasets

train_alg=dagger
ft_dim=768
features=clip768
ngpus=1
seed=0

outdir=${DATA_ROOT}/R2R/
aug_file=${DATA_ROOT}/R2R/annotations/prevalent_aug_train_enc.json
speaker_file=${DATA_ROOT}/R2R/speaker/transpeaker_r2r/state_dict/best_both_bleu.pt
r2r_pretrain_file=${DATA_ROOT}/R2R/pretrain/goat_r2r_pretrain/ckpts/model_step_best_42000.pt

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta
      --name ${name}   
      --mode train

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

      --batch_size 12
      --lr 2e-5
      --iters 150000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2 

      --feat_dropout 0.5
      --dropout 0.1

      --use_transpeaker
      --speaker ${speaker_file}
      --accumulateGrad
      --aug ${aug_file}

      --do_back_txt
      --do_back_img
      --do_back_txt_type type_2
      --do_back_imgobj_type type_1
      --do_add_method door
      --z_instr_update

      --do_front_txt
      --do_front_img
      --do_front_his
      "

# train
CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
      --bert_ckpt_file ${r2r_pretrain_file}