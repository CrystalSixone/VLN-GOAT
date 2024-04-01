name=goat_reverie
DATA_ROOT=../datasets

train_alg=dagger
features=clip768
ft_dim=768
obj_features=vitbase
obj_ft_dim=768
ngpus=1
seed=0

outdir=${DATA_ROOT}/REVERIE/
aug_file=${DATA_ROOT}/REVERIE/annotations/REVERIE_aug_train_enc.json
speaker_file=${DATA_ROOT}/REVERIE/speaker/transpeaker_reverie/ckpts/best_both_bleu.pt
reverie_pretrain_file=${DATA_ROOT}/REVERIE/pretrain/goat_reverie_pretrain/ckpts/model_step_best.pt

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta
      --mode train
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
      --lr 2e-5
      --iters 100000
      --log_every 1000
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2

      --feat_dropout 0.6
      --dropout 0.1

      --use_transpeaker
      --accumulateGrad
      --aug ${aug_file}
      --speaker ${speaker_file}

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
CUDA_VISIBLE_DEVICES='0' python -u reverie/main_nav_obj.py $flag  \
      --bert_ckpt_file ${reverie_pretrain_file}