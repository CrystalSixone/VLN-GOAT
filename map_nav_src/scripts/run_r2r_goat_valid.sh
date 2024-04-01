name=goat_r2r_valid

DATA_ROOT=../datasets
train_alg=dagger
ft_dim=768
features=clip768
ngpus=1
seed=0

outdir=${DATA_ROOT}/R2R/
backdoor_dict_file=${DATA_ROOT}/R2R/navigator/goat_r2r/logs/backdoor/backdoor_update_features.tsv
frontdoor_dict_file=${DATA_ROOT}/R2R/navigator/goat_r2r/logs/frontdoor/frontdoor_update_features.tsv
resume_file=${DATA_ROOT}/R2R/navigator/goat_r2r/ckpts/best_val_unseen.pt

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta
      --name ${name}   
      --mode valid

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
      --iters 100000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2  

      --feat_dropout 0
      --dropout 0.0

      --do_back_txt
      --do_back_img
      --do_back_txt_type type_2
      --do_back_imgobj_type type_1
      --do_add_method door

      --do_front_txt
      --do_front_img
      --do_front_his

      --backdoor_dict_file ${backdoor_dict_file}
      --frontdoor_dict_file ${frontdoor_dict_file}

      --submit
      "

# train
CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag \
      --resume_file ${resume_file}