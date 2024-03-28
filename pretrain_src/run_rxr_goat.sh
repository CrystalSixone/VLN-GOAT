name=goat_rxr_pretrain
DATA_ROOT=../datasets/RxR/
NODE_RANK=0
NUM_GPUS=1

# train
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port 8882 \
    train_r2r_goat.py --world_size ${NUM_GPUS} \
    --name ${name} \
    --vlnbert cmt \
    --model_config config/r2r_GOAT_model_config.json \
    --config config/rxr_GOAT_pretrain.json \
    --root_dir $DATA_ROOT \
    --cuda_first_device 0
