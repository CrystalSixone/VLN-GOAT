name=goat_reverie_pretrain
DATA_ROOT=../datasets/REVERIE/
NODE_RANK=0
NUM_GPUS=1

# train
CUDA_VISIBLE_DEVICES='1' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port 8889 \
    train_reverie_goat.py --world_size ${NUM_GPUS} \
    --name ${name} \
    --vlnbert cmt \
    --model_config config/reverie_GOAT_model_config.json \
    --config config/reverie_GOAT_pretrain.json \
    --root_dir ${DATA_ROOT} \
    --cuda_first_device 1

