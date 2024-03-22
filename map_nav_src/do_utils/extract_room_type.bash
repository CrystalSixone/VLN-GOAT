CUDA_VISIBLE_DEVICES=0 python do_utils/extract_room_type.py \
    --connectivity_dir connectivity \
    --scan_dir ../vln/v1/v1/scans \
    --output_file ../datasets/R2R/features/pano_roomtypes.tsv \
    --batch_size 2 \
    --num_workers 4 \
    --model_path ../datasets/pretrained/blip-vqa-base