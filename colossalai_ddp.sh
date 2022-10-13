CUBLAS_WORKSPACE_CONFIG=:4096:8  

CUDA_VISIBLE_DEVICES_set_n_least_memory_usage() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
        | tail -n +2 \
        | nl -v 0 \
        | tee /dev/tty \
        | sort -g -k 2 \
        | awk '{print $1}' \
        | head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

CUDA_VISIBLE_DEVICES_set_n_least_memory_usage 4

torchrun --nproc_per_node=4 --master_port 29505  main.py \
    --dataset_config configs/pretrain_test_flickr_only.json \
    --batch_size 4 \
    --lr_backbone 2e-5 \
    --text_encoder_lr 2e-5 \
    --lr 1e-4 \
    --num_queries 200 \
    --max_decoding_step 256 \
    --do_caption \
    --no_detection \
    --unitab_pretrain \
    --pretrain_seqcrop mixed \
    --ema \
    --output-dir weights/$exp_id \
    --distributed \
    --load /data2/users/lcmql/unitab/weights/pretrained_checkpoint.pth \
    --from_colossalai \
