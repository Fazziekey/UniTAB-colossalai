CUBLAS_WORKSPACE_CONFIG=:4096:8  

python main.py \
    --dataset_config configs/pretrain.json \
    --batch_size 2 \
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
    --load https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth \
    --host 127.0.0.1 \
    --port 29500 \
    --world_size 1 \
    --rank 0 \
    --backend nccl \

# torchrun --nproc_per_node=2 --master_port 29505  main.py \
#     --dataset_config configs/pretrain.json \
#     --batch_size 2 \
#     --lr_backbone 2e-5 \
#     --text_encoder_lr 2e-5 \
#     --lr 1e-4 \
#     --num_queries 200 \
#     --max_decoding_step 256 \
#     --do_caption \
#     --no_detection \
#     --unitab_pretrain \
#     --pretrain_seqcrop mixed \
#     --ema \
#     --output-dir weights/$exp_id \
#     --load https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth 