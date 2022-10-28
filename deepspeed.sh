CUBLAS_WORKSPACE_CONFIG=:4096:8


torchrun --nproc_per_node=2 --master_port 10101  main.py \
    --dataset_config configs/pretrain_test_flickr_only.json \
    --deepspeed_config deepspeed_config.json \
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
    --output-dir output/$exp_id \
    --distributed \
    --load /path/to/unitab/weights/pretrained_checkpoint.pth \
    --from_deepspeed \
    --mem_cap 16 \