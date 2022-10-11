CUBLAS_WORKSPACE_CONFIG=:4096:8  
# export LOCAL_RANK = 0
torchrun --nproc_per_node=4 --master_port 29505  main.py \
    --dataset_config configs/pretrain_test_flickr_only.json \
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
    --distributed \
    --load ./weights/pretrained_checkpoint.pth \
    --from_colossalai \