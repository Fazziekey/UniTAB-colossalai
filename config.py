from colossalai.amp import AMP_TYPE

BATCH_SIZE = 2
NUM_EPOCHS = 1

fp16=dict(mode=AMP_TYPE.TORCH)

torch_ddp=dict(find_unused_parameters=True)
