from colossalai.amp import AMP_TYPE

BATCH_SIZE = 2
NUM_EPOCHS = 40

CONFIG = dict(fp16=dict(mode=AMP_TYPE.TORCH))