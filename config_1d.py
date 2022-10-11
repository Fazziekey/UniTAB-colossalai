from colossalai.amp import AMP_TYPE

BATCH_SIZE = 2
NUM_EPOCHS = 1

TENSOR_PARALLEL = 2

torch_ddp=dict(find_unused_parameters=True)

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

parallel = dict(
    pipeline=1,
    tensor=dict(size=TENSOR_PARALLEL, mode='1d'),
)
