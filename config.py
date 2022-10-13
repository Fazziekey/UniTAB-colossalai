from colossalai.amp import AMP_TYPE
from colossalai.zero.shard_utils import TensorShardStrategy
BATCH_SIZE = 4
NUM_EPOCHS = 1

fp16 = dict(
    mode=AMP_TYPE.TORCH
)


torch_ddp=dict(find_unused_parameters=True,broadcast_buffers=False)

# zero = dict(
#     model_config=dict(
#         tensor_placement_policy='cpu',
#         shard_strategy=TensorShardStrategy(),
#         reuse_fp16_shard=True
#     ),
#     optimizer_config=dict()
# )