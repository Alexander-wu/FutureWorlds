# RGB 推理与权重选择

```python
from futureworlds.pipeline import WorldPipeline

pipe = WorldPipeline(
    bundle="/data/futureworlds_weights",
    dataset="rt1", variant="main", device="cuda:0",
    text_model="/data/t5-base-pinned-snapshot",
)
result = pipe.predict(
    observations=initial_rgb,  # uint8 [2, 256, 320, 3]
    actions=controls,          # [H+2, 13]; RoboCasa 为 [H+2, 52]
    instruction="move the can near the other can",
    horizon=32, beam_width=4, memory="full",
)
# result['prediction']: float32 [32,256,320,3], [0,1]
# result['tokens']: 原始预测 token；result['trace']: 各帧历史保留记录
```

推理入口拒绝传入多于两帧观测。生成期间不刷新真实未来帧；评测指标在浮点解码图像上计算，不能先转 uint8 再比较。

可选版本：`main`、`sft`、`memspo200`、`grpo200`、`ordinary200`。先校验整个 bundle，再严格匹配预测器、视觉编解码器、动作范围和文本适配器。RT-1 与 RoboCasa 使用固定 T5 快照和训练好的交叉注意力适配器；Bridge 本配方不接文本。

RoboCasa 的 52 维是四个有序 13-token 控制块，不能求平均成 13 维。动作值的量化规则由训练时的 `action_ranges.pt` 决定。数据准备及因果索引见 `docs/DATA.md`。

只需核心预测器时仍可使用 `checkpoints.load_world_model`；该低层接口接收已经编码的 token，不自行转换 RGB / 原始动作。
