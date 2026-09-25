# 数据与清单

本工程无损转换**原实验已确定的窗口清单**，避免重新抽样造成表格不可比。原始数据不在源码 ZIP 中，不能仅凭 case ID 重建图像。应将许可允许分发的清单 / 处理脚本与模型发布一并提供；当前公开下载链路尚未配置。

## 统一格式

`manifest.json`：

```json
{
  "schema_version": 1,
  "dataset": "bridge",
  "cases": [{
    "id": "example",
    "episode_id": "episode_identifier",
    "split": "train",
    "file": "000000.npz",
    "sha256": "ACTUAL_FILE_SHA256",
    "frames": 34
  }]
}
```

每个 NPZ 使用 `allow_pickle=False`：

- `images`：uint8 RGB `[T,256,320,3]`，包含两帧初始观测。
- `actions`：float32 `[T,13]`，RoboCasa 为 `[T,52]`。`actions[t]` 对齐帧 t 到 t+1 的控制。
- `instruction`：Unicode 标量，不允许把 gold diagnosis / 未来图像写入条件。

训练清单必须标为 `split: train`，并与评测 episode ID 不重合。RoboCasa 的原始任务标签用于分任务采样。ID、排序、动作和图像均不通过随机操作补齐。

## 转换原始清单

```bash
futureworlds prepare --dataset bridge --source /data/dev_manifest.jsonl \
  --ids configs/cohorts/bridge_128.json --output /data/prepared/bridge/evaluation
```

- RT-1：原始 `window_id` / TFRecord 偏移与长度清单，由 `data_readers/rt1.py` 解码，无需 TensorFlow 运行时。
- Bridge：原清单 `output` 名称确定 ID，`path` 指向带 `image` / `action` 的 NPZ；校验源文件 SHA。
- RoboCasa：原始 PNG blob / offsets / arrays 清单，校验源哈希，保留时间顺序与四个动作子步。
- `--path-map mapping.json` 映射清单内旧前缀到本机数据根；不会改动原清单。

统一转换只接受已支持的真实来源格式，不把任意相似数据默认为论文数据。输出目录已存在时拒绝覆盖；转换意外中断可检查并删除自己生成的不完整目录后重试。

`configs/cohorts/` 内是三个数据集各 128 个有序评测 ID；`configs/schedules/` 是后训练实际顺序。它们是实验元数据，不是数据本体。
