# Hugging Face 发布方案

## 发布物如何分工

- **GitHub：** 核心代码、环境依赖、数据准备、训练/推理/评测脚本和复现说明。
- **Hugging Face Model：** 自己训练的模型、配套视觉编解码器、文本适配器、动作统计与模型卡。
- **Hugging Face Dataset（可选）：** 后续可放有权限再分发的预测结果、逐样本指标及示例；原始 RT-1/Bridge/RoboCasa 数据当前不打包。
- **Space（可选）：** 待完整推理入口稳定后制作演示，不能把“已托管权重”等同于“已支持在线推理”。

建议先使用一个 `<账号>/FutureWorlds` 模型仓库，按数据集和实验版本分目录。该名称是规划，不代表仓库已经创建。首次验证使用私有仓库，公开时再填写正式论文链接与最终许可证。

## 已核验的文件

2026-09-24 通过 CPU 读取现有服务器文件并重新计算 SHA-256，核验 **15 份预测模型、3 份视觉编解码器**。其中 RT-1/RoboCasa 每份预测模型均包含文本适配器。

| 数据集 | 主表版 | SFT | 匹配 MemSPO | 匹配 GRPO | 普通束搜索 |
|---|---|---|---|---|---|
| RT-1 | 500 步，R0 | 50k | 200 步，R0 | 200 步 | 200 步 |
| BridgeV2 | 200 步，E4/R1 | 50k | 200 步，E2/R0 | 200 步，E1/R0 | 200 步 |
| RoboCasa | 400 步，R0 | 50k | 200 步，R0 | 200 步 | 200 步 |

更精确的大小、文件名和哈希见根目录 `weights_manifest.json`。单份预测骨干约 508.42 MB，文本适配器约 9.48 MB，单份编解码器约 517.85 MB。**三个主表模型加配套编解码器约 3.10 GB；全部上述模型与消融约 9.27 GB**，均为十进制大小，未计外部 T5。此处是文件大小，不是 GPU 显存。

## 每个数据集的目录

```text
rt1/                         # bridge/、robocasa/ 同结构
  README.md                  # 数据集对应模型卡
  futureworlds_config.json    # 文本、动作、记忆、解码与版本映射
  training_recipe.json        # 从原配置提取的训练超参数
  models/
    main/                    # 论文主表权重
    sft/
    memspo200/
    grpo200/
    ordinary200/
  codec/
  preprocessing/             # 配套 action_ranges
bridge/r1_reward_config.json  # 主表 E4 的奖励尺度/权重单独保存
code/                        # 可安装的核心包
licenses/                    # 上游许可证
weights_manifest.json        # 组件级来源、哈希和大小
manifest.json                # 最终发布目录每个文件的哈希
```

模型以原始 FP32 `safetensors` 保存，不为缩小体积改成 FP16，不把不同训练版本的适配器混合。各组共享本数据集的编解码器。完整训练恢复所需 optimizer/RNG 状态留在原服务器，不放入默认推理包；以后若提供续训包，应另设目录说明版本与体积。

T5-base 固定 `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1` 版本，优先指向上游，不重复上传。官方文本模型和外部 baseline 也不直接冒充 FutureWorlds 权重发布。

## 工具与执行顺序

1. `scripts/prepare_hf_release.py` 读取内部文件白名单；默认只检查路径/大小，显式 `--materialize` 才复制并校验 SHA-256。内部计划含服务器路径，只留工作目录，**不上传**。
2. `scripts/upload_hf_release.py --folder ... --repo-id ...` 默认只验证清单、哈希及未列入清单的文件，不访问 HF。只有加 `--upload` 才使用已有 HF 登录上传到私有模型仓库。
3. 上传后记录实际 Hub commit SHA；从干净目录重新下载，并对照 `manifest.json` 验证，再运行固定样本推理。
4. 完成完整入口与原实现对齐后再公开。公开是独立操作，当前脚本不会自动改变仓库可见性。

不要把访问 token 放到代码、配置或上传命令参数中；使用已有 HF 登录。不得把整个实验 `runs/` 目录作为上传源。

## 当前验证边界

已完成：原文件哈希；三个数据集配套关系；核心源码 import 迁移检查；10 项 CPU 检查，其中包括纯视觉/带文本预测器保存加载的逐值一致性、缺少适配器报错和文件篡改检测。

已补齐 RGB/动作/T5 流水线、统一训练、评测及本地一键入口；三套真实权重首帧与原入口对齐。仍待完成新入口全量 GPU 回归、干净安装验证、正式公开仓库与资产下载配置、原始新增代码的发布许可证与引用元信息。目前不承诺 `AutoModel.from_pretrained` 或通用文生视频 pipeline 能直接运行整个系统。

## 官方文档与上游来源

- [HF 上传文件与目录](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [HF 模型卡](https://huggingface.co/docs/hub/model-cards)
- [HF 许可证元信息](https://huggingface.co/docs/hub/repositories-licenses)
- [RT-1 base model：模型卡标注 MIT](https://huggingface.co/thuml/rt1-world-model-multi-step-base)
- [RT-1 tokenizer：模型卡标注 MIT](https://huggingface.co/thuml/rt1-compressive-tokenizer)
- [T5-base：模型卡标注 Apache-2.0](https://huggingface.co/google-t5/t5-base)

上游声明按组件保留，原始数据集的条款不能用模型代码许可证代替。
