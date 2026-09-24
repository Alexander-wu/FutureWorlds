# 方法对应关系

| 论文模块 | 实现 | 核心约束 |
|---|---|---|
| Construct | `search.rollout_group_beam` | 4 组 × 2 条活动路径，每组返回 1 条；搜索惩罚与原始策略评分分开保存 |
| Maintain | `memory.make_prefix` | 场景 token、状态锚点、近期历史与动作组成因果前缀；新帧重建 KV cache |
| Compare & Learn | `memory.frame_logp`、`objective`、`rewards` | old/current/reference 在候选对应的相同历史上评分；GT 仅进入奖励计算 |
| Text conditioning | `text_conditioning.TextWorldModel` | 冻结 T5 特征通过零初始化残差注意力接入；非离散文本 token 拼接 |
| Evaluation | `decoding.beam_decode` | 每帧固定长度的视觉 token 搜索，选择模型分数最高的帧，不以 GT 选候选 |

记忆限制的是模型前缀长度；Python 历史字典仍保存完整生成历史，不等于整个程序存储空间恒定。

MemSPO 使用确定性搜索候选上的代理更新目标，不把搜索输出当成普通 on-policy GRPO 样本。原始模型 log-probability 也不是搜索行为分布的概率。

初次迁移修改包内 import 路径，当前版本另统一格式并补充说明注释。原函数名、默认值、计算顺序和张量操作均保留；已用 AST 核对可执行逻辑一致性（忽略文档字符串空白和 import 模块名）。核心 SHA-256 见 `provenance/core_sources.json`。

