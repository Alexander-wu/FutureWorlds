# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from .batch import TensorBatch as DataProto


def plot_img(tensor, name):
    """Save the optional debug strip without relying on an external trainer helper."""
    from pathlib import Path
    from PIL import Image

    pixels = tensor.detach().float().cpu().squeeze(0).clamp(0, 1)
    array = (pixels.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    path = Path(str(name) + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def msp_reward_fn(
    self,
    batch: DataProto,
    pixels,
    return_reward_tensor=True,
    save_pred=False,
    pixels_before_repeat=None,
):
    if pixels_before_repeat is None:
        pixels_before_repeat = pixels

    batch_size = batch.batch["responses"].shape[0]
    segment_length = self.config.data.video.segment_length
    tokens_per_frame = self.config.processor.tokens_per_frame
    action_dim = self.config.processor.action_dim

    output_tokens = batch.batch["responses"].reshape(
        batch_size, segment_length - 1, tokens_per_frame + action_dim
    )
    output_tokens = output_tokens[:, :, :tokens_per_frame]
    output_tokens = output_tokens.clamp(
        0, self.config.processor.visual_token_num - 1
    ).long()

    ctx_tokens = batch.batch["ctx_tokens"]
    output_tokens = output_tokens.reshape(
        batch_size, segment_length - 1, tokens_per_frame
    )

    detokenize_output = self.tokenizer_wg.detokenize(
        DataProto.from_single_dict({"tokens": output_tokens, "ctx_tokens": ctx_tokens}),
        # DataProto.from_single_dict({"real": pixels_before_repeat[:, 2:]}, meta_info={'lpips': True}),
        DataProto.from_single_dict(
            {"dummy": torch.zeros((batch_size, 1))},
            meta_info={"lpips": True, "recon": self.config.trainer.reward_fn},
        ),
    )

    if "recon_loss" in detokenize_output.batch.keys():
        recon_loss = detokenize_output.batch["recon_loss"]
    else:
        pred = detokenize_output.batch["pixels"].clamp(0.0, 1.0)[:, 1:]
        real = pixels[:, 2:]
        if self.config.trainer.reward_fn == "mse":
            recon_loss = torch.mean((real - pred) ** 2, dim=(2, 3, 4))
        elif self.config.trainer.reward_fn == "mae":
            recon_loss = torch.mean(torch.abs(real - pred), dim=(2, 3, 4))
        else:
            raise NotImplementedError(
                f"Unsupported reward function: {self.config.trainer.reward_fn}"
            )

    if "perceptual_loss" in detokenize_output.batch.keys():
        perceptual_loss = detokenize_output.batch["perceptual_loss"]
    else:
        perceptual_loss = self.tokenizer_wg.perceptual_loss(
            DataProto.from_single_dict({"real": real, "pred": pred})
        )
        perceptual_loss = perceptual_loss.batch["perceptual_loss"]

    if self.config.trainer.msp_reward_aggregate == "mean":
        loss = (recon_loss + perceptual_loss).mean(-1)
    elif self.config.trainer.msp_reward_aggregate == "last":
        loss = (recon_loss + perceptual_loss)[:, -1]
    elif self.config.trainer.msp_reward_aggregate == "discount":
        discount = 0.9
        weight = discount ** torch.arange(recon_loss.shape[1], device=recon_loss.device)
        loss = (recon_loss + perceptual_loss) * weight.unsqueeze(0)
        loss = loss.sum(-1) / weight.sum()
    if not return_reward_tensor:
        return loss

    print("loss", loss.mean().item(), flush=True)
    print("recon_loss", recon_loss.mean().item(), flush=True)
    print("perceptual_loss", perceptual_loss.mean().item(), flush=True)
    if save_pred:
        pred = detokenize_output.batch["pixels"].clamp(0.0, 1.0)[:, 1:]
        real = pixels[:, 2:]
        for i in range(1):
            # for i in range(5):
            real_traj = torch.cat(
                [real[i : i + 1, j] for j in range(real.shape[1])], dim=-1
            )
            pred_traj = torch.cat(
                [pred[i : i + 1, j] for j in range(pred.shape[1])], dim=-1
            )
            plot_img(
                torch.cat(
                    [
                        real_traj.float(),
                        pred_traj.float(),
                        torch.abs(real_traj - pred_traj).float(),
                    ],
                    dim=-2,
                ),
                f"{self.config.trainer.experiment_name}-{i}",
            )

    reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    for i in range(len(batch)):
        data_item = batch[i]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = (
            data_item.batch["attention_mask"][prompt_length:].sum().long().item()
        )
        reward_tensor[i, valid_response_length - 1] = -loss[i].item()

    return reward_tensor, {
        "critic/recon_loss/mean": recon_loss.mean().item(),
        "critic/perceptual_loss/mean": perceptual_loss.mean().item(),
    }
