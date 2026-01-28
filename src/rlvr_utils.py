from __future__ import annotations

from typing import List

import torch
import tinker


def build_datums_from_group(
    prompt: tinker.ModelInput,
    sampled_tokens_G_T: List[List[int]],
    logprobs_G_T: List[List[float]],
    advantages_G: List[float],
) -> List[tinker.Datum]:
    datums: list[tinker.Datum] = []
    ob_len = prompt.length - 1
    for sampled_tokens, logprobs, advantage in zip(
        sampled_tokens_G_T, logprobs_G_T, advantages_G
    ):
        model_input = prompt.append(tinker.EncodedTextChunk(tokens=sampled_tokens[:-1]))
        target_tokens = [0] * ob_len + sampled_tokens
        padded_logprobs = [0.0] * ob_len + logprobs
        padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
        datum = tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": tinker.TensorData.from_torch(torch.tensor(padded_logprobs)),
                "advantages": tinker.TensorData.from_torch(torch.tensor(padded_advantages)),
            },
        )
        datums.append(datum)
    return datums
