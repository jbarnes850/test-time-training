import argparse
import asyncio
import math
import os

from src.tinker_dataset import KernelDatasetBuilder, SplitConfig
from src.utils.tinker_utils import ensure_tinker_cookbook_on_path
from src.utils.path_utils import repo_root
from src.utils.env_utils import load_dotenv

load_dotenv()
ensure_tinker_cookbook_on_path()

from tinker_cookbook.rl.train import Config as RLConfig, StreamMinibatchConfig, main as rl_main
from tinker_cookbook.utils import ml_log


class GuardedLogger(ml_log.Logger):
    def __init__(self, inner: ml_log.Logger, kl_threshold: float = 0.01):
        self.inner = inner
        self.kl_threshold = kl_threshold
        self.prev_reward: float | None = None

    def log_hparams(self, config):
        return self.inner.log_hparams(config)

    def log_metrics(self, metrics, step=None):
        self.inner.log_metrics(metrics, step=step)

        for key in ["kl_sample_train_v1", "kl_sample_train_v2", "kl_post"]:
            if key in metrics and metrics[key] is not None:
                if float(metrics[key]) > self.kl_threshold:
                    raise RuntimeError(f"KL guardrail triggered: {key}={metrics[key]}")

        reward = metrics.get("reward/total")
        if reward is not None:
            reward = float(reward)
            if math.isnan(reward):
                raise RuntimeError("Reward is NaN")
            if self.prev_reward is not None and self.prev_reward > 0 and reward < self.prev_reward * 0.1:
                raise RuntimeError(
                    f"Reward collapse: prev={self.prev_reward:.6f} curr={reward:.6f}"
                )
            self.prev_reward = reward

    def close(self):
        return self.inner.close()

    def sync(self):
        return self.inner.sync()

    def get_logger_url(self):
        return self.inner.get_logger_url()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="splits/l1_seed42.json")
    parser.add_argument("--log_path", type=str, default="runs/rlvr_smoke")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--kl_threshold", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--renderer_name", type=str, default="gpt_oss_no_sysprompt")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--eval_mode", type=str, default="full", choices=["full", "fast"])
    parser.add_argument("--num_correct_trials", type=int, default=None)
    parser.add_argument("--num_perf_trials", type=int, default=None)
    parser.add_argument("--normalize_reward", action="store_true")
    parser.add_argument("--reward_baseline_window", type=int, default=32)
    parser.add_argument("--correctness_bonus", type=float, default=0.01)
    args = parser.parse_args()

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY not set. Add it to .env or export it.")

    os.environ["KERNELBENCH_EVAL_MODE"] = args.eval_mode
    if args.num_correct_trials is not None:
        os.environ["KERNELBENCH_NUM_CORRECT_TRIALS"] = str(args.num_correct_trials)
    if args.num_perf_trials is not None:
        os.environ["KERNELBENCH_NUM_PERF_TRIALS"] = str(args.num_perf_trials)
    if args.normalize_reward:
        os.environ["KERNELBENCH_REWARD_MODE"] = "normalized"
    os.environ["KERNELBENCH_REWARD_BASELINE_WINDOW"] = str(args.reward_baseline_window)
    os.environ["KERNELBENCH_CORRECTNESS_BONUS"] = str(args.correctness_bonus)

    # Enable streaming telemetry for real-time visibility
    log_dir = (repo_root() / args.log_path).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KERNELBENCH_TELEMETRY_PATH"] = str(log_dir / "telemetry.jsonl")

    # Patch logger with guardrails
    orig_setup = ml_log.setup_logging

    def setup_logging_guarded(*args_, **kwargs_):
        inner = orig_setup(*args_, **kwargs_)
        return GuardedLogger(inner, kl_threshold=args.kl_threshold)

    ml_log.setup_logging = setup_logging_guarded

    dataset_builder = KernelDatasetBuilder(
        split_config=SplitConfig(split_path=args.split, seed=42),
        batch_size=args.batch_size,
        group_size=args.group_size,
        level=1,
        model_name_for_tokenizer=args.model,
        renderer_name=args.renderer_name,
        num_epochs=args.num_epochs,
        max_batches=args.max_batches,
    )

    cfg = RLConfig(
        learning_rate=args.learning_rate,
        dataset_builder=dataset_builder,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        lora_rank=args.lora_rank,
        num_substeps=1,
        log_path=str((repo_root() / args.log_path).resolve()),
        save_every=args.save_every,
        eval_every=args.eval_every,
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=args.batch_size,
            num_minibatches=1,
        ),
    )

    asyncio.run(rl_main(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
