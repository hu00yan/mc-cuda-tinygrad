"""Pure-TinyGrad Async RL (PPO + GRPO, with preemptive GPU burst updates).

Architecture:
- Actor worker process runs ``MineCUEnv`` + a local policy copy, continuously
  pushing ``RolloutBatch`` chunks into ``rollout_queue``.
- When ``preempt_batches`` chunks have accumulated in ``rollout_queue``, the
  Learner preempts the GPU (via ``preempt_event`` when ``preempt_gpu=True``),
  drains the accumulated batches into one large fixed-shape batch to saturate
  the GPU across ``epochs`` x ``minibatches`` updates, writes back updated
  weights, and releases the GPU for the Actor to resume sampling.
- Supports both ``algo="ppo"`` (``CNNActorCritic`` + GAE + value/entropy loss)
  and ``algo="grpo"`` (``CNNPolicy`` + group-normalized returns + KL penalty).

Usage:
    PYTHONPATH=. DEV=CPU python -m minecu.rl.async_rl --algo ppo --total-steps 64
    PYTHONPATH=. DEV=CPU python -m minecu.rl.async_rl --algo grpo --total-steps 64
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import multiprocessing as mp
import queue
import time

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict

from minecu import MineCUEnv
from minecu.rl.common import (
    ACTION_DIM,
    buttons_from_actions,
    categorical_sample_and_logp,
    compute_gae,
    obs_to_tensor,
)
from minecu.rl.models import CNNActorCritic, CNNPolicy


@dataclass
class AsyncConfig:
    """Configuration for pure-TinyGrad Async RL training (PPO or GRPO)."""

    algo: str = "ppo"  # "ppo" | "grpo"
    batch: int = 8
    batch_groups: int = 2  # used when algo == "grpo" (B = batch_groups * group_size)
    group_size: int = 4    # used when algo == "grpo"
    rollout_steps: int = 8
    total_steps: int = 256
    preempt_batches: int = 2
    preempt_gpu: bool = True
    minibatches: int = 2
    epochs: int = 2
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    kl_coef: float = 0.01
    hidden_dim: int = 128
    queue_size: int = 8
    world: int = 16
    render: int = 16
    max_steps: int = 16
    episode_length: int = 16
    accel: str = "off"
    seed: int = 42
    log_interval: int = 1

    @property
    def effective_batch(self) -> int:
        if self.algo == "grpo":
            return self.batch_groups * self.group_size
        return self.batch


@dataclass
class RolloutBatch:
    """Single rollout chunk produced by the actor worker."""

    obs: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray | None
    last_values: np.ndarray | None
    policy_version: int


def _build_model(algo: str, obs_shape: tuple[int, int, int], hidden_dim: int):
    if algo == "ppo":
        return CNNActorCritic(obs_shape, action_dim=ACTION_DIM, hidden_dim=hidden_dim)
    if algo == "grpo":
        return CNNPolicy(obs_shape, action_dim=ACTION_DIM, hidden_dim=hidden_dim)
    raise ValueError(f"unknown algo {algo!r}; expected 'ppo' or 'grpo'")


def _snapshot_weights(model) -> dict[str, np.ndarray]:
    return {k: v.numpy().copy() for k, v in get_state_dict(model).items()}


def _apply_weights(model, weights: dict[str, np.ndarray]) -> None:
    tensors = {k: Tensor(v, dtype=dtypes.float32) for k, v in weights.items()}
    load_state_dict(model, tensors, verbose=False)


def _group_advantages(
    rewards: np.ndarray, batch_groups: int, group_size: int, gamma: float
) -> np.ndarray:
    """Group-relative normalized returns [T, B] for GRPO."""
    T, B = rewards.shape
    if gamma == 1.0:
        traj_returns = rewards.sum(axis=0)
    else:
        discounts = (gamma ** np.arange(T, dtype=np.float32))[:, None]
        traj_returns = (rewards * discounts).sum(axis=0)
    grouped = traj_returns.reshape(batch_groups, group_size)
    adv = (grouped - grouped.mean(axis=1, keepdims=True)) / (grouped.std(axis=1, keepdims=True) + 1e-8)
    return np.broadcast_to(adv.reshape(1, B).astype(np.float32), (T, B)).copy()


def _actor_worker_process(
    cfg: AsyncConfig,
    rollout_queue,
    weight_queue,
    stop_event,
    preempt_event,
    error_queue,
) -> None:
    """Actor process: runs MineCUEnv + local model, yielding GPU when preempted."""
    env = None
    try:
        B = cfg.effective_batch
        env = MineCUEnv(
            batch=B,
            world=cfg.world,
            width=cfg.render,
            height=cfg.render,
            max_steps=cfg.max_steps,
            episode_length=cfg.episode_length,
            accel=cfg.accel,
            seed=cfg.seed,
        )
        actor_model = _build_model(cfg.algo, env.observation_shape, cfg.hidden_dim)
        obs = env.reset()
        rng = np.random.default_rng(cfg.seed + 1)
        actor_version = 0

        while not stop_event.is_set():
            # If learner is preempting the GPU to burst-update, wait cleanly.
            while preempt_event.is_set() and not stop_event.is_set():
                time.sleep(0.002)
            if stop_event.is_set():
                break

            latest_weights = None
            while True:
                try:
                    actor_version, latest_weights = weight_queue.get_nowait()
                except queue.Empty:
                    break
            if latest_weights is not None:
                _apply_weights(actor_model, latest_weights)

            T = cfg.rollout_steps
            obs_buf = np.empty((T, B, *env.observation_shape), dtype=obs.dtype)
            act_buf = np.empty((T, B), dtype=np.int32)
            logp_buf = np.empty((T, B), dtype=np.float32)
            rew_buf = np.empty((T, B), dtype=np.float32)
            done_buf = np.empty((T, B), dtype=np.bool_)
            val_buf = np.empty((T, B), dtype=np.float32) if cfg.algo == "ppo" else None

            for t in range(T):
                if stop_event.is_set():
                    return
                while preempt_event.is_set() and not stop_event.is_set():
                    time.sleep(0.002)
                if stop_event.is_set():
                    return

                with Context(TRAINING=0):
                    out = actor_model(obs_to_tensor(obs))
                    if cfg.algo == "ppo":
                        logits_t, val_t = out
                        logits_np = logits_t.numpy()
                        val_buf[t] = val_t.numpy()
                    else:
                        logits_np = out.numpy()

                actions, old_logp = categorical_sample_and_logp(logits_np, rng)
                buttons, look = buttons_from_actions(actions)
                next_obs, rew, done = env.step(buttons, look)

                obs_buf[t] = obs
                act_buf[t] = actions.astype(np.int32)
                logp_buf[t] = old_logp
                rew_buf[t] = np.asarray(rew, dtype=np.float32)
                done_buf[t] = np.asarray(done, dtype=np.bool_)
                obs = next_obs

            last_values = None
            if cfg.algo == "ppo":
                with Context(TRAINING=0):
                    _, last_v_t = actor_model(obs_to_tensor(obs))
                    last_values = last_v_t.numpy()

            batch = RolloutBatch(
                obs=obs_buf,
                actions=act_buf,
                old_log_probs=logp_buf,
                rewards=rew_buf,
                dones=done_buf,
                values=val_buf,
                last_values=last_values,
                policy_version=actor_version,
            )
            while not stop_event.is_set():
                try:
                    rollout_queue.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    continue
    except BaseException as exc:
        try:
            error_queue.put_nowait(repr(exc))
        except Exception:
            pass
        stop_event.set()
    finally:
        for q in (rollout_queue, weight_queue, error_queue):
            try:
                q.cancel_join_thread()
            except Exception:
                pass
        if env is not None:
            env.close()


def _prepare_superbatch(
    batches: list[RolloutBatch], cfg: AsyncConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute per-batch advantages (GAE or GRPO) and concatenate into one super-batch."""
    obs_list, act_list, logp_list, adv_list, ret_list = [], [], [], [], []
    for b in batches:
        T, B = b.actions.shape
        if cfg.algo == "ppo":
            assert b.values is not None and b.last_values is not None
            adv, ret = compute_gae(
                b.rewards, b.values, b.dones, b.last_values,
                gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
            )
            ret_list.append(ret.reshape(T * B))
        else:
            adv = _group_advantages(b.rewards, cfg.batch_groups, cfg.group_size, cfg.gamma)

        obs_list.append(b.obs.reshape(T * B, *b.obs.shape[2:]))
        act_list.append(b.actions.reshape(T * B))
        logp_list.append(b.old_log_probs.reshape(T * B))
        adv_list.append(adv.reshape(T * B))

    b_obs = np.concatenate(obs_list, axis=0)
    b_act = np.concatenate(act_list, axis=0)
    b_logp = np.concatenate(logp_list, axis=0)
    b_adv = np.concatenate(adv_list, axis=0)
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
    b_ret = np.concatenate(ret_list, axis=0) if ret_list else None
    return b_obs, b_act, b_logp, b_adv, b_ret


def train_async(cfg: AsyncConfig) -> dict[str, float]:
    """Run decoupled actor-learner Async RL (PPO or GRPO) with GPU-burst preemption."""
    if cfg.algo not in ("ppo", "grpo"):
        raise ValueError(f"unknown algo {cfg.algo!r}; expected 'ppo' or 'grpo'")
    B = cfg.effective_batch
    if B <= 0:
        raise ValueError("effective batch must be positive")

    obs_shape = (cfg.render, cfg.render, 3)
    learner_model = _build_model(cfg.algo, obs_shape, cfg.hidden_dim)
    opt = Adam(get_parameters(learner_model), lr=cfg.lr)
    init_weights = _snapshot_weights(learner_model)

    ctx = mp.get_context("spawn")
    rollout_queue = ctx.Queue(maxsize=max(cfg.queue_size, cfg.preempt_batches + 1))
    weight_queue = ctx.Queue(maxsize=1)
    error_queue = ctx.Queue(maxsize=1)
    weight_queue.put((0, init_weights))
    stop_event = ctx.Event()
    preempt_event = ctx.Event()

    worker = ctx.Process(
        target=_actor_worker_process,
        args=(cfg, rollout_queue, weight_queue, stop_event, preempt_event, error_queue),
        name="async-rl-actor",
        daemon=True,
    )
    worker.start()

    rng = np.random.default_rng(cfg.seed)
    learner_version = 0
    total_env_steps = 0
    last_loss = 0.0
    last_lag = 0
    last_mean_rew = 0.0
    t_start = time.perf_counter()

    try:
        while total_env_steps < cfg.total_steps:
            # Accumulate up to preempt_batches (capped by remaining steps needed)
            steps_per_chunk = cfg.rollout_steps * B
            remaining_chunks = max(1, (cfg.total_steps - total_env_steps + steps_per_chunk - 1) // steps_per_chunk)
            target_batches = min(cfg.preempt_batches, remaining_chunks)

            collected: list[RolloutBatch] = []
            while len(collected) < target_batches:
                try:
                    err = error_queue.get_nowait()
                    raise RuntimeError(f"async actor worker failed: {err}")
                except queue.Empty:
                    pass
                try:
                    collected.append(rollout_queue.get(timeout=0.5))
                except queue.Empty:
                    if not worker.is_alive() and rollout_queue.empty():
                        try:
                            err = error_queue.get_nowait()
                            raise RuntimeError(f"async actor worker failed: {err}")
                        except queue.Empty:
                            break

            if not collected:
                break

            # Preempt GPU so Learner has exclusive access during burst update
            if cfg.preempt_gpu:
                preempt_event.set()

            try:
                last_lag = learner_version - min(b.policy_version for b in collected)
                last_mean_rew = float(np.mean([b.rewards.mean() for b in collected]))
                b_obs, b_act, b_logp, b_adv, b_ret = _prepare_superbatch(collected, cfg)
                N = b_act.shape[0]
                mb_size = max(1, N // max(1, cfg.minibatches))

                with Context(TRAINING=1):
                    Tensor.training = True
                    for _ in range(cfg.epochs):
                        perm = rng.permutation(N)
                        for start in range(0, N, mb_size):
                            idx = perm[start : start + mb_size]
                            obs_t = obs_to_tensor(b_obs[idx])
                            act_t = Tensor(b_act[idx].reshape(-1, 1), dtype=dtypes.int32)
                            old_logp_t = Tensor(b_logp[idx], dtype=dtypes.float32)
                            adv_t = Tensor(b_adv[idx], dtype=dtypes.float32)

                            opt.zero_grad()
                            if cfg.algo == "ppo":
                                assert b_ret is not None
                                ret_t = Tensor(b_ret[idx], dtype=dtypes.float32)
                                logits, values = learner_model(obs_t)
                                log_probs = logits.log_softmax(axis=-1)
                                probs = log_probs.exp()
                                new_logp = log_probs.gather(1, act_t).squeeze(1)
                                ratio = (new_logp - old_logp_t).exp()
                                surr1 = ratio * adv_t
                                surr2 = ratio.clip(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t
                                pg_loss = -surr1.minimum(surr2).mean()
                                vf_loss = ((values - ret_t) ** 2).mean()
                                entropy = -(probs * log_probs).sum(axis=-1).mean()
                                loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy
                            else:
                                logits = learner_model(obs_t)
                                log_probs = logits.log_softmax(axis=-1)
                                new_logp = log_probs.gather(1, act_t).squeeze(1)
                                log_ratio = new_logp - old_logp_t
                                ratio = log_ratio.exp()
                                surr1 = ratio * adv_t
                                surr2 = ratio.clip(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t
                                pg_loss = -surr1.minimum(surr2).mean()
                                if cfg.kl_coef > 0.0:
                                    kl_t = ((-log_ratio).exp() + log_ratio - 1.0).mean()
                                    loss = pg_loss + cfg.kl_coef * kl_t
                                else:
                                    loss = pg_loss

                            loss.backward()
                            opt.step()
                            last_loss = float(loss.numpy())

                learner_version += 1
                new_weights = _snapshot_weights(learner_model)

                try:
                    weight_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    weight_queue.put_nowait((learner_version, new_weights))
                except queue.Full:
                    pass
            finally:
                if cfg.preempt_gpu:
                    preempt_event.clear()

            total_env_steps += N
            if learner_version % cfg.log_interval == 0 or total_env_steps >= cfg.total_steps:
                elapsed = max(time.perf_counter() - t_start, 1e-6)
                sps = total_env_steps / elapsed
                print(
                    f"[async_rl:{cfg.algo}] version={learner_version} batches={len(collected)} "
                    f"lag={last_lag} steps={total_env_steps}/{cfg.total_steps} "
                    f"loss={last_loss:.4f} mean_rew={last_mean_rew:.4f} sps={sps:.1f}"
                )
    finally:
        stop_event.set()
        preempt_event.clear()
        for q in (rollout_queue, weight_queue, error_queue):
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass
        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2.0)

    total_elapsed = max(time.perf_counter() - t_start, 1e-6)
    return {
        "algo": cfg.algo,
        "steps": float(total_env_steps),
        "policy_version": float(learner_version),
        "last_lag": float(last_lag),
        "loss": last_loss,
        "mean_reward": last_mean_rew,
        "sps": total_env_steps / total_elapsed,
    }


def _parse_args() -> AsyncConfig:
    p = argparse.ArgumentParser(description="Pure-TinyGrad Async RL (PPO & GRPO) on MineCUEnv")
    p.add_argument("--algo", type=str, default="ppo", choices=("ppo", "grpo"))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--batch-groups", type=int, default=2)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--rollout-steps", type=int, default=8)
    p.add_argument("--total-steps", type=int, default=256)
    p.add_argument("--preempt-batches", type=int, default=2)
    p.add_argument("--no-preempt-gpu", action="store_true")
    p.add_argument("--minibatches", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--kl-coef", type=float, default=0.01)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--queue-size", type=int, default=8)
    p.add_argument("--world", type=int, default=16)
    p.add_argument("--render", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--episode-length", type=int, default=16)
    p.add_argument("--accel", type=str, default="off")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=1)
    args = p.parse_args()
    return AsyncConfig(
        algo=args.algo,
        batch=args.batch,
        batch_groups=args.batch_groups,
        group_size=args.group_size,
        rollout_steps=args.rollout_steps,
        total_steps=args.total_steps,
        preempt_batches=args.preempt_batches,
        preempt_gpu=not args.no_preempt_gpu,
        minibatches=args.minibatches,
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        kl_coef=args.kl_coef,
        hidden_dim=args.hidden_dim,
        queue_size=args.queue_size,
        world=args.world,
        render=args.render,
        max_steps=args.max_steps,
        episode_length=args.episode_length,
        accel=args.accel,
        seed=args.seed,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    train_async(_parse_args())
