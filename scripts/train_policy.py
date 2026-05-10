import argparse
import os
from pathlib import Path
import importlib.util

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sys

repo_dir = Path(__file__).resolve().parents[1]
if str(repo_dir) not in sys.path:
    sys.path.append(str(repo_dir))

from modules import bot_utils as bu, utils


class RandomDesiderataResetWrapper(gym.Wrapper):
    """Sample a new desiderata uniformly from observation bounds on each reset."""

    def __init__(self, env):
        super().__init__(env)
        self._lows = env.observation_boundaries[:, 0]
        self._highs = env.observation_boundaries[:, 1]

    def reset(self, **kwargs):
        desiderata = np.random.uniform(low=self._lows, high=self._highs).astype(np.float32)
        kwargs["desiderata"] = desiderata
        return self.env.reset(**kwargs)


class TerminalRewardWrapper(gym.Wrapper):
    """Keep only terminal reward; all intermediate rewards are set to zero."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        if done:
            info["terminal_reward"] = reward
            return obs, reward, terminated, truncated, info
        info["terminal_reward"] = 0.0
        return obs, 0.0, terminated, truncated, info


def _observation_boundaries_from_config(cfg):
    obs_cfg = cfg.get("observation", {})
    rew_cfg = cfg.get("rewarder", {})

    n_objects_min = float(obs_cfg.get("n_objects_min", 1))
    n_objects_max = float(obs_cfg.get("n_objects_max", 500))
    n_objects_scale = float(rew_cfg.get("n_objects_scale", 100))

    if n_objects_min <= 0 or n_objects_max <= 0:
        raise ValueError("observation n_objects_min/max must be positive")
    if n_objects_min >= n_objects_max:
        raise ValueError("observation n_objects_min must be < n_objects_max")
    if n_objects_scale <= 0:
        raise ValueError("rewarder n_objects_scale must be positive")

    low = np.tanh(n_objects_min / n_objects_scale)
    high = np.tanh(n_objects_max / n_objects_scale)

    if low >= high:
        raise ValueError("invalid transformed observation range; check n_objects_scale")

    return np.array([[low, high]], dtype=np.float32)


def _build_env(cfg, randomize_desiderata=True):
    bot_cfg = cfg.get("bot", {})
    rew_cfg = cfg.get("rewarder", {})
    env_cfg = cfg.get("env", {})

    kernel_size = int(bot_cfg.get("kernel_size", 3))
    f_symm = bool(bot_cfg.get("f_symm", True))
    h_symm = bool(bot_cfg.get("h_symm", False))
    v_symm = bool(bot_cfg.get("v_symm", False))
    fixed_center = bool(bot_cfg.get("fixed_center", True))
    kernel_center = float(bot_cfg.get("kernel_center", 0.5))
    activation = str(bot_cfg.get("activation", "identity"))
    kernel_prec = int(bot_cfg.get("kernel_prec", 2))
    scale_kernel = bool(bot_cfg.get("scale_kernel", False))

    n_objects_scale = float(rew_cfg.get("n_objects_scale", 100))
    sigma = float(rew_cfg.get("sigma", 1.0))

    edge_len = int(env_cfg.get("edge_len", 400))
    t_max = int(env_cfg.get("t_max", 2000))
    mid_timesteps = int(env_cfg.get("mid_timesteps", 20))
    var_th = float(env_cfg.get("var_th", 1e-4))
    pixel_lbound = float(env_cfg.get("pixel_lbound", -1.0))
    pixel_ubound = float(env_cfg.get("pixel_ubound", 1.0))

    observation_boundaries = _observation_boundaries_from_config(cfg)

    bot = bu.Bot(
        kernel_size=kernel_size,
        f_symm=f_symm,
        h_symm=h_symm,
        v_symm=v_symm,
        fixed_center=fixed_center,
        kernel_center=kernel_center,
        activation=activation,
        kernel_prec=kernel_prec,
        scale_kernel=scale_kernel,
    )

    n_free_parameters = utils.n_free_parameters(
        f_symm=bot.f_symm,
        hv_symm=bot.hv_symm,
        h_symm=bot.h_symm,
        v_symm=bot.v_symm,
        kernel_size=bot.kernel_size,
        fixed_center=bot.fixed_center,
    )

    low = observation_boundaries[:, 0]
    high = observation_boundaries[:, 1]
    desiderata = np.random.uniform(low=low, high=high).astype(np.float32)

    rewarder = bu.Rewarder(
        desiderata=desiderata,
        sigma=sigma,
        n_objects_scale=n_objects_scale,
    )

    env = bu.Environment(
        bot=bot,
        n_free_parameters=n_free_parameters,
        observation_boundaries=observation_boundaries,
        rewarder=rewarder,
        max_timesteps=t_max,
        mid_timesteps=mid_timesteps,
        edge_len=edge_len,
        var_th=var_th,
        pixel_lbound=pixel_lbound,
        pixel_ubound=pixel_ubound,
    )

    if randomize_desiderata:
        env = RandomDesiderataResetWrapper(env)

    # Always enforce terminal-only reward for policy training.
    env = TerminalRewardWrapper(env)

    return Monitor(env)


def main():
    parser = argparse.ArgumentParser(description="Train PPO policy for inverse Neural CA control")
    parser.add_argument(
        "--config",
        default="parameters.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg.get("env", {})
    if not bool(env_cfg.get("terminal_reward_only", True)):
        print(
            "Warning: env.terminal_reward_only is false in YAML, "
            "but training enforces terminal-only rewards."
        )

    train_cfg = cfg.get("training", {})

    seed = int(train_cfg.get("seed", 42))
    np.random.seed(seed)

    best_model_dir = str(train_cfg.get("best_model_dir", "models/best_policy"))
    tb_log_dir = str(train_cfg.get("tb_log_dir", "models/tb_logs"))
    os.makedirs(best_model_dir, exist_ok=True)

    has_tensorboard = importlib.util.find_spec("tensorboard") is not None
    tensorboard_log = tb_log_dir
    if has_tensorboard:
        os.makedirs(tb_log_dir, exist_ok=True)
    else:
        tensorboard_log = None
        if tb_log_dir:
            print("Warning: tensorboard is not installed; disabling tensorboard logging.")

    train_env = DummyVecEnv([lambda: _build_env(cfg, randomize_desiderata=True)])
    eval_env = DummyVecEnv([lambda: _build_env(cfg, randomize_desiderata=True)])

    clip_range = float(train_cfg.get("clip_range", train_cfg.get("epsilon", 0.2)))

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        n_steps=int(train_cfg.get("n_steps", 512)),
        batch_size=int(train_cfg.get("batch_size", 128)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        clip_range=clip_range,
        ent_coef=float(train_cfg.get("ent_coef", 0.02)),
        vf_coef=float(train_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
        use_sde=bool(train_cfg.get("use_sde", True)),
        sde_sample_freq=int(train_cfg.get("sde_sample_freq", 4)),
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=best_model_dir,
        eval_freq=int(train_cfg.get("eval_freq", 5000)),
        n_eval_episodes=int(train_cfg.get("n_eval_episodes", 8)),
        deterministic=bool(train_cfg.get("deterministic_eval", True)),
        render=False,
    )

    total_timesteps = int(train_cfg.get("total_timesteps", 100000))
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    best_model_path = os.path.join(best_model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"Best model saved at: {best_model_path}")
    else:
        print("Training completed, but no best_model.zip was produced.")


if __name__ == "__main__":
    main()
