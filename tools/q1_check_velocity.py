"""
Load Q1 checkpoint and print lin_vel / ang_vel tracking.
"""
import sys, numpy as np
from pathlib import Path
PROJECT_ROOT = Path("/root/autodl-tmp/ASAP")
sys.path.insert(0, str(PROJECT_ROOT))

from isaacgym import gymapi
import torch
from omegaconf import OmegaConf

ckpt = PROJECT_ROOT / "logs/TEST_Q1_loco/20260529_164937-Q1_Stand_AntiSlip-legged_base-q1_22dof/model_600.pt"
ckpt_data = torch.load(str(ckpt), map_location="cpu")
config = OmegaConf.load(str(ckpt.parent / "config.yaml"))

# Build actor from state dict directly
sd = ckpt_data["actor_model_state_dict"]
actor = torch.nn.Sequential(
    torch.nn.Linear(sd["actor_module.module.0.weight"].shape[1], 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 256),
    torch.nn.ELU(),
    torch.nn.Linear(256, 128),
    torch.nn.ELU(),
    torch.nn.Linear(128, 22),
)
# Strip "actor_module." prefix, skip "std" (action log std, not part of Sequential)
stripped = {}
for k, v in sd.items():
    if "std" in k: continue
    new_k = k.replace("actor_module.module.", "")
    stripped[new_k] = v
actor.load_state_dict(stripped)
actor.to("cuda:0")
actor.eval()

# Build env
rc = config.robot
from omegaconf import open_dict
with open_dict(config):
    config.save_rendering_dir = None
    config.add_extra_action = False
    config.max_episode_length_s = config.env.config.max_episode_length_s

from humanoidverse.envs.locomotion.stand_smoke import LeggedRobotStandSmoke
env = LeggedRobotStandSmoke(config=config, device="cuda:0")
obs = env.reset_all()
obs_dim = obs["actor_obs"].shape[-1]
print("actor input dim:", obs_dim, " actions:", env.dim_actions)

print("\nRolling out...")
print("%-4s %7s %7s %7s %7s %7s %7s %7s %7s %7s" %
      ("f", "rz", "pgz", "lv_x", "lv_y", "lv_z", "av_x", "av_y", "av_z", "cmd_lx"))

log_std = sd["actor_module.std"].to("cuda:0")

all_lv, all_av = [], []
for step in range(500):
    with torch.no_grad():
        mean = actor(obs["actor_obs"].float())
        a = mean.detach()  # use mean directly (no exploration noise for eval)
    obs, rew, reset, extras = env.step({"actions": a})
    rs = env.simulator.robot_root_states[0].cpu().numpy()
    lv_x = rs[7]; lv_y = rs[8]; lv_z = rs[9]
    av_z = rs[12]
    all_lv.append((lv_x, lv_y)); all_av.append(av_z)
    if step % 25 == 0:
        qx,qy,qz_,qw=rs[3:7]; pgz_v=1-2*(qx*qx+qy*qy)
        print("%-4d %+7.3f %7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f" %
              (step, rs[2], pgz_v, lv_x, lv_y, lv_z, rs[10], rs[11], av_z, 0.0))

# Summary
all_lv = np.array(all_lv); all_av = np.array(all_av)
print("\n=== SUMMARY ===")
print("Mean lin_vel: [%.3f, %.3f, %.3f]" % (all_lv[:,0].mean(), all_lv[:,1].mean(),0))
print("Mean |lin_vel_xy|: %.4f" % np.linalg.norm(all_lv, axis=1).mean())
print("Mean ang_vel_z: %.4f" % all_av.mean())
print("Std ang_vel_z: %.4f" % all_av.std())
print("Max |ang_vel_z|: %.4f" % abs(all_av).max())
env.simulator.gym.destroy_sim(env.simulator.sim)
