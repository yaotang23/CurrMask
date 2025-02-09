import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import gym
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
from video import VideoRecorder
import wandb
import omegaconf
torch.backends.cudnn.benchmark = True

def get_domain(task):
    if task.startswith('point_mass_maze'):
        return 'point_mass_maze'
    return task.split('_', 1)[0]

def get_task(task):
    if task.startswith('point_mass_maze'):
        return task.split('_', 3)[3]
    return task.split('_', 1)[1]

def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1

def get_dir(cfg):
    snapshot_base_dir = Path(cfg.snapshot_base_dir)
    snapshot_dir = snapshot_base_dir / get_domain(cfg.task)
    snapshot = snapshot_dir / str(cfg.seed) / f'snapshot_{cfg.snapshot_ts}.pt'
    return snapshot

def eval_prompt(global_step, agent, env, logger, context_iter, device, num_eval_episodes, video_recorder, cfg):
    step, episode, total_reward, r1,r2,r3,r4,r5 = 0, 0, 0, 0,0,0,0,0
    eval_until_episode = utils.Until(num_eval_episodes)

    batch = next(context_iter)
    states, actions, physics, reward, remaining = utils.to_torch(
        batch, device)
    init_obs = states[:, -1]
    states = states[:, :-1]

    while eval_until_episode(episode):
        time_step = env.reset()
        with env.physics.reset_context():
            env.physics.set_state(physics[episode, -1].cpu())

        video_recorder.init(env, enabled=True)
        context_s = states[episode]
        context_a = actions[episode]
        video_recorder.add_context_frames(env, physics[episode])
        len = agent.forecast_length
        for t in range(agent.forecast_length):
            if t == 0:
                obs = init_obs[episode]
            else:
                obs = np.asarray(time_step.observation)
                obs = torch.as_tensor(obs, device=device)
            with torch.no_grad(), utils.eval_mode(agent):
                
                action = agent.act_once(obs,
                                    context_s,
                                    context_a,
                                    global_step,
                                    agent.forecast_length-t,
                                    eval_mode=True)
            context_s = torch.cat((context_s, obs.unsqueeze(0)), dim=0)
            context_a = torch.cat((context_a, action.unsqueeze(0)), dim=0)

            time_step = env.step(action.cpu().numpy())
            video_recorder.record(env)
            total_reward += time_step.reward
            step += 1

        episode += 1
        video_recorder.render_context()
        video_recorder.save(f'{global_step}.mp4')

    expert_reward = reward.sum()
    with logger.log_and_dump_ctx(global_step, ty='eval') as log:
        log('episode_reward', total_reward / episode)
        log('expert_reward', expert_reward / episode)

        log('episode_length', step / episode)
        log('step', global_step)

        
@hydra.main(config_path='.', config_name='eval_prompt')
def main(cfg):
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    utils.set_seed_everywhere(cfg.data_seed)
    
    print("np.random.get_state()[1][0]:",np.random.get_state()[1][0])
    device = torch.device(cfg.device)
    

    env = dmc.make(cfg.task, seed=cfg.data_seed)
    cfg.agent.obs_shape = obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = action_shape=env.action_spec().shape
    
    path = get_dir(cfg)
    
    agent = hydra.utils.instantiate(cfg.agent,
                                    obs_shape=obs_shape,
                                    action_shape=action_shape,
                                    path=path)

    # create logger
    cfg.agent.transformer_cfg = agent.config
    exp_name = '_'.join([cfg.agent.name,cfg.task, str(int(cfg.snapshot_ts/10000)),cfg.pretrained_mask_type,str(cfg.pretrained_mask_ratio),str(cfg.pretrained_mask_len), str(cfg.data_seed)])
    wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.project,
               entity="tangyao2020",
               name=exp_name,
               config=wandb_config,
               settings=wandb.Settings(
                   start_method="thread",
                   _disable_stats=True,
               ),
               mode="online" if cfg.use_wandb else "offline",
               notes=cfg.notes,
               )
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    # create data storage
    domain = get_domain(cfg.task)
    specific_task = get_task(cfg.task)
    replay_dir = Path(cfg.replay_buffer_dir) /domain/ cfg.finetuned_data /specific_task
    goal_dir = Path(cfg.goal_buffer_dir) / domain / cfg.finetuned_data /specific_task
    print(f'replay dir, context dir: {replay_dir, goal_dir}')
    
    context_loader = make_replay_loader(env, goal_dir, cfg.goal_buffer_size,
                                    cfg.num_eval_episodes,
                                    cfg.goal_buffer_num_workers,
                                    cfg.discount,
                                    domain,
                                    traj_length=1,
                                    mode='prompt',
                                    cfg=cfg.agent,
                                    relabel=False,base_seed=cfg.data_seed)
    context_iter = iter(context_loader)

    # create video recorders
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    
    timer = utils.Timer()
    global_step = 0
    eval_every_step = utils.Every(cfg.eval_every_steps)

    if eval_every_step(global_step):
        logger.log("eval_total_time", timer.total_time(), global_step)
        eval_prompt(
            global_step,
            agent,
            env,
            logger,
            context_iter,
            device,
            cfg.num_eval_episodes,
            video_recorder,
            cfg
        )

if __name__ == '__main__':
    main()
