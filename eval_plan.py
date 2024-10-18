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

        
def eval_mdp(global_step, agent, env, logger, goal_iter, device, num_eval_episodes, video_recorder):
    step, episode, total_dist2goal = 0, 0, []
    eval_until_episode = utils.Until(num_eval_episodes)
    batch = next(goal_iter)
    start_obs, start_physics, goal, goal_physics, time_budget = utils.to_torch(
        batch, device)

    while eval_until_episode(episode):
        time_step = env.reset()
        with env.physics.reset_context():
            env.physics.set_state(start_physics[episode].cpu())
        dist2goal = 1e6
        video_recorder.init(env, enabled=True)

        with torch.no_grad(), utils.eval_mode(agent):
            actions = agent.multi_goal_act(start_obs[episode].unsqueeze(0), goal[episode], time_budget[episode])

        states = []
        for a in actions:
            time_step = env.step(a)
            video_recorder.record(env)
            states.append(np.asarray(time_step.observation))
        states = np.array(states)
        episode_dist = []
        episode_budget = time_budget[episode]
        dist2goal = 1e5
        current_goal = goal[episode, i]
        for t in range(len(states)):
            dist = np.linalg.norm(states[t] - current_goal.cpu().numpy())
            dist2goal = min(dist2goal, dist)

        episode_dist.append(dist2goal)

        video_recorder.save(f'{global_step}.mp4')
        video_recorder.render_goal(env, goal_physics[episode, -1])
        episode += 1
        total_dist2goal.append(episode_dist)
            
    total_dist2goal = np.array(total_dist2goal)
    with logger.log_and_dump_ctx(global_step, ty='eval') as log:
        log('dist2goal1', np.mean(total_dist2goal[:, 0]))
        log('dist2goal2', np.mean(total_dist2goal[:, 1]))
        log('dist2goal3', np.mean(total_dist2goal[:, 2]))
        log('dist2goal4', np.mean(total_dist2goal[:, 3]))
        log('dist2goal5', np.mean(total_dist2goal[:, 4]))
        log('step', global_step)


def eval_seq_bc(global_step, agent, env, logger, goal_iter, device, num_eval_episodes, video_recorder):
    step, episode, total_dist2goal= 0, 0, []
    eval_until_episode = utils.Until(num_eval_episodes)
    batch = next(goal_iter)
    start_obs, start_physics, goal, goal_physics, time_budget = utils.to_torch(
        batch, device)

    while eval_until_episode(episode):
        time_step = env.reset()
        with env.physics.reset_context():
            env.physics.set_state(start_physics[episode].cpu())
        video_recorder.init(env, enabled=True)
        epi_budget = time_budget[episode]
        obs = start_obs[episode].unsqueeze(0)
        episode_dist = []
        for i in range(epi_budget.shape[0]):
            dist2goal = 1e6
            if obs.shape[0] > 1:
                obs = obs[-1].unsqueeze(0)
            current_goal = goal[episode, i]
            if i == 0:
                current_budget = epi_budget[i] + 2
            else:
                current_budget = epi_budget[i] - epi_budget[i-1] + 2
            for _ in range(current_budget):
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(obs,
                                       current_goal,
                                       global_step)
                time_step = env.step(action)
                obs_t = np.asarray(time_step.observation)
                obs_t = torch.as_tensor(obs_t, device=device)
                obs = torch.cat((obs, obs_t.unsqueeze(0)), dim=0)
                dist = np.linalg.norm(time_step.observation - current_goal.cpu().numpy())
                dist2goal = min(dist2goal, dist)
                video_recorder.record(env)
            episode_dist.append(dist2goal)   
        total_dist2goal.append(episode_dist)
        video_recorder.save(f'{global_step}.mp4')
        video_recorder.render_goal(env, goal_physics[episode, -1])
        episode += 1

    total_dist2goal = np.array(total_dist2goal)
    with logger.log_and_dump_ctx(global_step, ty='eval') as log:
        log('dist2goal1', np.mean(total_dist2goal[:, 0]))
        log('dist2goal2', np.mean(total_dist2goal[:, 1]))
        log('dist2goal3', np.mean(total_dist2goal[:, 2]))
        log('dist2goal4', np.mean(total_dist2goal[:, 3]))
        log('dist2goal5', np.mean(total_dist2goal[:, 4]))

        log('step', global_step)

@hydra.main(config_path='.', config_name='eval_plan')
def main(cfg):
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    utils.set_seed_everywhere(cfg.data_seed)
    device = torch.device(cfg.device)

    # create envs

    env = dmc.make(cfg.task, seed=cfg.data_seed)
    cfg.agent.obs_shape = obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = action_shape=env.action_spec().shape
    

    # create agent
    path = get_dir(cfg)
    agent = hydra.utils.instantiate(cfg.agent,
                                    obs_shape=obs_shape,
                                    action_shape=action_shape,
                                    path=path)

    # create logger
    cfg.agent.transformer_cfg = agent.config
    exp_name = '_'.join([cfg.agent.name,cfg.task, str(int(cfg.snapshot_ts/10000)),cfg.pretrained_mask_type,str(cfg.pretrained_mask_ratio), str(cfg.pretrained_mask_len),str(cfg.data_seed)])
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
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb, mode='multi_goal')


    domain = get_domain(cfg.task)
    specific_task = get_task(cfg.task)
    goal_dir = Path(cfg.goal_buffer_dir) / domain / cfg.finetuned_data /specific_task
    print(f'replay dir, goal dir: {goal_dir, goal_dir}')
    goal_loader = make_replay_loader(env, goal_dir, cfg.goal_buffer_size,
                                    cfg.num_eval_episodes,
                                    cfg.goal_buffer_num_workers,
                                    cfg.discount,
                                    domain=domain,
                                    traj_length=1,
                                    mode='multi_goal',
                                    cfg=agent.config,
                                    relabel=False,base_seed=cfg.data_seed)
    goal_iter = iter(goal_loader)

    # create video recorders
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    timer = utils.Timer()

    global_step = 0

    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    #log_every_step = utils.Every(cfg.log_every_steps)

    while train_until_step(global_step):
        # try to evaluate
        if eval_every_step(global_step):
            logger.log('eval_total_time', timer.total_time(), global_step)
            if cfg.agent.name == 'mdp_goal' or cfg.agent.name == "t5_goal":
                eval_mdp(global_step, agent, env, logger, goal_iter, device, cfg.num_eval_episodes,
                        video_recorder)
            elif cfg.agent.name == 'seq_goal':
                eval_seq_bc(global_step, agent, env, logger, goal_iter, device, cfg.num_eval_episodes,
                            video_recorder)
            else:
                raise NotImplementedError
        global_step += 1
        break

if __name__ == '__main__':
    main()
