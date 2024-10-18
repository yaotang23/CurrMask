import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs
import gym

import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
from video import VideoRecorder
import wandb
import omegaconf
import ptvsd
import h5py
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def get_dir(cfg):
    resume_dir = Path(cfg.resume_dir)
    snapshot = resume_dir /get_domain(cfg.task)/ str(
        cfg.seed) / f'snapshot_{cfg.resume_step}.pt'
    print('loading from', snapshot)
    return snapshot

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


        
@hydra.main(config_path='.', config_name='pretrain_ct')
def main(cfg):
    
    ############ for debug ############
    if cfg.debug==True:
        host = "localhost"
        port = 55557
        ptvsd.enable_attach(address=(host, port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    env = dmc.make(cfg.task, seed=cfg.seed)
    cfg.agent.obs_shape = obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = action_shape=env.action_spec().shape

    # create agent
    print('cfg.agent:',cfg.agent['name'])
    print('cfg.mask_type:',cfg.mask_type)
    if cfg.agent['name']=='mdp' or cfg.agent['name']=='t5':
        agent = hydra.utils.instantiate(cfg.agent,
                                    obs_shape=obs_shape,
                                    action_shape=action_shape,
                                    new_mask_ratio=cfg.mask_ratio,
                                    mask_type=cfg.mask_type,
                                    mask_len=cfg.mask_len,
                                    teacher_gamma=cfg.teacher_gamma,curr_init_mode = cfg.curr_init_mode)
    else:
        agent = hydra.utils.instantiate(cfg.agent,
                                        obs_shape=obs_shape,
                                        action_shape=action_shape)

    if cfg.resume is True:
        resume_dir = get_dir(cfg)
        payload = torch.load(resume_dir)
        agent.model.load_state_dict(payload['model'])

    domain = get_domain(cfg.task)
    snapshot_dir = work_dir / Path(cfg.snapshot_dir) / domain / str(cfg.seed)
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    # create logger
    exp_name = '_'.join([cfg.mask_type,domain,str(cfg.curr_init_mode),str(cfg.mask_len),str(cfg.seed)])
   
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

    replay_train_dir = Path(cfg.replay_buffer_dir) /domain/cfg.pretrained_data/'train'
    print(f'replay dir: {replay_train_dir}')
    train_loader = make_replay_loader(env, replay_train_dir, cfg.replay_buffer_size,
                                    cfg.batch_size,
                                    cfg.replay_buffer_num_workers,
                                    cfg.discount,
                                    domain,
                                    cfg.agent.transformer_cfg.traj_length,
                                    relabel=False)
    train_iter = iter(train_loader)
    replay_val_dir = Path(cfg.replay_buffer_dir) / domain / cfg.pretrained_data/'train'
    valid_loader = make_replay_loader(env, replay_val_dir, cfg.replay_buffer_size,
                                    int(cfg.batch_size/4),
                                    cfg.replay_buffer_num_workers,
                                    cfg.discount,
                                    domain,
                                    cfg.agent.transformer_cfg.traj_length,
                                    relabel=False,)
    valid_iter = iter(valid_loader)
    
    # create video recorders

    timer = utils.Timer()

    global_step = cfg.resume_step

    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    log_every_step = utils.Every(cfg.log_every_steps)

    while train_until_step(global_step):
        metrics = agent.update(train_iter, global_step,cfg.num_grad_steps)
        logger.log_metrics(metrics, global_step, ty='train')
        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty='train') as log:
                log('fps', cfg.log_every_steps / elapsed_time)
                log('total_time', total_time)
                log('step', global_step)

        #if global_step in cfg.snapshots:
        if global_step%cfg.save_snapshot_freq==0:
            snapshot = snapshot_dir / f'snapshot_{global_step}.pt'
            payload = {'model': agent.model.state_dict(), 'cfg': cfg.agent.transformer_cfg}
            with snapshot.open('wb') as f:
                torch.save(payload, f)
        # try to evaluate
        if eval_every_step(global_step) and (cfg.agent['name']=='mdp' or cfg.agent['name']=='t5' ):
            metrics = agent.eval_validation(valid_iter, num_eval=int(cfg.num_eval_episodes),step=global_step,mask_ratio=cfg.eval_mask_ratio,mask_type=cfg.eval_mask_type)
            logger.log_metrics(metrics, global_step, ty='valid')
            with logger.log_and_dump_ctx(global_step, ty='valid') as log:
                log('step', global_step)

        global_step += 1
        
if __name__ == '__main__':
    main()
