import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils
#from dm_control.utils import rewards
from einops import rearrange, reduce, repeat
from agent.modules.attention import Block, CausalSelfAttention


import agent.mask_utils as mask_utils

from agent import curriculum

class MaskedDP(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        # MAE encoder specifics
        self.n_embd = config.n_embd
        self.max_len = config.traj_length * 2
        self.pe = config.pe
        self.norm = config.norm
        print('norm', self.norm)
        self.state_embed = nn.Linear(obs_dim, self.n_embd)
        self.action_embed = nn.Linear(action_dim, self.n_embd)
        self.encoder_blocks = nn.ModuleList([Block(config) for _ in range(config.n_enc_layer)])
        self.encoder_norm = nn.LayerNorm(self.n_embd)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_state_embed = nn.Linear(self.n_embd, self.n_embd)
        self.decoder_action_embed = nn.Linear(self.n_embd, self.n_embd)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.n_embd))

        self.decoder_blocks = nn.ModuleList([Block(config) for _ in range(config.n_dec_layer)])

        self.action_head = nn.Sequential(nn.LayerNorm(self.n_embd), nn.ReLU(inplace=True), nn.Linear(self.n_embd, action_dim), nn.Tanh()) # decoder to patch
        self.state_head = nn.Sequential(nn.LayerNorm(self.n_embd), nn.ReLU(inplace=True), nn.Linear(self.n_embd, obs_dim))
        # --------------------------------------------------------------------------
        self.decoder_input = None
        self.encoder_input = None
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = utils.get_1d_sincos_pos_embed_from_grid(self.n_embd, self.max_len)
        pe = torch.from_numpy(pos_embed).float().unsqueeze(0) / 2.
        self.register_buffer('pos_embed', pe)
        self.register_buffer('decoder_pos_embed', pe)
        self.register_buffer('attn_mask', torch.ones(self.max_len, self.max_len)[None, None, ...])
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_encoder(self, states, actions, mask_ratio, mask_type=None,mask_len=None,current_step=None,total_step=None,denoiser_mode=None):
        batch_size, T, obs_dim = states.size()
        # print("states.size: ",states.size(),"actions.size: ",actions.size())
        s_emb = self.state_embed(states)
        a_emb = self.action_embed(actions)

        x = torch.stack([s_emb, a_emb], dim=1).permute(0, 2, 1, 3).reshape(batch_size, 2*T, self.n_embd)
        x = x + self.pos_embed
        #print('mask_type:',mask_type)
        x,mask,ids_restore=mask_utils.mask(x,mask_ratio,mask_type,mask_len,current_step,total_step,denoiser_mode)
        # apply Transformer blocks
        self.encoder_input = x
        for blk in self.encoder_blocks:
            #x = blk(x, self.enc_attn_mask)
            x = blk(x, self.attn_mask)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        s = self.decoder_state_embed(x[:, ::2])
        a = self.decoder_action_embed(x[:, 1::2])

        x = torch.stack([s, a], dim=1).permute(0, 2, 1, 3).reshape_as(x)

        # add pos embed
        x = x + self.decoder_pos_embed
        self.decoder_input = x
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            #x = blk(x, self.dec_attn_mask)
            x = blk(x, self.attn_mask)

        # predictor projection
        s = self.state_head(x[:, ::2])
        a = self.action_head(x[:, 1::2])

        return s, a

    def forward_loss(self, target_s, target_a, pred_s, pred_a, mask):
        batch_size, T, _ = target_s.size()
        # apply normalization
        if self.norm == 'l2':
            target_s = target_s / torch.norm(target_s, dim=-1, keepdim=True)
        elif self.norm == 'mae':
            mean = target_s.mean(dim=-1, keepdim=True)

            var = target_s.var(dim=-1, keepdim=True)
            target_s = (target_s - mean) / (var + 1.e-6)**.5

        loss_s = (pred_s - target_s) ** 2
        loss_a = (pred_a - target_a) ** 2
        loss = torch.stack([loss_s.mean(dim=-1), loss_a.mean(dim=-1)], dim=1).permute(0, 2, 1).reshape(batch_size, 2*T)
        masked_loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss_s = loss_s.mean()
        loss_a = loss_a.mean()
        return masked_loss, loss_s, loss_a


class MaskedDPAgent:
    def __init__(self,
                 name,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 batch_size,
                 use_tb,
                 mask_ratio,
                 transformer_cfg,
                 new_mask_ratio=0.55,
                 mask_type='random',
                 mask_len=1,teacher_gamma=0.2,curr_init_mode='1'):

        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.use_tb = use_tb
        self.config = transformer_cfg

        # models
        
        self.model = MaskedDP(obs_shape[0], action_shape[0], transformer_cfg).to(device)
        self.mask_ratio = mask_ratio
        self.new_mask_ratio = new_mask_ratio
        self.mask_type = mask_type
        self.mask_len = mask_len
        self.curr_init_mode = str(curr_init_mode)
        # optimizers
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        print("number of parameters: %e", sum(p.numel() for p in self.model.parameters()))
        if mask_utils.curriculum_true(mask_type):
            self.teacher = mask_utils.teacher_init(mask_type,teacher_gamma,mask_len,curr_init_mode)
            self.prev_total_loss = None
            self.current_task = self.teacher.get_task()
            self.hist_tasks = []
            self.hist_rewards = []
        
        self.train()
        
    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def update_mdp(self, states, actions,current_step=None,total_step=None):
        metrics = dict()
        if self.mask_type == 'random':
            mask_ratio = np.random.choice(self.mask_ratio)
            latent, mask, ids_restore = self.model.forward_encoder(states, actions, mask_ratio)
        elif mask_utils.curriculum_true(self.mask_type):
            if mask_utils.arm_length(self.mask_type)==2:
                mask_len, mask_ratio = self.current_task
                latent, mask, ids_restore = self.model.forward_encoder(states, actions, mask_ratio,self.mask_type,mask_len,current_step,total_step)
            elif mask_utils.arm_length(self.mask_type)==3:
                mask_len, mask_ratio, denoiser_mode = self.current_task
                latent, mask, ids_restore = self.model.forward_encoder(states, actions, mask_ratio,self.mask_type,mask_len,current_step,total_step,denoiser_mode)
            else:
                raise NotImplementedError
        else:
            latent, mask, ids_restore = self.model.forward_encoder(states, actions, self.new_mask_ratio,self.mask_type,self.mask_len,current_step,total_step)
        
        pred_s, pred_a = self.model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        mask_loss, state_loss, action_loss = self.model.forward_loss(states, actions, pred_s, pred_a, mask)
        if self.config.loss == 'masked':
            loss = mask_loss
        elif self.config.loss == 'total':
            loss = state_loss + action_loss
        else:
            raise NotImplementedError

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        if self.use_tb:
            metrics['mask_loss'] = mask_loss.item()
            metrics['state_loss'] = state_loss.item()
            metrics['action_loss'] = action_loss.item()

        return metrics

    def eval_validation(self, val_iter, num_eval,step=None,mask_ratio=0.5,mask_type='fixed_seq_masking2',mask_len=1):
        # self.model.eval()
        metrics = dict()
        total_mask_loss = total_state_loss = total_action_loss =0
        total_mask_losses = total_state_losses = total_action_losses = [0.,0.]
        for i in range(num_eval):
            batch = next(val_iter)
            obs, action, _, _, _, _ = utils.to_torch(
                batch, self.device)
            mask_ratio = np.random.choice(self.mask_ratio)
            if mask_type!='world_agent':
                with torch.no_grad():
                    latent, mask, ids_restore = self.model.forward_encoder(obs, action, mask_ratio,mask_type,mask_len)
                    pred_s, pred_a = self.model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
                    mask_loss, state_loss, action_loss = self.model.forward_loss(obs, action, pred_s, pred_a, mask)
                    total_mask_loss += mask_loss.item()
                    total_state_loss += state_loss.item()
                    total_action_loss += action_loss.item()
            else:
                mask_types = ['action_masking','state_masking']
                mask_len = 63
                for i in range(len(mask_types)):
                    with torch.no_grad():
                        latent, mask, ids_restore = self.model.forward_encoder(obs, action, mask_ratio,mask_types[i],mask_len)
                        pred_s, pred_a = self.model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
                        mask_loss, state_loss, action_loss = self.model.forward_loss(obs, action, pred_s, pred_a, mask)
                        total_mask_losses[i] += mask_loss.item()
                        total_state_losses[i] += state_loss.item()
                        total_action_losses[i] += action_loss.item()

        # For auto-curriculum
        if mask_utils.curriculum_true(self.mask_type):
            total_loss = (total_state_loss + total_action_loss) / num_eval

            if self.prev_total_loss is not None:
                reward = self.prev_total_loss - total_loss  # target prediction gain
                self.hist_tasks.append(self.current_task)
                self.hist_rewards.append(reward)
                print("reward:",reward)
            if len(self.hist_rewards) >= 2:
                # Update the teacher
                scaled_reward = self.teacher.normalize_reward(reward, self.hist_rewards)
                self.teacher.update(self.current_task, scaled_reward)
            
            # Update task & other info
            self.prev_total_loss = total_loss
            self.current_task = self.teacher.get_task()
            print(f"New task: {self.current_task}")
            arm_prob = self.teacher.task_probabilities

            
        if self.use_tb:
            metrics['val_mask_loss'] = total_mask_loss/num_eval
            metrics['val_state_loss'] = total_state_loss/num_eval
            metrics['val_action_loss'] = total_action_loss/num_eval
            metrics['val_total_loss'] = metrics['val_state_loss'] + metrics['val_action_loss']

        return metrics
    

    def update(self, replay_iter, current_step=None,total_step=None):
        metrics = dict()
        batch = next(replay_iter)
        obs, action, _, _, _, _ = utils.to_torch(
            batch, self.device)
        
        # update critic
        if self.mask_type=='random':
            metrics.update(self.update_mdp(obs, action))
        else:
            assert current_step!=None and total_step!=None,"current_step!=None and total_step!=None"
            metrics.update(self.update_mdp(obs, action,current_step,total_step))

        return metrics
    
    