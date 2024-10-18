import hydra
import numpy as np
import torch
from typing import Dict, Optional, Sequence, Tuple, Union
from agent.eval_mask import action_masking,state_masking,prompt_masking

def random_masking( x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

def fixed_seq_masking(x, mask_ratio, sub_len):
    if sub_len==1:
        x_masked, mask, ids_restore=random_masking(x, mask_ratio)
        return x_masked, mask, ids_restore
    N, L, D = x.shape  # batch, length, dim
    cnt_blk = int((L + sub_len - 1) // sub_len) - 1

    noise = torch.rand(N, cnt_blk, device=x.device)  # noise in [0, 1]
    blk_shuffle = torch.argsort(noise, dim=1).to(x.device)

    blk_offset = blk_shuffle.unsqueeze(-1) * sub_len
    add = torch.arange(1, sub_len).repeat(cnt_blk * N).view(N, cnt_blk, sub_len - 1).to(x.device)
    blk_offset = (blk_offset + add).view(N, -1)
    blk_offset += torch.randint(0, sub_len, (1,), device=x.device).item()
    mask_tmp = (blk_offset) < L
    ids_shuffle_tmp = blk_offset[mask_tmp].reshape(N, -1)

    tmp = torch.arange(L, device=x.device).view(1, -1).repeat(N, 1)
    tmp_expanded = tmp.unsqueeze(-1).repeat(1, 1, ids_shuffle_tmp.shape[-1])
    ids_shuffle_tmp_expanded = ids_shuffle_tmp.unsqueeze(1).repeat(1, tmp.shape[-1], 1)
    mask = torch.any(tmp_expanded == ids_shuffle_tmp_expanded, dim=-1)
    left_tmp = tmp[~mask].reshape(N, -1)

    ids_shuffle = torch.cat([ids_shuffle_tmp, left_tmp], dim=1).flip([1])
    #print("ids_shuffle=",ids_shuffle)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    len_keep = int(L * (1 - mask_ratio))
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = x.gather(dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device, dtype=int)
    mask[:, :len_keep] = 0
    mask = mask.gather(dim=1, index=ids_restore)
    return x_masked, mask, ids_restore

def MixedInv_4p(x,mask_ratio,current_step,total_step):
    assert current_step <= total_step,"current_step should be less than total_step"
    current_step = float(current_step)
    total_step = float(total_step)
    if current_step/total_step < 0.25:
        mask_len =  np.random.randint(1,20)
    elif current_step/total_step < 0.5:
        mask_len = np.random.randint(1,15)
    elif current_step/total_step < 0.75:
        mask_len = np.random.randint(1,10)
    else:
        mask_len = np.random.randint(1,5)
    x_masked, mask, ids_restore = fixed_seq_masking(x, mask_ratio,mask_len)
    return x_masked, mask, ids_restore


def MixedProg_4p(x,mask_ratio,current_step,total_step):
    assert current_step <= total_step,"current_step should be less than total_step"
    current_step = float(current_step)
    total_step = float(total_step)
    if current_step/total_step < 0.25:
        mask_len =  np.random.randint(1,5)
    elif current_step/total_step < 0.5:
        mask_len = np.random.randint(1,10)
    elif current_step/total_step < 0.75:
        mask_len = np.random.randint(1,15)
    else:
        mask_len = np.random.randint(1,20)
    x_masked, mask, ids_restore = fixed_seq_masking(x, mask_ratio,mask_len)
    return x_masked, mask, ids_restore

def Mixed_masking(x,mask_len):
    lst = [0.15, 0.35, 0.55, 0.75, 0.95]
    mask_ratio = np.random.choice(lst)
    sub_len = np.random.randint(1, mask_len+1)
    x_masked, mask, ids_restore =fixed_seq_masking(x, mask_ratio, sub_len)
    return x_masked, mask, ids_restore

def prefix_masking(x,mask_ratio, sub_len):
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(sub_len)

    noise = torch.arange(L, device=x.device)*0.001
    noise = noise.repeat(N).view(N,L)
    #noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore

def Prefix_masking(x,sub_len):
    sub_len = np.random.randint(1, int(sub_len+1))
    mask_ratio=0.
    x_masked, mask, ids_restore = prefix_masking(x,mask_ratio, sub_len)
    return x_masked, mask, ids_restore
