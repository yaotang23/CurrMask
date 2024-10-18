import hydra
import numpy as np
import torch
import agent.eval_mask as eval_mask
from agent import curriculum
import agent.mask_methods as mask_methods

def curriculum_true(mask_type):
    if mask_type in ['CurrMask',  
                     'CurrMask_v2', 
                     'CurrMask_v3', 
                     'currmask_mod','currmask_random_prefix','curr_mtm']:
        return True
    else:
        return False

def arm_length(mask_type):
    len = int(0)
    if mask_type in ['CurrMask']:
        len=2
    elif mask_type in ['currmask_mod','currmask_random_prefix']:
        len=3
    else:
        raise NotImplementedError
    
    return len

def mask(x,mask_ratio,mask_type=None,mask_len=None,current_step=None,total_step=None,denoiser_mode=None):
    if mask_type=='MixedProg_4p':
        x,mask,ids_restore = mask_methods.MixedProg_4p(x,mask_ratio,current_step,total_step)
    elif mask_type=='MixedInv_4p':
        x,mask,ids_restore = mask_methods.MixedInv_4p(x,mask_ratio,current_step,total_step)
    elif mask_type=='Mixed_masking':
        x,mask,ids_restore = mask_methods.Mixed_masking(x,mask_len)
    elif mask_type=='fixed_seq_masking':
        x,mask,ids_restore = mask_methods.fixed_seq_masking(x,mask_ratio,mask_len)
    elif curriculum_true(mask_type) and arm_length(mask_type)==2:
        x,mask,ids_restore = mask_methods.fixed_seq_masking(x,mask_ratio,mask_len)
    elif curriculum_true(mask_type) and arm_length(mask_type)==3:
        if denoiser_mode==0:
            x,mask,ids_restore = mask_methods.fixed_seq_masking(x,mask_ratio,mask_len)
        elif denoiser_mode==1:
            x,mask,ids_restore = mask_methods.prefix_masking(x,mask_ratio,mask_len)
        else:
            raise NotImplementedError
    else: #random mask
        x, mask,ids_restore = mask_methods.random_masking(x, mask_ratio)
    return  x, mask,ids_restore
    
def teacher_init(mask_type,teacher_gamma,mask_len=20,curr_init_mode='1'):
    if mask_type == 'CurrMask':
        print("gamma in exp3:",teacher_gamma)
        teacher = curriculum.TeacherExp3(
            tasks=list((sub_len, q) for sub_len in range(1, mask_len+1) for q in (0.15, 0.35, 0.55, 0.75, 0.95)),
            gamma=teacher_gamma,
            mode='single',
            num_sub_task=1,
            init_mode = curr_init_mode
        )
        print("mask_type:",mask_type)
        print("num of arm:",mask_len*5)
    return teacher


        