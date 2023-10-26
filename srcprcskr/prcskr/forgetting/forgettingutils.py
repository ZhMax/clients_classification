"""
Utilities to compute epochs when examples were learned and 
forgotten during model training

https://arxiv.org/pdf/2210.15031.pdf, https://github.com/pratyushmaini/ssft
"""


import torch
from typing import Any, Dict

def get_first_epoch_where_we_learn_forever(mask: torch.tensor):
    #never forget once you learned
    # Example:
    # >>> a = torch.tensor([0,0,1,1,0,1,1,1,1])
    # >>> z  = torch.flip(a, [0])
    # >>> z
    # tensor([1, 1, 1, 1, 0, 1, 1, 0, 0])
    # >>> z.argmax()
    # tensor(0)
    # >>> z.argmin()
    # tensor(4)

    # What if example is correct from the beginning? 
    # >> Just add an extra row at the top that has all 0s

    # What if we never overfit on that sample even after many epochs?
    # >> Just add an extra row at the bottom that has all 1s
    mask = torch.cat([torch.zeros(1, mask.shape[1]), mask, torch.ones(1, mask.shape[1])])
    z  = torch.flip(mask, [0])
    mins = z.argmin(dim = 0)
    total_epochs = mask.shape[0]

    
    return (total_epochs - mins).float().numpy()

def get_forgetting_counts(masks: torch.tensor):
    num_examples = masks.shape[1]
    num_epochs = masks.shape[0]
    mask2 = torch.ones((num_epochs, num_examples))
    # mask2 represents the accuracy for the same example at the next epoch
    # if mask1 is greater than mask2 then a forgetting event happened
    mask2[:-1] = masks[1:]
    diff_mask = masks - mask2
    diff_mask[diff_mask > 0] = 1
    diff_mask[diff_mask != 1] = 0
    num_forgetting_events = diff_mask.sum(dim = 0)
    return num_forgetting_events


def get_first_epoch_where_we_forget(mask: torch.tensor):
    # Example:
    # >>> a = torch.tensor([1,1,0,0,1,1,0,0,0,0])
    # >>> b = torch.tensor([1,1,1,1,1,1])
    # >>> a.argmin()
    # tensor(2)
    # What if example was never learnt? 
    # Just add an extra row at the top that has all 1s

    # What if example is never forgotten? 
    # Just add an extra row at the bottom that has all 0s
    mask = torch.cat([torch.ones(1, mask.shape[1]), mask, torch.zeros(1, mask.shape[1])])
    return mask.argmin(dim = 0).float().numpy()


def get_first_epoch_where_we_forget_forever(mask: torch.tensor):
    #this is same as get first epoch where we learn forever, 
    # except that we reverse the masks
    return get_first_epoch_where_we_learn_forever(1 - mask)



def get_forget_forever_epochs_for_examples(ret_ft: Dict['str', Any]):
    #Compute epochs when examples were forgotten forever
    masks_ft = ret_ft['acc_mask']
    masks_ft = torch.cat(masks_ft)
    forget_epochs = get_first_epoch_where_we_forget_forever(masks_ft)
    file_names = ret_ft['file_names']

    return forget_epochs, file_names

def get_learn_epoch_forgetting_counts_for_examples(ret_pre: Dict['str', Any]):
    #Compute epochs when examples were learned forever and 
    # quantity of forgetting for each example.
    # If the example was learned from the beginning or was not learned, 
    # the forgetting count is equal to 0
    masks_pre = ret_pre['acc_mask']
    masks_pre = torch.cat(masks_pre)
    learn_epochs = get_first_epoch_where_we_learn_forever(masks_pre)
    forgetting_counts = get_forgetting_counts(masks_pre)
    file_names = ret_pre['file_names']

    return learn_epochs, forgetting_counts, file_names 
