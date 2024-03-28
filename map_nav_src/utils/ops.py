import numpy as np
import torch

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    device = tensors[0].device
    output = torch.zeros(*size, dtype=dtype).to(device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def gen_seq_masks(seq_lens, max_len=None):
    if max_len is None:
        max_len = max(seq_lens)

    if isinstance(seq_lens, torch.Tensor):
        device = seq_lens.device
        masks = torch.arange(max_len).to(device).repeat(len(seq_lens), 1) < seq_lens.unsqueeze(1)
        return masks

    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=np.bool)
        
    seq_lens = np.array(seq_lens)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks

def pad_tensors_obj(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    dtype = tensors[0].dtype
    device = tensors[0].device

    # pad the first dimension
    area_lens = [t.size(0) for t in tensors]
    max_area_len = max(area_lens)
    bs = len(tensors)
    new_tensors = []
    for i in range(bs):
        origin_len, mid_len, last_dim = tensors[i].shape
        tmp_size = [max_area_len,mid_len,last_dim]
        tmp_tensor = torch.zeros(*tmp_size,dtype=dtype).to(device)
        tmp_tensor.data[:origin_len,...] = tensors[i].data
        new_tensors.append(tmp_tensor)

    # pad the second dimension
    tensors = new_tensors
    if lens is None:
        lens = [t.size(1) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[2:])

    mid_dim = tensors[0].shape[0]
    size = [bs, mid_dim, max_len] + hid
    output = torch.zeros(*size, dtype=dtype).to(device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :, :l, ...] = t.data
    return output

def pad_list(lists):
    lens = [len(t) for t in lists]
    max_len = max(lens)
    for i, l in enumerate(lists):
        if max_len - lens[i] > 0:
            pad_list = [0]*(max_len-lens[i])
            lists[i] = lists[i] + pad_list
    return lists