import torch


class XBM:
    """
    Cross-Batch Memory for Embedding Learning
    """

    def __init__(self, xbm_size):
        self.K = xbm_size
        self.feats = torch.zeros(self.K, 128).cuda()
        self.targets = torch.zeros(self.K, dtype=torch.long)
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != 0

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size
