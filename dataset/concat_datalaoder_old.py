from torch.utils.data.dataloader import DataLoader


class ConcatDataloader:
    """
    Deprecated, since it not compatible with tqdm
    Concat Dataloader that concat different dataloaders in one.
    Support polling_strategy are 'batch_wise' or 'dataset_wise'.
    'batch_wise' means using different datasets for each batch and loop over all datasets.
    'dataset_wise' means using after using all batches in one dataset and then move to the next dataset.
    """

    def __init__(self, datasets, polling_strategy='batch_wise',
                 batch_size=1, shuffle=False, sampler=None,
                 batch_samplers=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):

        self.polling_strategy = polling_strategy

        self.dataloaders = []
        for i, dataset in enumerate(datasets):
            dataloader = DataLoader(dataset, batch_size, shuffle, sampler,
                                    batch_samplers[i], num_workers, collate_fn,
                                    pin_memory, drop_last, timeout,
                                    worker_init_fn, multiprocessing_context)
            self.dataloaders.append(dataloader)
        self._len = sum([len(d) for d in self.dataloaders])

        self.count = 0
        self.iter_dataloaders = [iter(d) for d in self.dataloaders]
        self.idx_to_dataset_idx = self.generate_idx_to_dataset_idx(self.polling_strategy,
                                                                   {i: len(d) for i, d in enumerate(self.dataloaders)})

    def __len__(self):
        return self._len

    def __iter__(self):
        self.count = 0
        self.iter_dataloaders = [iter(d) for d in self.dataloaders]
        return self

    @staticmethod
    def generate_idx_to_dataset_idx(polling_strategy, dataset_remain_count):
        idx_to_dataset_idx = {}
        length = sum([v for v in dataset_remain_count.values()])
        if polling_strategy == 'batch_wise':
            idx = 0
            while idx < length:
                for dataset_idx in dataset_remain_count.keys():
                    if dataset_remain_count[dataset_idx] > 0:
                        idx_to_dataset_idx[idx] = dataset_idx
                        idx += 1
                        dataset_remain_count[dataset_idx] -= 1
        if polling_strategy == 'dataset_wise':
            idx = 0
            while idx < length:
                for dataset_idx in dataset_remain_count.keys():
                    while dataset_remain_count[dataset_idx] > 0:
                        idx_to_dataset_idx[idx] = dataset_idx
                        idx += 1
                        dataset_remain_count[dataset_idx] -= 1

        return idx_to_dataset_idx

    def __next__(self):
        if self.count == self._len:
            raise StopIteration

        cur_dataset_idx = self.idx_to_dataset_idx[self.count]
        self.count += 1
        return cur_dataset_idx, self.iter_dataloaders[cur_dataset_idx].next()


if __name__ == '__main__':
    print("===> batch_wise\n", ConcatDataloader.generate_idx_to_dataset_idx('batch_wise', {0: 8, 1: 10, 2: 7}))
    print("===> dataset_wise\n", ConcatDataloader.generate_idx_to_dataset_idx('dataset_wise', {0: 8, 1: 10, 2: 7}))
