import random
import time
import numpy as np
import torch.multiprocessing
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

import datasampler as dsamplers
import dataset
import evaluation as eval
from dataset.concat_datalaoder_old import ConcatDataloader
from dataset.feature_dataset import FeatureDataset


def set_seed(seed):
    """
    set seed for reproducibility
    """
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(opt, model):
    dataloaders = {}
    datasets = dataset.select(opt.dataset, opt, opt.source_path)

    dataloaders['evaluation'] = DataLoader(datasets['evaluation'], num_workers=opt.kernels,
                                           batch_size=opt.bs, shuffle=False)
    dataloaders['testing'] = DataLoader(datasets['testing'], num_workers=opt.kernels,
                                        batch_size=opt.bs,
                                        shuffle=False)
    if opt.use_tv_split:
        dataloaders['validation'] = DataLoader(datasets['validation'], num_workers=opt.kernels,
                                               batch_size=opt.bs, shuffle=False)

    train_data_sampler = dsamplers.select(opt.data_sampler, opt, datasets['training'].image_dict,
                                          datasets['training'].image_list)
    if train_data_sampler.requires_storage:
        train_data_sampler.create_storage(dataloaders['evaluation'], model, opt.device)

    dataloaders['training'] = DataLoader(datasets['training'], num_workers=opt.kernels,
                                         batch_sampler=train_data_sampler)

    return dataloaders, train_data_sampler


def get_dataset_fusion_dataloaders(opt, model):
    name_to_dataloaders = {}
    training_sets = []
    training_samplers = []
    name_to_datasampler = {}
    for i, dataset_name in enumerate(opt.feature_datasets):
        opt.dataset_idx = i

        dataloaders = {}

        datasets = dataset.select(dataset_name, opt, opt.source_path + '/' + dataset_name,
                                  TrainDatasetClass=FeatureDataset)

        training_sets.append(datasets['training'])

        dataloaders['evaluation'] = DataLoader(datasets['evaluation'], num_workers=opt.kernels,
                                               batch_size=opt.bs, shuffle=False)
        dataloaders['testing'] = DataLoader(datasets['testing'], num_workers=opt.kernels,
                                            batch_size=opt.bs, shuffle=False)
        if opt.use_tv_split:
            dataloaders['validation'] = DataLoader(datasets['validation'], num_workers=opt.kernels,
                                                   batch_size=opt.bs, shuffle=False)

        train_data_sampler = dsamplers.select(opt.data_sampler, opt, datasets['training'].image_dict,
                                              datasets['training'].image_list)

        training_samplers.append(train_data_sampler)

        if train_data_sampler.requires_storage:
            train_data_sampler.create_storage(dataloaders['evaluation'], model, opt.device)

        dataloaders['training'] = DataLoader(datasets['training'], num_workers=opt.kernels,
                                             batch_sampler=train_data_sampler)

        name_to_dataloaders[dataset_name] = dataloaders
        name_to_datasampler[dataset_name] = train_data_sampler

    name_to_dataloaders['joint_training'] = ConcatDataloader(training_sets,
                                                             polling_strategy=opt.polling_strategy,
                                                             num_workers=opt.kernels,
                                                             batch_samplers=training_samplers)

    return name_to_dataloaders, name_to_datasampler


def train_one_epoch(opt, epoch, scheduler, train_data_sampler, dataloader, model, criterion, optimizer, LOG,
                    feature_penalty=None):
    opt.epoch = epoch
    # Scheduling Changes specifically for cosine scheduling
    if opt.scheduler != 'none':
        print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

    if train_data_sampler.requires_storage:
        train_data_sampler.precompute_indices()

    # Train one epoch
    start = time.time()
    _ = model.train()

    loss_collect = []
    data_iterator = tqdm(dataloader, desc='Epoch {}/{} Training...'.format(epoch, opt.n_epochs))

    loss_args = {'batch': None, 'labels': None, 'batch_features': None, 'f_embed': None}

    for i, out in enumerate(data_iterator):
        global_steps = epoch * len(data_iterator) + i

        class_labels, input, input_indices = out

        # Compute Embedding
        input = input.to(opt.device)
        model_args = {'x': input.to(opt.device)}
        # Needed for MixManifold settings.
        if 'mix' in opt.arch:
            model_args['labels'] = class_labels
        embeds = model(**model_args)
        if isinstance(embeds, tuple):
            embeds, (avg_features, features) = embeds
            # if feature_penalty is not None:
            #     features = feature_penalty(features, epoch)
            loss_args['batch_features'] = features

        if feature_penalty is not None:
            embeds = feature_penalty(embeds, epoch)

        # Compute Loss
        loss_args['batch'] = embeds
        loss_args['labels'] = class_labels
        loss_args['f_embed'] = model.model.last_linear
        loss = criterion(**loss_args)

        optimizer.zero_grad()
        loss.backward()

        # Compute Model Gradients and log them!
        grads = np.concatenate(
            [p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        grad_l2, grad_max = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
        LOG.progress_saver['Model Grad'].log('Grad L2', grad_l2, group='L2')
        LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')
        LOG.tensorboard.add_scalar(tag='Grad/L2', scalar_value=grad_l2, global_step=global_steps)
        LOG.tensorboard.add_scalar(tag='Grad/Max', scalar_value=grad_max, global_step=global_steps)

        # Update network weights!
        optimizer.step()

        loss_collect.append(loss.item())

        if i == len(dataloader) - 1:
            data_iterator.set_description(
                'Epoch (Train) {0}/{1}: Mean Loss [{2:.4f}]'.format(epoch, opt.n_epochs, np.mean(loss_collect)))

        # A brilliant way to update embeddings!
        if train_data_sampler.requires_storage and train_data_sampler.update_storage:
            train_data_sampler.replace_storage_entries(embeds.detach().cpu(), input_indices)

    result_metrics = {'loss': np.mean(loss_collect)}

    LOG.progress_saver['Train'].log('epochs', epoch)
    for metric_name, metric_val in result_metrics.items():
        LOG.progress_saver['Train'].log(metric_name, metric_val)
        LOG.tensorboard.add_scalar(tag='Train/%s' % metric_name, scalar_value=metric_val, global_step=epoch)
    LOG.progress_saver['Train'].log('time', np.round(time.time() - start, 4))
    LOG.tensorboard.add_scalar(tag='Train/time', scalar_value=np.round(time.time() - start, 4), global_step=epoch)

    # Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()


def train_dataset_fusion_one_epoch(opt, epoch, scheduler, name_to_datasampler, dataloader, model, name_to_criteria,
                                   optimizer, name_to_log):
    opt.epoch = epoch
    # Scheduling Changes specifically for cosine scheduling
    if opt.scheduler != 'none':
        print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

    for train_data_sampler in name_to_datasampler.values():
        if train_data_sampler.requires_storage:
            train_data_sampler.precompute_indices()

    # Train one epoch
    start = time.time()
    model.train()

    # loss_collect = []
    dataset_idx_to_loss_collect = {i: [] for i in range(len(opt.feature_datasets))}
    dataset_idx_to_feature_loss_collect = {i: [] for i in range(len(opt.feature_datasets))}
    dataset_idx_to_total_loss_collect = {i: [] for i in range(len(opt.feature_datasets))}
    data_iterator = tqdm(dataloader, desc='Epoch {}/{} Training...'.format(epoch, opt.n_epochs))

    name_to_loss_arg = {}
    for dataset_name in opt.feature_datasets:
        name_to_loss_arg[dataset_name] = {'batch': None, 'labels': None, 'batch_features': None, 'f_embed': None}

    dataset_idx_to_dataset_name = {i: n for i, n in enumerate(opt.feature_datasets)}
    assert len(opt.feature_datasets) * 2 == len(opt.feature_indexes)
    dataset_idx_to_feature_indexes = {i: (opt.feature_indexes[i * 2], opt.feature_indexes[i * 2 + 1])
                                      for i in range(len(opt.feature_datasets))}
    feature_lambda = opt.feature_lambda

    for i, out in enumerate(data_iterator):
        global_steps = epoch * len(data_iterator) + i

        dataset_idx, (class_labels, input, low_dim_features, input_indices) = out
        dataset_name = dataset_idx_to_dataset_name[dataset_idx]

        loss_args = name_to_loss_arg[dataset_name]
        LOG = name_to_log[dataset_name]
        train_data_sampler = name_to_datasampler[dataset_name]
        criterion = name_to_criteria[dataset_name]

        # Compute Embedding
        input = input.to(opt.device)
        model_args = {'x': input.to(opt.device)}
        # Needed for MixManifold settings.
        if 'mix' in opt.arch:
            model_args['labels'] = class_labels
        embeds = model(**model_args)
        if isinstance(embeds, tuple):
            embeds, (avg_features, features) = embeds
            loss_args['batch_features'] = features

        # Compute Loss
        loss_args['batch'] = embeds
        loss_args['labels'] = class_labels
        loss_args['f_embed'] = model.model.last_linear
        loss = criterion(**loss_args)

        # calculate L2 loss here
        start_index, end_index = dataset_idx_to_feature_indexes[dataset_idx]
        low_dim_embeds = embeds[:, start_index:end_index]
        # print("====> start_index, end_index ", start_index, end_index)
        # print("====> low_dim_embeds.size() ", low_dim_embeds.size())
        # print("====> low_dim_features.size() ", low_dim_features.size())
        feature_loss = torch.nn.MSELoss()(low_dim_embeds, low_dim_features.to(opt.device))

        total_loss = loss + feature_loss * feature_lambda

        optimizer.zero_grad()
        total_loss.backward()

        # Compute Model Gradients and log them!
        grads = np.concatenate(
            [p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        grad_l2, grad_max = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
        LOG.progress_saver['Model Grad'].log('Grad L2', grad_l2, group='L2')
        LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')
        LOG.tensorboard.add_scalar(tag='Grad/L2', scalar_value=grad_l2, global_step=global_steps)
        LOG.tensorboard.add_scalar(tag='Grad/Max', scalar_value=grad_max, global_step=global_steps)

        # Update network weights!
        optimizer.step()

        dataset_idx_to_loss_collect[dataset_idx].append(loss.item())
        dataset_idx_to_feature_loss_collect[dataset_idx].append(feature_loss.item())
        dataset_idx_to_total_loss_collect[dataset_idx].append(total_loss.item())

        if i == len(dataloader) - 1:
            data_iterator.set_description(
                'Epoch (Train: {5}) {0}/{1}: Mean Loss [{2:.4f}], '
                'Mean Feature Loss [{3:.4f}], Mean Total Loss [{4:.4f}]'.format(
                    epoch, opt.n_epochs,
                    np.mean(dataset_idx_to_loss_collect[dataset_idx]),
                    np.mean(dataset_idx_to_feature_loss_collect[dataset_idx]),
                    np.mean(dataset_idx_to_total_loss_collect[dataset_idx]),
                    dataset_idx_to_dataset_name[dataset_idx]))

        # A brilliant way to update embeddings!
        if train_data_sampler.requires_storage and train_data_sampler.update_storage:
            train_data_sampler.replace_storage_entries(embeds.detach().cpu(), input_indices)

    for dataset_idx, dataset_name in dataset_idx_to_dataset_name.items():

        result_metrics = {
            'loss': np.mean(dataset_idx_to_loss_collect[dataset_idx]),
            'feature_loss': np.mean(dataset_idx_to_feature_loss_collect[dataset_idx]),
            'total_loss': np.mean(dataset_idx_to_total_loss_collect[dataset_idx])
        }

        LOG = name_to_log[dataset_name]

        LOG.progress_saver['Train'].log('epochs', epoch)
        for metric_name, metric_val in result_metrics.items():
            LOG.progress_saver['Train'].log(metric_name, metric_val)
            LOG.tensorboard.add_scalar(tag='Train/%s' % metric_name, scalar_value=metric_val, global_step=epoch)
        LOG.progress_saver['Train'].log('time', np.round(time.time() - start, 4))
        LOG.tensorboard.add_scalar(tag='Train/time', scalar_value=np.round(time.time() - start, 4), global_step=epoch)

    # Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()


def train_dataset_fusion_one_epoch_wo_feature(opt, epoch, scheduler, name_to_datasampler, dataloader, model,
                                              name_to_criteria,
                                              optimizer, name_to_log):
    opt.epoch = epoch
    # Scheduling Changes specifically for cosine scheduling
    if opt.scheduler != 'none':
        print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

    for train_data_sampler in name_to_datasampler.values():
        if train_data_sampler.requires_storage:
            train_data_sampler.precompute_indices()

    # Train one epoch
    start = time.time()
    model.train()

    # loss_collect = []
    dataset_idx_to_loss_collect = {i: [] for i in range(len(opt.feature_datasets))}
    data_iterator = tqdm(dataloader, desc='Epoch {}/{} Training...'.format(epoch, opt.n_epochs))

    name_to_loss_arg = {}
    for dataset_name in opt.feature_datasets:
        name_to_loss_arg[dataset_name] = {'batch': None, 'labels': None, 'batch_features': None, 'f_embed': None}

    dataset_idx_to_dataset_name = {i: n for i, n in enumerate(opt.feature_datasets)}
    assert len(opt.feature_datasets) * 2 == len(opt.feature_indexes)
    feature_lambda = opt.feature_lambda

    for i, out in enumerate(data_iterator):
        global_steps = epoch * len(data_iterator) + i

        dataset_idx, (class_labels, input, low_dim_features, input_indices) = out
        dataset_name = dataset_idx_to_dataset_name[dataset_idx]

        loss_args = name_to_loss_arg[dataset_name]
        LOG = name_to_log[dataset_name]
        train_data_sampler = name_to_datasampler[dataset_name]
        criterion = name_to_criteria[dataset_name]

        # Compute Embedding
        input = input.to(opt.device)
        model_args = {'x': input.to(opt.device)}
        # Needed for MixManifold settings.
        if 'mix' in opt.arch:
            model_args['labels'] = class_labels
        embeds = model(**model_args)
        if isinstance(embeds, tuple):
            embeds, (avg_features, features) = embeds
            loss_args['batch_features'] = features

        # Compute Loss
        loss_args['batch'] = embeds
        loss_args['labels'] = class_labels
        loss_args['f_embed'] = model.model.last_linear
        loss = criterion(**loss_args)

        optimizer.zero_grad()
        loss.backward()

        # Compute Model Gradients and log them!
        grads = np.concatenate(
            [p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        grad_l2, grad_max = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
        LOG.progress_saver['Model Grad'].log('Grad L2', grad_l2, group='L2')
        LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')
        LOG.tensorboard.add_scalar(tag='Grad/L2', scalar_value=grad_l2, global_step=global_steps)
        LOG.tensorboard.add_scalar(tag='Grad/Max', scalar_value=grad_max, global_step=global_steps)

        # Update network weights!
        optimizer.step()

        dataset_idx_to_loss_collect[dataset_idx].append(loss.item())

        if i == len(dataloader) - 1:
            data_iterator.set_description(
                'Epoch (Train: {3}) {0}/{1}: Mean Loss [{2:.4f}]'.format(
                    epoch, opt.n_epochs,
                    np.mean(dataset_idx_to_loss_collect[dataset_idx]),
                    dataset_idx_to_dataset_name[dataset_idx]))

        # A brilliant way to update embeddings!
        if train_data_sampler.requires_storage and train_data_sampler.update_storage:
            train_data_sampler.replace_storage_entries(embeds.detach().cpu(), input_indices)

    for dataset_idx, dataset_name in dataset_idx_to_dataset_name.items():

        result_metrics = {
            'loss': np.mean(dataset_idx_to_loss_collect[dataset_idx]),
        }

        LOG = name_to_log[dataset_name]

        LOG.progress_saver['Train'].log('epochs', epoch)
        for metric_name, metric_val in result_metrics.items():
            LOG.progress_saver['Train'].log(metric_name, metric_val)
            LOG.tensorboard.add_scalar(tag='Train/%s' % metric_name, scalar_value=metric_val, global_step=epoch)
        LOG.progress_saver['Train'].log('time', np.round(time.time() - start, 4))
        LOG.tensorboard.add_scalar(tag='Train/time', scalar_value=np.round(time.time() - start, 4), global_step=epoch)

    # Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()


@torch.no_grad()
def evaluate_dataset_fusion(opt, epoch, model, name_to_dataloaders, metric_computer, name_to_log):
    for dataset_name in opt.feature_datasets:
        print('\nEvaluating %s' % dataset_name, end='')
        evaluate(opt, epoch, model, name_to_dataloaders[dataset_name], metric_computer, name_to_log[dataset_name])


@torch.no_grad()
def evaluate(opt, epoch, model, dataloaders, metric_computer, LOG):
    # Evaluate Metric for Training & Test (& Validation)
    model.eval()
    print('\nEpoch {0}/{1} Computing Testing Metrics...'.format(epoch, opt.n_epochs))
    eval.evaluate(LOG, metric_computer, dataloaders['testing'], model, opt, opt.eval_types,
                  opt.device, log_key='Test')
    if opt.use_tv_split:
        print('\nEpoch {0}/{1} Computing Validation Metrics...'.format(epoch, opt.n_epochs))
        eval.evaluate(LOG, metric_computer, dataloaders['validation'], model, opt, opt.eval_types,
                      opt.device, log_key='Val')
    print('\nEpoch {0}/{1} Computing Training Metrics...'.format(epoch, opt.n_epochs))
    eval.evaluate(LOG, metric_computer, dataloaders['evaluation'], model, opt, opt.eval_types,
                  opt.device, log_key='Train')

    LOG.update(update_all=True)


def normalize_image(x):
    x = np.transpose(x, (1, 2, 0))
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return (x * 255).astype(np.uint8)


class FeaturePenalty(nn.Module):
    def __init__(self, total_dimension, total_epoch, reverse=False, base=8,
                 start_dimension=None, rescale=False, topK=False):
        assert total_dimension % base == 0, 'total_dimension must be totally split by base'
        if start_dimension is not None:
            assert total_dimension >= start_dimension, 'start_dimension must be smaller or equal to total_dimension'

        super(FeaturePenalty, self).__init__()
        self.total_dimension = total_dimension

        self.total_epoch = total_epoch
        self.base = base
        self.reverse = reverse

        self.start_dimension = start_dimension if start_dimension is not None else self.base
        self.rescale = rescale
        self.topK = topK

        self.encountered_epoch = []

        print(self)

        self.batchIdToEndIndex = self.get_batchIdToEndIndex(total_dimension, total_epoch, base,
                                                            reverse=reverse,
                                                            start_dimension=self.start_dimension)

    @staticmethod
    def get_batchIdToEndIndex(total_dimension, total_epoch, base, reverse=False, start_dimension=None):
        # example dim: 64, epoch: 150, base: 8
        batchIdToEndIndex = {}
        start_dimension = base if start_dimension is None else start_dimension
        # print("=====> start_dimension ", start_dimension)
        num_expand = (total_dimension - start_dimension) // base
        to_expand = total_epoch // num_expand
        # print("====> num_expand ", num_expand)
        # print("====> to_expand ", to_expand)
        dim = start_dimension
        for epoch in range(total_epoch):
            # if epoch % to_expand == 0 and epoch != 0 and dim < total_dimension:
            if epoch % to_expand == 0 and epoch != 0:
                # print("====> epoch: ", epoch)
                dim += base
            if not reverse:
                batchIdToEndIndex[epoch] = dim
            else:
                batchIdToEndIndex[total_epoch - epoch - 1] = dim

        # print("=====> batchIdToEndIndex:\n", batchIdToEndIndex)

        return batchIdToEndIndex

    @staticmethod
    def mask_feature_by_endIndex(batch_feature, end_index, scale_factor=1.0, topK=False):
        # batch_feature[:, end_index:] = batch_feature[:, end_index:] * 0

        batch_size, feature_size = batch_feature.size()

        if not topK:
            batch_ones = torch.ones((batch_size, end_index), device=batch_feature.device, requires_grad=False)
            batch_zeros = torch.zeros((batch_size, feature_size - end_index), device=batch_feature.device,
                                      requires_grad=False)
            # print("====> batch_ones.size() ", batch_ones.size())
            # print("====> batch_zeros.size() ", batch_zeros.size())

            batch_masks = torch.cat([batch_ones, batch_zeros], dim=1)
        else:
            topK_indices = torch.topk(torch.abs(batch_feature), end_index, dim=1, sorted=False)[1]
            # print("===> topK_indices.size() ", topK_indices.size())
            # print("===> topK_indices ", topK_indices)
            rang = torch.arange(batch_feature.size(0)).reshape((-1, 1)).to(batch_feature.device)
            batch_masks = torch.zeros_like(batch_feature)
            batch_masks[rang, topK_indices] = 1.0
            # print("====> batch_masks\n", batch_masks)

        # print("====> batch_masks.size() ", batch_masks.size())
        return scale_factor * batch_feature * batch_masks

    def forward(self, batch_feature, epoch):
        # print("====> epoch ", epoch)
        if epoch not in self.encountered_epoch:
            self.encountered_epoch.append(epoch)
            if 0 < epoch < len(self.batchIdToEndIndex) - 1 and \
                    self.batchIdToEndIndex[epoch - 1] != self.batchIdToEndIndex[epoch]:
                print("[FeaturePenalty] Changing feature_dim from %d to %d for epoch %d" % (
                    self.batchIdToEndIndex[epoch - 1],
                    self.batchIdToEndIndex[epoch], epoch))
            else:
                print("[FeaturePenalty] Using feature_dim %d for epoch %d" % (self.batchIdToEndIndex[epoch], epoch))

        scale_factor = 1.0 if not self.rescale else self.total_dimension / self.batchIdToEndIndex[epoch]
        return self.mask_feature_by_endIndex(batch_feature, self.batchIdToEndIndex[epoch],
                                             scale_factor, topK=self.topK)

    def __repr__(self):
        return 'FeaturePenalty(' \
               'total_dimension={total_dimension}, start_dimension={start_dimension}, ' \
               'base={base}, reverse={reverse}, rescale={rescale}, topK={topK})'.format(**self.__dict__)


if __name__ == '__main__':
    def run_feature_penalty():
        import torch

        total_dimension, total_epoch, base, start_dimension = 128, 150, 8, 64

        feature_penalty = FeaturePenalty(total_dimension=total_dimension, total_epoch=total_epoch, base=base,
                                         reverse=True, start_dimension=start_dimension, rescale=True, topK=True)

        for epoch in range(1, total_epoch):
            batch_features = torch.randn((1, total_dimension))
            batch_features = feature_penalty(batch_features, epoch)
            print("====> batch_features ", batch_features)


    def run_batchIdToEndIndex():
        # FeaturePenalty.get_batchIdToEndIndex(64, 150, 8, reverse=True)
        FeaturePenalty.get_batchIdToEndIndex(32, 150, 2, reverse=True, start_dimension=8)


    def plot_feature_penalty():
        total_dimension, total_epoch, base, start_dimension = 128, 150, 8, None
        feature_penalty_1 = FeaturePenalty(total_dimension=total_dimension, total_epoch=total_epoch, base=base,
                                           reverse=True, start_dimension=start_dimension)
        total_dimension, total_epoch, base, start_dimension = 128, 150, 8, 64
        feature_penalty_2 = FeaturePenalty(total_dimension=total_dimension, total_epoch=total_epoch, base=base,
                                           reverse=True, start_dimension=start_dimension)
        total_dimension, total_epoch, base, start_dimension = 128, 150, 16, 64
        feature_penalty_3 = FeaturePenalty(total_dimension=total_dimension, total_epoch=total_epoch, base=base,
                                           reverse=True, start_dimension=start_dimension)
        import matplotlib.pyplot as plt
        x = []
        y1 = []
        y2 = []
        y3 = []
        for i in range(total_epoch):
            x.append(i)
            y1.append(feature_penalty_1.batchIdToEndIndex[i])
            y2.append(feature_penalty_2.batchIdToEndIndex[i])
            y3.append(feature_penalty_3.batchIdToEndIndex[i])

        plt.plot(x, y1, label='old')
        plt.plot(x, y2, label='new')
        plt.plot(x, y3, label='to run')
        plt.xlabel('epoch')
        plt.ylabel('dim')
        plt.legend()
        plt.show()

    def run_topK():
        data = [[3, 5, 1, 7], [3, 8, 4, 9]]
        tensor = torch.tensor(data)
        ind = torch.topk(tensor, 2, dim=1)[1]
        print("===> ind: ", ind)
        rang = torch.arange(tensor.size(0)).reshape((-1, 1))
        print("===> rang: ", rang)
        ret = tensor[[[0], [1]], ind]
        print("===> ret: ", ret)
        tensor[[[0], [1]], ind] = -1000
        print("===> tensor: ", tensor)
        # ret = tensor[0, ind[0]]
        # print("===> ret: ", ret)
        # ret = tensor[1, ind[1]]
        # print("===> ret: ", ret)
        # tensor[1, ind[1]] = -1000000
        # ret = tensor[1, ind[1]]
        # print("===> ret: ", ret)
        # print("===> tensor: ", tensor)

    run_feature_penalty()
    # run_batchIdToEndIndex()
    # plot_feature_penalty()
    # run_topK()
