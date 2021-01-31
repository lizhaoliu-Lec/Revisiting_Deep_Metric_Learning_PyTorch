import argparse
import os
import random
import time
import warnings

import matplotlib
import numpy as np
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

import architectures as archs
import batchminer as bmine
import criteria as criteria
import datasampler as dsamplers
import dataset
import evaluation as eval
import metric
import parameters as par
from utilities import logger
from utilities import misc

torch.multiprocessing.set_sharing_strategy('file_system')

matplotlib.use('agg')
warnings.filterwarnings("ignore")


def read_arguments_from_cmd():
    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()

    parser = par.get_basic_training_parameters(parser)
    parser = par.get_batch_creation_parameters(parser)
    parser = par.get_batchmining_specific_parameters(parser)
    parser = par.get_loss_specific_parameters(parser)
    parser = par.get_wandb_parameters(parser)

    # Read in parameters
    opt = parser.parse_args()
    return opt


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


def main():
    # get argument from cmd
    opt = read_arguments_from_cmd()

    # The following setting is useful when logging to wandb and running multiple seeds per setup:
    # By setting the savename to <group_plus_seed>, the savename will instead comprise the group and the seed!
    if opt.savename == 'group_plus_seed':
        if opt.log_online:
            opt.savename = opt.group + '_s{}'.format(opt.seed)
        else:
            opt.savename = ''

    # If wandb-logging is turned on, initialize the wandb-run here:
    if opt.log_online:
        import wandb

        _ = os.system('wandb login {}'.format(opt.wandb_key))
        os.environ['WANDB_API_KEY'] = opt.wandb_key
        wandb.init(project=opt.project, group=opt.group, name=opt.savename, dir=opt.save_path)
        wandb.config.update(opt)

    # Load Remaining Libraries that neeed to be loaded after comet_ml

    full_training_start_time = time.time()

    opt.source_path += '/' + opt.dataset
    opt.save_path += '/' + opt.dataset

    # Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
    assert not opt.bs % opt.samples_per_class, \
        'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

    opt.pretrained = not opt.not_pretrained

    # GPU SETTINGS
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # if not opt.use_data_parallel:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu[0])

    # SEEDS FOR REPRODUCIBILITY.
    set_seed(opt.seed)

    # NETWORK SETUP
    opt.device = torch.device('cuda')
    model = archs.select(opt.arch, opt)

    if opt.fc_lr < 0:
        to_optim = [{'params': model.parameters(), 'lr': opt.lr, 'weight_decay': opt.decay}]
    else:
        all_but_fc_params = [x[-1] for x in list(filter(lambda x: 'last_linear' not in x[0], model.named_parameters()))]
        fc_params = model.model.last_linear.parameters()
        to_optim = [{'params': all_but_fc_params, 'lr': opt.lr, 'weight_decay': opt.decay},
                    {'params': fc_params, 'lr': opt.fc_lr, 'weight_decay': opt.decay}]

    model.to(opt.device)

    # DATALOADER SETUPS
    dataloaders, train_data_sampler = get_dataloaders(opt, model)

    opt.n_classes = len(dataloaders['training'].dataset.avail_classes)

    # CREATE LOGGING FILES
    sub_loggers = ['Train', 'Test', 'Model Grad']
    if opt.use_tv_split:
        sub_loggers.append('Val')
    LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=opt.log_online)

    # LOSS SETUP
    batchminer = bmine.select(opt.batch_mining, opt)
    criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)
    criterion.to(opt.device)

    if 'criterion' in train_data_sampler.name:
        train_data_sampler.internal_criterion = criterion

    # OPTIM SETUP
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(to_optim)
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(to_optim, momentum=0.9)
    else:
        raise Exception('Optimizer <{}> not available!'.format(opt.optim))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)

    # METRIC COMPUTER
    opt.rho_spectrum_embed_dim = opt.embed_dim
    metric_computer = metric.MetricComputer(opt.evaluation_metrics, opt)

    # Summary
    data_text = 'Dataset:\t {}'.format(opt.dataset.upper())
    setup_text = 'Objective:\t {}'.format(opt.loss.upper())
    miner_text = 'Batchminer:\t {}'.format(opt.batch_mining if criterion.REQUIRES_BATCHMINER else 'N/A')
    arch_text = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))
    summary = data_text + '\n' + setup_text + '\n' + miner_text + '\n' + arch_text
    print(summary)

    # SCRIPT MAIN

    loss_args = {'batch': None, 'labels': None, 'batch_features': None, 'f_embed': None}

    for epoch in range(opt.n_epochs):
        epoch_start_time = time.time()

        if epoch > 0 and opt.data_idx_full_prec and train_data_sampler.requires_storage:
            train_data_sampler.full_storage_update(dataloaders['evaluation'], model, opt.device)

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
        data_iterator = tqdm(dataloaders['training'], desc='Epoch {} Training...'.format(epoch))

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

            loss_collect.append(loss.item())

            if i == len(dataloaders['training']) - 1:
                data_iterator.set_description(
                    'Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect)))

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

        # Evaluate Metric for Training & Test (& Validation)
        model.eval()
        print('\nComputing Testing Metrics...')
        eval.evaluate(LOG, metric_computer, [dataloaders['testing']], model, opt, opt.eval_types,
                      opt.device, log_key='Test')
        if opt.use_tv_split:
            print('\nComputing Validation Metrics...')
            eval.evaluate(LOG, metric_computer, [dataloaders['validation']], model, opt, opt.eval_types,
                          opt.device, log_key='Val')
        print('\nComputing Training Metrics...')
        eval.evaluate(LOG, metric_computer, [dataloaders['evaluation']], model, opt, opt.eval_types,
                      opt.device, log_key='Train')

        LOG.update(update_all=True)

        # Learning Rate Scheduling Step
        if opt.scheduler != 'none':
            scheduler.step()

        print('Total Epoch Runtime: {0:4.2f}s'.format(time.time() - epoch_start_time))

    # CREATE A SUMMARY TEXT FILE
    summary_text = ''
    full_training_time = time.time() - full_training_start_time
    summary_text += 'Training Time: {} min.\n'.format(np.round(full_training_time / 60, 2))

    for sub_logger in LOG.sub_loggers:
        metrics = LOG.graph_writer[sub_logger].ov_title
        summary_text += '{} metric: {}\n'.format(sub_logger.upper(), metrics)

    with open(opt.save_path + '/training_summary.txt', 'w') as summary_file:
        summary_file.write(summary_text)


def simple_test():
    from torchvision.models.resnet import resnet18
    model = resnet18().cuda()
    import torch
    x = torch.randn((5, 3, 224, 224)).cuda()
    y = torch.randint(0, 100, [5]).long().cuda()
    out = model(x)
    from torch.nn import CrossEntropyLoss
    criteria = CrossEntropyLoss()
    loss = criteria(out, y)
    loss.backward()


if __name__ == '__main__':
    main()
    # simple_test()
