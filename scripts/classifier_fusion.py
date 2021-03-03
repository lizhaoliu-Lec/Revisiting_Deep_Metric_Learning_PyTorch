import os
import time
import numpy as np
import torch.multiprocessing

import architectures as archs
import batchminer as bmine
import criteria as criteria
import metric
from utilities import logger, misc
from scripts._share import set_seed, get_dataset_fusion_dataloaders, evaluate_dataset_fusion, \
    train_dataset_fusion_one_epoch_wo_feature
from utilities.logger import get_time_string


def main(opt):
    # The following setting is useful when logging to wandb and running multiple seeds per setup:
    # By setting the savename to <group_plus_seed>, the savename will instead comprise the group and the seed!
    if opt.savename == 'group_plus_seed':
        if opt.log_online:
            opt.savename = opt.group + '_s{}'.format(opt.seed)
        else:
            opt.savename = ''

    full_training_start_time = time.time()

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
    name_to_dataloaders, name_to_datasampler = get_dataset_fusion_dataloaders(opt, model,
                                                                              opt.classifier_fusion_datasets)

    # CREATE LOGGING FILES
    sub_loggers = ['Train', 'Test', 'Model Grad']
    if opt.use_tv_split:
        sub_loggers.append('Val')

    name_to_log = {}

    opt.save_path = opt.save_path + '/' + '-'.join(
        opt.classifier_fusion_datasets) + '/' + opt.arch.upper() + '-' + get_time_string()
    for dataset_name in name_to_dataloaders.keys():
        if dataset_name != 'joint_training':
            opt.savename = '{}_{}'.format(dataset_name, opt.arch.upper())
            name_to_log[dataset_name] = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True,
                                                      log_online=opt.log_online)

    # dataset name to index
    dataset_name_2_index = {}
    for index in range(len(opt.classifier_fusion_datasets)):
        dataset_name_2_index[opt.classifier_fusion_datasets[index]] = index

    dataset_idx_to_cls_indexes = {i: (opt.classifier_fusion_indexes[i * 2],
                                      opt.classifier_fusion_indexes[i * 2 + 1])
                                  for i in range(len(opt.feature_datasets))}

    # LOSS SETUP
    batchminer = bmine.select(opt.batch_mining, opt)
    name_to_criterion = {}
    embed_dim = opt.embed_dim
    for dataset_name in name_to_dataloaders.keys():
        if dataset_name != 'joint_training':
            opt.dataset = dataset_name
            opt.n_classes = len(name_to_dataloaders[dataset_name]['training'].dataset.avail_classes)

            # create low cls
            opt.embed_dim = opt.classifier_fusion_dim
            low_criterion, _ = criteria.select(opt.loss, opt, to_optim, batchminer)
            opt.embed_dim = embed_dim

            # load low cls
            checkpoint = torch.load(opt.classifier_fusion_path[dataset_name_2_index[dataset_name]])
            low_criterion.load_state_dict(checkpoint['state_dict'])

            criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)

            # merge low cls
            start_index, end_index = dataset_idx_to_cls_indexes[dataset_name_2_index[dataset_name]]
            criterion.load_low_dimensional_classifier(low_criterion, start_index, end_index,
                                                      not opt.classifier_fusion_not_freeze)

            criterion.to(opt.device)
            name_to_criterion[dataset_name] = criterion

    for dataset_name in name_to_datasampler.keys():
        train_data_sampler = name_to_datasampler[dataset_name]
        if 'criterion' in train_data_sampler.name:
            train_data_sampler.internal_criterion = name_to_criterion[dataset_name]

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
    miner_text = ''
    for dataset_name in name_to_criterion.keys():
        miner_text = 'Batchminer:\t {}\n'.format(opt.batch_mining
                                                 if name_to_criterion[dataset_name].REQUIRES_BATCHMINER else 'N/A')
    arch_text = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))
    summary = data_text + '\n' + setup_text + '\n' + miner_text + '\n' + arch_text
    print(summary)

    # SCRIPT MAIN

    for epoch in range(opt.n_epochs):
        epoch_start_time = time.time()

        if epoch > 0 and opt.data_idx_full_prec:
            for dataset_name in name_to_datasampler.keys():
                train_data_sampler = name_to_datasampler[dataset_name]
                if train_data_sampler.requires_storage:
                    train_data_sampler.full_storage_update(name_to_datasampler[dataset_name]['evaluation'], model,
                                                           opt.device)

        train_dataset_fusion_one_epoch_wo_feature(opt, epoch, scheduler, name_to_datasampler,
                                                  name_to_dataloaders['joint_training'],
                                                  model, name_to_criterion, optimizer, name_to_log,
                                                  datasets=opt.classifier_fusion_datasets)
        evaluate_dataset_fusion(opt, epoch, model, name_to_dataloaders, metric_computer, name_to_log,
                                datasets=opt.classifier_fusion_datasets)

        print('Total Epoch Runtime: {0:4.2f}s'.format(time.time() - epoch_start_time))

    for dataset_name, LOG in name_to_log.items():
        # CREATE A SUMMARY TEXT FILE
        summary_text = ''
        full_training_time = time.time() - full_training_start_time
        summary_text += 'Training Time: {} min.\n'.format(np.round(full_training_time / 60, 2))

        for sub_logger in LOG.sub_loggers:
            metrics = LOG.graph_writer[sub_logger].ov_title
            summary_text += '{} metric: {}\n'.format(sub_logger.upper(), metrics)

        with open(opt.save_path + '/%s_training_summary.txt' % dataset_name, 'w') as summary_file:
            summary_file.write(summary_text)
