import os
import time
import numpy as np
import torch.multiprocessing

import architectures as archs
import batchminer as bmine
import criteria as criteria
import metric
from utilities import logger
from utilities import misc
from scripts._share import set_seed, get_dataloaders, train_one_epoch, evaluate, FeaturePenalty


def main(opt):
    opt.source_path += '/' + opt.dataset
    opt.save_path += '/' + opt.dataset

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
    dataloaders, train_data_sampler = get_dataloaders(opt, model)

    opt.n_classes = len(dataloaders['training'].dataset.avail_classes)

    # CREATE LOGGING FILES
    sub_loggers = ['Train', 'Test', 'Model Grad']
    if opt.use_tv_split:
        sub_loggers.append('Val')
    LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=opt.log_online)

    # LOSS SETUP
    embed_dim = opt.embed_dim
    batchminer = bmine.select(opt.batch_mining, opt)
    criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)
    if opt.classifier_fusion_used:
        print("Using low dimensional classifier fusion!!!")
        # create low cls
        opt.embed_dim = opt.classifier_fusion_dim
        low_criterion, _ = criteria.select(opt.loss, opt, to_optim, batchminer)
        opt.embed_dim = embed_dim

        # load low cls
        print("Loading low dimensional classifier from %s " % opt.classifier_fusion_path[0])
        checkpoint = torch.load(opt.classifier_fusion_path[0])
        low_criterion.load_state_dict(checkpoint['state_dict'])

        criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)

        # merge low cls
        criterion.load_low_dimensional_classifier(low_criterion, 0, opt.classifier_fusion_dim,
                                                  not opt.classifier_fusion_not_freeze)
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
    feature_penalty = None
    if opt.feature_penalty_used:
        feature_penalty = FeaturePenalty(total_dimension=opt.embed_dim,
                                         total_epoch=opt.n_epochs,
                                         base=opt.feature_penalty_base,
                                         reverse=opt.feature_penalty_reversed,
                                         start_dimension=opt.feature_penalty_start_dimension,
                                         rescale=opt.feature_penalty_rescale,
                                         topK=opt.feature_penalty_topK)

    for epoch in range(opt.n_epochs):
        epoch_start_time = time.time()

        if epoch > 0 and opt.data_idx_full_prec and train_data_sampler.requires_storage:
            train_data_sampler.full_storage_update(dataloaders['evaluation'], model, opt.device)

        train_one_epoch(opt, epoch, scheduler, train_data_sampler, dataloaders['training'],
                        model, criterion, optimizer, LOG, feature_penalty=feature_penalty)
        evaluate(opt, epoch, model, dataloaders, metric_computer, LOG, criterion=criterion)

        print('Total Epoch Runtime: {0:4.2f}s'.format(time.time() - epoch_start_time))

    # CREATE A SUMMARY TEXT FILE
    summary_text = ''
    full_training_time = time.time() - full_training_start_time
    summary_text += 'Training Time: {} min.\n'.format(np.round(full_training_time / 60, 2))

    for sub_logger in LOG.sub_loggers:
        metrics = LOG.graph_writer[sub_logger].ov_title
        summary_text += '{} metric: {}\n'.format(sub_logger.upper(), metrics)

    with open(LOG.save_path + '/training_summary.txt', 'w') as summary_file:
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
