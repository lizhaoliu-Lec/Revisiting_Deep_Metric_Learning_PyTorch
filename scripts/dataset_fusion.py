import os
import time
import numpy as np
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import architectures as archs
import batchminer as bmine
import criteria as criteria
import metric
from utilities import logger, misc
import dataset
from scripts._share import set_seed, get_dataset_fusion_dataloaders, \
    train_dataset_fusion_one_epoch, evaluate_dataset_fusion, normalize_image, train_dataset_fusion_one_epoch_wo_feature
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
    name_to_dataloaders, name_to_datasampler = get_dataset_fusion_dataloaders(opt, model)

    # CREATE LOGGING FILES
    sub_loggers = ['Train', 'Test', 'Model Grad']
    if opt.use_tv_split:
        sub_loggers.append('Val')

    name_to_log = {}

    opt.save_path = opt.save_path + '/' + '-'.join(
        opt.feature_datasets) + '/' + opt.arch.upper() + '-' + get_time_string()
    for dataset_name in name_to_dataloaders.keys():
        if dataset_name != 'joint_training':
            opt.savename = '{}_{}'.format(dataset_name, opt.arch.upper())
            name_to_log[dataset_name] = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True,
                                                      log_online=opt.log_online)

    # LOSS SETUP
    batchminer = bmine.select(opt.batch_mining, opt)
    name_to_criterion = {}
    for dataset_name in name_to_dataloaders.keys():
        if dataset_name != 'joint_training':
            opt.dataset = dataset_name
            opt.n_classes = len(name_to_dataloaders[dataset_name]['training'].dataset.avail_classes)
            criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)
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

        if not opt.feature_not_used:
            train_dataset_fusion_one_epoch(opt, epoch, scheduler, name_to_datasampler,
                                           name_to_dataloaders['joint_training'],
                                           model, name_to_criterion, optimizer, name_to_log)
        else:
            train_dataset_fusion_one_epoch_wo_feature(opt, epoch, scheduler, name_to_datasampler,
                                                      name_to_dataloaders['joint_training'],
                                                      model, name_to_criterion, optimizer, name_to_log)
        evaluate_dataset_fusion(opt, epoch, model, name_to_dataloaders, metric_computer, name_to_log)

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


def run_feature_dataset(par):
    from dataset import cub200, cars196
    from dataset.feature_dataset import FeatureDataset
    from dataset.concat_dataloader import ConcatDataloader

    print("====> type(par.feature_datasets) ", type(par.feature_datasets))
    print("====> par.feature_datasets ", par.feature_datasets)

    # par.dataset_idx = 0
    # datasets_cub200 = cub200.get_dataset(opt=par,
    #                                      data_path=par.source_path + '/' + par.feature_datasets[par.dataset_idx],
    #                                      TrainDatasetClass=FeatureDataset)
    #
    # par.dataset_idx = 1
    # datasets_cars196 = cars196.get_dataset(opt=par,
    #                                        data_path=par.source_path + '/' + par.feature_datasets[par.dataset_idx],
    #                                        TrainDatasetClass=FeatureDataset)
    # dataloader = ConcatDataloader([datasets_cub200['training'],
    #                                datasets_cars196['training']],
    #                               polling_strategy=par.polling_strategy,
    #                               num_workers=par.kernels, batch_size=par.bs,
    #                               shuffle=False, drop_last=False)
    # # dataloader.test_loader_len()
    # #
    # # dataloader.test_loader_next_long()
    #
    # # tqdm_dataloader = dataloader
    # for ttt in range(3):
    #     tqdm_dataloader = tqdm(dataloader)
    #     # tqdm_dataloader = dataloader
    #     import time
    #     t = time.time()
    #     print("====> starting for the %d times" % ttt)
    #     print("======> len(dataloader) ", len(dataloader))
    #     dataset_count = {0: 0, 1: 0}
    #     for i, data in enumerate(tqdm_dataloader):
    #         # print('\r %d / %d' % (i, len(dataloader)), end='')
    #         # for i, data in enumerate(dataloader):
    #         dataset_idx, (img_p, img_t, feat, img_id) = data
    #         dataset_count[dataset_idx] += 1
    #         a = 1 + 2
    #         # print("===> i", i)
    #         # print("===> dataset_idx", dataset_idx)
    #         # print("===> img_t", img_t.size(), img_t.device)
    #         # print("===> feat", feat.size(), feat.device)
    #         # print("===> img_id", img_id)
    #     print("====> dataset_count\n", dataset_count)
    #     print('=====> Using %s to loop over all dataloader' % str(time.time() - t))


@torch.no_grad()
def visualization(opt):
    # get pretrained model
    model = archs.select(opt.arch, opt)

    checkpoint = torch.load(opt.visualization_checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model.cuda()

    datasets = dataset.select(opt.dataset, opt, opt.source_path + '/' + opt.dataset)

    dataloader = DataLoader(datasets['evaluation'], num_workers=opt.kernels,
                            batch_size=opt.bs, shuffle=False)

    data_iterator = tqdm(dataloader, desc='Evaluating dataset `%s`...' % opt.dataset)

    features = []
    images = []

    for i, out in enumerate(data_iterator):

        class_labels, x, input_indices = out

        # Compute Embedding
        model_args = {'x': x.cuda()}
        # Needed for MixManifold settings.
        if 'mix' in opt.arch:
            model_args['labels'] = class_labels
        embeds = model(**model_args)
        if isinstance(embeds, tuple):
            embeds, (_, _) = embeds

        features.append(embeds.cpu())
        images.append(x.cpu())

        break

    features = torch.cat(features, dim=0).numpy()
    images = torch.cat(images, dim=0).numpy()

    save_dir = opt.visualization_save_dir

    index = 0
    for feature, image in tqdm(zip(features, images)):
        image_normed = normalize_image(image)

        plt.subplot(2, 1, 1)
        plt.imshow(image_normed)
        plt.axis('off')
        plt.title('image')
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        plt.bar(range(len(feature)), feature)
        plt.xlabel('feature index')
        plt.ylabel('feature value')
        plt.title('feature')
        plt.tight_layout()
        plt.savefig("%s/%d_overall.png" % (save_dir, index), dpi=300)

        plt.close()

        index += 1

        if index == 10:
            break
