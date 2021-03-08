import argparse
import os


def get_basic_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training_script', default='standard', type=str, help='Script used to train the model.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducibility.')
    return parser


def get_basic_training_parameters(parser):
    # Dataset-related Parameters
    parser.add_argument('--dataset', default='cub200', type=str, choices=['cub200', 'cars196', 'online_products'],
                        help='Dataset to use. Currently supported: cub200, cars196, online_products.')
    parser.add_argument('--use_tv_split', action='store_true',
                        help='Flag. If set, split the training set into a training/validation set.')
    parser.add_argument('--tv_split_by_samples', action='store_true',
                        help='Flag. If set, create the validation set by taking a percentage of samples PER class. '
                             'Otherwise, the validation set is create by taking a percentage of classes.')
    parser.add_argument('--tv_split_perc', default=0.8, type=float,
                        help='Percentage with which the training dataset is split into training/validation.')
    parser.add_argument('--augmentation', default='base', type=str, choices=['base', 'adv', 'big', 'red'],
                        help='Type of preprocessing/augmentation to use on the data. '
                             'Available: base (standard), adv (with color/brightness changes), '
                             'big (Images of size 256x256), red (No RandomResizedCrop).')

    # General Training Parameters
    parser.add_argument('--lr', default=0.00001, type=float, help='Learning Rate for network parameters.')
    parser.add_argument('--fc_lr', default=-1, type=float,
                        help='Optional. If not -1, sets the learning rate for the final linear embedding layer.')
    parser.add_argument('--decay', default=0.0004, type=float, help='Weight decay placed on network weights.')
    parser.add_argument('--n_epochs', default=150, type=int, help='Number of training epochs.')
    parser.add_argument('--kernels', default=6, type=int, help='Number of workers for pytorch dataloader.')
    parser.add_argument('--bs', default=112, type=int, help='Mini-Batchsize to use.')
    parser.add_argument('--scheduler', default='step', type=str,
                        help='Type of learning rate scheduling. Currently supported: step')
    parser.add_argument('--gamma', default=0.3, type=float, help='Learning rate reduction after tau epochs.')
    parser.add_argument('--tau', default=[1000], nargs='+', type=int, help='Stepsize before reducing learning rate.')

    # Loss-specific Settings
    parser.add_argument('--optim', default='adam', type=str, choices=['adam', 'sgd'],
                        help='Optimization method to use. Currently supported: adam & sgd.')
    parser.add_argument('--loss', default='softmax', type=str,
                        help='Training criteria: For supported methods, please check criteria/__init__.py')
    parser.add_argument('--batch_mining', default='distance', type=str,
                        help='Batchminer for tuple-based losses: '
                             'For supported methods, please check batch_mining/__init__.py')

    # Network-related Flags
    parser.add_argument('--embed_dim', default=128, type=int,
                        help='Embedding dimensionality of the network. '
                             'Note: dim = 64, 128 or 512 is used in most papers, depending on the architecture.')
    parser.add_argument('--not_pretrained', action='store_true',
                        help='Flag. If set, no ImageNet pretraining is used to initialize the network.')
    parser.add_argument('--arch', default='resnet50_frozen_normalize', type=str,
                        help='Underlying network architecture. '
                             'Frozen denotes that existing pretrained batchnorm layers are frozen, and '
                             'normalize denotes normalization of the output embedding.')

    # Evaluation Parameters
    parser.add_argument('--no_train_metrics', action='store_true',
                        help='Flag. If set, evaluation metric are not computed for the training data. '
                             'Saves a forward pass over the full training dataset.')
    parser.add_argument('--evaluate_on_gpu', action='store_true',
                        help='Flag. If set, all metric, when possible, are computed on the GPU (requires Faiss-GPU).')
    parser.add_argument('--evaluation_metrics', nargs='+',
                        default=['e_recall@1', 'e_recall@2', 'e_recall@4', 'nmi', 'f1', 'mAP_1000', 'mAP_lim', 'mAP_c',
                                 'dists@intra', 'dists@inter', 'dists@intra_over_inter', 'rho_spectrum@0',
                                 'rho_spectrum@-1', 'rho_spectrum@1', 'rho_spectrum@2', 'rho_spectrum@10'],
                        type=str, help='Metrics to evaluate performance by.')

    parser.add_argument('--storage_metrics', nargs='+', default=['e_recall@1'], type=str,
                        help='Improvement in these metric on a dataset trigger checkpointing.')
    parser.add_argument('--eval_types', nargs='+', default=['discriminative'], type=str,
                        help='The network may produce multiple embeddings (ModuleDict, relevant for e.g. DiVA). '
                             'If the key is listed here, the entry will be evaluated on the evaluation metric. '
                             'Note: One may use Combined [embed1, embed2, ..., embedn], '
                             '[w1, w2, ..., wn] to compute evaluation metric on weighted (normalized) combinations.')

    # Setup Parameters
    parser.add_argument('--gpu', default=[0], nargs='+', type=int, help='Gpu to use.')
    parser.add_argument('--savename', default='group_plus_seed', type=str,
                        help='Run savename - if default, '
                             'the savename will comprise the project and group name (see wandb_parameters()).')
    parser.add_argument('--source_path', default='<folder_path_that_stores_cub200_cars196_online_products>.', type=str,
                        help='Path to training data.')
    parser.add_argument('--save_path', default=os.getcwd() + '/TrainingResults', type=str,
                        help='Where to save everything.')

    return parser


def get_wandb_parameters(parser):
    # Online Logging/Wandb Log Arguments
    parser.add_argument('--log_online', action='store_true',
                        help='Flag. If set, run metric are stored online in addition to offline logging. '
                             'Should generally be set.')
    parser.add_argument('--wandb_key', default='<your_api_key_here>', type=str, help='API key for W&B.')
    parser.add_argument('--project', default='Sample_Project', type=str,
                        help='Name of the project - relates to W&B project names. '
                             'In --savename default setting part of the savename.')
    parser.add_argument('--group', default='Sample_Group', type=str,
                        help='Name of the group - relates to W&B group names '
                             '- all runs with same setup but different seeds are logged into one group. '
                             'In --savename default setting part of the savename.')
    return parser


def get_loss_specific_parameters(parser):
    # Contrastive Loss
    parser.add_argument('--loss_contrastive_pos_margin', default=0, type=float,
                        help='positive margin for contrastive pairs.')
    parser.add_argument('--loss_contrastive_neg_margin', default=1, type=float,
                        help='negative margin for contrastive pairs.')

    # Triplet-based Losses
    parser.add_argument('--loss_triplet_margin', default=0.2, type=float, help='Margin for Triplet Loss')

    # MarginLoss
    parser.add_argument('--loss_margin_margin', default=0.2, type=float, help='Triplet margin.')
    parser.add_argument('--loss_margin_beta_lr', default=0.0005, type=float,
                        help='Learning Rate for learnable class margin parameters in MarginLoss')
    parser.add_argument('--loss_margin_beta', default=1.2, type=float,
                        help='Initial Class Margin Parameter in Margin Loss')
    parser.add_argument('--loss_margin_nu', default=0, type=float,
                        help='Regularisation value on betas in Margin Loss. Generally not needed.')
    parser.add_argument('--loss_margin_beta_constant', action='store_true',
                        help='Flag. If set, beta-values are left untrained.')

    # ProxyNCA
    parser.add_argument('--loss_proxynca_lrmulti', default=50, type=float,
                        help='Learning Rate multiplier for Proxies in proxynca. '
                             'NOTE: The number of proxies is determined by the number of data classes')

    # NPair
    parser.add_argument('--loss_npair_l2', default=0.005, type=float,
                        help='L2 weight in NPair. '
                             'Note: Set to 0.02 in paper, but multiplied with 0.25 in their implementation.')

    # Angular Loss
    parser.add_argument('--loss_angular_alpha', default=45, type=float, help='Angular margin in degrees.')
    parser.add_argument('--loss_angular_npair_ang_weight', default=2, type=float,
                        help='Relative weighting between angular and npair contribution.')
    parser.add_argument('--loss_angular_npair_l2', default=0.005, type=float,
                        help='L2 weight on NPair (as embeddings are not normalized).')

    # Multisimilary Loss
    parser.add_argument('--loss_multisimilarity_pos_weight', default=2, type=float,
                        help='Weighting on positive similarities.')
    parser.add_argument('--loss_multisimilarity_neg_weight', default=40, type=float,
                        help='Weighting on negative similarities.')
    parser.add_argument('--loss_multisimilarity_margin', default=0.1, type=float,
                        help='Distance margin for both positive and negative similarities.')
    parser.add_argument('--loss_multisimilarity_thresh', default=0.5, type=float, help='Exponential thresholding.')

    # Lifted Structure Loss
    parser.add_argument('--loss_lifted_neg_margin', default=1, type=float, help='Margin placed on similarities.')
    parser.add_argument('--loss_lifted_l2', default=0.005, type=float,
                        help='As embeddings are not normalized, they need to be placed under penalty.')

    # Quadruplet Loss
    parser.add_argument('--loss_quadruplet_margin_alpha_1', default=0.2, type=float,
                        help='Quadruplet Loss requires two margins. This is the first one.')
    parser.add_argument('--loss_quadruplet_margin_alpha_2', default=0.2, type=float, help='This is the second.')

    # Soft-Triple Loss
    parser.add_argument('--loss_softtriplet_n_centroids', default=2, type=int, help='Number of proxies per class.')
    parser.add_argument('--loss_softtriplet_margin_delta', default=0.01, type=float,
                        help='Margin placed on sample-proxy similarities.')
    parser.add_argument('--loss_softtriplet_gamma', default=0.1, type=float,
                        help='Weight over sample-proxies within a class.')
    parser.add_argument('--loss_softtriplet_lambda', default=8, type=float, help='Serves as a temperature.')
    parser.add_argument('--loss_softtriplet_reg_weight', default=0.2, type=float,
                        help='Regularization weight on the number of proxies.')
    parser.add_argument('--loss_softtriplet_lrmulti', default=1, type=float,
                        help='Learning Rate multiplier for proxies.')

    # Normalized Softmax Loss
    parser.add_argument('--loss_softmax_lr', default=0.00001, type=float, help='Learning rate on class proxies.')
    parser.add_argument('--loss_softmax_temperature', default=0.05, type=float, help='Temperature for NCA objective.')

    # Histogram Loss
    parser.add_argument('--loss_histogram_nbins', default=65, type=int,
                        help='Number of bins for histogram discretization.')

    # SNR Triplet (with learnable margin) Loss
    parser.add_argument('--loss_snr_margin', default=0.2, type=float, help='Triplet margin.')
    parser.add_argument('--loss_snr_reg_lambda', default=0.005, type=float,
                        help='Regularization of in-batch element sum.')

    # ArcFace
    parser.add_argument('--loss_arcface_lr', default=0.0005, type=float, help='Learning rate on class proxies.')
    parser.add_argument('--loss_arcface_angular_margin', default=0.5, type=float, help='Angular margin in radians.')
    parser.add_argument('--loss_arcface_feature_scale', default=16, type=float,
                        help='Inverse Temperature for NCA objective.')
    return parser


def get_batchmining_specific_parameters(parser):
    # Distance-based Batchminer
    parser.add_argument('--miner_distance_lower_cutoff', default=0.5, type=float,
                        help='Lower cutoff on distances - values below are sampled with equal prob.')
    parser.add_argument('--miner_distance_upper_cutoff', default=1.4, type=float,
                        help='Upper cutoff on distances - values above are IGNORED.')
    # Spectrum-Regularized Miner (as proposed in our paper) - utilizes a distance-based sampler that is regularized.
    parser.add_argument('--miner_rho_distance_lower_cutoff', default=0.5, type=float,
                        help='Lower cutoff on distances - values below are sampled with equal prob.')
    parser.add_argument('--miner_rho_distance_upper_cutoff', default=1.4, type=float,
                        help='Upper cutoff on distances - values above are IGNORED.')
    parser.add_argument('--miner_rho_distance_cp', default=0.2, type=float,
                        help='Probability to replace a negative with a positive.')
    return parser


def get_batch_creation_parameters(parser):
    parser.add_argument('--data_sampler', default='class_random', type=str,
                        help='How the batch is created. Available options: See datasampler/__init__.py.')
    parser.add_argument('--samples_per_class', default=2, type=int,
                        help='Number of samples in one class drawn before choosing the next class. '
                             'Set to >1 for tuple-based loss.')
    # Batch-Sample Flags - Have no relevance to default SPC-N sampling
    parser.add_argument('--data_batchmatch_bigbs', default=512, type=int,
                        help='Size of batch to be summarized into a smaller batch. '
                             'For distillation/coreset-based methods.')
    parser.add_argument('--data_batchmatch_ncomps', default=10, type=int,
                        help='Number of batch candidates that are evaluated, from which the best one is chosen.')
    parser.add_argument('--data_storage_no_update', action='store_true',
                        help='Flag for methods that need a sample storage. If set, storage entries are NOT updated.')
    parser.add_argument('--data_d2_coreset_lambda', default=1, type=float, help='Regularisation for D2-coreset.')
    parser.add_argument('--data_gc_coreset_lim', default=1e-9, type=float, help='D2-coreset value limit.')
    parser.add_argument('--data_sampler_lowproj_dim', default=-1, type=int,  # TODO look into the related paper
                        help='Optionally project embeddings into a lower dimension to ensure that '
                             'greedy coreset works better. Only makes a difference for large embedding dims.')
    parser.add_argument('--data_sim_measure', default='euclidean', type=str,
                        help='Distance measure to use for batch selection.')
    parser.add_argument('--data_gc_softened', action='store_true',
                        help='Flag. If set, use a soft version of greedy coreset.')
    parser.add_argument('--data_idx_full_prec', action='store_true', help='Deprecated.')
    parser.add_argument('--data_mb_mom', default=-1, type=float,
                        help='For memory-bank based samplers - momentum term on storage entry updates.')
    parser.add_argument('--data_mb_lr', default=1, type=float, help='Deprecated.')

    return parser


def get_feature_dataset_parameters(parser):
    parser.add_argument('--feature_arch_path', default=['<path_to_your_feature_arch1>',
                                                        '<path_to_your_feature_arch2>'], nargs='+',
                        help='Underlying architectures for feature extraction. '
                             'Description is same as arch.')
    parser.add_argument('--feature_embed_dim', default=64, type=int,
                        help='Embedding dimensionality of the architecture for feature extraction. '
                             'Description is same as embed_dim, but for architecture for feature extraction.')
    parser.add_argument('--feature_datasets', default=['cub200', 'cars196'], nargs='+',
                        help='datasets for multiple dataset ensemble.')
    parser.add_argument('--polling_strategy', default='batch_wise',
                        type=str, choices=['batch_wise', 'dataset_wise'],
                        help='Polling strategy that merges multiple datasets.')
    parser.add_argument('--feature_lambda', default=1.0,
                        type=float, help='Scalar to control the optimization strength between classification '
                                         'and feature regression.')
    parser.add_argument('--feature_indexes', default=[0, 64, 64, 128],
                        nargs='+', help='Feature indices with <start, end> pair to locate the primary feature '
                                        'dimension of each dataset.')
    parser.add_argument('--feature_not_used', action='store_true',
                        help='Flag. If set, no feature is used to train the network.')
    return parser


def get_visualization_parameters(parser):
    parser.add_argument('--visualization_checkpoint_path', default='<path_to_your_arch>', type=str,
                        help='Checkpoint path to load the model for visualization.')
    parser.add_argument('--visualization_save_dir', default='<path_to_save_vis_result>', type=str,
                        help='Path to save the visualization result.')
    return parser


def get_feature_penalty_parameters(parser):
    parser.add_argument('--feature_penalty_used', action='store_true',
                        help='Flag. If set, feature penalty is used to train the network.')
    parser.add_argument('--feature_penalty_base', default=8, type=int,
                        help='Base dimension to expand during feature penalty.')
    parser.add_argument('--feature_penalty_reversed', action='store_true',
                        help='Whether to reverse the feature penalty process.')
    parser.add_argument('--feature_penalty_start_dimension', default=64, type=int,
                        help='Start dimension to expand during feature penalty.')
    parser.add_argument('--feature_penalty_rescale', action='store_true',
                        help='Whether to rescale the masked features during the feature penalty process.')
    parser.add_argument('--feature_penalty_topK', action='store_true',
                        help='Whether to rescale the remain topK features during the feature penalty process.')
    parser.add_argument('--feature_penalty_lambda', default=10.0, type=float,
                        help='Lambda to control optimization strength on feature penalty.')
    return parser


def get_classifier_fusion_parameters(parser):
    parser.add_argument('--classifier_fusion_path', default=['<path_to_your_classifier_fusion_path1>',
                                                             '<path_to_your_classifier_fusion_path2>'], nargs='+',
                        help='Underlying low dimensional classifier to load to construct high dimensional classifier. '
                             'Description is same as arch.')
    parser.add_argument('--classifier_fusion_dim', default=64, type=int,
                        help='Embedding dimensionality of the low dimensional classifier.')
    parser.add_argument('--classifier_fusion_datasets', default=['cub200', 'cars196'], nargs='+',
                        help='datasets for multiple dataset ensemble.')
    # parser.add_argument('--polling_strategy', default='batch_wise',
    #                     type=str, choices=['batch_wise', 'dataset_wise'],
    #                     help='Polling strategy that merges multiple datasets.')
    parser.add_argument('--classifier_fusion_indexes', default=[0, 64, 64, 128],
                        nargs='+', help='Feature indices with <start, end> pair to '
                                        'locate the low dimensional classifier.')
    parser.add_argument('--classifier_fusion_not_freeze', action='store_true',
                        help='Flag. If set, the parameters from the low dimensional classifier is trained.')
    return parser


def get_standard_classifier_fusion_parameters(parser):
    parser.add_argument('--classifier_fusion_used', action='store_true',
                        help='Flag. If set, classifier fusion is used to train the network.')
    return parser


def read_arguments_from_cmd():
    parser = get_basic_parameters()

    parser = get_basic_training_parameters(parser)
    parser = get_batch_creation_parameters(parser)
    parser = get_batchmining_specific_parameters(parser)
    parser = get_loss_specific_parameters(parser)
    parser = get_wandb_parameters(parser)

    # for dataset fusion
    parser = get_feature_dataset_parameters(parser)

    # for visualization
    parser = get_visualization_parameters(parser)

    # for feature penalty
    parser = get_feature_penalty_parameters(parser)

    # for classifier fusion
    parser = get_classifier_fusion_parameters(parser)

    # for standard classfier fusion
    parser = get_standard_classifier_fusion_parameters(parser)

    return parser.parse_args()
