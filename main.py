from scripts.parameters import read_arguments_from_cmd
from scripts import standard
from scripts import dataset_fusion
from scripts import classifier_fusion

import torch
import matplotlib
import warnings

if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    matplotlib.use('agg')
    warnings.filterwarnings("ignore")

    par = read_arguments_from_cmd()

    training_type = par.training_script

    print("** Training with `training_type = %s` **" % training_type)

    if training_type == 'standard':
        standard.main(par)
    if training_type == 'dataset_fusion':
        dataset_fusion.main(par)
        # dataset_fusion.run_feature_dataset(par)
    if training_type == 'dataset_fusion_visualization':
        dataset_fusion.visualization(par)
    if training_type == 'classifier_fusion':
        classifier_fusion.main(par)
