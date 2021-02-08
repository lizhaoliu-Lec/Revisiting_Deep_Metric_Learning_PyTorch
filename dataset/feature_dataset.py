from dataset.basic_dataset_scaffold import BaseDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import architectures as archs


class FeatureDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.has_got_features = False
        self.features = self.get_features(self.pars)
        self.return_feature = True

    def set_return_feature(self, return_or_not):
        self.return_feature = return_or_not

    def reset_return_feature(self):
        self.return_feature = True

    @torch.no_grad()
    def get_features(self, opt):

        embed_dim = opt.embed_dim
        opt.embed_dim = opt.feature_embed_dim

        # get pretrained model
        model = archs.select(opt.arch, opt)

        checkpoint = torch.load(opt.feature_arch_path[opt.dataset_idx])

        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        model.eval()
        dataloader = DataLoader(self, num_workers=opt.kernels, batch_size=opt.bs, shuffle=False, drop_last=False)

        data_iterator = tqdm(dataloader,
                             desc='Extracting feature for dataset `%s`...' % opt.feature_datasets[opt.dataset_idx])

        features = []

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

        features = torch.cat(features, dim=0)

        opt.embed_dim = embed_dim

        self.has_got_features = True
        return features

    def __getitem__(self, idx):
        img_label, img_tensor, img_idx = super().__getitem__(idx)
        if not self.has_got_features:
            return img_label, img_tensor, img_idx
        else:
            feature = self.features[idx]
            if self.return_feature:
                return img_label, img_tensor, feature, img_idx
            else:
                return img_label, img_tensor, img_idx
