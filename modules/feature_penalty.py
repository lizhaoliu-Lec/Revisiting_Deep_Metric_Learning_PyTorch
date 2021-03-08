import torch
import torch.nn as nn


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

        self.current_dim = self.batchIdToEndIndex[0]

    @staticmethod
    def get_batchIdToEndIndex(total_dimension, total_epoch, base, reverse=False, start_dimension=None):

        # print("====> total_dimension ", total_dimension)
        # print("====> total_epoch ", total_epoch)
        # print("====> base ", base)

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

    @staticmethod
    def zero_feature_by_endIndex(batch_feature, end_index, topK=False):

        batch_size, feature_size = batch_feature.size()

        loss = None
        if feature_size - end_index > 0:
            sub_features = batch_feature[:, end_index:]
            batch_zeros = torch.zeros((batch_size, feature_size - end_index), device=batch_feature.device,
                                      requires_grad=False)

            # print("====> sub_features.size() ", sub_features.size())
            # print("====> batch_zeros.size() ", batch_zeros.size())

            loss = nn.L1Loss()(sub_features, batch_zeros)

        # print("====> batch_masks.size() ", batch_masks.size())
        return loss

    def forward_by_masking_zero(self, batch_feature, epoch):
        # print("====> epoch ", epoch)
        self.current_dim = self.batchIdToEndIndex[epoch]
        if epoch not in self.encountered_epoch:
            self.encountered_epoch.append(epoch)
            if 0 < epoch < len(self.batchIdToEndIndex) - 1 and \
                    self.batchIdToEndIndex[epoch - 1] != self.current_dim:
                print("[FeaturePenalty] Changing feature_dim from %d to %d for epoch %d" % (
                    self.batchIdToEndIndex[epoch - 1],
                    self.current_dim, epoch))
            else:
                print("[FeaturePenalty] Using feature_dim %d for epoch %d" % (self.current_dim, epoch))

        scale_factor = 1.0 if not self.rescale else self.total_dimension / self.current_dim
        return self.mask_feature_by_endIndex(batch_feature, self.current_dim,
                                             scale_factor, topK=self.topK)

    def forward(self, batch_feature, epoch):
        # print("====> epoch ", epoch)
        self.current_dim = self.batchIdToEndIndex[epoch]
        if epoch not in self.encountered_epoch:
            self.encountered_epoch.append(epoch)
            if 0 < epoch < len(self.batchIdToEndIndex) - 1 and \
                    self.batchIdToEndIndex[epoch - 1] != self.current_dim:
                print("[FeaturePenalty] Changing feature_dim from %d to %d for epoch %d" % (
                    self.batchIdToEndIndex[epoch - 1],
                    self.current_dim, epoch))
            else:
                print("[FeaturePenalty] Using feature_dim %d for epoch %d" % (self.current_dim, epoch))

        return self.zero_feature_by_endIndex(batch_feature, self.current_dim, topK=self.topK)

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
        ret = FeaturePenalty.get_batchIdToEndIndex(128, 150, 8, reverse=True, start_dimension=64)
        print("===> ret ", ret)
        # FeaturePenalty.get_batchIdToEndIndex(32, 150, 2, reverse=True, start_dimension=8)


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


    # run_feature_penalty()
    run_batchIdToEndIndex()
    # plot_feature_penalty()
    # run_topK()
