import torch


class Refine(torch.nn.Module):
    def __init__(self, model, joint_labels, average_tag=True, n_sigma=2):
        super(Refine, self).__init__()
        self.model = model
        self.flip_index = self.get_flip_index(joint_labels)
        self.average_tag = average_tag
        self.n_sigma = n_sigma

    @torch.no_grad()
    def forward(self, x: torch.FloatTensor):
        assert x.dim() == 4, 'Invalid tensor dimension'
        x_flip = torch.flip(x, dims=(-1,))
        x = torch.cat([x, x_flip], dim=0)
        seg_pred, *hm_preds, tag_preds = self.model(x)
        hm_preds = list(hm_preds)
        seg_pred = self._merge(seg_pred[0], seg_pred[1])
        hm_preds = [self._merge(hm_pred[0], hm_pred[1][self.flip_index]) for hm_pred in hm_preds]
        if self.average_tag:
            offset, sigma, seed = torch.split(tag_preds, [2, self.n_sigma, 1], dim=1)
            offset[1, 0] = torch.neg(offset[1, 0])  # invert x offset
            offset = self._merge(offset[0], offset[1])
            sigma = self._merge(torch.abs(sigma[0]), torch.abs(sigma[1]))  # sigma might be negative
            seed = self._merge(seed[0], seed[1])
            tag_pred = torch.cat([offset, sigma, seed], dim=1)
        else:
            tag_preds[1, 1] = torch.neg(tag_preds[1, 1])
            tag_preds[1] = torch.flip(tag_preds[1], dims=(-1,))
            tag_pred = tag_preds
        return seg_pred, hm_preds, tag_pred

    @staticmethod
    def _merge(x, x_flip):
        _x = torch.flip(x_flip, dims=(-1,))
        x = torch.mean(torch.stack([x, _x]), dim=0).unsqueeze(dim=0)
        return x

    @staticmethod
    def get_flip_index(labels):
        '''helper function calculates horiozontal flip index given label (names)
        Args:
            labels: `list` or `tuple` of labels.
        Returns:
            flip_index: `list` of index of counterpart
        '''

        flip_index = []
        for i, j in enumerate(labels):
            if '_' in j:
                side, *parts = j.split('_')
                opposite = 'r' if side == 'l' else 'l'
                counterpart = opposite + '_' + '_'.join(parts)
                flip_index.append(labels.index(counterpart))
            else:
                flip_index.append(i)
        return flip_index
