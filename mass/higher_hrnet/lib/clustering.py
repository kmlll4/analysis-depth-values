import torch
import torch.nn as nn


class SpatialClustering(nn.Module):
    def __init__(self,
                 threshold=.5,
                 margin=0.5,
                 min_pixels=64,
                 n_sigma=2):
        super().__init__()
        self.threshold = threshold
        self.margin = margin
        self.min_pixels = min_pixels
        self.n_sigma = n_sigma

    @torch.no_grad()
    def forward(self, tag_pred, *args, **kwargs):
        """
        Args:
            tag_pred: torch.Tensor, (b, 5, h, w)
        """
        return torch.stack([self.cluster_per_image(_x)[0] for _x in tag_pred])

    def cluster_per_image(self, x: torch.Tensor):
        """
        Args:
            x: torch.Tensor, (5, h, w)
        """
        offset, sigma, seed = x.split([2, 2, 1], dim=0)
        device = offset.device
        h, w = offset.shape[-2:]
        xy_grid = self.make_grid(h, w).to(device=offset.device)
        instance_map = torch.zeros(h, w, dtype=torch.uint8).to(device=offset.device)

        spa_emb = torch.tanh(offset) + xy_grid
        seed_map = torch.sigmoid(seed)

        instances = []
        seed_coords = []

        count = 1
        mask = seed > self.threshold

        # If foreground #pixel is larger than min_pixels
        if mask.sum() > self.min_pixels:
            spa_emb_masked = torch.masked_select(spa_emb, mask.expand_as(spa_emb)).view(2, -1)
            sigma_masked = torch.masked_select(sigma, mask.expand_as(sigma)).view(self.n_sigma, -1)
            seed_map_masked = torch.masked_select(seed_map, mask).view(1, -1)

            unclustered = torch.ones(mask.sum(), dtype=torch.uint8).to(device)
            instance_map_masked = torch.zeros(mask.sum(), dtype=torch.uint8).to(device)

            while unclustered.sum() > self.min_pixels:
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()

                # no seed
                if seed_score < self.threshold:
                    break

                center = spa_emb_masked[:, seed: seed + 1]
                unclustered[seed] = 0
                s = sigma_masked[:, seed: seed + 1]  # n_sigma x 1
                dist = torch.exp(-1 * torch.sum(torch.pow(spa_emb_masked - center, 2) / (2 * s ** 2), 0, keepdim=True))

                proposal = (dist > self.margin).squeeze()

                if proposal.sum() > self.min_pixels:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:  # ?
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(h, w, dtype=torch.uint8)
                        instance_mask[mask.squeeze().cpu()] = proposal.byte().cpu()
                        instances.append(
                            {'mask': instance_mask.squeeze() * 255, 'score': seed_score})
                        count += 1
                        seed_coords.append(seed)

                unclustered[proposal] = 0

            instance_map[mask.squeeze()] = instance_map_masked.byte()

        return instance_map, instances

    @staticmethod
    def make_grid(h, w):
        if w >= h:
            xm = torch.linspace(0, w / h, w).view(1, 1, -1).expand(1, h, w)
            ym = torch.linspace(0, 1, h).view(1, -1, 1).expand(1, h, w)
        else:
            xm = torch.linspace(0, 1, w).view(1, 1, -1).expand(1, h, w)
            ym = torch.linspace(0, h / w, h).view(1, -1, 1).expand(1, h, w)
        xy_grid = torch.cat([xm, ym], dim=0)
        return xy_grid
