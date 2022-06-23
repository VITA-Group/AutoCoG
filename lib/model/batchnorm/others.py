import torch
import torch.nn as nn
import torch.nn.functional as F

class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x, *args, **kwargs):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x

class mean_norm(torch.nn.Module):
    def __init__(self):
        super(mean_norm, self).__init__()
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x, *args, **kwargs):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        return x

class group_norm(torch.nn.Module):
    def __init__(self, dim_to_norm=None, dim_hidden=16, num_groups=None, skip_weight=None, **w):
        super(group_norm, self).__init__()
        self.num_groups = num_groups
        self.skip_weight = skip_weight

        dim_hidden = dim_hidden if dim_to_norm is None else dim_to_norm
        self.dim_hidden = dim_hidden

        # print(f'\n\n{dim_to_norm}\n\n');raise

        self.bn = torch.nn.BatchNorm1d(dim_hidden * self.num_groups, momentum=0.3)
        self.group_func = torch.nn.Linear(dim_hidden, self.num_groups, bias=True)
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        if self.num_groups == 1:
            x_temp = self.bn(x)
        else:
            N, _ = x.shape
            score_cluster = F.softmax(self.group_func(x), dim=1)
            x_temp = torch.cat([score_cluster[:, group].unsqueeze(dim=1) * x for group in range(self.num_groups)],
                               dim=-1)
            x_temp = self.bn(x_temp).view(-1, self.num_groups, self.dim_hidden).sum(dim=1)

        x = x + x_temp * self.skip_weight
        return x