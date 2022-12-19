import torch
import torch.nn as nn
from loss import batch_episym
from torch.nn import functional as F
import numpy as np


class GRA_block(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        sub_channels = channels // 4

        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1)
        )

        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, sub_channels, kernel_size=1),
            nn.BatchNorm2d(sub_channels),
            nn.ReLU(),
            nn.Conv2d(sub_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(sub_channels, eps=1e-3),
            nn.BatchNorm2d(sub_channels),
            nn.ReLU(),
            nn.Conv2d(sub_channels, sub_channels, kernel_size=1),
            nn.InstanceNorm2d(sub_channels, eps=1e-3),
            nn.BatchNorm2d(sub_channels),
            nn.ReLU(),
            nn.Conv2d(sub_channels, sub_channels, kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.InstanceNorm2d(sub_channels, eps=1e-3),
            nn.BatchNorm2d(sub_channels),
            nn.ReLU(),
            nn.Conv2d(sub_channels, sub_channels, kernel_size=1),
            nn.InstanceNorm2d(sub_channels, eps=1e-3),
            nn.BatchNorm2d(sub_channels),
            nn.ReLU(),
            nn.Conv2d(sub_channels, sub_channels, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            nn.InstanceNorm2d(sub_channels, eps=1e-3),
            nn.BatchNorm2d(sub_channels),
            nn.ReLU(),
            nn.Conv2d(sub_channels, sub_channels, kernel_size=1),
            nn.InstanceNorm2d(sub_channels, eps=1e-3),
            nn.BatchNorm2d(sub_channels),
            nn.ReLU(),
            nn.Conv2d(sub_channels, sub_channels, kernel_size=1)
        )
        self.conv4 = nn.Sequential(
            nn.InstanceNorm2d(sub_channels, eps=1e-3),
            nn.BatchNorm2d(sub_channels),
            nn.ReLU(),
            nn.Conv2d(sub_channels, sub_channels, kernel_size=1),
            nn.InstanceNorm2d(sub_channels, eps=1e-3),
            nn.BatchNorm2d(sub_channels),
            nn.ReLU(),
            nn.Conv2d(sub_channels, sub_channels, kernel_size=1)
        )

    def forward(self, x):
        spx = torch.split(x, 64, 1)
        #x_spatial = self.spatial_attpre(spx[0])

        x_spatial = spx[0]
        x_spatial = torch.cat((torch.max(x_spatial, 1)[0].unsqueeze(1), torch.mean(x_spatial, 1).unsqueeze(1)), dim=1)
        x_spatial = self.spatial_att(x_spatial)  
        scale_sa = torch.sigmoid(x_spatial)

        sp1 = spx[0] * scale_sa
        sp1 = self.conv1(sp1)

        sp2 = sp1 + spx[1] * scale_sa
        sp2 = self.conv2(sp2)

        sp3 = sp2 + spx[2] * scale_sa
        sp3 = self.conv3(sp3)

        sp4 = sp3 + spx[3] * scale_sa
        sp4 = self.conv4(sp4)

        cat = torch.cat((sp1, sp2, sp3, sp4), 1)
        out = cat

        xag = F.avg_pool2d(out, (out.size(2), x.size(3)), stride=(out.size(2), x.size(3)))  
        xmg = F.max_pool2d(out, (out.size(2), x.size(3)), stride=(out.size(2), x.size(3)))
        
        xam = self.channel_att(xag + xmg)

        scale_ca = torch.sigmoid(xam)
        out = out * scale_ca

        out = out + x
        out = shuffle_chnls(out, 4)
        return out

class PointCN_down128(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels , eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels // 2, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv(x)

        return out


class PointCN_256(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv(x)

        return out


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)  # b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        # x_up: b*c*n*1
        # x_down: b*c*k*1
        embed = self.conv(x_up)  # b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)  # b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),  # b*c*n*1
            trans(1, 2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class GRAModule(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        print('channels:' + str(channels) + ', layer_num:' + str(self.layer_num))
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)

        l2_nums = clusters

        self.l1_1 = []
        for _ in range(self.layer_num // 2):
            self.l1_1.append(GRA_block(channels))

        self.l1_down128 = PointCN_down128(channels)
        self.down1 = diff_pool(channels//2, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num // 2):
            self.l2.append(OAFilter(channels //2, l2_nums))

        self.up1 = diff_unpool(channels//2, l2_nums)

        self.l1_2 = []
        self.l1_exchange = PointCN_256(channels)

        for _ in range(self.layer_num // 2):
            self.l1_2.append(GRA_block(channels))

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, data, xs):
        # data: b*c*n*1
        batch_size, num_pts = data.shape[0], data.shape[2]
        x1_1 = self.conv1(data)
        x1_1 = self.l1_1(x1_1)

        x1_1 = self.l1_down128(x1_1)
        
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        x1_2 = self.l1_exchange(torch.cat([x1_1, x_up], dim=1))
        out = self.l1_2(x1_2)

        logits = torch.squeeze(torch.squeeze(self.output(out), 3), 1)
        e_hat = weighted_8points(xs, logits)

        x1, x2 = xs[:, 0, :, :2], xs[:, 0, :, 2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

        return logits, e_hat, residual


class PGFNet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        depth_each_stage = config.net_depth // (config.iter_num + 1)
        self.side_channel = (config.use_ratio == 2) + (config.use_mutual == 2)
        self.weights_init = GRAModule(config.net_channels, 4 + self.side_channel, depth_each_stage, config.clusters)
        self.weights_iter = [GRAModule(config.net_channels, 6 + self.side_channel, depth_each_stage, config.clusters) for
                             _ in range(config.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        # data: b*1*n*c
        # x_weight = self.positon(data['xs'])
        input = data['xs'].transpose(1, 3)
        if self.side_channel > 0:
            sides = data['sides'].transpose(1, 2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat = [], []
        logits, e_hat, residual = self.weights_init(input, data['xs'])
        res_logits.append(logits), res_e_hat.append(e_hat)
        logits_1 = logits
        logits_2 = torch.zeros(logits.shape).cuda() 

        index_k = logits.topk(k=num_pts//2, dim=-1)[1]  
        input_new = torch.stack(
            [ input[i].squeeze().transpose(0, 1)[index_k[i]] for i in range(input.size(0))]).unsqueeze(-1).transpose(1, 2)
        
        residual_new = torch.stack(
            [ residual[i].squeeze(0)[index_k[i]] for i in range(residual.size(0))]).unsqueeze(1)
        
        logits_new = logits.reshape(residual.shape)
        logits_new = torch.stack(
            [ logits_new[i].squeeze(0)[index_k[i]] for i in range(logits_new.size(0))]).unsqueeze(1)
        
        data_new = torch.stack(
            [ data['xs'][i].squeeze(0)[index_k[i]] for i in range(input.size(0))]).unsqueeze(1)
        


        for i in range(self.iter_num):
            logits, e_hat, residual = self.weights_iter[i](
                torch.cat([input_new, residual_new.detach(), torch.relu(torch.tanh(logits_new)).detach()],
                          dim=1),
                data_new)
            '''for i in range(logits_2.size(0)):
                for j in range(logits.size(-1)):                   
                    logits_2[i][index_k[i][j]] = logits[i][j]'''
            logits_2.scatter_(1, index_k, logits)
            
            logits_2 = logits_2 + self.gamma*logits_1
            e_hat = weighted_8points(data['xs'], logits_2) 
            
                          
            res_logits.append(logits_2), res_e_hat.append(e_hat)
        #print(self.gamma)
        return res_logits, res_e_hat

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


def shuffle_chnls(x, groups=4):
    """Channel Shuffle"""

    bs, chnls, h, w = x.data.size()
    if chnls % groups:
        return x
    chnls_per_group = chnls // groups
    x = x.view(bs, groups, chnls_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bs, -1, h, w)

    return x
