import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import models.basicblock as B

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class URRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=32, nb=20, gc=32, act_mode='L', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(URRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=2*nc, gc=gc)
        print([in_nc, out_nc, nc, nb, gc])

        #############################
        # Kernel Prediction Branch
        #############################

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        # Encoder
        self.m_down11 = nn.Conv2d(in_nc, nc, 3, 1, 1, bias=True)
        self.m_down12 = B.ResBlock(nc, nc, bias=True, mode='C' + act_mode + 'C')
        self.m_down13 = B.ResBlock(nc, nc, bias=True, mode='C' + act_mode + 'C')

        self.m_down21 = downsample_block(nc, 2*nc, bias=True, mode='2')
        self.m_down22 = B.ResBlock(2*nc, 2*nc, bias=True, mode='C' + act_mode + 'C')
        self.m_down23 = B.ResBlock(2 * nc, 2 * nc, bias=True, mode='C' + act_mode + 'C')

        self.m_down31 = downsample_block(2*nc, 4*nc, bias=True, mode='2')
        self.m_down32 = B.ResBlock(4*nc, 4*nc, bias=True, mode='C' + act_mode + 'C')
        self.m_down33 = B.ResBlock(4 * nc, 4 * nc, bias=True, mode='C' + act_mode + 'C')

        # Decoder
        self.m_up21 = upsample_block(4*nc, 2*nc, bias=True, mode='2') # 128 -> 64
        self.m_up22 = B.ResBlock(2*nc, 2*nc, bias=True, mode='C'+act_mode+'C')
        self.m_up23 = B.ResBlock(2*nc, 2*nc, bias=True, mode='C'+act_mode+'C')

        self.m_up11 = upsample_block(2*nc, nc, bias=True, mode='2') # 64 -> 32
        self.m_up12 = B.ResBlock(nc, nc, bias=True, mode='C'+act_mode+'C')
        self.m_up13 = B.ResBlock(nc, nc, bias=True, mode='C' + act_mode + 'C')

        self.conv_last = nn.Conv2d(nc, out_nc, 3, 1, 1, bias=True)

        # RRDBNet

        self.conv_first = nn.Conv2d(in_nc, 2*nc, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(2*nc, 2*nc, 3, 1, 1, bias=True)

        self.kernel_block1 = B.sequential(
            nn.Conv2d(2 * nc, 4 * nc, 1, 1, bias=True),
            B.ResBlock(4 * nc, 4 * nc, bias=True, mode='C'+act_mode+'C'),
            B.ResBlock(4 * nc, 4 * nc, bias=True, mode='C' + act_mode + 'C'))

        self.kernel_block2 = B.sequential(
            nn.Conv2d(2 * nc, 2 * nc, 1, 1, bias=True),
            B.ResBlock(2 * nc, 2 * nc, bias=True, mode='C' + act_mode + 'C'),
            B.ResBlock(2 * nc, 2 * nc, bias=True, mode='C' + act_mode + 'C'))

        self.kernel_block3 = B.sequential(
            nn.Conv2d(2 * nc, nc, 1, 1, bias=True),
            B.ResBlock(nc, nc, bias=True, mode='C' + act_mode + 'C'),
            B.ResBlock(nc, nc, bias=True, mode='C' + act_mode + 'C'))

        self.upconv1 = nn.Conv2d(2*nc, 2*nc, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(2*nc, 2*nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.HRconv = nn.Conv2d(nc, nc, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nc, out_nc, 3, 1, 1, bias=True)

    def forward(self, x):
        head = self.m_down11(x) # [1 32 256 256]
        E1 = self.m_down13(self.m_down12(head)) # [1 32 256 256]
        E2 = self.m_down23(self.m_down22(self.m_down21(E1))) # [1 64 128 128]
        E3 = self.m_down33(self.m_down32(self.m_down31(E2))) # [1 128 64 64]

        k = F.interpolate(x, scale_factor=0.25, mode='bicubic') # [1 3 64 64]
        k = self.conv_first(k) # [1 64 64 64]
        trunk = self.trunk_conv(self.RRDB_trunk(k)) # [1 64 64 64]
        k1 = k + trunk # [1 64 64 64]
        k2 = self.lrelu(self.upconv1(F.interpolate(k1, scale_factor=2, mode='nearest'))) # [1 64 128 128]
        k3 = self.lrelu(self.upconv2(F.interpolate(k2, scale_factor=2, mode='nearest'))) # [1 64 256 256]

        D = self.kernel_block1(k1) + E3 # [1 128 64 64]
        D = self.m_up23(self.m_up22(self.m_up21(D)+ self.kernel_block2(k2) + E2)) # [1 64 128 128]
        D = self.m_up13(self.m_up12(self.m_up11(D)+ self.kernel_block3(k3) + E1)) # [1 32 256 256]
        out = self.conv_last(self.HRconv(D) + head)# [1 3 256 256]

        return out

if __name__ == '__main__':
    x = torch.rand(1,3,256,256)
    net = URRDBNet()
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.size())