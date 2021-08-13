import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import models.basicblock as B
import numpy as np

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


class URRDBNetx2(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=32, nb=20, gc=32, scale=4, act_mode='L', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(URRDBNetx2, self).__init__()
        RRDB_block_f = functools.partial(B.RRDB, nc=2*nc, gc=gc)
        print([in_nc, out_nc, nc, nb, gc])
        self.scale = scale
        #############################
        # Kernel Prediction Branch
        #############################

        Block = B.ResBlock
        Conv = B.conv
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
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.m_down11 = Conv(in_nc, nc, bias=True, mode = 'C')
        self.m_down12 = Block(nc, nc, bias=True, mode='C' + act_mode + 'C')
        self.m_down13 = Block(nc, nc, bias=True, mode='C' + act_mode + 'C')

        self.m_down21 = downsample_block(nc, 2*nc, bias=True, mode='2')
        self.m_down22 = Block(2*nc, 2*nc, bias=True, mode='C' + act_mode + 'C')
        self.m_down23 = Block(2 * nc, 2 * nc, bias=True, mode='C' + act_mode + 'C')

        self.m_down31 = downsample_block(2*nc, 4*nc, bias=True, mode='2')
        self.m_down32 = Block(4*nc, 4*nc, bias=True, mode='C' + act_mode + 'C')
        self.m_down33 = Block(4 * nc, 4 * nc, bias=True, mode='C' + act_mode + 'C')

        # Decoder
        self.m_up21 = upsample_block(4 * nc, 2 * nc, bias=True, mode='2')  # 128 -> 64
        self.m_up22 = Block(2 * nc, 2 * nc, bias=True, mode='C' + act_mode + 'C')
        self.m_up23 = Block(2 * nc, 2 * nc, bias=True, mode='C' + act_mode + 'C')

        self.m_up11 = upsample_block(2 * nc, nc, bias=True, mode='2')  # 64 -> 32
        self.m_up12 = Block(nc, nc, bias=True, mode='C' + act_mode + 'C')
        self.m_up13 = Block(nc, nc, bias=True, mode='C' + act_mode + 'C')



        # RRDBNet

        self.conv_first = Conv(in_nc * 16, 2*nc, bias=True, mode = 'C')
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = Conv(2*nc, 2*nc, bias=True, mode='C')

        self.kernel_block1 = B.sequential(
            Conv(2 * nc, 4 * nc, bias=True, mode = 'C'),
            Block(4 * nc, 4 * nc, bias=True, mode='C'+act_mode+'C'),
            Block(4 * nc, 4 * nc, bias=True, mode='C' + act_mode + 'C'))

        self.kernel_block2 = B.sequential(
            Conv(2 * nc, 2 * nc, bias=True, mode = 'C'),
            Block(2 * nc, 2 * nc, bias=True, mode='C' + act_mode + 'C'),
            Block(2 * nc, 2 * nc, bias=True, mode='C' + act_mode + 'C'))

        self.kernel_block3 = B.sequential(
            Conv(2 * nc, nc, bias=True, mode = 'C'),
            Block(nc, nc, bias=True, mode='C' + act_mode + 'C'),
            Block(nc, nc, bias=True, mode='C' + act_mode + 'C'))

        self.upconv1 = Conv(2*nc, 2*nc, bias=True, mode='C')
        self.upconv2 = Conv(2*nc, 2*nc, bias=True, mode='C')
        self.lrelu = Conv(negative_slope=0.2, mode='L')

        self.H_conv0 = Conv(nc, nc, bias=True, mode='C')
        self.m_uper1 = upsample_block(nc, nc, mode = '2' + act_mode)
        self.H_conv1 =  Conv(nc, nc, bias=True, mode='C')
        self.act = Conv(mode='R')
        self.conv_last =  Conv(nc, out_nc, bias=True, mode='C')

    def forward(self, x):
        w, h = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingBottom, 0, paddingRight))(x)

        # x = F.interpolate(x, scale_factor=0.5, mode='nearest')
        head = self.m_down11(x) # [1 32 256 256]
        E1 = self.m_down13(self.m_down12(head)) # [1 32 256 256]
        E2 = self.m_down23(self.m_down22(self.m_down21(E1))) # [1 64 128 128]
        E3 = self.m_down33(self.m_down32(self.m_down31(E2))) # [1 128 64 64]

        # k = F.interpolate(x, scale_factor=0.25, mode='nearest') # [1 3 64 64]
        # k = self.conv_first(k) # [1 64 64 64]
        k = self.pixel_unshuffle(x) # [1 48 64 64]
        k = self.conv_first(k)# [1 64 64 64]
        trunk = self.trunk_conv(self.RRDB_trunk(k)) # [1 64 64 64]
        k1 = k + trunk # [1 64 64 64]
        k2 = self.lrelu(self.upconv1(F.interpolate(k1, scale_factor=2, mode='nearest'))) # [1 64 128 128]
        k3 = self.lrelu(self.upconv2(F.interpolate(k2, scale_factor=2, mode='nearest'))) # [1 64 256 256]

        D = self.kernel_block1(k1) + E3 # [1 128 64 64]
        D = self.m_up23(self.m_up22(self.m_up21(D)+ self.kernel_block2(k2) + E2)) # [1 64 128 128]
        D = self.m_up13(self.m_up12(self.m_up11(D)+ self.kernel_block3(k3) + E1)) # [1 32 256 256]
        out = self.conv_last(self.act(self.H_conv1(self.m_uper1(self.H_conv0(D) + head))))

        out = out[..., :self.scale * w, :self.scale * h]
        return out

if __name__ == '__main__':
    x = torch.rand(1,3,256,256)
    net = URRDBNetx2()
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.size())