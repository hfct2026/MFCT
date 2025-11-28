######-----Special thanks to "https://github.com/XPixelGroup/HAT/tree/main/hat" that helps to make our effort lighter-----######

##Import Pre-defined Modules
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
##Import User-defined Modules
from . import  MFCT
# from transformer_last import Trans_Last
from basicsr.archs.arch_util import trunc_normal_

#########--------Adaptive_Scaling--------#########
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor([init_value]))

    def forward(self, input):
        return input * self.scale
#########--------Adaptive_Scaling--------#########



#########--------Default_Convolution--------#########
def default_conv(in_channels, out_channels, kernel_size, bias=True, groups = 1):
    wn = lambda x:torch.nn.utils.weight_norm(x)
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups = groups)
#########--------Default_Convolution--------#########



#########--------Default_Convolution--------#########
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
#########--------Default_Convolution--------#########



#########--------Convolution+BN+ReLU_Module--------#########
class BasicConv(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=1, 
                 dilation=1, 
                 groups=1, 
                 relu=True,
                 bn=False, 
                 bias=False, 
                 up_size=0,
                 fan=False):
        super(BasicConv, self).__init__()
        
        ##Arguments
        self.up_size = up_size

        ##Modules
        if fan:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                           padding=padding, dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                  padding=padding,dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x
#########--------Convolution+BN+ReLU_Module--------#########



#########--------DS-Convolution_Module--------#########
class DS_Conv(nn.Module):
    """ Depth-wise Separable Convolution """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DS_Conv, self).__init__()
        
        ##Modules
        self.DepthConv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.PointConv = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1, dilation=2, padding=2)
        self.Relu = nn.ReLU(inplace=False)
        self.BatchNorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.DepthConv(x)
        x = self.Relu(x)
        x = self.PointConv(x)
        x = self.Relu(x)
        x = self.BatchNorm(x)
        return x
#########--------DS-Convolution_Module--------#########



#########--------Attention_Module--------#########
def Attention(in_channels, factor=2):
    layers = []
    layers.append(nn.Conv2d(in_channels, in_channels//factor, 1, padding=0, bias=True))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(in_channels//factor, in_channels, 1, padding=0, bias=True))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)
    
class Att_Low(nn.Module):
    def __init__(self, 
                 in_channels, 
                 factor):
        super(Att_Low, self).__init__()

        ##Modules
        self.conv1 = BasicConv(in_channels, in_channels, 3,1,1)
        self.conv2 = BasicConv(in_channels, in_channels, 3,1,1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.att = Attention(in_channels, factor)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avg_pool(x)
        y = self.att(y)
        return self.conv2(x * y)
#########--------Attention_Module--------#########



#########--------Low_Frequency_Module_1--------#########
class Low_Freq_Block1(nn.Module):
    def __init__(self, in_channels, factor, compress_ratio=2):  #compress_ratio=3 for in_channels=180 #compress_ratio=2 or 4 for in_channels=64/128
        super(Low_Freq_Block1, self).__init__()

        ##Arguments
        self.in_channels = in_channels
        factor1 = compress_ratio
        factor2 = compress_ratio*compress_ratio
        
        ##Modules
        self.Conv1 = DS_Conv(in_channels,in_channels//factor1,kernel_size=5,stride=1,padding=2)
        self.Conv2 = DS_Conv(in_channels//factor1,in_channels//factor1,kernel_size=5,stride=1,padding=2)
        self.Conv3 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.ConvRed1 = nn.Conv2d(in_channels,in_channels//factor2,kernel_size=3,stride=1,padding=1)
        self.ConvExp1 = nn.Conv2d(in_channels//factor2,in_channels//factor1,kernel_size=3,stride=1,padding=1)
        self.ConvRed2 = nn.Conv2d(in_channels//factor1,in_channels//factor2,kernel_size=3,stride=1,padding=1)
        self.ConvExp2 = nn.Conv2d(in_channels//factor2,in_channels//factor1,kernel_size=3,stride=1,padding=1)
        self.Relu = nn.ReLU(inplace=True)
        self.Attention = Att_Low(in_channels, factor)
        self.Weight1 = Scale(1)
        self.Weight2 = Scale(1)

    def forward(self, x):
        x_res = x
        x11 = self.Conv1(x)
        x12 = self.Conv2(self.Relu(x11))
        x_conv = torch.cat((x11, x12),1)
        x21 = self.ConvExp1(self.Relu(self.ConvRed1(x)))
        x22 = self.ConvExp2(self.Relu(self.ConvRed2(x21)))
        x_RE = torch.cat((x21, x22),1)
        x_add = self.Weight1(x_RE) + self.Weight2(x_conv) + x_res

        return self.Attention(x_add)
#########--------Low_Frequency_Module_1--------#########



#########--------Low_Frequency_Module_2--------#########
class Low_Freq_Block2(nn.Module):
    def __init__(self, in_channels, factor, compress_ratio=3):    #compress_ratio=3 for in_channels=180 #compress_ratio=2 or 4 for in_channels=64/128
        super(Low_Freq_Block2, self).__init__()
        
        ##Arguments
        self.in_channels = in_channels
        factor1 = compress_ratio
        factor2 = compress_ratio*compress_ratio
        
        ##Modules
        self.Conv1 = DS_Conv(in_channels,in_channels//factor1,kernel_size=5,stride=1,padding=2)
        self.Conv2 = DS_Conv(in_channels//factor1,in_channels//factor1,kernel_size=5,stride=1,padding=2)
        self.Conv3 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.ConvRed1 = nn.Conv2d(in_channels//factor1,in_channels//factor2,kernel_size=3,stride=1,padding=1)
        self.ConvExp1 = nn.Conv2d(in_channels//factor2,in_channels//factor1,kernel_size=3,stride=1,padding=1)
        self.ConvRed2 = nn.Conv2d(in_channels//factor1,in_channels//factor2,kernel_size=3,stride=1,padding=1)
        self.ConvExp2 = nn.Conv2d(in_channels//factor2,in_channels//factor1,kernel_size=3,stride=1,padding=1)
        self.Relu = nn.ReLU(inplace=True)
        self.Attention = Att_Low(in_channels, factor)
        self.Weight1 = Scale(1)
        self.Weight2 = Scale(1)
        self.Weight3 = Scale(1)
        self.Weight4 = Scale(1)

    def forward(self, x):
        x_res = x
        x1 = self.Conv1(x)
        x2 = self.ConvExp1(self.Relu(self.ConvRed1(x1)))
        x3 = self.Conv2(self.Relu(x2))
        x4 = self.ConvExp2(self.Relu(self.ConvRed2(x3)))
        x_low1 = self.Weight2(x2) + self.Weight1(x1 - x4)
        x_low2 = self.Weight4(x3) + self.Weight3(x1 - x3)
        x_cat = torch.cat((x_low1, x_low2),1)

        return self.Attention(x_cat + x_res)
#########--------Low_Frequency_Module_2--------#########



#########--------High_Frequency_Preserving_Module--------#########
class HFPB(nn.Module):
    """ HFPB:: High Frequency Preserving Block"""
    
    def __init__(self, in_channels, factor, compress_ratio):
        super(HFPB, self).__init__()

        ##Arguments
        self.in_channels = in_channels
        fraction=2
        self.fraction = fraction

        ##Modules
        self.Conv1 = BasicConv(in_channels, in_channels//fraction, 3,1,1)
        self.Conv2 = BasicConv(in_channels//fraction, in_channels//fraction, 3,1,1)
        self.LowPool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.LowExplorer1 = Low_Freq_Block1(in_channels//fraction, factor, compress_ratio)
        self.LowExplorer2 = Low_Freq_Block2(in_channels//fraction, factor, compress_ratio)
        self.LowConv1 = BasicConv(in_channels//fraction, in_channels//fraction, 3,1,1)
        self.LowConv2 = BasicConv(in_channels//fraction, in_channels//fraction, 3,1,1)
        self.Relu = nn.ReLU()
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)

    def forward(self, x): #[32, 32, 64, 64]
        x_res = x
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Relu(x1))
        x = torch.cat([x1,x2],dim=1)
        x_low = self.LowPool(x)
        x_high = x - F.interpolate(x_low, size = x.size()[-2:], mode='bilinear', align_corners=True)
        x_list = torch.split(x_high, math.ceil(self.in_channels//self.fraction), dim=1)
        x_high1 = x_list[0]
        x_high1 = self.LowExplorer1(x_high1)
        x_high2 = x_list[1]
        x_high2 = self.LowExplorer2(x_high2)
        x_cat = torch.cat((self.LowConv1(x_high1), self.LowConv2(x_high2)),1)

        return self.weight1(x_res) + self.weight2(x_cat)
#########--------High_Frequency_Preserving_Module--------#########



#########--------High_Frequency_Module--------#########
class HFM(nn.Module):   #P:1814351=1.81M
    """ HFM:: High Frequency Module """
    
    def __init__(self, in_channels, factor, compress_ratio, hfm_blocks=2):
        super(HFM, self).__init__()
        
        ##Arguments
        self.hfm_blocks = hfm_blocks
        
        ##Modules
        self.blocks = nn.ModuleList([HFPB(in_channels, factor, compress_ratio) for i in range(hfm_blocks)])
        #self.Conv_cat = nn.Conv2d(hfm_blocks*in_channels, in_channels, 3, 1, 1, groups=hfm_blocks)

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        #x = self.Conv_cat(x)
        return x
#########--------High_Frequency_Module--------#########



#########--------Frequency_Enhanced_Transformer_Block--------#########
class FETB(nn.Module):
    r""" FETB: Frequency Enhanced Transformer Block
    Args:
        in_channels    (int): Number of channels after shallow conv. Default 180
        img_size       (int | tuple(int)): Height and width of the input image. Default 192/4  
    """
    def __init__(self, 
                 in_channels, 
                 img_size,
                 num_layers
                 ):
        super(FETB, self).__init__()
        
        ##Modules
        #Transformer
        self.trans1 = mfct(in_channels=in_channels, img_size=img_size, num_layers=num_layers)   #P:755532=0.75M
        self.trans2 = mfct(in_channels=in_channels, img_size=img_size, num_layers=num_layers)   #P:755532=0.75M
        self.trans3 = mfct(in_channels=in_channels, img_size=img_size, num_layers=num_layers)   #P:755532=0.75M
        #Convolutor
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv6 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        #Weight(Learned)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)
        self.weight3 = Scale(1)
        self.weight4 = Scale(1)
        self.weight5 = Scale(1)
        self.weight6 = Scale(1)
        self.weight7 = Scale(1)
        self.weight8 = Scale(1)
        self.weight9 = Scale(1)
        self.weight10 = Scale(1)
        self.weight11 = Scale(1)
        self.weight12 = Scale(1)

    def forward(self, x, x_h):             #[32, 32, 64, 64]
        x_res = x
        xh_res = x_h
        x, x_h = self.trans1(x, x_h)  #xh_mix=[2, 48, 64, 64]) xl_mix=[2, 48, 64, 64]
        x = self.weight1(x_res) + self.weight2(self.conv1(x))
        x_h = self.weight3(xh_res) + self.weight4(self.conv2(x_h))
        x_res = x
        xh_res = x_h
        x, x_h = self.trans2(x, x_h)
        x = self.weight5(x_res) + self.weight6(self.conv3(x))
        x_h = self.weight7(xh_res) + self.weight8(self.conv4(x_h))
        x_res = x
        xh_res = x_h
        x, x_h = self.trans3(x, x_h)  #xh_mix=[2, 48, 64, 64]) xl_mix=[2, 48, 64, 64]
        return self.weight9(x_res) + self.weight10(self.conv5(x)), self.weight11(xh_res) + self.weight12(self.conv6(x_h))
#########--------Frequency_Enhanced_Transformer_Block--------#########


#########--------Multi-feature_Parallel_Transformer_Module--------#########
class MFPT(nn.Module):
    r""" FETB: Frequency Enhanced Transformer Block
    Args:
        args     (Namespace): Model related arguments
        n_blocks       (int): Number of Frequency Enhanced Transformer Block  -Default=1
        factor         (int): Channel reduction ratio in attention module  -Default=2
        compress_ratio (int): Channel reduction ratio in frequency block 1 $ 2  -Default=3
        hfm_blocks     (int): NUmber of subsequent high frequency module  -Default=2
        hfm_blocks     (int): NUmber of subsequent low frequency module  -Default=2
        up_channels    (int): Number of intermediate channels during upsampling  -Default=64
        kernel_size    (int): Size of the kernels during convolution  -Default=3
        img_range      (int): Range of the images during mean calculation  -Default=1.
        
    """
    def __init__(self, 
                 args, 
                 n_blocks = 3,
                 factor=2,
                 compress_ratio=2,
                 hfm_blocks=2,
                 lfm_blocks=2,
                 num_layers=8,
                 conv=default_conv,
                 up_channels = 64, 
                 kernel_size = 3,
                 img_range=1.):
        super(MFPT, self).__init__()

        ##Arguments
        channel_in = args.n_colors
        in_channels = args.in_channels
        patch_size = args.patch_size
        scale = args.scale
        img_size = patch_size/scale
        self.n_blocks = n_blocks
              
        ##RGB_mean_for_DIV2K
        self.img_range = img_range
        if channel_in == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1.0, 1.0, 1.0, 1.0)
        
        ##Extractor
        self.ExplorerHigh = HFM(in_channels, factor, compress_ratio,  hfm_blocks)
        ##Head
        modules_head = [default_conv(3, in_channels, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        ##Body
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(MFCT.mfct(in_channels, img_size, num_layers=num_layers))
        self.body = nn.Sequential(*modules_body)

        ##Conv_Before_Upsample
        self.reduce = default_conv(in_channels, in_channels, kernel_size)
        
        ##Tail
        modules_tail = [default_conv(in_channels, up_channels, kernel_size),
                        nn.LeakyReLU(inplace=True), 
                        Upsampler(conv, scale, up_channels, act=False), 
                        default_conv(up_channels, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)
        
        ##Upsample_as_residual
        #self.up = nn.Sequential(Upsampler(conv,scale,in_channels,act=False), BasicConv(in_channels, 3,3,1,1))
        
        ##Weight_Initialization
        self.apply(self._init_weights)
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_in,x2 = None, test=False):
        self.mean = self.mean.type_as(x_in)      #[1, 3, 1, 1]
        x_mean = (x_in - self.mean) * self.img_range  #[1, 3, 64, 64]
        #x_res1 = x_mean
        
        ####----Head----####
        x_int = self.head(x_mean)  #[1, 180, 64, 64]
        ####----Head----####
        
        x_res2 = x_int
        
        #body_out = []
        
        xh_mod = self.ExplorerHigh(x_int)
        ####----Body----####
        loss_sum = []
        for i in range(self.n_blocks):
            x_int, xh_mod, load_balance_loss = self.body[i](x_int, xh_mod)
            loss_sum.append(load_balance_loss)
            #body_out.append(x_int)
        
        ####----Body----####
        
        #x_out = torch.cat(body_out,1)
        x_out = self.reduce(x_int)
        x_out = x_out + x_res2
        
        ####----Tail----####      
        x_out = self.tail(x_out)
        #x_out = self.up(x_res1) + x_out
        ####----Tail----####
        
        x_out = x_out / self.img_range + self.mean

        return x_out, loss_sum
#########--------Multi-feature_Parallel_Transformer_Module--------#########
