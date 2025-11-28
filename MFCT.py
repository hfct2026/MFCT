##Import Pre-defined Modules
import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
##Import User-defined Modules
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from einops import rearrange
NEG_INF = -1000000


############------Scale------############ 
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor([init_value]))

    def forward(self, input):
        return input * self.scale
############------Scale------############ 



############------DropPath------############    
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
############------DropPath------############



############------PatchEmbed------############
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):     #[2, 180, 64, 64]
        x = x.flatten(2).transpose(1, 2)  #[2, 180, 64, 64]->[2, 180, 4096]->[2, 4096, 180]
        if self.norm is not None:
            x = self.norm(x)  #[2, 4096, 180]
        return x
############------PatchEmbed------############


############------PatchUnEmbed------############
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x
############------PatchUnEmbed------############



############------Multi-feature Attention Block(MFAB)------############
class Attention(nn.Module):
    def __init__(self, embed_dim, factor=30):
        super(Attention, self).__init__()
        self.att = nn.Sequential(nn.AdaptiveAvgPool2d(1),                                  #P:0
                                 nn.Conv2d(embed_dim, embed_dim // factor, 1, padding=0),  #P:1086
                                 nn.ReLU (inplace=True),                                   #P:0
                                 nn.Conv2d(embed_dim // factor, embed_dim, 1, padding=0),  #P:1260
                                 nn.Sigmoid())                                             #P:0

    def forward(self, x):
        y = self.att(x)
        return x * y


class MFAB(nn.Module):
    """Multi-feature Attention Block(MFAB)
    Args:
        num_feat (int): Channel number of intermediate features.
        fraction (int): Channel copression factor for Enhancement Block. Default: 3.
        factor (int): Channel copression factor for Attention Block. Default: 16.
    """
    def __init__(self, embed_dim, fraction=3, factor=30):
        super(MFAB, self).__init__()

        self.Enhance = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // fraction, 3, 1, 1),   #P:97260
                                     nn.GELU (),                                             #P:0
                                     nn.Conv2d(embed_dim // fraction, embed_dim, 3, 1, 1))   #P:97380
        self.Attentive = Attention(embed_dim, factor=factor)                              

    def forward(self, x):
        return self.Attentive(self.Enhance(x))
############------Multi-feature Attention Block (MFAB)------############




############------Multi Layer Perceptron (MLP)------############
class Mlp(nn.Module):  #All r 1D
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.Linear1 = nn.Linear(in_features, hidden_features)   #P:32580
        self.Activation = act_layer()                            
        self.Linear2 = nn.Linear(hidden_features, out_features)  #P:32580
        self.DropOut = nn.Dropout(drop)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.Activation(x)
        x = self.DropOut(x)
        x = self.Linear2(x)
        x = self.DropOut(x)
        return x
############------Multi Layer Perceptron (MLP)------############



class MlpExpert(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MoE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_experts=4, k=1, drop=0.):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(in_features, num_experts)
        self.experts = nn.ModuleList([
            MlpExpert(in_features, hidden_features, out_features, drop=drop)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, N, C = x.shape
        x_flat = x.view(-1, C)  # [B*N, C]

        # Gumbel-Softmax routing
        logits = self.gate(x_flat)  # [B*N, E]
        gates = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)  # [B*N, E]

        # Load balancing loss (maximize entropy of average gate usage)
        mean_gates = gates.mean(dim=0)  # [E]
        load_balance_loss = (mean_gates * torch.log(mean_gates + 1e-8)).sum()

        # Top-k routing
        topk_vals, topk_idxs = torch.topk(gates, self.k, dim=1)  # both [B*N, k]
        output = torch.zeros_like(x_flat)  # [B*N, C]

        for i in range(self.k):
            expert_indices = topk_idxs[:, i]  # [B*N]
            weights = topk_vals[:, i]         # [B*N]

            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    x_selected = x_flat[mask]  # inputs for this expert
                    out_selected = self.experts[expert_id](x_selected)  # [num_tokens, C]
                    output[mask] += weights[mask].unsqueeze(1) * out_selected

        return output.view(B, N, C), load_balance_loss



########################------------DynamicPosBias------------########################
class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(nn.LayerNorm(self.pos_dim),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(self.pos_dim, self.pos_dim))
        self.pos2 = nn.Sequential(nn.LayerNorm(self.pos_dim),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(self.pos_dim, self.pos_dim))
        self.pos3 = nn.Sequential(nn.LayerNorm(self.pos_dim),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(self.pos_dim, self.num_heads))
                                  
    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos
########################------------DynamicPosBias------------########################


    
########################------------Window Attention------------########################
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        embed_dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, 
                 embed_dim, 
                 window_size, 
                 num_heads, 
                 qkv_bias=True, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.embed_dim = embed_dim   #180
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads  #6
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.position_bias = position_bias

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        if position_bias:
            self.pos = DynamicPosBias(self.embed_dim // 4, self.num_heads)
        
        self.qv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        
        #self.qkv_x2 = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, H, W, rpi, mask1=None, mask2=None):  #x=[16, 16*16, 32], x1=[16, 16*16, 32]
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x1.shape  #[16, 256, 32]
        group_size = (H, W)
        assert H * W == n
        
        ####################----------------Define[q, k, v]----------------####################
        qv = self.qv(x1).reshape(b_, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4) #[16,256,180]->[16,256,360]->[16,256,2,6,30]->[2,16,6,256,30] P:64800
        q, v = qv[0], qv[1]               #[16, 6, 256, 30] Each
        kv = self.kv(x2).reshape(b_, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4) #[16,256,180]->[16,256,540]->[16,256,3,6,30]-[3,16,6,256,30]
        k_h, v_h = kv[0], kv[1] #[16, 6, 256, 30] Each
        ####################----------------Define[q, k, v]----------------####################
        
        ###############-------------Define[qkv_high, q_high, k_high, v_high]-------------###############
        #qkv_high = self.qkv_x2(x2).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2,0,3,1,4)#[16,256,180]->[16,256,540]->[16,256,3,6,30]->[3,16,6,256,30] P:97200
        #q_high, k_high, v_high = qkv_high[0], qkv_high[1], qkv_high[2]   #[16, 6, 256, 30] Each
        ###############-------------Define[qkv_high, q_high, k_high, v_high]-------------###############
        
        #########--------Scaling--------#########
        q = q * self.scale                #scale=0.18257418583505536 #[16, 6, 256, 30]
        #########--------Scaling--------#########
        
        #########--------QxTransposed(K)--------#########
        attn = (q @ k_h.transpose(-2, -1))          #[16, 6, 256, 256]
        #########--------QxTransposed(K)--------#########
        
        ###############--------Relative_Positional_Bias_for_attn_1--------###############
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  
                                                                                       # Wh*Ww,Wh*Ww,nH #[256, 256, 6]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww #[6, 256, 256]
        attn1 = attn + relative_position_bias.unsqueeze(0)  #[16, 6, 256, 256]+[1, 6, 256, 256]=[16, 6, 256, 256]
        ###############--------Relative_Positional_Bias_for_attn_1--------###############
        
        ###############--------Relative_Positional_Bias_for_attn_2--------###############
        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw
            
            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn2 = attn + relative_position_bias.unsqueeze(0)
        ###############--------Add_Relative_Positional_Bias--------###############
        
        #########--------Masking_if_True--------#########
        if mask1 is not None:
            nw = mask1.shape[0]
            attn1 = attn1.view(b_ // nw, nw, self.num_heads, n, n) + mask1.unsqueeze(1).unsqueeze(0)
            attn1 = attn1.view(-1, self.num_heads, n, n)
            attn1 = self.softmax(attn1)
        else:
            attn1 = self.softmax(attn1)
        
        if mask2 is not None:
            nw = mask2.shape[0]
            attn2 = attn2.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn2 = attn2.view(-1, self.num_heads, n, n)
            attn2 = self.softmax(attn2)
        else:
            attn2 = self.softmax(attn2)
        #########--------Masking_if_True--------#########
        
        #########--------DropOut(Attn) $ (QK)xV--------#########
        attn1 = self.attn_drop(attn1)  #[16, 6, 256, 256]
        attn2 = self.attn_drop(attn2)  #[16, 6, 256, 256]
        #########--------DropOut(Attn) $ (QK)xV--------#########
        
        ############-----------(QK)xV_high------------############
        x1 = (attn1 @ v).transpose(1, 2).reshape(b_, n, c)  #[16, 6, 256, 256]*[16, 6, 256, 30]=[16, 6, 256, 30]->[16, 256, 180]
        x2 = (attn2 @ v_h).transpose(1, 2).reshape(b_, n, c)  #[16, 6, 256, 256]*[16, 6, 256, 30]=[16, 6, 256, 30]->[16, 256, 180]
        ############-----------(QK)xV------------############
        
        #########--------Linear+DropOut(Linear)--------#########
        x1 = self.proj1(x1)  #[16, 256, 180]      P:32580
        x1 = self.proj_drop(x1)  #[16, 256, 180]  P:0
        x2 = self.proj2(x2)  #[16, 256, 180]      P:32580
        x2 = self.proj_drop(x2)  #[16, 256, 180]  P:0
        #########--------Linear+DropOut(Linear)--------#########
        return x1, x2
########################------------Window Attention------------########################



############------window_partition------############
def window_partition(x, window_size):   #[1, 64, 64, 180]
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)    #[1, 64//16, 16, 64//16, 16, 180]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c) #[1, 64//16, 64//16, 16, 16, 180]->#[16, 16, 16, 180]
    return windows
############------window_partition------############



############------window_reverse------############
def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x
############------window_reverse------############



###############---------Overlapping window-based Channel Attention Module(OW_CAM)---------################
class OW_CAM(nn.Module):
    def __init__(self, 
                 embed_dim,
                 input_resolution,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=2,
                 norm_layer=nn.LayerNorm,
                 num_experts=4):

        super(OW_CAM, self).__init__()
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.qv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size),
                                stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) ** 2, num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)

        self.norm3 = norm_layer(embed_dim)
        self.norm4 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp1 = MoE(embed_dim, mlp_hidden_dim, embed_dim, num_experts=num_experts)
        self.mlp2 = MoE(embed_dim, mlp_hidden_dim, embed_dim, num_experts=num_experts)

    def forward(self, x1, x2, x_size, rpi):
        h, w = x_size
        b, _, c = x1.shape

        shortcut1, shortcut2 = x1, x2

        x1 = self.norm1(x1).view(b, h, w, c)
        x2 = self.norm2(x2).view(b, h, w, c)

        qv = self.qv(x1).reshape(b, h, w, 2, c).permute(3, 0, 4, 1, 2)
        kv = self.kv(x2).reshape(b, h, w, 2, c).permute(3, 0, 4, 1, 2)
        q = qv[0].permute(0, 2, 3, 1)
        kv = torch.cat((kv[0], kv[1]), dim=1) # b, 2*c, h, w

        q_windows = window_partition(q, self.window_size)
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)

        qv_windows = self.unfold(qv[1])
        qv_windows = rearrange(qv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch',
                               nc=1, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size)[0]

        kv_windows = self.unfold(kv)
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch',
                               nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size)
        k_windows, v_windows = kv_windows[0], kv_windows[1]

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.embed_dim // self.num_heads

        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        v_h = qv_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size,
            self.overlap_win_size * self.overlap_win_size,
            -1).permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x1 = (attn @ v).transpose(1, 2).reshape(b_, nq, self.embed_dim)
        x2 = (attn @ v_h).transpose(1, 2).reshape(b_, nq, self.embed_dim)

        x1 = x1.view(-1, self.window_size, self.window_size, self.embed_dim)
        x2 = x2.view(-1, self.window_size, self.window_size, self.embed_dim)
        x1 = window_reverse(x1, self.window_size, h, w).view(b, h * w, self.embed_dim)
        x2 = window_reverse(x2, self.window_size, h, w).view(b, h * w, self.embed_dim)

        x1 = self.proj1(x1) + shortcut1
        x2 = self.proj2(x2) + shortcut2

        x1_moe, lb1 = self.mlp1(self.norm3(x1))
        x2_moe, lb2 = self.mlp2(self.norm4(x2))

        x1 = x1 + x1_moe
        x2 = x2 + x2_moe

        load_balance_loss = (lb1 + lb2) / 2
        return x1, x2, load_balance_loss





###############------Triangular_Window_Partition------###############
def window_partition_triangular(x, window_size, masks):   #[1, 64, 64, 180]
    b, h, w, c = x.shape
    m = len(masks)
    ws = window_size
    h_ws = h // ws
    w_ws = w // ws
    x = x.view(b, h_ws, ws, w_ws, ws, c)    #b, h/ws, ws, w/ws, ws, c
    windows = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(-1, ws, ws) #b, h/ws, w/ws, c, ws, ws-->b*(h_ws)*(w_ws)*c, ws, ws
    #window_mask=torch.zeros((len(masks), windows.shape[0], ws//2 * ws//2), dtype=windows.dtype).to(x.device)
    window_masks = []
    for mask in masks:
        mask = mask.expand(windows.shape[0], -1, -1)
        window_mask = windows[mask]
        window_masks.append(window_mask.unsqueeze(0))
    window_masks = torch.cat(window_masks, dim=0)
    window_masks = window_masks.view(m, windows.shape[0], -1)
    m, _, n = window_masks.shape
    window_masks = window_masks.view(m, -1, c, n).permute(1, 0, 3, 2).contiguous()  #[m, b*(h_ws)*(w_ws)*c, n]->[b*(h_ws)*(w_ws), m, n, c]
    return window_masks
###############------Triangular_Window_Partition------###############



###############------Triangular_Window_Reverse------###############
def window_reverse_triangular(x, window_size, masks):
    b_, m, n, c = x.shape   #[b*(h_ws)*(w_ws), m, n, c]
    x = x.permute(1, 0, 3, 2).contiguous().view(m, -1)  #[m, b*(h_ws)*(w_ws)*c, n]
    reconstructed = torch.zeros((b_*c, window_size, window_size), dtype=x.dtype).to(x.device)
    for mask, x_ in zip(masks, x):
        mask = mask.expand(b_*c, -1, -1)
        reconstructed[mask] = x_   #[b*(h_ws)*(w_ws)*c, ws, ws]
    return reconstructed
###############------Triangular_Window_Reverse------###############



###############---------Hybrid Cross Attention Module(HCAM=NSW-CAM+NDW-CAM)---------################
class HCAM(nn.Module):
    r""" Hybrid Cross Attention Module.

    Args:
        embed_dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 embed_dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 interval=4,
                 ds_flag=0,
                 shift_size=0,
                 triangular_flag = 0,
                 fraction=3,
                 factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(HCAM, self).__init__()
        #print(dim, input_resolution, num_heads, window_size, shift_size, compress_ratio, squeeze_factor, conv_scale)
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.triangular_flag = triangular_flag
        self.mlp_ratio = mlp_ratio
        self.interval = interval
        self.ds_flag = ds_flag

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        #assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.norm3 = norm_layer(embed_dim)
        self.norm4 = norm_layer(embed_dim)
        
        self.attn = WindowAttention(embed_dim,
                                    window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop,
                                    position_bias=True)

        self.conv_scale = conv_scale
        self.conv_block1 = MFAB(embed_dim, fraction=fraction, factor=factor)
        self.conv_block2 = MFAB(embed_dim, fraction=fraction, factor=factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x1, x2, x_size, rpi_sa, attn_mask, triangular_masks):  #x, x_l, x_h = [2,4096,180], x_size = (64, 64)  #P:753012=0.75M
        h, w = x_size
        b, n_, c = x1.shape
        assert n_ == h * w, "input feature has wrong size"
        if min(h, w) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.window_size = min(h, w)
        
        ####----ShortCut----####
        shortcut1 = x1
        shortcut2 = x2
        ####----ShortCut----####
        
        ####----LN----####
        x1 = self.norm1(x1)  #[2, 4096, 180]  P:360
        x2 = self.norm2(x2)  #[2, 4096, 180]  P:360
        ####----LN----####

        ####--------####
        x1 = x1.view(b, h, w, c)   #[2, 64, 64, 180]
        x2 = x2.view(b, h, w, c)   #[2, 64, 64, 180]
        ####--------####

        ####----MFAB----####
        conv_x1 = self.conv_block1(x1.permute(0, 3, 1, 2))    #[2, 180, 64, 64]  P:196866
        conv_x2 = self.conv_block2(x2.permute(0, 3, 1, 2))   #[2, 180, 64, 64]  P:196866
        conv_x1 = conv_x1.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)   #[2, 4096, 180]
        conv_x2 = conv_x2.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)   #[2, 4096, 180]
        ####----MFAB----####
        
        
        ################################-------shifted_window_partitioning_for_x1-------################################
        # cyclic shift
        if self.shift_size > 0:
            shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.shift_size == 8:
                attn_mask1 = attn_mask[0]
            if self.shift_size == 16:
                attn_mask1 = attn_mask[1]
            if self.shift_size == 24:
                attn_mask1 = attn_mask[2]
        else:
            shifted_x1 = x1     #[1, 64, 64, 180]
            attn_mask1 = None
        # partition windows
        if not self.triangular_flag:
            x1_windows = window_partition(shifted_x1, self.window_size) # nw*b, window_size, window_size, c #[2, 64, 64, 180]->[32, 16, 16, 180]
            x1_windows = x1_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c #[32, 16*16, 180]

        if self.triangular_flag:
            x1_windows  = window_partition_triangular(shifted_x1, 2*self.window_size, triangular_masks)  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
            _, m, n, _ = x1_windows.shape           #[b*(h_ws)*(w_ws), m, n, c]
            x1_windows  = x1_windows.view(-1, n, c)  #[b*(h_ws)*(w_ws)*m, n, c]  #[16, 16*16, 180]
        ###############################-------shifted_window_partitioning_for_x1-------################################
        
        ###############################-------dilated_window_partitioning_for_x2-------################################
        ########-------padding-------########
        size_par = self.interval if self.ds_flag == 1 else self.window_size
        pad_l = pad_t = 0
        pad_r = (size_par - w % size_par) % size_par
        pad_b = (size_par - h % size_par) % size_par
        x2 = F.pad(x2, (0, 0, pad_l, pad_r, pad_t, pad_b))
        B, Hd, Wd, C = x2.shape
        ########-------padding-------########
        
        ########-------mask-------########
        mask = torch.zeros((1, Hd, Wd, 1), device=x2.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1
        ########-------mask-------########

        ########-------Partition_Dense_Attention+attn_mask-------########
        if self.ds_flag == 0:  
            G = Gh = Gw = self.window_size
            x2 = x2.reshape(B, Hd // G, G, Wd // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            x2_windows = x2.reshape(B * Hd * Wd // G ** 2, G ** 2, C)
            nP = Hd * Wd // G ** 2 # number of partitioning groups
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Hd // G, G, Wd // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
                mask = mask.reshape(nP, 1, G * G)
                attn_mask2 = torch.zeros((nP, G * G, G * G), device=x1.device)
                attn_mask2 = attn_mask2.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask2 = None
        ########-------Partition_Dense_Attention+attn_mask-------########
        
        ########-------Partition_Sparse_Attention+attn_mask-------########
        if self.ds_flag == 1:
            I, Gh, Gw = self.interval, Hd // self.interval, Wd // self.interval
            x2 = x2.reshape(B, Gh, I, Gw, I, C).permute(0, 2, 4, 1, 3, 5).contiguous()
            x2_windows = x2.reshape(B * I * I, Gh * Gw, C)
            nP = I ** 2  # number of partitioning groups
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Gh, I, Gw, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
                mask = mask.reshape(nP, 1, Gh * Gw)
                attn_mask2 = torch.zeros((nP, Gh * Gw, Gh * Gw), device=x1.device)
                attn_mask2 = attn_mask2.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask2 = None
        ########-------Partition_Sparse_Attention+attn_mask-------########
        ###############################-------dilated_window_partitioning_for_x2-------################################
        
        
        attn_windows1, attn_windows2 = self.attn(x1_windows, x2_windows, Gh, Gw, rpi=rpi_sa, mask1=attn_mask1, mask2=attn_mask2)  #[32, 256, 180]  P:227160
        
        
        ###############################-------dilated_window_merging_for_x2-------################################
        ########-------Merge_Dense_Attention&Sparse_Attention-------########
        if self.ds_flag == 0:
            attn_x2 = attn_windows2.reshape(B, Hd // G, Wd // G, G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()  # B, Hd//G, G, Wd//G, G, C
        else:
            attn_x2 = attn_windows2.reshape(B, I, I, Gh, Gw, C).permute(0, 3, 1, 4, 2, 5).contiguous()  # B, Gh, I, Gw, I, C
        attn_x2 = attn_x2.reshape(B, Hd, Wd, C)
        ########-------Merge_Dense_Attention&Sparse_Attention-------########
        
        ########-------remove_padding-------########
        if pad_r > 0 or pad_b > 0:
            attn_x2 = attn_x2[:, :h, :w, :].contiguous()
        attn_x2 = attn_x2.view(B, h * w, C)   #h,w of x
        ########-------remove_padding-------########
        ###############################-------dilated_window_merging_for_x2-------################################
        
        
        ################################-------shifted_window_merging_for_x1-------################################
        if not self.triangular_flag:
            attn_windows1 = attn_windows1.view(-1, self.window_size, self.window_size, c)  #[32, 16, 16, 180]
            shifted_x1 = window_reverse(attn_windows1, self.window_size, h, w)  # b h' w' c  #[32, 16, 16, 180]->[2, 64, 64, 180]
        if self.triangular_flag:
            attn_windows1 = attn_windows1.view(-1, m, n, c)   #[b*(h_ws)*(w_ws), m, n, c]
            shifted_x1  = window_reverse_triangular(attn_windows1, 2*self.window_size, triangular_masks)  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
            shifted_x1 = shifted_x1.view(b, h // (2*self.window_size), w // (2*self.window_size), c, 2*self.window_size, 2*self.window_size)
            shifted_x1 = shifted_x1.permute(0, 1, 4, 2, 5, 3).contiguous().view(b, h, w, c)   #[1, 64, 64, 180]
        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x1 = torch.roll(shifted_x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x1 = shifted_x1
        attn_x1 = attn_x1.view(b, h * w, c)  #[2, 64, 64, 180]->[2, 4096, 180]
        ##################################-------shifted_window_merging_for_x1-------################################

        ####--------####
        x1 = shortcut1 + self.drop_path(attn_x1) + conv_x1 * self.conv_scale  #[2, 4096, 180]
        x2 = shortcut2 + self.drop_path(attn_x2) + conv_x2 * self.conv_scale  #[2, 4096, 180]
        ####--------####

        ####----LN+MLP----####
        x1 = x1 + self.drop_path(self.mlp1(self.norm3(x1)))  #[2, 4096, 180]  P:360+32580+32580
        x2 = x2 + self.drop_path(self.mlp2(self.norm4(x2)))  #[2, 4096, 180]  P:360+32580+32580 
        ####----LN+MLP----####

        
        return x1, x2   #[2, 4096, 180]
###############---------Hybrid Cross Attention Module(HCAM)---------################


####################-------------MFCT-------------######################
class mfct(nn.Module):
    def __init__(self, 
                 in_channels, 
                 img_size,
                 num_layers,
                 patch_size=1,
                 window_size=16,
                 shift_size = (0, 0, 8, 8, 16, 16, 24, 24),  #if num_layers=8 then change shift_size to(0, 0, 8, 8, 16, 16) and pass shift_size=(8,16,24) while calculating attn_mask, change in HCAM
                 interval=4,  #3 if input_size=48 and 4 if input_size=64
                 fraction=3, 
                 factor=30,
                 num_heads=6, 
                 overlap_ratio=0.5,
                 mlp_ratio=1., 
                 qkv_bias=False, 
                 qk_scale=None,
                 proj_drop=0., 
                 attn_drop=0., 
                 mlp_path=0., 
                 HAB_drop=0., 
                 act_layer=nn.ReLU, 
                 norm_layer=nn.LayerNorm, 
                 conv_scale=0.01, 
                 drop_path_rate=0.1,
                 patch_norm=True, 
                 pos_emb=False):
        super(mfct, self).__init__()
        
        embed_dim =in_channels
        self.conv_scale = conv_scale
        self.patch_norm = patch_norm
        self.pos_emb = pos_emb
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.shift_size = shift_size
        self.num_features = in_channels
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)

        ##Relative_positional_Encoding::relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)

        ##Patch_Embed::split image into non-overlapping patches
        self.patch_embed1 = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size, 
                                      in_chans=embed_dim, 
                                      embed_dim=embed_dim, 
                                      norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed2 = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size, 
                                      in_chans=embed_dim, 
                                      embed_dim=embed_dim, 
                                      norm_layer=norm_layer if self.patch_norm else None)
        
        ##Required for absolute positional encoding and HCAM
        num_patches = self.patch_embed1.num_patches
        patches_resolution = self.patch_embed1.patches_resolution
        self.patches_resolution = patches_resolution

        ##Patch_Unembed::merge non-overlapping patches into image
        self.patch_unembed1 = PatchUnEmbed(img_size=img_size, 
                                          patch_size=patch_size, 
                                          in_chans=embed_dim, 
                                          embed_dim=embed_dim, 
                                          norm_layer=norm_layer if self.patch_norm else None)
        self.patch_unembed2 = PatchUnEmbed(img_size=img_size, 
                                          patch_size=patch_size, 
                                          in_chans=embed_dim, 
                                          embed_dim=embed_dim, 
                                          norm_layer=norm_layer if self.patch_norm else None)
        
        ##Absolute_Position_Embedding
        if self.pos_emb:
            self.absolute_pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) #(1, 4096, 180)
            trunc_normal_(self.absolute_pos_embed1, std=.02)
            self.absolute_pos_embed2 = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) #(1, 4096, 180)
            trunc_normal_(self.absolute_pos_embed2, std=.02)
        self.pos_drop = nn.Dropout(p=attn_drop)
        
        ## stochastic depth
        ##block_number=interation of this block
        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(block_number*num_layers))]  # stochastic depth decay rule
        
        ##Transformer_Blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(HCAM(embed_dim=embed_dim,
                                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    interval=interval,
                                    ds_flag = 0 if (i % 2 == 0) else 1,
                                    #shift_size=0 if (i % 2 == 0) else window_size // 2,
                                    shift_size=shift_size[i],
                                    triangular_flag = 0 if (i%2==0) else 1,
                                    fraction=fraction,
                                    factor=factor,
                                    conv_scale=conv_scale,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=HAB_drop,
                                    attn_drop=attn_drop,
                                    drop_path=0.,       #drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
                                    norm_layer=norm_layer))
        
        self.strided_attn = OW_CAM(embed_dim=embed_dim,
                                      input_resolution=(patches_resolution[0], patches_resolution[1]),
                                      window_size=window_size,
                                      overlap_ratio=overlap_ratio,
                                      num_heads=num_heads,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qk_scale,
                                      mlp_ratio=mlp_ratio,
                                      norm_layer=norm_layer)
        
        ##Layer_Normalization
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        
    #########---------calculate relative position index for SA--------#########
    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        
        return relative_position_index
    #########---------calculate relative position index for SA--------#########
    
    #########---------calculate relative position index for SA--------#########
    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index
    #########---------calculate relative position index for SA--------#########
    
    ###########----------calculate attention mask for SW-MSA---------###########
    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
    

    ######----Attention_Mask_for_HAB(SW-MSA)----######
    def calculate_mask1(self, x_size, shift_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,-shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,-shift_size), slice(-shift_size, None))
        cnt = 0
        for h_s in h_slices:
         for w_s in w_slices:
             img_mask[:, h_s, w_s, :] = cnt
             cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    ######----Attention_Mask_for_HAB----######


    ###############------Triangular_Window_Mask------###############
    def triangle_masks(self, x):
        ws = 2*self.window_size
        rows = torch.arange(ws).unsqueeze(1).repeat(1, ws)
        cols = torch.arange(ws).unsqueeze(0).repeat(ws, 1)
    
        upper_triangle_mask = (cols > rows) & (rows + cols < ws)
        right_triangle_mask = (cols >= rows) & (rows + cols >= ws)
        bottom_triangle_mask = (cols < rows) & (rows + cols >= ws-1)
        left_triangle_mask = (cols <= rows) & (rows + cols < ws-1)
    
        return [upper_triangle_mask.to(x.device), right_triangle_mask.to(x.device), bottom_triangle_mask.to(x.device), left_triangle_mask.to(x.device)]
    ###############------Triangular_Window_Mask------###############

    
    ###########----------calculate attention mask for SW-MSA---------###########

    def forward(self, x1, x2):   #x, x_h=[2, 180, 64, 64]  #P:755532=0.75M
        B, C, H, W = x1.shape
        x_size = (x1.shape[2], x1.shape[3])

        #attn_mask = self.calculate_mask(x_size).to(x1.device)  #[16, 256, 256]
        attn_mask = attn_masks = [self.calculate_mask1(x_size, shift_size).to(x1.device) for shift_size in (8, 16, 24)]
        triangular_masks = tuple(self.triangle_masks(x1))  #[16, 256, 256]   #changed to tuple
        params = {'attn_mask': attn_mask, 'triangular_masks': triangular_masks, 'rpi_sa': self.relative_position_index_SA, 'rpi_oca': self.relative_position_index_OCA}

        ## Embed
        x1 = self.patch_embed1(x1)              #[2, 180, 64, 64]->[2, 4096, 180] P:360  Faltten+LayerNorm 
        x2 = self.patch_embed2(x2)           #[2, 180, 64, 64]->[2, 4096, 180] P:360   
                       
        ##Positional Embedding
        if self.pos_emb:
            x1 = x1 + self.absolute_pos_embed1    #[2, 4096, 180]  P:0
            x2 = x2 + self.absolute_pos_embed2 #[2, 4096, 180]  P:0
        
        ##Drop_out
        x1 = self.pos_drop(x1)                #[2, 4096, 180]  P:0
        x2 = self.pos_drop(x2)              #[2, 4096, 180]  P:0
        
        for block in self.blocks:
            x1, x2 = block(x1, x2, x_size, params['rpi_sa'], params['attn_mask'], params['triangular_masks'])    #[2, 4096, 180]  P:753012
        
        x1, x2, load_balance_loss  = self.strided_attn(x1, x2, x_size, params['rpi_oca'])  #[1, 4096, 180]->[1, 4096, 180]
        
        x1 = self.norm1(x1)  # b seq_len c     #[2, 4096, 32]  P:360
        x2 = self.norm2(x2)  # b seq_len c     #[2, 4096, 32]  P:360
        ## Unembed
        x1 = self.patch_unembed1(x1, x_size)   #[2, 4096, 32]->[2, 32, 64, 64]  P:360
        x2 = self.patch_unembed2(x2, x_size)   #[2, 4096, 32]->[2, 32, 64, 64]  P:360
        
        return x1, x2, load_balance_loss
####################-------------MFCT-------------######################