import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from torch import nn, einsum
from einops import rearrange, repeat

#https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads = 4):
        super(SelfAttention, self).__init__()
        self.channels = channels
        #self.H,self.W = size # input image heights and weights
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
    def forward(self, x):
        # x is B x C x H x W ----> B x C x (H X W)--> B x (H x W) X C
        B, C, H, W = x.shape
        x = x.view(B, self.channels, H * W).swapaxes(1, 2)# specify layer norm axis to make it to the end
        #print(x.shape)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln) # keep the shape, as multihead attention
        attention_value = attention_value + x # Residual block
        # Residual block
        attention_value = self.ff_self(attention_value) + attention_value # Residual block
        return attention_value.swapaxes(2, 1).view(-1, self.channels, H, W)



class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 kernel_size = 3,
                 padding = 1,
                 residual=False):
        super().__init__()
        """ two convolutional layers to increase the channel depth, since the padding is 1, there is
        no change of shapes """
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # input shape (B x in_channels x H x W )
            nn.Conv2d(in_channels, mid_channels, kernel_size= kernel_size , padding=padding, bias=False),
            # equivalent with layer norm
            nn.GroupNorm(1, mid_channels),
            nn.GELU(), # can also be nn.SiLU()
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size , padding=padding, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 maxpool_size = 2,
                 maxpool_dilation = 1,
                 emb_dim=256):
        super().__init__()
        self.maxpool_size =  maxpool_size
        self.maxpool_dilation = 1
        self.emb_dim = emb_dim
        # note that https://github.com/CompVis/latent-diffusion/blob/5a6571e384f9a9b492bbfaca594a2b00cad55279/ldm/modules/diffusionmodules/openaimodel.py#L74
        # Open Ai also use average pooling

        self.maxpool_conv = nn.Sequential(
            #[B,64,64,64]--->[B,64,64,64]
            nn.MaxPool2d(kernel_size=self.maxpool_size, stride=self.maxpool_size, padding=0, dilation=self.maxpool_dilation),
            #nn.MaxPool2d(self.maxpool_size),
            #x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            DoubleConv(in_channels, in_channels, kernel_size = kernel_size, residual=True), #[B,64,64,64]--->[B,64,64,64]
            DoubleConv(in_channels, out_channels, kernel_size = kernel_size),#[B,64,64,64]--->[B,128,64,64]
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x) # image after downsample
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size =3, upscale_factor = 2,  emb_dim=256):
        super().__init__()
        self.scale_factor = upscale_factor
        #x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
        self.up = nn.Upsample(scale_factor=self.scale_factor, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, kernel_size = kernel_size, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size = kernel_size),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

def inout(dims):
    return(list(zip(dims[:-1], dims[1:])))


class UNet(nn.Module):
    def __init__(self,
                 c_in=3, # input image channel
                 c_out=3, # output image channel
                 time_dim=256, # time dimension
                 down_factor = 2, # sample factor used for maxpooling
                 num_heads = 4, # number of attention heads
                 kernel_size = 3, # kernel size
                 channels_down = [64, 128, 256, 256],# increase channel depth in Unet
                 num_classes=None,
                 multi_class = None,
                 device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.maxpool_size = down_factor# downsample by a factor of 2
        self.embed_dim = time_dim # embeddim is eventually the for time_step and other conditional generation
        ################################################################################
        # Attention layer and Conv layer set up
        ################################################################################
        self.num_heads = num_heads #
        self.kernel_size = kernel_size # uniform kernel size across conv2D
        ################################################################################
        # Depth of Unet
        ################################################################################
        self.channels_down = channels_down  # [64, 128, 256, 256]
        self.inc_dim = self.channels_down[0]
        self.bot_dim = self.channels_down[-1]
        self.channels_up = [self.inc_dim] + self.channels_down[:-1]
        self.down_dims = inout(self.channels_down)  # length should be dims_down -1
        self.up_dims = inout(self.channels_up)
        # inc
        self.inc = DoubleConv(in_channels = c_in,
                              out_channels = self.inc_dim,
                              mid_channels = None,
                              kernel_size = self.kernel_size,
                              padding = 1,
                              residual = False) # B x 64 x H x W
        ################################################################################
        # downsampling block: # Add downsample maxpooling parameter
        ################################################################################
        self.downs = nn.ModuleList([])
        # append a dowmsample layer and a selfAttention block
        for index, (in_channel, out_channel) in enumerate(self.down_dims):
            self.downs.append(
                nn.ModuleList([
                    # downsample from maxpooling + 2 conv2d
                    Down(in_channels = in_channel,
                         out_channels = out_channel,
                          kernel_size = self.kernel_size, # downsampling layer put 2 doubleconv in, this is to set the kernel size
                          maxpool_size = self.maxpool_size,
                          maxpool_dilation = 1,
                          emb_dim=  self.embed_dim
                          ),
                    SelfAttention(channels = out_channel, num_heads = self.num_heads),
                    ])

            )
        ################################################################################
        # Comment out non-modulized  code for reference
        ################################################################################
        # self.down1 = Down(in_channels = 64, out_channels = 128, maxpool_size = self.maxpool_size, embeddim = self.embed_dim) # B x 128 x H/2 x W/2 # image size get smaller but channel get larger
        # # size refers to height and weights after downsampling
        # self.sa1 = SelfAttention(channels = 128, num_heads = self.num_heads) # self attention:# B x 128 x H/2 x W/2,
        #
        # self.down2 = Down(in_channels =128, out_channels =256, maxpool_size = self.maxpool_size, embeddim = self.embed_dim)
        # self.sa2 = SelfAttention(channels = 256, num_heads = self.num_heads)
        #
        # self.down3 = Down(in_channels =256, out_channels =256,maxpool_size = self.maxpool_size, embeddim = self.embed_dim)
        # self.sa3 = SelfAttention(channels = 256, num_heads = self.num_heads)
        ################################################################################
        # Bottom layers of Unet
        ################################################################################
        self.bot1 = DoubleConv(in_channels =self.bot_dim, out_channels =self.bot_dim * 2, kernel_size = self.kernel_size, padding = 1)
        #self.sa_bot = SelfAttention(channels = self.bot_dim * 2, num_heads = self.num_heads)
        self.bot2 = DoubleConv(in_channels =self.bot_dim * 2, out_channels = self.bot_dim * 2,kernel_size = self.kernel_size, padding = 1)
        self.bot3 = DoubleConv(in_channels =self.bot_dim * 2, out_channels =self.bot_dim, kernel_size = self.kernel_size, padding = 1)
        ################################################################################
        # Upsampling block
        ################################################################################
        self.ups = nn.ModuleList([])
        for index, (dim_out, dim_in) in enumerate(reversed(self.up_dims)):
            #print((dim_in * 2, dim_out)) check
            self.ups.append(
                nn.ModuleList([
                    Up(in_channels = dim_in * self.maxpool_size, # for dimensions after upsamping (first upsample and then conv2D)
                                  out_channels =dim_out,
                                  kernel_size = self.kernel_size,
                                  upscale_factor = self.maxpool_size,
                                  emb_dim= self.embed_dim ),
                    SelfAttention(channels = dim_out, num_heads = self.num_heads),
                ])
            )
        ################################################################################
        # Comment out non-modulized  code for reference
        ################################################################################
        # self.up1 = Up(in_channels = 512,
        #               out_channels =128,
        #               kernel_size = self.kernel_size,
        #               upscale_factor = self.maxpool_size,
        #               emb_dim= self.embed_dim )
        # self.sa4 = SelfAttention(channels = 128)
        # self.up2 = Up(256, 64)
        # self.sa5 = SelfAttention(channels = 64)
        # self.up3 = Up(128, 64)
        # self.sa6 = SelfAttention(channels = 64)
        # outc
        self.outc = nn.Conv2d(self.inc_dim, c_out, kernel_size=1)
        ################################################################################
        # Adding conditions
        ################################################################################
        if not isinstance(num_classes, int) and num_classes is not None:
            assert f"number of class either is None or a interger for single condition"
        if not isinstance(multi_class, list) and multi_class is not None:
            assert f"multi_class either is None or a interger list for multiple condition"
        if num_classes is not None and multi_class is not None:
            assert f"specify either multiple condition or single condition "
        self.multi_class = multi_class
        self.num_classes = num_classes
        if num_classes is not None: # only one class
            self.label_emb = nn.Embedding(num_classes, time_dim)
        if multi_class is not None: # multilevel
            self.label_emb1 = nn.Embedding(multi_class[0], time_dim) # Age
            self.label_emb2 = nn.Embedding(multi_class[1], time_dim) # plate_date
            self.label_emb3 = nn.Embedding(multi_class[2], time_dim) # mutants
            self.concat_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    time_dim * 4,
                    time_dim
                ),
            )
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    #def add_att(self, Sampling):
        #for index, (down, sa) in enumerate(self.downs):
            #x_d= down(x_d, t)
            #x_d = sa(x_d)
    def forward(self, x, t, y = None):
        # time embedding
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        ## Dealing with multilabel class
        if y is not None:
            if self.num_classes is not None:
                t += self.label_emb(y)
            elif self.multi_class is not None: # Gerald: Concatenate ?
                emb1 = self.label_emb1(y[0])
                emb2 = self.label_emb2(y[1])
                emb3 = self.label_emb3(y[2])
                t = self.concat_layer(torch.cat([t, emb1, emb2, emb3], dim = 1))
        x_d = self.inc(x)
        ################################################################################
        # down sample and keep the input for up sample
        ###########################################lmao
        # #####################################
        #NOTE modulate this component to add attention

        down_ = [x_d] #x1
        for index, (down,sa) in enumerate(self.downs):
            x_d= down(x_d, t)
            x_d = sa(x_d)
            down_.append(x_d) # [x1, x2, x3, x4] if 4 layers
        down_ = down_[:-1] #[x1, x2, x3, x4]-->[x1,x2,x3] --->[x3,x2,x1]
        down_.reverse()
        ################################################################################
        # Comment out non-modulized  code for reference
        ################################################################################
        # x2 = self.down1(x1, t)
        # x2 = self.sa1(x2)
        #
        # x3 = self.down2(x2, t)
        # x3 = self.sa2(x3)
        #
        # x4 = self.down3(x3, t)
        # x4 = self.sa3(x4)
        ################################################################################
        # Bottom layers of Unet
        ################################################################################
        x_bot = self.bot1(x_d)
        x_bot = self.bot2(x_bot) # # can be replaced with attention
        #x_bot = self.sa_bot(x_bot)
        x_up = self.bot3(x_bot)
        ################################################################################
        # Comment out non-modulized  code for reference
        ################################################################################
        # x4 = self.bot1(x4)
        # x4 = self.bot2(x4)
        # x4 = self.bot3(x4)
        ################################################################################
        #  up sample
        ################################################################################
        for index, (up, sa) in enumerate(self.ups):
            x_up = up(x_up, down_[index] ,t)
            #print(x_up.shape)
            x_up = sa(x_up)
        ################################################################################
        # Comment out non-modulized  code for reference
        ################################################################################
        # x = self.up1(x4, x3, t)
        # x = self.sa4(x)
        # x = self.up2(x, x2, t)
        # x = self.sa5(x)
        # x = self.up3(x, x1, t)
        # x = self.sa6(x)
        ################################################################################
        # Output layers
        ################################################################################
        output = self.outc(x_up)
        return output






class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, multi_class = None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.num_classes = num_classes
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)
        self.bot1 = DoubleConv(256, 512)
        #self.bot2 = DoubleConv(512, 512)
        self.sa_bot = SelfAttention(512)
        self.bot3 = DoubleConv(512, 256)
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
        if not isinstance(num_classes, int) and num_classes is not None:
            assert f"number of class either is None or a interger for single condition"
        if not isinstance(multi_class, list) and multi_class is not None:
            assert f"multi_class either is None or a interger list for multiple condition"
        if num_classes is not None and multi_class is not None:
            assert f"specify either multiple condition or single condition "
        self.multi_class = multi_class
        self.num_classes = num_classes
        if num_classes is not None: # only one class
            self.label_emb = nn.Embedding(num_classes, time_dim)
        if multi_class is not None: # multilevel
            self.label_emb1 = nn.Embedding(multi_class[0], time_dim) # Age
            self.label_emb2 = nn.Embedding(multi_class[1], time_dim) # plate_date
            self.label_emb3 = nn.Embedding(multi_class[2], time_dim) # mutants
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        if y is not None:
            if self.num_classes is not None:
                t += self.label_emb(y)
            elif self.multi_class is not None: # Gerald: Concatenate ?
                if y[0] is not None:
                    t += self.label_emb1(y[0])
                if y[1] is not None:
                    t += self.label_emb2(y[1])
                if y[2] is not None:
                    t += self.label_emb3(y[2])
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x4 = self.bot1(x4)
        x4 = self.sa_bot(x4)#self.bot2(x4)
        x4 = self.bot3(x4)
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


################################################################################
# Different Attention Class
################################################################################
#https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/ldm/modules/attention.py
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels, kernel_size = 1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-6, affine=True)
        # query, key, value
        ###? if kernel size = 1, what is the point here ?
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=self.kernel_size,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=self.kernel_size,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=self.kernel_size,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=self.kernel_size,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False) # query
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False) # context key
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False) # context value

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_heads,
                 d_head,
                 dropout=0.,
                 context_dim=None,
                 gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        #self.norm = Normalize(in_channels)
        self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in



















# print(sum([p.numel() for p in net.parameters()]))
# x = torch.randn(3, 3, 64, 64)
# t = x.new_tensor([500] * x.shape[0]).long()
# y = x.new_tensor([1] * x.shape[0]).long()
# print(net(x, t, y).shape)