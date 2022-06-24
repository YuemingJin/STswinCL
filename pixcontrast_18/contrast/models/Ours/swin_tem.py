import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows ## shape: (num_windows*B, window_size, window_size, C)


'''def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x ## turn windows into image features'''

def window_reverse(windows, window_size, H, W, T):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, T, window_size, window_size, -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H, W, -1)
    return x ## turn windows into image features

class WindowAttention(nn.Module):

	def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

		super().__init__()
		self.dim = dim
		self.window_size = window_size  # Wh, Ww
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim ** -0.5

		# define a parameter table of relative position bias
		self.relative_position_bias_table = nn.Parameter(
			torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

		# get pair-wise relative position index for each token inside the window
		coords_h = torch.arange(self.window_size[0])
		coords_w = torch.arange(self.window_size[1])
		coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
		coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
		relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
		relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
		relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
		relative_coords[:, :, 1] += self.window_size[1] - 1
		relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
		relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
		self.register_buffer("relative_position_index", relative_position_index)

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		trunc_normal_(self.relative_position_bias_table, std=.02)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x_v, mask=None):
		"""
		Args:
			x: input features with shape of (num_windows*B, T, N, C)
			mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
		"""
		B_, T, N, C = x_v.shape
		qkv = self.qkv(x_v.reshape(-1, N, C)).reshape(B_, T*N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#.contiguous()
		q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) ## (B, NUM_HEADS, T*N, C // NUM_HEADS)

		q = q * self.scale
		attn = (q @ k.transpose(-2, -1)) ## (B, NUM_HEADS, T*N, T*N)

		relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
			self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().repeat(1, T, T) # nH, T*Wh*Ww, T*Wh*Ww
		attn = attn + relative_position_bias.unsqueeze(0)

		if mask is not None:
			mask = mask.repeat(1, T, T) #nW, T*N, T*N
			nW = mask.shape[0]
			attn = attn.view(B_ // nW, nW, self.num_heads, T*N, T*N) + mask.unsqueeze(1).unsqueeze(0)
			attn = attn.view(-1, self.num_heads, T*N, T*N)
			attn = self.softmax(attn)
		else:
			attn = self.softmax(attn)

		attn = self.attn_drop(attn)

		x_v = (attn @ v).transpose(1, 2).reshape(B_, T, N, C)
		x_v = self.proj(x_v)
		x_v = self.proj_drop(x_v)
		return x_v

class SwinTransformerBlock(nn.Module):

	def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
					mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
					act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()
		self.dim = dim
		self.input_resolution = input_resolution
		self.num_heads = num_heads
		self.window_size = window_size
		self.shift_size = shift_size
		self.mlp_ratio = mlp_ratio
		if min(self.input_resolution) <= self.window_size:
			# if window size is larger than input resolution, we don't partition windows
			self.shift_size = 0
			self.window_size = min(self.input_resolution)
		assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

		self.norm1 = norm_layer(dim)
		self.attn = WindowAttention(
			dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
			qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

		if self.shift_size > 0:
			# calculate attention mask for SW-MSA
			H, W = self.input_resolution
			img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
			h_slices = (slice(0, -self.window_size),
						slice(-self.window_size, -self.shift_size),
						slice(-self.shift_size, None))
			w_slices = (slice(0, -self.window_size),
						slice(-self.window_size, -self.shift_size),
						slice(-self.shift_size, None))
			cnt = 0
			for h in h_slices:
				for w in w_slices:
					img_mask[:, h, w, :] = cnt
					cnt += 1

			mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
			mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
			attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
			attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
		else:
			attn_mask = None

		self.register_buffer("attn_mask", attn_mask)

	def forward(self, x_v): # x_v: B, T, L, C
		H, W = self.input_resolution
		x_v = x_v.contiguous()
		B, T, L, C = x_v.shape
		assert L == H * W, "input feature has wrong size"
		assert T == 2, "input feature has wrong size"

		
		shortcut = x_v.view(B*T, L, C)
		#x_v = self.norm1(x_v)
		x_v = x_v.view(B*T, H, W, C)
		

		# cyclic shift
		if self.shift_size > 0:
			shifted_x = torch.roll(x_v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
		else:
			shifted_x = x_v

		# partition windows
		x_windows = window_partition(shifted_x, self.window_size)  # B*T*nW, window_size, window_size, C
		x_windows = x_windows.view(B, T, -1, self.window_size * self.window_size, C)  # B, T, nW, window_size*window_size, C
		x_windows = x_windows.permute(0, 2, 1, 3, 4).contiguous().view(-1, T, self.window_size * self.window_size, C) # B*nW, T, window_size*window_size, C

		# W-MSA/SW-MSA
		attn_windows = self.attn(x_windows, mask=self.attn_mask)  # B*nW, T, window_size*window_size, C

		# merge windows
		shifted_x = window_reverse(attn_windows, self.window_size, H, W, T)  # B T H' W' C
		shifted_x = shifted_x.view(B*T, H, W, C)
		# reverse cyclic shift
		if self.shift_size > 0:
			x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
		else:
			x = shifted_x
		x = x.view(B*T, H * W, C)

		# FFN
		x = shortcut + self.drop_path(x)
		x = self.norm1(x + self.drop_path(self.mlp(self.norm2(x))))
		x = x.view(B, T, L, C)
		return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, T, L, C = x.shape
        assert T == 4, "wrong time dimension"
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B*T, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B*T, -1, 4 * C)  # B*T H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x.view(B, T, L//4, 2*C)


class SwinTransformerLayerv5(nn.Module):
	def __init__(self, dim=512, input_resolution=(32,56), num_heads=4):
		super().__init__()
		self.dim = dim
		self.input_resolution = input_resolution
		self.num_heads = num_heads
		self.num_layers = 3
		#self.pairs = [[slice(0, 2), slice(2, 4), slice(4, 6)], [slice(1, 3), slice(3, 5)]]
		#self.pairs = [[slice(0, 2),slice(2, 4)], [slice(1, 3), slice(3, 5)]] #t=5
		self.pairs = [[slice(0, 2), slice(2, 4)], [slice(1, 3)], [slice(0, 2), slice(2, 4)]]  # t=4

		self.layers = nn.ModuleList()
		for i in range(self.num_layers):
			layer = nn.Sequential(SwinTransformerBlock(self.dim, self.input_resolution, self.num_heads),
									SwinTransformerBlock(self.dim, self.input_resolution, self.num_heads, shift_size=4))
			self.layers.append(layer)

		for i in range(self.num_layers):
			layer = nn.Sequential(SwinTransformerBlock(self.dim * 2, (self.input_resolution[0] // 2, self.input_resolution[1] // 2), self.num_heads, window_size=4),
									SwinTransformerBlock(self.dim * 2, (self.input_resolution[0] // 2, self.input_resolution[1] // 2), self.num_heads, window_size=4, shift_size=2))
			self.layers.append(layer)

		self.downsample = PatchMerging(self.input_resolution, self.dim)

	def _single_layer_forward(self, x_v, pairs, layer_idx):
		x_y = x_v.clone()
		for p in pairs:
			layer = self.layers[layer_idx]
			x_y[:,p,:,:] = layer(x_v[:,p,:,:])
		return x_y

	def forward(self, x_v): #x_v: B, T, C, H, W

		B, T, C, H, W = x_v.shape
		#print(x_v.shape)
		assert T == 4, "input feature has wrong size"
		x_v = x_v.permute(0, 1, 3, 4, 2).contiguous().view(B, T, H * W, C)
		x_layer1 = self._single_layer_forward(x_v, self.pairs[0], 0)
		x_layer2 = self._single_layer_forward(x_layer1, self.pairs[1], 1)
		x_layer3 = self._single_layer_forward(x_layer2, self.pairs[2], 2)

		x_layer3_output = x_layer3.permute(0, 1, 3, 2).contiguous().view(B, T, C, H, W)

		x_layer3 = self.downsample(x_layer3)

		x_layer4 = self._single_layer_forward(x_layer3, self.pairs[0], 3)
		x_layer5 = self._single_layer_forward(x_layer4, self.pairs[1], 4)
		x_layer6 = self._single_layer_forward(x_layer5, self.pairs[2], 5)
		x_layer6 = x_layer6.permute(0, 1, 3, 2).contiguous().view(B, T, 2 * C, H // 2, W // 2)
		return x_layer3_output, x_layer6


