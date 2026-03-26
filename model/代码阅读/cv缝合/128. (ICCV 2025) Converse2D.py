import torch
import torch.nn as nn

class Converse2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale=1, padding=2, padding_mode='circular', eps=1e-5):                                                                                                                                                                     # 微信公众号:CV缝合救星
        super(Converse2D, self).__init__()
        """
        Converse2D Operator for Image Restoration Tasks.

        Args:
            x (Tensor): Input tensor of shape (N, in_channels, H, W), where
                        N is the batch size, H and W are spatial dimensions.
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the operation.
            kernel_size (int): Size of the kernel.
            scale (int): Upsampling factor. For example, `scale=2` doubles the resolution.
            padding (int): Padding size. Recommended value is `kernel_size - 1`.
            padding_mode (str, optional): Padding method. One of {'reflect', 'replicate', 'circular', 'constant'}.
                                        Default is `circular`.
            eps (float, optional): Small value added to denominators for numerical stability.
                                Default is a small value like 1e-5.

        Returns:
            Tensor: Output tensor of shape (N, out_channels, H * scale, W * scale), where spatial dimensions
                    are upsampled by the given scale factor.
        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size =  kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps


        # ensure depthwise
        assert self.out_channels == self.in_channels
        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        self.weight.data = nn.functional.softmax(self.weight.data.view(1,self.in_channels,-1), dim=-1).view(1, self.in_channels, self.kernel_size, self.kernel_size)                                                                                                                                                                     # 微信公众号:CV缝合救星

        
    def forward(self, x):

        if self.padding > 0:
            x = nn.functional.pad(x, pad=[self.padding, self.padding, self.padding, self.padding], mode=self.padding_mode, value=0)                                                                                                                                                                     # 微信公众号:CV缝合救星

        self.biaseps = torch.sigmoid(self.bias-9.0) + self.eps
        _, _, h, w = x.shape
        STy = self.upsample(x, scale=self.scale)
        if self.scale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')                                                                                                                                                                     # 微信公众号:CV缝合救星
            # x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear',align_corners=False)
        # x = torch.zeros_like(x)

        FB = self.p2o(self.weight, (h*self.scale, w*self.scale))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
        
        FR = FBFy + torch.fft.fftn(self.biaseps*x, dim=(-2,-1))
        x1 = FB.mul(FR)
        FBR = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)                                                                                                                                                                     # 微信公众号:CV缝合救星
        invW = torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)                                                                                                                                                                     # 微信公众号:CV缝合救星
        invWBR = FBR.div(invW + self.biaseps)
        FCBinvWBR = FBC*invWBR.repeat(1, 1, self.scale, self.scale)
        FX = (FR-FCBinvWBR)/self.biaseps
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if self.padding > 0:
            out = out[..., self.padding*self.scale:-self.padding*self.scale, self.padding*self.scale:-self.padding*self.scale]                                                                                                                                                                     # 微信公众号:CV缝合救星

        return out

    def splits(self, a, scale):
        '''
        Split tensor `a` into `scale x scale` distinct blocks.                                                                                                                                                                     # 微信公众号:CV缝合救星
        Args:
            a: Tensor of shape (..., W, H)
            scale: Split factor
        Returns:
            b: Tensor of shape (..., W/scale, H/scale, scale^2)                                                                                                                                                                     # 微信公众号:CV缝合救星
        '''
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale

        # Reshape to separate the scale factors
        b = a.view(*leading_dims, scale, W_s, scale, H_s)

        # Generate the permutation order
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1, len(leading_dims) + 3, len(leading_dims), len(leading_dims) + 2]                                                                                                                                                                     # 微信公众号:CV缝合救星
        b = b.permute(*permute_order).contiguous()

        # Combine the scale dimensions
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b


    def p2o(self, psf, shape):
        '''
        Convert point-spread function to optical transfer function.
        otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
        point-spread function (PSF) array and creates the optical transfer
        function (OTF) array that is not influenced by the PSF off-centering.
        Args:
            psf: NxCxhxw
            shape: [H, W]
        Returns:
            otf: NxCxHxWx2
        '''
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[...,:psf.shape[-2],:psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2]/2), -int(psf.shape[-1]/2)), dims=(-2, -1))                                                                                                                                                                     # 微信公众号:CV缝合救星
        otf = torch.fft.fftn(otf, dim=(-2,-1))

        return otf

    def upsample(self, x, scale=3):
        '''s-fold upsampler
        Upsampling the spatial size by filling the new entries with zeros                                                                                                                                                                     # 微信公众号:CV缝合救星
        x: tensor image, NxCxWxH
        '''
        st = 0
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*scale, x.shape[3]*scale)).type_as(x)                                                                                                                                                                     # 微信公众号:CV缝合救星
        z[..., st::scale, st::scale].copy_(x)
        return z
    
if __name__ == "__main__":

    # 输入张量：形状为 (B, C, H, W)
    x = torch.randn(1, 32, 64, 64)  # batch=1, 通道=32, 高=64, 宽=64

    # 初始化（这里设置 in_channels=out_channels=32, kernel_size=5, scale=2）
    converse = Converse2D(in_channels=32, out_channels=32, kernel_size=5, scale=2)                                                                                                                                                                     # 微信公众号:CV缝合救星

    # 前向传播测试
    output = converse(x)

    # 输出结果形状
    print(converse)
    print("\n微信公众号:CV缝合救星\n")
    print("输入张量形状:", x.shape)       # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, C, H*scale, W*scale]
