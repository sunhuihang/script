import torch.nn as nn
import torch
import torch.nn.functional as F


class SRAdaIN(torch.nn.Module):
    def __init__(self, in_channels, representation_dim):
        super(SRAdaIN, self).__init__()
        self.inns = torch.nn.InstanceNorm2d(in_channels, affine=False)
        self.compress_gamma = torch.nn.Sequential(
            torch.nn.Linear(representation_dim, in_channels, bias=False),
            torch.nn.LeakyReLU(0.1, True)
        )
        self.compress_beta = torch.nn.Sequential(
            torch.nn.Linear(representation_dim, in_channels, bias=False),
            torch.nn.LeakyReLU(0.1, True)
        )

    def forward(self, x, representation):
        gamma = self.compress_gamma(representation)
        beta = self.compress_beta(representation)

        b, c = gamma.size()

        gamma = gamma.view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)

        out = self.inns(x)
        out = out * gamma + beta
        return out


class SConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, representation_dim):
        super(SConv, self).__init__()
        reflection_padding = kernel_size // 2  # same dimension after padding
        self.reflection_padding = reflection_padding
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)

        self.kernel_size = kernel_size

        self.compress_key = torch.nn.Sequential(
            torch.nn.Linear(representation_dim, out_channels * kernel_size * kernel_size, bias=False),
            torch.nn.LeakyReLU(0.1, True)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

    def forward(self, x, representation):
        out = self.reflection_pad(x)

        b, c, h, w = out.size()
        kernel = self.compress_key(representation).view(b, self.out_channels, -1, self.kernel_size, self.kernel_size)

        # 1,64,1,kh,kw -> 1,64,4,kh,kw
        features_per_group = int(self.in_channels / self.groups)
        kernel = kernel.repeat_interleave(features_per_group, dim=2)

        # 1,64,4,kh,kw
        k_batch, k_outputchannel, k_feature_pergroup, kh, kw = kernel.size()

        out = F.conv2d(out.view(1, -1, h, w), kernel.view(-1, k_feature_pergroup, kh, kw), groups=b * self.groups,
                       padding=0)

        b, c, h, w = x.size()
        out = out.view(b, -1, h, w)

        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2  # same dimension after padding
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)  # remember this dimension

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class SCM(nn.Module):
    def __init__(self, representation_dim, channels_out, reduction):
        super(SCM, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(representation_dim, representation_dim // reduction, 1, 1, 0, bias=False),
            nn.PReLU(),
            nn.Conv2d(representation_dim // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.conv_du(x[1][:, :, None, None])
        return x[0] * att


class GatingMechanism(nn.Module):
    def __init__(self, representation_dim):
        super(GatingMechanism, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(representation_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, representation):
        gate = self.fc(representation)
        return gate


class SAB(torch.nn.Module):
    def __init__(self, in_channels, out_channels, groups, representation_dim):
        super(SAB, self).__init__()

        # branch 2 begin-----------------------------------------
        self.conv1 = SConv(in_channels, out_channels, kernel_size=3, groups=groups,
                           representation_dim=representation_dim)
        self.cm1 = SRAdaIN(out_channels, representation_dim)
        self.relu = torch.nn.ReLU()
        # branch 1 end-----------------------------------------

        # branch 2 begin-----------------------------------------
        self.ca = SCM(representation_dim=representation_dim, channels_out=out_channels, reduction=4)
        # branch 2 end-----------------------------------------

        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)

        self.gating = GatingMechanism(representation_dim)

    def forward(self, x, representation):
        residual = x

        # branch 1 begin-----------------------------------------
        out1 = self.conv1(x, representation)
        out1 = self.cm1(out1, representation)
        out1 = self.relu(out1)
        # branch 1 end-----------------------------------------

        # branch 2-----------------------------------------
        out2 = self.ca([residual, representation])
        # branch 2-----------------------------------------

        gate = self.gating(representation)
        out = gate * out1 + (1 - gate) * out2

        out = self.conv2(out)

        return out


if __name__ == '__main__':
    batch_size = 1
    style_representation_length = 16
    content_feature_channels = 128
    content_feature_heights = 64
    content_feature_width = 64
    groups_for_SConv = 128  # SConv performs group convolution, you can set group number manually (e.g., 64, 32...).

    style_representation = torch.ones(batch_size, style_representation_length).cuda()
    content_feature = torch.ones(
        (batch_size, content_feature_channels, content_feature_heights, content_feature_width)).cuda()

    SAB = SAB(in_channels=content_feature_channels, out_channels=content_feature_channels, groups=groups_for_SConv,
              representation_dim=style_representation_length).cuda()
    print(SAB)
    output_feature = SAB(content_feature, style_representation)
    print(f"哔哩哔哩: CV缝合救星！")
    print(f"\nInput shape: {content_feature.size()}")
    print(f"Output shape: {output_feature.size()}")
