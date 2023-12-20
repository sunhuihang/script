""" Full assembly of the parts to form the complete network """
import torch
import sys
sys.path.append('../model')
try:
    from .unet_parts import *
except:
    from unet_parts import *

class UNet_attention(nn.Module):
    def __init__(self,
                 scale, 
                 n_channels, 
                 n_classes = 2, 
                 bilinear = False):
        super(UNet_attention, self).__init__()
        self.channel_list=[64, 128, 256, 512, 1024]
        self.scale = scale
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, self.channel_list[0]))
        self.down1 = (Down(self.channel_list[0], self.channel_list[1]))
        self.down2 = (Down(self.channel_list[1], self.channel_list[2]))
        self.down3 = (Down(self.channel_list[2], self.channel_list[3]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.channel_list[3], self.channel_list[4] // factor))

        #att + up + conv
        self.up4 = Attention_Up_Conv(self.channel_list[4]// factor, self.channel_list[3] // factor, bilinear)
        self.up3 = Attention_Up_Conv(self.channel_list[3]// factor, self.channel_list[2] // factor, bilinear)
        self.up2 = Attention_Up_Conv(self.channel_list[2]// factor, self.channel_list[1] // factor, bilinear)
        self.up1 = Attention_Up_Conv(self.channel_list[1]// factor, self.channel_list[0], bilinear)

        self.outc = (OutConv(64, n_classes))
        self.line_last2 =  nn.Sequential(nn.Dropout(p=0.5),
                            nn.Linear(scale**2, 6,), 
                            nn.LogSoftmax(dim=1),
                            )
        self.line_last =  nn.Sequential(nn.Dropout(p=0.5),
                            nn.Linear(scale**2, 1,), 
                            nn.ReLU(inplace=True),
                            )



    def forward(self, x):
        # x.shape  [batch, channel, height, width]
        batch = x.shape[0]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(g=x5, x=x4)
        x = self.up3(g=x, x=x3)
        x = self.up2(g=x, x=x2)
        x = self.up1(g=x, x=x1)

        x = self.outc(x)
        cls = self.line_last2(x[:,0].view(batch,-1))
        out_stage = self.line_last(x[:,1].view(batch,-1))
        
        return out_stage,cls

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)

        # self.self_attention1 = torch.utils.checkpoint(self.self_attention1)
        # self.self_attention2 = torch.utils.checkpoint(self.self_attention2)
        # self.self_attention3 = torch.utils.checkpoint(self.self_attention3)
        # self.self_attention4 = torch.utils.checkpoint(self.self_attention4)

        self.up4 = torch.utils.checkpoint(self.up4)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.outc = torch.utils.checkpoint(self.outc)
        self.line_last2 = torch.utils.checkpoint(self.line_last2)
        self.line_last = torch.utils.checkpoint(self.line_last)
if __name__ == '__main__':
    model = UNet_attention(16,11,2)
    prep,_ = model(torch.randn(2,11,16,16))
    print(prep.shape) 
