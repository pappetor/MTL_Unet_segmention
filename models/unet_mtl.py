""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv2d_mtl import Conv2dMtl, ConvTranspose2dMtl
#这个类用于MSNet
class CNN1(nn.Module):
    def __init__(self,channel,map_size,pad):
        super(CNN1,self).__init__()
        self.weight = nn.Parameter(torch.ones(channel,channel,map_size,map_size),requires_grad=False).cuda()
        self.bias = nn.Parameter(torch.zeros(channel),requires_grad=False).cuda()
        self.pad = pad
        self.norm = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = F.conv2d(x,self.weight,self.bias,stride=1,padding=self.pad)
        out = self.norm(out)
        out = self.relu(out)
        return out

#普通的两次卷积、normal、relu
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#DoubleConvMtl的工具类
def conv3x3mtl(in_planes, out_planes, stride=1):
    return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
#mtl的两次卷积、normal、relu
class DoubleConvMtl(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = conv3x3mtl(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3mtl(mid_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out

#下采样
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, mtl):
        super().__init__()
        if mtl:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConvMtl(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)

#上采样
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mtl):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        if mtl:
            # self.up = ConvTranspose2dMtl(in_channels , in_channels, 2, 2)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv = DoubleConvMtl(in_channels, out_channels)
        else:
            # self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        """自己定义的上采样"""
        outputs = torch.cat([self.up(x1),x2],1)
        outputs = self.conv(outputs)
        return outputs


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, mtl):
        super(OutConv, self).__init__()
        if mtl:
            self.conv = Conv2dMtl(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNetMtl(nn.Module):
    def __init__(self, n_channels, n_classes, mtl=True):
        super(UNetMtl, self).__init__()
        #************************
        self.x5_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.x5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))

        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.x5_dem_5 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        # ************************
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = False
        self.mtl = mtl
        if mtl:
            self.inc = DoubleConvMtl(n_channels, 64)
        else:
            self.inc = DoubleConv(n_channels, 64)
            
        self.down1 = Down(64, 128, mtl)

        self.down2 = Down(128, 256, mtl)

        self.down3 = Down(256, 512, mtl)
        factor = 2 if bilinear else 1

        self.down4 = Down(512, 512, mtl)
        # self.up1 = Up(64, 512, mtl)
        # self.up2 = Up(64, 256, mtl)
        # self.up3 = Up(64, 128, mtl)
        # self.up4 = Up(64, 64 * factor, mtl)
        self.up1 = Up(128, 64, mtl)
        self.up2 = Up(128, 64, mtl)
        self.up3 = Up(128, 64, mtl)
        self.up4 = Up(128, 64, mtl)
        self.outc = OutConv(64, n_classes, mtl)




    def forward(self, x):
        # print("x的尺寸：" + str(x.shape))
        x1 = self.inc(x)
        # print("x1的尺寸：" + str(x1.shape))

        x2 = self.down1(x1)
        # print("x2的尺寸：" + str(x2.shape))

        x3 = self.down2(x2)
        # print("x3的尺寸：" + str(x3.shape))

        x4 = self.down3(x3)
        # print("x4的尺寸：" + str(x4.shape))

        x5 = self.down4(x4)
        # print("x5的尺寸：" + str(x5.shape))

        #将第五层到第二层全部转化为64通道，这是第一列
        x5_dem_1 = self.x5_dem_1(x5)
        x4_dem_1 = self.x4_dem_1(x4)
        x3_dem_1 = self.x3_dem_1(x3)
        x2_dem_1 = self.x2_dem_1(x2)

        #这是第二列
        x5_4 = self.x5_x4(abs(F.upsample(x5_dem_1, size=x4.size()[2:], mode='bilinear') - x4_dem_1))
        x4_3 = self.x4_x3(abs(F.upsample(x4_dem_1, size=x3.size()[2:], mode='bilinear') - x3_dem_1))
        x3_2 = self.x3_x2(abs(F.upsample(x3_dem_1, size=x2.size()[2:], mode='bilinear') - x2_dem_1))
        x2_1 = self.x2_x1(abs(F.upsample(x2_dem_1, size=x1.size()[2:], mode='bilinear') - x1))

        #这是第三列
        x5_4_3 = self.x5_x4_x3(abs(F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear') - x4_3))
        x4_3_2 = self.x4_x3_x2(abs(F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear') - x3_2))
        x3_2_1 = self.x3_x2_x1(abs(F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear') - x2_1))

        #这是第四列
        x5_4_3_2 = self.x5_x4_x3_x2(abs(F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear') - x4_3_2))
        x4_3_2_1 = self.x4_x3_x2_x1(abs(F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear') - x3_2_1))

        #这是第五列
        x5_dem_4 = self.x5_dem_4(x5_4_3_2)
        x5_4_3_2_1 = self.x5_x4_x3_x2_x1(
            abs(F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear') - x4_3_2_1))

        #汇聚所有的层，level5就是x5_dem_5
        level4 = x5_4
        level3 = self.level3(x4_3 + x5_4_3)
        level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2)
        level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)

        x5_dem_5 = self.x5_dem_5(x5)
        output4 = self.output4(F.upsample(x5_dem_5, size=level4.size()[2:], mode='bilinear') + level4)
        output3 = self.output3(F.upsample(output4, size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3, size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2, size=level1.size()[2:], mode='bilinear') + level1)

        x = self.up1(x5_dem_5, output4)
        x = self.up2(x, output3)
        x = self.up3(x, output2)
        print("x:" + str(x.size()))
        print("output1:" + str(output1.size()))
        x = self.up4(x, output1)
        logits = self.outc(x)
        return logits
        # output4 = self.output4(F.upsample(x5_dem_5, size=level4.size()[2:], mode='bilinear') + level4)
        # output3 = self.output3(F.upsample(output4, size=level3.size()[2:], mode='bilinear') + level3)
        # output2 = self.output2(F.upsample(output3, size=level2.size()[2:], mode='bilinear') + level2)
        # output1 = self.output1(F.upsample(output2, size=level1.size()[2:], mode='bilinear') + level1)

        # output = F.upsample(output1, size=input.size()[2:], mode='bilinear')
        # if self.training:
        #     return output
        # return output

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # # print(x.size())
        # logits = self.outc(x) #segmented ouput
        
        # Classifier Use and the layer changed
        # logits = logits.view(logits.size(0), -1)
        # print(logits.size())
        # return logits
if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    # model = res2net50_v1b_26w_4s(pretrained=False)
    model = UNetMtl(n_channels=3,n_classes=100)

    model = model.cuda(0)
    res = model(images)
    # images = model.inc(images)
    print(res.shape)
    # print(model(images).size())