import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_d311(nn.Module):
    def __init__(self):            
        super(Conv_d311, self).__init__()
        kernel = [[-1, 0, 0],
                  [ 0, 1, 0],
                  [ 0, 0, 0],
                  ]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1) 
        
class Conv_d312(nn.Module):
    def __init__(self):            
        super(Conv_d312, self).__init__()
        kernel = [[ 0, -1, 0],
                  [ 0,  1, 0],
                  [ 0,  0, 0],
                  ]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)    

class Conv_d313(nn.Module):
    def __init__(self):            
        super(Conv_d313, self).__init__()
        kernel = [[ 0, 0, -1],
                  [ 0,  1, 0],
                  [ 0,  0, 0],
                  ]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)    

class Conv_d314(nn.Module):
    def __init__(self):            
        super(Conv_d314, self).__init__()
        kernel = [[ 0, 0, 0],
                  [ 0,  1, -1],
                  [ 0,  0, 0],
                  ]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)    

class Conv_d315(nn.Module):
    def __init__(self):            
        super(Conv_d315, self).__init__()
        kernel = [[ 0, 0, 0],
                  [ 0,  1, 0],
                  [ 0,  0, -1],
                  ]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)         
        
class Conv_d316(nn.Module):
    def __init__(self):            
        super(Conv_d316, self).__init__()
        kernel = [[ 0, 0, 0],
                  [ 0,  1, 0],
                  [ 0,  -1, 0],
                  ]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)   

class Conv_d317(nn.Module):
    def __init__(self):            
        super(Conv_d317, self).__init__()
        kernel = [[ 0, 0, 0],
                  [ 0,  1, 0],
                  [ -1,  0, 0],
                  ]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)         
        
class Conv_d318(nn.Module):
    def __init__(self):            
        super(Conv_d318, self).__init__()
        kernel = [[ 0, 0, 0],
                  [ -1,  1, 0],
                  [ 0,  0, 0],
                  ]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)  

class Conv_d511(nn.Module):
    def __init__(self):
        super(Conv_d511, self).__init__()
        kernel = [[-1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)

class Conv_d512(nn.Module):
    def __init__(self):
        super(Conv_d512, self).__init__()
        kernel = [[0, 0, -1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)

class Conv_d513(nn.Module):
    def __init__(self):
        super(Conv_d513, self).__init__()
        kernel = [[0, 0, 0, 0, -1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)

class Conv_d514(nn.Module):
    def __init__(self):
        super(Conv_d514, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, -1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)

class Conv_d515(nn.Module):
    def __init__(self):
        super(Conv_d515, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -1]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)

class Conv_d516(nn.Module):
    def __init__(self):
        super(Conv_d516, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, -1, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)

class Conv_d517(nn.Module):
    def __init__(self):
        super(Conv_d517, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [-1, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)

class Conv_d518(nn.Module):
    def __init__(self):
        super(Conv_d518, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [-1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)

class Conv_d711(nn.Module):
    def __init__(self):
        super(Conv_d711, self).__init__()
        kernel = [[-1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=3)

class Conv_d712(nn.Module):
    def __init__(self):
        super(Conv_d712, self).__init__()
        kernel = [[0, 0, 0, -1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=3)

class Conv_d713(nn.Module):
    def __init__(self):
        super(Conv_d713, self).__init__()
        kernel = [[0, 0, 0, 0, 0, 0, -1],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=3)

class Conv_d714(nn.Module):
    def __init__(self):
        super(Conv_d714, self).__init__()
        kernel = [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, -1],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=3)

class Conv_d715(nn.Module):
    def __init__(self):
        super(Conv_d715, self).__init__()
        kernel = [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -1]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=3)

class Conv_d716(nn.Module):
    def __init__(self):
        super(Conv_d716, self).__init__()
        kernel = [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, -1, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=3)

class Conv_d717(nn.Module):
    def __init__(self):
        super(Conv_d717, self).__init__()
        kernel = [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [-1, 0, 0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=3)

class Conv_d718(nn.Module):
    def __init__(self):
        super(Conv_d718, self).__init__()
        kernel = [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [-1, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=3)

class dw_conv(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True):
        super(dw_conv, self).__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        self.dw_conv_k3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            activation())
    def forward(self, x):
        x = self.dw_conv_k3(x)
        return x

class M2AM(nn.Module):
    def __init__(self):
        super(M2AM, self).__init__()

        self.d711 = Conv_d711()
        self.d712 = Conv_d712()
        self.d713 = Conv_d713()
        self.d714 = Conv_d714()
        self.d715 = Conv_d715()
        self.d716 = Conv_d716()
        self.d717 = Conv_d717()
        self.d718 = Conv_d718()

        self.d511 = Conv_d511()
        self.d512 = Conv_d512()
        self.d513 = Conv_d513()
        self.d514 = Conv_d514()
        self.d515 = Conv_d515()
        self.d516 = Conv_d516()
        self.d517 = Conv_d517()
        self.d518 = Conv_d518()

        self.d311 = Conv_d311()
        self.d312 = Conv_d312()
        self.d313 = Conv_d313()
        self.d314 = Conv_d314()
        self.d315 = Conv_d315()
        self.d316 = Conv_d316()
        self.d317 = Conv_d317()
        self.d318 = Conv_d318()

    def forward(self, x):
        d711 = self.d711(x)
        d712 = self.d712(x)
        d713 = self.d713(x)
        d714 = self.d714(x)
        d715 = self.d715(x)
        d716 = self.d716(x)
        d717 = self.d717(x)
        d718 = self.d718(x)
        LoC7 = d711.mul(d715) + d712.mul(d716) + d713.mul(d717) + d714.mul(d718)

        d511 = self.d511(x)
        d512 = self.d512(x)
        d513 = self.d513(x)
        d514 = self.d514(x)
        d515 = self.d515(x)
        d516 = self.d516(x)
        d517 = self.d517(x)
        d518 = self.d518(x)
        LoC5 = d511.mul(d515) + d512.mul(d516) + d513.mul(d517) + d514.mul(d518)

        d311 = self.d311(x)
        d312 = self.d312(x)
        d313 = self.d313(x)
        d314 = self.d314(x)
        d315 = self.d315(x)
        d316 = self.d316(x)
        d317 = self.d317(x)
        d318 = self.d318(x)
        LoC3 = d311.mul(d315) + d312.mul(d316) + d313.mul(d317) + d314.mul(d318)

        md = torch.max(LoC3, LoC5)
        md = torch.max(md, LoC7)
        md = torch.sigmoid(md)
        return md

class ChannelAttention(nn.Module):   
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False) 
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False) 
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):   
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False) 
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)      
        
def conv_relu_bn(in_channel, out_channel, dirate):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=dirate,
                  dilation=dirate),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class dconv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(dconv_block, self).__init__()
        self.conv1 = conv_relu_bn(in_ch, out_ch, 1)
        self.dconv1 = conv_relu_bn(out_ch, out_ch // 2, 2)         
        self.dconv2 = conv_relu_bn(out_ch // 2, out_ch // 2, 4)   
        self.dconv3 = conv_relu_bn(out_ch, out_ch, 2)              
        self.conv2 = conv_relu_bn(out_ch * 2, out_ch, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        dx1 = self.dconv1(x1)
        dx2 = self.dconv2(dx1)
        dx3 = self.dconv3(torch.cat((dx1, dx2), dim=1))
        out = self.conv2(torch.cat((x1, dx3), dim=1))
        return out    
    
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention() 
    def forward(self, x):
        residual = x

        if self.shortcut is not None:
            residual = self.shortcut(x)      
        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.ca(out) * out

        out = self.sa(out) * out

        out += residual

        out = self.relu(out)

        return out

class FeatureFusionModule(nn.Module): 
    def __init__(self, planes_high, planes_low, planes_out):
        super(FeatureFusionModule, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(planes_low, planes_low // 4, kernel_size=1),
            nn.BatchNorm2d(planes_low // 4),
            nn.ReLU(True),

            nn.Conv2d(planes_low // 4, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.plus_conv = nn.Sequential(
            nn.Conv2d(planes_low*3, planes_low, kernel_size=1), 
            nn.BatchNorm2d(planes_low),
            nn.ReLU(True)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes_low, planes_low // 4, kernel_size=1),
            nn.BatchNorm2d(planes_low // 4),
            nn.ReLU(True),

            nn.Conv2d(planes_low // 4, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.end_conv = nn.Sequential(     
            nn.Conv2d(planes_high, planes_low, 3, 1, 1),  
            nn.BatchNorm2d(planes_low),
            nn.ReLU(True),
        )
        self.dconv = nn.ConvTranspose2d(
            in_channels=planes_high, out_channels=planes_low, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x_high, x_low):      
        x_high = self.dconv(x_high, output_size=x_low.size())       
        feat = x_low + x_high

        pa = self.pa(x_low) * x_high
        ca = self.ca(x_high) * x_low
        feat = self.plus_conv(torch.cat([pa, ca, feat], 1))

        return feat

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())

class Net(nn.Module):
    def __init__(self, input_channels =1, block=ResNet):
        super().__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        self.pool = nn.MaxPool2d(2, 2) 
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.first = M2AM()
        self.conv1 = conv_batch(1, 16)
        self.conv2 = conv_batch(16,32)
        self.conv3 = conv_batch(32,16)
        self.conv4 = conv_batch(32,64)
        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[2], param_channels[2], block, param_blocks[1])
        self.middle_layer = dconv_block(param_channels[2], param_channels[3])
        self.FeaFus2 = FeatureFusionModule(128, 64, 64)
        self.FeaFus1 = FeatureFusionModule(64, 32, 32)
        self.FeaFus0 = FeatureFusionModule(32, 16, 16)
        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.final = nn.Conv2d(3, 1, 3, 1, 1)


    def _make_layer(self, in_channels, out_channels, block, block_num=1): 
        layer = []        
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x, warm_flag):
        md = self.first(x)
        x= self.conv1(x)                       
        x_e0 = self.encoder_0(x)
        out = x_e0.mul(md)
        out = out + x_e0
        x_e1 = self.encoder_1(self.pool(out))        
        out_2 = self.conv2(out)                     
        x_e2 = self.encoder_2(torch.cat((self.pool(x_e1),self.pool(self.pool(out_2))), dim=1))
        x_m = self.middle_layer(self.pool(x_e2))
        x_d2 = self.FeaFus2(x_m, x_e2)
        x_d1 = self.FeaFus1(x_d2, x_e1)
        x_d0 = self.FeaFus0(x_d1, x_e0)
 

        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            output = self.final(torch.cat([mask0, self.up(mask1), self.up4(mask2)], dim=1))
            return [mask0, mask1, mask2], output  
        else:
            output = self.output_0(x_d0)
            return [], output  