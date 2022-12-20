import torch,math
import torch.nn as nn
import torch.nn.functional as F
import sys,time
sys.path.insert(0,'/raid/wjc/code/RealtimeSegmentation/')

from net.utils.helpers import maybe_download
from net.utils.layer_factory import conv1x1, conv3x3, convbnrelu, CRPBlock


data_info = {21: "VOC"}

models_urls = {
    "mbv2_voc": "https://cloudstor.aarnet.edu.au/plus/s/nQ6wDnTEFhyidot/download",
    "mbv2_imagenet": "https://cloudstor.aarnet.edu.au/plus/s/uRgFbkaRjD3qOg5/download",
}

class TimeProcesser(nn.Module):
    def __init__(self, inplanes, planes, size, batch_size, tag, group=1):
        super(TimeProcesser, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.size = size
        self.tag = tag
        self.batch_size = batch_size
        if not inplanes==planes:
            self.refine = conv1x1(planes, inplanes)
        else:
            pass
    def forward(self, x):
        x = self.processer(x)
        return x
    
class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block from https://arxiv.org/abs/1801.04381"""

    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super(InvertedResidualBlock, self).__init__()
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1)
        self.output = nn.Sequential(
            convbnrelu(in_planes, intermed_planes, 1),
            convbnrelu(
                intermed_planes,
                intermed_planes,
                3,
                stride=stride,
                groups=intermed_planes,
            ),
            convbnrelu(intermed_planes, out_planes, 1, act=False),
        )

    def forward(self, x):
        residual = x
        out = self.output(x)
        if self.residual:
            return out + residual
        else:
            return out

class MemoryCore(nn.Module):
    def __init__(self):
        super(MemoryCore, self).__init__()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,t,c,h,w
        B, T, D_e, H, W = m_in.size() #mem key: 2,1,256,64,80
        _, _, D_o, _, _ = m_out.size() #mem value

        mi = m_in.transpose(1,2).contiguous().view(B, D_e, T*H*W) 
        mi = torch.transpose(mi, 1, 2).contiguous()  # b, THW, emb
 
        qi = q_in.view(B, D_e, H*W) #query key:2,256,5120 # b, emb, HW
 
        p = torch.bmm(mi, qi) # b, THW, HW: 前几帧和当前帧做点积
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1) # b, THW, HW
        p = self.dropout(p) #normalized weights
         
        mo = m_out.view(B, D_o, T*H*W) 
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW, previous value*weight
        mem = mem.view(B, D_o, H, W) #2,256,64,80

        mem_out = torch.cat([mem, q_out], dim=1) #2,512,64,80

        return mem_out, p

class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
 
    def forward(self, x):  
        return [self.Key(x), self.Value(x)]
    
class Memory(nn.Module):
    def __init__(self,c):
        super(Memory, self).__init__()
        self.mem_core = MemoryCore()
        self.kv = KeyValue(c, c//2, c//2)
        
    def forward(self, mem, query):
        _, T, _, _, _ = mem.size()
#         print('Memory:{}'.format(T))
        keys = []
        values = []
        for t in range(T):
            k,v = self.kv(mem[:,t]) #mem里面全部帧会经过conv生成自己的key value
            keys.append(k.unsqueeze(1))
            values.append(v.unsqueeze(1))
        MemoryKeys = torch.cat(keys, dim=1)
        MemoryValues = torch.cat(values, dim=1)
        CurrentKey, CurrentValue = self.kv(query) #当前帧也是经过conv生成自己的key value
        mem_out, p = self.mem_core(MemoryKeys, MemoryValues, CurrentKey, CurrentValue)
        return mem_out, p
    
class MobileEncoder(nn.Module):
    """Encoder mobilev2"""
    
    mobilenet_config = [
        [1, 16, 1, 1],  # expansion rate, output channels, number of repeats, stride
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    in_planes = 32  # number of input channels
    num_layers = len(mobilenet_config)
    def __init__(self):
        super(MobileEncoder, self).__init__()
        self.layer1 = convbnrelu(3, self.in_planes, kernel_size=3, stride=2)
        c_layer = 2
        for t, c, n, s in self.mobilenet_config:
            layers = []
            for idx in range(n):
                layers.append(
                    InvertedResidualBlock(
                        self.in_planes,
                        c,
                        expansion_factor=t,
                        stride=s if idx == 0 else 1,
                    )
                )
                self.in_planes = c
            setattr(self, "layer{}".format(c_layer), nn.Sequential(*layers))
            c_layer += 1
        
        ## Light-Weight RefineNet ##
        self.conv8 = conv1x1(320, 256, bias=False)
        self.conv7 = conv1x1(160, 256, bias=False)
        
        self.relu = nn.ReLU6(inplace=True)
        
    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)  # x / 2
        l3 = self.layer3(l2)  # 24, x / 4
        l4 = self.layer4(l3)  # 32, x / 8
        l5 = self.layer5(l4)  # 64, x / 16
        l6 = self.layer6(l5)  # 96, x / 16
        l7 = self.layer7(l6)  # 160, x / 32
        l8 = self.layer8(l7)  # 320, x / 32
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)# 256, x/32
        return [l3,l4,l5,l6,l7]
    
class RefineDecoder(nn.Module):
    def __init__(self, num_classes):
        super(RefineDecoder, self).__init__()
        self.conv6 = conv1x1(96, 256, bias=False)
        self.conv5 = conv1x1(64, 256, bias=False)
        self.conv4 = conv1x1(32, 256, bias=False)
        self.conv3 = conv1x1(24, 256, bias=False)
        self.crp4 = self._make_crp(256, 256, 4)
        self.crp3 = self._make_crp(256, 256, 4)
        self.crp2 = self._make_crp(256, 256, 4)
        self.crp1 = self._make_crp(256, 256, 4)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)

        self.segm = conv3x3(256, num_classes, bias=True)
        self.relu = nn.ReLU6(inplace=True) 
    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)    
    def forward(self, x):
        l3,l4,l5,l6,l7 = x
        
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = nn.Upsample(size=l6.size()[2:], mode="bilinear", align_corners=True)(l7)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = nn.Upsample(size=l4.size()[2:], mode="bilinear", align_corners=True)(l5)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = nn.Upsample(size=l3.size()[2:], mode="bilinear", align_corners=True)(l4)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        out_segm = self.segm(l3)
        return out_segm
    
