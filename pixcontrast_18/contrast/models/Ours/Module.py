import torch,math
import torch.nn as nn
import torch.nn.functional as F
import sys,time
sys.path.insert(0,'/raid/wjc/code/RealtimeSegmentation/')


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