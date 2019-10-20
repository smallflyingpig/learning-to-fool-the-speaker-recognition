import torch
import torch.nn.functional as F
import torch.nn as nn


class SampleLayer(nn.Module):
    def __init__(self, sample_factor):
        super(SampleLayer, self).__init__()
        self.sample_factor = sample_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.sample_factor, mode='linear', align_corners=False)

def get_act(type, **kwargs):
    act_dict = {'relu':nn.ReLU, 'leaky_relu':nn.LeakyReLU}
    return act_dict[type](inplace=True, **kwargs)



def make_cnn1d(channel, kernel_size, dilation, sample, bn=False, in_channel=1, ac_type='relu'):
    assert len(channel)==len(kernel_size)==len(dilation)==len(sample), "param for cnn layer should have the same length"
    model = nn.ModuleList()
    in_c = in_channel
    bias = not bn
    for c,k,d,s in zip(channel, kernel_size, dilation, sample):
        p = k//2*d
        layer = []
        layer.append(nn.Conv1d(in_c, c, k, 1, p, dilation=d, bias=bias))
        if bn:
            layer.append(nn.BatchNorm1d(c))
        layer.append(get_act(ac_type))
        layer.append(SampleLayer(s))
        model.append(nn.Sequential(*layer))
        in_c = c
    return model

        
class SpeechTransformer(nn.Module):
    """
    input: Bx3200
    output: Bx3200
    """
    def __init__(self, channel=[32,32,32,32,32], 
        kernel_size=[3,3,3,3,3], dilation=[1,2,5,2,1], 
        sample=[1,1,1,1,1], scale=1.0):
        super(SpeechTransformer, self).__init__()
        self.model = make_cnn1d(channel, kernel_size, dilation, sample, bn=True)
        self.scale = scale
        self.out_layer = nn.Sequential(
            nn.Conv1d(channel[-1], 1, 3, 1, 1, bias=True),
            nn.Tanh()
        )
        self.last_zero_init()

    def last_zero_init(self):
        nn.init.constant_(self.out_layer[0].weight, val=0)
        nn.init.constant_(self.out_layer[0].bias, val=0)

    def last_norm_init(self):
        nn.init.normal_(self.model[-1][0].weight, mean=0, std=0.05)
        nn.init.constant_(self.model[-1][0].bias, val=0)

    def forward(self, x):
        assert len(x.shape)==2
        _x = x.unsqueeze(1)
        for idx, m in enumerate(self.model):
            # print(idx)
            _x = m(_x)
        noise = self.scale * self.out_layer(_x)
        x = x + noise.squeeze(1)
        x = torch.clamp(x, -1,1)
        return x





    
   