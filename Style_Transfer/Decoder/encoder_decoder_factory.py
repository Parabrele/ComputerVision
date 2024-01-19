import torch
import torch.nn as nn

from tqdm import tqdm

import time

from models.autoencoder_vgg19.vgg19_1 import vgg_normalised_conv1_1, feature_invertor_conv1_1
from models.autoencoder_vgg19.vgg19_2 import vgg_normalised_conv2_1, feature_invertor_conv2_1
from models.autoencoder_vgg19.vgg19_3 import vgg_normalised_conv3_1, feature_invertor_conv3_1
from models.autoencoder_vgg19.vgg19_4 import vgg_normalised_conv4_1, feature_invertor_conv4_1
from models.autoencoder_vgg19.vgg19_5 import vgg_normalised_conv5_1, feature_invertor_conv5_1

import images as im

class Encoder(nn.Module):

    def __init__(self, depth):
        super(Encoder, self).__init__()

        assert(type(depth).__name__ == 'int' and 1 <= depth <= 5)
        self.depth = depth

        if depth == 1:
            self.model = vgg_normalised_conv1_1.vgg_normalised_conv1_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_1/vgg_normalised_conv1_1.pth"))
        elif depth == 2:
            self.model = vgg_normalised_conv2_1.vgg_normalised_conv2_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_2/vgg_normalised_conv2_1.pth"))
        elif depth == 3:
            self.model = vgg_normalised_conv3_1.vgg_normalised_conv3_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_3/vgg_normalised_conv3_1.pth"))
        elif depth == 4:
            self.model = vgg_normalised_conv4_1.vgg_normalised_conv4_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_4/vgg_normalised_conv4_1.pth"))
        elif depth == 5:
            self.model = vgg_normalised_conv5_1.vgg_normalised_conv5_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_5/vgg_normalised_conv5_1.pth"))


    def forward(self, x):
        out = self.model(x)
        return out


class Decoder(nn.Module):
    def __init__(self, depth):
        super(Decoder, self).__init__()

        assert (type(depth).__name__ == 'int' and 1 <= depth <= 5)
        self.depth = depth

        if depth == 1:
            self.model = feature_invertor_conv1_1.feature_invertor_conv1_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_1/feature_invertor_conv1_1.pth"))
        elif depth == 2:
            self.model = feature_invertor_conv2_1.feature_invertor_conv2_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_2/feature_invertor_conv2_1.pth"))
        elif depth == 3:
            self.model = feature_invertor_conv3_1.feature_invertor_conv3_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_3/feature_invertor_conv3_1.pth"))
        elif depth == 4:
            self.model = feature_invertor_conv4_1.feature_invertor_conv4_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_4/feature_invertor_conv4_1.pth"))
        elif depth == 5:
            self.model = feature_invertor_conv5_1.feature_invertor_conv5_1
            self.model.load_state_dict(torch.load("models/autoencoder_vgg19/vgg19_5/feature_invertor_conv5_1.pth"))

    def forward(self, x):
        out = self.model(x)
        return out


class MultiLevelWCT(nn.Module):
    def __init__(self, alpha=1):
        super(MultiLevelWCT, self).__init__()
        
        self.alpha = alpha

        self.e1 = Encoder(1)
        self.e2 = Encoder(2)
        self.e3 = Encoder(3)
        self.e4 = Encoder(4)
        self.e5 = Encoder(5)
        self.encoders = [self.e5, self.e4, self.e3, self.e2, self.e1]

        self.d1 = Decoder(1)
        self.d2 = Decoder(2)
        self.d3 = Decoder(3)
        self.d4 = Decoder(4)
        self.d5 = Decoder(5)
        self.decoders = [self.d5, self.d4, self.d3, self.d2, self.d1]
    
    def forward(self, content, style, alpha=None, verbose=False):
        c = im.wct(content, style)
        alpha = alpha if alpha else self.alpha
        beta = alpha / 5
        l = tqdm(range(5)) if verbose else range(5)
        for i in l:
            gamma = beta / (1 - (4-i) * beta)
            content_feature = self.encoders[i](c).data.cpu()
            style_feature = self.encoders[i](style).data.cpu()
            blend_feature = im.wct(content_feature, style_feature, gamma)
            c = self.decoders[i](blend_feature)
            c = im.wct(c, style)
        
        return c



class FasterMultiLevelWCT_v1(nn.Module):
    def __init__(self, alpha=1):
        super(FasterMultiLevelWCT_v1, self).__init__()
        
        self.alpha = alpha

        self.encoder = Encoder(5)
        self.decoder = Decoder(5)

        self.e1 = self.encoder.model[:4]
        self.e2 = self.encoder.model[4:11]
        self.e3 = self.encoder.model[11:18]
        self.e4 = self.encoder.model[18:31]
        self.e5 = self.encoder.model[31:44]
        
        self.d1 = self.decoder.model[40:42]
        self.d2 = self.decoder.model[33:40]
        self.d3 = self.decoder.model[26:33]
        self.d4 = self.decoder.model[13:26]
        self.d5 = self.decoder.model[:13]

        self.encoders = [self.e1, self.e2, self.e3, self.e4, self.e5]
        self.decoders = [self.d5, self.d4, self.d3, self.d2, self.d1]
    
    def forward(self, content, style, alpha=None, verbose=False, timit = False):
        t1 = time.perf_counter()
        c = im.wct(content, style)#.cuda()
        s = style
        t2 = time.perf_counter()
        wct_time = t2 - t1
        encoding_time = 0
        decoding_time = 0
        cpu2gpu_time = 0
        gpu2cpu_time = 0

        alpha = alpha if alpha else self.alpha
        beta = alpha / 10
        l = tqdm(range(5)) if verbose else range(5)
        for i in l:
            gamma = beta / (1 - (9-i) * beta)

            t1 = time.perf_counter()
            c = self.encoders[i](c)
            s = self.encoders[i](s)
            t2 = time.perf_counter()
            encoding_time += t2 - t1

            t1 = time.perf_counter()
            #c = c.data#.cpu()
            #s = s.data#.cpu()
            t2 = time.perf_counter()
            gpu2cpu_time += t2 - t1

            t1 = time.perf_counter()
            c = im.wct(c, s, gamma)
            t2 = time.perf_counter()
            wct_time += t2 - t1

            t1 = time.perf_counter()
            #c = c.cuda()
            #s = s.cuda()
            t2 = time.perf_counter()
            cpu2gpu_time += t2 - t1
        for i in l:
            j = i + 5
            gamma = beta / (1 - (9-j) * beta)

            t1 = time.perf_counter()
            c = self.decoders[i](c)
            s = self.decoders[i](s)
            t2 = time.perf_counter()
            decoding_time += t2 - t1

            t1 = time.perf_counter()
            #c = c.data#.cpu()
            #s = s.data#.cpu()
            t2 = time.perf_counter()
            gpu2cpu_time += t2 - t1

            t1 = time.perf_counter()
            c = im.wct(c, s, gamma)
            t2 = time.perf_counter()
            wct_time += t2 - t1

            t1 = time.perf_counter()
            #c = c.cuda()
            #s = s.cuda()
            t2 = time.perf_counter()
            cpu2gpu_time += t2 - t1
        
        if timit:
            print("wct_time: ", wct_time)
            print("encoding_time: ", encoding_time)
            print("decoding_time: ", decoding_time)
            print("cpu2gpu_time: ", cpu2gpu_time)
            print("gpu2cpu_time: ", gpu2cpu_time)

        return c


class FasterMultiLevelWCT_v2(nn.Module):
    def __init__(self, alpha=1):
        super(FasterMultiLevelWCT_v2, self).__init__()
        
        self.alpha = alpha

        self.encoder = Encoder(5)
        self.decoder = Decoder(5)
        
        self.d1 = self.decoder.model[40:42]
        self.d2 = self.decoder.model[33:40]
        self.d3 = self.decoder.model[26:33]
        self.d4 = self.decoder.model[13:26]
        self.d5 = self.decoder.model[:13]

        self.decoders = [self.d5, self.d4, self.d3, self.d2, self.d1]
    
    def forward(self, content, style, alpha=None, verbose=False, timit = False):
        wct_time = 0
        encoding_time = 0
        decoding_time = 0
        cpu2gpu_time = 0
        gpu2cpu_time = 0

        t1 = time.perf_counter()
        c = im.wct(content, style)#.cuda()
        s = style
        t2 = time.perf_counter()
        wct_time += t2 - t1
        
        t1 = time.perf_counter()
        c = self.encoder(c)
        s = self.encoder(s)
        t2 = time.perf_counter()
        encoding_time += t2 - t1

        alpha = alpha if alpha else self.alpha
        beta = alpha / 5

        l = tqdm(range(5)) if verbose else range(5)
        for i in l:
            gamma = beta / (1 - (5-i) * beta)

            t1 = time.perf_counter()
            c = im.wct(c, s, gamma)
            t2 = time.perf_counter()
            wct_time += t2 - t1

            t1 = time.perf_counter()
            c = self.decoders[i](c)
            s = self.decoders[i](s)
            t2 = time.perf_counter()
            decoding_time += t2 - t1
        
        if timit:
            print("wct_time: ", wct_time)
            print("encoding_time: ", encoding_time)
            print("decoding_time: ", decoding_time)
            print("cpu2gpu_time: ", cpu2gpu_time)
            print("gpu2cpu_time: ", gpu2cpu_time)

        return c