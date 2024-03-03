import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import framework.config as config


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_v_rms(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock_v_rms, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 1), stride=(1, 1),
                               padding=(1, 0), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 1), stride=(1, 1),
                               padding=(1, 0), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')

        return x



class Proposed_CNN(nn.Module):
    def __init__(self, event_num):

        super(Proposed_CNN, self).__init__()

        self.event_num = event_num

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        pann_dim = 2048

        self.fc1 = nn.Linear(pann_dim, pann_dim, bias=True)

        self.fc1_rate = nn.Linear(pann_dim, 1, bias=True)
        self.fc1_event = nn.Linear(pann_dim, event_num, bias=True)


    def forward(self, input):
        # print(input.size())
        # torch.Size([64, 480, 64])

        (_, seq_len, mel_bins) = input.shape

        # x = input.view(-1, 1, seq_len, mel_bins)
        # '''(samples_num, feature_maps, time_steps, freq_num)'''
        # # pann using mel, already normal
        # # another method:
        x = input[:, None, :, :]

        x_clip = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x_clip = F.dropout(x_clip, p=0.2, training=self.training)
        x_clip = self.conv_block2(x_clip, pool_size=(2, 2), pool_type='avg')
        x_clip = F.dropout(x_clip, p=0.2, training=self.training)
        x_clip = self.conv_block3(x_clip, pool_size=(2, 2), pool_type='avg')
        x_clip = F.dropout(x_clip, p=0.2, training=self.training)
        x_clip = self.conv_block4(x_clip, pool_size=(2, 2), pool_type='avg')
        x_clip = F.dropout(x_clip, p=0.2, training=self.training)
        x_clip = self.conv_block5(x_clip, pool_size=(2, 2), pool_type='avg')
        x_clip = F.dropout(x_clip, p=0.2, training=self.training)
        x_clip = self.conv_block6(x_clip, pool_size=(1, 1), pool_type='avg')
        x_clip = F.dropout(x_clip, p=0.2, training=self.training)

        x_clip = torch.mean(x_clip, dim=3)
        (x1_clip, _) = torch.max(x_clip, dim=2)
        x2_clip = torch.mean(x_clip, dim=2)
        x_clip = x1_clip + x2_clip
        # print('x_clip: ', x_clip.size())  # 10s clip: torch.Size([128, 2048])

        x_clip = F.dropout(x_clip, p=0.5, training=self.training)

        x_clip = F.relu_(self.fc1(x_clip))

        linear_rate = self.fc1_rate(x_clip)
        # print(linear_rate.size())  # torch.Size([64, 1])

        linear_each_events = self.fc1_event(x_clip)

        return _, linear_rate



class ConvBlock_rms(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock_rms, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 1), stride=(1, 1),
                               padding=(1, 0), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 1), pool_type='avg'):

        x = input
        # print(x.size()) torch.Size([64, 1, 482, 1])
        x = F.relu_(self.bn1(self.conv1(x)))
        # print(x.size())  torch.Size([64, 64, 482, 1])

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')

        return x


class DNN(nn.Module):
    def __init__(self, event_class):

        super(DNN, self).__init__()

        self.fc_64 = nn.Linear(64, 64, bias=True)
        self.fc_128 = nn.Linear(64, 128, bias=True)
        self.fc_256 = nn.Linear(128, 256, bias=True)
        self.fc_512 = nn.Linear(256, 512, bias=True)

        self.fc_event = nn.Linear(512, event_class, bias=True)
        self.fc_rate = nn.Linear(512, 1, bias=True)


    def forward(self, input):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (16, 481, 64)

        (_, seq_len, mel_bins) = input.shape

        x = input

        # print(x.size())  # torch.Size([64, 480, 64])
        x = F.relu_(self.fc_64(x))
        x = F.max_pool2d(x, kernel_size=(4, 1))
        # print(x.size())  # torch.Size([64, 160, 64])

        x = F.relu_(self.fc_128(x))
        x = F.max_pool2d(x, kernel_size=(4, 1))

        x = F.relu_(self.fc_256(x))
        x = F.max_pool2d(x, kernel_size=(4, 1))

        x = F.relu_(self.fc_512(x))
        x = F.max_pool2d(x, kernel_size=(4, 1))
        # print(x.size())
        # torch.Size([64, 1, 512])

        x_all = torch.flatten(x, start_dim=1)

        x_rate_linear = self.fc_rate(x_all)
        x_event_linear = self.fc_event(x_all)

        return x_event_linear, x_rate_linear


class Simple_CNN(nn.Module):
    def __init__(self, event_class):

        super(Simple_CNN, self).__init__()

        out_channels = 64
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 128
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 256
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 512
        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=out_channels,
                               kernel_size=(3, 1), stride=(1, 1),
                               padding=(0, 0), bias=False)

        self.fc_event = nn.Linear(512, event_class, bias=True)
        self.fc_rate = nn.Linear(512, 1, bias=True)


    def forward(self, input):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (16, 481, 64)

        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        # print(x.size())  # torch.Size([64, 1, 480, 64])
        x = F.relu_(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([64, 32, 95, 12])

        x = F.relu_(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([64, 64, 18, 1])

        x = F.relu_(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([64, 256, 6, 1])

        x = F.relu_(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=(4, 1))
        # print(x.size())  # torch.Size([64, 512, 1, 1])

        x = torch.flatten(x, start_dim=1)

        x = x.view(x.size()[0], -1)
        # print(x.size())  # torch.Size([64, 1152])

        x_rate_linear = self.fc_rate(x)

        x_event_linear = self.fc_event(x)

        return x_event_linear, x_rate_linear


#################################### mha ########################################################
import numpy as np
# transformer
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_heads = 8  # number of heads in Multi-Head Attention

class ScaledDotProductAttention_nomask(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention_nomask, self).__init__()

    def forward(self, Q, K, V, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention_nomask(nn.Module):
    def __init__(self, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads,
                 output_dim=d_model):
        super(MultiHeadAttention_nomask, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * d_v, output_dim)

    def forward(self, Q, K, V, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        context, attn = ScaledDotProductAttention_nomask()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        x = self.layernorm(output + residual)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self, output_dim=d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention_nomask(output_dim=output_dim)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, input_dim, n_layers, output_dim=d_model):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(output_dim) for _ in range(n_layers)])
        self.mel_projection = nn.Linear(input_dim, d_model)

    def forward(self, enc_inputs):
        # print(enc_inputs.size())  # torch.Size([64, 54, 8, 8])
        size = enc_inputs.size()
        enc_inputs = enc_inputs.reshape(size[0], size[1], -1)
        enc_outputs = self.mel_projection(enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k=d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
#################################################################################################


class CNN_Transformer(nn.Module):
    def __init__(self, event_class):

        super(CNN_Transformer, self).__init__()

        out_channels = 64
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 128
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        out_channels = 256
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), bias=False)

        d_model = 512
        self.mha = Encoder(input_dim=256, n_layers=1, output_dim=d_model)

        self.fc_event = nn.Linear(512*6, event_class, bias=True)
        self.fc_rate = nn.Linear(512*6, 1, bias=True)


    def forward(self, input):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (16, 481, 64)

        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''

        # print(x.size())  # torch.Size([64, 1, 480, 64])
        x = F.relu_(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([64, 32, 95, 12])

        x = F.relu_(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([64, 64, 18, 1])

        x = F.relu_(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=(4, 3))
        # print(x.size())  # torch.Size([64, 256, 6, 1])

        x = x.transpose(1, 2)  # torch.Size([64, 6, 256, 1])
        x, x_self_attns = self.mha(x)  # already have reshape
        # print(x_event.size())  # torch.Size([64, 6, 512])

        x = torch.flatten(x, start_dim=1)

        x_rate_linear = self.fc_rate(x)

        x_event_linear = self.fc_event(x)

        return x_event_linear, x_rate_linear





