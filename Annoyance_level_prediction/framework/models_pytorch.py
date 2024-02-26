import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def move_data_to_gpu(x, cuda, half=False):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")
    if cuda:
        x = x.cuda()
        if half:
            x = x.half()
    return x



def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.
    """

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



class PANN(nn.Module):
    def __init__(self, event_num):

        super(PANN, self).__init__()

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

        return linear_each_events, linear_rate



