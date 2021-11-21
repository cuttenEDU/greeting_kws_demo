import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class SubSpectralNorm(nn.Module):
    def __init__(self, C, S, eps=1e-5):
        super(SubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.bn = nn.BatchNorm2d(C*S)

    def forward(self, x):
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.view(N, C * self.S, F // self.S, T)

        x = self.bn(x)

        return x.view(N, C, F, T)


class BroadcastedBlock(nn.Module):
    def __init__(
            self,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(BroadcastedBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=dilation,
                                      stride=stride, bias=False)
        self.ssn1 = SubSpectralNorm(planes, 5)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.channel_drop = nn.Dropout2d(p=0.5)
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x.clone()

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        out = self.ssn1(out)
        ##########################

        auxilary = out.clone()
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        ############################

        out = out + identity + auxilary
        out = self.relu(out)

        return out


class TransitionBlock(nn.Module):

    def __init__(
            self,
            inplanes: int,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ) -> None:
        super(TransitionBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride,
                                      dilation=dilation, bias=False)
        self.ssn = SubSpectralNorm(planes, 5) #planes = 8
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.channel_drop = nn.Dropout2d(p=0.5)
        self.swish = nn.SiLU()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # f2
        #############################
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.freq_dw_conv(out)
        out = self.ssn(out) #
        #############################
        auxilary = out.clone()
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        #############################
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        out = self.channel_drop(out)
        #############################

        out = auxilary + out
        out = self.relu(out)

        return out


class BCResNet(torch.nn.Module):
    def __init__(self, num_labels = 12):
        super(BCResNet, self).__init__()
        self.num_labels = num_labels
        if num_labels == 2:
            self.num_labels = 1
        self.conv1 = nn.Conv2d(1, 16, 5, stride=(2, 1), padding=(2, 2))
        self.block1_1 = TransitionBlock(16, 8)
        self.block1_2 = BroadcastedBlock(8)

        self.block2_1 = TransitionBlock(8, 12, stride=(2, 1), dilation=(1, 2), temp_pad=(0, 2))
        self.block2_2 = BroadcastedBlock(12, dilation=(1, 2), temp_pad=(0, 2))

        self.block3_1 = TransitionBlock(12, 16, stride=(2, 1), dilation=(1, 4), temp_pad=(0, 4))
        self.block3_2 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_3 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))
        self.block3_4 = BroadcastedBlock(16, dilation=(1, 4), temp_pad=(0, 4))

        self.block4_1 = TransitionBlock(16, 20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_2 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_3 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))
        self.block4_4 = BroadcastedBlock(20, dilation=(1, 8), temp_pad=(0, 8))

        self.conv2 = nn.Conv2d(20, 20, 5, groups=20, padding=(0, 2))
        self.conv3 = nn.Conv2d(20, 32, 1, bias=False)
        self.conv4 = nn.Conv2d(32, self.num_labels, 1, bias=False)

    def forward(self, x):

        #print('INPUT SHAPE:', x.shape)
        out = self.conv1(x)

        #print('BLOCK1 INPUT SHAPE:', out.shape)
        out = self.block1_1(out)
        out = self.block1_2(out)

        #print('BLOCK2 INPUT SHAPE:', out.shape)
        out = self.block2_1(out)
        out = self.block2_2(out)

        #print('BLOCK3 INPUT SHAPE:', out.shape)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)

        #print('BLOCK4 INPUT SHAPE:', out.shape)
        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = self.block4_4(out)

        #print('Conv2 INPUT SHAPE:', out.shape)
        out = self.conv2(out)

        #print('Conv3 INPUT SHAPE:', out.shape)
        out = self.conv3(out)
        out = out.mean(-1, keepdim=True)

        #print('Conv4 INPUT SHAPE:', out.shape)
        out = self.conv4(out)
        out = out.squeeze(-1)
        out = out.mean(-1)

        if self.num_labels == 1:
            out = torch.sigmoid(out)
        #print('OUTPUT SHAPE:', out.shape)
        return out


class MHAttKWS(nn.Module):
    def __init__(
            self,
            num_classes: int = None,
            in_channel: int = 1,
            hidden_dim: int = 128,
            n_head: int = 4,
            dropout: float = 0.1
    ):
        super(MHAttKWS, self).__init__()

        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        if self.num_classes == 2:
            output_dim = 2
        else:
            output_dim = self.num_classes

        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(in_channel, 10, (5, 1), stride=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 1, (5, 1), stride=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )

        self.rnn = nn.LSTM(1, self.hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.q_emb = nn.Linear(self.hidden_dim << 1, (self.hidden_dim << 1) * self.n_head)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.cnn_extractor(x)
        x = x.reshape(x.size(0), -1, x.size(1))
        x, _ = self.rnn(x)

        middle = x.size(1) // 2
        mid_feature = x[:, middle, :]

        multiheads = []
        queries = self.q_emb(mid_feature).view(self.n_head, batch_size, -1, self.hidden_dim << 1)

        for query in queries:
            att_weights = torch.bmm(query, x.transpose(1, 2))
            att_weights = F.softmax(att_weights, dim=-1)
            multiheads.append(torch.bmm(att_weights, x).view(batch_size, -1))

        x = torch.cat(multiheads, dim=-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    x = torch.ones(5, 1, 40, 128)
    bcresnet = BCResNet()
    from torchsummary import summary
    #summary(bcresnet,(1,40,128),device="cpu",batch_size=5)
    res = bcresnet(x)
    # print(res[0])
    # print('num parameters:', sum(p.numel() for p in bcresnet.parameters() if p.requires_grad))