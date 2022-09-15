# coding: utf-8
import math

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se

        if self.se:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv3 = conv1x1(planes, planes // 16)
            self.conv4 = conv1x1(planes // 16, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.se:
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()

            out = out * w

        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, se=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.bn = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        return x


class VideoCNN(nn.Module):
    def __init__(self, se=False):
        super(VideoCNN, self).__init__()

        # frontend3D
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                64,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        # resnet
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)
        self.dropout = nn.Dropout(p=0.5)

        # backend_gru
        # initialize
        self._initialize_weights()

    def visual_frontend_forward(self, x):
        x = x.transpose(1, 2)
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet18(x)
        return x

    def forward(self, x):
        b, t = x.size()[:2]

        x = self.visual_frontend_forward(x)

        # x = self.dropout(x)
        feat = x.view(b, -1, 512)

        x = x.view(b, -1, 512)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class LRWVideoModel(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(LRWVideoModel, self).__init__()

        self.args = args

        self.video_cnn = VideoCNN(se=self.args["se"])
        if self.args["border"]:
            in_dim = 512 + 1
        else:
            in_dim = 512
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)

        self.v_cls = nn.Linear(1024 * 2, self.args["n_classes"])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v, border=None):
        self.gru.flatten_parameters()

        if self.training:
            f_v = self.video_cnn(v)
            f_v = self.dropout(f_v)
            f_v = f_v.float()
        else:
            f_v = self.video_cnn(v)
            f_v = self.dropout(f_v)

        if self.args["border"]:
            border = border[:, :, None]
            h, _ = self.gru(torch.cat([f_v, border], -1))
        else:
            h, _ = self.gru(f_v)

        y_v = self.v_cls(self.dropout(h)).mean(1)

        return y_v


class ScalableLRWNet(nn.Module):
    """
    Idea:
    >> use 3D CNN, freeze the weights
    >> replace feature encoder from resnet to more lightweight model
    >> replace GRU with LSTM or Simplify GRU hidden layer
    """

    def __init__(
        self,
        border: bool = False,
        n_classes: int = 500,
        se: bool = False,
        outplane_3d: int = 64,
        recurrent_hidden_size: int = 1024,
        recurrent_biderctional: bool = True,
        stack_recurrent: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.video_cnn = ScalableVideoCNN(outplane_3d=outplane_3d, se=se)  # pass as keyword args
        self.border = border
        if border:
            in_dim = (outplane_3d * 8) + 1
        else:
            in_dim = outplane_3d * 8

        if recurrent_biderctional:
            self.gru = nn.GRU(
                in_dim,
                recurrent_hidden_size,
                num_layers=stack_recurrent,
                batch_first=True,
                bidirectional=True,
                dropout=0.2,
            )
            self.v_cls = nn.Linear(recurrent_hidden_size * 2, n_classes)
        else:
            self.gru = nn.GRU(
                in_dim,
                recurrent_hidden_size,
                num_layers=stack_recurrent,
                batch_first=True,
                bidirectional=False,
                dropout=0.2,
            )
            self.v_cls = nn.Linear(recurrent_hidden_size, n_classes)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v, border=None):
        self.gru.flatten_parameters()

        if self.training:
            f_v = self.video_cnn(v)
            f_v = self.dropout(f_v)
            f_v = f_v.float()
        else:
            f_v = self.video_cnn(v)
            f_v = self.dropout(f_v)

        if self.border:
            border = border[:, :, None]
            h, _ = self.gru(torch.cat([f_v, border], -1))
        else:
            h, _ = self.gru(f_v)

        y_v = self.v_cls(self.dropout(h)).mean(1)

        return y_v


class ScalableVideoCNN(nn.Module):
    def __init__(self, outplane_3d: int = 64, se: bool = False):
        super().__init__()

        self.outplane_3d = outplane_3d

        # frontend3D
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                outplane_3d,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(outplane_3d),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        # resnet
        self.feature_net = ScalableResNet(
            BasicBlock, [2, 2, 2, 2], in_planes=self.outplane_3d, se=se
        )
        self.dropout = nn.Dropout(p=0.5)

        # backend_gru
        # initialize
        self._initialize_weights()

    def visual_frontend_forward(self, x):
        x = x.transpose(1, 2)
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, self.outplane_3d, x.size(3), x.size(4))
        x = self.feature_net(x)
        return x

    def forward(self, x):
        b, t = x.size()[:2]

        x = self.visual_frontend_forward(x)

        # x = self.dropout(x)
        feat = x.view(
            b, -1, self.outplane_3d * 8
        )  # FIXME: depended on current implementation of ScalableResnet

        x = x.view(b, -1, self.outplane_3d * 8)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ScalableResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        in_planes=64,
        se=False,
    ):
        self.inplanes = in_planes
        super().__init__()
        self.se = se
        self.layer1 = self._make_layer(block, in_planes, layers[0])
        self.layer2 = self._make_layer(block, in_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.bn = nn.BatchNorm1d(in_planes * 8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        return x
