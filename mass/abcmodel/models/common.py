import torch
from torch import nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()
        self.arguments = {'in_filters': in_filters, 'out_filters': out_filters, 'reps': reps, 
						'strides': strides, 'start_with_relu': start_with_relu, 'grow_first': grow_first}

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep=[]

        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class CoordConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, 
				stride=1, padding=0, dilation=1, bias=True, groups=1):
        """
        Coord Convolution Module
        Coord2d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True)
        Use it just like using a normal pytorch nn.Module
        """

        super().__init__(
			in_channels=in_channels + 2, out_channels=out_channels, 
			kernel_size=kernel_size, stride=stride, padding=padding, 
			dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x_with_coord = self.add_coord(x)
        return super().forward(x_with_coord)

    @staticmethod
    def add_coord(x):
        b, _, h, w = x.size()
        _h = torch.arange(0, h)
        _w = torch.arange(0, w)
        # draw grid
        y_grid, x_grid = torch.meshgrid(_h, _w)
        # normalize -1 to 1
        x_coord = torch.true_divide(x_grid, w / 2) - 1
        y_coord = torch.true_divide(y_grid, h / 2) - 1
        # Did not use .register_buffer because I was not sure if that can be converted with tensorRT
        x_coord = x_coord.expand(b, 1, -1, -1).to(x.device)
        y_coord = y_coord.expand(b, 1, -1, -1).to(x.device)
        x = torch.cat([x, x_coord, y_coord], dim=1)
        return x


class ConvBNReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, repeat, 
				use_coord_conv=True, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        layers = []
        conv_op = CoordConv2d if use_coord_conv else nn.Conv2d
        for i in range(repeat):
            layers.append(
                conv_op(
					in_channels=in_channels if not i else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias
				))
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU(inplace=True))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
