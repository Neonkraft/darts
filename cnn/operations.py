import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, affine, use_lora, lora_r: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine, use_lora, lora_r: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine, use_lora, lora_r: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine, use_lora, lora_r: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine, use_lora, lora_r: SepConv(C, C, 3, stride, 1, affine=affine, use_lora=use_lora, lora_r=lora_r),
  'sep_conv_5x5' : lambda C, stride, affine, use_lora, lora_r: SepConv(C, C, 5, stride, 2, affine=affine, use_lora=use_lora, lora_r=lora_r),
  'sep_conv_7x7' : lambda C, stride, affine, use_lora, lora_r: SepConv(C, C, 7, stride, 3, affine=affine, use_lora=use_lora, lora_r=lora_r),
  'dil_conv_3x3' : lambda C, stride, affine, use_lora, lora_r: DilConv(C, C, 3, stride, 2, 2, affine=affine, use_lora=use_lora, lora_r=lora_r),
  'dil_conv_5x5' : lambda C, stride, affine, use_lora, lora_r: DilConv(C, C, 5, stride, 4, 2, affine=affine, use_lora=use_lora, lora_r=lora_r),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, use_lora=False, lora_r=4):
    super(ReLUConvBN, self).__init__()
    self.activation = nn.ReLU(inplace=False)
    self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    self.use_lora = use_lora
    self.lora_r = lora_r

    self.lora_down = nn.Conv2d(C_in, lora_r, kernel_size=1, stride=stride, padding=0, bias=False)
    self.lora_up = nn.Conv2d(lora_r, C_out, kernel_size=1, stride=1, padding=0, bias=False)

  def forward(self, x):
    out = self.bn(self.conv(self.activation(x)))

    if self.use_lora is True:
      lora_out = self.lora_up(self.lora_down(x))
      out = out + lora_out

    return out
class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, use_lora=False, lora_r=4):
    super(DilConv, self).__init__()
    self.activation = nn.ReLU(inplace=False)
    self.depthwise = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False)
    self.pointwise = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    self.use_lora = use_lora
    self.lora_r = lora_r

    self.lora_down = nn.Conv2d(C_in, lora_r, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=lora_r, bias=False)
    self.lora_up = nn.Conv2d(lora_r, C_out, kernel_size=1, padding=0, bias=False)

  def forward(self, x):
    out = self.bn(self.pointwise(self.depthwise(self.activation(x))))

    if self.use_lora is True:
      lora_out = self.lora_up(self.lora_down(x))
      out = out + lora_out

    return out


class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, use_lora=False, lora_r=4):
    super(SepConv, self).__init__()

    self.activation = nn.ReLU(inplace=False)
    self.conv1_depthwise = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False)
    self.conv1_pointwise = nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(C_in, affine=affine)

    # nn.ReLU(inplace=False)
    self.conv2_depthwise = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False)
    self.conv2_pointwise = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
    self.bn2 = nn.BatchNorm2d(C_out, affine=affine)

    self.use_lora = use_lora
    self.lora_r = lora_r

    self.lora_down = nn.Conv2d(C_in, lora_r, kernel_size=kernel_size, stride=stride, padding=padding, groups=lora_r, bias=False)
    self.lora_up = nn.Conv2d(lora_r, C_out, kernel_size=1, padding=0, bias=False)

  def forward(self, x):
    out1 = self.bn1(self.conv1_pointwise(self.conv1_depthwise(self.activation(x))))
    out2 = self.bn2(self.conv2_pointwise(self.conv2_depthwise(self.activation(out1))))

    if self.use_lora is True:
      lora_out = self.lora_up(self.lora_down(x))
      out = out2 + lora_out

    return out

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True, use_lora=False, lora_r=4):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

    self.use_lora = use_lora
    self.lora_r = lora_r

    self.conv1_lora_down = nn.Conv2d(C_in, lora_r, kernel_size=1, stride=2, padding=0, bias=False)
    self.conv1_lora_up = nn.Conv2d(lora_r, C_out//2, kernel_size=1, stride=1, padding=0, bias=False)
    self.conv2_lora_down = nn.Conv2d(C_in, lora_r, kernel_size=1, stride=2, padding=0, bias=False)
    self.conv2_lora_up = nn.Conv2d(lora_r, C_out//2, kernel_size=1, stride=1, padding=0, bias=False)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)

    if self.use_lora is True:
      lora_out = torch.cat([self.conv1_lora_up(self.conv1_lora_down(x)), self.conv2_lora_up(self.conv2_lora_down(x[:,:,1:,1:]))], dim=1)
      out = out + lora_out

    return out
