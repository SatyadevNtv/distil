import torch
import torch.nn.functional as F

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.activation = activation

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.activation is not None:
            x = self.activation(x)
        return x


class DepthwiseSeparableConv2d(torch.nn.Module):
    """
    A kxk depthwise convolution + a 1x1 convolution.
    In :paper:`xception`, norm & activation are applied on the second conv.
    :paper:`mobilenet` uses norm & activation on both convs.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        dilation=1,
        activation=None
    ):
        super().__init__()
        self.depthwise = Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=True,
        )
        self.pointwise = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
            activation=activation,
        )


    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class BaseConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            3,
            padding="same",
            activation=torch.nn.ReLU()
        )
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            3,
            padding="same",
            activation=torch.nn.ReLU()
        )
        self.pool = torch.nn.MaxPool2d(2)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.pool(out)


class BaseSeperableConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seperableConv1 = DepthwiseSeparableConv2d(in_channels, out_channels, activation=torch.nn.ReLU())
        self.seperableConv2 = DepthwiseSeparableConv2d(out_channels, out_channels, activation=torch.nn.ReLU())
        self.norm = torch.nn.BatchNorm2d(out_channels)
        self.pool = torch.nn.MaxPool2d(2)


    def forward(self, x):
        out = self.seperableConv1(x)
        out = self.seperableConv2(out)
        out = self.norm(out)
        return self.pool(out)


class BaseSeperableConvDropBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = BaseSeperableConvBlock(in_channels, out_channels)
        self.drop = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        return self.drop(self.conv(x))


class XRayNet(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = BaseConvBlock(in_channels, 16)
        self.sepConv1 = BaseSeperableConvBlock(16, 32)
        self.sepConv2 = BaseSeperableConvBlock(32, 64)
        self.sepConv3 = BaseSeperableConvDropBlock(64, 128)
        self.sepConv4 = BaseSeperableConvDropBlock(128, 256)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(4096, 512)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(p=0.7)
        self.fc2 = torch.nn.Linear(512, 128)
        self.relu2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(p=0.5)
        self.fc3 = torch.nn.Linear(128, 64)
        self.relu3 = torch.nn.ReLU()
        self.drop3 = torch.nn.Dropout(p=0.3)
        self.classification_layer = torch.nn.Linear(64, out_channels)


    def forwardUptoLast(self, x):
        out = self.conv1(x)
        out = self.sepConv1(out)
        out = self.sepConv2(out)
        out = self.sepConv3(out)
        out = self.sepConv4(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.drop3(out)
        return out


    # Support for CORDS
    def forward(self, x, last=False, freeze=False):
        if freeze:
            self.eval()
            with torch.no_grad():
                emb = self.forwardUptoLast(x)
            self.train()
        else:
            emb = self.forwardUptoLast(x)
        out = self.classification_layer(emb)
        if last:
            return out, emb
        return out

    def get_embedding_dim(self):
        return 64
