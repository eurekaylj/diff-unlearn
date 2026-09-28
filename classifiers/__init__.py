from .resnet import resnet18, resnet50
from .vgg import vgg16_bn
from .densenet import densenet121
v001 = {'resnet18': lambda v086: resnet18(3, v086), 'resnet50': lambda v086: resnet50(3, v086), 'vgg16_bn': lambda v086: vgg16_bn(3, v086), 'densenet121': lambda v086: densenet121(num_classes=v086)}
