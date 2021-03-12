import torch.nn as nn

cfg = {
       'VGG11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'VGG13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'VGG16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'VGG19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
       }
class VGG(nn.Module):
    def __init__(self, VGG_name, num_classes):
        super(VGG, self).__init__()
        self.vgg = self.make_layers(cfg[VGG_name])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.v(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
    def make_layers(self,cfg):
        layers = []
        input_channel = 3
        for i in cfg:
            if i == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(input_channel, i, kernel_size=3, padding=1), nn.BatchNorm2d(i), nn.ReLU(inplace=True)]
                input_channel = i
        layers += [nn.AvgPool2d(kernel_size=7, stride=7)]
        return nn.Sequential(*layers)
