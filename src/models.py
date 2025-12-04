import torch
import torch.nn as nn
import torchvision.models as models

# --- 1. ResNet-based CRNN (สำหรับ OCR) ---
class ResNetCRNN(nn.Module):
    def __init__(self, img_channel, num_classes, hidden_size=256, num_rnn_layers=2, dropout_rnn=0.3):
        super().__init__()
        # Load ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # ปรับ Input Layer รับ Grayscale (1 channel)
        resnet.conv1 = nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # ปรับ Stride เพื่อรักษาความกว้าง (Width)
        resnet.layer2[0].conv1.stride = (2, 1)
        resnet.layer2[0].downsample[0].stride = (2, 1)
        resnet.layer3[0].conv1.stride = (2, 1)
        resnet.layer3[0].downsample[0].stride = (2, 1)
        resnet.layer4[0].conv1.stride = (2, 1)
        resnet.layer4[0].downsample[0].stride = (2, 1)

        self.cnn = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        self.out_channels = 512
        self.rnn = nn.LSTM(self.out_channels, hidden_size, num_layers=num_rnn_layers,
                           bidirectional=True, batch_first=True, dropout=dropout_rnn)
        self.linear = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = torch.mean(features, dim=2) # Mean Pool Height
        features = features.permute(0, 2, 1)   # [Batch, Time, Features]
        rnn_out, _ = self.rnn(features)
        output = self.linear(rnn_out)
        return output

# --- 2. Province Classifier ---
class ProvinceClassifier(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = models.mobilenet_v2(weights=weights)
        
        # Classifier Head
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(self.model.last_channel, n_classes)
        )

    def forward(self, x):
        return self.model(x)