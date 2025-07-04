import torch
import torch.nn as nn
from vit_pytorch import SimpleViT

class CombinedNetwork(nn.Module):
    def __init__(self, fimage_size=(3, 588), fpatch_size = 3, FCN_input_dim=9, inchannels=2):
        super(CombinedNetwork, self).__init__()
        self.vit = SimpleViT(image_size=fimage_size, patch_size=fpatch_size, num_classes = 256, dim = 512, depth = 6, heads = 8, mlp_dim = 2048, channels = inchannels)
        self.relu = nn.ReLU(inplace=True)
        self.fc1=nn.Linear(FCN_input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2=nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3=nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.bn4 = nn.BatchNorm1d(256)

        self.fc5=nn.Linear(128+256, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc6=nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc7=nn.Linear(256, 2)


    def forward(self, vit_input, feature_input):
        
        feature_input=self.fc1(feature_input)
        feature_input=self.bn1(feature_input)
        feature_input=self.relu(feature_input)
        feature_input=self.fc2(feature_input)
        feature_input=self.bn2(feature_input)
        feature_input=self.relu(feature_input)
        feature_input=self.fc3(feature_input)
        feature_input=self.bn3(feature_input)
        feature_input=self.relu(feature_input)

        vit_input = self.vit(vit_input)
        vit_input = self.bn4(vit_input)
        vit_input = self.relu(vit_input)

        combined_input = torch.cat((vit_input, feature_input), dim=1)

        output = self.fc5(combined_input)
        output = self.bn5(output)
        output = self.relu(output)
        output = self.fc6(output)
        output = self.bn6(output)
        output = self.relu(output)
        output = self.fc7(output)
        
        return output