import torch.nn as nn
from torchvision.models import swin_t

class FrameClassifierSwinT(nn.Module):
    def __init__(self, gru_model):
        super().__init__()

        self.swin_t = swin_t(weights=None)
        self.swin_t.head = nn.Identity() # remove the classification head, (N, 768)
        self.gru = gru_model

    def forward(self, x, h=None):
        # x is (N, 3, 224, 224)
        # h is None at the beginning of the video
        x = self.swin_t(x) # (N, 768)
        out, h = self.gru(x, h) # (N, 2_classes), (Layers, 1, Hidden)
        return out, h # (N, 2_classes), (Layers, 1, Hidden)
    

if __name__ == "__main__":
    from ConvNeXtBased import EfficientGRUModel
    gru_model = EfficientGRUModel()
    model = FrameClassifierSwinT(gru_model)