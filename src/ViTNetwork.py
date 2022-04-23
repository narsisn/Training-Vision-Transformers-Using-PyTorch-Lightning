# Importing PyTorch Libraries 
import torch
import torch.nn
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# Importing Ligthening Libraries 

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
import torchmetrics


# Importing ViT Libraries

from vit_pytorch import ViT
from transformers import ViTConfig

# ViT configs class: ViT paramerts setting 
class ViTConfigExtended(ViTConfig):
    def __init__(self, hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

# ViT Network 
class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ViT(
            image_size = config.image_size,
            patch_size = config.patch_size,
            num_classes = config.num_classes,
            dim = config.hidden_size,
            depth = config.num_hidden_layers,
            heads = config.num_attention_heads,
            mlp_dim = config.intermediate_size,
            dropout = config.hidden_dropout_prob,
            emb_dropout = config.attention_probs_dropout_prob
        )
    
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
            
        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)
    
    def forward(self, x):
        return self.model(x)

# ViT Backbone Class 

class Backbone(torch.nn.Module):
    def __init__(self, model_type, config):
        super().__init__()
        if model_type == 'vit':
            self.model = VisionTransformer(config)
        
    def forward(self, x):
        return self.model(x)

# Lightning Classifier  

class LitClassifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-3):
        super().__init__()
        self.backbone = backbone
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_acc(y_hat.softmax(dim=-1), y)
        metrics = {'val_acc': self.val_acc, 'val_loss': loss}
        self.log_dict(metrics, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.log('test_acc', self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


