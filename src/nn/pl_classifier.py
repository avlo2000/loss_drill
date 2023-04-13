import lightning as pl
import torch
from torch import nn
from torch import optim
from torchmetrics import Accuracy


class LitClassifer(pl.LightningModule):
    def __init__(self, model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.model = model
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, *args):
        x, y_trg = batch
        y = self(x)
        loss = nn.functional.cross_entropy(y, y_trg)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, *args):
        x, y_trg = batch
        y = self(x)
        loss = nn.functional.cross_entropy(y, y_trg)
        preds = torch.argmax(y, dim=1)
        self.val_accuracy.update(preds, y)
        
        self.log('val_loss', loss)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y_trg = batch
        y = self(x)
        loss = nn.functional.cross_entropy(y, y_trg)
        preds = torch.argmax(y, dim=1)
        self.test_accuracy.update(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
