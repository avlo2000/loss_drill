import lightning as pl
import torch
from torch import nn
from torch import optim


class LitClassifer(pl.LightningModule):
    def __init__(self, model: nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, *args):
        x, y_trg = batch
        y = self.model(x)
        loss = nn.functional.cross_entropy(y, y_trg)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
