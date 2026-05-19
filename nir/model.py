import torch
import torch.nn as nn
import torchvision as tv
import lightning as L
import torchmetrics
import numpy as np


class LightningCIFARClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 100,
        lr: float = 1e-3,
        orth_lambda: float = 0.0,
        bn_gamma_lambda: float = 0.0,
        weight_decay: float = 5e-4,
        weight_decay_conv: float | None = None,
        warmup_fraction: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.orth_lambda = orth_lambda
        self.bn_gamma_lambda = bn_gamma_lambda
        self.weight_decay = weight_decay
        self.weight_decay_conv = (
            weight_decay if weight_decay_conv is None else weight_decay_conv
        )
        self.warmup_fraction = warmup_fraction
        self.num_classes = num_classes

        self.model = self._get_model(model_name)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def _get_model(self, model_name: str):
        if model_name == "resnet18":
            model = tv.models.resnet18(num_classes=self.num_classes)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            model.maxpool = nn.Identity()
            return model
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # --- Orthogonal / BN gamma losses ---
    @staticmethod
    def deconv_orth_dist(kernel, stride=1, padding=1):
        o_c, i_c, w, h = kernel.shape
        output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
        target = torch.zeros(
            o_c, o_c, output.shape[-2], output.shape[-1], device=kernel.device
        )
        ct = int(np.floor(output.shape[-1] / 2))
        target[:, :, ct, ct] = torch.eye(o_c, device=kernel.device)
        return torch.norm(output - target)

    @staticmethod
    def orth_dist(mat):
        mat = mat.reshape(mat.shape[0], -1)
        if mat.shape[0] < mat.shape[1]:
            mat = mat.permute(1, 0)
        return torch.norm(
            torch.t(mat) @ mat - torch.eye(mat.shape[1], device=mat.device)
        )

    def compute_orthogonal_loss(self):
        if self.orth_lambda <= 0:
            return 0.0
        diff = 0.0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if name == "conv1":
                    continue
                ker = module.kernel_size
                if ker == (3, 3):
                    diff += self.orth_dist(module.weight)
                else:
                    diff += self.deconv_orth_dist(module.weight)
        return self.orth_lambda * diff

    def compute_bn_gamma_loss(self):
        if self.bn_gamma_lambda <= 0:
            return 0.0
        l1 = 0.0
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                l1 += torch.sum(torch.abs(m.weight))
        return self.bn_gamma_lambda * l1

    # --- Optimizer, scheduler ---
    def configure_optimizers(self):
        # Группы параметров
        conv_weights = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "conv" in name and "weight" in name:
                conv_weights.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": conv_weights, "weight_decay": self.weight_decay_conv},
            {"params": other_params, "weight_decay": self.weight_decay},
        ]

        optimizer = torch.optim.SGD(
            param_groups, lr=self.lr, momentum=0.9, nesterov=True
        )

        # Планировщик: warmup + cosine annealing
        total_epochs = self.trainer.max_epochs if self.trainer else 200
        warmup_epochs = max(1, int(total_epochs * self.warmup_fraction))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "lr_scheduler",
            },
        }

    # --- Шаги обучения / валидации ---
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)

        orth = self.compute_orthogonal_loss()
        bn = self.compute_bn_gamma_loss()
        total_loss = loss + orth + bn

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        if orth > 0:
            self.log("train_orth_loss", orth, on_step=True, on_epoch=True)
        if bn > 0:
            self.log("train_bn_gamma_loss", bn, on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "test")

    def _shared_eval_step(self, batch, batch_idx, prefix):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        if prefix == "val":
            acc = self.val_acc(logits.argmax(1), y)
            self.log(
                f"{prefix}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True
            )
            self.log(f"{prefix}_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
        else:
            acc = self.test_acc(logits.argmax(1), y)
            self.log(f"{prefix}_loss", loss, on_epoch=True, sync_dist=True)
            self.log(f"{prefix}_acc", acc, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()

    def on_test_epoch_end(self):
        self.log("test_acc_epoch", self.test_acc.compute(), prog_bar=True)
        self.test_acc.reset()
