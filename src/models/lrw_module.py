from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy


class LRWLitModule(LightningModule):
    """Example of LightningModule for LRW classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        scheduler_max_epochs: int = 30,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False
        )  # NOTE: can't ignore saving networks otherwise it will fail to load!

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch["video"], batch["label"].long()  # convert label to long tensor
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduluer = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.scheduler_max_epochs, eta_min=5e-6
        )
        return [optimizer], [scheduluer]


class LRWKDLitModule(LightningModule):
    """Training student model using Knowledge Distillation."""

    def __init__(
        self,
        teacher_net: str,
        border: bool,
        net: torch.nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        loss_scaler: float = 1.0,
        scheduler_max_epochs: int = 30,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])  # ignore saving networks

        # model
        self.net = net
        # load teacher model, load weights from checkpoint
        self.teacher_model = LRWKDLitModule.load_lrw_model_from_checkpoint(teacher_net)

        # define two loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        self.distill_loss = torch.nn.L1Loss(reduction="sum")

        # metrics for train, val and test step
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch["video"], batch["label"].long()  # convert label to long tensor
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        x, y = batch["video"], batch["label"].long()  # convert label to long tensor

        logits_student = self.forward(x)  # student prediction

        border = batch["duration"].float()
        if self.hparams.border:
            logits_teacher = self.teacher_model(x, border)  # prediction using border
        else:
            logits_teacher = self.teacher_model.forward(x)  # teacher prediction

        loss = self.hparams.loss_scaler * self.distill_loss(  # FIXME: using factor
            logits_student, logits_teacher
        )  # loss based on mse loss between student_pred and teacher_pred

        # student predictions
        preds_student = torch.argmax(logits_student, dim=1)

        # log train metrics
        acc = self.train_acc(preds_student, y)  # accuracy of student based on true target label
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds_student, "targets": y}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Similar to validation step."""
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduluer = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.scheduler_max_epochs, eta_min=5e-6
        )
        return [optimizer], [scheduluer]

    @staticmethod
    def load_lrw_model_from_checkpoint(net: str) -> torch.nn.Module:
        CHECKPOINT_ROOT = "/home/iwawiwi/research/22/lipreading-lightning/data/checkpoint/"
        model_path = ""
        model_hparams = {}
        if net == "baseline":
            model_path = CHECKPOINT_ROOT + "lrw-cosine-lr-acc-0.85080.pt"
            model_hparams = {"se": False, "border": False, "n_classes": 500}
        elif net == "border":
            model_path = CHECKPOINT_ROOT + "lrw-border-cosine-lr-acc-0.87520.pt"
            model_hparams = {"se": False, "border": True, "n_classes": 500}
        elif net == "border-se-mixup-smooth":
            model_hparams = {"se": True, "border": True, "n_classes": 500}
            model_path = (
                CHECKPOINT_ROOT
                + "lrw-border-se-mixup-label-smooth-cosine-lr-wd-1e-4-acc-0.88460.pt"
            )

        state_dict = torch.load(model_path)["video_model"]  # return state dictionary
        try:
            # model implementation from https://github.com/VIPL-Audio-Visual-Speech-Understanding/learn-an-effective-lip-reading-model-without-pains/
            from src.models.components.lrw_video_model import VideoModel

            model = VideoModel(args=model_hparams)
        except Exception:
            raise Exception("Model not found")

        return model  # return model
