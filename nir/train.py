import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    LearningRateFinder,
)
import torch

from myutils import fix_seed
from data import GenericDataModule
from model import LightningCIFARClassifier


def build_callbacks(cfg: DictConfig):
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            filename="best",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]

    # Early stopping
    if cfg.trainer.get("early_stopping", True):
        callbacks.append(
            EarlyStopping(
                monitor="val_acc",
                patience=cfg.trainer.get("patience", 20),
                mode="max",
                verbose=True,
            )
        )

    # Опциональный LR Finder
    if cfg.trainer.get("lr_finder", False):
        callbacks.append(
            LearningRateFinder(min_lr=1e-6, max_lr=1e-2, num_training_steps=100)
        )

    return callbacks


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    fix_seed(cfg.trainer.seed)

    # --- Данные ---
    num_classes_map = {"cifar10": 10, "cifar100": 100, "svhn": 10}
    dataset_name = cfg.data.dataset.lower()
    num_classes = num_classes_map[dataset_name]

    dm = GenericDataModule(
        dataset=dataset_name,
        data_path=cfg.data.data_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
    )

    wd_conv = (
        cfg.trainer.wd
        if cfg.trainer.get("wd_on_conv") is None
        else cfg.trainer.wd_on_conv
    )

    # --- Модель ---
    model = LightningCIFARClassifier(
        model_name=cfg.model.name,
        num_classes=num_classes,
        lr=cfg.trainer.lr,
        orth_lambda=cfg.trainer.get("alpha", 0.0),
        bn_gamma_lambda=cfg.trainer.get("beta", 0.0),
        weight_decay=cfg.trainer.wd,
        weight_decay_conv=wd_conv,
        warmup_fraction=cfg.trainer.get("warmup_fraction", 0.1),
    )

    # --- MLflow Logger ---
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.get("experiment_name", "cifar_experiment"),
        tracking_uri=cfg.get("tracking_uri", None),
    )
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # --- Коллбэки ---
    callbacks = build_callbacks(cfg)

    # --- Trainer ---
    trainer = L.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.trainer.get("devices", 1),
        logger=mlflow_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    # Если указан чекпоинт для возобновления
    if cfg.trainer.get("ckpt_path"):
        trainer.fit(model, dm, ckpt_path=cfg.trainer.ckpt_path)
    else:
        trainer.fit(model, dm)

    # Тест с лучшей моделью
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        trainer.test(model, dm, ckpt_path=trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
