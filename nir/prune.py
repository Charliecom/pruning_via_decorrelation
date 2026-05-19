import copy
from pathlib import Path
import tempfile

import hydra
import mlflow
import pandas as pd
import torch
import torch.nn as nn
import torch_pruning as tp
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from myutils import fix_seed
from data import GenericDataModule
from model import LightningCIFARClassifier


def get_importance(name: str):
    """Преобразует строку в класс важности torch_pruning."""
    mapping = {
        "l1": tp.importance.MagnitudeImportance(p=1),
        "l2": tp.importance.MagnitudeImportance(p=2),
        "taylor": tp.importance.TaylorImportance(),
        "bnscale": tp.importance.BNScaleImportance(),
    }
    return mapping[name.lower()]


def evaluate(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
):
    """Возвращает точность (accuracy) на датасете."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def prune_model(
    model: nn.Module,
    pruning_ratio: float,
    importance: tp.importance.Importance,
    device: torch.device,
):
    """Прунит модель и возвращает (pruned_model, cr_flops, cr_params)."""
    new_model = copy.deepcopy(model)
    example_inputs = torch.rand(1, 3, 32, 32, dtype=torch.float32, device=device)

    base_flops, base_params = tp.utils.count_ops_and_params(model, example_inputs)

    pruner = tp.pruner.BasePruner(
        model=new_model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=pruning_ratio,
        global_pruning=True,
        ignored_layers=[new_model.fc, new_model.conv1],
        round_to=8,
    )
    pruner.step()

    flops, params = tp.utils.count_ops_and_params(new_model, example_inputs)

    cr_flops = base_flops / flops if flops > 0 else float("inf")
    cr_params = base_params / params if params > 0 else float("inf")

    return new_model, cr_flops, cr_params


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    fix_seed(42)

    prune_cfg = cfg.prune
    experiment_name = prune_cfg.experiment_name
    run_id = prune_cfg.run_id

    client = mlflow.tracking.MlflowClient(tracking_uri=cfg.get("tracking_uri", None))

    if run_id is None:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found.")
        runs = client.search_runs(
            experiment.experiment_id, order_by=["start_time DESC"], max_results=1
        )
        if not runs:
            raise ValueError("No runs found in experiment.")
        run_id = runs[0].info.run_id
        print(f"Using latest run: {run_id}")
    else:
        print(f"Using provided run_id: {run_id}")

    # Получаем путь к директории run через artifact_uri
    run_info = client.get_run(run_id)
    artifact_uri = run_info.info.artifact_uri

    if artifact_uri.startswith("file://"):
        artifact_path = Path(artifact_uri[7:])
    else:
        artifact_path = Path(artifact_uri)

    # В структуре mlruns чекпоинты лежат в родительской директории от artifacts
    run_dir = artifact_path.parent
    print(f"Run directory: {run_dir}")

    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    ckpt_name = prune_cfg.ckpt_name
    ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
    print(f"Available checkpoints: {[f.name for f in ckpt_files]}")

    local_path = None
    for ckpt_file in ckpt_files:
        if ckpt_name in ckpt_file.name:
            local_path = ckpt_file
            break

    if local_path is None:
        fallback_path = checkpoints_dir / "last.ckpt"
        if fallback_path.exists():
            local_path = fallback_path
            print("Using last.ckpt as fallback")
        else:
            raise FileNotFoundError(
                f"Checkpoint containing '{ckpt_name}' not found in {checkpoints_dir}. "
                f"Available: {[f.name for f in ckpt_files]}"
            )

    print(f"Loading checkpoint: {local_path}")

    lit_model = LightningCIFARClassifier.load_from_checkpoint(str(local_path))
    model = lit_model.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset_name = prune_cfg.get("dataset", cfg.data.dataset)
    dm = GenericDataModule(
        dataset=dataset_name,
        data_path=prune_cfg.get("data_path", cfg.data.data_path),
        batch_size=prune_cfg.get("batch_size", cfg.data.batch_size),
        num_workers=prune_cfg.get("num_workers", cfg.data.num_workers),
        val_split=0.0,
    )
    dm.setup()
    test_loader = dm.test_dataloader()

    importance = get_importance(prune_cfg.importance)
    ratios = prune_cfg.pruning_ratios

    results = []

    with mlflow.start_run(run_id=run_id):
        print("Evaluating original model...")
        base_acc = evaluate(model, test_loader, device)
        print(f"Original accuracy: {base_acc:.4f}")
        results.append(
            {
                "pruning_ratio": 0.0,
                "accuracy": base_acc,
                "cr_flops": 1.0,
                "cr_params": 1.0,
            }
        )

        for ratio in ratios:
            print(f"\nPruning with ratio {ratio:.2f}...")
            pruned_model, cr_flops, cr_params = prune_model(
                model, ratio, importance, device
            )
            acc = evaluate(pruned_model, test_loader, device)
            print(
                f"Pruned accuracy: {acc:.4f}, CR_FLOPs: {cr_flops:.2f}, CR_Params: {cr_params:.2f}"
            )
            results.append(
                {
                    "pruning_ratio": ratio,
                    "accuracy": acc,
                    "cr_flops": cr_flops,
                    "cr_params": cr_params,
                }
            )

    # Сохраняем результаты во временную директорию
    tmp_dir = Path(tempfile.mkdtemp())
    df = pd.DataFrame(results)
    csv_path = tmp_dir / "pruning_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults:\n{df}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    crf_arr = df["cr_flops"].values
    crp_arr = df["cr_params"].values
    acc_arr = df["accuracy"].values

    ax1.plot(crf_arr, acc_arr, "o-", label="Accuracy")
    ax1.set_xlabel("Compression Ratio (FLOPs)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs FLOPs Compression")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(crp_arr, acc_arr, "o-", label="Accuracy")
    ax2.set_xlabel("Compression Ratio (Params)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs Params Compression")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plot_path = tmp_dir / "pruning_analysis.png"
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close()

    # Логируем артефакты в MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(csv_path), artifact_path="pruning")
        mlflow.log_artifact(str(plot_path), artifact_path="pruning")
        mlflow.log_params(
            {
                "prune_importance": prune_cfg.importance,
                "prune_ratios": str(prune_cfg.pruning_ratios),
            }
        )
        print(f"Artifacts logged to run {run_id}")

    # Очистка временной директории
    import shutil

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
