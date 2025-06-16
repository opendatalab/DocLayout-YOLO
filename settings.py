from dataclasses import dataclass
from pathlib import Path

@dataclass
class LayoutParserTrainingSettings:
    from_dataset_repo: str = "agomberto/historical-layout"
    local_data_dir: str = "/home/ubuntu/datasets/data"
    local_model_dir: str = "/home/ubuntu/models"
    from_model_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    from_model_name: str = "doclayout_yolo_docstructbench_imgsz1024.pt"
    pushed_model_name: str = "my_ft_model.pt"
    pushed_model_repo: str = "agomberto/historical-layout-ft-test"
    local_ft_model_dir: str = "/home/ubuntu/yolo_ft"

    # hyperparameters
    batch_size: int = 8
    epochs: int = 5
    image_size: int = 1024
    lr0: float = 0.001
    optimizer: str = "Adam"
    base_model: str = "m-doclayout"
    patience: int = 5
    
    # Optional training parameters (with defaults)
    warmup_epochs: float = 3.0
    momentum: float = 0.9
    mosaic: float = 1.0
    workers: int = 4
    device: str = "0"
    val_period: int = 1
    save_period: int = 10
    plots: bool = False

    @property
    def local_ft_model_name(self) -> str:
        """Get the path to the fine-tuned model"""
        name = (f"yolov10{self.base_model}_{self.local_data_dir}_"
               f"epoch{self.epochs}_imgsz{self.image_size}_"
               f"bs{self.batch_size}_pretrain_docstruct")
        return str(Path(self.local_ft_model_dir) / name / "weights/best.pt")
