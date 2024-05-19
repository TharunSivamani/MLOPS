from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import gradio as gr
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:

    assert cfg.ckpt_path

    log.info("Running Demo Function")

    log.info(f"Instantiating Scripted Model <{cfg.ckpt_path}>")

    model = torch.jit.load(cfg.ckpt_path)

    log.info(f"Loaded Model using JIT: {model}")

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def recognize_image(img):
        if img is None:
            return None
        
        image = torch.tensor(img[None,...], dtype=torch.float32)

        preds = model.forward_jit(image)
        preds = preds[0].tolist()
        out = {
            classes[i]: preds[i] for i in range(classes)
        }

        return out
    
    im = gr.Image(shape=(32, 32))

    demo = gr.Interface(
        fn = recognize_image,
        inputs = [ 
            im 
        ],
        outputs = [
            gr.Label(num_top_classes=10)
        ],
        live = True,
    )

    demo.launch(server_name="0.0.0.0")

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="demo_trace.yaml")
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()

    