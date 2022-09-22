import pathlib

import hydra
from omegaconf import OmegaConf

import src


@hydra.main(config_path="configs", config_name="base_config.yaml")
def main(cfg):
    outdir = pathlib.Path.cwd()  # set by hydra
    cfg = OmegaConf.to_container(cfg, resolve=True)
    tr = cfg["training_regimes"]
    indices = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    cfg["training_regimes"] = [tr[indices[i]] for i in range(len(tr))]
    print(cfg)
    src.main(**cfg, outdir=outdir, config_dict=cfg)

if __name__ == "__main__":
    main()
