from .graspcvae import GraspCVAE
from .contactnet import ContactNet
def build_model(cfg):
    if cfg.name.lower() == "graspcvae":
        return GraspCVAE(cfg)
    elif cfg.name.lower() == "contactnet":
        return ContactNet(cfg)
    else:
        raise NotImplementedError(f"No such model: {cfg.name}")
