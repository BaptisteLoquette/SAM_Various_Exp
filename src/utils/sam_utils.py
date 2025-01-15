import torch
import os
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

def get_sam_model(model_type = "vit_l"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    checkpoint_path = os.path.join(models_dir, f"sam_{model_type}_4b8939.pth")

    ## If SAM isn't downloaded, download it
    if not os.path.exists(checkpoint_path):
        import urllib.request
        
        print(f"Downloadinbf SAM {model_type} ...")
        if model_type == "vit_h":
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        elif model_type == "vit_l":
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        elif model_type == "vit_b":
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        
        # Create class for progress bar
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None): 
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        with DownloadProgressBar(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                            desc="SAM checkpoint") as t:
            urllib.request.urlretrieve(url, checkpoint_path, reporthook=t.update_to)
        
        print("Download done!")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    PREDICTOR = SamPredictor(sam)
    return PREDICTOR