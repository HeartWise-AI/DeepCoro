# This script download necessary models at Docker Build Time

import os

from utils_refac import choose_model
from segmentation_models import SegmentationModels

from huggingface_hub import snapshot_download
from dotenv import load_dotenv


class ModelsDownloader:
    """
    A class to handle the downloading of models required for DeepCoro.
    """
        
    deepcoro_path = '/opt/deepcoro/models'
    repo_id = 'heartwise/DeepCoro'
    env_path = '/opt/deepcoro/.env'
    
    models_list = (SegmentationModels.DeepLabV3, SegmentationModels.FPN, SegmentationModels.DeepLabV3Plus, SegmentationModels.PAN)
    
    
    @classmethod
    def download_hf_models(cls):
        """
        Download models from Hugging Face Hub.
        """        
        
        os.makedirs(cls.deepcoro_path, exist_ok=True)
               
        load_dotenv(cls.env_path)
        hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')

        snapshot_download(repo_id=cls.repo_id, local_dir=cls.deepcoro_path, use_auth_token=hf_token)
    
    @classmethod
    def download_pytorch_models(cls):
        """
        Download PyTorch models.
        """        
        
        for model in cls.models_list:
            choose_model(model)
        


if __name__ == '__main__':
    ModelsDownloader.download_hf_models()
    ModelsDownloader.download_pytorch_models()

