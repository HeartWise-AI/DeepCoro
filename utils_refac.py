import numpy as np
from scipy.ndimage import uniform_filter1d
import cv2
import torch
import torchvision
import segmentation_models_pytorch
import scipy.ndimage
import torch.nn as nn
import torchvision.models.video
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Union
import os 

from segmentation_models import SegmentationModels




def to_camel_case(string: str) -> str:
    """
    Convert a string from snake_case to camelCase.

    Args:
        string (str): The input string in snake_case.

    Returns:
        str: The string converted to camelCase.
    """
    if '_' not in string:
        return string

    parts = string.split('_')
    return parts[0] + ''.join(part.capitalize() for part in parts[1:])


###################
### Algorithm 3 ###
###################

def resize_coordinates(dict_coordinates: Dict[str, float], pixel_spacing: float, img_shape: Tuple[int, int, int], target_size: float) -> Dict[str, int]:
    """
    Resize the bounding box coordinates based on the target size.

    Args:
        dict_coordinates (Dict[str, float]): Dictionary containing the bounding box coordinates.
        pixel_spacing (float): The pixel spacing of the DICOM image.
        img_shape (Tuple[int, int, int]): The shape of the DICOM image.
        target_size (float): The target size for resizing the bounding box.

    Returns:
        Dict[str, int]: Dictionary containing the resized bounding box coordinates.
    """
    xc = np.floor((dict_coordinates['x2'] + dict_coordinates['x1']) / 2)
    yc = np.floor((dict_coordinates['y2'] + dict_coordinates['y1']) / 2)

    nPixel = target_size / pixel_spacing

    (x1n, y1n, x2n, y2n) = (
        int(np.floor(xc - nPixel / 2)),
        int(np.floor(yc - nPixel / 2)),
        int(np.floor(xc + nPixel / 2)),
        int(np.floor(yc + nPixel / 2))
    )

    if x1n < 0:
        x2n -= x1n
        x1n = 0
    if x2n > img_shape[1]:
        if x1n != 0:
            x1n -= x2n - img_shape[1]
        x2n = img_shape[1]
    if y1n < 0:
        y2n -= y1n
        y1n = 0
    if y2n > img_shape[2]:
        if y1n != 0:
            y1n -= y2n - img_shape[2]
        y2n = img_shape[2]

    assert abs(y2n - y1n) == abs(x2n - x1n), "Resized region is not square."

    return {
        'x1': x1n,
        'y1': y1n,
        'x2': x2n,
        'y2': y2n
    }


def register(img: np.ndarray, index: int, box_coordinates: Dict[str, int]) -> Dict[int, List[float]]:
    """
    Perform image registration using the CSRT tracker.

    Args:
        img (np.ndarray): The input image array.
        index (int): The index of the frame to start registration from.
        box_coordinates (Dict[str, int]): Dictionary containing the bounding box coordinates.

    Returns:
        Dict[int, List[float]]: Dictionary containing the registration shifts for each frame.
    """
    img_after = img[index:]
    img_before = np.flip(img[: index + 1], 0)
    plx, ply = [], []
    tracker = cv2.TrackerCSRT.create()

    tracker.init(
        img_before[0],
        [
            box_coordinates['x1'],
            box_coordinates['y1'],
            box_coordinates['x2'] - box_coordinates['x1'],
            box_coordinates['y2'] - box_coordinates['y1']
        ]
    )

    for i in range(img_before.shape[0]):
        moving = img_before[i]
        _, bbox = tracker.update(moving)
        (x, y, w, h) = (int(v) for v in bbox)

        yoff = int(y + h / 2 - (box_coordinates['y2'] - box_coordinates['y1']) / 2) - box_coordinates['y1']
        xoff = int(x + w / 2 - (box_coordinates['x2'] - box_coordinates['x1']) / 2) - box_coordinates['x1']

        plx.append(xoff)
        ply.append(yoff)

    plx.reverse()
    ply.reverse()

    plx.pop()
    ply.pop()

    tracker.init(img_after[0], [box_coordinates['x1'], box_coordinates['y1'], box_coordinates['x2'] - box_coordinates['x1'], box_coordinates['y2'] - box_coordinates['y1']])
    for i in range(img_after.shape[0]):
        moving = img_after[i]
        _, bbox = tracker.update(moving)
        (x, y, w, h) = (int(v) for v in bbox)

        yoff = int(y + h / 2 - (box_coordinates['y2'] - box_coordinates['y1']) / 2) - box_coordinates['y1']
        xoff = int(x + w / 2 - (box_coordinates['x2'] - box_coordinates['x1']) / 2) - box_coordinates['x1']

        plx.append(xoff)
        ply.append(yoff)

    plxo1 = uniform_filter1d(plx, size=5, output=float)
    plyo1 = uniform_filter1d(ply, size=5, output=float)

    reg_shift = {}
    for i in range(img.shape[0]):
        reg_shift[i] = [-plyo1[i], -plxo1[i]]

    return reg_shift


###################
### Algorithm 4 ###
###################

def choose_model(model_name: str) -> 'segmentation_models_pytorch.SegmentationModel':
    """
    Choose the appropriate segmentation model based on the model name.

    Args:
        model_name (str): The name of the segmentation model.

    Returns:
        segmentation_models_pytorch.SegmentationModel: The chosen segmentation model.

    Raises:
        ValueError: If an invalid model name is provided.
    """
    if model_name == SegmentationModels.Unet:
        return segmentation_models_pytorch.Unet(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == SegmentationModels.UnetPlusPlus:
        return segmentation_models_pytorch.UnetPlusPlus(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == SegmentationModels.MAnet:
        return segmentation_models_pytorch.MAnet(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == SegmentationModels.Linknet:
        return segmentation_models_pytorch.Linknet(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == SegmentationModels.FPN:
        return segmentation_models_pytorch.FPN(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == SegmentationModels.PSPNet:
        return segmentation_models_pytorch.PSPNet(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == SegmentationModels.DeepLabV3:
        return segmentation_models_pytorch.DeepLabV3(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == SegmentationModels.DeepLabV3Plus:
        return segmentation_models_pytorch.DeepLabV3Plus(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == SegmentationModels.PAN:
        return segmentation_models_pytorch.PAN(encoder_weights="imagenet", in_channels=3, classes=26)
    else:
        raise ValueError(f"Invalid model name: {model_name}. Please choose a valid model.")


def load_model(checkpoint_path: str, device: str) -> 'torch.nn.Module':
    """
    Load a segmentation model from a checkpoint.

    Args:
        checkpoint_path (str): The path to the model checkpoint.
        device (str): The device to load the model on ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded segmentation model.
    """
    split_parts = checkpoint_path.split('_')
    model_index = split_parts.index("model")
    model_name = split_parts[model_index + 1]
    model = choose_model(model_name)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model


def histogram_equalization_batch(image_batch: 'torch.Tensor') -> 'torch.Tensor':
    """
    Perform histogram equalization on a batch of images.

    Args:
        image_batch (torch.Tensor): The input batch of images.

    Returns:
        torch.Tensor: The batch of images after histogram equalization.
    """
    equalized_images = []
    for image in image_batch:
        image_pil = torchvision.transforms.ToPILImage()(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(np.array(image_pil))
        equalized_images.append(torch.from_numpy(cl1).unsqueeze(0))

    return torch.cat(equalized_images, dim=0)


def process_batch(images_batch: np.ndarray) -> 'torch.Tensor':
    """
    Process a batch of images by performing histogram equalization and normalization.

    Args:
        images_batch (np.ndarray): The input batch of images.

    Returns:
        torch.Tensor: The processed batch of images.
    """
    images_batch_tensor = torch.from_numpy(images_batch).float()
    images_batch_tensor = images_batch_tensor.to(torch.uint8)
    images_batch_tensor = histogram_equalization_batch(images_batch_tensor)
    images_batch_float = images_batch_tensor.float()
    min_vals = images_batch_float.amin(dim=(1, 2), keepdim=True)
    max_vals = images_batch_float.amax(dim=(1, 2), keepdim=True)
    normalized_images_batch = (images_batch_float - min_vals) / (max_vals - min_vals)
    normalized_images_batch = normalized_images_batch.unsqueeze(1).repeat(1, 3, 1, 1)
    return normalized_images_batch


def perform_segmentation_inference_batch(models: List['torch.nn.Module'], batch: 'torch.Tensor', device: str) -> np.ndarray:
    """
    Perform segmentation inference on a batch of images using multiple models.

    Args:
        models (List[torch.nn.Module]): The list of segmentation models.
        batch (torch.Tensor): The input batch of images.
        device (str): The device to perform inference on ('cpu' or 'cuda').

    Returns:
        np.ndarray: The combined segmentation output.
    """
    batch = batch.to(device)
    outputs = []
    with torch.no_grad():
        for model in models:
            outputs.append(model(batch))

    combined_outputs = torch.stack(outputs).mean(dim=0).argmax(1).to('cpu').numpy()
    return combined_outputs


###################
### Algorithm 5 ###
###################

retinanet_artery_labels = {
    1: 'prox_rca',  # 1: RCA proximal
    2: 'mid_rca',  # 2: RCA mid
    3: 'dist_rca',  # 3: RCA distal
    4: 'pda',  # 4: Posterior descending
    5: 'leftmain',  # 5: Left main
    6: 'lad',  # 6: LAD proximal
    7: 'mid_lad',  # 7: LAD mid
    8: 'dist_lad',  # 8: LAD apical
    9: 'other',  # 9: First diagonal
    10: 'other',  # 9a: First diagonal a
    11: 'other', # 10: Second diagonal
    12: 'other', # 10a: Second diagonal a
    13: 'lcx', # 11: Proximal circumflex 
    14: 'other', # 12: Intermediate/anterolateral
    15: 'other', # 12a: Obtuse marginal a 
    16: 'dist_lcx', # 13: Distal circumflex 
    17: 'other', # 14: Left posterolateral 
    18: 'other', # 14a: Left posterolateral a 
    19: 'other', # 15: Posterior descending
    20: 'posterolateral', # 16: Posterolateral from RCA
    21: 'posterolateral', # 16a: Posterolateral from RCA 
    22: 'posterolateral', # 16b: Posterolateral from RCA
    23: 'posterolateral', # 16c: Posterolateral from RCA
    24: 'other', # 12b: Obtuse marginal b
    25: 'other', # 14b: Left posterolateral b
    26: 'other' # stenosis
}

which_artery = {
    "RCA": [1, 2, 3, 4, 20, 21, 22, 23],
    "LCA": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25]
}


def get_segment_region(region: np.ndarray, anatomical_structure: str) -> str:
    """
    Get the segment region based on the object value.

    Args:
        region (np.ndarray): The region of interest.
        anatomical_structure (str): The object value (e.g., 'RCA' or 'LCA').

    Returns:
        str: The segment region.
    """
    
    unique, counts = np.unique(region, return_counts=True)
    region_counts = dict(zip(unique, counts))
    sorted_counts = dict(sorted(region_counts.items(), key=lambda item: item[1], reverse=True))
    if 0 in sorted_counts:
        del sorted_counts[0]
        
    segment = 'None'
    for value in sorted_counts:
        if value in which_artery[anatomical_structure]:
            segment = retinanet_artery_labels[value]
            break
        
    return segment


def get_segment_center(region: np.ndarray, object_value: str) -> str:
    """
    Get the segment at the center of the region.

    Args:
        region (np.ndarray): The region of interest.
        object_value (str): The object value (e.g., 'RCA' or 'LCA').

    Returns:
        str: The segment at the center of the region.
    """
    center = [int(region.shape[0] / 2), int(region.shape[0] / 2)]
    min_length = min(center)
    for length in range(min_length):
        if length != 0:
            subregion = region[center[0] - length: center[0] + length, center[1] - length: center[1] + length]
        else:
            subregion = region[center[0]: center[0] + 1, center[1]: center[1] + 1]
        if subregion.sum() == 0:
            segment = 'None'
        else:
            segment = get_segment_region(subregion, object_value)
        if segment not in ['None']:
            break
    return segment


###################
### Algorithm 6 ###
###################

def create_cropped_registered_video(dicom: np.ndarray, frame: int, reg_shifts: dict, box_coordinates: dict) -> torch.Tensor:
    """
    Create a cropped and registered video from the DICOM data.

    Args:
        dicom (np.ndarray): The DICOM data.
        frame (int): The frame number.
        reg_shifts (dict): The registration shifts for each frame.
        box_coordinates (dict): The bounding box coordinates.

    Returns:
        torch.Tensor: The cropped and registered video.
    """
    FRAMES = 24
    RESIZE = 224
    mean = [144.80116, 144.80116, 144.80116]
    std = [52.669548, 52.669548, 52.669548]

    cropped_img_reg = np.zeros(dicom[:, box_coordinates['y1']:box_coordinates['y2'], box_coordinates['x1']:box_coordinates['x2']].shape)
    for j in range(dicom.shape[0]):
        reg_shift = reg_shifts[j]
        img_reg = scipy.ndimage.shift(dicom[j], shift=(reg_shift[0], reg_shift[1]), order=5)
        cropped_img_reg[j] = img_reg[box_coordinates['y1']:box_coordinates['y2'], box_coordinates['x1']:box_coordinates['x2']]

    cropped_img_reg = np.stack((cropped_img_reg,) * 3, 1)
    cropped_img_reg = torch.from_numpy(cropped_img_reg)
    cropped_img_reg = torchvision.transforms.Resize((RESIZE, RESIZE), antialias=None)(cropped_img_reg)
    cropped_img_reg = torchvision.transforms.Normalize(mean, std)(cropped_img_reg).numpy()

    pad_value = int(FRAMES / 2)
    zero_frame = np.zeros(cropped_img_reg[0:1].shape)
    padded_video = np.concatenate([zero_frame] * pad_value + [cropped_img_reg] + [zero_frame] * pad_value, axis=0)
    pad_frame = frame + pad_value
    final_video = padded_video[pad_frame - pad_value: pad_frame + pad_value]
    final_video = np.stack((final_video,) * 1, 0).transpose((0, 2, 1, 3, 4))

    return torch.from_numpy(final_video)


class Identity(nn.Module):
    """An identity module to pass through the features."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Regressor(nn.Module):
    def __init__(self, num_class: int = 1, num_features: int = 12):
        super(Regressor, self).__init__()

        # Initialize the Swin3D model pre-trained weights
        model = torchvision.models.video.swin3d_b(weights='KINETICS400_IMAGENET22K_V1', progress=False)
        n_inputs = model.head.in_features

        # Replace the original head with Identity to use as a feature extractor
        model.head = Identity()

        self.model = model

        # The fc layer now also includes additional features from age and segment alongside the extracted features
        self.fc = nn.Linear(n_inputs + num_features, num_class)
        self.layer = nn.Sigmoid()

    def forward(self, x: tuple) -> torch.Tensor:
        # Assuming x is a tuple or list of (video, age, segment)
        video_data, age, segment = x

        # Extract features from the video
        features = self.model(video_data)

        # Normalize age
        normalized_age = age / 120

        # Handle segment encoding gracefully for any batch size
        encoded_segment = torch.nn.functional.one_hot(segment.long().squeeze(), num_classes=11).float()

        if encoded_segment.shape == torch.Size([11]):
            encoded_segment = encoded_segment.unsqueeze(0)

        # Concatenate features, encoded segment, and normalized age for the final classification
        combined_features = torch.cat((features, encoded_segment, normalized_age), dim=1)

        # Pass through the fully connected layer and activation
        output = self.fc(combined_features)
        return self.layer(output)


def get_model() -> Regressor:
    """
    Get the Regressor model.

    Returns:
        Regressor: The Regressor model.
    """
    model = Regressor()
    return model


def get_age(dicom_info: 'pydicom.dataset.FileDataset') -> int:
    """
    Get the age from the DICOM information.

    Args:
        dicom_info (pydicom.dataset.FileDataset): The DICOM dataset information.

    Returns:
        int: The age in years.
    """
    study_date = str(dicom_info["StudyDate"].value)
    dob = str(dicom_info[(0x0010, 0x0030)].value)

    study_year, study_month, study_day = int(study_date[0:4]), int(study_date[4:6]), int(study_date[6:8])
    dob_year, dob_month, dob_day = int(dob[0:4]), int(dob[4:6]), int(dob[6:8])

    study_date_obj = date(study_year, study_month, study_day)
    dob_obj = date(dob_year, dob_month, dob_day)

    age = relativedelta(study_date_obj, dob_obj).years

    return age


def object_recon(dicom: np.ndarray, config: dict, device: str, model: 'torch.nn.Module') -> torch.Tensor:
    """
    Perform object reconstruction on the DICOM data.

    Args:
        dicom (np.ndarray): The DICOM data.
        config (dict): The configuration dictionary.
        device (str): The device to use for computation.
        model (torch.nn.Module): The model to use for object reconstruction.

    Returns:
        torch.Tensor: The output of the object reconstruction model.
    """
        
    checkpoint = torch.load(config["model_path"])
    
    model = torchvision.models.video.swin3d_s(weights="KINETICS400_V1")
    n_inputs = model.head.in_features
    model.head = nn.Linear(n_inputs, 11)
    
    model_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    model.load_state_dict(model_state_dict)
    model.to(device)

    output = model(dicom)
    return output



### Model loading 

def load_segmentation_UNet_models(models_dir, model_weights, device):
    return [load_model(os.path.join(models_dir, path), device) for path in model_weights]
    
def load_structure_recognition_Swin3D_model(models_dir, model_weights, device):
        
    model = torchvision.models.video.swin3d_s(weights="KINETICS400_V1")
    n_inputs = model.head.in_features
            
    model.head = nn.Linear(n_inputs, 11)
    
    checkpoint = torch.load(os.path.join(models_dir,model_weights))
    checkpoint = {key.replace('module.model.', ''): value for key, value in checkpoint.items()}
    
    model_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))

    # Check and consume "_orig_mod.module." prefix if present
    if any(k.startswith("_orig_mod.module.") for k in model_state_dict.keys()):
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            model_state_dict, "_orig_mod.module."
        )
        print("Removed prefix '_orig_mod.module.' from state dict")

    # Check and consume "module." prefix if present
    elif any(k.startswith("module.") for k in model_state_dict.keys()):
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            model_state_dict, "module."
        )
        print("Removed prefix 'module.' from state dict")

    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)
    return model
    
def load_stenosis_severity_Swin3D_model(models_dir, model_weights, device):
    severity_prediction_model = get_model().to(device).double()
    checkpoint = torch.load(os.path.join(models_dir, model_weights), map_location=device)["state_dict"]
    
    if device == 'cpu':
        checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
    else:
        severity_prediction_model = torch.nn.DataParallel(severity_prediction_model)
    severity_prediction_model.eval()
    severity_prediction_model.load_state_dict(checkpoint)
    return severity_prediction_model
        