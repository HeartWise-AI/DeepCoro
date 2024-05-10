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
import yaml
# from orion.models import pytorchvideo_model
def to_camel_case(string):
    if '_' not in string:
        return string
    
    parts = string.split('_')
    return parts[0] + ''.join(part.capitalize() for part in parts[1:])


def perform_checks(sub_df, dicom_path, dicom_info):
    dicom = dicom_info.pixel_array
    
    if len(sub_df['artery_view'].unique()) != 1:
        print(f"More than one artery view {sub_df['artery_view'].unique()} is associated with the same DICOM {dicom_path}")
        return False
    
    if sub_df['frame'].max() >= dicom.shape[0]:
        print(f"Maximum frame number ({sub_df['frame'].max()}) for dicom {dicom_path} is higher than or equal to the number of frames in the video ({dicom.shape[0]}).")
        return False
    
    try:
        imager_spacing = float(dicom_info['ImagerPixelSpacing'][0])
        factor = float(dicom_info['DistanceSourceToDetector'].value) / float(dicom_info['DistanceSourceToPatient'].value)
        pixel_spacing = float(imager_spacing / factor)
        if imager_spacing <= 0 or factor <= 0:
            raise ValueError(f'Pixel spacing information invalid for dicom {dicom_path}.')
    except (KeyError, ValueError):
        print(f'Pixel spacing information missing or invalid for dicom {dicom_path}.')
        return False
    
    try:
        fps = int(dicom_info["RecommendedDisplayFrameRate"].value)
        if fps != 15:
            print(f"Frame rate not equal to 15 FPS ({fps} FPS) for dicom {dicom_path}.")
            return False
    except KeyError:
        print(f'Frame rate information missing {dicom_path}.')
        return False
    
    try:
        StudyDate = str(dicom_info["StudyDate"].value)
    except KeyError as e:
        print(f"Study date information missing. :: Error {str(e)}")
    
    try:
        DOB = str(dicom_info[(0x0010, 0x0030)].value)
    except KeyError:
        print("Patient age information missing.")
        return False
    
    object_pred = sub_df['artery_view'].iloc[0]
    if object_pred not in ["RCA", "LCA"]:
        print("DICOM does not display RCA or LCA.")
        return False
    
    return True

###################
### Algorithm 3 ###
###################

### Refactored function for classes @otastet
def resize_coordinates_refac(dict_coordinates, pixel_spacing, img_shape, target_size):
    
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
            'x1' : x1n,
            'y1' : y1n,
            'x2' : x2n,
            'y2' : y2n
        }

def register_refac(img, index, box_coordinates):
    img_after = img[index:]
    img_before = np.flip(
        img[: index + 1], 0
    )  
    plx = []
    ply = []
    k = 0
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









def resize_coordinates(x1, y1, x2, y2, pixel_spacing, img_shape, target_size):
    xc = np.floor((x2 + x1) / 2)
    yc = np.floor((y2 + y1) / 2)

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
    return x1n, y1n, x2n, y2n

def register(img, index, x1, y1, x2, y2):
    img_after = img[index:]
    img_before = np.flip(
        img[: index + 1], 0
    )  
    plx = []
    ply = []
    k = 0
    tracker = cv2.TrackerCSRT.create() 

    tracker.init(img_before[0], [x1, y1, x2 - x1, y2 - y1])
    for i in range(img_before.shape[0]):
        moving = img_before[i]
        _, bbox = tracker.update(moving)
        (x, y, w, h) = (int(v) for v in bbox)

        yoff = int(y + h / 2 - (y2 - y1) / 2) - y1
        xoff = int(x + w / 2 - (x2 - x1) / 2) - x1

        plx.append(xoff)
        ply.append(yoff)

    plx.reverse()
    ply.reverse()

    plx.pop()
    ply.pop()

    tracker.init(img_after[0], [x1, y1, x2 - x1, y2 - y1])
    for i in range(img_after.shape[0]):
        moving = img_after[i]
        _, bbox = tracker.update(moving)
        (x, y, w, h) = (int(v) for v in bbox)

        yoff = int(y + h / 2 - (y2 - y1) / 2) - y1
        xoff = int(x + w / 2 - (x2 - x1) / 2) - x1

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

def choose_model(model_name):
    if model_name == 'Unet':
        return segmentation_models_pytorch.Unet(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == 'UnetPlusPlus':
        return segmentation_models_pytorch.UnetPlusPlus(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == 'MAnet':
        return segmentation_models_pytorch.MAnet(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == 'Linknet':
        return segmentation_models_pytorch.Linknet(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == 'FPN':
        return segmentation_models_pytorch.FPN(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == 'PSPNet':
        return segmentation_models_pytorch.PSPNet(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == 'DeepLabV3':
        return segmentation_models_pytorch.DeepLabV3(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == 'DeepLabV3Plus':
        return segmentation_models_pytorch.DeepLabV3Plus(encoder_weights="imagenet", in_channels=3, classes=26)
    elif model_name == 'PAN':
        return segmentation_models_pytorch.PAN(encoder_weights="imagenet", in_channels=3, classes=26)
    else:
        raise ValueError(f"Invalid model name: {model_name}. Please choose a valid model.")

def load_model(checkpoint_path, device):
    split_parts = checkpoint_path.split('_')
    model_index = split_parts.index("model")
    model_name = split_parts[model_index+1]
    model = choose_model(model_name)

    checkpoint = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model


def histogram_equalization_batch(image_batch):
    # Assuming image_batch is a tensor of shape [batch_size, height, width]
    # Since we're working with batches, we need to individually process each image
    equalized_images = []
    for image in image_batch:
        image_pil = torchvision.transforms.ToPILImage()(image)        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(np.array(image_pil))
        equalized_images.append(torch.from_numpy(cl1).unsqueeze(0))
        
    # Stack the list of tensors along a new dimension
    return torch.cat(equalized_images, dim=0)

def process_batch(images_batch):
    images_batch_tensor = torch.from_numpy(images_batch).float()
    # Assuming images_batch is a tensor of shape [batch_size, height, width], and dtype float32
    images_batch_tensor = images_batch_tensor.to(torch.uint8) # Convert to uint8 if not already
    images_batch_tensor = histogram_equalization_batch(images_batch_tensor)
    # Normalize each image in the batch individually
    images_batch_float = images_batch_tensor.float() # Convert back to float for normalization
    min_vals = images_batch_float.amin(dim=(1, 2), keepdim=True)
    max_vals = images_batch_float.amax(dim=(1, 2), keepdim=True)
    normalized_images_batch = (images_batch_float - min_vals) / (max_vals - min_vals)
    normalized_images_batch = normalized_images_batch.unsqueeze(1).repeat(1, 3, 1, 1)
    return normalized_images_batch

def histogram_equalization(image):
    image_pil = torchvision.transforms.ToPILImage()(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(np.asarray(image_pil))
    cl1_copy = cl1.copy()
    return torch.from_numpy(cl1_copy).unsqueeze(0)

def process_image(img):
    assert(len(img.shape) == 2)
    image = torch.tensor(img).float().unsqueeze(0)
    image = image.to(torch.uint8)
    image = histogram_equalization(image)
    image = (image - image.min()) / (image.max() - image.min())
    return image

def perform_segmentation_inference_batch(models, batch, device):
    
    batch = batch.to(device) 
    outputs = []
    with torch.no_grad():
        for model in models:
            outputs.append(model(batch))  # Keep on device

    # Combine outputs on GPU and only then transfer to CPU
    combined_outputs = torch.stack(outputs).mean(dim=0).argmax(1).to('cpu').numpy()
    return combined_outputs


def perform_segmentation_inference(models, img, device):
    image = process_image(img).repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        outputs = [model(image) for model in models]
    combined_output = torch.stack(outputs).mean(dim=0).argmax(1).squeeze().cpu().numpy()
    return combined_output

###################
### Algorithm 5 ###
###################

retinanet_artery_labels = { # https://syntaxscore.org/index.php/tutorial/definitions
    1: 'prox_rca', # 1: RCA proximal
    2: 'mid_rca', # 2: RCA mid 
    3: 'dist_rca', # 3: RCA distal
    4: 'pda', # 4: Posterior descending
    5: 'leftmain', # 5: Left main 
    6: 'lad', # 6: LAD proximal 
    7: 'mid_lad', # 7: LAD mid 
    8: 'dist_lad', # 8: LAD apical
    9: 'other', # 9: First diagonal 
    10: 'other', # 9a: First diagonal a
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

which_artery = {"RCA": [1,2,3,4,20,21,22,23],
                "LCA": [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,24,25]}

def get_segment_region(region, object_value):
    unique, counts = np.unique(region, return_counts=True)
    d = dict(zip(unique, counts))
    d1 = dict(sorted(d.items(), key=lambda item: item[1], reverse = True))
    if 0 in d1.keys():
        del d1[0]
    segment = 'None'
    for value in d1.keys():
        if (value in which_artery[object_value]):
            segment = retinanet_artery_labels[value]
            break
    return segment

def get_segment_center(region, object_value):
    center = [int(region.shape[0] / 2), int(region.shape[0] / 2)]
    min_L = min(center)
    d1 = {"None": "region2"}
    for l in range(min_L):
        if l != 0:
            subregion = region[center[0] - l: center[0] + l, center[1] - l: center[1] + l]
        else:
            subregion = region[center[0]: center[0] + 1, center[1]: center[1] + 1]
        if subregion.sum() == 0:
            segment = 'None'
        else:
            segment = get_segment_region(subregion, object_value)
        if not(segment in ['None']):
            break
    return segment

###################
### Algorithm 6 ###
###################

### Refac
def create_cropped_registered_video_refac(dicom, frame, reg_shifts, box_coordinates):
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
    cropped_img_reg = torchvision.transforms.Resize((RESIZE, RESIZE),antialias=None)(cropped_img_reg)
    cropped_img_reg = torchvision.transforms.Normalize(mean, std)(cropped_img_reg).numpy()
    
    pad_value = int(FRAMES / 2)
    zero_frame = np.zeros(cropped_img_reg[0:1].shape)
    padded_video = np.concatenate([zero_frame] * pad_value + [cropped_img_reg] + [zero_frame] * pad_value, axis=0)
    pad_frame = frame + pad_value
    final_video = padded_video[pad_frame - pad_value: pad_frame + pad_value]
    final_video = np.stack((final_video,) * 1, 0).transpose((0, 2, 1, 3, 4))
    
    return torch.from_numpy(final_video)







def create_cropped_registered_video(dicom, frame, reg_shifts, x1, y1, x2, y2):
    FRAMES = 24
    RESIZE = 224
    mean = [144.80116, 144.80116, 144.80116] 
    std = [52.669548, 52.669548, 52.669548]

    cropped_img_reg = np.zeros(dicom[:, y1:y2, x1:x2].shape)
    for j in range(dicom.shape[0]):
        reg_shift = reg_shifts[j]
        img_reg = scipy.ndimage.shift(dicom[j], shift=(reg_shift[0], reg_shift[1]), order=5)
        cropped_img_reg[j] = img_reg[y1:y2, x1:x2]
        
    cropped_img_reg = np.stack((cropped_img_reg,) * 3, 1)
    cropped_img_reg = torch.from_numpy(cropped_img_reg)
    cropped_img_reg = torchvision.transforms.Resize((RESIZE, RESIZE),antialias=None)(cropped_img_reg)
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
    def forward(self, x):
        return x

class Regressor(nn.Module):
    def __init__(self, num_class=1, num_features=12):
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

    def forward(self, x):
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

def get_model():
    model = Regressor()
    return model

def get_age(dicom_info):
    StudyDate = str(dicom_info["StudyDate"].value)
    DOB = str(dicom_info[(0x0010, 0x0030)].value)
    
    [ySD, mSD, dSD] = [int(StudyDate[0:4]), int(StudyDate[4:6]), int(StudyDate[6:8])]
    [yDOB, mDOB, dDOB] = [int(DOB[0:4]), int(DOB[4:6]), int(DOB[6:8])]
    d1 = date(yDOB,mDOB,dDOB)
    d2 = date(ySD,mSD,dSD)
    age = relativedelta(d2, d1).years
    
    return age


def object_recon(dicom, config, device):
    
    model = pytorchvideo_model(config["model_name"], config["num_classes"], config["task"])
    checkpoint = torch.load(config["model_path"])
    model_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    model.load_state_dict(model_state_dict)
    model.to(device)
    
    output = model(dicom)
    return output
    

