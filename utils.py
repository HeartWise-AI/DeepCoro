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


###################
### Algorithm 3 ###
###################

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

def histogram_equalization(image):
    image_pil = torchvision.transforms.ToPILImage()(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(np.asarray(image_pil))
    return torch.from_numpy(cl1).unsqueeze(0)

def process_image(img):
    assert(len(img.shape) == 2)
    image = torch.tensor(img).float().unsqueeze(0)
    image = image.to(torch.uint8)
    image = histogram_equalization(image)
    image = (image - image.min()) / (image.max() - image.min())
    return image

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
    cropped_img_reg = torchvision.transforms.Resize((RESIZE, RESIZE))(cropped_img_reg)
    cropped_img_reg = torchvision.transforms.Normalize(mean, std)(cropped_img_reg).numpy()
    
    pad_value = int(FRAMES / 2)
    zero_frame = np.zeros(cropped_img_reg[0:1].shape)
    padded_video = np.concatenate([zero_frame] * pad_value + [cropped_img_reg] + [zero_frame] * pad_value, axis=0)
    pad_frame = frame + pad_value
    final_video = padded_video[pad_frame - pad_value: pad_frame + pad_value]
    final_video = np.stack((final_video,) * 1, 0).transpose((0, 2, 1, 3, 4))
    
    return torch.from_numpy(final_video)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class Regressor(nn.Module):
    def __init__(self, num_class = 1, num_features = 12):
        super(Regressor, self).__init__()
        
        
        model = torchvision.models.video.swin3d_b(weights='KINETICS400_IMAGENET22K_V1')
        n_inputs = model.head.in_features
        model.head = nn.Linear(n_inputs, num_class)
        #model = nn.Sequential(model, nn.Sigmoid())

        #model = model[0]
        n_inputs = model.head.in_features
        model.head = Identity()
        
        self.model = model
        self.fc = nn.Linear(n_inputs + num_features, num_class)
        self.layer = nn.Sigmoid()

    def forward(self, x):
        features = self.model(x[0])
        
        age = x[1]/120
        
        segment = torch.nn.functional.one_hot(x[2].squeeze(), num_classes=11).to(torch.float32)
        if len(segment.size()) == 1:
            segment = segment.unsqueeze(0)
        x = torch.cat([features, segment, age], 1)

        x = self.fc(x)
        x = self.layer(x)
        return x

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

