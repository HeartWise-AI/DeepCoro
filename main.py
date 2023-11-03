import argparse
import pydicom
import numpy as np
import torch
import pandas as pd
import tensorflow as tf
import keras_retinanet.models
from keras_retinanet.utils.image import preprocess_image, resize_image
import cv2
import utils
from tqdm import tqdm
import imageio
import scipy.ndimage
import time
from collections import Counter

def deepcoro(dicom_path, artery_view, save_dir, models_dir, device):
    dicom_info = pydicom.dcmread(dicom_path)
    dicom = dicom_info.pixel_array
    since = time.time()
    
    try:
        primary_angle = float(dicom_info['PositionerPrimaryAngle'].value)
    except:
        assert(1 == 0), "Positioner Primary Angle information missing."
        
    try:
        secondary_angle = float(dicom_info['PositionerSecondaryAngle'].value)
    except:
        assert(1 == 0), "Positioner Secondary Angle information missing."

    try:
        imager_spacing = float(dicom_info['ImagerPixelSpacing'][0])
        factor = float(dicom_info['DistanceSourceToDetector'].value) / float(
            dicom_info['DistanceSourceToPatient'].value)
        pixel_spacing = float(imager_spacing / factor)
        assert (imager_spacing > 0)
        assert (factor > 0)
    except:
        assert(1 == 0), 'Pixel spacing information missing.'

    try:
        fps = int(dicom_info["RecommendedDisplayFrameRate"].value)
    except:
        assert(1 == 0), 'Frame rate information missing.'

    assert(fps == 15), "Frame rate not equal to 15 FPS (" + str(fps) + " FPS)."

    try:
        StudyDate = str(dicom_info["StudyDate"].value)
    except:
        assert(1 == 0), "Study date information missing."

    try:
        DOB = str(dicom_info[(0x0010, 0x0030)].value)
    except:
        assert(1 == 0), "Patient age information missing."
        
        
    if len(models_dir) > 0:
        if models_dir[-1] != "/":
            models_dir = models_dir + "/"
        
    ###################
    ### Algorithm 1 ###
    ###################
        
    print("\nAlgorithm 1 / 6 starts")

    since1 = time.time()

    object_pred = artery_view
    assert(object_pred in ["RCA","LCA"]), "DICOM does not display RCA or LCA."

    print("Algorithm 1 / 6 finished in", round(time.time() - since1, 2), 's')
    
    ###################
    ### Algorithm 2 ###
    ###################
    
    print("\nAlgorithm 2 / 6 starts")

    since2 = time.time()
    
    if 'cuda' in device:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    elif "cpu" in device:
        tf.config.set_visible_devices([], 'GPU')

    primary_angle = float(dicom_info['PositionerPrimaryAngle'].value)
    secondary_angle = float(dicom_info['PositionerSecondaryAngle'].value)
    
    primary_angle = (primary_angle + 180) % 360 - 180
    secondary_angle = (secondary_angle + 180) % 360 - 180
    if (-45 <= primary_angle <= -15) and (15 <= secondary_angle <= 45):
        angle_pred = "RAO Cranial"
    elif (15 <= primary_angle <= 45) and (-15 <= secondary_angle <= 15):
        angle_pred = "LAO Straight"
    else:
        angle_pred = "Other"

    if (object_pred == "RCA") & (angle_pred == "LAO Straight"):
        angle_model = keras_retinanet.models.load_model(models_dir + 'algorithm2b/RCA_LAO_straight.h5', backbone_name="resnet50")
        print("RetinaNet model of the", angle_pred, "projection angle from the", object_pred, "loaded.")
    elif (object_pred == "LCA") & (angle_pred == "RAO Cranial"):
        angle_model = keras_retinanet.models.load_model(models_dir + 'algorithm2b/LAD_RAO_cranial.h5', backbone_name="resnet50")
        print("RetinaNet model of the", angle_pred, "projection angle from the", object_pred, "loaded.")
    else:
        angle_model = keras_retinanet.models.load_model(models_dir + 'algorithm2b/general.h5', backbone_name="resnet50")
        print("General RetinaNet model loaded.")
    
    c = 0
    stenoses = {}
    for i in tqdm(range(dicom.shape[0])):
        image = cv2.cvtColor(dicom[i].astype(np.uint8), cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)

        boxes, scores, labels = angle_model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale

        idx_score = set(np.where(scores[0] >= 0.5)[0])
        idx_labels = set(np.where(labels[0] == 15.0)[0])
        idx = list(idx_score.intersection(idx_labels))

        for j in idx:
            stenoses[c] = {'frame': i, 'box': tuple(boxes[0, j])}
            c += 1
    
    del angle_model

    print("Algorithm 2 / 6 finished in", round(time.time() - since2, 2), 's')
    
    ###################
    ### Algorithm 3 ###
    ###################
    
    print("\nAlgorithm 3 / 6 starts")

    since3 = time.time()
    imager_spacing = float(dicom_info['ImagerPixelSpacing'][0])
    factor = float(dicom_info['DistanceSourceToDetector'].value) / float(
        dicom_info['DistanceSourceToPatient'].value)
    pixel_spacing = float(imager_spacing / factor)

    for i in tqdm(stenoses.keys()):
        frame = stenoses[i]['frame']
        (x1, y1, x2, y2) = stenoses[i]['box']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1n, y1n, x2n, y2n = utils.resize_coordinates(x1, y1, x2, y2, pixel_spacing, dicom.shape, 17.5)
        stenoses[i]['box_resized'] = (x1n, y1n, x2n, y2n) 
        stenoses[i]['reg_shift'] = utils.register(dicom, frame, x1n, y1n, x2n, y2n)

    print("Algorithm 3 / 6 finished in", round(time.time() - since3, 2), 's')
    
    ###################
    ### Algorithm 4 ###
    ###################
    
    print("\nAlgorithm 4 / 6 starts")

    since4 = time.time()
    model_paths = [
        models_dir + "algorithm4/bs_64_lr_0.00107809_model_FPN_loss_LovaszLoss.pth",
        models_dir + "algorithm4/bs_64_lr_0.00242160_model_DeepLabV3Plus_loss_LovaszLoss.pth",
        models_dir + "algorithm4/bs_64_lr_0.00129894_model_PAN_loss_TverskyLoss.pth",
        models_dir + "algorithm4/bs_4_lr_0.00059902_model_DeepLabV3_loss_TverskyLoss.pth",
        models_dir + "algorithm4/bs_16_lr_0.00993245_model_FPN_loss_LovaszLoss.pth",
        models_dir + "algorithm4/bs_16_lr_0.00232125_model_DeepLabV3_loss_LovaszLoss.pth",
        models_dir + "algorithm4/bs_64_lr_0.00646546_model_PAN_loss_DiceLoss.pth"
    ]

    models = [utils.load_model(path, device) for path in model_paths]

    output = np.zeros(dicom.shape)
    for i in tqdm(range(dicom.shape[0])):
        output[i] = utils.perform_segmentation_inference(models, dicom[i], device)

    del models
    if 'cuda' in device:
        torch.cuda.empty_cache()

    print("Algorithm 4 / 6 finished in", round(time.time() - since4, 2), 's')
    
    ###################
    ### Algorithm 5 ###
    ###################
    
    print("\nAlgorithm 5 / 6 starts")

    since5 = time.time()
    
    segments_of_interest = {}
    segments_of_interest["rca"] = ["prox_rca","mid_rca","dist_rca","pda","posterolateral"]
    segments_of_interest["lca"] = ["leftmain","lad","mid_lad","dist_lad","lcx","dist_lcx"]
    
    for i in tqdm(range(len(stenoses))):
        frame = stenoses[i]['frame']
        (x1, y1, x2, y2) = stenoses[i]['box_resized']

        preds = []
        pad_value = max([(x2 - x1), (y2 - y1)])
        for j in range(output.shape[0]):
            reg_shift = stenoses[i]['reg_shift'][j]
            padded_output = np.pad(output[j], ((pad_value, pad_value), (pad_value, pad_value)), mode='constant', constant_values=0)

            y_shift = -reg_shift[0] + pad_value
            x_shift = -reg_shift[1] + pad_value
            region = padded_output[int(y1 + y_shift): int(y2 + y_shift), int(x1 + x_shift): int(x2 + x_shift)].astype('uint8')

            segment = utils.get_segment_center(region, object_pred)
            preds.append(segment)

        cleaned_preds = [i for i in preds if not(i in ['None', 'other'])]
        if len(cleaned_preds) == 0:
            pred_segment = 'None'
        else:
            counts = Counter(cleaned_preds)
            pred_segment = max(counts, key=lambda x: (counts[x], -(segments_of_interest["rca"] + segments_of_interest["lca"]).index(x)))

        stenoses[i]['artery_segment'] = pred_segment

    def get_middle(group):
        return group.iloc[int(len(group) / 2)]

    df_stenoses = pd.DataFrame(stenoses).T
    df_stenoses = df_stenoses.loc[df_stenoses['artery_segment'] != 'None'].reset_index(drop=True)
    df_stenoses = df_stenoses.groupby(['artery_segment']).apply(get_middle).reset_index(drop=True)

    print("Algorithm 5 / 6 finished in", round(time.time() - since5, 2), 's')
    
    ###################
    ### Algorithm 6 ###
    ###################
    
    print("\nAlgorithm 6 / 6 starts")

    since6 = time.time()
    artery_labels = {'leftmain': 0, 'lad': 1, 'lcx': 2, 'mid_lad': 3, 'dist_lcx': 4, 'dist_lad': 5, 'prox_rca': 6, 'mid_rca': 7, 'dist_rca': 8, 'pda': 9, 'posterolateral': 10}
    THRESH = 0.23

    model = utils.get_model()
    model = model.to(device).double()
    checkpoint = torch.load(models_dir + 'algorithm6.pt', map_location=device)["state_dict"]
    if device == 'cpu':
        checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
    else:
        model = torch.nn.DataParallel(model)
    model.eval()
    model.load_state_dict(checkpoint)

    patient_age = utils.get_age(dicom_info)

    df_stenoses['percent_stenosis'] = None
    df_stenoses['severe_stenosis'] = None
    for i in tqdm(range(len(df_stenoses))):
        (x1, y1, x2, y2) = df_stenoses['box_resized'].iloc[i]
        reg_shifts = df_stenoses['reg_shift'].iloc[i]
        frame = df_stenoses['frame'].iloc[i]
        video = utils.create_cropped_registered_video(dicom, frame, reg_shifts, x1, y1, x2, y2)
        age = torch.tensor([[patient_age]])
        segment = torch.tensor([[artery_labels[df_stenoses['artery_segment'].iloc[i]]]])
        outputs = model([video.to(device), age.to(device), segment.to(device)])

        df_stenoses.loc[i, 'percent_stenosis'] = outputs.item() * 100
        df_stenoses.loc[i, 'severe_stenosis'] = (outputs.item() >= THRESH)

    print("Algorithm 6 / 6 finished in", round(time.time() - since6, 2), 's')
    
    ####################
    ### Save outputs ###
    ####################

    print("\nSaving output...")
    if len(save_dir) > 0:
        if save_dir[-1] != "/":
            save_dir = save_dir + "/"

    df_stenoses['video_path'] = None
    font = cv2.FONT_HERSHEY_PLAIN
    for i in tqdm(range(len(df_stenoses))):
        if df_stenoses['severe_stenosis'].iloc[i]:
            save_path = 'severe_stenosis.mp4'
        else:
            save_path = 'nonsevere_stenosis.mp4'
        save_path = save_dir + df_stenoses['artery_segment'].iloc[i] + "_" + str(int(df_stenoses['percent_stenosis'].iloc[i])) + "_" + save_path
        df_stenoses.loc[i, 'video_path'] = save_path

        (x1, y1, x2, y2) = df_stenoses['box_resized'].iloc[i]
        img_reg = np.zeros(dicom.shape)
        for j in range(dicom.shape[0]):
            reg_shift = df_stenoses['reg_shift'].iloc[i][j]
            img_reg[j] = scipy.ndimage.shift(dicom[j], shift=(reg_shift[0], reg_shift[1]), order=5)
            cv2.rectangle(img_reg[j], (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(img_reg[j], df_stenoses['artery_segment'].iloc[i] + " - " + str(int(df_stenoses['percent_stenosis'].iloc[i])) + '%', org=(x1, y1), fontFace=font, fontScale=1,color=(255, 255, 255))
        imageio.mimwrite(save_path, img_reg.astype(np.uint8), fps=15)

    df_stenoses[['video_path', 'frame', 'box_resized', 'artery_segment', 'percent_stenosis', 'severe_stenosis']].to_csv(save_dir + "df_stenosis.csv")

    total_time = int(time.time() - since)
    print("Total running time:", int(total_time // 60), 'min', int(total_time % 60), "s")


def main(args = None):
    parser = argparse.ArgumentParser(description = 'DeepCoro')
    
    parser.add_argument('--dicom_path')
    parser.add_argument('--save_dir')
    parser.add_argument('--artery_view')
    parser.add_argument('--models_dir', default = 'models/')
    parser.add_argument('--device', default = 'cuda')
    parser = parser.parse_args(args)
    
    dicom_path = parser.dicom_path
    artery_view = parser.artery_view.upper()
    save_dir = parser.save_dir
    models_dir = parser.models_dir
    device = parser.device
 
    deepcoro(dicom_path, artery_view, save_dir, models_dir, device)


if __name__ == '__main__':
    main()
