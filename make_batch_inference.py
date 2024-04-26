import argparse
import pydicom
import numpy as np
import torch
import pandas as pd
import cv2
import utils
from tqdm import tqdm
import imageio
import scipy.ndimage
import time
from collections import Counter
import json 

## Get params 
with open('params.json', 'r') as file:
    params = json.load(file)

### Reads inputs from the provided csv files. 
### Algorithms 1 and 2 are mimicked by extracting the previously computed results from the csv 
def read_videos(input_file_path):
    df = pd.read_csv(input_file_path, dtype={
        'artery_view': str,
        'frame': 'int64',
        'x1': 'float64',
        'y1': 'float64',
        'x2': 'float64',
        'y2': 'float64'
    })
    prefix = input_file_path[:input_file_path.find('input_file.csv')]
    
    df['artery_view'] = df['artery_view'].str.upper()
    
    dicom_paths = df['dicom_path'].unique().tolist()
    
    dicom_data = {}
    
    ## For each input row in the csv file, extract the artery view (Algo 1) and the stenosis location (Algo 2)
    for dicom_id, dicom_path in enumerate(dicom_paths):
        sub_df = df[df['dicom_path'] == dicom_path].copy().reset_index(drop=True)
        
        ## Checks if more than 1 artery view is detected in the same dicom (which would be unusual)
        if len(sub_df['artery_view'].unique()) != 1:
            print(f"More than one artery view {sub_df['artery_view'].unique()} is associated with the same DICOM {dicom_path}")
            continue
        
        dicom_info = pydicom.dcmread(prefix + dicom_path)
        dicom = dicom_info.pixel_array
        
        ## Checks if the frames referenced in the csv 
        if sub_df['frame'].max() >= dicom.shape[0]:
            print(f"Maximum frame number ({sub_df['frame'].max()}) for dicom {dicom_path} is higher than or equal to the number of frames in the video ({dicom.shape[0]}).")
            continue
        
        ## This block of trys attempts to extract important values from the dicom and will raise specific errors if some are unavailable 
        try:
            imager_spacing = float(dicom_info['ImagerPixelSpacing'][0])
            factor = float(dicom_info['DistanceSourceToDetector'].value) / float(dicom_info['DistanceSourceToPatient'].value)
            pixel_spacing = float(imager_spacing / factor)
            if imager_spacing <= 0 or factor <= 0:
                raise ValueError(f'Pixel spacing information invalid for dicom {dicom_path}.')
        except (KeyError, ValueError):
            print(f'Pixel spacing information missing or invalid for dicom {dicom_path}.')
            continue
        
        try:
            fps = int(dicom_info["RecommendedDisplayFrameRate"].value)
            if fps != 15:
                print(f"Frame rate not equal to 15 FPS ({fps} FPS) for dicom {dicom_path}.")
                continue
        except KeyError:
            print(f'Frame rate information missing {dicom_path}.')
            continue
        
        try:
            StudyDate = str(dicom_info["StudyDate"].value)
        except KeyError as e:
            print(f"Study date information missing. :: Error {str(e)}")
        
        try:
            DOB = str(dicom_info[(0x0010, 0x0030)].value)
        except KeyError:
            print("Patient age information missing.")
            continue
        
        ###################
        ### Algorithm 1 ###
        ###################
        
        ## Extract artery view 
        object_pred = sub_df['artery_view'].iloc[0]
        if object_pred not in ["RCA", "LCA"]:
            print("DICOM does not display RCA or LCA.")
            continue
        
        ###################
        ### Algorithm 2 ###
        ###################
        
        ## Extract stenosis boxes 
        stenoses = sub_df[['frame', 'x1', 'y1', 'x2', 'y2']].to_dict('records')
        
        for stenosis in stenoses:
            frame = stenosis['frame']
            x1, y1, x2, y2 = stenosis['x1'], stenosis['y1'], stenosis['x2'], stenosis['y2']
            x1n, y1n, x2n, y2n = utils.resize_coordinates(x1, y1, x2, y2, pixel_spacing, dicom.shape, 17.5)
            stenosis['box_resized'] = (x1n, y1n, x2n, y2n)
            stenosis['reg_shift'] = utils.register(dicom, frame, x1n, y1n, x2n, y2n)
        
        ## Create dictionary with extracted features 
        dicom_data[dicom_id] = {
            'dicom': dicom,
            'dicom_info': dicom_info,
            'stenoses': stenoses,
            'object_pred': object_pred,
            'dicom_path': dicom_path
        }
    
    return dicom_data

### Apply segmentation models to dicom videos 
### The frames of a video are batched for batch inference  
def segment_artery_subclass(dicom_data, models_dir, device):

    models = [utils.load_model(path, device) for path in params["segmentation_models_weights"]]
    all_outputs = []
    
    ### Each dicom in dicom_data is a video 
    for value in tqdm(dicom_data.values()):
        output = np.zeros(value['dicom'].shape)
        output = utils.perform_segmentation_inference_batch(models,value['dicom'],device)
        all_outputs.append(output)
    
    ### Delete models from gpu to optimize memory usage 
    del models
    if 'cuda' in device:
        torch.cuda.empty_cache()
    return all_outputs

## Assignment of detected stenosis to segmented artery sub classes 
def assign_stenosis_to_segment(data_dict, output, segments_of_interest):
    df_dict = {}
    for key, value in data_dict.items():
        df_stenoses = pd.DataFrame(value['stenoses'])
        df_stenoses['preds'] = df_stenoses.apply(lambda row: get_preds(row, output[key], value['object_pred'], segments_of_interest), axis=1)
        df_stenoses['artery_segment'] = df_stenoses['preds'].apply(lambda preds: get_pred_segment(preds, segments_of_interest))
        df_stenoses = df_stenoses.loc[df_stenoses['artery_segment'] != 'None'].reset_index(drop=True)
        df_stenoses = df_stenoses.groupby(['artery_segment']).apply(lambda group: group.iloc[len(group) // 2]).reset_index(drop=True)
        df_dict[key] = df_stenoses[['frame', 'box_resized', 'reg_shift', 'artery_segment']]
    return df_dict

def get_preds(row, output, object_pred, segments_of_interest):
    frame = int(row['frame'])
    (x1, y1, x2, y2) = row['box_resized']
    preds = []
    pad_value = max([(x2 - x1), (y2 - y1)])
    for j in range(output.shape[0]):
        reg_shift = row['reg_shift'][j]
        padded_output = np.pad(output[j], ((pad_value, pad_value), (pad_value, pad_value)), mode='constant', constant_values=0)
        y_shift = -reg_shift[0] + pad_value
        x_shift = -reg_shift[1] + pad_value
        region = padded_output[int(y1 + y_shift): int(y2 + y_shift), int(x1 + x_shift): int(x2 + x_shift)].astype('uint8')
        segment = utils.get_segment_center(region, object_pred)
        preds.append(segment)
    return preds

def get_pred_segment(preds, segments_of_interest):
    cleaned_preds = [i for i in preds if i not in ['None', 'other']]
    if len(cleaned_preds) == 0:
        return 'None'
    else:
        counts = Counter(cleaned_preds)
        return max(counts, key=lambda x: (counts[x], -(segments_of_interest["rca"] + segments_of_interest["lca"]).index(x)))

## This function uses a swin3d model to predict the % of stenosis (indicating the severtiy of the stenosis)
def predict_percentage_stenosis(data_dict, device, df_dict,models_dir):
    
    model = utils.get_model()
    model = model.to(device).double()
    
    checkpoint = torch.load(params["swin3d_model_weights"], map_location=device)["state_dict"]
    if device == 'cpu':
        checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
    else:
        model = torch.nn.DataParallel(model)
    model.eval()
    model.load_state_dict(checkpoint)

    for key, value in data_dict.items():
        patient_age = utils.get_age(value['dicom_info'])

        df_dict[key]['percent_stenosis'] = None
        df_dict[key]['severe_stenosis'] = None
        
        for i in range(len(df_dict[key])):
            (x1, y1, x2, y2) = df_dict[key]['box_resized'].iloc[i]
            reg_shifts = df_dict[key]['reg_shift'].iloc[i]
            frame = df_dict[key]['frame'].iloc[i]
            video = utils.create_cropped_registered_video(value['dicom'], frame, reg_shifts, x1, y1, x2, y2)
            age = torch.tensor([[patient_age]])
            segment = torch.tensor([[params["artery_labels"][df_dict[key]['artery_segment'].iloc[i]]]])
            outputs = model([video.to(device), age.to(device), segment.to(device)])
            df_dict[key].loc[i, 'percent_stenosis'] = outputs.item() * 100
            df_dict[key].loc[i, 'severe_stenosis'] = (outputs.item() >= params["percentage_threshold"])
            
    return df_dict


### This function saves results in a mp4 video and data frame describing the predictions
def save_results(data_dict,df_dict,save_dir):
    df_stenoses_cat = pd.DataFrame({'dicom_path':[],'video_path':[], 'frame':[], 'box_resized':[], 'artery_segment':[], 'percent_stenosis':[], 'severe_stenosis':[]})
    
    for key, value in data_dict.items():
        df_dict[key]['video_path'] = None
        df_dict[key]['dicom_path'] = value['dicom_path']
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(df_dict[key])):
            if df_dict[key]['severe_stenosis'].iloc[i]:
                save_path = 'severe_stenosis.mp4'
            else:
                save_path = 'nonsevere_stenosis.mp4'
            save_path = save_dir + "DICOM" + str(key) + "_" + df_dict[key]['artery_segment'].iloc[i] + "_" + str(int(df_dict[key]['percent_stenosis'].iloc[i])) + "_" + save_path
            df_dict[key].loc[i, 'video_path'] = save_path

            (x1, y1, x2, y2) = df_dict[key]['box_resized'].iloc[i]
            img_reg = np.zeros(value['dicom'].shape)
            for j in range(value['dicom'].shape[0]):
                reg_shift = df_dict[key]['reg_shift'].iloc[i][j]
                img_reg[j] = scipy.ndimage.shift(value['dicom'][j], shift=(reg_shift[0], reg_shift[1]), order=5)
                cv2.rectangle(img_reg[j], (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(img_reg[j], df_dict[key]['artery_segment'].iloc[i] + " - " + str(int(df_dict[key]['percent_stenosis'].iloc[i])) + '%', org=(x1, y1), fontFace=font, fontScale=1,color=(255, 255, 255))
            imageio.mimwrite(save_path, img_reg.astype(np.uint8), fps=15)

        df_dict[key] = df_dict[key][['dicom_path', 'video_path', 'frame', 'box_resized', 'artery_segment', 'percent_stenosis', 'severe_stenosis']]

        df_stenoses_cat = pd.concat([df_stenoses_cat, df_dict[key]])
    df_stenoses_cat.to_csv(save_dir + "df_stenosis.csv")


def main(args = None):
    parser = argparse.ArgumentParser(description = 'DeepCoro')
    
    parser.add_argument('--input_file_path', default='/dcm_input/input_file.csv')
    parser.add_argument('--models_dir', default = '/opt/deepcoro/models/')
    parser.add_argument('--save_dir', default='/results/batch_inference/')
    
    parser = parser.parse_args(args)
    
    input_file_path = parser.input_file_path
    models_dir = parser.models_dir
    save_dir = parser.save_dir
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    total_start_time = time.time()
    
    start_time = time.time()
    data_dict = read_videos(input_file_path)
    end_time = time.time()
    print(f'Elapse Time (read_video): {end_time-start_time}')
    
    start_time = time.time()
    segmentation_output = segment_artery_subclass(data_dict,models_dir,device)
    end_time = time.time()
    print(f'Elapse Time (segment_artery_subclass): {end_time-start_time}')
    
    print(len(segmentation_output))
    
    start_time = time.time()
    stenosis_assignment_dict = assign_stenosis_to_segment(data_dict, segmentation_output, params["segments_of_interest"])
    end_time = time.time()
    print(f'Elapse Time (assign_stenosis_to_segment): {end_time-start_time}')
    
    start_time = time.time()
    percentage_stenosis_results = predict_percentage_stenosis(data_dict, device, stenosis_assignment_dict, models_dir)
    end_time = time.time()
    print(f'Elapse Time (predict_percentage_stenosis): {end_time-start_time}')
    
    start_time = time.time()
    save_results(data_dict,percentage_stenosis_results,save_dir)
    end_time = time.time()
    print(f'Elapse Time (save_results): {end_time-start_time}')
    
    total_end_time = time.time()
    print(f"Elapsed time: {total_end_time-total_start_time}")

if __name__ == '__main__':
    main()