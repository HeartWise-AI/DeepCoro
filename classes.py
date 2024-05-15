import utils_refac
import numpy as np
from collections import Counter
import torch
from tqdm import tqdm
import os
import pandas as pd
import cv2
import scipy
import imageio
import json
from typing import List
import pydicom
import torchvision
torchvision.disable_beta_transforms_warning()

import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageColor
from torchvision.transforms import v2

class StenosisDataset:
    """
    Represents a dataset of stenosis in DICOM exams.

    Args:
        dicom_input_files (str): Path to the input DICOM files.
        params_file (str): Path to the parameters file.
        device (str): The device to use for computations (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, dicom_input_files: str, params_file: str, models_dir: str, device: str):
        self.dicom_input_files = dicom_input_files # TODO: Unused variable ?
        self.dicoms = []
        self.device = device

        with open(params_file, 'r') as file:
            self.params = json.load(file)

        self.models_dir = models_dir
        
        self.segmentation_model = None
        self.severity_prediction_model = None
        self.object_recon_model = None
        

        self.load_models()

    def add_dicom(self, dicom_exam: 'DicomExam') -> None:
        """
        Adds a DICOM exam to the dataset.

        Args:
            dicom_exam (DicomExam): The DICOM exam to add.
        """
        self.dicoms.append(dicom_exam)

    def load_models(self) -> None:
        """
        Loads the segmentation and severity prediction models.
        """
        ### Load segmentation models
        self.segmentation_model = utils_refac.load_segmentation_UNet_models(
            self.models_dir, 
            self.params['unet_artery_segmentation_params']['weights'], 
            self.device
        )
        
        ### Load swin3d regression stenosis severity model
        self.severity_prediction_model = utils_refac.load_stenosis_severity_Swin3D_model(
            self.models_dir, 
            self.params['swin3d_severity_prediction_params']['weights'],
            self.device
        )
        
        ### Load swin3d object classification model
        self.object_recon_model = utils_refac.load_structure_recognition_Swin3D_model(
            self.models_dir, 
            self.params['swin3d_object_recon_params']['weights'],
            self.device
        )

    def segment_artery_subclass(self, device: str) -> None:
        """
        Performs artery segmentation on the DICOM exams.

        Args:
            device (str): The device to use for computations.
        """
        for idx, dicom_exam in tqdm(enumerate(self.dicoms),desc="(Algo 4) Inferring Artery Segmentation from DICOM: ",total=len(self.dicoms)):
            dicom_exam.batch_segmentation(self.segmentation_model, device)
        
        if 'cuda' in device:
            torch.cuda.empty_cache()

    def predict_stenosis_severity(self, device: str) -> None:
        """
        Predicts the severity of stenosis in the DICOM exams.

        Args:
            device (str): The device to use for computations.
        """
        batches = {
            'batch_videos': [],
            'batch_ages': [],
            'batch_segments': [],
            'batch_keys': [],
            'batch_stenosis_index': []
        }

        ### Small local function to make batch inference on stenosis videos
        def process_batch() -> None:
            video_batch = torch.cat(batches['batch_videos'], dim=0).to(device)
            age_batch = torch.tensor(batches['batch_ages']).to(device)
            segment_batch = torch.tensor(batches['batch_segments']).to(device)

            output = self.severity_prediction_model([video_batch, age_batch.unsqueeze(-1), segment_batch.unsqueeze(-1)])

            for i, key in enumerate(batches['batch_keys']):
                self.dicoms[int(key.split('-')[0])].stenoses[int(key.split('-')[1])].set_percent_stenosis(output.cpu().detach().numpy()[i][0])

            # Clear lists
            batches['batch_videos'].clear()
            batches['batch_ages'].clear()
            batches['batch_segments'].clear()
            batches['batch_keys'].clear()

        ## Iterate over all dicom > stenoses and create batches of videos, ages and segments for the swin3d model
        for dicom_idx, dicom_exam in tqdm(enumerate(self.dicoms),desc="(Algo 6) Swin3D severity prediction: ",total=len(self.dicoms)):
            for stenosis_idx, stenosis in enumerate(self.dicoms[dicom_idx].stenoses):
                batches['batch_videos'].append(self.dicoms[dicom_idx].stenoses[stenosis_idx].video)
                batches['batch_ages'].append(utils_refac.get_age(dicom_exam.dicom_info))
                batches['batch_segments'].append(self.params['swin3d_severity_prediction_params']['artery_labels'][stenosis.artery_segment])
                batches['batch_keys'].append(f"{dicom_idx}-{stenosis_idx}")

                ## Process the batch if it reaches the target length specified in params
                if len(batches['batch_videos']) == self.params['swin3d_severity_prediction_params']['batch_size']:
                    process_batch()

        ## If there are leftovers, process what is left 
        if len(batches['batch_videos']) > 0:
            process_batch()
            
        if 'cuda' in device:
            torch.cuda.empty_cache()

    def save_run(self, save_dir: str) -> None:
        """
        Saves the results of the stenosis analysis.

        Args:
            save_dir (str): The directory to save the results.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        df_stenoses_catalog = pd.DataFrame(
            {
                'dicom_path': [],
                'video_path': [],
                'frame': [],
                'resized_box_coordinates': [],
                'artery_segment': [],
                'percent_stenosis': [],
                'stenosis_severity': [],
                'swin3d_structure': []
            }
        )

        font = cv2.FONT_HERSHEY_PLAIN

        print(f"\n\nSaving video results to {save_dir}")
        for dicom_idx, dicom in tqdm(enumerate(self.dicoms), desc="Saving Results", total=len(self.dicoms)):
            for stenosis_idx, stenosis in enumerate(dicom.stenoses):
                video_save_path = f"{save_dir}dicom{dicom_idx}_stenosis{stenosis_idx}AtFrame{stenosis.frame}_{int(stenosis.percent_stenosis)}pct_{utils_refac.to_camel_case(stenosis.artery_segment)}.mp4"
                img_reg = np.zeros(dicom.dicom_info.pixel_array.shape)

                for frame_id in range(dicom.dicom_info.pixel_array.shape[0]):
                    reg_shift = stenosis.reg_shift[frame_id]
                    img_reg[frame_id] = scipy.ndimage.shift(dicom.dicom_info.pixel_array[frame_id], shift=(reg_shift[0], reg_shift[1]), order=5)
                    cv2.rectangle(img_reg[frame_id], (stenosis.resized_box_coordinates['x1'], stenosis.resized_box_coordinates['y1']), (stenosis.resized_box_coordinates['x2'], stenosis.resized_box_coordinates['y2']), (255, 255, 255), 1)
                    cv2.putText(img_reg[frame_id], stenosis.artery_segment + " - " + str(int(stenosis.percent_stenosis)) + '%', org=(stenosis.resized_box_coordinates['x1'], stenosis.resized_box_coordinates['y1']), fontFace=font, fontScale=1, color=(255, 255, 255))

                imageio.mimwrite(video_save_path, img_reg.astype(np.uint8), fps=15)

                df_stenoses_catalog = pd.concat([
                    df_stenoses_catalog,
                    pd.DataFrame([{
                        'dicom_path': dicom.dicom_path,
                        'video_path': video_save_path,
                        'frame': stenosis.frame,
                        'resized_box_coordinates': stenosis.resized_box_coordinates,
                        'artery_segment': stenosis.artery_segment,
                        'percent_stenosis': stenosis.percent_stenosis,
                        'stenosis_severity': stenosis.severe_stenosis,
                        'swin3d_structure': dicom.swin3d_structure
                    }])], ignore_index=True)

        print(f"\n\nSaving summary results to: \n{save_dir}df_stenosis.csv")
        df_stenoses_catalog.to_csv(save_dir + "df_stenosis.csv")


class DicomExam:
    """
    Represents a DICOM exam.

    Args:
        dicom_path (str): The path to the DICOM file.
        dicom_info (pydicom.dataset.FileDataset): The DICOM dataset information.
        anatomical_structure (str): The object value associated with the DICOM exam.
        pixel_spacing (float): The pixel spacing of the DICOM exam.
        params (dict): Additional parameters for the DICOM exam.
    """

    def __init__(self, dicom_path: str, anatomical_structure: str, params: dict):
        self.params = params
        self.dicom_path = dicom_path
        self.dicom_info = self.get_dicom_info() # TODO: Dicom VIEWER exists ? // return None but never handle
        self.processed_dicom = utils_refac.process_batch(self.dicom_info.pixel_array)
        self.pixel_spacing = self.get_pixel_spacing()
        self.fps = self.get_fps()
        self.anatomical_structure = anatomical_structure
        self.swin3d_structure = None
        self.segmentation_map = None
        self.stenoses = []
    
    def qc_skip(self, dicom_stenoses: 'pd.DataFrame') -> bool:
        """
        Perform various checks on the DICOM file and associated data.

        Args:
            sub_df (pd.DataFrame): The DataFrame containing stenosis information.
            dicom_path (str): The path to the DICOM file.
            dicom_info (pydicom.dataset.FileDataset): The DICOM dataset information.

        Returns:
            bool: True if all checks pass, False otherwise.
        """

        skip = False
        if len(dicom_stenoses['artery_view'].unique()) != 1:
            print(f"More than one artery view {dicom_stenoses['artery_view'].unique()} is associated with the same DICOM {dicom.dicom_path}")
            skip = True
            
        if dicom_stenoses['frame'].max() >= self.dicom_info.pixel_array.shape[0]:
            print(f"Maximum frame number ({dicom_stenoses['frame'].max()}) for dicom {self.dicom_path} is higher than or equal to the number of frames in the video ({self.dicom_info.pixel_array.shape[0]}).")
            skip = True
        
        anatomical_structure = dicom_stenoses['artery_view'].iloc[0]
        if anatomical_structure not in ["RCA", "LCA"]:
            print("DICOM does not display RCA or LCA.")
            skip = True
        
        if len(dicom_stenoses['artery_view'].unique()) != 1:
            print(f"More than one artery view {dicom_stenoses['artery_view'].unique()} is associated with the same DICOM {self.dicom_path}")
            skip = True

        if dicom_stenoses['frame'].max() >= self.dicom_info.pixel_array.shape[0]:
            print(f"Maximum frame number ({dicom_stenoses['frame'].max()}) for dicom {self.dicom_path} is higher than or equal to the number of frames in the video ({self.dicom_info.pixel_array.shape[0]}).")
            skip = True

        return skip
    
    def get_dicom_info(self):
        """
        
        Sets dicom_infos
        
        """
        
        try:
            dicom_info = pydicom.dcmread(self.dicom_path)
            required_tags = ['ImagerPixelSpacing', 'DistanceSourceToDetector', 'DistanceSourceToPatient']
            if not all(tag in dicom_info for tag in required_tags):
                missing_tags = [tag for tag in required_tags if tag not in dicom_info]
                print(f"Missing required DICOM tags in file: {self.dicom_path}. Missing tags: {', '.join(missing_tags)}.")
                return None
            return dicom_info 
        
        except pydicom.errors.InvalidDicomError as e:
            print(f"Error reading DICOM file: {self.dicom_path}. Skipping.")
            print(f"Error message: {str(e)}")
            return None
    
    def get_fps(self):
        """
        
        Finds the fps of the dicom video and prints a warning if fps != 15 
        
        """
        try:
            return int(self.dicom_info["RecommendedDisplayFrameRate"].value)
            if fps != 15:
                print(f"Frame rate not equal to 15 FPS ({fps} FPS) for dicom {self.dicom_path}.")
                
        except KeyError:    
            print(f'Frame rate information missing {self.dicom_path}.')
            return None
            
    def get_pixel_spacing(self):
        """
        
        Calculates the pixal spacing value if possible. Sets to None if absent from dicom infos.
        Needs to have access to dicom_info. 
        
        """
        try:
            imager_spacing = float(self.dicom_info['ImagerPixelSpacing'][0])
            factor = float(self.dicom_info['DistanceSourceToDetector'].value) / float(self.dicom_info['DistanceSourceToPatient'].value)
            pixel_spacing = float( imager_spacing / factor)
            if imager_spacing <= 0 or factor <= 0:
                raise ValueError(f'Pixel spacing information invalid for dicom {self.dicom_path}.')
            
            return pixel_spacing
        except (KeyError, ValueError):
            print(f'Pixel spacing information missing or invalid for dicom {self.dicom_path}. Setting to None') 
            return None
    
    
    def add_stenosis(self, stenosis: 'Stenosis') -> None:
        """
        Adds a stenosis to the DICOM exam.

        Args:
            stenosis (Stenosis): The stenosis to add.
        """
        self.stenoses.append(stenosis)
        

    def batch_segmentation(self, models: List['torch.nn.Module'], device: str) -> None:
        """
        Performs batch segmentation on the DICOM exam.

        Args:
            models (List[torch.nn.Module]): The segmentation models.
            device (str): The device to use for computations.
        """
        
        batch = self.processed_dicom.to(device)
        outputs = []
        with torch.no_grad():
            for model in models:
                outputs.append(model(batch))

        self.segmentation_output = torch.stack(outputs).mean(dim=0).argmax(1).to('cpu').numpy()

        for idx, stenosis in enumerate(self.stenoses):
            self.stenoses[idx].assign_artery_segment(self.segmentation_output, self.anatomical_structure)
    
    def object_recon(self, device: str, model) -> torch.Tensor:
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
        mean = [104.86392211914062, 104.86392211914062, 104.86392211914062]
        std = [53.32306671142578, 53.32306671142578, 53.32306671142578]
        video_length = 32
        period = 2 
        
        video = torch.from_numpy(self.dicom_info.pixel_array)
        video = video.unsqueeze(0).repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
        
        video = v2.Resize((224, 224), antialias=None)(video).to(torch.float32)
        video = v2.Normalize(mean, std)(video)
        
        video = video.permute(1, 0, 2, 3)
        c, f, h, w = video.shape
        video.numpy()
        
        if f < video_length * period:
            # Pad video_lenght with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate(
                (video, np.zeros((c, video_length * period - f, h, w), video.dtype)), axis=1
            )
            c, f, h, w = video.shape 
        
        video = tuple(video[:, s + period * np.arange(video_length), :, :] for s in np.array([0]))
        
        extracted_frames = extract_frames(video)
        extracted_frames = extracted_frames[0].unsqueeze(0).to(device)
        
        model.to(device)    
        output = model(extracted_frames)
        self.swin3d_structure = self.params["swin3d_object_recon_params"]["object_dict"][str(np.argmax(output.cpu().detach().numpy()))]
        
        if 'cuda' in device:
            torch.cuda.empty_cache()
            
        


class Stenosis(DicomExam):
    """
    Represents a stenosis in a DICOM exam.

    Args:
        dicom_exam (DicomExam): The DICOM exam object containing the stenosis.
        frame (int): The frame number of the stenosis.
        stenosis_box (dict): The bounding box coordinates of the stenosis.
    """

    def __init__(self, dicom_exam: DicomExam, frame: int, stenosis_box: dict):
        super().__init__(dicom_exam.dicom_path, dicom_exam.anatomical_structure, dicom_exam.params)
        self.frame = frame
        self.initial_box_coordinates = stenosis_box
        self.resized_box_coordinates = utils_refac.resize_coordinates(
            self.initial_box_coordinates, self.pixel_spacing, self.dicom_info.pixel_array.shape, self.params['stenosis_params']['target_size']
        )
        self.reg_shift = utils_refac.register(self.dicom_info.pixel_array, self.frame, self.resized_box_coordinates)
        self.video = utils_refac.create_cropped_registered_video(
            self.dicom_info.pixel_array, self.frame, self.reg_shift, self.resized_box_coordinates
        )

        self.artery_segment = None
        self.percent_stenosis = None
        self.severe_stenosis = None

    def set_percent_stenosis(self, percent_stenosis: float) -> None:
        """
        Sets the percent stenosis and determines the severity of the stenosis.

        Args:
            percent_stenosis (float): The percent stenosis value.
        """
        self.percent_stenosis = percent_stenosis * 100
        self.severe_stenosis = int(percent_stenosis > self.params['swin3d_object_recon_params']['percentage_threshold'])

    def assign_artery_segment(self, segmentation_output: np.ndarray, anatomical_structure: str) -> None:
        """
        Assigns the artery segment to the stenosis based on the segmentation output.

        Args:
            segmentation_output (np.ndarray): The segmentation output array.
            anatomical_structure (str): The anatomical structure of the stenosis.
        """
        pad_value = max([
            (self.resized_box_coordinates['x2'] - self.resized_box_coordinates['x1']),
            (self.resized_box_coordinates['y2'] - self.resized_box_coordinates['y1'])
        ])

        preds = []
        for j in range(segmentation_output.shape[0]):
            padded_output = np.pad(
                segmentation_output[j], ((pad_value, pad_value), (pad_value, pad_value)),
                mode='constant', constant_values=0
            )
            y_shift = -self.reg_shift[j][0] + pad_value
            x_shift = -self.reg_shift[j][1] + pad_value
            region = padded_output[
                int(self.resized_box_coordinates['y1'] + y_shift): int(self.resized_box_coordinates['y2'] + y_shift),
                int(self.resized_box_coordinates['x1'] + x_shift): int(self.resized_box_coordinates['x2'] + x_shift)
            ].astype('uint8')
            segment = utils_refac.get_segment_center(region, anatomical_structure)
            preds.append(segment)

        cleaned_preds = [i for i in preds if i not in ['None', 'other']]
        if len(cleaned_preds) == 0:
            self.artery_segment = 'None'
        else:
            counts = Counter(cleaned_preds)
            self.artery_segment = max(counts, key=lambda x: (
                counts[x],
                -(self.params["assign_artery_segment_params"]["segments_of_interest"]["rca"] + self.params["assign_artery_segment_params"]["segments_of_interest"]["lca"]).index(x)
                if x in (self.params["assign_artery_segment_params"]["segments_of_interest"]["rca"] + self.params["assign_artery_segment_params"]["segments_of_interest"]["lca"])
                else float('inf')
            ))
            
            
def extract_frames(video, num_frames=32):
    total_frames = video[0].shape[0]
    middle_frame = total_frames // 2
    start_frame = max(0, middle_frame - num_frames // 2)
    end_frame = min(total_frames, start_frame + num_frames)
    extracted_frames = video[start_frame:end_frame:2]
    return extracted_frames
