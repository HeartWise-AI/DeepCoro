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


class StenosisDataset:
    """
    Represents a dataset of stenosis in DICOM exams.

    Args:
        dicom_input_files (str): Path to the input DICOM files.
        params_file (str): Path to the parameters file.
        device (str): The device to use for computations (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, dicom_input_files: str, params_file: str, device: str):
        self.dicom_input_files = dicom_input_files
        self.dicoms = []
        self.device = device

        with open(params_file, 'r') as file:
            self.params = json.load(file)

        self.segmentation_model = None
        self.severity_prediction_model = None

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
        self.segmentation_model = [utils_refac.load_model(path, self.device) for path in self.params["segmentation_models_weights"]]

        severity_prediction_model = utils_refac.get_model().to(self.device).double()
        checkpoint = torch.load(self.params["swin3d_model_weights"], map_location=self.device)["state_dict"]
        if self.device == 'cpu':
            checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
        else:
            severity_prediction_model = torch.nn.DataParallel(severity_prediction_model)
        severity_prediction_model.eval()
        severity_prediction_model.load_state_dict(checkpoint)
        self.severity_prediction_model = severity_prediction_model

    def segment_artery_subclass(self, device: str) -> None:
        """
        Performs artery segmentation on the DICOM exams.

        Args:
            device (str): The device to use for computations.
        """
        print("\n\n(Algo 4) Inferring Artery Segmentation from DICOM ...")
        for idx, dicom_exam in enumerate(self.dicoms):
            dicom_exam.batch_segmentation(self.segmentation_model, device)

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

        def process_batch() -> None:
            video_batch = torch.cat(batches['batch_videos'], dim=0).to(device)
            age_batch = torch.tensor(batches['batch_ages']).to(device)
            segment_batch = torch.tensor(batches['batch_segments']).to(device)

            output = self.severity_prediction_model([video_batch, age_batch.unsqueeze(-1), segment_batch.unsqueeze(-1)])

            for i, key in enumerate(batches['batch_keys']):
                self.dicoms[int(key.split('-')[0])].stenoses[int(key.split('-')[1])].set_percent_stenosis(output.cpu().detach().numpy()[i][0])

            # Clear lists
            batches.clear()

        for dicom_idx, dicom_exam in tqdm(enumerate(self.dicoms), desc="Batching Stenosis", total=len(self.dicoms)):
            for stenosis_idx, stenosis in enumerate(self.dicoms[dicom_idx].stenoses):
                batches['batch_videos'].append(self.dicoms[dicom_idx].stenoses[stenosis_idx].video)
                batches['batch_ages'].append(utils_refac.get_age(dicom_exam.dicom_info))
                batches['batch_segments'].append(self.params['artery_labels'][stenosis.artery_segment])
                batches['batch_keys'].append(f"{dicom_idx}-{stenosis_idx}")

                if len(batches['batch_videos']) == self.params['batch_size']:
                    process_batch()

        if len(batches['batch_videos']) > 0:
            process_batch()

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
                'stenosis_severity': []
            }
        )

        font = cv2.FONT_HERSHEY_PLAIN

        print(f"\n\nSaving video results to {save_dir}")
        for dicom_idx, dicom in tqdm(enumerate(self.dicoms), desc="Saving Results", total=len(self.dicoms)):
            for stenosis_idx, stenosis in enumerate(dicom.stenoses):
                video_save_path = f"{save_dir}dicom{dicom_idx}_stenosis{stenosis_idx}AtFrame{stenosis.frame}_{int(stenosis.percent_stenosis)}pct_{utils_refac.to_camel_case(stenosis.artery_segment)}.mp4"

                img_reg = np.zeros(dicom.dicom_data.shape)

                for frame_id in range(dicom.dicom_data.shape[0]):
                    reg_shift = stenosis.reg_shift[frame_id]
                    img_reg[frame_id] = scipy.ndimage.shift(dicom.dicom_data[frame_id], shift=(reg_shift[0], reg_shift[1]), order=5)
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
                        'stenosis_severity': stenosis.severe_stenosis
                    }])], ignore_index=True)

        print(f"\n\nSaving summary results to: \n{save_dir}df_stenosis.csv")
        df_stenoses_catalog.to_csv(save_dir + "df_stenosis.csv")


class DicomExam:
    """
    Represents a DICOM exam.

    Args:
        dicom_path (str): The path to the DICOM file.
        dicom_info (pydicom.dataset.FileDataset): The DICOM dataset information.
        dicom_data (np.ndarray): The DICOM pixel data.
        object_value (str): The object value associated with the DICOM exam.
        pixel_spacing (float): The pixel spacing of the DICOM exam.
        params (dict): Additional parameters for the DICOM exam.
    """

    def __init__(self, dicom_path: str, dicom_info: 'pydicom.dataset.FileDataset', dicom_data: np.ndarray,
                 object_value: str, pixel_spacing: float, params: dict):
        self.params = params
        self.dicom_path = dicom_path
        self.dicom_info = dicom_info
        self.dicom_data = dicom_data
        self.processed_dicom = utils_refac.process_batch(self.dicom_data)

        self.pixel_spacing = float(self.dicom_info['ImagerPixelSpacing'][0] / (self.dicom_info['DistanceSourceToDetector'].value / self.dicom_info['DistanceSourceToPatient'].value))
        self.object_value = object_value

        self.segmentation_output = None
        self.stenoses = []

    def add_stenosis(self, stenosis: 'Stenosis') -> None:
        """
        Adds a stenosis to the DICOM exam.

        Args:
            stenosis (Stenosis): The stenosis to add.
        """
        self.stenoses.append(stenosis)

    def set_segmentation_output(self, segmentation_output: np.ndarray) -> None:
        """
        Sets the segmentation output for the DICOM exam.

        Args:
            segmentation_output (np.ndarray): The segmentation output array.
        """
        self.segmentation_output = segmentation_output

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
            self.stenoses[idx].assign_artery_segment(self.segmentation_output, self.object_value)


class Stenosis(DicomExam):
    """
    Represents a stenosis in a DICOM exam.

    Args:
        dicom_exam (DicomExam): The DICOM exam object containing the stenosis.
        frame (int): The frame number of the stenosis.
        stenosis_box (dict): The bounding box coordinates of the stenosis.
    """

    def __init__(self, dicom_exam: DicomExam, frame: int, stenosis_box: dict):
        super().__init__(dicom_exam.dicom_path, dicom_exam.dicom_info, dicom_exam.dicom_data,
                         dicom_exam.object_value, dicom_exam.pixel_spacing, dicom_exam.params)
        self.frame = frame
        self.initial_box_coordinates = stenosis_box
        self.resized_box_coordinates = utils_refac.resize_coordinates(
            self.initial_box_coordinates, self.pixel_spacing, self.dicom_data.shape, self.params['target_size']
        )
        self.reg_shift = utils_refac.register(self.dicom_data, self.frame, self.resized_box_coordinates)
        self.video = utils_refac.create_cropped_registered_video(
            self.dicom_data, self.frame, self.reg_shift, self.resized_box_coordinates
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
        self.severe_stenosis = int(percent_stenosis > self.params['percentage_threshold'])

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
            final_pred_segment = max(counts, key=lambda x: (
                counts[x],
                -(self.params["segments_of_interest"]["rca"] + self.params["segments_of_interest"]["lca"]).index(x)
                if x in (self.params["segments_of_interest"]["rca"] + self.params["segments_of_interest"]["lca"])
                else float('inf')
            ))
            self.artery_segment = final_pred_segment