import argparse
import pydicom
import torch
import pandas as pd
import utils
from tqdm import tqdm
import time
import logging
import sys 

from classes import Stenosis, StenosisDataset, DicomExam


def read_videos(input_file_path: str, params_file: str) -> StenosisDataset:
    """
    Read DICOM files and perform video registration.

    Args:
        input_file_path (str): Path to the input CSV file.
        params_file (str): Path to the parameters file.

    Returns:
        StenosisDataset: Dataset containing processed DICOM data.
    """
    df = pd.read_csv(input_file_path, dtype={
        'artery_view': str,
        'frame': 'int64',
        'x_start': 'float64',
        'y_start': 'float64',
        'x_end': 'float64',
        'y_end': 'float64'
    })
    df['artery_view'] = df['artery_view'].str.upper()

    prefix = input_file_path[:input_file_path.find('input_file.csv')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    stenosis_dataset = StenosisDataset(input_file_path, params_file, device)

    logging.info("\t\tReading DICOM files (resizing box coordinates + video registration) ...")
    for dicom_path, stenosis_df in tqdm(df.groupby('dicom_path'), desc="Processing DICOM files"):
        try:
            dicom_info = pydicom.dcmread(prefix + dicom_path)
        except pydicom.errors.InvalidDicomError as e:
            logging.error(f"Error reading DICOM file: {dicom_path}. Skipping.")
            logging.error(f"Error message: {str(e)}")
            continue

        if not utils.perform_checks(stenosis_df, dicom_path, dicom_info):
            continue

        required_tags = ['ImagerPixelSpacing', 'DistanceSourceToDetector', 'DistanceSourceToPatient']
        if not all(tag in dicom_info for tag in required_tags):
            missing_tags = [tag for tag in required_tags if tag not in dicom_info]
            logging.warning(f"Missing required DICOM tags in file: {dicom_path}. Missing tags: {', '.join(missing_tags)}. Skipping.")
            continue

        predicted_artery_view = stenosis_df['artery_view'].iloc[0]

        dicom_exam = DicomExam(dicom_path, dicom_info, dicom_info.pixel_array, predicted_artery_view, stenosis_dataset.params)

        for _, row in stenosis_df.iterrows():
            stenosis_box = {
                'x1': row['x1_stenosis'],
                'y1': row['y1_stenosis'],
                'x2': row['x2_stenosis'],
                'y2': row['y2_stenosis']
            }
            new_stenosis = Stenosis(dicom_exam, row['frame'], stenosis_box)
            dicom_exam.add_stenosis(new_stenosis)

        stenosis_dataset.add_dicom(dicom_exam)

    return stenosis_dataset
def parse_args(args):
    """
    Parsing arguments for running DeepCoro. 

    Args:
        args (Union[argparse.Namespace, None]): Command-line arguments. Defaults to None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    
    parser = argparse.ArgumentParser(description='DeepCoro')

    parser.add_argument('--input_file_path', default='/volume/deepcoro/repotest/DeepCoro/random_dicoms/input_file.csv', help='Path to the input CSV file')
    parser.add_argument('--models_dir', default='/volume/deepcoro/repotest/DeepCoro/models/', help='Directory containing the models')
    parser.add_argument('--save_dir', default='/volume/deepcoro/repotest/DeepCoro/results/4_stenosis/batch_inference/', help='Directory to save the results')
    parser.add_argument('--params_file', default='/volume/deepcoro/repotest/DeepCoro/params.json', help='Path to the parameters file')

    return parser.parse_args(args)


def main(args: None) -> None:
    """
    Main function to run the DeepCoro algorithm.

    Args:
        args (Union[argparse.Namespace, None]): Command-line arguments. Defaults to None.
    """
    
    parsed_args = parse_args(args)
    input_file_path = parsed_args.input_file_path
    save_dir = parsed_args.save_dir
    params_file = parsed_args.params_file

    total_start_time = time.time()

    logging.info("\t\tStarting DeepCORO algorithm suite with the following parameters:")
    logging.info(f"\t* input_file_path: {input_file_path}")
    logging.info(f"\t* save_dir: {save_dir}")

    start_time = time.time()
    stenosis_dataset = read_videos(input_file_path, params_file)
    end_time = time.time()
    logging.info(f'Elapsed Time (read_video): {end_time - start_time}s')

    start_time = time.time()
    stenosis_dataset.segment_artery_subclass(stenosis_dataset.device)
    end_time = time.time()
    logging.info(f'Elapsed Time (segment_artery_subclass): {end_time - start_time}s')

    start_time = time.time()
    stenosis_dataset.predict_stenosis_severity(stenosis_dataset.device)
    end_time = time.time()
    logging.info(f'Elapsed Time (predict_stenosis_severity): {end_time - start_time}s')

    start_time = time.time()
    stenosis_dataset.save_run(save_dir)
    end_time = time.time()
    logging.info(f'Elapsed Time (save_run): {end_time - start_time}s')

    total_end_time = time.time()

    logging.info(f'Total Elapsed Time: {total_end_time - total_start_time}s')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])