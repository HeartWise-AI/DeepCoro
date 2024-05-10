import argparse
import torch
import pandas as pd
from tqdm import tqdm
import time
import sys 
import os

from classes import Stenosis, StenosisDataset, DicomExam


def process_dicoms(input_path: str, params_file: str, models_dir: str) -> StenosisDataset:
    """
    Read DICOM files and perform video registration.

    Args:
        input_path (str): Path to the input CSV file.
        params_file (str): Path to the parameters file.

    Returns:
        StenosisDataset: Dataset containing processed DICOM data.
    """
    df = pd.read_csv(input_path, dtype={
        'artery_view': str,
        'frame': 'int64',
        'x_start': 'float64',
        'y_start': 'float64',
        'x_end': 'float64',
        'y_end': 'float64'
    })
    df['artery_view'] = df['artery_view'].str.upper()

    input_folder = input_path[:input_path.find('input_file.csv')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    stenosis_dataset = StenosisDataset(input_path, params_file, models_dir, device)

    print("\t\tReading DICOM files (resizing box coordinates + video registration) ...")
    for dicom_path, stenosis_df in tqdm(df.groupby('dicom_path'), desc="Processing DICOM files"):
        
        predicted_anatomical_structure = stenosis_df['artery_view'].iloc[0]
        dicom_exam = DicomExam(os.path.join(input_folder, dicom_path), predicted_anatomical_structure, stenosis_dataset.params)
        
        dicom_exam.object_recon(stenosis_dataset.device, stenosis_dataset.object_recon_model)
        
        if dicom_exam.qc_skip(stenosis_df):
            continue
        
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
        args : Command-line arguments. Defaults to None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    
    
    parser = argparse.ArgumentParser(description='DeepCoro')


    parser.add_argument('--workdir', help='Path to the input CSV file',default="/volume/deepcoro/repotest/DeepCoro/")

    parser.add_argument('--input_path', help='Path to the input CSV file',default="random_dicoms/input_file.csv")
    parser.add_argument('--save_dir', help='Directory to save the results',default="results/inference")
    parser.add_argument('--models_dir', help='Directory to save the results',default="models/")
    parser.add_argument('--params_file', help='Path to the parameters file',default="params.json")

    return parser.parse_args(args)


def main(args: None) -> None:
    """
    Main function to run the DeepCoro algorithm.

    Args:
        args (Union[argparse.Namespace, None]): Command-line arguments. Defaults to None.
    """
    
        
    parsed_args = parse_args(args)
    
    workspace_dir = parsed_args.workdir
    input_path = os.path.join(workspace_dir, parsed_args.input_path)    
    save_dir = os.path.join(workspace_dir, parsed_args.save_dir)
    params_file = os.path.join(workspace_dir, parsed_args.params_file)
    
    print(workspace_dir)
    if workspace_dir == '/':
        model_root_dir = '/opt/deepcoro/'
    else:
        model_root_dir = parsed_args.workdir
    models_dir = os.path.join(model_root_dir, parsed_args.models_dir)
    
    
    total_start_time = time.time()

    print("\tStarting DeepCORO algorithm suite with the following parameters:")
    print(f"\t\t* input_path: {input_path}")
    print(f"\t\t* save_dir: {save_dir}")

    start_time = time.time()
    stenosis_dataset = process_dicoms(input_path, params_file, models_dir)
    end_time = time.time()
    print(f'\t ** Elapsed Time (read_video): {end_time - start_time}s ** \n\n')

    start_time = time.time()
    stenosis_dataset.segment_artery_subclass(stenosis_dataset.device)
    end_time = time.time()
    print(f'\t ** Elapsed Time (segment_artery_subclass): {end_time - start_time}s ** \n\n')

    start_time = time.time()
    stenosis_dataset.predict_stenosis_severity(stenosis_dataset.device)
    end_time = time.time()
    print(f'\t ** Elapsed Time (predict_stenosis_severity): {end_time - start_time}s ** \n\n')

    start_time = time.time()
    stenosis_dataset.save_run(save_dir)
    end_time = time.time()
    print(f'\t ** Elapsed Time (save_run): {end_time - start_time}s ** \n\n')

    total_end_time = time.time()

    print(f'\t ** Total Elapsed Time: {total_end_time - total_start_time}s ** \n\n')


if __name__ == '__main__':
    main(sys.argv[1:])