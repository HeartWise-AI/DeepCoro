# DeepCoro
This repository contains the material necessary to run inference on a DICOM coronary angiography video. Models for the primary anatomic struction classification (Algorithm 1) and stenosis detection (Algorithm 2) could not be made public. Therefore, coronary arteries viewed and stenoses need to be manually identified in an input csv file.

### Input folder
The input folder ```dcm_input/``` contains the following inputs:
1. Your DICOMs to run inference on. The DICOM on which inference is ran must:
    * Contain information about:
      - Imager Pixel Spacing
      - Distance Source To Detector
      - Distance Source To Patient
      - Recommended Display Frame Rate
      - Study Date
      - Patient's Birth Date
    * Display the right coronary arteries (RCA) or the left coronary arteries (LCA).
    * Be aquired at 15 frames per seconds.
2. An input csv file named ```input_file.csv``` that contains 7 columns:
    * dicom_path: The path to the DICOM on which you want to run inference.
    * artery_view: The manual identification of the coronary arteries viewed in the coronary angiography (RCA or LCA).
    * frame: Frame index at which a stenosis has been detected. 
    * x1: Lower-valued x-coordinate of the rectangle's corner identifying the stenosis location. 
    * y1: Lower-valued y-coordinate of the rectangle's corner identifying the stenosis location. 
    * x2: Higher-valued x-coordinate of the rectangle's corner identifying the stenosis location. 
    * y2: Higher-valued y-coordinate of the rectangle's corner identifying the stenosis location. 

Example of ```input_file.csv```:
| dicom_path | artery_view | frame | x1 | y1 | x2 | y2 |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| dicom1.dcm | RCA | 50 | 122 | 214 | 211 | 303 |
| dicom1.dcm | RCA | 62 | 195 | 84 | 284 | 173 |
| dicom2.dcm | LCA | 21 | 30 | 105 | 51 | 159 |
| dicom2.dcm | LCA | 34 | 259 | 110 | 305 | 187 |
| dicom2.dcm | LCA | 60 | 203 | 217 | 248 | 276 |

### Output folder
The outputs are generated from inference in a folder ```results/``` which contains the following outputs:
1. Registered videos of detected stenoses with the identification of the stenosis location, its associated coronary artery segment and its percentage of obstruction.
2. A ```df_stenosis.csv``` file where, for each output registered video, there is information about:
     * dicom_path: Path to the DICOM on which inference is run.
     * video_path: Path of the ouput video.
     * frame: Reference frame used for registration.
     * box: Coordinates of the input stenosis box.
     * box_resized: Coordinates of the resized stenosis box (displayed in the video).
     * artery_segment: Coronary artery segment on which the stenosis has been identified. 
     * percent_stenosis: Percentage of obstruction predicted. 
     * severe_stenosis: Mention of if the stenosis is severe or not according to our predetermined threshold. 

### Run inference
To run inference on DICOMs identified in your input csv file, you must run the following command in a terminal:
1. Clone the DeepCoro repository:
 ```
 https://github.com/HeartWise-AI/DeepCoro.git
 ```
2. Build the docker that contains the environment on which inference will be run, and all the weights and files necessary for the task. In the repository, run:
 ```
 docker build -t deepcoro_inference .
 ```
3. Place your input files in the ```dcm_input/``` input folder created (see ```Input folder``` section above).
4. Run inference:
 * To run on CPU:
 ```
   docker run -v /path/to/dcm_input:/dcm_input -v /path/to/results:/results deepcoro_inference
 ```
 * To run on GPU:
 ```
   docker run --gpus all -v /path/to/dcm_input:/dcm_input -v /path/to/results:/results deepcoro_inference
 ```
 where 
    - /path/to/dcm_input: The path to the input folder ```dcm_input/```.
    - /path/to/results: The path to the output folder ```results/```.

### Models
Trained models are available on HuggingFace to perform inference: [DeepCoro models](https://huggingface.co/heartwise/DeepCoro/tree/main)
