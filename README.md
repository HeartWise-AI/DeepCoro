# DeepCoro
This repository contains the material necessary to run inference on a DICOM coronary angiography video. Models for the primary anatomic struction classification (Algorithm 1) and stenosis detection (Algorithm 2) could not be made public. Therefore, coronary arteries viewed and stenoses need to be manually identified in an input csv file.

### Outputs
The outputs generated from inference are:
- Registered videos of detected stenoses with the identification of the stenosis location, its associated coronary artery segment and its percentage of obstruction.
- A ```df_stenosis.csv``` file where, for each output registered video, there is information about:
  * video_path: Path of the ouput video.
  * frame: Reference frame used for registration.
  * box_resized: Coordinates of the resized stenosis box (displayed in the video).
  * artery_segment: Coronary artery segment on which the stenosis has been identified. 
  * percent_stenosis: Percentage of obstruction predicted. 
  * severe_stenosis: Mention of if the stenosis is severe or not according to our predetermined threshold. 

### DICOM requirements 
The DICOM on which inference is ran must:
- Contain information about:
  * Imager Pixel Spacing
  * Distance Source To Detector
  * Distance Source To Patient
  * Recommended Display Frame Rate
  * Study Date
  * Patient's Birth Date
- Display the right coronary arteries (RCA) or the left coronary arteries (LCA).
- Be aquired at 15 frames per seconds.

### Models
Trained models are available on HuggingFace to perform inference: [DeepCoro models](https://huggingface.co/heartwise/DeepCoro/tree/main)

### Environment 
The file ```deepcoro.yml``` contains the environment on which DeepCoro inference must be ran on. 

### Input csv file
This file contains 7 columns:
* dicom_path: The path to the DICOM on which you want to run inference.
* artery_view: The manual identification of the coronary arteries viewed in the coronary angiography (RCA or LCA).
* frame: Frame index at which a stenosis has been detected. 
* x1: Lower-valued x-coordinate of the rectangle's corner identifying the stenosis location. 
* y1: Lower-valued y-coordinate of the rectangle's corner identifying the stenosis location. 
* x2: Higher-valued x-coordinate of the rectangle's corner identifying the stenosis location. 
* y2: Higher-valued y-coordinate of the rectangle's corner identifying the stenosis location. 

Example of the input csv file:
| dicom_path | artery_view | frame | x1 | y1 | x2 | y2 |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| dicom1.dcm | RCA | 50 | 122 | 214 | 211 | 303 |
| dicom1.dcm | RCA | 62 | 195 | 84 | 284 | 173 |
| dicom2.dcm | LCA | 21 | 30 | 105 | 51 | 159 |
| dicom2.dcm | LCA | 34 | 259 | 110 | 305 | 187 |
| dicom2.dcm | LCA | 60 | 203 | 217 | 248 | 276 |

### Run inference
To run inference on a DICOM, you must run the following command in a terminal:

```
python main.py --dicom_path YOUR_DICOM_PATH --save_dir YOUR_SAVE_DIR --artery_view DICOM_ARTERY_VIEW --models_dir YOUR_MODELS_DIR --device YOUR_DEVICE
```

where the inputs are
- input_file_path: The path to the unput csv file.
- save_dir: The directory where you want the outputs to be saved.
- models_dir: The directory where the models (the content of the 'models' file available on HuggingFace) are located (optional, default = models/).
- device: The device on which you want to run inference (cuda or cpu) (optional, default = cuda).

