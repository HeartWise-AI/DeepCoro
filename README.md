# DeepCoro
This repository contains the material necessary to run inference on a DICOM coronary angiography video. Models for the primary anatomic struction classification and projection angle classification could not be made public. Therefore, as an alternative, primary structure is identified manually, and projection angle is identified with the DICOM metadata. 

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
  * Positioner Primary Angle
  * Positioner Secondary Angle
- Display the right coronary arteries (RCA) or the left coronary arteries (LCA).
- Be aquired at 15 frames per seconds.

### Models
Trained models are available on HuggingFace to perform inference: [DeepCoro models](https://huggingface.co/heartwise/DeepCoro/tree/main)

### Environment 
The file ```deepcoro.yml``` contains the environment on which DeepCoro inference must be ran on. 

### Run inference
To run inference on a DICOM, you must run the following command in a terminal:

```
python main.py --dicom_path YOUR_DICOM_PATH --save_dir YOUR_SAVE_DIR --artery_view DICOM_ARTERY_VIEW --models_dir YOUR_MODELS_DIR --device YOUR_DEVICE
```

where the inputs are
- dicom_path: The path to the DICOM on which you want to run inference.
- save_dir: The directory where you want the outputs to be saved.
- artery_view: The manual identification of the coronary arteries viewed in the coronary angiography (RCA or LCA). 
- models_dir: The directory where the models (the content of the 'models' file available on HuggingFace) are located (optional, default = models/).
- device: The device on which you want to run inference (cuda or cpu) (optional, default = cuda).

