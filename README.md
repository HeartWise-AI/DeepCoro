# DeepCoro
This repository contains the material necessary to run inference on a DICOM coronary angioography video. 

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
- Display the right coronary arteries or the left coronary arteries.
- Be aquired at 15 frames per seconds.

### Models
Trained models are available on HuggingFace to perform inference: [DeepCoro models](https://huggingface.co/heartwise/DeepCoro/tree/main)

### Environment 
The file ```deepcoro.yml``` contains the environment on which DeepCoro inference must be ran on. 

### Run inference
To run inference on a DICOM, you must run the following command in a terminal:

```
python main.py --dicom_path YOUR_DICOM_PATH --save_dir YOUR_SAVE_DIR --models_dir YOUR_MODELS_DIR --device YOUR_DEVICE
```

where the inputs are
- dicom_path: The path to the DICOM on which you want to run inference.
- save_dir: The directory where you want the outputs to be saved.
- models_dir: The directory where the models (the content of the 'models' file available on HuggingFace) are located (optional, default = models/).
- device: Device on which you want to run inference (cuda or cpu) (optional, default = cuda).

