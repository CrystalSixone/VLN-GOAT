# Instructions for Extracting Backdoor Dictionaries
- The process is streamlined into two main steps: extracting room types and generating backdoor dictionaries. Please follow the steps below carefully to ensure successful extraction. 
- Our extracted room types are available here.

## 1. Extract Room Types
### 1. Prepare the Matterport3D (MP3D) Dataset:
- Access the Matterport3D dataset by requesting permission [here](https://niessner.github.io/Matterport/).
- Once access is granted, use the provided download script to obtain the `matterport_skybox_images`. These images are essential for room type extraction.

### 2. Set up the BLIP Model:
- For some reasons that it may not possible to access huggingface's website directly. In this case, I suggest to download the model from [here](https://huggingface.co/Salesforce/blip-vqa-base) and save it locally for subsequent use.

### 3. Execute Room Type Extraction
- Now, let's extract the room types for each sub-image. Execute the following commands, ensuring that you adjust the path in the `extract_room_type.bash` script to match your local setup.
```bash
bash map_nav_src/do_utils/extract_room_type.bash
```

## 2. Generate Backdoor Dictionaries
- Once you have the `pano_roomtype.tsv` file, you're ready to generate backdoor dictionaries that correlate image room types with text keywords.
- Using the command below, you will generate the backdoor dictionaries. Please pay close attention to the specific paths within the code. Modify them as needed to reflect your directory structure accurately.
```python
python map_nav_src/do_utils/do_intervention.py
```
- After obtaining the files, e.g., `image_z_dict_clip_50.tsv` and `r2r_z_instr_dict.tsv`, put them to the dictionary `datasets/R2R/features`. Make sure the names of these files are the same with the statements in `map_nav_src/r2r/parser.py`.