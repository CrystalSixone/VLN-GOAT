# Vision-and-Language Navigation via Causal Learning
**Abstract:** In the pursuit of robust and generalizable environment perception and language understanding, the ubiquitous challenge of dataset bias continues to plague vision-and-language navigation (VLN) agents, hindering their performance in unseen environments. This paper introduces the generalized cross-modal causal transformer (GOAT), a pioneering solution rooted in the paradigm of causal inference. By delving into both observable and unobservable confounders within vision, language, and history, we propose the back-door and front-door adjustment causal learning (BACL and FACL) modules to promote unbiased learning by comprehensively mitigating potential spurious correlations. 
Additionally, to capture global confounder features, we propose a cross-modal feature pooling (CFP) module supervised by contrastive learning, which is also shown to be effective in improving cross-modal representations during pre-training. Extensive experiments across multiple VLN datasets (R2R, REVERIE, RxR, and SOON) underscore the superiority of our proposed method over previous state-of-the-art approaches.


## Setup Instructions

### 1. Requirements and Installation

1. **Install MatterPort3D Simulator:** Start by installing the MatterPort3D simulator from the official [repository](https://github.com/peteanderson80/Matterport3DSimulator).

2. **Install Python Dependencies:** Run the following command to install the necessary Python packages. Make sure to match the versions in `requirements.txt` to avoid compatibility issues, particularly when loading pre-trained weights for fine-tuning.
    ```setup
    pip install -r requirements.txt
    ```
3. **Install en_core_web_sm**: Run the following command:
    ```setup
    pip install spacy
    wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz
    pip install en_core_web_sm-2.3.0.tar.gz
    ```
3. **Install nltk_data**: Run the following command to use the NLTK Downloader to obtain the resource:
    ```setup
    python
    >>> import nltk
    >>> nltk.download('wordnet')
    ```
3. **Download Resources**:
    1. **Datasets, Features and Trained-weights:**: Available [here](https://huggingface.co/crystal61/VLN-GOAT).
    3. **METER Pre-training (Optional):** If you wish to pre-train GOAT using METER, download the model `meter_clip16_224_roberta_pretrain.ckpt` from [here](https://github.com/zdou0830/METER).
    4. **EnvEdit Weights (Optional)**: Available [here](https://github.com/jialuli-luka/EnvEdit).
    5. **RoBERTa Tokenizer**: If direct access to Hugging Face models is restricted, manually download `roberta-base` from [Hugging Face](https://huggingface.co/FacebookAI/roberta-base/tree/main) and store it locally under `datasets/pretrained/roberta`.

    Ensure your `datasets` directory follows this structure:
    ```
    datasets
    ├── R2R
    │   ├── annotations
    │   │   ├──pretrain_map
    │   │   └──RxR
    │   ├── connectivity
    │   ├── features
    │   ├── speaker
    │   ├── navigator
    │   ├── pretrain
    │   ├── test
    │   └── id_paths.json
    ├── REVERIE
    │   ├── annotations
    │   │   └──pretrain
    │   ├── speaker
    │   └── features
    ├── SOON
    │   ├── annotations
    │   ├── speaker
    │   └── features
    ├── RxR
    ├── EnvEdit
    └── pretrained
        ├── METER
        └── roberta

    ```

### 2. Pre-training

To pre-train the model, navigate to the pre-training source directory and execute the provided shell script. Replace r2r with the desired dataset name as needed.
```pre-train
cd pretrain_src
bash run_r2r_goat.sh
```

### 3. Confounder Feature Extraction
1) **Extract BACL Features:**

    Navigate to the map navigation source directory and execute the scripts to extract BACL features. Refer to the [ducumentation](map_nav_src/do_utils/README.md) for more details.

    ``` BACL
    cd map_nav_src
    bash do_utils/extract_room_type.bash
    python do_intervention.py
    ```

2) **Extract FACL features:**

    Run the following script to extract FACL features, and store them in the respective `features` directory for each dataset.
    ``` cfp
    cd map_nav_src
    bash scripts/run_r2r_goat_CFPextract.sh
    ```

### 4. Fine-tuning
To fine-tune the model, use the command below:
``` fine-tune
cd map_nav_src
bash scripts/run_r2r.sh
```
Note that we have observed that the use of speaker coupled with causal intervention is critical.

### 5. Validation
For model validation, execute the following:
``` valid
cd map_nav_src
bash scripts/run_r2r_valid.sh
```

### 6. Additional Resources
1) Panoramic trajectory visualization is provided by [Speaker-Follower](https://gist.github.com/ronghanghu/d250f3a997135c667b114674fc12edae).
2) Top-down maps for Matterport3D are available in [NRNS](https://github.com/meera1hahn/NRNS).
3) Instructions for extracting image features from Matterport3D scenes can be found in [VLN-HAMT](https://github.com/cshizhe/VLN-HAMT).

We extend our gratitude to all the authors for their significant contributions and for sharing their resources.

### 7. TODO
- [ ] Clean the code for SOON.
- [x] Release the features and weights.


## Acknowledgements
This project builds upon the work found in [MP3DSim](https://github.com/peteanderson80/Matterport3DSimulator), [DUET](https://github.com/cshizhe/VLN-DUET), [EnvDrop](https://github.com/airsplay/R2R-EnvDrop), and [METER](https://github.com/zdou0830/METER). Some augmented datasets and features are from [PREVALENT](https://github.com/weituo12321/PREVALENT), [RxR-Marky](https://github.com/google-research-datasets/RxR/tree/main/marky-mT5), and [EnvEdit](https://github.com/jialuli-luka/EnvEdit).

We express our sincere thanks to these authors for their outstanding work and generosity in sharing their resources.

## BibTeX
If you find our work useful in your research, please consider citing:
```
@InProceedings{Wang2024GOAT,
    author    = {Wang, Liuyi and He, Zongtao and Dang, Ronghao and Shen, Mengjiao and Liu, Chengju and Chen, Qijun},
    title     = {Vision-and-Language Navigation via Causal Learning},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024}
}
```