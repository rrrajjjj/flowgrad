# Flowgrad - Using Motion for Visual Sound Source Localization

## Official Implementation for the [Paper](https://arxiv.org/abs/2211.08367)

Recent work in visual sound source localization relies on semantic audio-visual representations learned in a self-supervised manner, and by design excludes temporal information present in videos. While it proves to be effective for widely used benchmark datasets, the method falls short for challenging scenarios like urban traffic. This work introduces temporal context into the state-of-the-art methods for sound source localization in urban scenes using optical flow as a means to encode motion information. 
# Inference on a Video
https://user-images.githubusercontent.com/70520320/210113500-9ed320d7-b9e9-4158-8052-112c5d220fac.mp4

# Setup
## Install requirements
    pip install -r requirements.txt

# Evaluation on Urbansas 
We've used the [Urban Sound and Sight](https://ieeexplore.ieee.org/document/9747644) dataset to train and evaluate our models. Evaluation can be run on urbansas as follows -

1. Prepare data for evaluation


        python src/data/prepare_data.py -d PATH_TO_URBANSAS
        python src/data/calc_flow.py -d PATH_TO_URBANSAS
2. Run evaluation - Several different models can be used for evaluation and the model can be passed as an argument. 

        python evaluate.py -m MODEL
The following model choices are available - 
1. **rcgrad** - Pretrained model from **How to Listen? Rethinking Visual Sound Source Localization** [(Paper)](https://arxiv.org/abs/2204.05156) [(Repo)](https://github.com/hohsiangwu/rethinking-visual-sound-localization)
2. **flow** - optical flow used as localization maps
3. **flowgrad-H, flowgrad-EN, flowgrad-IC** - variants of flowgrad (refer to paper for details)
4. **yolo_baseline, yolo_topline** - vision only object detection models used as baselines. The topline includes motion based filtering (stationary objects are discarded). 

# Performance 
| **Model**               | **IoU (τ = 0.5)** | **AUC** |
|-----------------------------|-------------------|---------|
| Vision-only+CF+TF (topline) | 0.68              | 0.51    |
| Optical flow (baseline)     | 0.33              | 0.23    |
| RCGrad                      | 0.16              | 0.13    |
| FlowGrad-H                  | 0.50              | 0.30    |
| FlowGrad-IC                 | 0.26              | 0.18    |
| FlowGrad-EN                 | 0.37              | 0.23    |

#
![main_result](https://user-images.githubusercontent.com/70520320/210113600-7425f095-bfa1-4a71-a317-de12f141287d.jpg)

# Upcoming! 
1. Inference on custom videos 
2. Training 
3. Support for other benchmark datasets for Visual Sound Source Localization

If you have any ideas or feature requests, please feel free to raise an issue!

## Credits
This work is built upon and borrows heavily from [hohsiangwu/rethinking-visual-sound-localization](https://github.com/hohsiangwu/rethinking-visual-sound-localization)


