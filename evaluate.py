import tqdm
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
from src.utils import get_vision_baseline_pred, get_vision_topline_pred, generate_mask_from_pred, compute_metrics, cal_CIOU
from src.data import UrbansasDataset
from src.model import RCGrad, FlowGradEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLOWGRAD_IC_PATH = "models/flowgrad_IC.ckpt"
FLOWGRAD_EN_PATH = "models/flowgrad_EN.ckpt"


def main():
    urbansas_dataset = UrbansasDataset(data_root = data_root)

    # get predictions
    if model == "flow":
        # optical flow as localisation maps
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            try:
                flow = np.array(Image.open(f"{data_root}Flow/{ft}.jpg").resize((224, 224)))
            except:
                continue
            preds.append((ft, flow, gt_map))

    elif model == "rcgrad":
        rc_grad = RCGrad()
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            preds.append((ft, rc_grad.pred_audio(img, audio), gt_map))
    
    
    elif model == "flowgrad-H":
        rc_grad = RCGrad()
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            flow_norm = None
            try:
                flow = np.array(Image.open(f"{data_root}Flow/{ft}.jpg").resize((224, 224)))
                flow_norm = (flow+5)/200
            except:
                continue
            pred = rc_grad.pred_audio(img, audio, flow_norm)
            pred*=flow
            preds.append((ft, pred, gt_map))

    elif model == "flowgrad-IC":
        checkpoint = torch.load(FLOWGRAD_IC_PATH, map_location=device)["state_dict"]
        rc_grad = RCGrad(modal="flow_IC", checkpoint = checkpoint)
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            flow_norm = None
            try:
                flow = np.array(Image.open(f"{data_root}Flow/{ft}.jpg").resize((224, 224)))
                flow_norm = (flow+5)/200
            except:
                continue
            pred = rc_grad.pred_audio(img, audio, flow_norm)
            preds.append((ft, pred, gt_map))
    
    elif model == "flowgrad-EN":
        checkpoint = torch.load(FLOWGRAD_EN_PATH, map_location=device)["state_dict"]
        flowgrad = FlowGradEN(checkpoint = checkpoint)
    
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            flow_norm = None
        
            try:
                flow = np.array(Image.open(f"{data_root}Flow/{ft}.jpg").resize((224, 224)))
                flow_norm = (flow+5)/200
            except:
                continue
            pred = flowgrad.pred_audio(img, audio, flow_norm)
            preds.append((ft, pred, gt_map))
    

    elif model == "yolo_topline":
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            pred = get_vision_topline_pred(ft)
            if pred is not None:
                pred_mask = generate_mask_from_pred(pred)
                preds.append((ft, pred_mask, gt_map))

            
    elif model == "yolo_baseline":
        preds = []
        for ft, img, audio, gt_map in tqdm.tqdm(urbansas_dataset):
            pred = get_vision_baseline_pred(ft)
            if pred is not None:
                pred_mask = generate_mask_from_pred(pred)
                preds.append((ft, pred_mask, gt_map))
            
    # save image-wise cIoU
    ious = [cal_CIOU(pred, gt_map)[0] for _, pred, gt_map in preds]
    filenames = [ft for ft, _, _ in preds]
    iou_df = pd.DataFrame()
    iou_df["filename"] = filenames
    iou_df["iou"] = ious
    os.makedirs(f"evaluation/", exist_ok=True)
    iou_df.to_csv(f"evaluation/{model}.csv", index=None)

    # compute metrics
    metrics = compute_metrics(preds)
    print(metrics)  



if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='evaluates ssl on urbansas')
    parser.add_argument(
        '-filtered', '--f', action='store_true',
        help='The filtered version of the dataset will be used if the argument is passed'
        )

    parser.add_argument(
        "-model", action = "store", default="rcgrad",
        help='Which model to use for inference? \
            Available options - rcgrad, flow, flowgrad-H, \
            flowgrad-IC, flowgrad-EN, yolo_topline, yolo_baseline'
        )

    filtered = parser.parse_args().f
    allowed_models = ['rcgrad', 'flow', 'flowgrad-H', 'flowgrad-IC', 'flowgrad-EN', 'yolo_topline', 'yolo_baseline']
    model = parser.parse_args().model
    if model not in allowed_models:
        raise Exception("Invalid Model! Available options - rcgrad, flow, flowgrad-H, flowgrad-IC, flowgrad-EN, yolo_topline, yolo_baseline")

    dataset = "urbansas"
    if filtered:
        dataset = "urbansas_filtered"
    data_root = f"data/{dataset}/"

    print(f"Using model - {model}")
    print(f"Dataset - {dataset}")

    # setup evaluation directory
    if not os.path.isdir("evaluation/"):
        os.mkdir("evaluation/")

    main()

