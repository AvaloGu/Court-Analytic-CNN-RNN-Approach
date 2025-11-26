from Models.ResNet_RS import ResNet_RS
from Models.ConvNeXt import ConvNeXt
from Models.Gru import EfficientGRUModel, GRUStage2
from Classifiers.Classifier import FrameClassifier, ShotPredictor
from Loaders.Dataloader import DataLoaderStage1, DataLoaderStage2, DataLoader
import torch
import numpy as np
import pandas as pd
import argparse
from Loaders.vocab import POINTSEPARATOR, ITOSSTAGE1, ITOSSTAGE2


def load_model(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()


def predictor(model, loader, stage, device):
    prediction = []
    gru_hidden = None # currently only used by stage1

    while loader.not_empty:
        x, _ = loader.next_batch()
        x = x.to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _, gru_hidden = model(x, h=gru_hidden) # (num_examples, num_classes)

        pred = logits.argmax(dim = 1) # (B,)
        prediction.append(pred)

        if stage == "stage2": # current implementation is tearcher forcing accuracy for stage2
            point_token = torch.tensor([POINTSEPARATOR], dtype=torch.long, device=pred.device) # point token z
            prediction.append(point_token)

    out = torch.cat(prediction, dim=0).cpu() # (total_num_of_examples,)
    return out


def run_inference():
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conv1 = ConvNeXt()
    rnn1 = EfficientGRUModel()
    model_stage1 = FrameClassifier(conv1, rnn1)

    img_folder = "Dataset/test_set"

    load_model(model_stage1, "model_stage1.pth", device_type)
    loader_stage1 = DataLoaderStage1(mode="infer", infer_folder=img_folder) 

    stage1_result = predictor(model_stage1, loader_stage1, "stage1", device_type) # (total_num_of_frames,)

    conv2 = ConvNeXt()
    rnn2 = GRUStage2()
    model_stage2 = ShotPredictor(conv2, rnn2)

    load_model(model_stage2, "model_stage2.pth", device_type)
    loader_stage2 = DataLoaderStage2(mode="infer", infer_folder=img_folder, stage1_labels=stage1_result) 

    stage2_result = predictor(model_stage1, loader_stage2, "stage2", device_type) # (num_shots,)
    out = stage2_result.numpy().astype(int)
    decode = [ITOSSTAGE2[i] for i in out]

    df = pd.DataFrame(decode, columns=["predictions"])
    df.to_csv("infer_result.csv", index=False)

    
if __name__ == "__main__":
    run_inference()

