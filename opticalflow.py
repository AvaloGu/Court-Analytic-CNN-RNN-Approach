import cv2
import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm

def save_flow(video_id, flow, root):
    # flow: (2, 224, 224) np array, float32
    # Create a folder for the video
    os.makedirs(root, exist_ok=True)   # Safe: ignores if folder exists

    # Save the flows as .npy
    out_path = os.path.join(root, f"{video_id}")
    np.save(out_path, flow)


def process_frames(frames, crop, folder): 
    # frames: a list of len 2, two consecutive file names
    assert len(frames) == 2

    gray_frames = []
    for file in frames:
        frame = Image.open(os.path.join(folder, file)).convert("RGB") # PIL Image
        frame = crop(frame) # type PIL Image, croped 
        np_frames = np.array(frame)  # (224, 224, 3) np arrays, dtype uint8 [0, 255]

        # BGR expected by OpenCV; also convert to gray, 
        x_bgr = np_frames[:, :, ::-1] # RGB to BGR, last dim is the channel dim
        gray = cv2.cvtColor(x_bgr, cv2.COLOR_BGR2GRAY) # (224, 224)
        gray_frames.append(gray)

    return gray_frames[0], gray_frames[1]


def process_flow(mode="train"):
    if mode == "train":
        imgfolder = "Dataset/frames"
        save_path = "Dataset/frames_flow"
    elif mode == "val":
        imgfolder = "Dataset/val_set"
        save_path = "Dataset/val_set_flow"
    else: 
        imgfolder = "Dataset/test_set"
        save_path = "Dataset/val_set_flow"
    
    files = sorted([f for f in os.listdir(imgfolder) if f.endswith(".jpg")]) # a list of file names
    crop = transforms.Resize((224, 224))

    clamp = 50.0
    flow_algo = cv2.optflow.DualTVL1OpticalFlow_create() # flow algo, TV-L1
    
    for i in tqdm(range(len(files) - 1)):
        frames = files[i:i+2]
        frame1, frame2 = process_frames(frames=frames, crop=crop, folder=imgfolder) # (224, 224) grey frames
        
        flow = flow_algo.calc(frame1, frame2, None)  # (224,224,2) float32 np array, (dx,dy)
        # np.clip limit the values within an array to a specified range
        flow = np.clip(flow, -clamp, clamp) # (224,224,2)
        flow = flow.transpose(2, 0, 1) # (2, 224, 224)
        flow_normalized = flow / clamp # normalized in [-1, 1]

        video_id = f"flow_{files[i][:-4]}.npy" # -4 since we don't want .jpg extension
        save_flow(video_id=video_id, flow=flow_normalized, root=save_path)

    # the very last flow, we'll pad it with 0
    C, H, W = flow_normalized.shape
    last_flow_pad = np.zeros((C, H, W), dtype=np.float32) # (2, 224, 224)

    video_id = f"flow_{files[-1][:-4]}.npy"
    save_flow(video_id=video_id, flow=last_flow_pad, root=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process optical flow")
    parser.add_argument("mode", help="which dataset to process, train, val, or test")

    args = parser.parse_args()
    process_flow(args.mode)


    

     