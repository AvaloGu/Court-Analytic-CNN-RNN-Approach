# import cv2
import supervision as sv
from torchvision import transforms
from PIL import Image
import torch
from Models.ConvNeXt import ConvNeXt
from Models.Gru import EfficientGRUModel, GRUStage2
from Classifiers.Classifier import FrameClassifier, ShotPredictor
from Loaders.vocab import DURINGPOINT, OUTOFPOINT, ITOSSTAGE2
from Loaders.Dataloader import DataLoaderStage1
import pandas as pd
from itertools import groupby
import matplotlib.pyplot as plt
import gradio as gr
from prometheus_client import start_http_server, Summary, Counter, Gauge, CollectorRegistry  # Import Prometheus metrics

def gen_wrapper_with_last_flag(gen):
    # this wrapper touches the frame once, checking whether it is the last frame
    # or not before handing it to the enumerator
    prev = next(gen, None)
    
    if prev is None:
        return
    
    for item in gen:
        yield prev, False
        prev = item
    
    yield prev, True  # Only the last item from the generator will be True


def load_model(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    

def predictor(model, x, device, hidden=None):
    # model: FrameClassifier or ShotPredictor
    # x: (B, 3, 224, 224)

    x = x.to(device)

    with torch.no_grad():
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, _, hidden = model(x, h=hidden) # (num_examples, num_classes)

    pred = logits.argmax(dim = 1) # (B,)

    return pred, hidden


def find_idx(frame_label):
    # frame_label: (B,)
    for i in range(2, 6):
        frame_label[frame_label == i] = 1 # we'll set anything besides during_pt as out_of_pt

    label_original = frame_label[:-1]
    label_shift_by_1 = frame_label[1:]
    diff = label_shift_by_1 - label_original
    in_point_out_point_indices = (torch.where((diff==1) | (diff==-1)))[0]

    return in_point_out_point_indices


def get_stats(player_shots):
    # player_shots: a list of shots (str) of a player 
    df = pd.DataFrame(player_shots, columns=["shots"])
    stats = {}
    back_hand = []
    forehand = []
    serve_ad = []
    serve_deuce = []

    # backhand
    back_hand.append(df[df["shots"] == 'back_hand_cross'].shape[0])
    back_hand.append(df[df["shots"] == 'back_hand_mid'].shape[0])
    back_hand.append(df[df["shots"] == 'back_hand_line'].shape[0])
    stats["back_hand_cross", "back_hand_mid", "back_hand_line"] = back_hand


    # forhand
    forehand.append(df[df["shots"] == 'forehand_cross'].shape[0])
    forehand.append(df[df["shots"] == 'forehand_mid'].shape[0])
    forehand.append(df[df["shots"] == 'forehand_line'].shape[0])
    stats["forehand_cross", "forehand_mid", "forehand_line"] = forehand

    # serve ad
    serve_ad.append(df[df["shots"] == 'serve_ad_wide'].shape[0])
    serve_ad.append(df[df["shots"] == 'serve_ad_mid'].shape[0])
    serve_ad.append(df[df["shots"] == 'serve_ad_t'].shape[0])
    stats["serve_ad_wide", "serve_ad_mid", "serve_ad_t"] = serve_ad

    # serve deuce
    serve_deuce.append(df[df["shots"] == 'serve_deuce_wide'].shape[0])
    serve_deuce.append(df[df["shots"] == 'serve_deuce_mid'].shape[0])
    serve_deuce.append(df[df["shots"] == 'serve_deuce_t'].shape[0])
    stats["serve_deuce_wide", "serve_deuce_mid", "serve_deuce_t"] = serve_deuce

    return stats
    

def make_figures(df):
    all_point = [] # a list of lists of shots in a point
    for k, g in groupby(df["predictions"], lambda x: x=="z"):
        if not k:
            all_point.append(list(g))

    top_player_shots = []
    bot_player_shots = []

    for pt in all_point:
        starting_player = 'top'
        if pt[0].startswith('bot'):
            starting_player = 'bot'
        exist_serve_miss = 'serve_miss' in pt

        if starting_player == 'top' and not exist_serve_miss:
            top_player_shots += pt[1::2]
            bot_player_shots += pt[2::2]
        elif starting_player == 'top' and exist_serve_miss:
            top_player_shots += ['serve_miss']
            top_player_shots += pt[2::2]
            bot_player_shots += pt[3::2]
        elif starting_player == 'bot' and not exist_serve_miss:
            bot_player_shots += pt[1::2]
            top_player_shots += pt[2::2]
        else: # starting_player == 'bot' and exist_serve_miss:
            bot_player_shots += ['serve_miss']
            bot_player_shots += pt[2::2]
            top_player_shots += pt[3::2]

    players = {}
    players["top player"] = top_player_shots
    players["bot player"] = bot_player_shots

    shot_categories = ["back hand", "forehand", "serve ad", "serve deuce"]

    plt.figure(figsize=(15,5))
    for i, (player, shots) in enumerate(players.items()):
        stats = get_stats(shots)
        for j, (key, item) in enumerate(stats.items()):
            pos_global = i*4 + j
            plt.subplot(2, 4, pos_global + 1)
            if sum(item) != 0:
                key_cut = [k.split("_")[-1] for k in key]
                plt.pie(item, labels = key_cut, autopct='%1.1f%%', wedgeprops={"width": 0.4}) # textprops={'fontsize': 8})
            else: # no data
                plt.pie([1], labels = ["no data"], autopct='%1.1f%%', wedgeprops={"width": 0.4}) #textprops={'fontsize': 8})

            plt.title(f"{shot_categories[j]}")

        y_pos = 0.75 if i == 0 else 0.25
        plt.figtext(0.01, y_pos, player, fontsize=14, va='center')

    plt.suptitle("Shot Distribution by Player", fontsize=20)
    plt.tight_layout(rect=[0.1,0,1,0.95])
    plt.savefig("analysis.png")
    plt.close()
    return "analysis.png"

# Create a Prometheus Summary to track time spent processing each request
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

# Start Prometheus metrics server to expose metrics on port 8000
start_http_server(8000)

@REQUEST_TIME.time()  # Decorator to measure time spent in this function
def deploy(video):

    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conv1 = ConvNeXt()
    rnn1 = EfficientGRUModel()
    model_stage1 = FrameClassifier(conv1, rnn1)
    load_model(model_stage1, "model_stage1.pth", device_type)

    conv2 = ConvNeXt()
    rnn2 = GRUStage2()
    model_stage2 = ShotPredictor(conv2, rnn2)
    load_model(model_stage2, "model_stage2.pth", device_type)

    # video_info = sv.VideoInfo.from_video_path('val_vid.mov') # holds video info
    # print(video_info.fps)

    original_fps = 60
    target_fps = 4
    stride = original_fps // target_fps

    frame_generator = sv.get_video_frames_generator(video, stride=stride)

    img_process = transforms.Compose([transforms.Resize((224, 224)), 
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                           std=[0.229, 0.224, 0.225])])

    batch_size = 128
    buffer_stage1 = []
    buffer_stage2 = []
    predictions = [] # a list of (B,) torch tensors
    gru_hidden = None

    for frame_idx, (frame, is_last) in enumerate(gen_wrapper_with_last_flag(frame_generator)):
        img_rgb = frame[..., ::-1].copy()  # BGR to RGB
        img = Image.fromarray(img_rgb) # convert it to Image
        torch_img = img_process(img) # (3, 224, 224)

        buffer_stage1.append(torch_img)

        if (frame_idx != 0 and frame_idx % (batch_size - 1) == 0) or is_last:
            x = torch.stack(buffer_stage1, dim=0) # (128, 3, 224, 224)
            out, gru_hidden = predictor(model=model_stage1, x=x, device=device_type, hidden=gru_hidden) # (B,)

            # loader = DataLoaderStage1(mode="val")
            # x, y = loader.next_batch()
            # x, y = x.to(device_type), y.to(device_type)
            # out_loader, gru_hidden = predictor(model=model_stage1, x=x, device=device_type, hidden=gru_hidden) # (B,)
            
            in_out_indices = find_idx(out) # 1-D torch tensor

            start_is_during_pt = out[0].item() == DURINGPOINT 

            if len(in_out_indices) == 0 and start_is_during_pt:
                # the entire buffer contains during point frames
                buffer_stage2 += buffer_stage1

            elif len(in_out_indices) == 1:
                if start_is_during_pt: 
                    buffer_stage2 +=  buffer_stage1[:in_out_indices[0].item()]
                if buffer_stage2:
                    # if the buffer is non empty
                    frames = torch.stack(buffer_stage2, dim=0) # (B, 3, 224, 224)
                    predictions.append(predictor(model_stage2, frames, device_type)[0]) # (B,)
                    buffer_stage2 = [] # reset buffer
                if not start_is_during_pt:
                    buffer_stage2 +=  buffer_stage1[in_out_indices[0].item():]

            elif len(in_out_indices) > 1:
                if start_is_during_pt:
                    start_idx = 0
                for a, b, in zip(in_out_indices[:-1:2], in_out_indices[1::2]):
                    if start_is_during_pt:
                        end_idx = a.item()
                    else:
                        start_idx = a.item()
                        end_idx = b.item()

                    if buffer_stage2:
                        # buffer is not empty
                        if start_is_during_pt:
                            buffer_stage2 +=  buffer_stage1[:end_idx]

                        # we want to flush the buffer when it is not empty (carried from previous) 
                        # and the current buffer started with out of point frame
                        frames = torch.stack(buffer_stage2, dim=0) # (B, 3, 224, 224)
                        predictions.append(predictor(model_stage2, frames, device_type)[0]) # (B,)
                        buffer_stage2 = [] # reset buffer
                        start_idx = b.item()

                        if start_is_during_pt:
                            continue

                    if (end_idx - start_idx) > 2 or start_idx == 0: 
                        buffer_stage2 += buffer_stage1[start_idx:end_idx+1]
                    else: # else we consider it a noise
                        continue

                     # the point ends within this buffer
                    frames = torch.stack(buffer_stage2, dim=0) # (B, 3, 224, 224)
                    predictions.append(predictor(model_stage2, frames, device_type)[0]) # (B,)
                    buffer_stage2 = [] # reset buffer
                    
                    if start_is_during_pt:
                        start_idx = b.item()

                if len(in_out_indices) % 2 == 1:
                    if start_is_during_pt:
                        end_idx = in_out_indices[-1].item()
                        buffer_stage2 += buffer_stage1[start_idx:end_idx+1]

                        frames = torch.stack(buffer_stage2, dim=0) # (B, 3, 224, 224)
                        predictions.append(predictor(model_stage2, frames, device_type)[0]) # (B,)
                        buffer_stage2 = [] # reset buffer
                    else:
                        start_idx = in_out_indices[-1].item()
                        buffer_stage2 += buffer_stage1[start_idx:]

                if start_is_during_pt and len(in_out_indices) % 2 == 0:
                    buffer_stage2 += buffer_stage1[start_idx:]

            # print(predictions)
            buffer_stage1 = [] # reset buffer

    out = torch.cat(predictions, dim=0).cpu() # (num_of_examples,)
    out = out.numpy().astype(int)
    decode = [ITOSSTAGE2[i] for i in out]
    df = pd.DataFrame(decode, columns=["predictions"])
    # df.to_csv("out_tensor_stage2.csv", index=False)
    analysis_file = make_figures(df)

    return analysis_file, df


def collect_feedback(feedback_type, feedback_match, open_feedback):
    """
    Collect user feedback on the prediction.
    """
    # Truncate open feedback to 50 characters for use as a label
    feedback_excerpt = open_feedback[:50] if open_feedback else "No additional feedback"
    
    # Increment Prometheus counter with feedback details
    feedback_counter.labels(
        feedback_type=feedback_type,
        feedback_match = feedback_match,
        feedback_excerpt=feedback_excerpt
    ).inc()
    
    print(f"Feedback: {open_feedback} | Type: {feedback_type} for match {feedback_match}")
    return "Thank you for your feedback!"


with gr.Blocks() as app_interface:
    gr.Markdown("<h1 style='text-align: center;'>Court Analytic</h1>")

    video_in = gr.Video(label="Upload a match video")
    run_btn = gr.Button("Run analysis")

    analysis_img = gr.Image(type="filepath", label="Analysis")

    # hidden by default
    with gr.Accordion("Show Data", open=False):
        df_out = gr.Dataframe()

    # Clear button clears listed components
    clear_btn = gr.ClearButton(
        components=[video_in, analysis_img, df_out],
        value="Clear"
    )

    run_btn.click(fn=deploy, inputs=video_in, outputs=[analysis_img, df_out])


feedback_counter = Counter(
    'gradio_feedback_total', 
    'Total number of feedback entries received', 
    ['feedback_type', 'feedback_match', 'feedback_excerpt']
)

feedback_interface = gr.Interface(
    fn=collect_feedback,
    inputs=[
        gr.Textbox(label="Match analyzed", placeholder="Enter the match analyzed"),
        gr.Radio(choices=["Good", "Bad"], label="Feedback Type"),  # Radio button for Good/Bad
        gr.Textbox(label="Additional Feedback", placeholder="Provide additional feedback here")
    ],
    outputs="text",
    title="Feedback Collector",
    description="Provide feedback on the model's prediction."
)

gr.TabbedInterface([app_interface, feedback_interface], ["Predict", "Feedback"]).launch(server_name="0.0.0.0", server_port=7860)

# if __name__ == "__main__":
#     deploy()











    
