import time
from Models.ConvNeXt import ConvNeXt
from Models.Gru import EfficientGRUModel, GRUStage2
from Models.ResNet_RS import ResNet_RS
from Classifiers.Classifier import FrameClassifier, ShotPredictor
import torch
import os
import math
from Loaders.Dataloader import DataLoaderStage0, DataLoaderStage1, DataLoaderStage2
from timm.utils import ModelEma
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from evaluation import eval

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def train(stage, steps):
    """ stage: "stage0", "stage1", "stage2"
        steps: max_steps
    """
    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    print(f"using ddp: {ddp}")

    if stage == "stage1":
        ddp = False # current stage1 implementation does not support ddp

    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK']) # rank globally, 0 to world_size-1
        ddp_local_rank = int(os.environ['LOCAL_RANK']) # rank on this node, 0 to num_gpu_per_node-1
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}' # tells the process which GPU to use
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     device = "mps"
        print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(6216)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(6216)

    # print("I am GPUv ", ddp_rank)
    # import sys; sys.exit(0)

    if stage == "stage0":
        train_loader = DataLoaderStage0(process_rank=ddp_rank, num_processes=ddp_world_size)
    elif stage == "stage1":
        train_loader = DataLoaderStage1(process_rank=ddp_rank, num_processes=ddp_world_size)
    else: # stage2
        train_loader = DataLoaderStage2(process_rank=ddp_rank, num_processes=ddp_world_size) 


    # use tensor float32 precision for all matrix multiplications
    # faster than the normal float32 matmul. But all variables (tensors) are still stored in float32
    torch.set_float32_matmul_precision('high') 

    if stage == "stage0":
        ConvNet = ConvNeXt(stage0=True)
    else:
        ConvNet = ConvNeXt()

    # ConvNet = ResNet_RS()
    # rnn = EfficientGRUModel(input_size=2048)

    if stage ==  "stage2": # must run stage1 before stage2
        rnn = GRUStage2()
        model = ShotPredictor(ConvNet, rnn)
        # transfer learning
        state_dict_convnet = torch.load("convnet_stage1.pth", map_location="cpu")
        state_dict_gru = torch.load("gru_stage1.pth", map_location="cpu")
        model.convnet.load_state_dict(state_dict_convnet) # load_state_dict matches key and tensor shape
        model.rnn.encoder.gru.load_state_dict(state_dict_gru)
    elif stage == "stage1":
        rnn = EfficientGRUModel()
        model = FrameClassifier(ConvNet, rnn)
        state_dict_convnet = torch.load("convnet_stage0.pth", map_location="cpu")
        model.convnet.load_state_dict(state_dict_convnet, strict=False)

        for param in model.convnet.parameters():
            param.requires_grad = False  # freeze convolutional base
            
    else: # stage0
        model = FrameClassifier(ConvNet, stage0=True)

    model.to(device_type)

    # TODO: turn it off for debugging and tunning
    model = torch.compile(model) # compiler for Neural networks, compile the model to make it faster

    optimizer = model.configure_optimizers(weight_decay = 0.0, learning_rate = 4e-3, device_type = device_type, master_process = master_process)

    # create EMA wrapper, use it for validation
    model_ema = ModelEma(model, decay=0.999, device='cpu')

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank]) # wrap the model into DDP container
    # raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # TODO
    max_lr = 4e-4
    min_lr = max_lr * 0.1
    warmup_steps = int(0.06 * steps)
    max_steps = steps

    def get_lr(iter):
        # 1) linear warmup for warmup_iters steps
        if iter < warmup_steps:
            return max_lr * (iter+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if iter > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    model.train()

    loss_plot = []
    val_plot = []

    scaler = torch.amp.GradScaler("cuda") # initialize once before training

    # torch.autograd.set_detect_anomaly(True) # track every operation in the backward graph, debug backward passs

    last_step = max_steps - 1

    for step in range(max_steps):
        t0 = time.time()

        # evaluate validation loss per 1000 steps
        if (step % 1000 == 0 or step == last_step) and master_process:
            model.eval()
            # with torch.no_grad() toggled in eval 
            accuracy, _ = eval(stage=stage, model=model)
            val_plot.append(accuracy)
            model.train()

        optimizer.zero_grad()
        loss_accum = 0.0

        x, y, f = train_loader.next_batch() # (B, 3, 224, 224), (Num of shots in this point,), # (B, 8, 224, 224)
        x, y, f = x.to(device), y.to(device), f.to(device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss, _ = model(x, y, f)

        loss_accum += loss.detach()
        
        if ddp: 
            # average the loss across all processes for reporting
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) 

            num_examples = train_loader.num_examples
            total_num_examples = torch.tensor(num_examples, device=device_type)
            dist.all_reduce(total_num_examples, op=dist.ReduceOp.SUM) 
            # scaling so we get the right gradient when all reduce avg happens 
            # during .backward()
            loss *= ddp_world_size * num_examples / total_num_examples

        # scaled loss to prevent underflow/overflow
        # protects gradients while they're computed
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # gradient clipping, clip the global norm of the gradients at 1.0, prevent exploding gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        lr = get_lr(step) 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        scaler.step(optimizer)
        scaler.update()  # adjusts scale up/down depending on underflow/overflow

        raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
        model_ema.update(raw_model)

        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work, measure time accurately

        t1 = time.time()
        dt = t1 - t0 # time differenece in seconds
        if stage == "stage2":
            examples_processed = train_loader.num_examples * ddp_world_size
        else:
            examples_processed = train_loader.B * ddp_world_size
        examples_per_sec = examples_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | examples: {examples_processed:5d} | examples/sec: {examples_per_sec:.2f}")
            # with open(log_file, "a") as f:
            #     f.write(f"{step} train {loss_accum.item():.6f}\n")
            loss_plot.append(loss_accum.item())

    if master_process:
        loss_plot = torch.tensor(loss_plot).view(-1, 50).mean(1) # (20,)
        loss_plot = loss_plot.numpy()
        df_loss = pd.DataFrame(loss_plot)
        df_val_acc = pd.DataFrame(val_plot)
        
        
        if stage == "stage0":
            df_loss.to_csv("loss_plot_stage0.csv", index=False)
            df_val_acc.to_csv("val_plot_stag0.csv", index=False)
        elif stage == "stage1":
            df_loss.to_csv("loss_plot_stage1.csv", index=False)
            df_val_acc.to_csv("val_plot_stage1.csv", index=False)
        else: #stage2
            df_loss.to_csv("loss_plot_stage2.csv", index=False) 
            df_val_acc.to_csv("val_plot_stage2.csv", index=False)

        if stage == "stage0":
            torch.save(model_ema.ema.convnet.state_dict(), "convnet_stage0.pth")
            torch.save(model_ema.ema.state_dict(), "model_stage0.pth")
        elif stage == "stage1":
            torch.save(model_ema.ema.convnet.state_dict(), "convnet_stage1.pth")
            torch.save(model_ema.ema.rnn.gru.state_dict(), "gru_stage1.pth") # save just the gru from the model
            torch.save(model_ema.ema.state_dict(), "model_stage1.pth")
        else: # stage2
            torch.save(model_ema.ema.state_dict(), "model_stage2.pth")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("stage", help="which training stage to run, stage0 or stage1 or stage2")
    parser.add_argument("--steps", type=int, default=10, help="number of iteration steps to run")

    args = parser.parse_args()
    train(args.stage, args.steps)






