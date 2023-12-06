import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import multiprocessing
import random
import numpy as np
from tqdm.auto import tqdm
from torchinfo import summary

from relic import ReLIC, relic_loss

from relic.utils import accuracy, get_dataset, get_encoder

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


# cosine EMA schedule (increase from tau_base to one) as defined in https://arxiv.org/abs/2010.07922
# k -> current training step, K -> maximum number of training steps
def update_gamma(k, K, tau_base):
    k = torch.tensor(k, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)

    tau = 1 - (1 - tau_base) * (torch.cos(torch.pi * k / K) + 1) / 2
    return tau.item()


def train_relic(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = get_encoder(args.encoder_model_name)
    relic_model = ReLIC(encoder,
                        mlp_out_dim=args.proj_out_dim,
                        mlp_hidden=args.proj_hidden_dim)

    if args.ckpt_path:
        model_state = torch.load(args.ckpt_path)
        relic_model.load_state_dict(model_state)
    relic_model = relic_model.to(device)

    summary(relic_model, input_size=[(1, 3, 32, 32), (1, 3, 32, 32)])

    optimizer = torch.optim.Adam(list(relic_model.online_encoder.parameters()),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    ds = get_dataset(args.dataset_name, args.dataset_path)
    train_loader = DataLoader(ds,
                              batch_size=args.batch_size,
                              num_workers=multiprocessing.cpu_count() - 4,
                              drop_last=True,
                              pin_memory=True,
                              shuffle=True)

    scaler = GradScaler(enabled=args.fp16_precision)

    total_num_steps = (len(train_loader) *
                       (args.num_epochs + 2)) - args.update_gamma_after_step
    gamma = args.gamma
    global_step = 0
    total_loss = 0.0
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        progress_bar = tqdm(train_loader,
                            desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, (images, _) in enumerate(progress_bar):
            x1, x2 = images
            x1 = x1.to(device)
            x2 = x2.to(device)

            with autocast(enabled=args.fp16_precision):
                o1, o2, t1, t2 = relic_model(x1, x2)
                loss1, logits_1, labels, invariance_loss1 = relic_loss(
                    o1, t2, args.tau, args.alpha)
                loss2, logits_2, labels, invariance_loss2 = relic_loss(
                    o2, t1, args.tau, args.alpha)
                loss = (loss1 + loss2) / 2

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if global_step > args.update_gamma_after_step and global_step % args.update_gamma_every_n_steps == 0:
                relic_model.update_params(gamma)
                gamma = update_gamma(global_step, total_num_steps, args.gamma)

            if global_step <= args.update_gamma_after_step:
                relic_model.copy_params()

            total_loss += loss.item()
            epoch_loss += loss.item()
            avg_loss = total_loss / (global_step + 1)
            ep_loss = epoch_loss / (step + 1)

            epoch_kl_loss += (invariance_loss1.item() +
                              invariance_loss2.item()) / 2
            ep_kl_loss = epoch_kl_loss / (step + 1)

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_description(
                f"Epoch {epoch+1}/{args.num_epochs} | "
                f"Step {global_step+1} | "
                f"Epoch Loss: {ep_loss:.4f} |"
                f"Total Loss: {avg_loss:.4f} |"
                f"KL Loss: {ep_kl_loss:.6f} |"
                f"Gamma: {gamma:.6f} |"
                f"Alpha: {args.alpha:.3f} |"
                f"Tau: {args.tau:.3f} |"
                f"Lr: {current_lr:.6f}")

            global_step += 1
            if global_step % args.log_every_n_steps == 0:
                top1, top5 = accuracy(logits_1, labels, topk=(1, 5))
                print('acc/top1 logits1', top1[0].item())
                print('acc/top5 logits1', top5[0].item())
                top1, top5 = accuracy(logits_2, labels, topk=(1, 5))
                print('acc/top1 logits2', top1[0].item())
                print('acc/top5 logits2', top5[0].item())
                print("#" * 100)

                torch.save(relic_model.state_dict(),
                           f"{args.save_model_dir}/relic_model.pth")
                relic_model.save_encoder(f"{args.save_model_dir}/encoder.pth")
