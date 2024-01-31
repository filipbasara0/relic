import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import multiprocessing
import random
import numpy as np
from tqdm.auto import tqdm
from torchinfo import summary

from relic import ReLIC, relic_loss

from relic.utils import accuracy, get_dataset, get_encoder
from relic.stl10_eval import STL10Eval

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

    modify_model = True if "cifar" in args.dataset_name else False
    encoder = get_encoder(args.encoder_model_name, modify_model)
    relic_model = ReLIC(encoder,
                        mlp_out_dim=args.proj_out_dim,
                        mlp_hidden=args.proj_hidden_dim)

    if args.ckpt_path:
        model_state = torch.load(args.ckpt_path)
        relic_model.load_state_dict(model_state)
    relic_model = relic_model.to(device)

    summary(relic_model, input_size=[(1, 3, 32, 32), (1, 3, 32, 32)])

    params = list(relic_model.online_encoder.parameters())
    if not args.use_siglip:
        params += [relic_model.t_prime]
    optimizer = torch.optim.Adam(params,
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    ds = get_dataset(args)
    train_loader = DataLoader(ds,
                              batch_size=args.batch_size,
                              num_workers=multiprocessing.cpu_count() - 8,
                              drop_last=True,
                              pin_memory=True,
                              shuffle=True)

    scaler = GradScaler(enabled=args.fp16_precision)

    stl10_eval = STL10Eval()
    total_num_steps = (len(train_loader) *
                       (args.num_epochs + 2)) - args.update_gamma_after_step
    gamma = args.gamma
    global_step = 0
    total_loss = 0.0
    n_global, n_local = args.num_global_views, args.num_local_views
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        progress_bar = tqdm(train_loader,
                            desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, (views, _) in enumerate(progress_bar):
            views = [v.to(device) for v in views]
            global_views = views[:n_global]
            local_views = views[:n_local]

            with autocast(enabled=args.fp16_precision):
                projections_online = []
                projections_target = []
                for view in global_views:
                    projections_online.append(relic_model.get_online_pred(view))
                    projections_target.append(relic_model.get_target_pred(view))
                for view in local_views:
                    projections_online.append(relic_model.get_online_pred(view))
                loss = 0
                # invariance_loss used only for debug
                invariance_loss = 0
                scale = 0
                for i_t, target_pred in enumerate(projections_target):
                    for i_o, online_pred in enumerate(projections_online):
                        if i_t != i_o:
                            relic_loss_, invar_loss = relic_loss(online_pred, target_pred,
                                                                relic_model.t_prime, args.alpha,
                                                                use_siglip=args.use_siglip)
                            loss += relic_loss_
                            invariance_loss += invar_loss
                            scale += 1

                loss /= scale
                invariance_loss /= scale

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

            epoch_kl_loss += invariance_loss.item()
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
                f"Temp: {relic_model.t_prime.exp().item():.3f} |"
                f"Lr: {current_lr:.6f}")

            global_step += 1
            if global_step % args.log_every_n_steps == 0:
                with torch.no_grad():
                    x, x_prime = projections_online[0], projections_target[1]
                    x, x_prime = F.normalize(x, p=2, dim=-1), F.normalize(x_prime, p=2, dim=-1)
                    logits = torch.mm(x, x_prime.t()) * relic_model.t_prime.exp()
                labels = torch.arange(logits.size(0)).to(logits.device)
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                print("#" * 100)
                print('acc/top1 logits1', top1[0].item())
                print('acc/top5 logits1', top5[0].item())
                print("#" * 100)

                torch.save(relic_model.state_dict(),
                           f"{args.save_model_dir}/relic_model.pth")
                relic_model.save_encoder(f"{args.save_model_dir}/encoder.pth")

            if global_step % (args.log_every_n_steps * 5) == 0:
                stl10_eval.evaluate(relic_model)
                print("!" * 100)
