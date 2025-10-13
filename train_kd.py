import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from tqdm import tqdm
from DDRNet import DDRNet
from kd_losses.DCSFKDLoss import DCSFKDLoss
from kd_losses.OutputKDLoss import OutputKDLoss
from DDRNet_39 import *
from functions import *
from pathlib import Path
from tensorboardX import SummaryWriter
import math
import numpy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_parameters(model):
    print("Calculating total model parameters...")

    # 전체 모델 파라미터 수
    total_params = count_parameters(model)
    print(f"Total parameters in the model: {total_params}")

def _update_confmat(confmat, preds, targets, num_classes, ignore_index=255):
    # preds: (B,H,W) argmax 결과, targets: (B,H,W)
    valid = (targets != ignore_index)
    if not valid.any():
        return confmat

    t = targets[valid].view(-1)
    p = preds[valid].view(-1)
    k = t * num_classes + p  # (t,p) 쌍을 1D 인덱스로
    hist = torch.bincount(k, minlength=num_classes * num_classes)
    hist = hist.view(num_classes, num_classes).to(confmat.device)
    confmat += hist.to(dtype=confmat.dtype)
    return confmat


def compute_miou_from_confmat(confmat):
    """
    evaluation.py의 compute_miou와 동일한 정의:
    IoU(cls) = TP / (TP + FP + FN), 분모=0이면 NaN, mIoU는 NaN 무시 평균
    """
    confmat = confmat.to(torch.float64)  # 안전한 정밀도
    TP = torch.diag(confmat)             # (K,)
    FP = confmat.sum(0) - TP             # 예측이 cls인데 정답 아님
    FN = confmat.sum(1) - TP             # 정답이 cls인데 예측 아님
    denom = TP + FP + FN

    ious = torch.where(denom > 0, TP / denom.clamp(min=1), torch.full_like(TP, float('nan')))
    miou = torch.nanmean(ious)
    iou_list = [float(v) if not torch.isnan(v) else float('nan') for v in ious]
    return float(miou), iou_list


def compute_pixel_accuracy_from_confmat(confmat):
    total = confmat.sum().clamp(min=1)
    correct = torch.trace(confmat)
    return float((correct / total).item())


def _seed_worker(worker_id):
    import random, numpy as np, torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def train(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    AUG_NAMES = ["haze", "rain", "raindrop", "low_light", "overbright"]
    name_to_idx = {n: i for i, n in enumerate(AUG_NAMES)}

    # -------------------- Dataset & Dataloader --------------------
    train_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'train', args.scale_range,
                                        val_resize_size=(1080, 1920),
                                        normal_aug_prob=args.normal_aug_prob,
                                        severity_range=(args.severity_min, args.severity_max),
                                        )
    display_dataset_info(args.dataset_dir, train_dataset)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank,
                                       drop_last=True, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=16, pin_memory=True, worker_init_fn=_seed_worker, collate_fn=collate_with_meta)

    val_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'val', args.scale_range, val_resize_size=(1080, 1920))
    display_dataset_info(args.dataset_dir, val_dataset)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank,
                                     drop_last=False, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=max(1, args.batch_size//2), sampler=val_sampler,
                            num_workers=16, pin_memory=True, worker_init_fn=_seed_worker, collate_fn=collate_with_meta)

    # Model
    print(f"[GPU {local_rank }] Before model setup")
    # model = DDRNet(num_classes=args.num_classes).to(device)

    teacher_model = DualResNet(BasicBlock, [3, 4, 6, 3], num_classes=19, planes=64, spp_planes=128, head_planes=256, augment=False).to(device)
    print_model_parameters(teacher_model)
    if args.teacher_loadpath is not None:
        map_location = {f'cuda:{0}': f'cuda:{local_rank}'}
        ckpt = torch.load(args.teacher_loadpath, map_location=map_location)
        teacher_model.load_state_dict(ckpt, strict=False)

    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()  # 꼭 eval 모드로 설정

    model = DDRNet(num_classes=args.num_classes).to(device)
    print_model_parameters(model)
    model = DDP(model, device_ids=[local_rank])
    print(f"[GPU {local_rank }] DDP initialized")

    # Loss, Optimizer, Scheduler
    criterion = OhemCrossEntropy(ignore_label=255)
    criterion_kd = DCSFKDLoss(loss_weight=1.0).to(device)
    criterion_output_kd = OutputKDLoss(temperature=4.0, loss_weight=0.1, ignore_index=255)

    params = list(model.parameters()) + list(criterion_kd.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs=args.epochs, warmup_epochs=10, eta_min=1e-5)
    # scheduler = WarmupPolyEpochLR(optimizer, total_epochs=args.epochs, warmup_epochs=5, warmup_ratio=5e-4)

    if args.loadpath is not None:
        map_location = {f'cuda:{0}': f'cuda:{local_rank}'}
        state = torch.load(args.loadpath, map_location=map_location)
        # weights-only(모델 state_dict)인 경우
        try:
            load_state_dict(model, state)  # 네가 이미 쓰는 헬퍼
            print(f"[Rank {local_rank}] Loaded weights-only from {args.loadpath}")
        except Exception:
            # 혹시 모를 키 구조 차이 대비
            (model.module if isinstance(model, DDP) else model).load_state_dict(state, strict=False)
            print(f"[Rank {local_rank}] Loaded weights-only (strict=False)")

    # -------------------- Logging/TensorBoard --------------------
    writer = None
    if local_rank == 0:
        os.makedirs(args.result_dir, exist_ok=True)
        log_path = os.path.join(args.result_dir, "log.txt")
        with open(log_path, 'w') as f:
            f.write("Epoch\t\tTrain-loss\t\tlearningRate\n")
        writer = SummaryWriter(log_dir=os.path.join(args.result_dir, "tb"))

    def _get_state_dict(m):
        return m.module.state_dict() if isinstance(m, DDP) else m.state_dict()

    best_miou = float("-inf")
    eps = 1e-6
    start_epoch = args.start_epoch

    for _ in range(start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        criterion_kd.train()

        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        ce_loss_sum = 0.0
        kd_loss_sum = 0.0
        kd_output_loss_sum = 0.0

        num_steps = 0  # 에폭 내 배치 수

        aug_counts_local = torch.zeros(len(AUG_NAMES), device=device, dtype=torch.long)

        if local_rank == 0:
            loop = tqdm(train_loader, desc=f"[GPU {local_rank}] Epoch [{epoch + 1}/{args.epochs}]", ncols=110)
        else:
            loop = train_loader

        for i, (imgs, labels, metas) in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs, features = model(imgs)
            with torch.no_grad():
                outputs_t, features_t = teacher_model(imgs)  # teacher forward (eval)

            # print("Student feature shape:", features.shape)
            # print("Teacher feature shape:", features_t.shape)

            loss_ce = criterion(outputs, labels)
            loss_kd_feat = criterion_kd(features, features_t)  # 기존 feature KD
            loss_kd_output = criterion_output_kd(outputs, outputs_t, labels=labels)  # 새로 추가된 output KD

            ce_loss_sum += loss_ce.item()
            kd_loss_sum += loss_kd_feat.item()
            kd_output_loss_sum += loss_kd_output.item()

            loss = loss_ce + loss_kd_feat + loss_kd_output
            loss.backward()
            optimizer.step()

            # === 증강 카운트 집계 ===
            if isinstance(metas, (list, tuple)):  # ✅ collate_with_meta 경로
                for m in metas:
                    for (name, sev) in m.get("applied", []):
                        idx = name_to_idx.get(name)
                        if idx is not None:
                            aug_counts_local[idx] += 1
            elif isinstance(metas, dict):  # (혹시 다른 collate를 쓰는 경우 호환)
                for applied in metas.get("applied", []):
                    for (name, sev) in applied:
                        idx = name_to_idx.get(name)
                        if idx is not None:
                            aug_counts_local[idx] += 1

            # torch.cuda.synchronize()

            total_loss += loss.item()

            num_steps += 1

            if local_rank == 0:
                loop.set_postfix(loss=loss.item(),
                                 avg_loss=total_loss / max(1, num_steps),
                                 lr=scheduler.get_last_lr()[0])

        torch.cuda.empty_cache()
        dist.barrier()
        scheduler.step()

        # ------ Train epoch 평균(DDP 전체 평균으로 산출) ------
        ce_loss_sum_t = torch.tensor([ce_loss_sum], device=device)
        kd_loss_sum_t = torch.tensor([kd_loss_sum], device=device)
        kd_output_loss_sum_t = torch.tensor([kd_output_loss_sum], device=device)
        dist.all_reduce(ce_loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(kd_loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(kd_output_loss_sum_t, op=dist.ReduceOp.SUM)

        train_loss_sum = torch.tensor([total_loss], device=device)
        train_step_sum = torch.tensor([num_steps], device=device, dtype=torch.float32)
        dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_step_sum, op=dist.ReduceOp.SUM)

        train_ce_loss_epoch = (ce_loss_sum_t / train_step_sum).item()
        train_kd_loss_epoch = (kd_loss_sum_t / train_step_sum).item()
        train_kd_output_loss_epoch = (kd_output_loss_sum_t / train_step_sum).item()
        train_loss_epoch = (train_loss_sum / train_step_sum).item()

        # ===== Validation =====
        model.eval()
        criterion_kd.eval()

        val_ce_loss_sum = torch.tensor([0.0], device=device)
        val_kd_loss_sum = torch.tensor([0.0], device=device)
        val_kd_output_loss_sum = torch.tensor([0.0], device=device)
        val_loss_sum = torch.tensor([0.0], device=device)
        val_batches = torch.tensor([0.0], device=device)

        # evaluation.py 의미와 맞추기 위해 int64 혼동행렬 사용
        confmat = torch.zeros((args.num_classes, args.num_classes), device=device, dtype=torch.int64)

        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"[GPU {local_rank}] Validate",
                            ncols=110) if local_rank == 0 else val_loader
            for imgs, labels, metas in val_iter:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                logits, features = model(imgs)
                outputs_t, features_t = teacher_model(imgs)

                vloss_ce = criterion(logits, labels)
                vloss_kd = criterion_kd(features, features_t)
                vloss_kd_output = criterion_output_kd(logits, outputs_t, labels=labels)

                val_ce_loss_sum += vloss_ce.item()
                val_kd_loss_sum += vloss_kd.item()
                val_kd_output_loss_sum += vloss_kd_output.item()

                vloss = vloss_ce + vloss_kd + vloss_kd_output

                val_loss_sum += vloss.detach()
                val_batches += 1.0

                preds = torch.argmax(logits, dim=1)
                confmat = _update_confmat(confmat, preds, labels, args.num_classes, ignore_index=255)

        # ---- DDP 집계 ----
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_kd_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_ce_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_kd_output_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches, op=dist.ReduceOp.SUM)
        dist.all_reduce(confmat, op=dist.ReduceOp.SUM)

        val_loss_epoch = (val_loss_sum / val_batches).item()
        val_ce_loss_epoch = (val_ce_loss_sum / val_batches).item()
        val_kd_loss_epoch = (val_kd_loss_sum / val_batches).item()
        val_kd_output_loss_epoch = (val_kd_output_loss_sum / val_batches).item()
        miou, iou_list = compute_miou_from_confmat(confmat)
        acc = compute_pixel_accuracy_from_confmat(confmat)

        aug_counts = aug_counts_local.clone()
        dist.all_reduce(aug_counts, op=dist.ReduceOp.SUM)

        # ===== Logging / Checkpoint on rank0 =====
        if local_rank == 0:
            lr = scheduler.get_last_lr()
            lr = sum(lr) / len(lr)

            counts_str = ", ".join(f"{n}:{int(aug_counts[i].item())}" for i, n in enumerate(AUG_NAMES))
            print(f"[Epoch {epoch + 1}] Aug Applied Counts -> {counts_str}")

            # ⬇️ 에폭당 한 번만 기록
            if writer is not None:
                writer.add_scalar("train/loss_ce", train_ce_loss_epoch, epoch + 1)
                writer.add_scalar("train/loss_kd", train_kd_loss_epoch, epoch + 1)
                writer.add_scalar("train/loss_kd_output", train_kd_output_loss_epoch, epoch + 1)
                writer.add_scalar("train/loss_total", train_loss_epoch, epoch + 1)
                writer.add_scalar("val/loss_ce", val_ce_loss_epoch, epoch + 1)
                writer.add_scalar("val/loss_kd", val_kd_loss_epoch, epoch + 1)
                writer.add_scalar("val/loss_kd_output", val_kd_output_loss_epoch, epoch + 1)
                writer.add_scalar("val/loss_total", val_loss_epoch, epoch + 1)
                writer.add_scalar("val/mIoU", miou, epoch + 1)
                writer.add_scalar("val/Acc", acc, epoch + 1)
                writer.add_scalar("train/lr_epoch", lr, epoch + 1)

                for c, iou_c in enumerate(iou_list):
                    if not math.isnan(iou_c):
                        writer.add_scalar(f"val/IoU_cls/{c}", iou_c, epoch + 1)

                for i, n in enumerate(AUG_NAMES):
                    writer.add_scalar(f"aug/count/{n}", int(aug_counts[i].item()), epoch + 1)

            with open(log_path, "a") as f:
                f.write("\n%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.8f" %
                        (epoch + 1, train_loss_epoch, train_kd_output_loss_epoch,
                         val_loss_epoch, val_kd_output_loss_epoch, miou, acc, lr))

            # 베스트(Val Loss 기준)
            if (miou > best_miou + eps) or (abs(miou - best_miou) <= eps and (epoch + 1) > 0):
                best_miou = miou
                best_epoch = epoch + 1
                # mIoU를 파일명에 포함
                ckpf = os.path.join(args.result_dir, f"model_best_e{best_epoch}_miou{best_miou:.4f}.pth")
                torch.save(_get_state_dict(model), ckpf)
                # 항상 최신 베스트를 가리키는 포인터(복사본)
                torch.save(_get_state_dict(model), os.path.join(args.result_dir, "model_best.pth"))
                torch.save(criterion_kd.state_dict(), os.path.join(args.result_dir, "kd_loss_best.pth"))

        dist.barrier()  # 다음 epoch 동기화

    if local_rank == 0 and writer is not None:
        writer.close()

    dist.destroy_process_group()

# ---------- Argparse ----------
if __name__ == "__main__":
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_P2P_DISABLE"] = "1" 
    os.environ["NCCL_IB_DISABLE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str,  help="Path to dataset root",
                        default="/content/drive/MyDrive/SemanticDataset_lednet")
    # parser.add_argument("--loadpath", type=str,  help="Path to dataset root", 
    #                     default="./DDRNet23s_imagenet.pth")    # "ex: ./pths/DDRNet23s_imagenet.pth"
    parser.add_argument("--loadpath", type=str,  help="Path to dataset root", 
                        default="/content/drive/MyDrive/DDRNet23s_kd_001/model_best_e64_miou0.7418.pth")    # "ex: ./pths/DDRNet23s_imagenet.pth"
    parser.add_argument("--teacher_loadpath", type=str, help="Path to dataset root",
                        default="/content/drive/MyDrive/DDRNet39_001/teacher_weights_only_2.pth")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--result_dir", type=str, default='/content/drive/MyDrive/DDRNet23s_kd_001')
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--crop_size", default=[1080, 1920], type=arg_as_list, help="crop size (H W)")
    parser.add_argument("--scale_range", default=[0.75, 1.25], type=arg_as_list,  help="resize Input")
    parser.add_argument("--normal_aug_prob", type=float, default=0.8, help="normal 이미지에 degradation 조합을 적용할 확률")
    parser.add_argument("--severity_min", type=int, default=3)
    parser.add_argument("--severity_max", type=int, default=5)
    parser.add_argument("--start-epoch", type=int, default=64,
                    help="가중치만 로드 시, 이어서 시작할 에폭(마지막 완료 에폭+1)")
    
    args = parser.parse_args()
    
    print(f'Initial learning rate: {args.lr}')
    print(f'Total epochs: {args.epochs}')
    print(f'dataset path: {args.dataset_dir}')
                  
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)               
    torch.multiprocessing.set_start_method('spawn', force=True)
    train(args)
