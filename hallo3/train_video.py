# ============================================================
# train_video.py — 训练入口文件
# 作用：把数据、模型、训练循环组装在一起，启动训练
# ============================================================

import os
import argparse
from functools import partial   # 把函数"预填"部分参数，方便后面复用
import numpy as np
import torch.distributed         # PyTorch 多卡分布式训练工具
from omegaconf import OmegaConf  # 读取 .yaml 配置文件的库
import imageio                   # 把帧列表写成 mp4 视频文件

import torch
torch.backends.cudnn.enabled = False  # cuDNN 9.2.0 与驱动 570.86.15 在 H200 上不兼容

from sat import mpu              # SAT 库：管理"模型并行"的进程组
from sat.training.deepspeed_training import training_main  # DeepSpeed 训练主循环

from sgm.util import get_obj_from_str, isheatmap   # 工具函数：按字符串找类、判断是否热图
from diffusion_video import SATVideoDiffusionEngine # 模型类（扩散模型主体）
from arguments import get_args   # 解析命令行 + yaml 里的所有参数
import warnings

from einops import rearrange     # 张量维度重排工具，比 permute 更直观
from icecream import ic          # 调试打印工具，类似 print 但更好看

try:
    import wandb                 # 实验记录平台，记录 loss 曲线、视频样本等
except ImportError:
    print("warning: wandb not installed")

# 屏蔽 FutureWarning 和 DeprecationWarning，防止终端被警告刷屏
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def print_debug(args, s):
    """只有开了 --debug 模式才打印，并附上当前 GPU 的 rank 编号"""
    if args.debug:
        s = f"RANK:[{torch.distributed.get_rank()}]:" + s
        print(s)


def save_texts(texts, save_dir, iterations):
    """把这一步用到的文本 caption 保存成文件，文件名用步数编号，方便对照样本"""
    output_path = os.path.join(save_dir, f"{str(iterations).zfill(8)}")  # 补零对齐，如 00001000
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, T: int, fps: int = 5, args=None, key=None):
    """
    把一批生成的视频帧保存为 .mp4 文件
    video_batch: shape [B, T, C, H, W]，B 是视频数量，T 是帧数
    """
    os.makedirs(save_path, exist_ok=True)  # 目录不存在就创建

    for i, vid in enumerate(video_batch):  # 遍历每一个视频
        gif_frames = []
        for frame in vid:                              # 遍历每一帧
            frame = rearrange(frame, "c h w -> h w c") # 把 [C,H,W] 变成 [H,W,C]，imageio 需要这个格式
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)  # 从 [0,1] 浮点 → [0,255] 整数
            gif_frames.append(frame)

        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")  # 每个视频单独存一个文件
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)  # 一帧一帧写入 mp4

        # 如果开了 wandb，顺便上传到网页端方便远程查看
        if args is not None and args.wandb:
            wandb.log(
                {key + f"_video_{i}": wandb.Video(now_save_path, fps=fps, format="mp4")}, step=args.iteration + 1
            )


def log_video(batch, model, args, only_log_video_latents=False):
    """
    验证时调用：让模型生成样本视频，保存到磁盘（以及 wandb）
    only_log_video_latents=True 时只保存潜变量（latent），不解码成像素，省时间
    """
    texts = batch["txt"]                          # 取出这批数据的文本描述
    text_save_dir = os.path.join(args.save, "video_texts")
    os.makedirs(text_save_dir, exist_ok=True)
    save_texts(texts, text_save_dir, args.iteration)  # 把文本保存下来

    # 保持当前的混合精度设置（bf16/fp16），传给 autocast
    gpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled(),
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    # torch.no_grad()：验证时不需要计算梯度，省内存
    with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
        videos = model.log_video(batch, only_log_video_latents=only_log_video_latents)

    # 只让 rank=0（主进程）负责保存文件，避免多卡重复写
    if torch.distributed.get_rank() == 0:
        root = os.path.join(args.save, "video")

        if only_log_video_latents:
            # 只保存 latent（压缩后的特征向量），不解码成视频
            root = os.path.join(root, "latents")
            filename = "{}_gs-{:06}".format("latents", args.iteration)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            os.makedirs(path, exist_ok=True)
            torch.save(videos["latents"], os.path.join(path, "latent.pt"))  # 保存为 .pt 文件
        else:
            # 解码成像素视频，整理后保存
            for k in videos:
                N = videos[k].shape[0]
                if not isheatmap(videos[k]):
                    videos[k] = videos[k][:N]          # 截取 N 个样本
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().float().cpu()   # 搬到 CPU，转 float32
                    if not isheatmap(videos[k]):
                        videos[k] = torch.clamp(videos[k], -1.0, 1.0)  # 像素值限制在 [-1, 1]

            num_frames = batch["num_frames"][0]
            fps = batch["fps"][0].cpu().item()

            if only_log_video_latents:
                # 这段理论上不会执行到（上面已经判断过了），保险起见保留
                root = os.path.join(root, "latents")
                filename = "{}_gs-{:06}".format("latents", args.iteration)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                os.makedirs(path, exist_ok=True)
                torch.save(videos["latents"], os.path.join(path, "latents.pt"))
            else:
                for k in videos:
                    samples = (videos[k] + 1.0) / 2.0  # 从 [-1,1] 映射回 [0,1]（正常图像范围）
                    filename = "{}_gs-{:06}".format(k, args.iteration)
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    save_video_as_grid_and_mp4(samples, path, num_frames // fps, fps, args, k)


def broad_cast_batch(batch):
    """
    模型并行时，rank=0 读到了数据，需要把数据广播给同一个"模型并行组"里的其他 GPU
    因为一个大模型被切成几块分放在不同 GPU 上，每块都需要看到相同的输入
    """
    mp_size = mpu.get_model_parallel_world_size()  # 模型并行的 GPU 数量
    global_rank = torch.distributed.get_rank() // mp_size
    src = global_rank * mp_size                    # 数据来源的 GPU rank

    # 先广播张量的 shape，让其他 GPU 知道要准备多大的内存
    if batch["mp4"] is not None:
        broadcast_shape = [batch["mp4"].shape, batch["fps"].shape, batch["num_frames"].shape]
    else:
        broadcast_shape = None

    # 广播文本（txt 是字符串列表，用 broadcast_object_list 而不是 broadcast）
    txt = [batch["txt"], broadcast_shape]
    torch.distributed.broadcast_object_list(txt, src=src, group=mpu.get_model_parallel_group())
    batch["txt"] = txt[0]

    # 取出广播过来的 shape
    mp4_shape = txt[1][0]
    fps_shape = txt[1][1]
    num_frames_shape = txt[1][2]

    # 非主 rank 先分配空张量占位，然后接收广播数据
    if mpu.get_model_parallel_rank() != 0:
        batch["mp4"] = torch.zeros(mp4_shape, device="cuda")
        batch["fps"] = torch.zeros(fps_shape, device="cuda", dtype=torch.long)
        batch["num_frames"] = torch.zeros(num_frames_shape, device="cuda", dtype=torch.long)

    # 广播视频张量、fps、帧数
    torch.distributed.broadcast(batch["mp4"], src=src, group=mpu.get_model_parallel_group())
    torch.distributed.broadcast(batch["fps"], src=src, group=mpu.get_model_parallel_group())
    torch.distributed.broadcast(batch["num_frames"], src=src, group=mpu.get_model_parallel_group())
    return batch


def forward_step_eval(data_iterator, model, args, timers, only_log_video_latents=False, data_class=None):
    """
    验证步骤：取一批数据 → 生成样本视频保存 → 算 loss（用于监控，不更新权重）
    """
    if mpu.get_model_parallel_rank() == 0:  # 只有模型并行主 rank 负责读数据
        timers("data loader").start()
        batch_video = next(data_iterator)   # 从数据迭代器取下一批
        timers("data loader").stop()

        # 如果 mp4 是 6 维（多视角），把前两维 [B, V] 合并成 [B*V]
        if len(batch_video["mp4"].shape) == 6:
            b, v = batch_video["mp4"].shape[:2]
            batch_video["mp4"] = batch_video["mp4"].view(-1, *batch_video["mp4"].shape[2:])
            txt = []
            for i in range(b):
                for j in range(v):
                    txt.append(batch_video["txt"][j][i])
            batch_video["txt"] = txt

        # 把所有张量搬到 GPU
        for key in batch_video:
            if isinstance(batch_video[key], torch.Tensor):
                batch_video[key] = batch_video[key].cuda()
    else:
        # 非主 rank 先用空占位，等广播
        batch_video = {"mp4": None, "fps": None, "num_frames": None, "txt": None}

    broad_cast_batch(batch_video)  # 广播数据到模型并行组内所有 GPU

    # 只让数据并行 rank=0 的进程保存样本视频（避免重复写文件）
    if mpu.get_data_parallel_rank() == 0:
        log_video(batch_video, model, args, only_log_video_latents=only_log_video_latents)

    batch_video["global_step"] = args.iteration  # 把当前步数塞进 batch，模型可能用得到
    loss, loss_dict = model.shared_step(batch_video)  # 算 loss

    # bfloat16 的 loss 转成 float32，方便打印和记录
    for k in loss_dict:
        if loss_dict[k].dtype == torch.bfloat16:
            loss_dict[k] = loss_dict[k].to(torch.float32)
    return loss, loss_dict


def forward_step(data_iterator, model, args, timers, data_class=None):
    """
    训练步骤：取一批数据 → 算 loss → 返回给 DeepSpeed 做反向传播和权重更新
    （DeepSpeed 会自动处理 backward 和 optimizer.step，这里只负责 forward）
    """
    if mpu.get_model_parallel_rank() == 0:
        timers("data loader").start()
        batch = next(data_iterator)   # 取下一批训练数据
        timers("data loader").stop()

        # 把所有张量搬到 GPU
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()

        # 只在 rank=0 且第一次运行时，把完整配置保存一份到实验目录，方便复现
        if torch.distributed.get_rank() == 0:
            if not os.path.exists(os.path.join(args.save, "training_config.yaml")):
                configs = [OmegaConf.load(cfg) for cfg in args.base]
                config = OmegaConf.merge(*configs)       # 合并多个 yaml 配置
                os.makedirs(args.save, exist_ok=True)
                OmegaConf.save(config=config, f=os.path.join(args.save, "training_config.yaml"))
    else:
        # 非主 rank 先空占位，等广播
        batch = {"mp4": None, "fps": None, "num_frames": None, "txt": None, "audio_emb": None}

    batch["global_step"] = args.iteration  # 告诉模型当前是第几步

    broad_cast_batch(batch)  # 模型并行时把数据广播给其他 GPU

    loss, loss_dict = model.shared_step(batch)  # 模型 forward + 算扩散 loss

    return loss, loss_dict  # 返回给 DeepSpeed，由它做 backward + optimizer step


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":

    # 兼容 OpenMPI 启动方式：把 MPI 的环境变量映射到 PyTorch 分布式的变量
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    # 解析命令行参数（--base、--lr 等），也会顺带读取 yaml 配置
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))  # 把两部分参数合并

    # 根据配置文件里的字符串（如 "data_video.SFTDataset"）动态加载数据集类
    data_class = get_obj_from_str(args.data_config["target"])

    # partial：预先把数据集参数填进去，后面调用时只需传 path 和 args
    create_dataset_function = partial(data_class.create_dataset_function, **args.data_config["params"])

    import yaml

    # 把所有 yaml 配置读出来，附到 args 上，方便 wandb 等工具记录实验配置
    configs = []
    for config in args.base:
        with open(config, "r") as f:
            base_config = yaml.safe_load(f)
        configs.append(base_config)
    args.log_config = configs

    # 正式启动训练！DeepSpeed 会接管训练循环：
    # - 调用 forward_step 算 loss
    # - 自动做 backward（反向传播）
    # - 自动做 optimizer.step（更新权重）
    # - 按 save_interval 保存 checkpoint
    # - 按 eval_interval 调用 forward_step_eval 做验证
    training_main(
        args,
        model_cls=SATVideoDiffusionEngine,           # 用哪个模型类
        forward_step_function=partial(forward_step, data_class=data_class),        # 训练用的 forward
        forward_step_eval=partial(
            forward_step_eval, data_class=data_class, only_log_video_latents=args.only_log_video_latents
        ),                                           # 验证用的 forward
        create_dataset_function=create_dataset_function,  # 数据集工厂函数
    )
