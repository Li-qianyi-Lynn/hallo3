# ============================================================
# arguments.py — 训练参数解析与初始化
# 作用：读取命令行参数 + yaml 配置，初始化分布式训练和随机种子
# ============================================================

import argparse
import os
import torch
import json
import warnings
import omegaconf
from omegaconf import OmegaConf
from sat.helpers import print_rank0          # 只让 rank=0 打印，避免多卡重复输出
from sat import mpu                          # 模型并行进程组管理
from sat.arguments import set_random_seed
from sat.arguments import add_training_args, add_evaluation_args, add_data_args  # SAT 库预置参数
import torch.distributed


def add_model_config_args(parser):
    """添加模型相关的命令行参数"""

    group = parser.add_argument_group("model", "model configuration")
    group.add_argument("--base", type=str, nargs="*",
                       help="yaml 配置文件路径，可以传多个，会按顺序合并")
    group.add_argument("--model-parallel-size", type=int, default=1,
                       help="模型并行度：把模型切成几块分放在几张 GPU 上，普通用户保持 1 即可")
    group.add_argument("--force-pretrain", action="store_true",
                       help="强制重新加载预训练权重，即使已有 checkpoint")
    group.add_argument("--device", type=int, default=-1,
                       help="指定 GPU 编号，-1 表示自动分配")
    group.add_argument("--debug", action="store_true",
                       help="开启调试模式，打印更多信息")
    group.add_argument("--log-image", type=bool, default=True,
                       help="是否保存样本图/视频")

    return parser


def add_sampling_config_args(parser):
    """添加推理采样相关的命令行参数（训练时一般不用改）"""

    group = parser.add_argument_group("sampling", "Sampling Configurations")
    group.add_argument("--output-dir", type=str, default="samples",
                       help="生成结果的保存目录")
    group.add_argument("--input-dir", type=str, default=None)
    group.add_argument("--input-type", type=str, default="cli",
                       help="输入方式：cli（命令行）或 file（文件）")
    group.add_argument("--input-file", type=str, default="input.txt",
                       help="批量推理时的输入文件")
    group.add_argument("--final-size", type=int, default=2048)
    group.add_argument("--sdedit", action="store_true",
                       help="开启 SDEdit 模式（图像编辑）")
    group.add_argument("--grid-num-rows", type=int, default=1)
    group.add_argument("--force-inference", action="store_true")
    group.add_argument("--lcm_steps", type=int, default=None,
                       help="LCM（少步推理）的步数")
    group.add_argument("--sampling-num-frames", type=int, default=32,
                       help="推理时生成多少帧")
    group.add_argument("--sampling-fps", type=int, default=8,
                       help="推理时的帧率")
    group.add_argument("--only-save-latents", type=bool, default=False,
                       help="只保存 latent，不解码成像素（省时间）")
    group.add_argument("--only-log-video-latents", type=bool, default=False,
                       help="验证时只记录 latent，不生成完整视频")
    group.add_argument("--latent-channels", type=int, default=32,
                       help="VAE latent 的通道数")
    group.add_argument("--image2video", action="store_true",
                       help="图生视频模式（以图像为起始帧）")

    return parser


def add_others_config_args(parser):
    """添加音频、人脸分析等模型路径参数"""

    group = parser.add_argument_group("others", "Others Configurations")
    group.add_argument("--sample_rate", type=int, default=16000,
                       help="音频采样率，wav2vec2 标准输入是 16kHz")
    group.add_argument("--wav2vec_model_path", type=str,
                       help="wav2vec2 音频编码器的本地路径")
    group.add_argument("--wav2vec_features", type=str, default="all",
                       help="使用 wav2vec2 的哪些特征层：all / last / ...")
    group.add_argument("--audio_separator_model_path", type=str,
                       help="人声分离模型路径（把背景音乐从人声中分离）")
    group.add_argument("--face_analysis_model_path", type=str,
                       help="人脸分析模型路径（提取人脸 embedding）")

    return parser


def get_args(args_list=None, parser=None):
    """
    核心函数：解析所有参数，组装 DeepSpeed 配置，初始化分布式环境
    返回填好所有字段的 args 对象
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="sat")
    else:
        assert isinstance(parser, argparse.ArgumentParser)

    # 把各组参数逐一注册到 parser
    parser = add_model_config_args(parser)
    parser = add_sampling_config_args(parser)
    parser = add_training_args(parser)      # SAT 预置：lr、batch_size、train_iters 等
    parser = add_evaluation_args(parser)    # SAT 预置：eval_interval、eval_iters 等
    parser = add_data_args(parser)          # SAT 预置：train_data、num_workers 等
    parser = add_others_config_args(parser)

    import deepspeed
    parser = deepspeed.add_config_arguments(parser)  # DeepSpeed 自己的参数（如 --deepspeed_config）

    args = parser.parse_args(args_list)      # 正式解析命令行
    args = process_config_to_args(args)      # 再把 yaml 里的 args 字段覆盖进来

    if not args.train_data:
        print_rank0("No training data specified", level="WARNING")

    # train_iters 和 epochs 只能二选一，防止冲突
    assert (args.train_iters is None) or (args.epochs is None), \
        "only one of train_iters and epochs should be set."

    # 两个都没设置时，默认训练 10000 步
    if args.train_iters is None and args.epochs is None:
        args.train_iters = 10000
        print_rank0("No train_iters (recommended) or epochs specified, use default 10k iters.", level="WARNING")

    args.cuda = torch.cuda.is_available()   # 检查是否有 GPU

    # 从环境变量读取分布式训练信息（torchrun/srun 会自动设置这些）
    args.rank = int(os.getenv("RANK", "0"))            # 当前进程在所有进程中的全局编号
    args.world_size = int(os.getenv("WORLD_SIZE", "1")) # 总进程数 = 总 GPU 数
    if args.local_rank is None:
        args.local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 当前进程在本机上的编号

    # 自动把进程绑定到对应 GPU
    if args.device == -1:
        if torch.cuda.device_count() == 0:
            args.device = "cpu"
        elif args.local_rank is not None:
            args.device = args.local_rank  # 本机第 local_rank 号 GPU
        else:
            args.device = args.rank % torch.cuda.device_count()

    # local_rank 和 device 不一致时报错（推理模式除外，推理时可以手动指定）
    if args.local_rank != args.device and args.mode != "inference":
        raise ValueError(
            "LOCAL_RANK (default 0) and args.device inconsistent. "
            "This can only happens in inference mode. "
            "Please use CUDA_VISIBLE_DEVICES=x for single-GPU training. "
        )

    if args.rank == 0:
        print_rank0("using world size: {}".format(args.world_size))  # 打印总 GPU 数

    # 如果指定了数据权重，数量必须和数据集数量一致
    if args.train_data_weights is not None:
        assert len(args.train_data_weights) == len(args.train_data)

    # ---- 组装 DeepSpeed 配置 ----
    if args.mode != "inference":  # 只有训练模式才启用 DeepSpeed
        args.deepspeed = True
        if args.deepspeed_config is None:  # 没有手动指定 deepspeed config 文件时
            # 自动找项目内预置的 deepspeed_zero{N}.json（N 是 ZeRO stage）
            deepspeed_config_path = os.path.join(
                os.path.dirname(__file__), "training", f"deepspeed_zero{args.zero_stage}.json"
            )
            with open(deepspeed_config_path) as file:
                args.deepspeed_config = json.load(file)
            override_deepspeed_config = True   # 标记：用 args 的值覆盖 deepspeed config
        else:
            override_deepspeed_config = False  # 标记：用 deepspeed config 的值覆盖 args

    # fp16 和 bf16 不能同时开
    assert not (args.fp16 and args.bf16), "cannot specify both fp16 and bf16."

    # 用了 ZeRO 但没开混合精度时，自动开 fp16
    if args.zero_stage > 0 and not args.fp16 and not args.bf16:
        print_rank0("Automatically set fp16=True to use ZeRO.")
        args.fp16 = True
        args.bf16 = False

    if args.deepspeed:
        # 是否用 gradient checkpointing（激活检查点）省显存
        if args.checkpoint_activations:
            args.deepspeed_activation_checkpointing = True
        else:
            args.deepspeed_activation_checkpointing = False

        if args.deepspeed_config is not None:
            deepspeed_config = args.deepspeed_config

        if override_deepspeed_config:
            # 用命令行/yaml 的参数值填写 deepspeed config 里的字段
            if args.fp16:
                deepspeed_config["fp16"]["enabled"] = True
            elif args.bf16:
                deepspeed_config["bf16"]["enabled"] = True
                deepspeed_config["fp16"]["enabled"] = False
            else:
                deepspeed_config["fp16"]["enabled"] = False

            # 把 batch_size 写入 deepspeed（micro batch = 每张卡每步处理的样本数）
            deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
            # 梯度累积步数：攒 N 步再更新一次权重，相当于把 batch 扩大 N 倍
            deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            optimizer_params_config = deepspeed_config["optimizer"]["params"]
            optimizer_params_config["lr"] = args.lr               # 学习率
            optimizer_params_config["weight_decay"] = args.weight_decay  # 权重衰减
        else:
            # 反过来：用 deepspeed config 文件里的值覆盖 args
            if args.rank == 0:
                print_rank0("Will override arguments with manually specified deepspeed_config!")
            if "fp16" in deepspeed_config and deepspeed_config["fp16"]["enabled"]:
                args.fp16 = True
            else:
                args.fp16 = False
            if "bf16" in deepspeed_config and deepspeed_config["bf16"]["enabled"]:
                args.bf16 = True
            else:
                args.bf16 = False
            if "train_micro_batch_size_per_gpu" in deepspeed_config:
                args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
            if "gradient_accumulation_steps" in deepspeed_config:
                args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
            else:
                args.gradient_accumulation_steps = None
            if "optimizer" in deepspeed_config:
                optimizer_params_config = deepspeed_config["optimizer"].get("params", {})
                args.lr = optimizer_params_config.get("lr", args.lr)
                args.weight_decay = optimizer_params_config.get("weight_decay", args.weight_decay)
        args.deepspeed_config = deepspeed_config

    # 初始化分布式进程组 + 随机种子
    initialize_distributed(args)
    # 每张卡用不同的随机种子，防止所有卡采样到相同数据
    args.seed = args.seed + mpu.get_data_parallel_rank()
    set_random_seed(args.seed)
    return args


def initialize_distributed(args):
    """
    初始化 PyTorch 分布式通信（多卡训练的基础）
    设置：数据并行组、模型并行组、上下文并行组
    """
    # 如果已经初始化过，检查模型并行配置是否一致，避免重复初始化
    if torch.distributed.is_initialized():
        if mpu.model_parallel_is_initialized():
            if args.model_parallel_size != mpu.get_model_parallel_world_size():
                raise ValueError(
                    "model_parallel_size is inconsistent with prior configuration."
                    "We currently do not support changing model_parallel_size."
                )
            return False
        else:
            if args.model_parallel_size > 1:
                warnings.warn(
                    "model_parallel_size > 1 but torch.distributed is not initialized via SAT."
                    "Please carefully make sure the correctness on your own."
                )
            mpu.initialize_model_parallel(args.model_parallel_size)
        return True

    # 把当前进程绑定到对应 GPU
    if args.device == "cpu":
        pass
    else:
        torch.cuda.set_device(args.device)

    # 构建 master 节点地址（所有进程通过这个地址互相找到对方）
    init_method = "tcp://"
    args.master_ip = os.getenv("MASTER_ADDR", "localhost")

    if args.world_size == 1:
        # 单卡训练时随机找一个空闲端口
        from sat.helpers import get_free_port
        default_master_port = str(get_free_port())
    else:
        default_master_port = "6000"  # 多卡训练默认端口

    args.master_port = os.getenv("MASTER_PORT", default_master_port)
    init_method += args.master_ip + ":" + args.master_port

    # 正式初始化分布式进程组（nccl 是 GPU 间通信的高效后端）
    torch.distributed.init_process_group(
        backend=args.distributed_backend,   # 通常是 "nccl"
        world_size=args.world_size,
        rank=args.rank,
        init_method=init_method
    )

    # 初始化 SAT 的模型并行进程组
    mpu.initialize_model_parallel(args.model_parallel_size)

    # 设置上下文并行组（视频长序列切分用，减少单卡的 attention 计算量）
    from sgm.util import set_context_parallel_group, initialize_context_parallel
    if args.model_parallel_size <= 2:
        set_context_parallel_group(args.model_parallel_size, mpu.get_model_parallel_group())
    else:
        initialize_context_parallel(2)  # 上下文并行固定用 2

    if args.deepspeed:
        import deepspeed
        # DeepSpeed 也需要独立初始化它自己的分布式环境
        deepspeed.init_distributed(
            dist_backend=args.distributed_backend,
            world_size=args.world_size,
            rank=args.rank,
            init_method=init_method
        )
    else:
        # 非 deepspeed 模式下，仍然需要初始化 RNG tracker（随机数跟踪器），
        # 因为模型并行中 dropout 需要每个并行组用相同的随机数
        try:
            import deepspeed
            from deepspeed.runtime.activation_checkpointing.checkpointing import (
                _CUDA_RNG_STATE_TRACKER,
                _MODEL_PARALLEL_RNG_TRACKER_NAME,
            )
            _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 1)  # 默认种子 1
        except Exception as e:
            from sat.helpers import print_rank0
            print_rank0(str(e), level="DEBUG")

    return True


def process_config_to_args(args):
    """
    把 yaml 配置文件里 args: 字段的内容，覆盖到命令行解析出来的 args 中
    yaml 优先级 > 命令行默认值，命令行显式指定 > yaml

    同时把 yaml 里的 model/deepspeed/data 三个字段分别存到 args 对应属性上
    """
    # 读取并合并所有 yaml 配置文件（后面的会覆盖前面的同名字段）
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    config = OmegaConf.merge(*configs)

    # 取出 yaml 里的 args 字段，把每个 key 设置到 args 对象上
    args_config = config.pop("args", OmegaConf.create())
    for key in args_config:
        if isinstance(args_config[key], omegaconf.DictConfig) or isinstance(args_config[key], omegaconf.ListConfig):
            arg = OmegaConf.to_object(args_config[key])  # 转成普通 Python dict/list
        else:
            arg = args_config[key]
        if hasattr(args, key):
            setattr(args, key, arg)  # 覆盖 args 中对应字段

    # 把 yaml 里的 model 字段存为 args.model_config（供模型初始化时使用）
    if "model" in config:
        model_config = config.pop("model", OmegaConf.create())
        args.model_config = model_config

    # 把 yaml 里的 deepspeed 字段存为 args.deepspeed_config（供 DeepSpeed 使用）
    if "deepspeed" in config:
        deepspeed_config = config.pop("deepspeed", OmegaConf.create())
        args.deepspeed_config = OmegaConf.to_object(deepspeed_config)

    # 把 yaml 里的 data 字段存为 args.data_config（供数据集初始化时使用）
    if "data" in config:
        data_config = config.pop("data", OmegaConf.create())
        args.data_config = data_config

    return args
