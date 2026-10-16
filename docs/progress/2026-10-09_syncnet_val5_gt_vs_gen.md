# 2026-10-09：val 5 条的 SyncNet 打分（真值 vs 新 encoder 7000 步 vs 原版 Hallo3）

承接 [2026-10-08_checkpoint_selection_inference.md](2026-10-08_checkpoint_selection_inference.md)。

## 一、结论（先看这个）

| 项 | 结果 |
|----|------|
| 评估对象 | val 集随机 5 条（`--seed 0`），新 encoder 累计 7000 步 checkpoint（`10-06-23-37/2100`）的生成视频 |
| SyncNet 流程是否可信 | 可信。真值视频 conf 6.0–8.1、min dist 6.7–8.6、offset 稳定在 −1 到 −2，接近 SyncNet README 示例（offset 3 / dist 5.353 / conf 10.021） |
| 生成视频 | conf 0.74–1.32（平均 1.04）、min dist 13.1–14.3（平均 13.65）、offset 在 −15…+15 之间乱跳，其中 3 条贴着搜索范围边界 ±15。每一条都远差于对应的真值，口型基本没有被音频驱动 |
| 原版 Hallo3 | conf 5.29–7.28（平均 6.33，约为真值的 89%）、min dist 8.3–10.5（平均 9.29）、offset 稳定在 −1 到 −2，和真值一致。同样的 5 条输入、同样的 SyncNet 流程下原版能正常对上口型 |
| 由此排除的 | SyncNet 流程、输入数据（图片 / 音频）、推理流程（25 fps、分块、拼音频）本身有问题：这些在原版上都正常，所以生成视频的低分来自新 encoder 的模型或训练 |
| 还不能下结论的 | 差距由新 encoder 本身、训练不足（7000 步、2702 条，而原版是大规模训练）、训练帧率杂（vs 推理固定 25 fps）还是别的原因造成，现有数据区分不开。5 条样本也偏少 |

---

## 二、结果表

指标含义：

- AV offset：音画偏移（帧），理想接近 0，且多条之间应该稳定
- Min dist：最小距离，越低越好
- Confidence：置信度，越高越好（经验上大于 3 到 5 可认为基本同步，这是经验值，不是仓库给的阈值）

### 2.1 逐条

| clip | 真值 offset / dist / conf | 原版 Hallo3 offset / dist / conf | 新 encoder 7000 步 offset / dist / conf |
|------|---------------------------|-----------------------------------|------------------------------------------|
| 4610_qJNcXjKzC0I-scene4-scene1 | −1 / 7.077 / 7.478 | −2 / 8.286 / 7.278 | 6 / 13.777 / 0.736 |
| GXJQQqfB-ew-scene203-scene5 | −1 / 6.705 / 8.113 | −2 / 10.507 / 5.294 | −13 / 13.639 / 0.814 |
| 4V21SWg1V9Q-scene5-scene2 | −2 / 7.435 / 7.181 | −1 / 8.774 / 6.495 | −9 / 13.441 / 1.213 |
| _nllSIHloqz8-scene40-scene1 | −2 / 7.989 / 5.992 | −1 / 8.994 / 6.277 | −15 / 13.104 / 1.118 |
| 988_Rx-xZfFjVDQ-scene5-scene4 | −2 / 8.551 / 6.812 | −1 / 9.882 / 6.328 | 15 / 14.300 / 1.324 |

### 2.2 平均（5 条）

| | 真值 | 原版 Hallo3 | 新 encoder 7000 步 |
|---|------|-------------|---------------------|
| Min dist 平均 | 7.55 | 9.29 | 13.65 |
| Confidence 平均 | 7.12 | 6.33 | 1.04 |
| offset | 稳定 −1…−2 | 稳定 −1…−2 | 乱跳 −15…+15 |

参照：SyncNet README 示例 offset 3、min dist 5.353、conf 10.021。

### 2.3 观察

- 生成视频的 conf 约为真值的 15%，min dist 约为真值的 1.8 倍，每条都比对应真值差得多
- 偏移量没有规律（正负都有），贴边界（±15）的结果说明没有找到真实的对齐峰值，偏移量本身不可信
- 逐帧置信度在约 +8 和 −6 之间摆动：局部偶然对上，整体没有稳定的口型到音频对应
- 原版 Hallo3 比真值略差（conf 6.33 vs 7.12，min dist 9.29 vs 7.55），但偏移量和真值同样稳定在 −1 到 −2，说明它确实在用音频驱动口型。新 encoder 的 conf 只有原版的约 16%
- 5 条都检测到 1 张脸、1 个场景，人脸检测不是问题；视频本身已是 25 fps，没有重采样
- `988_Rx-xZfFjVDQ` 有一条 warning：音频 4.911s、视频 4.920s，差 9 ms，可忽略

---

## 三、方法与命令

环境：集群 `syncnet` conda 环境，节点 `rtx-devel`，仓库 `~/syncnet_python`，输出目录 `OUT=/scratch/li_qiany_neu/syncnet_out`。

### 3.1 生成视频

生成视频（`/scratch/li_qiany_neu/infer_out/step7000/<clip>/000000_with_audio.mp4`）自带音轨，直接用 `run_pipeline.py` + `run_syncnet.py`，`--reference step7000_<clip>`。

### 3.2 真值视频

真值视频需要先把音频合回去（见 3.3 的第二个坑），再打分：

```bash
cd ~/syncnet_python
OUT=/scratch/li_qiany_neu/syncnet_out
MAP=~/hallo3/my_inference/input_val5.txt.map.tsv
rm -rf $OUT/*/gt_*
mkdir -p $OUT/gt_mux

while IFS=$'\t' read -r -u 3 stem gt img aud; do
  echo "=== gt_$stem"
  mux=$OUT/gt_mux/$stem.mp4
  ffmpeg -nostdin -y -loglevel error -i "$gt" -i "$aud" \
    -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest "$mux"
  python run_pipeline.py --videofile "$mux" --reference "gt_$stem" --data_dir "$OUT" </dev/null
  python run_syncnet.py  --videofile "$mux" --reference "gt_$stem" --data_dir "$OUT" </dev/null
done 3< <(tail -n +2 "$MAP") 2>&1 | tee $OUT/gt.log | grep -E "^===|AV offset|Min dist|Confidence|Error|error"
```

日志：`$OUT/gt.log`。

### 3.3 踩过的两个坑

1. **`while read` 循环里的 ffmpeg 吃掉 stdin。** `run_pipeline.py` 调用的 ffmpeg 会读 stdin，而 `while read` 的输入正好也是 stdin，结果 map 文件后面的行被吃掉，路径变成残缺片段（如 `videovideo4610...`），只跑了 3 条且路径错。解决：用 `read -u 3` + `3< <(...)` 让循环输入走 fd 3，命令加 `</dev/null`，ffmpeg 加 `-nostdin`。
2. **TalkVid 真值视频没有音轨。** 视频在 `videos/`，音频单独在 `audios/`，直接对真值视频跑 `run_pipeline.py` 会在抽音频时报 `Output file #0 does not contain any stream`。解决：先用 `ffmpeg` 把对应音频合进去，再打分。之后对**任何不带音轨的输出视频**打分都要先合音频。

---

## 四、待办

- [x] 原版 Hallo3 跑同样 5 条并打分（已完成，结果见 2.1、2.2）
  - 用 `git worktree add ~/hallo3_orig origin/main`（`e342dce`）检出原版，`pretrained_models` 软链到 `/scratch/li_qiany_neu/pretrained_models`
  - 当前分支 `train7/new` 的 AudioProjModel 维度已改，不能直接加载原版权重，所以必须用 worktree 的 `main`
  - 输出目录：`/scratch/li_qiany_neu/infer_out/orig_hallo3/000000-<clip>-seed_42/000000_with_audio.mp4`，SyncNet 日志 `$OUT/orig.log`
  - 打分时新 shell 需先 `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`，否则 scipy 报 `GLIBCXX_3.4.30 not found`
- [ ] 区分"新 encoder 本身"和"训练量 / 数据量"：A 组对照，原 wav2vec 编码器在同一份 2702 条数据上训同样步数，再用同样 5 条打分。原版是大规模训练，直接和新 encoder 比，分不清是 encoder 的问题还是训练量的问题
- [ ] 噪声基线：真值视频配另一条 clip 的音频再打分，得到"完全没同步"时的 conf 下限，用来判断生成视频的 1.04 是否已贴在下限上
- [ ] 同样 5 条输入换 checkpoint（4900、9800、10400、约 11000）推理并打分，看 conf 是否随步数上升
- [ ] 音频消融：同一张图换成完全不同的音频生成，看口型是否还一样
- [ ] 如果确认问题在帧率不一致：把训练 clip 统一重采样到 25 fps 后重训（训练 fps 分布：30 fps 31%、23.98 fps 21%、29.97 fps 20%、25 fps 11%、24 fps 6%、约 60 fps 5%）
- [ ] 选定 checkpoint 后，在 val 全部 291 条上打分，再用 eval 350 条对比原版 Hallo3
