# Scratch 30天清理防护

## 问题
集群 `/scratch/li_qiany_neu/` 下的文件 30 天未访问会被自动清理删除。

## 解决方案
写了 `scripts/touch_scratch.sh`，定期 touch 所有文件和目录，刷新时间戳。

### 关键发现
1. **必须用 `touch`，不能用 `touch -a`**：很多集群清理策略看的是 mtime（修改时间），而非 atime（访问时间）。`touch -a` 只更新 atime，不管用。用 `touch`（不加参数）同时更新 atime 和 mtime。
2. **目录也要 touch**：`find -type f` 只处理文件，目录的时间戳不会更新。去掉 `-type f` 限制，对文件和目录一起 touch。
3. **不影响 GPU 排队**：`touch` 只改元数据，不占 GPU/CPU，与 SLURM 调度无关。

### 用法
```bash
bash scripts/touch_scratch.sh
```

### 建议设 cron job 每两周自动执行
```bash
crontab -e
# 每两周周日凌晨3点执行
0 3 1,15 * * bash /path/to/touch_scratch.sh >> ~/touch_scratch.log 2>&1
```
