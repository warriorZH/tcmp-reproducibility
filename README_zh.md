# PropGate 复现材料

[English](README.md) · [论文图表与文件对照](docs/PAPER_MAPPING.md) · [评估协议](docs/PROTOCOL.md) · [数据](data/README.md) · [验证记录](docs/VALIDATION.md) · [运行环境](environment/README.md)

本目录包含以下论文的代码、锁定的数据划分、已保存的实验结果和核验脚本：

> Hu J, Zhang P, Li X, Wang Z. **PropGate: graph label propagation with a reference-calibrated acceptance gate for low-label medicinal plant recognition when unlabeled archives contain unknown species.** 已投稿 *Plant Methods*（Methodology），2026。

PropGate 用每个物种少量带标签的参考图片，加上一个可能混有参考名单之外物种的无标签图库，来识别药用植物照片。图片用冻结的 BioCLIP 2 编码器提取一次特征；在参考图与图库图片构成的 k 近邻图上做标签传播（k = 10；每类4张标签时 α = 0.9，16张时 0.8）；只用参考集拟合的逻辑回归头给出接受阈值（交叉拟合最大 softmax 概率的第10百分位），每张查询照片要么给出已知物种，要么转交专家。

核对论文中的数字只需普通电脑的 CPU，几分钟即可完成。若要从图片重新跑实验，还需要下载两个公开数据集，最好有 GPU。

## 目录内容

| 路径 | 内容 |
|---|---|
| `src/tcmp_repro/` | 核心方法（LR、固定/软自训练、迭代与一步标签传播、参考集 kNN/NCM）、接受门、评价指标、特征提取和实验运行程序 |
| `revision/` | 正文 3.3–3.7 节的补充实验：OOD 过滤自训练、零样本与微调基线、k × α 敏感性、DINOv2/CLIP 编码器、开发类网格；`run_all.sh` 一键运行 |
| `configs/` | 固定的实验设置（`experiment.json`：随机种子、k、α、阈值、bootstrap）；BioCLIP 2 模型配置 |
| `data/` | 两个数据集的锁定清单（图片 ID、相对路径、SHA-256、标签、相似组、参考集抽样、物种角色）及 336 个污染情境的图库索引。**不含图片。** |
| `results/reference/` | 主体污染实验的已保存结果：350 个逐配置记录（TCMP + Indian）、476 个 TCMP 原始实验记录、审稿阶段对照 |
| `results/revision/` | 补充实验的已保存结果：497 个逐配置文件、14 个零样本/微调运行结果、10 个汇总文件、特征元数据和运行日志（服务器路径已替换为占位符） |
| `scripts/` | 核验与分析脚本（见下） |
| `docs/` | 评估协议、论文图表对照表、验证记录、插图源文件 |
| `provenance/` | 原始代码快照、BioCLIP 2 权重版本与哈希 |

## 第一级：核对论文报告的全部数字（只需 CPU，无需下载数据）

需要 Python 3.10–3.12。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate      Windows: .venv\Scripts\Activate.ps1
python -m pip install -e . "numpy==1.24.3" "scipy==1.11.2"
python scripts/run_offline_checks.py
```

依次执行8个步骤，报告写入 `outputs/offline_check_report.txt`：

| 步骤 | 脚本 | 检查内容 |
|---|---|---|
| 1 | `verify_release.py` | 清单计数、训练/查询图片及相似组互不重叠、4/16 张嵌套抽样、重新计算全部 336 个图库索引、350 + 476 个结果记录、文件校验和 |
| 2 | `smoke_test.py` | 7 种分类器 × 3 种接受门在合成特征上的功能测试（不是精度评测） |
| 3 | `analyze_formal.py` | 由 476 个原始记录重算 TCMP 闭集/开集汇总 |
| 4 | `analyze_results.py` | 污染梯度、组件对照、接受门与伪标签分析；重新生成图4–6及表1–2所依据的统计量 |
| 5 | `compare_analysis.py` | 将第4步与已发布的参考统计量比对（894 个数值，容差 1e-10） |
| 6 | `verify_revision.py` | 由逐配置文件重算补充实验的每个汇总结果 |
| 7 | `check_paper_numbers.py` | 正文 3.3–3.7 节和表3–7中印出的 308 个数值 |
| 8 | `make_revision_tables.py --check` | 重新生成附加文件1的表 S16–S26 |

每张表、每幅图对应哪些文件和命令，见 [docs/PAPER_MAPPING.md](docs/PAPER_MAPPING.md)。

## 第二级：从图片重新运行

### 数据

按 [data/README.md](data/README.md) 下载两个公开数据集（CC BY 4.0）：

- TCMP-300（Figshare，DOI 10.6084/m9.figshare.29432726）：根目录下有 `train/` 和 `val/`。
- Indian Medicinal Leaves Image Datasets 第3版（Mendeley Data，DOI 10.17632/748f8jkphb.3）：使用其中的 **Medicinal Leaf dataset** 文件夹。

先确认图片与本研究所用图片逐字节一致：

```bash
python scripts/verify_images.py --dataset tcmp --data-root /path/to/tcmp-300
python scripts/verify_images.py --dataset indian --data-root "/path/to/Medicinal Leaf dataset"
```

### 安装

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e ".[revision]" "numpy==1.24.3" "scipy==1.11.2"
python scripts/download_checkpoint.py --output checkpoints/bioclip-2   # 按 provenance/model.json 校验版本与 SHA-256
```

### 主体实验（正文 3.1–3.2、3.6、3.8 节；表1–2；图4–6）

```bash
python -m tcmp_repro.extract --dataset tcmp   --data-root /path/to/tcmp-300 --model-dir checkpoints/bioclip-2 --output outputs/features/tcmp   --device cuda:0 --batch-size 4
python -m tcmp_repro.extract --dataset indian --data-root "/path/to/Medicinal Leaf dataset" --model-dir checkpoints/bioclip-2 --output outputs/features/indian --device cuda:0 --batch-size 2
python -m tcmp_repro.run --dataset tcmp   --features outputs/features/tcmp/features.npy   --ids outputs/features/tcmp/ids.json   --output outputs/rerun/tcmp --suite gradients
python -m tcmp_repro.run --dataset indian --features outputs/features/indian/features.npy --ids outputs/features/indian/ids.json --output outputs/rerun/indian --suite all
python -m tcmp_repro.original --features outputs/features/tcmp/features.npy --ids outputs/features/tcmp/ids.json --output outputs/original_tcmp
python scripts/analyze_results.py --input outputs/rerun --output outputs/analysis_rerun
```

建议先只跑一个配置（`--seed 4001 --role 202610130 --rho 0.5 --max-contexts 1`）检查环境。特征提取也可用 `--device cpu`，但较慢。

### 补充实验（正文 3.3–3.5、3.7 节；表3–7；附加文件1 第 S7–S11 节）

```bash
TCMP_ROOT=/path/to/tcmp-300 INDIAN_ROOT="/path/to/Medicinal Leaf dataset" \
TCMP_FEATURES=outputs/features/tcmp/features.npy INDIAN_FEATURES=outputs/features/indian/features.npy \
OUT=outputs/revision bash revision/run_all.sh
```

`run_all.sh` 支持断点续跑（再次运行即可继续），可用 `STAGES="w2 w4 zeroshot w5 dev finetune"` 只跑部分阶段；缺少 BioCLIP 2 特征或权重时会自动提取或下载。使用特征前，它会先检查这些特征能否复现一个已发布配置的结果。各阶段与输出：

| 阶段 | 脚本 | 对应论文 | 输出（`$OUT/` 下） |
|---|---|---|---|
| `w2` | `exp_w2_ood_selftrain.py` | 表3，表 S19–S20 | `w2/<数据集>/`，`w2/<数据集>_summary.json` |
| `zeroshot` | `exp_w3_zeroshot.py` | 表4，表 S21 | `w3/zeroshot_<数据集>.json` |
| `finetune` | `exp_w3_finetune.py` | 表4，表 S22 | `w3/finetune_<骨干网络>_<数据集>_s<种子>.json` |
| `w4` | `exp_w4_sensitivity.py` | 表6–7，表 S16–S18 | `w4/<数据集>/`，`w4/<数据集>_summary.json` |
| `w5` | `extract_features.py`、`exp_w5_encoders.py` | 表5，表 S23–S25 | `features/<编码器>_<数据集>/`，`w5/` |
| `dev` | `extract_features.py`、`exp_dev_grid.py` | 3.7 节，表 S26 | `features/bioclip2_tcmp_dev/`，`dev/` |

如需把重跑结果与论文比对：在本目录的另一份副本中，用 `$OUT/{w2,w3,w4,w5,dev}` 覆盖 `results/revision/`，再运行第6–8步。在参考服务器（一块 Tesla P40，8 个并行 CPU 任务）上，补充实验全部跑完约需一天，大部分时间花在微调和其他编码器的特征提取上。

## 已知限制

- **微调结果不能逐位复现。** 随机数据增强和 GPU 内核会使不同机器上的微调准确率在末位小数上有差异。表4的均值应在报告的标准差范围内一致，而不是完全相同。
- **CLIP 的激活函数（已修正）。** 第一次 CLIP 实验用的是 open_clip 的 `ViT-L-14` 配置（GELU），而 OpenAI 权重是用 QuickGELU 训练的。2026年10月22日的稿件（第9版）报告的是用 `ViT-L-14-quickgelu` 重跑的结果（`--encoder clip-quickgelu`，`revision/run_clip_quickgelu.sh`）：CLIP 各方法准确率提高 1.5–6.5 点，TCMP 上的比较结论也随之改变（50% 污染时传播优于 LR 和固定阈值自训练）。被替换的 GELU 结果保留在 `results/revision/superseded/clip_gelu/`，以便查对。
- **开发类图片。** 有一张 TCMP 开发类图片文件损坏，已排除（`results/revision/features/bioclip2_tcmp_dev/excluded_images.json`）；开发类图片未做近重复审核。
- **不分发图片。** 清单中的 SHA-256 保证图片身份一致；如果公开数据集版本有变动，`verify_images.py` 会列出不一致的文件。
- **未在新机器上端到端重跑。** 所有已保存结果都已由保存的记录复核（第一级）；从图片到结果的完整流程（第二级）只在参考服务器上运行过一次。

## 引用与许可

请引用上述论文（正式发表后补充 DOI），以及 `references.bib` 中列出的数据集和 BioCLIP 2。`CITATION.cff` 描述本软件版本（0.2.1）。代码和文档采用 MIT 许可；已保存的数值结果采用 CC BY 4.0，引用时请注明本研究。数据集元数据和模型权重保留其原有许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
