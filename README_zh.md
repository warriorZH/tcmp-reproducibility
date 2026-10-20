# 低标注药用植物识别：未知类别污染容忍性与接受决策取舍

本仓库对应当前论文的实验与分析材料。研究在冻结 BioCLIP 2 特征上，比较逻辑回归（LR）、固定阈值／软加权自训练和标签传播（LP），并独立评估类别预测及接受／拒答规则。

## 内容

- TCMP 240类正式子集与 Indian 60类叶片子集的图片清单、SHA-256、类别映射、支持集抽样、未知角色及分组记录。
- 全部336个污染池索引；双来源350份逐配置结果（每配置7分类器×3接受规则）、原TCMP正式评估476份逐方法结果及审稿控制的已有元数据。
- 核心代码、可配置运行入口、模型版本与校验信息、结果重分析、数值对照和图表生成脚本。

## 最快检查路径：不需要图片或GPU

解压并进入仓库根目录，使用Python3.10–3.12与独立虚拟环境：

```bash
python -m pip install -e .
python scripts/verify_release.py
python scripts/smoke_test.py
python scripts/analyze_formal.py
python scripts/analyze_results.py --input results/reference --output outputs/analysis
python scripts/compare_analysis.py
```

最后一步将重新计算的数值与论文使用的参考统计逐项对照。报告、CSV和图表生成在 `outputs/analysis/`。

## 从图片重新运行

完整命令见[英文README](README.md)。依次进行：

1. 从原发布源下载两个数据集，按 `data/README.md` 的目录说明解压。
2. 使用 `scripts/verify_images.py` 核对原始文件哈希；不在确认集上重新抽样或调整参数。
3. 下载固定revision的BioCLIP 2；网络需要时可通过 `--endpoint https://hf-mirror.com` 使用镜像。
4. 用 `python -m tcmp_repro.extract` 提取冻结特征；论文使用P40 24 GB和float32。
5. 使用 `tcmp_repro.run` 跑双来源污染梯度／Indian闭集，使用 `tcmp_repro.original` 跑TCMP原闭集与全量池对照。
6. 用相同分析入口整理完整结果。

包内保留已有标签用于支持抽样与独立评价；核心学习器只接收图片特征、支持索引和支持标签，不接收无标签池／查询真值。原始图片、模型权重、特征数组及大体积预测缓存不放入Git仓库，由使用者下载或再生成。

## 复现范围

已验证数据清单、全部污染索引、合成数据上的7分类器×3接受规则、核心源函数一致性，以及已有结果的重分析。尚未在全新下载的数据和新机器上从头完成全部GPU特征提取与全量重跑；这一区别在验证说明中明确记录。数据最初获取的归档版本未完整记录，是否匹配以文件SHA-256为准。

核心结论是低标注未知混入条件下的污染容忍性和接受决策取舍，包含不利条件与类别尾部结果。干净池、容量受限0%池和总量固定池分别报告，不能混为同一基线。

## 上传GitHub

将本目录下文件上传到新仓库即可，目录内已配置 `.gitignore`、许可证、引用文件和CPU检查工作流。`CITATION.cff` 与 `LICENSE` 已填写作者信息（Jingyi Hu、Peihua Zhang、Xiangmeng Li、Zibin Wang）；论文正式发表后补充DOI与论文链接。

代码采用MIT许可证；数据集和模型按各自来源的许可与署名要求使用。详见 `THIRD_PARTY_NOTICES.md`。
