# UMIS v1.3

UMIS（Unified Multimodal Injection Sentinel）用于在图像、音频进入多模态助手之前，
检测其中可能携带的提示注入。输出 `BLOCK`、`QUARANTINE+CONFIRM`、`ALLOW+AUDIT`
及定位证据。实际拦截、确认和审计由接入应用实现。

这是研究实现，不保证检出所有攻击；不支持纯文本检测。

## 安装与快速使用

需要 Python 3.10+。在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell：.venv\Scripts\Activate.ps1
# Linux/macOS：source .venv/bin/activate
python -m pip install -e .
umis --version
umis probe 图片.png
```

`probe` 不下载模型，仅输出文件结构与信号特征，不输出完整安全裁决。

```bash
python -m pip install -e ".[ocr]"
umis detect --image 图片.png --user-text "描述这张图片" --out results/report.json

python -m pip install -e ".[audio]"
umis detect --audio 音频.wav --user-text "转录这段音频"
```

默认采用启发式评分，未启用语义编码器，不能视作论文中的完整配置。
完整实验安装：`python -m pip install -e ".[ocr,audio,encoder,judge,bench,plots]"`。
模型首次使用需要下载；MP3 转换和基准音频生成需要系统安装 FFmpeg。
完整 GPU 实验建议在 Linux 环境运行，按 CUDA 环境单独配置 PyTorch 和 vLLM。

`clip_clap` 编码器需要设置 `CLAP_CKPT` 为本地 LAION-CLAP 权重路径。
仓库不包含数据集、融合模型权重或第三方模型；须自行获取并训练。
训练、特征收集和推理须使用一致的编码器、漂移参考、语音检测器配置。

## 版本改动

- 增加安装配置、`umis` 命令、MIT 许可证、中英文文档、GitHub CI 和离线测试。
- 去除硬编码服务器目录，基准媒体路径相对于 manifest 保存。
- 修复 ASR 迭代器被耗尽、CPU 精度配置、短音频返回值和回声计算错误。
- 校准使用验证集；特征提取复用检测流程；L2 第二遍保留用户任务与完整抽取文本。
- 缓存按媒体内容与配置生成键；异常 L2 答复保留隔离状态。
- 历史 JSON 单独存放，排除字节码、论文草稿、Overleaf 日志和私有服务器记录。

原基准按攻击族和样本类别划分，但不同分割可能共享底图或原音频；不能宣称底层媒体完全独立。
旧实验结果不是 v1.3 新测结果；修复后需要重新实验才能发布新版性能数字。
检测器放行率与目标模型攻击成功率是不同指标。

详细运行命令见 [English README](README.md) 与 [复现说明](docs/REPRODUCIBILITY.md)。
GitHub 发布步骤见 [发布说明](docs/RELEASE.md)。
