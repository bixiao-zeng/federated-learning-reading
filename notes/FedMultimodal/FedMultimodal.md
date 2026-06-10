# 【KDD 2023】FedMultimodal: A Benchmark for Multimodal Federated Learning

## Metadata

- Title: FedMultimodal: A Benchmark for Multimodal Federated Learning
- Authors: Tiantian Feng, Digbalay Bose, Tuo Zhang, Rajat Hebbar, Anil Ramakrishna, Rahul Gupta, Mi Zhang, Salman Avestimehr, Shrikanth Narayanan
- Venue / Year: ACM SIGKDD 2023
- Paper: https://doi.org/10.1145/3580305.3599825
- Code / Benchmark: https://github.com/usc-sail/fed-multimodal
- Local PDF: `C:\Users\cjkd3\Zotero\storage\I7QJAQW9\Feng 等 - 2023 - FedMultimodal A Benchmark for Multimodal Federated Learning.pdf`
- Scope / Subfield: Multimodal Federated Learning Benchmark
- Tags: Multimodal Federated Learning, Benchmark, Missing Modality, Data Heterogeneity, Robustness, Cross-modal Learning
- Status: DONE

## TL;DR

FedMultimodal 是面向多模态联邦学习的端到端 benchmark，覆盖 5 类应用、10 个公开数据集和 8 类模态，统一提供数据划分、特征处理、轻量多模态模型、融合方式、FL optimizer 与噪声模拟。它最大的价值不是提出新算法，而是让图像-文本、音频-文本、音频-视频、加速度计-陀螺仪等真正异质跨模态任务能够在统一联邦设置下比较。Benchmark 还提供缺失模态、缺失标签和错误标签模拟，其中缺失模态基线使用 Bernoulli 缺失、零填充与 attention mask。

## 毒舌评论

这篇论文终于给多模态 FL 提供了一把统一尺子，但这把尺子偏轻量、偏早期：跨模态覆盖广，却主要依赖预提取特征和基础 late fusion；缺失模态实验也只是填零加 mask，而不是研究如何恢复或重配置缺失语义。因此它非常适合做算法起点和公平比较，却不应被误解为已经解决了跨模态联邦学习。

## Research Question

- 研究对象：多模态数据下的联邦学习算法与鲁棒性评估。
- 具体问题：不同多模态 FL 工作使用各自的数据、划分、模型和评价协议，如何建立可复现、可公平比较的统一 benchmark？
- 为什么重要：真实联邦应用常同时包含音频、视频、文本、图像、可穿戴传感器和医疗信号；没有统一 benchmark 时，新算法的改进难以横向验证。
- 论文边界：该工作主要构建 benchmark 和 baseline，不提出针对缺失模态的高级恢复算法。

## Motivation and Basic Idea

- Motivation：已有 FL benchmark 主要聚焦单模态 CV、NLP、语音或医疗任务；已有 MMFL 方法又各自采用不同实验设置，难以公平比较。
- Basic idea：选择具有代表性的公开多模态数据集，把数据划分、特征提取、模型、融合、FL optimizer 和噪声模拟统一成端到端流水线。
- 如何回应 motivation：统一组件后，研究者可以只替换目标算法，并在相同数据、划分与评价协议下比较性能和鲁棒性。
- 我的判断：FedMultimodal 的主要创新属于 benchmark / infrastructure novelty，而非模型方法 novelty。

## Benchmark Coverage

| 应用 | 数据集 | 客户端划分 | 客户端数 | 模态 | 特征/输入 | 指标 |
|---|---|---:|---:|---|---|---|
| 情绪识别 | MELD | Natural | 86 | Audio + Text | MFCC + MobileBERT | UAR |
| 情绪识别 | CREMA-D | Natural | 72 | Audio + Video | MFCC + MobileNetV2 | UAR |
| 多媒体动作识别 | UCF101 | Synthetic | 100 | Audio + Video | MFCC + MobileNetV2 | Top-1 Acc |
| 多媒体动作识别 | MiT10 | Synthetic | 200 | Audio + Video | MFCC + MobileNetV2 | Top-1 Acc |
| 多媒体动作识别 | MiT51 | Synthetic | 2000 | Audio + Video | MFCC + MobileNetV2 | Top-1 Acc |
| 人体活动识别 | UCI-HAR | Synthetic | 105 | Accelerometer + Gyroscope | Raw | F1 |
| 人体活动识别 | KU-HAR | Natural | 66 | Accelerometer + Gyroscope | Raw | F1 |
| 医疗 | PTB-XL | Natural | 34 | 两组 ECG leads | Raw | F1 |
| 社交媒体 | Hateful Memes | Synthetic | 50 | Image + Text | MobileNetV2 + MobileBERT | AUC |
| 社交媒体 | CrisisMMD | Synthetic | 100 | Image + Text | MobileNetV2 + MobileBERT | F1 |

覆盖的独特模态包括 Audio、Text、Video、Image、Accelerometer、Gyroscope 和 ECG lead groups 等。相比 PEPSY 只验证多通道信号，FedMultimodal 提供了更接近真正异质跨模态的实验入口。

## Background

- 背景：联邦学习保护本地数据隐私，多模态学习利用不同信息源的互补性。
- 问题：现有 FL benchmark 多为单模态，现有 MMFL 工作实验协议分散。
- Gap：缺少覆盖多应用、多模态组合、数据异质性和现实数据损坏的统一 MMFL benchmark。

## Assumptions

- 客户端划分：有天然用户/说话人/机构 ID 时采用 natural partition；否则使用 Dirichlet 分布构造 synthetic non-IID。
- Synthetic partition 的异质性参数：`alpha ∈ {0.1, 5.0}`，分别代表高和低数据异质性。
- 特征处理：图像/视频和文本主要使用移动友好的预训练模型提取特征，而非在联邦过程中端到端训练大型 backbone。
- 缺失模态：每个模态的可用性服从 Bernoulli 分布，并对所有模态设置相同缺失率 `q`。

## Method / Benchmark Pipeline

FedMultimodal 包含六个主要组件：

1. **Non-IID Data Partitioning**：优先使用真实客户端标识；缺少真实标识的数据集使用 Dirichlet 划分。
2. **Feature Processing**：
   - Visual：MobileNetV2 / MobileViT；
   - Text：MobileBERT / DistilBERT；
   - Audio：MFCC；
   - Accelerometer、Gyroscope、ECG：raw data。
3. **Multimodal Models**：轻量 Conv-RNN、RNN 或 MLP，之后进行 late fusion 和分类。
4. **Fusion Schemes**：concatenation-based fusion 与 attention-based fusion。
5. **Federated Optimizers**：FedAvg、FedProx、FedRS、SCAFFOLD、FedOpt 等。
6. **Noise Factor Emulator**：模拟 missing modalities、missing labels、erroneous labels。

### 缺失模态模拟

- 每种模态独立以统一缺失率 `q ∈ {0.1, 0.2, 0.3, 0.4, 0.5}` 进行 Bernoulli 缺失。
- 缺失输入用 `0` 填充。
- Attention fusion 计算注意力分数时 mask 掉对应缺失数据点。
- 该流程用于评估 baseline 鲁棒性，不会生成缺失模态内容，也不会学习 PEPSY 式 data-missing profile。

## Evaluation

| 实验问题 | 设计 | 主要结论 |
|---|---|---|
| 不同 fusion 和 FL optimizer 谁更强？ | 10 个数据集，对比 concatenation / attention 与 FedAvg、FedProx、FedRS、FedOpt | Attention fusion 在多数数据集和高异质条件下更强；FedOpt 通常表现最好，但需要额外调参 |
| 多模态一定优于单模态吗？ | 比较最佳 unimodal 和 multimodal FL | 多模态多数情况下更好，但多数数据集的优势小于 5%，说明弱模态可能贡献有限 |
| 缺失模态影响多大？ | Bernoulli 缺失率 `q=0.1...0.5`，零填充 + attention mask | 缺失率低于 30% 时影响通常有限；50% 时性能明显下降；CrisisMMD 与 HAR 更敏感 |
| 缺失标签影响多大？ | 标签以不同概率缺失，不使用半监督补救 | 多数数据集在 50% 以下缺失标签时下降较小 |
| 错误标签影响多大？ | 使用 transition matrix 模拟标签噪声 | 错误标签通常比缺失模态和缺失标签造成更大性能下降 |

## Key Artifacts

- Figure 1：FedMultimodal 端到端架构，展示应用、数据划分、特征处理、模型、融合、optimizer 和 noise emulator。
- Table 2：10 个数据集、客户端数、模态组合、特征、指标和验证协议，是 benchmark 覆盖范围的核心证据。
- Figure 3：单模态与多模态 FL 对比，说明多模态优势并非总是巨大。
- Figure 4：不同缺失模态率下的相对性能变化。
- Figure 5-7：缺失标签、错误标签及三类 corruption 的影响比较。
- Table 4：不同 fusion 和 FL optimizer 的系统 benchmark 结果。

## Findings

- Attention fusion 在高数据异质性场景中通常优于简单拼接，但并非所有数据集都如此。
- 多模态模型通常优于单模态模型，但多数任务的差距不足 5%。
- 基础 attention mask 对一定比例的缺失模态具有容忍度，但在高缺失率和部分任务上明显失效。
- 错误标签通常比缺失模态或缺失标签更具破坏性。
- 跨模态任务难度差异很大，社交媒体和 MiT 动作识别明显更难。

## Strengths

- 提供了统一、开源、端到端的 MMFL benchmark。
- 数据覆盖真正异质模态，包括图像-文本、音频-文本和音频-视频，而不只是多通道信号。
- 同时覆盖 natural 和 synthetic client partition，便于研究真实和可控 Non-IID。
- 把缺失模态、缺失标签和错误标签纳入标准化鲁棒性评估。

## Limitations

- 缺失模态主要通过独立同分布 Bernoulli 随机缺失模拟，不能覆盖与标签、客户端资源或事件相关的非随机缺失。
- 缺失模态 baseline 仅采用零填充和 attention mask，缺少重建、蒸馏、跨模态生成或表示重配置方法。
- 主要使用预提取特征和轻量模型，不能充分反映端到端大型多模态模型在 FL 中的行为。
- 仅包含 concatenation 和 attention 两种基础 fusion。
- 尚未覆盖医疗影像、自动驾驶、虚拟现实等重要跨模态应用。

## My Takeaways

- FedMultimodal 是验证 PEPSY 是否能从多通道信号推广到真正跨模态场景的合适起点。
- 可优先选择 Hateful Memes / CrisisMMD 验证 image-text 缺失，MELD 验证 audio-text 缺失，CREMA-D / UCF101 验证 audio-video 缺失。
- 将 PEPSY 迁移到 FedMultimodal 的首要难点是不同模态 encoder 输出、语义和信息量不对称，不能再简单地用其他模态特征均值替换缺失模态。
- 更合理的后续实验应比较：零填充 + mask、跨模态重建、modality dropout、PEPSY-style controls 和预训练多模态模型适配。

## Related Papers

- PEPSY：学习 data-missing profile 并重配置缺失条件下的表示；原实验仅覆盖信号模态。
- FedMSplit：针对多模态 split networks 的缺失模态问题。
- CreamFL：通过 contrastive representation ensemble 处理多模态联邦学习。
- MultiBench：集中式多模态学习 benchmark，可作为 FedMultimodal 的非联邦对照。

## Open Questions

- PEPSY 的 control pool 在 image-text、audio-video 等语义差异巨大的模态组合上是否仍能有效？
- 当强模态缺失和弱模态缺失造成完全不同的信息损失时，统一缺失率 `q` 是否合理？
- 如何模拟真实的非随机跨模态缺失，例如相机故障与环境条件相关、文本缺失与用户行为相关？
- 在跨模态 FL 中，缺失模态、客户端 Non-IID 和 encoder 异构如何共同影响模型聚合？
