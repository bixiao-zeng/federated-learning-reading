# 【ICLR 2024】Multimodal Patient Representation Learning with Missing Modalities and Labels

## Metadata

- Title: Multimodal Patient Representation Learning with Missing Modalities and Labels
- Authors: Zhenbang Wu, Anant Dadu, Nicholas Tustison, Brian Avants, Mike Nalls, Jimeng Sun, Faraz Faghri
- Venue / Year: ICLR 2024
- Note Name: 【ICLR 2024】MUSE
- Paper: https://openreview.net/forum?id=Je5SHCKpPa
- PDF: https://openreview.net/pdf?id=Je5SHCKpPa
- Code: https://github.com/zzachw/MUSE
- Dataset / Artifact: MIMIC-IV, eICU, ADNI/TADPOLE；官方代码仓库采用 MIT License
- Scope / Subfield: 缺失模态与缺失标签条件下的多模态患者表示学习
- Tags: Multimodal Learning, Missing Modality, Missing Label, Semi-supervised Learning, Graph Neural Network, Contrastive Learning, Healthcare, EHR
- Status: DONE

## TL;DR

MUSE 研究比“缺失模态”更贴近临床现实的联合问题：患者可能同时缺少部分模态和任务标签。它把一个批次构造成患者节点—模态节点二部图，以患者的实际模态特征作为边属性；再通过模态边丢弃生成同一患者的另一种缺失视图，联合无监督同患者对比损失、同标签患者监督对比损失和分类损失，学习对模态缺失稳定且与任务相关的患者表示。MUSE 在 MIMIC-IV、eICU 和 ADNI 上优于插补、直接融合及图模型基线；加入无标签患者训练的 MUSE+ 通常进一步带来约 4 个百分点的绝对 AUC-ROC 优势。

## 毒舌评论

这篇论文真正漂亮的地方不是又造了一个医疗 GNN，而是把“患者少了哪些模态”直接变成图的边结构，并用 edge dropout 对准部署时的缺失机制；这比对 latent feature 随手做 dropout 更有因果味道。但论文最强的卖点——同时处理缺失模态和标签——在 eICU、ADNI 上仍主要靠随机遮掉 50% 标签来证明，且主体分析默认 missing completely at random。它证明了方法在受控缺失下很能打，却没有充分证明医院里由病情、费用、流程与选择偏差共同造成的真实缺失也同样听话。

## Research Question

- 研究对象：包含 EHR、时间序列、文本和影像生物标志物等多模态数据的患者表示。
- 小领域范围：Multimodal patient representation learning with missing modalities and labels。
- 具体问题：当每位患者拥有的模态组合不同，且部分患者没有下游任务标签时，如何利用全部可用数据训练一个对模态缺失稳健的预测模型？
- 为什么重要：完整模态和完整随访标签在临床数据中往往不可得；丢弃不完整患者既浪费数据，也会引入选择偏差。另一方面，多模态模型容易出现 modality collapse，即过度依赖少数强模态，部署时一旦这些模态缺失就明显退化。
- 论文边界：研究的是静态训练集上的预测，不处理跨机构分布漂移、时序模态不同步、缺失机制的因果建模、隐私保护或联邦学习。

## Motivation and Basic Idea

- Motivation：
  1. 插补方法需要学习“已有模态到缺失模态”的映射，依赖较强分布假设，并可能引入伪造噪声。
  2. 直接预测方法通常不能灵活扩展到任意模态组合，或者忽略患者间关系。
  3. 既有缺失模态方法通常仍需要完整标签，因此无法利用无标签患者。
  4. 普通多模态融合可能过度依赖某个强模态，产生 modality collapse。
- Basic idea：将患者与模态建模成二部图；删掉部分“患者—模态”边得到同一患者的另一种模态视图，要求两种视图下的患者表示一致，同时要求同标签患者的表示相近。
- 这个 idea 如何回应 motivation：二部图让不同患者自然拥有不同度数，因此无需为每种模态组合设计模型；edge dropout 直接模拟整种模态缺失；无监督同患者对比不依赖标签，可以吸收无标签患者；监督对比和分类目标则抑制“稳定但与任务无关”的特征。
- 作者给出的证据：主实验中 MUSE/MUSE+ 全面优于基线；缺失率升高时 MUSE+ 与基线差距扩大；同一患者在不同模态视图下的表示具有最高 cosine similarity 和最低 Euclidean distance；移除 edge dropout 或任一对比目标均导致性能下降。
- 我的判断：方法逻辑链是闭合的。最关键的设计并非二部图本身——GRAPE 已采用类似结构——而是用“结构级模态扰动 + 双重一致性目标”明确优化缺失模态下的不变性，并借无监督分支引入缺标签样本。

## Background

- 背景：临床决策通常联合人口统计、诊断、处置、用药、检验、生命体征、文本和影像等信息。
- 问题：现实数据同时存在模态缺失、标签缺失和模型对强模态过度依赖。
- Gap：已有方法多处理完整多模态，或只处理模态缺失；图方法虽然能容纳不规则模态组合，但并未显式要求同一患者在不同模态集合下保持表示一致，也通常无法利用无标签患者。

## Threat Model / Assumptions

- 数据假设：主体实验与既有工作一致，主要采用 missing completely at random（MCAR）设定；部分数据集保留真实模态/标签缺失，eICU 与 ADNI 的标签缺失通过随机遮蔽 50% 标签模拟。
- 图建模假设：同一批次内患者可以通过共享模态节点交换信息；患者表示因此依赖批次图中的其他患者。
- 增强假设：随机 edge dropout 能近似部署时的模态缺失；默认 dropout rate 为 15%。
- 标签假设：至少存在一部分有标签患者，监督对比与分类损失只在这些患者上计算。
- 不覆盖：系统性检查偏倚、因病情导致的非随机缺失、跨医院数据漂移、标签噪声、恶意或隐私攻击。附录只在 eICU mortality 上补充了一个人工 MNAR 场景。

## Method

### 1. 患者—模态二部图

给定患者 $p$ 的多模态输入

$$
X^{(p)}=(x_1^{(p)},x_2^{(p)},\ldots,x_M^{(p)})
$$

以及模态可用矩阵 $A\in\{0,1\}^{N\times M}$，MUSE 构造无向二部图

$$
G=(V,E),\qquad V=V_P\cup V_M
$$

其中：

- $V_P=\{u_1,\ldots,u_N\}$ 是患者节点；
- $V_M=\{v_1,\ldots,v_M\}$ 是模态节点；
- 若患者 $p$ 拥有模态 $m$，即 $A[p,m]=1$，则建立边 $e_{u_pv_m}$；
- 模态并不是患者节点的普通输入，而是边属性：

$$
e_{u_pv_m}^{(0)}=\operatorname{Encoder}_m(x_m^{(p)}).
$$

模态节点初始化为对应模态的 one-hot 向量，患者节点初始化为全一向量。这样，模态缺失直接表现为边不存在，不需要补零或为每种模态组合建立独立网络。

### 2. 带边属性的消息传播

模型使用两层、支持 edge attributes 的 GraphSAGE。一般形式为：

$$
h_i^{(l)}=\operatorname{Aggregate}^{(l)}
\left(
h_i^{(l-1)},
\left\{
\operatorname{Message}^{(l)}
(h_j^{(l-1)},h_i^{(l-1)},e_{ji}^{(l-1)})
\right\}_{j\in\mathcal N(i)}
\right).
$$

图上的多跳传播使患者能够经由共享模态节点间接交换信息，也使模态节点聚合不同患者对该模态的观测。最终取得患者节点矩阵

$$
Z=\operatorname{GNN}(G)\in\mathbb R^{N\times d}.
$$

### 3. 结构级数据增强

从原图 $G$ 随机删除部分患者—模态边，得到增强图 $G'$。两张图表示同一批患者，但模态可用集合不同：

$$
Z'=\operatorname{GNN}(G').
$$

原图和增强图共享同一个 GNN，即 Siamese encoder。edge dropout 与 feature dropout 的差别很重要：前者删除完整模态关系，后者只破坏模态内部特征，不能真实模拟整模态不可得。

### 4. Mutual-Consistent Contrastive Learning

无监督对比损失将同一患者在两种模态视图下的表示作为正样本：

$$
\mathcal L_{\text{Unsup}}(Z,Z')
=-\sum_{p=1}^{N}
\log
\frac{\exp(s(z_p,z'_p)/\tau)}
{\sum_{q=1}^{N}\exp(s(z_p,z'_q)/\tau)}.
$$

它要求表示对模态组合变化保持一致，只使用“同一患者”这一自监督信号，因此无标签患者也能参与。

监督对比损失仅选择标签可用的患者，把同标签患者作为正对、异标签患者作为负对：

$$
\mathcal L_{\text{Sup}}(Z,Y,L).
$$

其作用是让模型不只学习模态无关特征，还聚焦于 label-decisive features。实现上，监督对比损失作用于患者表示经过 projection layer 后的张量，而分类损失直接作用于原始患者表示，避免两个目标完全重复。

分类头为：

$$
\hat Y=\operatorname{MLP}(Z),
$$

并只对有标签患者计算交叉熵：

$$
\mathcal L_{\text{CE}}
=\sum_{p=1}^{N}\mathbf 1(L[p]=1)
\operatorname{CE}(\hat y_p,y_p).
$$

总目标是三项损失的加权和：

$$
\mathcal L
=\lambda_1\mathcal L_{\text{Unsup}}
+\lambda_2\mathcal L_{\text{Sup}}
+\lambda_3\mathcal L_{\text{CE}}.
$$

### 5. MUSE 与 MUSE+

- MUSE：只用有标签患者训练，用于与只能使用标签监督的基线公平比较。
- MUSE+：有标签和无标签患者都进入图及无监督对比分支；监督对比和分类损失仍只使用有标签患者。
- 推理阶段：不再构造增强分支，只将实际可用模态构成的图送入 GNN 和 MLP。

### 6. 实现细节

- GNN：2-layer GraphSAGE with edge attributes。
- Edge dropout rate：15%。
- Temperature $\tau=0.05$。
- 数据划分：70% / 10% / 20% train / validation / test。
- 训练：100 epochs，以验证集指标选择最佳模型。
- 统计：对测试集 bootstrap 1000 次，报告均值和标准差；用 independent two-sample t-test 检验 MUSE 对最佳基线的提升。
- 编码器：
  - 人口统计与序列医疗编码：2-layer Transformer，embedding 128，2 heads；
  - lab/vital time series：2-layer bidirectional RNN；
  - clinical notes：TinyBERT，最大长度 256；
  - ADNI 已提取特征：2-layer MLP；
  - 各模态投影至 128 维共同空间。

## Evaluation

### 数据与任务

| 数据集 | 实际使用规模 | 模态 | 任务 | 标签缺失 |
|---|---:|---|---|---|
| MIMIC-IV | 172,113 patients / 389,761 admissions | demographics, diagnosis, procedure, medication, labs, notes | 90-day mortality；15-day readmission | 真实缺失：mortality 58%，readmission 41% |
| eICU | 120,892 patients / 158,248 admissions | demographics, diagnosis, procedure, medication, labs, vital signals | ICU mortality；15-day ICU readmission | 随机遮蔽 50% |
| ADNI/TADPOLE | 1,572 patients / 5,974 visits | DTI, PET, MRI biomarkers | NC / MCI / AD 三分类 | 随机遮蔽 50%；原始统计约 8% 缺失 |

### 对比方法

- 插补：CM-AE、SMIL。
- 直接预测：MT。
- 图方法：GRAPE、HGMF、M3Care。
- 公平设置：所有基线与 MUSE 只用有标签患者；MUSE+ 额外使用无标签患者。

### 主实验结果

MIMIC-IV 和 eICU：

| Method | MIMIC Mortality ROC | MIMIC Readmission ROC | eICU Mortality ROC | eICU Readmission ROC |
|---|---:|---:|---:|---:|
| Best baseline | 0.8896 | 0.7085 | 0.8964 | 0.7663 |
| MUSE | 0.9004 | 0.7152 | 0.9017 | 0.7709 |
| MUSE+ | **0.9201** | **0.7351** | **0.9332** | **0.8003** |

MUSE 在相同有标签训练集上已优于最佳基线；MUSE+ 利用无标签患者后进一步扩大优势。AUC-PRC 上也保持一致趋势，例如 MIMIC mortality 从最佳基线 0.4603 提升至 MUSE 0.4735、MUSE+ 0.4883。

ADNI：

| Method | Macro AUC-ROC | Balanced Accuracy |
|---|---:|---:|
| M3Care | 0.9101 | 0.7822 |
| MUSE | 0.9158 | 0.7973 |
| MUSE+ | **0.9309** | **0.8291** |

ADNI 样本较小，bootstrap 标准差普遍更大，但方法排序与 ICU 场景一致。

### 缺失率分析

- 模态随机缺失率：$\{0.1,0.2,0.3,0.5,0.7\}$。
- 标签随机缺失率：$\{0,0.1,0.2,0.3,0.4\}$。
- 任务：MIMIC-IV mortality。
- 结果：MUSE/MUSE+ 在全部配置下优于 GRAPE 和 M3Care；缺失率越高，MUSE+ 的优势通常越明显。

### 表示分析

作者比较同一患者在完整视图与随机遮蔽 30% 模态视图下的表示距离：

- MUSE/MUSE+ 的 cosine similarity 最高。
- Euclidean distance：CM-AE 0.7352，SMIL 0.6337，MT 0.5839，GRAPE 0.3001，M3Care 0.3286，MUSE 0.2437，MUSE+ **0.2232**。

这直接支撑“模型学到了对模态组合更稳定的表示”，而不仅是最终分类分数更高。

### 消融

| Variant | MIMIC Mortality | eICU Mortality | ADNI |
|---|---:|---:|---:|
| w/o Edge Dropout | 0.8830 | 0.8655 | 0.8811 |
| w/o Contrastive Loss | 0.8846 | 0.8895 | 0.9040 |
| w/o Supervised CL | 0.8934 | 0.8916 | 0.9086 |
| w/o Unsupervised CL | 0.8901 | 0.8911 | 0.9003 |
| MUSE | **0.9004** | **0.9017** | **0.9158** |

edge dropout 的消融退化最明显，说明“按模态删边”的结构增强是方法成立的核心；两类 contrastive loss 均有独立贡献。

### 非随机缺失补充实验

作者在 eICU mortality 上让“缺 vital signals 的死亡患者”具有更高标签缺失率，同时保持总体标签缺失率 50%。AUC-ROC 结果为：

- M3Care 0.8700；
- MUSE 0.8753；
- MUSE+ 0.8911。

该实验说明方法不只在纯随机缺失下有效，但它仍是单任务、人工定义的 MNAR 场景，不能等价为真实临床选择机制验证。

## Key Artifacts

- Figure 1：区分完整多模态、仅缺失模态、同时缺失模态与标签三种问题设定；支撑论文的问题新颖性。
- Figure 2：完整框架图；展示二部图、edge dropout、Siamese GNN、无监督/监督对比与分类三条训练信号。
- Equations 2–4：定义模态特征作为边属性，以及带边属性的图消息传播。
- Equations 5–6：分别定义同患者跨模态视图一致性和同标签患者一致性，是方法的核心创新。
- Table 1 / Table 2：三数据集主结果，支撑总体有效性与无标签患者的增益。
- Figure 3：缺失率变化实验，支撑鲁棒性。
- Figure 4 / Table 8：直接衡量跨模态视图表示稳定性，支撑机制解释。
- Table 3 / Table 7：证明 edge dropout、无监督对比、监督对比均有贡献。
- Algorithm 1：给出训练和推理流程，说明推理时无需增强分支。

## Findings

- 二部图能用统一结构容纳任意模态子集，并让不同患者通过共享模态节点交换信息。
- modality-level edge dropout 比普通 feature dropout 更符合实际缺失模态语义，是最关键的增强设计。
- 无监督同患者对比使缺标签患者可参与训练；MUSE+ 的稳定增益表明，这些患者不只是“可以使用”，而是提供了有效表示监督。
- 仅保证跨视图一致还不够；监督对比和分类损失负责从大量模态无关信息中筛出与任务相关的特征。
- 图方法整体优于插补和普通 Transformer 融合，说明患者间关系在这些医疗任务上确实有价值。

## Strengths

- 问题设定真实且明确：把缺失模态与缺失标签放入同一框架。
- 方法与问题高度匹配：缺模态对应缺边，缺标签对应仅关闭监督损失，结构自然。
- 与最近邻工作区分清楚：相对 GRAPE，关键增量是跨模态缺失视图的一致性；相对 M3Care/HGMF，图结构更简单且可扩展。
- 实验证据链完整：主结果、缺失率、表示距离、消融、运行时间和一个 MNAR 补充实验互相支持。
- 代码已公开，包含 MIMIC-IV/eICU 的预处理和训练流程；论文也给出了主要架构与超参数。

## Limitations

- 主体实验主要建立在 MCAR 或随机遮蔽标签上；真实临床缺失通常与病情、医生决策、费用、资源和医院流程相关。
- eICU 与 ADNI 的“缺标签能力”主要通过人工遮蔽 50% 标签验证，缺失模式较理想化。
- 二部图让患者表示依赖同批其他患者，可能对 batch composition、类别比例和部署批大小敏感；论文没有系统分析这种 transductive-style 依赖。
- 训练图中的患者间信息传播可能引入跨 split 或批处理设计风险。论文描述按 train/validation/test 划分，但复现时仍需确认图只在各自 split/batch 内构造，避免泄漏。
- 对比学习依赖足够的负样本与同标签正样本；类别极不平衡时，监督对比项的采样质量可能不稳定。
- 没有与专门的半监督学习、missing-label learning 或 pseudo-labeling 基线系统比较，因此 MUSE+ 的增益有多少来自“使用无标签数据”本身、有多少来自其特定图对比设计，仍不完全清楚。
- ADNI 的样本量小且方差较大；三个数据集均属公开研究数据，跨医院外部验证不足。
- 复现成本不低：MIMIC-IV/eICU 需资质访问与复杂预处理；论文训练环境为多张 RTX A6000。官方仓库提供代码，但 README 的复现说明主要覆盖 MIMIC-IV/eICU，ADNI 流程不够显式。

## My Takeaways

- 对缺失模态问题，增强操作应与缺失的语义层级一致：缺整模态就删整条模态关系，而不是只对特征维度加噪声。
- “不变性 + 判别性”是处理 modality collapse 的通用组合：无监督目标学跨模态稳定成分，监督目标防止模型稳定在无用信息上。
- 在缺标签场景中，不必急于伪标签；先设计一个不依赖标签、又与部署扰动一致的自监督任务，往往更稳。
- 若将该思路迁移到联邦多模态学习，可以把每个客户端的患者—模态子图作为本地图结构，但需重新处理跨客户端共享模态节点、对比负样本与隐私泄漏问题。
- 与 PEPSY 的互补关系很清楚：MUSE 在集中式医疗数据中通过图对比学习缺失不变性；PEPSY 则在联邦设置中显式学习并聚合 data-missing profiles。

## Related Papers

- 前置阅读：
  - GRAPE: Handling Missing Data with Graph Representation Learning（相似二部图结构）。
  - HGMF: Heterogeneous Graph-based Fusion for Multimodal Data with Incompleteness。
  - M3Care: Learning with Missing Modalities in Multimodal Healthcare Data。
- 可对比论文：
  - SMIL: Modality Incomplete Multimodal Learning with Bayesian Meta-Learning。
  - Mitigating Modality Collapse in Multimodal VAEs via Impartial Optimization。
  - PEPSY: Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data。
- 关键差异：
  - 相比插补方法，MUSE 不重建原始缺失模态。
  - 相比 GRAPE，MUSE 显式对齐同一患者的不同模态视图。
  - 相比 M3Care，MUSE 使用一个统一患者—模态二部图，而非每模态患者相似图。
  - 相比 PEPSY，MUSE 是集中式、样本关系驱动的方法；PEPSY 面向联邦客户端的缺失上下文与表示重配置。

## Open Questions

- 若缺失机制是 MNAR，且“缺失本身”携带诊断信息，强制表示对缺失模式不变是否会抹掉有用信号？
- 同一患者的两种缺失视图是否总应被完全拉近？某些关键模态缺失后，合理的不确定性应该如何保留？
- 如何将 batch graph 改造成 inductive、单患者即可推理且不依赖同批其他患者的模型？
- 与 FixMatch、Mean Teacher、pseudo-labeling 或 consistency regularization 等半监督方法组合后，MUSE+ 是否仍有独立优势？
- 在联邦场景中，能否只交换模态节点原型或图统计量，从而保留 MUSE 的跨患者关系建模能力？
