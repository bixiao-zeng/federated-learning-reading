# 【NeurIPS 2025】Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data

## Metadata

- Title: Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data
- Authors: Duong M. Nguyen, Trong Nghia Hoang, Thanh Trung Huynh, Quoc Viet Hung Nguyen, Phi Le Nguyen
- Venue / Year: NeurIPS 2025
- Note Name: 【NeurIPS 2025】PEPSY
- Paper: https://papers.neurips.cc/paper_files/paper/2025/hash/29978617eb250d16969e3681121ad6d7-Abstract-Conference.html
- OpenReview: https://openreview.net/forum?id=rMqQdJJz5r
- Code: https://github.com/nmduonggg/PEPSY
- Local PDF: `C:\Users\cjkd3\Zotero\storage\K5LRK39D\Nguyen 等 - 2025 - Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data.pdf`
- Dataset / Artifact: PTBXL, Sleep-EDF; official code repository above
- Scope / Subfield: Multimodal Federated Learning with Missing Data
- Tags: Federated Learning, Multimodal Learning, Missing Modality, Missing Features, Representation Reconfiguration, Medical Signal Classification, Non-IID
- Status: DONE

## TL;DR

这篇论文研究一个更真实的联邦多模态学习问题：不同客户端不仅可能拥有不同模态集合，每个模态内部还可能存在输入特征缺失。作者提出 PEPSY，用客户端侧可学习的 data-missing profile / embedding controls 编码本地缺失模式，并把这些控制信号用于重配置共享表示，使全局聚合后的表示能适配客户端局部数据上下文。实验在 PTBXL 和 Sleep-EDF 上验证，PEPSY 在 IID、Non-IID、训练/测试缺失统计不一致、严重缺失等场景下整体优于 FedProx、MIFL、FedInMM、FedMSplit、FedMAC，论文声称严重数据不完整时最高提升 36.45%。

## 毒舌评论

PEPSY 的价值在于把“缺失模式”从要被补零或绕开的麻烦，提升成一种可以学习、同步、聚合的控制变量；这比单纯做缺失模态补全更像一个顶会问题。但它的脆弱点也很明显：实验仍是基于公开数据集的人为缺失模拟，且所有模态都是信号模态，离真实多机构、多设备、多模态医疗系统中的异步采集、标签偏差、协议差异还有距离。换句话说，这篇论文把问题定义推前了一大步，但真实世界的“脏”还没有完全进实验室。

## Research Question

- 研究对象：多模态联邦学习中，客户端数据存在异构缺失时的表示学习与模型聚合。
- 小领域范围：Federated Multimodal Learning with missing modalities and missing input features。
- 具体问题：当每个客户端看到的模态子集不同，且每个样本/模态内部也可能缺特征时，如何让本地表示在全局聚合后仍然可用，并能适配客户端自身的缺失上下文？
- 为什么重要：真实 IoT、可穿戴健康、分布式医疗和环境感知系统中，数据不能集中，设备类型不同，传感器失效和间歇记录普遍存在。普通 FL 默认客户端特征空间一致，多模态 FL 又常默认模态完整，这两个假设都不稳。
- 论文边界：论文主要验证信号型多模态数据，采用模拟缺失模式；没有直接处理跨机构标签标准不一致、跨模态时间不同步、真实设备协议差异或隐私攻击。

## Motivation and Basic Idea

- Motivation：传统 FL 聚合默认本地模型基于共同特征空间训练；但在 MMFL 中，客户端可能缺模态，也可能在模态内部缺输入特征。本地模型会把不同缺失视角映射到不兼容的表示空间，直接聚合会导致表示错位和性能下降。
- Basic idea：为每个客户端学习一个 data-missing profile，即一组 embedding controls，用来编码该客户端“缺什么、怎么缺”。这些控制信号参与本地表示构造，把带缺失偏差的表示重配置成更接近 data-complete 的表示。
- 这个 idea 如何回应 motivation：服务器看不到原始数据，不知道每个客户端的缺失上下文；客户端知道自己的缺失模式，但无法独自解释全局表示。PEPSY 让客户端把缺失上下文压缩成可共享的 embedding controls，服务器再聚合同类缺失 profile，从而在不共享数据的情况下传播“缺失模式知识”。
- 作者给出的证据：Fig. 1 显示 FedAvg 随缺失程度上升快速退化而 PEPSY 更稳；Table 1/2 显示主实验和训练-测试缺失统计不一致时 PEPSY 多数领先；Fig. 3、Fig. 4、Fig. 5 支撑 alignment loss、profile 和表示对齐的必要性。
- 我的判断：核心想法是合理的。真正新颖的地方不是“处理缺失模态”，而是把缺失模式 profile 作为模型重配置的条件信号，并让这些 profile 在联邦服务端被非参数聚合。

## Background

- 背景：多模态学习希望利用不同模态互补信息；联邦学习希望多个客户端不共享原始数据也能协同训练。
- 问题：在真实 MMFL 中，客户端可能拥有不同模态组合，也可能样本级/模态级输入特征缺失。不同客户端因此学习到的表示空间不一致。
- Gap：已有 MMFL 方法多假设模态完整，或只处理客户端模态集合不同；缺失模态/缺失特征方法通常分别处理两类问题。论文认为同时存在 missing modalities 和 missing input features 的一般设定仍未被充分解决。

## Threat Model / Assumptions (Optional)

- 参与者能力：标准横向联邦学习设置，K 个客户端保存本地数据，服务器聚合参数和 profile；没有恶意客户端建模。
- 隐私假设：客户端不上传原始数据；论文没有证明 embedding controls 不泄露缺失模式或敏感属性。
- 数据假设：缺失模式通过 `p_s` 和 `p_m` 模拟；训练集分给客户端，可设 IID 或 Non-IID。Non-IID 用 Dirichlet 分布，论文实现细节中为 alpha = 0.5。
- 不覆盖的情况：恶意更新、隐私攻击、防推断机制、真实医院标注不一致、跨模态时间错配、跨设备协议偏移。

## Method

- 核心思路：学习一个可共享、可聚合、可查询的 data-missing profile，用它重配置客户端本地表示。
- 核心思路是怎么想到的：既然客户端表示错位来自缺失上下文不同，那么只补缺失值不够；模型需要知道本地缺失模式，并把这种模式作为条件信号注入表示学习。
- 从 motivation 到 method 的逻辑链：
  1. 缺失模态/缺失特征导致本地表示偏置。
  2. 服务器无法观察数据，无法直接理解客户端缺失上下文。
  3. 客户端可学习一组 embedding controls，编码本地缺失 profile。
  4. 查询相关 controls 后得到 missing-pattern representation。
  5. 将 modality-specific、data-specific、missing-pattern 三类信息拼接并重配置为更完整的表示。
  6. 服务器对普通神经参数用 FedAvg，对 data-missing profile 用非参数聚类/概率同步处理错位。
- 关键设计取舍：
  - 不直接补原始模态，而是在表示空间做 reconfiguration，避免依赖完整公共多模态数据或全模态 foundation model。
  - 不直接平均 embedding controls，因为不同客户端本地 profile 的顺序和大小都可能不同；因此采用非参数聚类式聚合。
  - 用 top-k 相关 controls，避免一个样本把缺失信息分散到太多 embedding 中。
- 系统流程或算法步骤：
  1. Client training：每个客户端从本地不完整多模态数据提取 modality-specific features `w_mod` 和 data-specific features `w_ins`。
  2. Missing profile query：用 query-key matching 从本地 profile `Ψ={ψ_p}` 中选择最相关的 embedding controls。
  3. Reconfiguration：得到 missing-pattern representation `w_mis`，拼接为 `w=[w_mod, w_ins, w_mis]`，再通过对比正则和注意力式融合得到最终表示。
  4. Local objective：优化任务损失、data-specific loss、reconfiguration contrastive loss，并最大化 relevance regularizer。
  5. Server aggregation：普通模型参数用 FedAvg；data-missing profiles 通过 PFPT 式非参数聚类/概率同步聚合，形成全局 profile。
- 关键定义 / 公式 / 不变量：
  - 标准 MMFL 目标：最小化所有客户端平均损失 `1/K * Σ_k l_k(θ)`。
  - 重配置模型：`f(D_k; θ, Ψ) = f_p(f_e(D_k; θ_e) ⊙ r(D_k; Ψ); θ_p)`，其中 `r` 返回与样本缺失模式相关的 embeddings。
  - Data-specific feature reconstruction：缺失模态特征用已观测模态特征平均近似。
  - Data-specific loss `L_ds`：拉近同一样本不同可用模态的 normalized features，推远不同样本特征。
  - Relevance `γ(x_di, ψ_p)`：用 cosine similarity 衡量样本模态与某个 embedding control 的相关度。
  - Local objective：`L = L_task + λ(L_ds + L_rc) - ηR`。
- 实现细节：使用 Inception Network 作为 modality encoder；embedding dimension `C=128`；总客户端 `K=32`，每轮采样 10 个客户端；本地训练 3 epoch；PTBXL 训练 1000 communication rounds，EDF 训练 500 rounds；优化器为 SGD。

## Evaluation

| 实验阶段 | 顶会科研问题 | 实验设计 | 指标/设置 | 主要结果 | 支撑的结论 |
|---|---|---|---|---|---|
| 动机实验 | 缺失程度是否真的破坏普通 FL？ | Fig. 1 比较 FedAvg 与 PEPSY 随缺失程度变化的趋势 | 缺失程度增加 | FedAvg 快速退化，PEPSY 更稳定 | 表示错位是关键问题，不能只用普通聚合 |
| 主实验 1 | 同一训练/测试缺失统计下是否优于基线？ | Table 1，PTBXL 和 EDF，IID/Non-IID，多组 `p_m/p_s` | Accuracy on server dataset | PEPSY 在大多数场景领先；EDF IID 下论文称 40/40 case 最高 | profile reconfiguration 在常规缺失模拟中有效 |
| 主实验 2 | 训练和测试缺失统计不一致时是否泛化？ | Table 2，客户端训练缺失统计与服务器测试缺失统计不同 | Accuracy | PEPSY 多数测试缺失条件下领先；论文报告平均提升和极端场景优势 | 不是只拟合单一缺失模式 |
| 聚合消融 | profile 聚合是否必要？ | Table 3 比较 FedAvg、FedProx、SynFedProx、PEPSY | Accuracy | PEPSY 在更高缺失率下更稳 | profile alignment / probabilistic synchronization 有贡献 |
| Alignment loss 消融 | 理论中的对齐损失是否影响鲁棒性？ | Fig. 3 调整 alignment weight，在完整训练、完整/极端缺失测试下比较 gap | Performance gap | alignment weight 增大时 gap 降低 | 支撑 Theorem 3.1 中 `L_ds` 控制缺失偏差的解释 |
| Data-missing profile 消融 | profile 本身是否有用？ | Fig. 5a 比较 PEPSY 与 PEPSY-NP | Accuracy | 缺失模态越多，profile 带来的增益越明显 | 缺失模式越复杂，profile 越关键 |
| 表示可视化 | 是否真的学到更对齐的表示？ | Fig. 4 t-SNE 比较 PEPSY、FedProx、FedMAC | 2D modality representation | PEPSY 表示更清晰对齐，FedProx/FedMAC 对齐不足 | 性能提升不是纯参数调优，和表示对齐相关 |
| Profile 收敛分析 | embedding controls 是否稳定学习？ | Fig. 5b 可视化 global profile 500 轮轨迹 | t-SNE / centroid trajectory | centroid 更新距离逐渐变小，embedding spread 反映多样缺失模式 | profile 能收敛并捕获客户端缺失差异 |

## Key Artifacts

- 关键图：
  - Fig. 1：论文动机图，展示 FedAvg 在缺失数据增加时退化，并区分两类 missing events：missing modalities 和 missing input features。
  - Fig. 2：PEPSY 总体 server-client workflow 与 client design。支撑“profile 选择 -> 表示重配置 -> 服务端聚合”的方法链。
  - Fig. 3：alignment loss 对完整/极端缺失测试 gap 的影响。支撑理论损失项和鲁棒性的关系。
  - Fig. 4：不同方法的 modality representation t-SNE。支撑 PEPSY 改善表示对齐。
  - Fig. 5：data-missing profile 消融、global control embeddings 的收敛和多样性。
  - Fig. 6：附录中的缺失模式模拟示例，说明 `p_m`、`p_s` 如何构造 missing matrix。
- 关键表：
  - Table 1：PTBXL / EDF 在 IID 和 Non-IID、不同缺失统计下的主结果。
  - Table 2：训练和测试缺失统计不同的泛化实验。
  - Table 3：server aggregation 消融。
  - Table 4：超参数设置，支撑复现实验配置。
- 关键公式 / 定义 / 算法：
  - Eq. 1：标准 MMFL 优化目标。
  - Eq. 2-4：引入 data-missing profile 后的 reconfigured formulation。
  - Eq. 5：缺失模态的 data-specific feature reconstruction。
  - Eq. 6：`L_ds`，约束同一样本跨模态特征接近。
  - Eq. 7-8：embedding controls 的相关度与 top-k relevance regularizer。
  - Eq. 9 / Theorem 3.1：缺失条件输出与全模态输出之间的期望偏差界。
  - Eq. 10-11：附录中的 missing matrix 与 incomplete dataset 构造。
- 这些证据分别支撑哪些结论：
  - Fig. 1 + Table 1 支撑“普通 FL / 单类缺失方法不够”。
  - Fig. 2 + Eq. 2-8 支撑“缺失 profile 可以作为重配置信号”。
  - Theorem 3.1 + Fig. 3 支撑“对齐损失与缺失鲁棒性有关”。
  - Fig. 5 + Table 3 支撑“profile 与其聚合机制是必要组件”。

## Findings

- 发现 1：当缺失程度较低时，PEPSY 与强 baseline 的差距较小；当缺失程度升高或 Non-IID 更强时，PEPSY 的优势更明显。
- 发现 2：FedMAC 在部分场景是强 baseline，但面对两类缺失事件同时存在时仍不如 PEPSY 稳定。
- 发现 3：profile 的贡献随缺失模式复杂度增加而变大，说明论文核心不是普通多模态融合，而是缺失上下文建模。
- 发现 4：训练/测试缺失统计不一致时 PEPSY 仍保持优势，说明它学习到的不是单一缺失 mask，而是较泛化的 reconfiguration signal。

## Strengths

- 论文最有说服力的地方：问题设定比已有工作更一般，同时覆盖客户端模态集合异构和模态内部特征缺失，这确实更接近真实 MMFL。
- 方法优势：把 data-missing profile 建成可学习、可查询、可聚合的 embedding controls，使缺失模式成为模型适配条件，而不是单纯的噪声或补全目标。
- 实验优势：覆盖 PTBXL、Sleep-EDF、IID/Non-IID、不同 `p_m/p_s`、训练/测试缺失统计不一致、聚合消融、profile 消融、alignment loss 消融和表示可视化。
- 相比已有工作的有效推进：相较 FedMSplit/MIFL 只偏向 modality-missing，FedInMM/FedMAC 偏向 feature-missing，PEPSY 处理两类 missing events 的组合。

## Limitations

- 适用范围限制：实验数据主要是 signal-based modalities，尚不能直接说明图像-文本-表格-病理等更异构模态场景同样有效。
- 缺失机制限制：缺失由 `p_m` 和 `p_s` 人工模拟，未验证真实机构中非随机缺失、设备故障机制、患者流程导致的系统性缺失。
- 隐私与安全限制：embedding controls 会显式编码客户端缺失模式，论文没有分析这些 profile 是否可能泄露机构设备配置、病种分布或工作流信息。
- Baseline 限制：虽然 baseline 覆盖了 FL、missing modality、feature missing 三类，但没有和大规模预训练多模态模型、真实公共数据检索补全或强生成补全方案在同一联邦设定下比较。
- 复现成本：1000 communication rounds、32 clients、A6000 GPU，对普通复现实验仍有一定成本。

## My Takeaways

- 对联邦多模态数据质量研究的启发：可以把“数据质量/缺失模式”建模为可学习的 profile，而不是只在预处理阶段统计缺失率。
- 可复用的方法：client-side context embedding、profile aggregation、reconfiguration regularization 可迁移到数据质量评分、客户端可信度估计、异构数据治理等方向。
- 后续问题：
  - 能否把 data-missing profile 扩展为更通用的 data-quality profile，包含噪声、标签质量、时间对齐、设备协议等质量维度？
  - profile 聚合是否会泄露客户端敏感属性？是否需要差分隐私或安全聚合？
  - 当缺失不是随机的，而与标签、疾病严重程度、机构资源水平相关时，PEPSY 是否仍稳定？
  - 能否把 PEPSY 与公共数据检索、生成式补全、foundation model adapter 结合？

## Related Papers

- 前置阅读：FedAvg, FedProx, FedMSplit, MIFL, FedInMM, FedMAC, Harmony, FedMM, FedDAT。
- 后续阅读：federated prompt-tuning with heterogeneous and incomplete multimodal client data；modality-heterogeneous client drift；cross-modal augmentation by retrieval for MMFL missing modalities。
- 可对比论文：
  - FedMSplit：更关注 split networks 下的 multimodal federated multi-task learning。
  - MIFL：更关注 missing modalities 的 contrastive / graph-based route。
  - FedInMM：更偏 incomplete modalities 的鲁棒 MMFL。
  - FedMAC：处理 partial-modality missing，使用 cross-modal aggregation 和 contrastive regularization。
- 最接近的相关工作：FedMAC，因为它也是面向 incomplete / partial modality MMFL 的强 baseline。
- 关键差异：PEPSY 显式学习并聚合 data-missing profile，把缺失上下文作为 reconfiguration signal；FedMAC 更偏 cross-modal aggregation 和 contrastive regularization。

## Open Questions

- PEPSY 的 embedding controls 是否可以解释为客户端数据质量画像？如果可以，如何量化每个 control 对具体缺失模式的贡献？
- 如果真实缺失模式与标签强相关，profile 是否会强化偏差？
- 服务端聚合 profile 时，是否应该考虑客户端样本量、质量、标签分布或不确定性，而不仅是 profile 相似性？
- 在医疗多中心场景中，缺失 profile 是否会泄露医院设备能力或诊疗流程？
- PEPSY 能否推广到异步多模态，比如影像时间点和文本诊断时间点不一致？

## From Motivation Experiment to Final Experiment

| 阶段 | 论文中的科研动作 | 为什么这样设计 | 对应证据 | 最终服务的主张 |
|---|---|---|---|---|
| 1. 现象暴露 | 用 FedAvg 展示缺失数据增加时性能退化 | 先证明问题真实存在，而不是为了方法硬造问题 | Fig. 1a | 普通 FL 聚合无法处理异构缺失导致的表示错位 |
| 2. 问题拆解 | 区分 missing modalities 与 missing input features | 把现有工作覆盖范围拆清楚，说明论文设定更一般 | Fig. 1b, Introduction | 同时处理两类缺失事件是必要的新问题 |
| 3. 研究缺口 | 指出现有 MMFL、imputation、FM、FL-specific missing 方法的不足 | 建立 PEPSY 的位置：不是简单补全，也不是单一缺失事件方法 | Related Work, Limitation of Prior Work | 需要一种能表达客户端缺失上下文的机制 |
| 4. 核心假设 | 缺失模式可以被压缩为 data-missing profile，并用于重配置表示 | 把“缺失”转成可学习控制信号 | Solution Vision | profile 是连接客户端本地上下文和服务器全局表示的桥 |
| 5. 方法落地 | 设计 modality-specific、data-specific、missing-pattern 三类表示 | 分离不同信息源，避免缺失模式污染全部表示 | Fig. 2b, Eq. 5-8 | PEPSY 能构造更接近 data-complete 的表示 |
| 6. 联邦聚合 | 用概率同步/非参数聚合处理 profile 顺序和数量不一致 | 直接平均 profile 会错位；缺失复杂度不同导致 profile size 可变 | Section 2.3, Table 3 | profile 可以跨客户端协作，而不是仅本地使用 |
| 7. 理论支撑 | 给出缺失输出与全模态输出偏差界 | 解释为什么 `L_ds` 与缺失鲁棒性有关 | Theorem 3.1, Fig. 3 | 方法设计和鲁棒性之间有可解释关系 |
| 8. 主实验验证 | 在 PTBXL / EDF、IID / Non-IID、不同 `p_m/p_s` 下比较 baselines | 覆盖从轻度到严重缺失、从同分布到异分布的主场景 | Table 1 | PEPSY 在复杂缺失条件下更强 |
| 9. 泛化验证 | 训练缺失统计和测试缺失统计不同 | 检查模型是否只记住某种缺失模式 | Table 2 | PEPSY 对未知/变化缺失统计更稳 |
| 10. 组件归因 | 消融 aggregation、alignment loss、data-missing profile | 证明不是单纯调参或 encoder 强，而是关键组件有效 | Table 3, Fig. 3, Fig. 5 | profile、alignment、同步聚合共同贡献性能 |
| 11. 机制解释 | t-SNE 可视化 modality representations 和 global control embeddings | 检查方法是否真的产生表示对齐和 profile 收敛 | Fig. 4, Fig. 5b | 性能提升与表示重配置机制一致 |
| 12. 最终结论 | 宣称 PEPSY 是复杂缺失 MMFL 的 flexible and stable solution | 汇总主实验、泛化、消融、理论 | Conclusion | PEPSY 在异构缺失联邦多模态设置中建立新 SOTA |
