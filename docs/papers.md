# cs.CL 

| Item |Content|
| --- |---|
|idx| 2403.08763v1 |
|title| Simple and Scalable Strategies to Continually Pre-train Large Language Models |
|authors| Adam IbrahimBenjamin ThérienKshitij GuptaMats L. RichterQuentin AnthonyTimothée LesortEugene BelilovskyIrina Rish
|links| http://arxiv.org/abs/2403.08763v1 |
|updated| 2024-03-13 17:58:57 UTC |
|summary| Large language models LLMs are routinely pre-trained on billions of tokensonly to start the process over again once new data becomes available. A muchmore efficient solution is to continually pre-train these models savingsignificant compute compared to re-training. However the distribution shiftinduced by new data typically results in degraded performance on previous dataor poor adaptation to the new data. In this work we show that a simple andscalable combination of learning rate LR re-warming LR re-decaying andreplay of previous data is sufficient to match the performance of fullyre-training from scratch on all available data as measured by final loss andlanguage model LM evaluation benchmarks. Specifically we show this for aweak but realistic distribution shift between two commonly used LLMpre-training datasets EnglishrightarrowEnglish and a stronger distributionshift EnglishrightarrowGerman at the 405M parameter model scale withlarge dataset sizes hundreds of billions of tokens. Selecting the weak butrealistic shift for larger-scale experiments we also find that our continuallearning strategies match the re-training baseline for a 10B parameter LLM. Ourresults demonstrate that LLMs can be successfully updated via simple andscalable continual learning strategies matching the re-training baseline usingonly a fraction of the compute. Finally inspired by previous work we proposealternatives to the cosine learning rate schedule that help circumventforgetting induced by LR re-warming and that are not bound to a fixed tokenbudget. |


| Item |Content|
| --- |---|
|idx| 2403.08755v1 |
|title| DAM: Dynamic Adapter Merging for Continual Video QA Learning |
|authors| Feng ChengZiyang WangYi-Lin SungYan-Bo LinMohit BansalGedas Bertasius
|links| http://arxiv.org/abs/2403.08755v1 |
|updated| 2024-03-13 17:53:47 UTC |
|summary| We present a parameter-efficient method for continual videoquestion-answering VidQA learning. Our method named DAM uses the proposedDynamic Adapter Merging to i mitigate catastrophic forgetting ii enableefficient adaptation to continually arriving datasets iii handle inputs fromunknown datasets during inference and iv enable knowledge sharing acrosssimilar dataset domains. Given a set of continually streaming VidQA datasetswe sequentially train dataset-specific adapters for each dataset while freezingthe parameters of a large pretrained video-language backbone. During inferencegiven a video-question sample from an unknown domain our method first uses theproposed non-parametric router function to compute a probability for eachadapter reflecting how relevant that adapter is to the current video-questioninput instance. Subsequently the proposed dynamic adapter merging schemeaggregates all the adapter weights into a new adapter instance tailored forthat particular test sample to compute the final VidQA prediction mitigatingthe impact of inaccurate router predictions and facilitating knowledge sharingacross domains. Our DAM model outperforms prior state-of-the-art continuallearning approaches by 9.1 while exhibiting 1.9 less forgetting on 6 VidQAdatasets spanning various domains. We further extend DAM to continual imageclassification and image QA and outperform prior methods by a large margin. Thecode is publicly available at: https://github.com/klauscc/DAM |


| Item |Content|
| --- |---|
|idx| 2403.08743v1 |
|title| Steering LLMs Towards Unbiased Responses: A Causality-Guided Debiasing Framework |
|authors| Jingling LiZeyu TangXiaoyu LiuPeter SpirtesKun ZhangLiu LeqiYang Liu
|links| http://arxiv.org/abs/2403.08743v1 |
|updated| 2024-03-13 17:46:28 UTC |
|summary| Large language models LLMs can easily generate biased and discriminativeresponses. As LLMs tap into consequential decision-making e.g. hiring andhealthcare it is of crucial importance to develop strategies to mitigatethese biases. This paper focuses on social bias tackling the associationbetween demographic information and LLM outputs. We propose a causality-guideddebiasing framework that utilizes causal understandings of 1 thedata-generating process of the training corpus fed to LLMs and 2 theinternal reasoning process of LLM inference to guide the design of prompts fordebiasing LLM outputs through selection mechanisms. Our framework unifiesexisting de-biasing prompting approaches such as inhibitive instructions andin-context contrastive examples and sheds light on new ways of debiasing byencouraging bias-free reasoning. Our strong empirical performance on real-worlddatasets demonstrates that our framework provides principled guidelines ondebiasing LLM outputs even with only the black-box access. |


| Item |Content|
| --- |---|
|idx| 2403.08739v1 |
|title| The Garden of Forking Paths: Observing Dynamic Parameters Distribution in Large Language Models |
|authors| Carlo NicoliniJacopo StaianoBruno LepriRaffaele Marino
|links| http://arxiv.org/abs/2403.08739v1 |
|updated| 2024-03-13 17:42:32 UTC |
|summary| A substantial gap persists in understanding the reasons behind theexceptional performance of the Transformer architecture in NLP. A particularlyunexplored area involves the mechanistic description of how the distribution ofparameters evolves over time during training. In this work we suggest thatlooking at the time evolution of the statistic distribution of modelparameters and specifically at bifurcation effects can help understanding themodel quality potentially reducing training costs and evaluation efforts andempirically showing the reasons behind the effectiveness of weightssparsification. |


| Item |Content|
| --- |---|
|idx| 2403.08738v1 |
|title| Improving Acoustic Word Embeddings through Correspondence Training of Self-supervised Speech Representations |
|authors| Amit MeghananiThomas Hain
|links| http://arxiv.org/abs/2403.08738v1 |
|updated| 2024-03-13 17:42:03 UTC |
|summary| Acoustic word embeddings AWEs are vector representations of spoken words.An effective method for obtaining AWEs is the Correspondence Auto-EncoderCAE. In the past the CAE method has been associated with traditional MFCCfeatures. Representations obtained from self-supervised learning SSL-basedspeech models such as HuBERT Wav2vec2 etc. are outperforming MFCC in manydownstream tasks. However they have not been well studied in the context oflearning AWEs. This work explores the effectiveness of CAE with SSL-basedspeech representations to obtain improved AWEs. Additionally the capabilitiesof SSL-based speech models are explored in cross-lingual scenarios forobtaining AWEs. Experiments are conducted on five languages: PolishPortuguese Spanish French and English. HuBERT-based CAE model achieves thebest results for word discrimination in all languages despite Hu-BERT beingpre-trained on English only. Also the HuBERT-based CAE model works well incross-lingual settings. It outperforms MFCC-based CAE models trained on thetarget languages when trained on one source language and tested on targetlanguages. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2403.08770v1 |
|title| FastMAC: Stochastic Spectral Sampling of Correspondence Graph |
|authors| Yifei ZhangHao ZhaoHongyang LiSiheng Chen
|links| http://arxiv.org/abs/2403.08770v1 |
|updated| 2024-03-13 17:59:56 UTC |
|summary| 3D correspondence i.e. a pair of 3D points is a fundamental concept incomputer vision. A set of 3D correspondences when equipped with compatibilityedges forms a correspondence graph. This graph is a critical component inseveral state-of-the-art 3D point cloud registration approaches e.g. the onebased on maximal cliques MAC. However its properties have not been wellunderstood. So we present the first study that introduces graph signalprocessing into the domain of correspondence graph. We exploit the generalizeddegree signal on correspondence graph and pursue sampling strategies thatpreserve high-frequency components of this signal. To address time-consumingsingular value decomposition in deterministic sampling we resort to astochastic approximate sampling strategy. As such the core of our method isthe stochastic spectral sampling of correspondence graph. As an application webuild a complete 3D registration algorithm termed as FastMAC that reachesreal-time speed while leading to little to none performance drop. Throughextensive experiments we validate that FastMAC works for both indoor andoutdoor benchmarks. For example FastMAC can accelerate MAC by 80 times whilemaintaining high registration success rate on KITTI. Codes are publiclyavailable at https://github.com/Forrest-110/FastMAC. |


| Item |Content|
| --- |---|
|idx| 2403.08763v1 |
|title| Simple and Scalable Strategies to Continually Pre-train Large Language Models |
|authors| Adam IbrahimBenjamin ThérienKshitij GuptaMats L. RichterQuentin AnthonyTimothée LesortEugene BelilovskyIrina Rish
|links| http://arxiv.org/abs/2403.08763v1 |
|updated| 2024-03-13 17:58:57 UTC |
|summary| Large language models LLMs are routinely pre-trained on billions of tokensonly to start the process over again once new data becomes available. A muchmore efficient solution is to continually pre-train these models savingsignificant compute compared to re-training. However the distribution shiftinduced by new data typically results in degraded performance on previous dataor poor adaptation to the new data. In this work we show that a simple andscalable combination of learning rate LR re-warming LR re-decaying andreplay of previous data is sufficient to match the performance of fullyre-training from scratch on all available data as measured by final loss andlanguage model LM evaluation benchmarks. Specifically we show this for aweak but realistic distribution shift between two commonly used LLMpre-training datasets EnglishrightarrowEnglish and a stronger distributionshift EnglishrightarrowGerman at the 405M parameter model scale withlarge dataset sizes hundreds of billions of tokens. Selecting the weak butrealistic shift for larger-scale experiments we also find that our continuallearning strategies match the re-training baseline for a 10B parameter LLM. Ourresults demonstrate that LLMs can be successfully updated via simple andscalable continual learning strategies matching the re-training baseline usingonly a fraction of the compute. Finally inspired by previous work we proposealternatives to the cosine learning rate schedule that help circumventforgetting induced by LR re-warming and that are not bound to a fixed tokenbudget. |


| Item |Content|
| --- |---|
|idx| 2403.08755v1 |
|title| DAM: Dynamic Adapter Merging for Continual Video QA Learning |
|authors| Feng ChengZiyang WangYi-Lin SungYan-Bo LinMohit BansalGedas Bertasius
|links| http://arxiv.org/abs/2403.08755v1 |
|updated| 2024-03-13 17:53:47 UTC |
|summary| We present a parameter-efficient method for continual videoquestion-answering VidQA learning. Our method named DAM uses the proposedDynamic Adapter Merging to i mitigate catastrophic forgetting ii enableefficient adaptation to continually arriving datasets iii handle inputs fromunknown datasets during inference and iv enable knowledge sharing acrosssimilar dataset domains. Given a set of continually streaming VidQA datasetswe sequentially train dataset-specific adapters for each dataset while freezingthe parameters of a large pretrained video-language backbone. During inferencegiven a video-question sample from an unknown domain our method first uses theproposed non-parametric router function to compute a probability for eachadapter reflecting how relevant that adapter is to the current video-questioninput instance. Subsequently the proposed dynamic adapter merging schemeaggregates all the adapter weights into a new adapter instance tailored forthat particular test sample to compute the final VidQA prediction mitigatingthe impact of inaccurate router predictions and facilitating knowledge sharingacross domains. Our DAM model outperforms prior state-of-the-art continuallearning approaches by 9.1 while exhibiting 1.9 less forgetting on 6 VidQAdatasets spanning various domains. We further extend DAM to continual imageclassification and image QA and outperform prior methods by a large margin. Thecode is publicly available at: https://github.com/klauscc/DAM |


| Item |Content|
| --- |---|
|idx| 2403.08743v1 |
|title| Steering LLMs Towards Unbiased Responses: A Causality-Guided Debiasing Framework |
|authors| Jingling LiZeyu TangXiaoyu LiuPeter SpirtesKun ZhangLiu LeqiYang Liu
|links| http://arxiv.org/abs/2403.08743v1 |
|updated| 2024-03-13 17:46:28 UTC |
|summary| Large language models LLMs can easily generate biased and discriminativeresponses. As LLMs tap into consequential decision-making e.g. hiring andhealthcare it is of crucial importance to develop strategies to mitigatethese biases. This paper focuses on social bias tackling the associationbetween demographic information and LLM outputs. We propose a causality-guideddebiasing framework that utilizes causal understandings of 1 thedata-generating process of the training corpus fed to LLMs and 2 theinternal reasoning process of LLM inference to guide the design of prompts fordebiasing LLM outputs through selection mechanisms. Our framework unifiesexisting de-biasing prompting approaches such as inhibitive instructions andin-context contrastive examples and sheds light on new ways of debiasing byencouraging bias-free reasoning. Our strong empirical performance on real-worlddatasets demonstrates that our framework provides principled guidelines ondebiasing LLM outputs even with only the black-box access. |


| Item |Content|
| --- |---|
|idx| 2403.08739v1 |
|title| The Garden of Forking Paths: Observing Dynamic Parameters Distribution in Large Language Models |
|authors| Carlo NicoliniJacopo StaianoBruno LepriRaffaele Marino
|links| http://arxiv.org/abs/2403.08739v1 |
|updated| 2024-03-13 17:42:32 UTC |
|summary| A substantial gap persists in understanding the reasons behind theexceptional performance of the Transformer architecture in NLP. A particularlyunexplored area involves the mechanistic description of how the distribution ofparameters evolves over time during training. In this work we suggest thatlooking at the time evolution of the statistic distribution of modelparameters and specifically at bifurcation effects can help understanding themodel quality potentially reducing training costs and evaluation efforts andempirically showing the reasons behind the effectiveness of weightssparsification. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2403.08763v1 |
|title| Simple and Scalable Strategies to Continually Pre-train Large Language Models |
|authors| Adam IbrahimBenjamin ThérienKshitij GuptaMats L. RichterQuentin AnthonyTimothée LesortEugene BelilovskyIrina Rish
|links| http://arxiv.org/abs/2403.08763v1 |
|updated| 2024-03-13 17:58:57 UTC |
|summary| Large language models LLMs are routinely pre-trained on billions of tokensonly to start the process over again once new data becomes available. A muchmore efficient solution is to continually pre-train these models savingsignificant compute compared to re-training. However the distribution shiftinduced by new data typically results in degraded performance on previous dataor poor adaptation to the new data. In this work we show that a simple andscalable combination of learning rate LR re-warming LR re-decaying andreplay of previous data is sufficient to match the performance of fullyre-training from scratch on all available data as measured by final loss andlanguage model LM evaluation benchmarks. Specifically we show this for aweak but realistic distribution shift between two commonly used LLMpre-training datasets EnglishrightarrowEnglish and a stronger distributionshift EnglishrightarrowGerman at the 405M parameter model scale withlarge dataset sizes hundreds of billions of tokens. Selecting the weak butrealistic shift for larger-scale experiments we also find that our continuallearning strategies match the re-training baseline for a 10B parameter LLM. Ourresults demonstrate that LLMs can be successfully updated via simple andscalable continual learning strategies matching the re-training baseline usingonly a fraction of the compute. Finally inspired by previous work we proposealternatives to the cosine learning rate schedule that help circumventforgetting induced by LR re-warming and that are not bound to a fixed tokenbudget. |


| Item |Content|
| --- |---|
|idx| 2403.08757v1 |
|title| Efficient Combinatorial Optimization via Heat Diffusion |
|authors| Hengyuan MaWenlian LuJianfeng Feng
|links| http://arxiv.org/abs/2403.08757v1 |
|updated| 2024-03-13 17:55:34 UTC |
|summary| Combinatorial optimization problems are widespread but inherently challengingdue to their discrete nature.The primary limitation of existing methods is thatthey can only access a small fraction of the solution space at each iterationresulting in limited efficiency for searching the global optimal. To overcomethis challenge diverging from conventional efforts of expanding the solverssearch scope we focus on enabling information to actively propagate to thesolver through heat diffusion. By transforming the target function whilepreserving its optima heat diffusion facilitates information flow from distantregions to the solver providing more efficient navigation. Utilizing heatdiffusion we propose a framework for solving general combinatorialoptimization problems. The proposed methodology demonstrates superiorperformance across a range of the most challenging and widely encounteredcombinatorial optimizations. Echoing recent advancements in harnessingthermodynamics for generative artificial intelligence our study furtherreveals its significant potential in advancing combinatorial optimization. |


| Item |Content|
| --- |---|
|idx| 2403.08755v1 |
|title| DAM: Dynamic Adapter Merging for Continual Video QA Learning |
|authors| Feng ChengZiyang WangYi-Lin SungYan-Bo LinMohit BansalGedas Bertasius
|links| http://arxiv.org/abs/2403.08755v1 |
|updated| 2024-03-13 17:53:47 UTC |
|summary| We present a parameter-efficient method for continual videoquestion-answering VidQA learning. Our method named DAM uses the proposedDynamic Adapter Merging to i mitigate catastrophic forgetting ii enableefficient adaptation to continually arriving datasets iii handle inputs fromunknown datasets during inference and iv enable knowledge sharing acrosssimilar dataset domains. Given a set of continually streaming VidQA datasetswe sequentially train dataset-specific adapters for each dataset while freezingthe parameters of a large pretrained video-language backbone. During inferencegiven a video-question sample from an unknown domain our method first uses theproposed non-parametric router function to compute a probability for eachadapter reflecting how relevant that adapter is to the current video-questioninput instance. Subsequently the proposed dynamic adapter merging schemeaggregates all the adapter weights into a new adapter instance tailored forthat particular test sample to compute the final VidQA prediction mitigatingthe impact of inaccurate router predictions and facilitating knowledge sharingacross domains. Our DAM model outperforms prior state-of-the-art continuallearning approaches by 9.1 while exhibiting 1.9 less forgetting on 6 VidQAdatasets spanning various domains. We further extend DAM to continual imageclassification and image QA and outperform prior methods by a large margin. Thecode is publicly available at: https://github.com/klauscc/DAM |


| Item |Content|
| --- |---|
|idx| 2403.08750v1 |
|title| Neural reproducing kernel Banach spaces and representer theorems for deep networks |
|authors| Francesca BartolucciErnesto De VitoLorenzo RosascoStefano Vigogna
|links| http://arxiv.org/abs/2403.08750v1 |
|updated| 2024-03-13 17:51:02 UTC |
|summary| Studying the function spaces defined by neural networks helps to understandthe corresponding learning models and their inductive bias. While in somelimits neural networks correspond to function spaces that are reproducingkernel Hilbert spaces these regimes do not capture the properties of thenetworks used in practice. In contrast in this paper we show that deep neuralnetworks define suitable reproducing kernel Banach spaces.  These spaces are equipped with norms that enforce a form of sparsityenabling them to adapt to potential latent structures within the input data andtheir representations. In particular leveraging the theory of reproducingkernel Banach spaces combined with variational results we derive representertheorems that justify the finite architectures commonly employed inapplications. Our study extends analogous results for shallow networks and canbe seen as a step towards considering more practically plausible neuralarchitectures. |


| Item |Content|
| --- |---|
|idx| 2403.08743v1 |
|title| Steering LLMs Towards Unbiased Responses: A Causality-Guided Debiasing Framework |
|authors| Jingling LiZeyu TangXiaoyu LiuPeter SpirtesKun ZhangLiu LeqiYang Liu
|links| http://arxiv.org/abs/2403.08743v1 |
|updated| 2024-03-13 17:46:28 UTC |
|summary| Large language models LLMs can easily generate biased and discriminativeresponses. As LLMs tap into consequential decision-making e.g. hiring andhealthcare it is of crucial importance to develop strategies to mitigatethese biases. This paper focuses on social bias tackling the associationbetween demographic information and LLM outputs. We propose a causality-guideddebiasing framework that utilizes causal understandings of 1 thedata-generating process of the training corpus fed to LLMs and 2 theinternal reasoning process of LLM inference to guide the design of prompts fordebiasing LLM outputs through selection mechanisms. Our framework unifiesexisting de-biasing prompting approaches such as inhibitive instructions andin-context contrastive examples and sheds light on new ways of debiasing byencouraging bias-free reasoning. Our strong empirical performance on real-worlddatasets demonstrates that our framework provides principled guidelines ondebiasing LLM outputs even with only the black-box access. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2403.08770v1 |
|title| FastMAC: Stochastic Spectral Sampling of Correspondence Graph |
|authors| Yifei ZhangHao ZhaoHongyang LiSiheng Chen
|links| http://arxiv.org/abs/2403.08770v1 |
|updated| 2024-03-13 17:59:56 UTC |
|summary| 3D correspondence i.e. a pair of 3D points is a fundamental concept incomputer vision. A set of 3D correspondences when equipped with compatibilityedges forms a correspondence graph. This graph is a critical component inseveral state-of-the-art 3D point cloud registration approaches e.g. the onebased on maximal cliques MAC. However its properties have not been wellunderstood. So we present the first study that introduces graph signalprocessing into the domain of correspondence graph. We exploit the generalizeddegree signal on correspondence graph and pursue sampling strategies thatpreserve high-frequency components of this signal. To address time-consumingsingular value decomposition in deterministic sampling we resort to astochastic approximate sampling strategy. As such the core of our method isthe stochastic spectral sampling of correspondence graph. As an application webuild a complete 3D registration algorithm termed as FastMAC that reachesreal-time speed while leading to little to none performance drop. Throughextensive experiments we validate that FastMAC works for both indoor andoutdoor benchmarks. For example FastMAC can accelerate MAC by 80 times whilemaintaining high registration success rate on KITTI. Codes are publiclyavailable at https://github.com/Forrest-110/FastMAC. |


| Item |Content|
| --- |---|
|idx| 2403.08768v1 |
|title| 3DFIRES: Few Image 3D REconstruction for Scenes with Hidden Surface |
|authors| Linyi JinNilesh KulkarniDavid Fouhey
|links| http://arxiv.org/abs/2403.08768v1 |
|updated| 2024-03-13 17:59:50 UTC |
|summary| This paper introduces 3DFIRES a novel system for scene-level 3Dreconstruction from posed images. Designed to work with as few as one view3DFIRES reconstructs the complete geometry of unseen scenes including hiddensurfaces. With multiple view inputs our method produces full reconstructionwithin all camera frustums. A key feature of our approach is the fusion ofmulti-view information at the feature level enabling the production ofcoherent and comprehensive 3D reconstruction. We train our system onnon-watertight scans from large-scale real scene dataset. We show it matchesthe efficacy of single-view reconstruction methods with only one input andsurpasses existing techniques in both quantitative and qualitative measures forsparse-view 3D reconstruction. |


| Item |Content|
| --- |---|
|idx| 2403.08766v1 |
|title| MonoOcc: Digging into Monocular Semantic Occupancy Prediction |
|authors| Yupeng ZhengXiang LiPengfei LiYuhang ZhengBu JinChengliang ZhongXiaoxiao LongHao ZhaoQichao Zhang
|links| http://arxiv.org/abs/2403.08766v1 |
|updated| 2024-03-13 17:59:04 UTC |
|summary| Monocular Semantic Occupancy Prediction aims to infer the complete 3Dgeometry and semantic information of scenes from only 2D images. It hasgarnered significant attention particularly due to its potential to enhancethe 3D perception of autonomous vehicles. However existing methods rely on acomplex cascaded framework with relatively limited information to restore 3Dscenes including a dependency on supervision solely on the whole networksoutput single-frame input and the utilization of a small backbone. Thesechallenges in turn hinder the optimization of the framework and yieldinferior prediction results particularly concerning smaller and long-tailedobjects. To address these issues we propose MonoOcc. In particular we iimprove the monocular occupancy prediction framework by proposing an auxiliarysemantic loss as supervision to the shallow layers of the framework and animage-conditioned cross-attention module to refine voxel features with visualclues and ii employ a distillation module that transfers temporalinformation and richer knowledge from a larger image backbone to the monocularsemantic occupancy prediction framework with low cost of hardware. With theseadvantages our method yields state-of-the-art performance on the camera-basedSemanticKITTI Scene Completion benchmark. Codes and models can be accessed athttps://github.com/ucaszyp/MonoOcc |


| Item |Content|
| --- |---|
|idx| 2403.08764v1 |
|title| VLOGGER: Multimodal Diffusion for Embodied Avatar Synthesis |
|authors| Enric CoronaAndrei ZanfirEduard Gabriel BazavanNikos KolotourosThiemo AlldieckCristian Sminchisescu
|links| http://arxiv.org/abs/2403.08764v1 |
|updated| 2024-03-13 17:59:02 UTC |
|summary| We propose VLOGGER a method for audio-driven human video generation from asingle input image of a person which builds on the success of recentgenerative diffusion models. Our method consists of 1 a stochastichuman-to-3d-motion diffusion model and 2 a novel diffusion-based architecturethat augments text-to-image models with both spatial and temporal controls.This supports the generation of high quality video of variable length easilycontrollable through high-level representations of human faces and bodies. Incontrast to previous work our method does not require training for eachperson does not rely on face detection and cropping generates the completeimage not just the face or the lips and considers a broad spectrum ofscenarios e.g. visible torso or diverse subject identities that are criticalto correctly synthesize humans who communicate. We also curate MENTOR a newand diverse dataset with 3d pose and expression annotations one order ofmagnitude larger than previous ones 800000 identities and with dynamicgestures on which we train and ablate our main technical contributions.  VLOGGER outperforms state-of-the-art methods in three public benchmarksconsidering image quality identity preservation and temporal consistency whilealso generating upper-body gestures. We analyze the performance of VLOGGER withrespect to multiple diversity metrics showing that our architectural choicesand the use of MENTOR benefit training a fair and unbiased model at scale.Finally we show applications in video editing and personalization. |


| Item |Content|
| --- |---|
|idx| 2403.08761v1 |
|title| Segmentation of Knee Bones for Osteoarthritis Assessment: A Comparative Analysis of Supervised, Few-Shot, and Zero-Shot Learning Approaches |
|authors| Yun Xin TeohAlice OthmaniSiew Li GohJuliana UsmanKhin Wee Lai
|links| http://arxiv.org/abs/2403.08761v1 |
|updated| 2024-03-13 17:58:34 UTC |
|summary| Knee osteoarthritis is a degenerative joint disease that induces chronic painand disability. Bone morphological analysis is a promising tool to understandthe mechanical aspect of this disorder. This study proposes a 2D bonemorphological analysis using manually segmented bones to explore morphologicalfeatures related to distinct pain conditions. Furthermore six semanticsegmentation algorithms are assessed for extracting femur and tibia bones fromX-ray images. Our analysis reveals that the morphology of the femur undergoessignificant changes in instances where pain worsens. Conversely improvementsin pain may not manifest pronounced alterations in bone shape. Thefew-shot-learning-based algorithm UniverSeg demonstrated superiorsegmentation results with Dice scores of 99.69 for femur and 99.60 for tibia.Regarding pain condition classification the zero-shot-learning-basedalgorithm CP-SAM achieved the highest accuracy at 66 among all models.UniverSeg is recommended for automatic knee bone segmentation while SAM modelsshow potential with prompt encoder modifications for optimized outcomes. Thesefindings highlight the effectiveness of few-shot learning for semanticsegmentation and the potential of zero-shot learning in enhancingclassification models for knee osteoarthritis diagnosis. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2403.08757v1 |
|title| Efficient Combinatorial Optimization via Heat Diffusion |
|authors| Hengyuan MaWenlian LuJianfeng Feng
|links| http://arxiv.org/abs/2403.08757v1 |
|updated| 2024-03-13 17:55:34 UTC |
|summary| Combinatorial optimization problems are widespread but inherently challengingdue to their discrete nature.The primary limitation of existing methods is thatthey can only access a small fraction of the solution space at each iterationresulting in limited efficiency for searching the global optimal. To overcomethis challenge diverging from conventional efforts of expanding the solverssearch scope we focus on enabling information to actively propagate to thesolver through heat diffusion. By transforming the target function whilepreserving its optima heat diffusion facilitates information flow from distantregions to the solver providing more efficient navigation. Utilizing heatdiffusion we propose a framework for solving general combinatorialoptimization problems. The proposed methodology demonstrates superiorperformance across a range of the most challenging and widely encounteredcombinatorial optimizations. Echoing recent advancements in harnessingthermodynamics for generative artificial intelligence our study furtherreveals its significant potential in advancing combinatorial optimization. |


| Item |Content|
| --- |---|
|idx| 2403.08750v1 |
|title| Neural reproducing kernel Banach spaces and representer theorems for deep networks |
|authors| Francesca BartolucciErnesto De VitoLorenzo RosascoStefano Vigogna
|links| http://arxiv.org/abs/2403.08750v1 |
|updated| 2024-03-13 17:51:02 UTC |
|summary| Studying the function spaces defined by neural networks helps to understandthe corresponding learning models and their inductive bias. While in somelimits neural networks correspond to function spaces that are reproducingkernel Hilbert spaces these regimes do not capture the properties of thenetworks used in practice. In contrast in this paper we show that deep neuralnetworks define suitable reproducing kernel Banach spaces.  These spaces are equipped with norms that enforce a form of sparsityenabling them to adapt to potential latent structures within the input data andtheir representations. In particular leveraging the theory of reproducingkernel Banach spaces combined with variational results we derive representertheorems that justify the finite architectures commonly employed inapplications. Our study extends analogous results for shallow networks and canbe seen as a step towards considering more practically plausible neuralarchitectures. |


| Item |Content|
| --- |---|
|idx| 2403.08699v1 |
|title| Implicit Regularization of Gradient Flow on One-Layer Softmax Attention |
|authors| Heejune SheenSiyu ChenTianhao WangHarrison H. Zhou
|links| http://arxiv.org/abs/2403.08699v1 |
|updated| 2024-03-13 17:02:27 UTC |
|summary| We study gradient flow on the exponential loss for a classification problemwith a one-layer softmax attention model where the key and query weightmatrices are trained separately. Under a separability assumption on the datawe show that when gradient flow achieves the minimal loss value it furtherimplicitly minimizes the nuclear norm of the product of the key and queryweight matrices. Such implicit regularization can be described by a SupportVector Machine SVM problem with respect to the attention weights. Thisfinding contrasts with prior results showing that the gradient descent inducesan implicit regularization on the Frobenius norm on the product weight matrixwhen the key and query matrices are combined into a single weight matrix fortraining. For diagonal key and query matrices our analysis builds upon thereparameterization technique and exploits approximate KKT conditions of the SVMassociated with the classification data. Moreover the results are extended togeneral weights configurations given proper alignment of the weight matricessingular spaces with the data features at initialization. |


| Item |Content|
| --- |---|
|idx| 2403.08673v1 |
|title| When can we Approximate Wide Contrastive Models with Neural Tangent Kernels and Principal Component Analysis? |
|authors| Gautham Govind AnilPascal EsserDebarghya Ghoshdastidar
|links| http://arxiv.org/abs/2403.08673v1 |
|updated| 2024-03-13 16:25:55 UTC |
|summary| Contrastive learning is a paradigm for learning representations fromunlabelled data that has been highly successful for image and text data.Several recent works have examined contrastive losses to claim that contrastivemodels effectively learn spectral embeddings while few works show relationsbetween wide contrastive models and kernel principal component analysisPCA. However it is not known if trained contrastive models indeed correspondto kernel methods or PCA. In this work we analyze the training dynamics oftwo-layer contrastive models with non-linear activation and answer when thesemodels are close to PCA or kernel methods. It is well known in the supervisedsetting that neural networks are equivalent to neural tangent kernel NTKmachines and that the NTK of infinitely wide networks remains constant duringtraining. We provide the first convergence results of NTK for contrastivelosses and present a nuanced picture: NTK of wide networks remains almostconstant for cosine similarity based contrastive losses but not for lossesbased on dot product similarity. We further study the training dynamics ofcontrastive models with orthogonality constraints on output layer which isimplicitly assumed in works relating contrastive learning to spectralembedding. Our deviation bounds suggest that representations learned bycontrastive models are close to the principal components of a certain matrixcomputed from random features. We empirically show that our theoretical resultspossibly hold beyond two-layer networks. |


| Item |Content|
| --- |---|
|idx| 2403.08652v1 |
|title| Extracting Explanations, Justification, and Uncertainty from Black-Box Deep Neural Networks |
|authors| Paul ArdisArjuna Flenner
|links| http://arxiv.org/abs/2403.08652v1 |
|updated| 2024-03-13 16:06:26 UTC |
|summary| Deep Neural Networks DNNs do not inherently compute or exhibitempirically-justified task confidence. In mission critical applications it isimportant to both understand associated DNN reasoning and its supportingevidence. In this paper we propose a novel Bayesian approach to extractexplanations justifications and uncertainty estimates from DNNs. Our approachis efficient both in terms of memory and computation and can be applied to anyblack box DNN without any retraining including applications to anomalydetection and out-of-distribution detection tasks. We validate our approach onthe CIFAR-10 dataset and show that it can significantly improve theinterpretability and reliability of DNNs. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2403.08700v1 |
|title| Diffusion-based Iterative Counterfactual Explanations for Fetal Ultrasound Image Quality Assessment |
|authors| Paraskevas PegiosManxi LinNina WengMorten Bo Søndergaard SvendsenZahra BashirSiavash BigdeliAnders Nymark ChristensenMartin TolsgaardAasa Feragen
|links| http://arxiv.org/abs/2403.08700v1 |
|updated| 2024-03-13 17:04:56 UTC |
|summary| Obstetric ultrasound image quality is crucial for accurate diagnosis andmonitoring of fetal health. However producing high-quality standard planes isdifficult influenced by the sonographers expertise and factors like thematernal BMI or the fetus dynamics. In this work we propose usingdiffusion-based counterfactual explainable AI to generate realistichigh-quality standard planes from low-quality non-standard ones. Throughquantitative and qualitative evaluation we demonstrate the effectiveness ofour method in producing plausible counterfactuals of increased quality. Thisshows future promise both for enhancing training of clinicians by providingvisual feedback as well as for improving image quality and consequentlydownstream diagnosis and monitoring. |


| Item |Content|
| --- |---|
|idx| 2403.08564v1 |
|title| Non-discrimination Criteria for Generative Language Models |
|authors| Sara SterlieNina WengAasa Feragen
|links| http://arxiv.org/abs/2403.08564v1 |
|updated| 2024-03-13 14:19:08 UTC |
|summary| Within recent years generative AI such as large language models hasundergone rapid development. As these models become increasingly available tothe public concerns arise about perpetuating and amplifying harmful biases inapplications. Gender stereotypes can be harmful and limiting for theindividuals they target whether they consist of misrepresentation ordiscrimination. Recognizing gender bias as a pervasive societal construct thispaper studies how to uncover and quantify the presence of gender biases ingenerative language models. In particular we derive generative AI analogues ofthree well-known non-discrimination criteria from classification namelyindependence separation and sufficiency. To demonstrate these criteria inaction we design prompts for each of the criteria with a focus on occupationalgender stereotype specifically utilizing the medical test to introduce theground truth in the generative AI context. Our results address the presence ofoccupational gender bias within such conversational language models. |


| Item |Content|
| --- |---|
|idx| 2403.08538v1 |
|title| Calibrating coordinate system alignment in a scanning transmission electron microscope using a digital twin |
|authors| Dieter WeberDavid LandersChen HuangEmanuela LibertiEmiliya PoghosyanMatthew BryanAlexander ClausenDaniel G. StroppaAngus I. KirklandElisabeth MüllerAndrew StewartRafal E. Dunin-Borkowski
|links| http://arxiv.org/abs/2403.08538v1 |
|updated| 2024-03-13 13:52:23 UTC |
|summary| In four-dimensional scanning transmission electron microscopy 4D STEM afocused beam is scanned over a specimen and a diffraction pattern is recordedat each position using a pixelated detector. During the experiment it must beensured that the scan coordinate system of the beam is correctly calibratedrelative to the detector coordinate system. Various simplified and approximatemodels are used implicitly and explicitly for understanding and analyzing therecorded data requiring translation between the physical reality of theinstrument and the abstractions used in data interpretation. Here we introducea calibration method where interactive live data processing in combination witha digital twin is used to match a set of models and their parameters with theaction of a real-world instrument. |


| Item |Content|
| --- |---|
|idx| 2403.08396v1 |
|title| A Picture Is Worth a Thousand Words: Exploring Diagram and Video-Based OOP Exercises to Counter LLM Over-Reliance |
|authors| Bruno Pereira CiprianoPedro AlvesPaul Denny
|links| http://arxiv.org/abs/2403.08396v1 |
|updated| 2024-03-13 10:21:29 UTC |
|summary| Much research has highlighted the impressive capabilities of large languagemodels LLMs like GPT and Bard for solving introductory programmingexercises. Recent work has shown that LLMs can effectively solve a range ofmore complex object-oriented programming OOP exercises with text-basedspecifications. This raises concerns about academic integrity as studentsmight use these models to complete assignments unethically neglecting thedevelopment of important skills such as program design problem-solving andcomputational thinking. To address this we propose an innovative approach toformulating OOP tasks using diagrams and videos as a way to fosterproblem-solving and deter students from a copy-and-prompt approach in OOPcourses. We introduce a novel notation system for specifying OOP assignmentsencompassing structural and behavioral requirements and assess its use in aclassroom setting over a semester. Student perceptions of this approach areexplored through a survey n56. Generally students responded positively todiagrams and videos with video-based projects being better received thandiagram-based exercises. This notation appears to have several benefits withstudents investing more effort in understanding the diagrams and feeling moremotivated to engage with the video-based projects. Furthermore studentsreported being less inclined to rely on LLM-based code generation tools forthese diagram and video-based exercises. Experiments with GPT-4 and Bardsvision abilities revealed that they currently fall short in interpreting thesediagrams to generate accurate code solutions. |


| Item |Content|
| --- |---|
|idx| 2403.08363v1 |
|title| ShareYourReality: Investigating Haptic Feedback and Agency in Virtual Avatar Co-embodiment |
|authors| Karthikeya Puttur VenkatrajWo MeijerMonica Perusquía-HernándezGijs HuismanAbdallah El Ali
|links| http://dx.doi.org/10.1145/3613904.3642425 |
|updated| 2024-03-13 09:23:53 UTC |
|summary| Virtual co-embodiment enables two users to share a single avatar in VirtualReality VR. During such experiences the illusion of shared motion controlcan break during joint-action activities highlighting the need forposition-aware feedback mechanisms. Drawing on the perceptual crossingparadigm we explore how haptics can enable non-verbal coordination betweenco-embodied participants. In a within-subjects study 20 participant pairs weexamined the effects of vibrotactile haptic feedback None Present and avatarcontrol distribution 25-75 50-50 75-25 across two VR reaching tasksTargeted Free-choice on participants Sense of Agency SoA co-presencebody ownership and motion synchrony. We found a lower SoA in the free-choicewith haptics than without b higher SoA during the shared targeted task cco-presence and body ownership were significantly higher in the free-choicetask d players hand motions synchronized more in the targeted task. Weprovide cautionary considerations when including haptic feedback mechanisms foravatar co-embodiment experiences. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2403.08610v1 |
|title| An Algorithmic Theory of Simplicity in Mechanism Design |
|authors| Diodato FerraioliCarmine Ventre
|links| http://arxiv.org/abs/2403.08610v1 |
|updated| 2024-03-13 15:22:18 UTC |
|summary| A growing body of work in economics and computation focuses on the trade-offbetween implementability and simplicity in mechanism design. The goal is todevelop a theory that not only allows to design an incentive structure easy tograsp for imperfectly rational agents but also understand the ensuinglimitations on the class of mechanisms that enforce it. In this context theconcept of OSP mechanisms has assumed a prominent role since they provablyaccount for the absence of contingent reasoning skills a specific cognitivelimitation. For single-dimensional agents it is known that OSP mechanisms needto use certain greedy algorithms.  In this work we introduce a notion that interpolates between OSP and SOSP amore stringent notion where agents only plan a subset of their own futuremoves. We provide an algorithmic characterization of this novel class ofmechanisms for single-dimensional domains and binary allocation problems thatprecisely measures the interplay between simplicity and implementability. Webuild on this to show how mechanisms based on reverse greedy algorithmsa.k.a. deferred acceptance auctions are algorithmically more robust toimperfectly rationality than those adopting greedy algorithms. |


| Item |Content|
| --- |---|
|idx| 2403.08291v1 |
|title| CleanAgent: Automating Data Standardization with LLM-based Agents |
|authors| Danrui QiJiannan Wang
|links| http://arxiv.org/abs/2403.08291v1 |
|updated| 2024-03-13 06:54:15 UTC |
|summary| Data standardization is a crucial part in data science life cycle. Whiletools like Pandas offer robust functionalities their complexity and the manualeffort required for customizing code to diverse column types pose significantchallenges. Although large language models LLMs like ChatGPT have shownpromise in automating this process through natural language understanding andcode generation it still demands expert-level programming knowledge andcontinuous interaction for prompt refinement. To solve these challenges ourkey idea is to propose a Python library with declarative unified APIs forstandardizing column types simplifying the code generation of LLM with conciseAPI calls. We first propose Dataprep.Clean which is written as a component ofthe Dataprep Library offers a significant reduction in complexity by enablingthe standardization of specific column types with a single line of code. Thenwe introduce the CleanAgent framework integrating Dataprep.Clean and LLM-basedagents to automate the data standardization process. With CleanAgent datascientists need only provide their requirements once allowing for ahands-free automatic standardization process. |


| Item |Content|
| --- |---|
|idx| 2403.08251v1 |
|title| Emergence of Social Norms in Large Language Model-based Agent Societies |
|authors| Siyue RenZhiyao CuiRuiqi SongZhen WangShuyue Hu
|links| http://arxiv.org/abs/2403.08251v1 |
|updated| 2024-03-13 05:08:10 UTC |
|summary| The emergence of social norms has attracted much interest in a wide array ofdisciplines ranging from social science and cognitive science to artificialintelligence. In this paper we propose the first generative agent architecturethat empowers the emergence of social norms within a population of largelanguage model-based agents. Our architecture named CRSEC consists of fourmodules: Creation  Representation Spreading Evaluation and Compliance. Ourarchitecture addresses several important aspects of the emergent processes allin one: i where social norms come from ii how they are formallyrepresented iii how they spread through agents communications andobservations iv how they are examined with a sanity check and synthesized inthe long term and v how they are incorporated into agents planning andactions. Our experiments deployed in the Smallville sandbox game environmentdemonstrate the capability of our architecture to establish social norms andreduce social conflicts within large language model-based multi-agent systems.The positive outcomes of our human evaluation conducted with 30 evaluatorsfurther affirm the effectiveness of our approach. |


| Item |Content|
| --- |---|
|idx| 2403.07769v1 |
|title| Transforming Competition into Collaboration: The Revolutionary Role of Multi-Agent Systems and Language Models in Modern Organizations |
|authors| Carlos Jose Xavier Cruz
|links| http://arxiv.org/abs/2403.07769v1 |
|updated| 2024-03-12 15:56:10 UTC |
|summary| This article explores the dynamic influence of computational entities basedon multi-agent systems theory SMA combined with large language models LLMwhich are characterized by their ability to simulate complex humaninteractions as a possibility to revolutionize human user interaction from theuse of specialized artificial agents to support everything from operationalorganizational processes to strategic decision making based on appliedknowledge and human orchestration. Previous investigations reveal that thereare limitations particularly in the autonomous approach of artificial agentsespecially when dealing with new challenges and pragmatic tasks such asinducing logical reasoning and problem solving. It is also considered thattraditional techniques such as the stimulation of chains of thoughts requireexplicit human guidance. In our approach we employ agents developed from largelanguage models LLM each with distinct prototyping that considers behavioralelements driven by strategies that stimulate the generation of knowledge basedon the use case proposed in the scenario role-play business using adiscussion approach between agents guided conversation. We demonstrate thepotential of developing agents useful for organizational strategies based onmulti-agent system theories SMA and innovative uses based on large languagemodels LLM based offering a differentiated and adaptable experiment todifferent applications complexities domains and capabilities from LLM. |


| Item |Content|
| --- |---|
|idx| 2403.07748v1 |
|title| Ariadne and Theseus: Exploration and Rendezvous with Two Mobile Agents in an Unknown Graph |
|authors| Romain Cosson
|links| http://arxiv.org/abs/2403.07748v1 |
|updated| 2024-03-12 15:33:09 UTC |
|summary| We investigate two fundamental problems in mobile computing: exploration andrendezvous with two distinct mobile agents in an unknown graph. The agents canread and write information on whiteboards that are located at all nodes. Theyboth move along one adjacent edge at every time-step. In the explorationproblem both agents start from the same node of the graph and must traverseall of its edges. We show that a simple variant of depth-first search achievescollective exploration in m synchronous time-steps where m is the numberof edges of the graph. This improves the competitive ratio of collective graphexploration. In the rendezvous problem the agents start from different nodesof the graph and must meet as fast as possible. We introduce an algorithmguaranteeing rendezvous in at most frac32m time-steps. This improvesover the so-called wait for Mommy algorithm which requires 2m time-steps.All our guarantees are derived from a more general asynchronous setting inwhich the speeds of the agents are controlled by an adversary at all times. Ourguarantees also generalize to weighted graphs if the number of edges m isreplaced by the sum of all edge lengths. |


