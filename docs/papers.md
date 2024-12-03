# cs.CL 

| Item |Content|
| --- |---|
|idx| 2411.19951v2 |
|title| T2Vid: Translating Long Text into Multi-Image is the Catalyst for Video-LLMs |
|authors| Shukang YinChaoyou FuSirui ZhaoYunhang ShenChunjiang GeYan YangZuwei LongYuhan DaiTong XuXing SunRan HeCaifeng ShanEnhong Chen
|links| http://arxiv.org/abs/2411.19951v2 |
|updated| 2024-12-02 06:54:47 UTC |
|summary| The success of Multimodal Large Language Models MLLMs in the image domainhas garnered wide attention from the research community. Drawing on previoussuccessful experiences researchers have recently explored extending thesuccess to the video understanding realms. Apart from training from scratch anefficient way is to utilize the pre-trained image-LLMs leading to twomainstream approaches i.e. zero-shot inference and further fine-tuning withvideo data. In this work our study of these approaches harvests an effectivedata augmentation method. We first make a deeper inspection of the zero-shotinference way and identify two limitations i.e. limited generalization andlack of temporal understanding capabilities. Thus we further investigate thefine-tuning approach and find a low learning efficiency when simply using allthe video data samples which can be attributed to a lack of instructiondiversity. Aiming at this issue we develop a method called T2Vid to synthesizevideo-like samples to enrich the instruction diversity in the training corpus.Integrating these data enables a simple and efficient training scheme whichachieves performance comparable to or even superior to using full videodatasets by training with just 15 the sample size. Meanwhile we find that theproposed scheme can boost the performance of long video understanding withouttraining with long video samples. We hope our study will spark more thinkingabout using MLLMs for video understanding and curation of high-quality data.The code is released at https://github.com/xjtupanda/T2Vid. |


| Item |Content|
| --- |---|
|idx| 2411.19943v2 |
|title| Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability |
|authors| Zicheng LinTian LiangJiahao XuXing WangRuilin LuoChufan ShiSiheng LiYujiu YangZhaopeng Tu
|links| http://arxiv.org/abs/2411.19943v2 |
|updated| 2024-12-02 06:26:38 UTC |
|summary| Large Language Models LLMs have exhibited remarkable performance onreasoning tasks. They utilize autoregressive token generation to constructreasoning trajectories enabling the development of a coherent chain ofthought. In this work we explore the impact of individual tokens on the finaloutcomes of reasoning tasks. We identify the existence of critical tokensthat lead to incorrect reasoning trajectories in LLMs. Specifically we findthat LLMs tend to produce positive outcomes when forced to decode other tokensinstead of critical tokens. Motivated by this observation we propose a novelapproach - cDPO - designed to automatically recognize and conduct token-levelrewards for the critical tokens during the alignment process. Specifically wedevelop a contrastive estimation approach to automatically identify criticaltokens. It is achieved by comparing the generation likelihood of positive andnegative models. To achieve this we separately fine-tune the positive andnegative models on various reasoning trajectories consequently they arecapable of identifying identify critical tokens within incorrect trajectoriesthat contribute to erroneous outcomes. Moreover to further align the modelwith the critical token information during the alignment process we extend theconventional DPO algorithms to token-level DPO and utilize the differentiallikelihood from the aforementioned positive and negative model as importantweight for token-level DPO learning.Experimental results on GSM8K and MATH500benchmarks with two-widely used models Llama-3 8B and 70B and deepseek-math7B demonstrate the effectiveness of the propsoed approach cDPO. |


| Item |Content|
| --- |---|
|idx| 2411.19941v1 |
|title| Perception Test 2024: Challenge Summary and a Novel Hour-Long VideoQA Benchmark |
|authors| Joseph HeywardJoão CarreiraDima DamenAndrew ZissermanViorica Pătrăucean
|links| http://arxiv.org/abs/2411.19941v1 |
|updated| 2024-11-29 18:57:25 UTC |
|summary| Following the successful 2023 edition we organised the Second PerceptionTest challenge as a half-day workshop alongside the IEEE/CVF EuropeanConference on Computer Vision ECCV 2024 with the goal of benchmarkingstate-of-the-art video models and measuring the progress since last year usingthe Perception Test benchmark. This year the challenge had seven tracks upfrom six last year and covered low-level and high-level tasks with languageand non-language interfaces across video audio and text modalities theadditional track covered hour-long video understanding and introduced a novelvideo QA benchmark 1h-walk VQA. Overall the tasks in the different trackswere: object tracking point tracking temporal action localisation temporalsound localisation multiple-choice video question-answering grounded videoquestion-answering and hour-long video question-answering. We summarise inthis report the challenge tasks and results and introduce in detail the novelhour-long video QA benchmark 1h-walk VQA. |


| Item |Content|
| --- |---|
|idx| 2411.19939v1 |
|title| VLSBench: Unveiling Visual Leakage in Multimodal Safety |
|authors| Xuhao HuDongrui LiuHao LiXuanjing HuangJing Shao
|links| http://arxiv.org/abs/2411.19939v1 |
|updated| 2024-11-29 18:56:37 UTC |
|summary| Safety concerns of Multimodal large language models MLLMs have graduallybecome an important problem in various applications. Surprisingly previousworks indicate a counter-intuitive phenomenon that using textual unlearning toalign MLLMs achieves comparable safety performances with MLLMs trained withimage-text pairs. To explain such a counter-intuitive phenomenon we discover avisual safety information leakage VSIL problem in existing multimodal safetybenchmarks i.e. the potentially risky and sensitive content in the image hasbeen revealed in the textual query. In this way MLLMs can easily refuse thesesensitive text-image queries according to textual queries. However image-textpairs without VSIL are common in real-world scenarios and are overlooked byexisting multimodal safety benchmarks. To this end we construct multimodalvisual leakless safety benchmark VLSBench preventing visual safety leakagefrom image to textual query with 2.4k image-text pairs. Experimental resultsindicate that VLSBench poses a significant challenge to both open-source andclose-source MLLMs including LLaVA Qwen2-VL Llama3.2-Vision and GPT-4o.This study demonstrates that textual alignment is enough for multimodal safetyscenarios with VSIL while multimodal alignment is a more promising solutionfor multimodal safety scenarios without VSIL. Please see our code and data at:http://hxhcreate.github.io/VLSBench |


| Item |Content|
| --- |---|
|idx| 2411.19930v1 |
|title| On Domain-Specific Post-Training for Multimodal Large Language Models |
|authors| Daixuan ChengShaohan HuangZiyu ZhuXintong ZhangWayne Xin ZhaoZhongzhi LuanBo DaiZhenliang Zhang
|links| http://arxiv.org/abs/2411.19930v1 |
|updated| 2024-11-29 18:42:28 UTC |
|summary| Recent years have witnessed the rapid development of general multimodal largelanguage models MLLMs. However adapting general MLLMs to specific domainssuch as scientific fields and industrial applications remains less explored.This paper systematically investigates domain adaptation of MLLMs throughpost-training focusing on data synthesis training pipelines and taskevaluation. 1 Data Synthesis: Using open-source models we develop a visualinstruction synthesizer that effectively generates diverse visual instructiontasks from domain-specific image-caption pairs. Our synthetic tasks surpassthose generated by manual rules GPT-4 and GPT-4V in enhancing thedomain-specific performance of MLLMs. 2 Training Pipeline: While thetwo-stage training--initially on image-caption pairs followed by visualinstruction tasks--is commonly adopted for developing general MLLMs we apply asingle-stage training pipeline to enhance task diversity for domain-specificpost-training. 3 Task Evaluation: We conduct experiments in two domainsbiomedicine and food by post-training MLLMs of different sources and scalese.g. Qwen2-VL-2B LLaVA-v1.6-8B Llama-3.2-11B and then evaluating MLLMperformance on various domain-specific tasks. To support further research inMLLM domain adaptation we will open-source our implementations. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2411.19946v1 |
|title| DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation |
|authors| Zhiqiang ShenAmmar SherifZeyuan YinShitong Shao
|links| http://arxiv.org/abs/2411.19946v1 |
|updated| 2024-11-29 18:59:46 UTC |
|summary| Recent advances in dataset distillation have led to solutions in two maindirections. The conventional batch-to-batch matching mechanism is ideal forsmall-scale datasets and includes bi-level optimization methods on models andsyntheses such as FRePo RCIG and RaT-BPTT as well as other methods likedistribution matching gradient matching and weight trajectory matching.Conversely batch-to-global matching typifies decoupled methods which areparticularly advantageous for large-scale datasets. This approach has garneredsubstantial interest within the community as seen in SRe2L G-VBSM WMDDand CDA. A primary challenge with the second approach is the lack of diversityamong syntheses within each class since samples are optimized independently andthe same global supervision signals are reused across different syntheticimages. In this study we propose a new Diversity-driven EarlyLate TrainingDELT scheme to enhance the diversity of images in batch-to-global matchingwith less computation. Our approach is conceptually simple yet effective itpartitions predefined IPC samples into smaller subtasks and employs localoptimizations to distill each subset into distributions from distinct phasesreducing the uniformity induced by the unified optimization process. Thesedistilled images from the subtasks demonstrate effective generalization whenapplied to the entire task. We conduct extensive experiments on CIFARTiny-ImageNet ImageNet-1K and its sub-datasets. Our approach outperforms theprevious state-of-the-art by 2sim5 on average across different datasets andIPCs images per class increasing diversity per class by more than 5 whilereducing synthesis time by up to 39.3 for enhancing the training efficiency.Code is available at: https://github.com/VILA-Lab/DELT. |


| Item |Content|
| --- |---|
|idx| 2411.19943v2 |
|title| Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability |
|authors| Zicheng LinTian LiangJiahao XuXing WangRuilin LuoChufan ShiSiheng LiYujiu YangZhaopeng Tu
|links| http://arxiv.org/abs/2411.19943v2 |
|updated| 2024-12-02 06:26:38 UTC |
|summary| Large Language Models LLMs have exhibited remarkable performance onreasoning tasks. They utilize autoregressive token generation to constructreasoning trajectories enabling the development of a coherent chain ofthought. In this work we explore the impact of individual tokens on the finaloutcomes of reasoning tasks. We identify the existence of critical tokensthat lead to incorrect reasoning trajectories in LLMs. Specifically we findthat LLMs tend to produce positive outcomes when forced to decode other tokensinstead of critical tokens. Motivated by this observation we propose a novelapproach - cDPO - designed to automatically recognize and conduct token-levelrewards for the critical tokens during the alignment process. Specifically wedevelop a contrastive estimation approach to automatically identify criticaltokens. It is achieved by comparing the generation likelihood of positive andnegative models. To achieve this we separately fine-tune the positive andnegative models on various reasoning trajectories consequently they arecapable of identifying identify critical tokens within incorrect trajectoriesthat contribute to erroneous outcomes. Moreover to further align the modelwith the critical token information during the alignment process we extend theconventional DPO algorithms to token-level DPO and utilize the differentiallikelihood from the aforementioned positive and negative model as importantweight for token-level DPO learning.Experimental results on GSM8K and MATH500benchmarks with two-widely used models Llama-3 8B and 70B and deepseek-math7B demonstrate the effectiveness of the propsoed approach cDPO. |


| Item |Content|
| --- |---|
|idx| 2411.19939v1 |
|title| VLSBench: Unveiling Visual Leakage in Multimodal Safety |
|authors| Xuhao HuDongrui LiuHao LiXuanjing HuangJing Shao
|links| http://arxiv.org/abs/2411.19939v1 |
|updated| 2024-11-29 18:56:37 UTC |
|summary| Safety concerns of Multimodal large language models MLLMs have graduallybecome an important problem in various applications. Surprisingly previousworks indicate a counter-intuitive phenomenon that using textual unlearning toalign MLLMs achieves comparable safety performances with MLLMs trained withimage-text pairs. To explain such a counter-intuitive phenomenon we discover avisual safety information leakage VSIL problem in existing multimodal safetybenchmarks i.e. the potentially risky and sensitive content in the image hasbeen revealed in the textual query. In this way MLLMs can easily refuse thesesensitive text-image queries according to textual queries. However image-textpairs without VSIL are common in real-world scenarios and are overlooked byexisting multimodal safety benchmarks. To this end we construct multimodalvisual leakless safety benchmark VLSBench preventing visual safety leakagefrom image to textual query with 2.4k image-text pairs. Experimental resultsindicate that VLSBench poses a significant challenge to both open-source andclose-source MLLMs including LLaVA Qwen2-VL Llama3.2-Vision and GPT-4o.This study demonstrates that textual alignment is enough for multimodal safetyscenarios with VSIL while multimodal alignment is a more promising solutionfor multimodal safety scenarios without VSIL. Please see our code and data at:http://hxhcreate.github.io/VLSBench |


| Item |Content|
| --- |---|
|idx| 2411.19922v1 |
|title| Dynamic EEG-fMRI mapping: Revealing the relationship between brain connectivity and cognitive state |
|authors| Guiran LiuBinrong Zhu
|links| http://arxiv.org/abs/2411.19922v1 |
|updated| 2024-11-29 18:36:58 UTC |
|summary| This study investigated the dynamic connectivity patterns between EEG andfMRI modalities contributing to our understanding of brain networkinteractions. By employing a comprehensive approach that integrated static anddynamic analyses of EEG-fMRI data we were able to uncover distinctconnectivity states and characterize their temporal fluctuations. The resultsrevealed modular organization within the intrinsic connectivity networks ICNsof the brain highlighting the significant roles of sensory systems and thedefault mode network. The use of a sliding window technique allowed us toassess how functional connectivity varies over time further elucidating thetransient nature of brain connectivity. Additionally our findings align withprevious literature reinforcing the notion that cognitive states can beeffectively identified through short-duration data specifically within the30-60 second timeframe. The established relationships between connectivitystrength and cognitive processes particularly during different visual statesunderscore the relevance of our approach for future research into braindynamics. Overall this study not only enhances our understanding of theinterplay between EEG and fMRI signals but also paves the way for furtherexploration into the neural correlates of cognitive functions and theirimplications in clinical settings. Future research should focus on refiningthese methodologies and exploring their applications in various cognitive andclinical contexts. |


| Item |Content|
| --- |---|
|idx| 2411.19921v1 |
|title| SIMS: Simulating Human-Scene Interactions with Real World Script Planning |
|authors| Wenjia WangLiang PanZhiyang DouZhouyingcheng LiaoYuke LouLei YangJingbo WangTaku Komura
|links| http://arxiv.org/abs/2411.19921v1 |
|updated| 2024-11-29 18:36:15 UTC |
|summary| Simulating long-term human-scene interaction is a challenging yet fascinatingtask. Previous works have not effectively addressed the generation of long-termhuman scene interactions with detailed narratives for physics-based animation.This paper introduces a novel framework for the planning and controlling oflong-horizon physical plausible human-scene interaction. On the one hand filmsand shows with stylish human locomotions or interactions with scenes areabundantly available on the internet providing a rich source of data forscript planning. On the other hand Large Language Models LLMs can understandand generate logical storylines.  This motivates us to marry the two by using an LLM-based pipeline to extractscripts from videos and then employ LLMs to imitate and create new scriptscapturing complex time-series human behaviors and interactions withenvironments. By leveraging this we utilize a dual-aware policy that achievesboth language comprehension and scene understanding to guide character motionswithin contextual and spatial constraints. To facilitate training andevaluation we contribute a comprehensive planning dataset containing diversemotion sequences extracted from real-world videos and expand them with largelanguage models. We also collect and re-annotate motion clips from existingkinematic datasets to enable our policy learn diverse skills. Extensiveexperiments demonstrate the effectiveness of our framework in versatile taskexecution and its generalization ability to various scenarios showingremarkably enhanced performance compared with existing methods. Our code anddata will be publicly available soon. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2411.19951v2 |
|title| T2Vid: Translating Long Text into Multi-Image is the Catalyst for Video-LLMs |
|authors| Shukang YinChaoyou FuSirui ZhaoYunhang ShenChunjiang GeYan YangZuwei LongYuhan DaiTong XuXing SunRan HeCaifeng ShanEnhong Chen
|links| http://arxiv.org/abs/2411.19951v2 |
|updated| 2024-12-02 06:54:47 UTC |
|summary| The success of Multimodal Large Language Models MLLMs in the image domainhas garnered wide attention from the research community. Drawing on previoussuccessful experiences researchers have recently explored extending thesuccess to the video understanding realms. Apart from training from scratch anefficient way is to utilize the pre-trained image-LLMs leading to twomainstream approaches i.e. zero-shot inference and further fine-tuning withvideo data. In this work our study of these approaches harvests an effectivedata augmentation method. We first make a deeper inspection of the zero-shotinference way and identify two limitations i.e. limited generalization andlack of temporal understanding capabilities. Thus we further investigate thefine-tuning approach and find a low learning efficiency when simply using allthe video data samples which can be attributed to a lack of instructiondiversity. Aiming at this issue we develop a method called T2Vid to synthesizevideo-like samples to enrich the instruction diversity in the training corpus.Integrating these data enables a simple and efficient training scheme whichachieves performance comparable to or even superior to using full videodatasets by training with just 15 the sample size. Meanwhile we find that theproposed scheme can boost the performance of long video understanding withouttraining with long video samples. We hope our study will spark more thinkingabout using MLLMs for video understanding and curation of high-quality data.The code is released at https://github.com/xjtupanda/T2Vid. |


| Item |Content|
| --- |---|
|idx| 2411.19950v1 |
|title| AlphaTablets: A Generic Plane Representation for 3D Planar Reconstruction from Monocular Videos |
|authors| Yuze HeWang ZhaoShaohui LiuYubin HuYushi BaiYu-Hui WenYong-Jin Liu
|links| http://arxiv.org/abs/2411.19950v1 |
|updated| 2024-11-29 18:59:52 UTC |
|summary| We introduce AlphaTablets a novel and generic representation of 3D planesthat features continuous 3D surface and precise boundary delineation. Byrepresenting 3D planes as rectangles with alpha channels AlphaTablets combinethe advantages of current 2D and 3D plane representations enabling accurateconsistent and flexible modeling of 3D planes. We derive differentiablerasterization on top of AlphaTablets to efficiently render 3D planes intoimages and propose a novel bottom-up pipeline for 3D planar reconstructionfrom monocular videos. Starting with 2D superpixels and geometric cues frompre-trained models we initialize 3D planes as AlphaTablets and optimize themvia differentiable rendering. An effective merging scheme is introduced tofacilitate the growth and refinement of AlphaTablets. Through iterativeoptimization and merging we reconstruct complete and accurate 3D planes withsolid surfaces and clear boundaries. Extensive experiments on the ScanNetdataset demonstrate state-of-the-art performance in 3D planar reconstructionunderscoring the great potential of AlphaTablets as a generic 3D planerepresentation for various applications. Project page is available at:https://hyzcluster.github.io/alphatablets |


| Item |Content|
| --- |---|
|idx| 2411.19946v1 |
|title| DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation |
|authors| Zhiqiang ShenAmmar SherifZeyuan YinShitong Shao
|links| http://arxiv.org/abs/2411.19946v1 |
|updated| 2024-11-29 18:59:46 UTC |
|summary| Recent advances in dataset distillation have led to solutions in two maindirections. The conventional batch-to-batch matching mechanism is ideal forsmall-scale datasets and includes bi-level optimization methods on models andsyntheses such as FRePo RCIG and RaT-BPTT as well as other methods likedistribution matching gradient matching and weight trajectory matching.Conversely batch-to-global matching typifies decoupled methods which areparticularly advantageous for large-scale datasets. This approach has garneredsubstantial interest within the community as seen in SRe2L G-VBSM WMDDand CDA. A primary challenge with the second approach is the lack of diversityamong syntheses within each class since samples are optimized independently andthe same global supervision signals are reused across different syntheticimages. In this study we propose a new Diversity-driven EarlyLate TrainingDELT scheme to enhance the diversity of images in batch-to-global matchingwith less computation. Our approach is conceptually simple yet effective itpartitions predefined IPC samples into smaller subtasks and employs localoptimizations to distill each subset into distributions from distinct phasesreducing the uniformity induced by the unified optimization process. Thesedistilled images from the subtasks demonstrate effective generalization whenapplied to the entire task. We conduct extensive experiments on CIFARTiny-ImageNet ImageNet-1K and its sub-datasets. Our approach outperforms theprevious state-of-the-art by 2sim5 on average across different datasets andIPCs images per class increasing diversity per class by more than 5 whilereducing synthesis time by up to 39.3 for enhancing the training efficiency.Code is available at: https://github.com/VILA-Lab/DELT. |


| Item |Content|
| --- |---|
|idx| 2411.19943v2 |
|title| Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability |
|authors| Zicheng LinTian LiangJiahao XuXing WangRuilin LuoChufan ShiSiheng LiYujiu YangZhaopeng Tu
|links| http://arxiv.org/abs/2411.19943v2 |
|updated| 2024-12-02 06:26:38 UTC |
|summary| Large Language Models LLMs have exhibited remarkable performance onreasoning tasks. They utilize autoregressive token generation to constructreasoning trajectories enabling the development of a coherent chain ofthought. In this work we explore the impact of individual tokens on the finaloutcomes of reasoning tasks. We identify the existence of critical tokensthat lead to incorrect reasoning trajectories in LLMs. Specifically we findthat LLMs tend to produce positive outcomes when forced to decode other tokensinstead of critical tokens. Motivated by this observation we propose a novelapproach - cDPO - designed to automatically recognize and conduct token-levelrewards for the critical tokens during the alignment process. Specifically wedevelop a contrastive estimation approach to automatically identify criticaltokens. It is achieved by comparing the generation likelihood of positive andnegative models. To achieve this we separately fine-tune the positive andnegative models on various reasoning trajectories consequently they arecapable of identifying identify critical tokens within incorrect trajectoriesthat contribute to erroneous outcomes. Moreover to further align the modelwith the critical token information during the alignment process we extend theconventional DPO algorithms to token-level DPO and utilize the differentiallikelihood from the aforementioned positive and negative model as importantweight for token-level DPO learning.Experimental results on GSM8K and MATH500benchmarks with two-widely used models Llama-3 8B and 70B and deepseek-math7B demonstrate the effectiveness of the propsoed approach cDPO. |


| Item |Content|
| --- |---|
|idx| 2411.19942v1 |
|title| Free-form Generation Enhances Challenging Clothed Human Modeling |
|authors| Hang YeXiaoxuan MaHai CiWentao ZhuYizhou Wang
|links| http://arxiv.org/abs/2411.19942v1 |
|updated| 2024-11-29 18:58:17 UTC |
|summary| Achieving realistic animated human avatars requires accurate modeling ofpose-dependent clothing deformations. Existing learning-based methods heavilyrely on the Linear Blend Skinning LBS of minimally-clothed human models likeSMPL to model deformation. However these methods struggle to handle looseclothing such as long dresses where the canonicalization process becomesill-defined when the clothing is far from the body leading to disjointed andfragmented results. To overcome this limitation we propose a novel hybridframework to model challenging clothed humans. Our core idea is to usededicated strategies to model different regions depending on whether they areclose to or distant from the body. Specifically we segment the human body intothree categories: unclothed deformed and generated. We simply replicateunclothed regions that require no deformation. For deformed regions close tothe body we leverage LBS to handle the deformation. As for the generatedregions which correspond to loose clothing areas we introduce a novelfree-form part-aware generator to model them as they are less affected bymovements. This free-form generation paradigm brings enhanced flexibility andexpressiveness to our hybrid framework enabling it to capture the intricategeometric details of challenging loose clothing such as skirts and dresses.Experimental results on the benchmark dataset featuring loose clothingdemonstrate that our method achieves state-of-the-art performance with superiorvisual fidelity and realism particularly in the most challenging cases. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2411.19951v2 |
|title| T2Vid: Translating Long Text into Multi-Image is the Catalyst for Video-LLMs |
|authors| Shukang YinChaoyou FuSirui ZhaoYunhang ShenChunjiang GeYan YangZuwei LongYuhan DaiTong XuXing SunRan HeCaifeng ShanEnhong Chen
|links| http://arxiv.org/abs/2411.19951v2 |
|updated| 2024-12-02 06:54:47 UTC |
|summary| The success of Multimodal Large Language Models MLLMs in the image domainhas garnered wide attention from the research community. Drawing on previoussuccessful experiences researchers have recently explored extending thesuccess to the video understanding realms. Apart from training from scratch anefficient way is to utilize the pre-trained image-LLMs leading to twomainstream approaches i.e. zero-shot inference and further fine-tuning withvideo data. In this work our study of these approaches harvests an effectivedata augmentation method. We first make a deeper inspection of the zero-shotinference way and identify two limitations i.e. limited generalization andlack of temporal understanding capabilities. Thus we further investigate thefine-tuning approach and find a low learning efficiency when simply using allthe video data samples which can be attributed to a lack of instructiondiversity. Aiming at this issue we develop a method called T2Vid to synthesizevideo-like samples to enrich the instruction diversity in the training corpus.Integrating these data enables a simple and efficient training scheme whichachieves performance comparable to or even superior to using full videodatasets by training with just 15 the sample size. Meanwhile we find that theproposed scheme can boost the performance of long video understanding withouttraining with long video samples. We hope our study will spark more thinkingabout using MLLMs for video understanding and curation of high-quality data.The code is released at https://github.com/xjtupanda/T2Vid. |


| Item |Content|
| --- |---|
|idx| 2411.19950v1 |
|title| AlphaTablets: A Generic Plane Representation for 3D Planar Reconstruction from Monocular Videos |
|authors| Yuze HeWang ZhaoShaohui LiuYubin HuYushi BaiYu-Hui WenYong-Jin Liu
|links| http://arxiv.org/abs/2411.19950v1 |
|updated| 2024-11-29 18:59:52 UTC |
|summary| We introduce AlphaTablets a novel and generic representation of 3D planesthat features continuous 3D surface and precise boundary delineation. Byrepresenting 3D planes as rectangles with alpha channels AlphaTablets combinethe advantages of current 2D and 3D plane representations enabling accurateconsistent and flexible modeling of 3D planes. We derive differentiablerasterization on top of AlphaTablets to efficiently render 3D planes intoimages and propose a novel bottom-up pipeline for 3D planar reconstructionfrom monocular videos. Starting with 2D superpixels and geometric cues frompre-trained models we initialize 3D planes as AlphaTablets and optimize themvia differentiable rendering. An effective merging scheme is introduced tofacilitate the growth and refinement of AlphaTablets. Through iterativeoptimization and merging we reconstruct complete and accurate 3D planes withsolid surfaces and clear boundaries. Extensive experiments on the ScanNetdataset demonstrate state-of-the-art performance in 3D planar reconstructionunderscoring the great potential of AlphaTablets as a generic 3D planerepresentation for various applications. Project page is available at:https://hyzcluster.github.io/alphatablets |


| Item |Content|
| --- |---|
|idx| 2411.19946v1 |
|title| DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation |
|authors| Zhiqiang ShenAmmar SherifZeyuan YinShitong Shao
|links| http://arxiv.org/abs/2411.19946v1 |
|updated| 2024-11-29 18:59:46 UTC |
|summary| Recent advances in dataset distillation have led to solutions in two maindirections. The conventional batch-to-batch matching mechanism is ideal forsmall-scale datasets and includes bi-level optimization methods on models andsyntheses such as FRePo RCIG and RaT-BPTT as well as other methods likedistribution matching gradient matching and weight trajectory matching.Conversely batch-to-global matching typifies decoupled methods which areparticularly advantageous for large-scale datasets. This approach has garneredsubstantial interest within the community as seen in SRe2L G-VBSM WMDDand CDA. A primary challenge with the second approach is the lack of diversityamong syntheses within each class since samples are optimized independently andthe same global supervision signals are reused across different syntheticimages. In this study we propose a new Diversity-driven EarlyLate TrainingDELT scheme to enhance the diversity of images in batch-to-global matchingwith less computation. Our approach is conceptually simple yet effective itpartitions predefined IPC samples into smaller subtasks and employs localoptimizations to distill each subset into distributions from distinct phasesreducing the uniformity induced by the unified optimization process. Thesedistilled images from the subtasks demonstrate effective generalization whenapplied to the entire task. We conduct extensive experiments on CIFARTiny-ImageNet ImageNet-1K and its sub-datasets. Our approach outperforms theprevious state-of-the-art by 2sim5 on average across different datasets andIPCs images per class increasing diversity per class by more than 5 whilereducing synthesis time by up to 39.3 for enhancing the training efficiency.Code is available at: https://github.com/VILA-Lab/DELT. |


| Item |Content|
| --- |---|
|idx| 2411.19942v1 |
|title| Free-form Generation Enhances Challenging Clothed Human Modeling |
|authors| Hang YeXiaoxuan MaHai CiWentao ZhuYizhou Wang
|links| http://arxiv.org/abs/2411.19942v1 |
|updated| 2024-11-29 18:58:17 UTC |
|summary| Achieving realistic animated human avatars requires accurate modeling ofpose-dependent clothing deformations. Existing learning-based methods heavilyrely on the Linear Blend Skinning LBS of minimally-clothed human models likeSMPL to model deformation. However these methods struggle to handle looseclothing such as long dresses where the canonicalization process becomesill-defined when the clothing is far from the body leading to disjointed andfragmented results. To overcome this limitation we propose a novel hybridframework to model challenging clothed humans. Our core idea is to usededicated strategies to model different regions depending on whether they areclose to or distant from the body. Specifically we segment the human body intothree categories: unclothed deformed and generated. We simply replicateunclothed regions that require no deformation. For deformed regions close tothe body we leverage LBS to handle the deformation. As for the generatedregions which correspond to loose clothing areas we introduce a novelfree-form part-aware generator to model them as they are less affected bymovements. This free-form generation paradigm brings enhanced flexibility andexpressiveness to our hybrid framework enabling it to capture the intricategeometric details of challenging loose clothing such as skirts and dresses.Experimental results on the benchmark dataset featuring loose clothingdemonstrate that our method achieves state-of-the-art performance with superiorvisual fidelity and realism particularly in the most challenging cases. |


| Item |Content|
| --- |---|
|idx| 2411.19941v1 |
|title| Perception Test 2024: Challenge Summary and a Novel Hour-Long VideoQA Benchmark |
|authors| Joseph HeywardJoão CarreiraDima DamenAndrew ZissermanViorica Pătrăucean
|links| http://arxiv.org/abs/2411.19941v1 |
|updated| 2024-11-29 18:57:25 UTC |
|summary| Following the successful 2023 edition we organised the Second PerceptionTest challenge as a half-day workshop alongside the IEEE/CVF EuropeanConference on Computer Vision ECCV 2024 with the goal of benchmarkingstate-of-the-art video models and measuring the progress since last year usingthe Perception Test benchmark. This year the challenge had seven tracks upfrom six last year and covered low-level and high-level tasks with languageand non-language interfaces across video audio and text modalities theadditional track covered hour-long video understanding and introduced a novelvideo QA benchmark 1h-walk VQA. Overall the tasks in the different trackswere: object tracking point tracking temporal action localisation temporalsound localisation multiple-choice video question-answering grounded videoquestion-answering and hour-long video question-answering. We summarise inthis report the challenge tasks and results and introduce in detail the novelhour-long video QA benchmark 1h-walk VQA. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2411.19933v1 |
|title| Transfer Learning for High-dimensional Quantile Regression with Distribution Shift |
|authors| Ruiqi BaiYijiao ZhangHanbo YangZhongyi Zhu
|links| http://arxiv.org/abs/2411.19933v1 |
|updated| 2024-11-29 18:49:55 UTC |
|summary| Information from related source studies can often enhance the findings of atarget study. However the distribution shift between target and source studiescan severely impact the efficiency of knowledge transfer. In thehigh-dimensional regression setting existing transfer approaches mainly focuson the parameter shift. In this paper we focus on the high-dimensionalquantile regression with knowledge transfer under three types of distributionshift: parameter shift covariate shift and residual shift. We propose a noveltransferable set and a new transfer framework to address the above threediscrepancies. Non-asymptotic estimation error bounds and source detectionconsistency are established to validate the availability and superiority of ourmethod in the presence of distribution shift. Additionally an orthogonaldebiased approach is proposed for statistical inference with knowledgetransfer leading to sharper asymptotic results. Extensive simulation resultsas well as real data applications further demonstrate the effectiveness of ourproposed procedure. |


| Item |Content|
| --- |---|
|idx| 2411.19923v1 |
|title| Scalable Out-of-distribution Robustness in the Presence of Unobserved Confounders |
|authors| Parjanya PrashantSeyedeh Baharan KhatamiBruno RibeiroBabak Salimi
|links| http://arxiv.org/abs/2411.19923v1 |
|updated| 2024-11-29 18:38:17 UTC |
|summary| We consider the task of out-of-distribution OOD generalization where thedistribution shift is due to an unobserved confounder Z affecting both thecovariates X and the labels Y. In this setting traditional assumptionsof covariate and label shift are unsuitable due to the confounding whichintroduces heterogeneity in the predictor i.e. hatY  f_ZX. OODgeneralization differs from traditional domain adaptation by not assumingaccess to the covariate distribution Xtextte of the test samples duringtraining. These conditions create a challenging scenario for OOD robustness:a Ztexttr is an unobserved confounder during training bPtextteZ neq PtexttrZ c Xtextte is unavailable duringtraining and d the posterior predictive distribution depends onPtextteZ i.e. hatY  E_PtextteZf_ZX. In generalaccurate predictions are unattainable in this scenario and existing literaturehas proposed complex predictors based on identifiability assumptions thatrequire multiple additional variables. Our work investigates a set ofidentifiability assumptions that tremendously simplify the predictor whoseresulting elegant simplicity outperforms existing approaches. |


| Item |Content|
| --- |---|
|idx| 2411.19920v1 |
|title| Geometry of fibers of the multiplication map of deep linear neural networks |
|authors| SImon Pepin LehalleurRichárd Rimányi
|links| http://arxiv.org/abs/2411.19920v1 |
|updated| 2024-11-29 18:36:03 UTC |
|summary| We study the geometry of the algebraic set of tuples of composable matriceswhich multiply to a fixed matrix using tools from the theory of quiverrepresentations. In particular we determine its codimension C and the numbertheta of its top-dimensional irreducible components. Our solution ispresented in three forms: a Poincare series in equivariant cohomology aquadratic integer program and an explicit formula. In the course of the proofwe establish a surprising property: C and theta are invariant underarbitrary permutations of the dimension vector. We also show that the reallog-canonical threshold of the function taking a tuple to the square Frobeniusnorm of its product is C/2. These results are motivated by the study of deeplinear neural networks in machine learning and Bayesian statistics singularlearning theory and show that deep linear networks are in a certain sensemildly singular. |


| Item |Content|
| --- |---|
|idx| 2411.19908v1 |
|title| Another look at inference after prediction |
|authors| Jessica GronsbellJianhui GaoYaqi ShiZachary R. McCawDavid Cheng
|links| http://arxiv.org/abs/2411.19908v1 |
|updated| 2024-11-29 18:12:50 UTC |
|summary| Prediction-based PB inference is increasingly used in applications wherethe outcome of interest is difficult to obtain but its predictors are readilyavailable. Unlike traditional inference PB inference performs statisticalinference using a partially observed outcome and a set of covariates byleveraging a prediction of the outcome generated from a machine learning MLmodel. Motwani and Witten 2023 recently revisited two innovative PB inferenceapproaches for ordinary least squares. They found that the method proposed byWang et al. 2020 yields a consistent estimator for the association ofinterest when the ML model perfectly captures the underlying regressionfunction. Conversely the prediction-powered inference PPI method proposed byAngelopoulos et al. 2023 yields valid inference regardless of the modelsaccuracy. In this paper we study the statistical efficiency of the PPIestimator. Our analysis reveals that a more efficient estimator proposed 25years ago by Chen and Chen 2000 can be obtained by simply adding a weight tothe PPI estimator. We also contextualize PB inference with methods from theeconomics and statistics literature dating back to the 1960s. Our extensivetheoretical and numerical analyses indicate that the Chen and Chen CCestimator offers a balance between robustness to ML model specification andstatistical efficiency making it the preferred choice for use in practice. |


| Item |Content|
| --- |---|
|idx| 2411.19902v1 |
|title| Noncommutative Model Selection for Data Clustering and Dimension Reduction Using Relative von Neumann Entropy |
|authors| Araceli Guzmán-TristánAntonio Rieser
|links| http://arxiv.org/abs/2411.19902v1 |
|updated| 2024-11-29 18:04:11 UTC |
|summary| We propose a pair of completely data-driven algorithms for unsupervisedclassification and dimension reduction and we empirically study theirperformance on a number of data sets both simulated data in three-dimensionsand images from the COIL-20 data set. The algorithms take as input a set ofpoints sampled from a uniform distribution supported on a metric space thelatter embedded in an ambient metric space and they output a clustering orreduction of dimension of the data. They work by constructing a natural familyof graphs from the data and selecting the graph which maximizes the relativevon Neumann entropy of certain normalized heat operators constructed from thegraphs. Once the appropriate graph is selected the eigenvectors of the graphLaplacian may be used to reduce the dimension of the data and clusters in thedata may be identified with the kernel of the associated graph Laplacian.Notably these algorithms do not require information about the size of aneighborhood or the desired number of clusters as input in contrast to popularalgorithms such as k-means and even more modern spectral methods such asLaplacian eigenmaps among others.  In our computational experiments our clustering algorithm outperformsk-means clustering on data sets with non-trivial geometry and topology inparticular data whose clusters are not concentrated around a specific pointand our dimension reduction algorithm is shown to work well in several simpleexamples. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2411.19922v1 |
|title| Dynamic EEG-fMRI mapping: Revealing the relationship between brain connectivity and cognitive state |
|authors| Guiran LiuBinrong Zhu
|links| http://arxiv.org/abs/2411.19922v1 |
|updated| 2024-11-29 18:36:58 UTC |
|summary| This study investigated the dynamic connectivity patterns between EEG andfMRI modalities contributing to our understanding of brain networkinteractions. By employing a comprehensive approach that integrated static anddynamic analyses of EEG-fMRI data we were able to uncover distinctconnectivity states and characterize their temporal fluctuations. The resultsrevealed modular organization within the intrinsic connectivity networks ICNsof the brain highlighting the significant roles of sensory systems and thedefault mode network. The use of a sliding window technique allowed us toassess how functional connectivity varies over time further elucidating thetransient nature of brain connectivity. Additionally our findings align withprevious literature reinforcing the notion that cognitive states can beeffectively identified through short-duration data specifically within the30-60 second timeframe. The established relationships between connectivitystrength and cognitive processes particularly during different visual statesunderscore the relevance of our approach for future research into braindynamics. Overall this study not only enhances our understanding of theinterplay between EEG and fMRI signals but also paves the way for furtherexploration into the neural correlates of cognitive functions and theirimplications in clinical settings. Future research should focus on refiningthese methodologies and exploring their applications in various cognitive andclinical contexts. |


| Item |Content|
| --- |---|
|idx| 2411.19727v1 |
|title| SoK: Detection and Repair of Accessibility Issues |
|authors| Liming NieHao LiuJing SunKabir Sulaiman SaidShanshan HongLei XueZhiyuan WeiYangyang ZhaoMeng Li
|links| http://arxiv.org/abs/2411.19727v1 |
|updated| 2024-11-29 14:19:19 UTC |
|summary| There is an increasing global emphasis on information accessibility withnumerous researchers actively developing automated tools to detect and repairaccessibility issues thereby ensuring that individuals with diverse abilitiescan independently access software products and services. However currentresearch still encounters significant challenges in two key areas: the absenceof a comprehensive taxonomy of accessibility issue types and the lack ofcomprehensive analysis of the capabilities of detection and repair tools aswell as the status of corresponding datasets. To address these challenges thispaper introduces the Accessibility Issue Analysis AIA framework. Utilizingthis framework we develop a comprehensive taxonomy that categorizes 55 typesof accessibility issues across four pivotal dimensions: PerceivabilityOperability Understandability and Robustness. This taxonomy has beenrigorously recognized through a questionnaire survey n130. Building on thistaxonomy we conduct an in-depth analysis of existing detection and repairtools as well as the status of corresponding datasets. In terms of tools ourfindings indicate that 14 detection tools can identify 31 issue typesachieving a 56.3 rate 31/55. Meanwhile 9 repair tools address just 13 issuetypes with a 23.6 rate. In terms of datasets those for detection tools cover21 issue types at a 38.1 coverage rate whereas those for repair tools coveronly 7 types at a 12.7 coverage rate. |


| Item |Content|
| --- |---|
|idx| 2411.19576v1 |
|title| A Review of LLM-based Explanations in Recommender Systems |
|authors| Alan Said
|links| http://arxiv.org/abs/2411.19576v1 |
|updated| 2024-11-29 09:47:32 UTC |
|summary| The rise of Large Language Models LLMs such as LLaMA and ChatGPT hasopened new opportunities for enhancing recommender systems through improvedexplainability. This paper provides a systematic literature review focused onleveraging LLMs to generate explanations for recommendations -- a criticalaspect for fostering transparency and user trust. We conducted a comprehensivesearch within the ACM Guide to Computing Literature covering publications fromthe launch of ChatGPT November 2022 to the present November 2024. Oursearch yielded 232 articles but after applying inclusion criteria only sixwere identified as directly addressing the use of LLMs in explainingrecommendations. This scarcity highlights that despite the rise of LLMs theirapplication in explainable recommender systems is still in an early stage. Weanalyze these select studies to understand current methodologies identifychallenges and suggest directions for future research. Our findings underscorethe potential of LLMs improving explanations of recommender systems andencourage the development of more transparent and user-centric recommendationexplanation solutions. |


| Item |Content|
| --- |---|
|idx| 2411.19554v1 |
|title| Unimib Assistant: designing a student-friendly RAG-based chatbot for all their needs |
|authors| Chiara AnticoStefano GiordanoCansu KoyuturkDimitri Ognibene
|links| http://arxiv.org/abs/2411.19554v1 |
|updated| 2024-11-29 09:07:21 UTC |
|summary| Natural language processing skills of Large Language Models LLMs areunprecedented having wide diffusion and application in different tasks. Thispilot study focuses on specializing ChatGPT behavior through aRetrieval-Augmented Generation RAG system using the OpenAI custom GPTsfeature. The purpose of our chatbot called Unimib Assistant is to provideinformation and solutions to the specific needs of University of Milano-BicoccaUnimib students through a question-answering approach. We provided the systemwith a prompt highlighting its specific purpose and behavior as well asuniversity-related documents and links obtained from an initial need-findingphase interviewing six students. After a preliminary customization phase aqualitative usability test was conducted with six other students to identifythe strengths and weaknesses of the chatbot with the goal of improving it in asubsequent redesign phase. While the chatbot was appreciated for itsuser-friendly experience perceived general reliability well-structuredresponses and conversational tone several significant technical andfunctional limitations emerged. In particular the satisfaction and overallexperience of the users was impaired by the systems inability to alwaysprovide fully accurate information. Moreover it would often neglect to reportrelevant information even if present in the materials uploaded and promptgiven. Furthermore it sometimes generated unclickable links undermining itstrustworthiness since providing the source of information was an importantaspect for our users. Further in-depth studies and feedback from other users aswell as implementation iterations are planned to refine our Unimib Assistant. |


| Item |Content|
| --- |---|
|idx| 2411.19502v1 |
|title| Knowledge-Data Fusion Based Source-Free Semi-Supervised Domain Adaptation for Seizure Subtype Classification |
|authors| Ruimin PengJiayu AnDongrui Wu
|links| http://arxiv.org/abs/2411.19502v1 |
|updated| 2024-11-29 06:40:45 UTC |
|summary| Electroencephalogram EEG-based seizure subtype classification enhancesclinical diagnosis efficiency. Source-free semi-supervised domain adaptationSF-SSDA which transfers a pre-trained model to a new dataset with no sourcedata and limited labeled target data can be used for privacy-preservingseizure subtype classification. This paper considers two challenges in SF-SSDAfor EEG-based seizure subtype classification: 1 How to effectively fuse bothraw EEG data and expert knowledge in classifier design 2 How to align thesource and target domain distributions for SF-SSDA We propose a Knowledge-DataFusion based SF-SSDA approach KDF-MutualSHOT for EEG-based seizure subtypeclassification. In source model training KDF uses Jensen-Shannon Divergence tofacilitate mutual learning between a feature-driven Decision Tree-based modeland a data-driven Transformer-based model. To adapt KDF to a new targetdataset an SF-SSDA algorithm MutualSHOT is developed which features aconsistency-based pseudo-label selection strategy. Experiments on the publicTUSZ and CHSZ datasets demonstrated that KDF-MutualSHOT outperformed othersupervised and source-free domain adaptation approaches in cross-subjectseizure subtype classification. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2411.19866v1 |
|title| Misinformation Dissemination: Effects of Network Density in Segregated Communities |
|authors| Soroush KarimiMarcos OliveiraDiogo Pacheco
|links| http://arxiv.org/abs/2411.19866v1 |
|updated| 2024-11-29 17:27:54 UTC |
|summary| Understanding the relationship between network features and misinformationpropagation is crucial for mitigating the spread of false information. Here weinvestigate how network density and segregation affect the dissemination ofmisinformation using a susceptible-infectious-recovered framework. We find thata higher density consistently increases the proportion of misinformationbelievers. In segregated networks our results reveal that minorities affectthe majority: denser minority groups increase the number of believers in themajority demonstrating how the structure of a segregated minority caninfluence misinformation dynamics within the majority group. |


| Item |Content|
| --- |---|
|idx| 2411.19747v1 |
|title| A Multi-Loss Strategy for Vehicle Trajectory Prediction: Combining Off-Road, Diversity, and Directional Consistency Losses |
|authors| Ahmad RahimiAlexandre Alahi
|links| http://arxiv.org/abs/2411.19747v1 |
|updated| 2024-11-29 14:47:08 UTC |
|summary| Trajectory prediction is essential for the safety and efficiency of planningin autonomous vehicles. However current models often fail to fully capturecomplex traffic rules and the complete range of potential vehicle movements.Addressing these limitations this study introduces three novel loss functions:Offroad Loss Direction Consistency Error and Diversity Loss. These functionsare designed to keep predicted paths within driving area boundaries alignedwith traffic directions and cover a wider variety of plausible drivingscenarios. As all prediction modes should adhere to road rules and conditionsthis work overcomes the shortcomings of traditional winner takes all trainingmethods by applying the loss functions to all prediction modes. These lossfunctions not only improve model training but can also serve as metrics forevaluating the realism and diversity of trajectory predictions. Extensivevalidation on the nuScenes and Argoverse 2 datasets with leading baselinemodels demonstrates that our approach not only maintains accuracy butsignificantly improves safety and robustness reducing offroad errors onaverage by 47 on original and by 37 on attacked scenes. This work sets a newbenchmark for trajectory prediction in autonomous driving offering substantialimprovements in navigating complex environments. Our code is available athttps://github.com/vita-epfl/stay-on-track . |


| Item |Content|
| --- |---|
|idx| 2411.19746v1 |
|title| HVAC-DPT: A Decision Pretrained Transformer for HVAC Control |
|authors| Anaïs Berkes
|links| http://arxiv.org/abs/2411.19746v1 |
|updated| 2024-11-29 14:46:37 UTC |
|summary| Building operations consume approximately 40 of global energy with HeatingVentilation and Air Conditioning HVAC systems responsible for up to 50 ofthis consumption. As HVAC energy demands are expected to rise optimisingsystem efficiency is crucial for reducing future energy use and mitigatingclimate change. Existing control strategies lack generalisation and requireextensive training and data limiting their rapid deployment across diversebuildings. This paper introduces HVAC-DPT a Decision-Pretrained Transformerusing in-context Reinforcement Learning RL for multi-zone HVAC control.HVAC-DPT frames HVAC control as a sequential prediction task training a causaltransformer on interaction histories generated by diverse RL agents. Thisapproach enables HVAC-DPT to refine its policy in-context without modifyingnetwork parameters allowing for deployment across different buildings withoutthe need for additional training or data collection. HVAC-DPT reduces energyconsumption in unseen buildings by 45 compared to the baseline controlleroffering a scalable and effective approach to mitigating the increasingenvironmental impact of HVAC systems. |


| Item |Content|
| --- |---|
|idx| 2411.19639v1 |
|title| RMIO: A Model-Based MARL Framework for Scenarios with Observation Loss in Some Agents |
|authors| Shi ZifengLiu MeiqinZhang SenlinZheng RonghaoDong Shanling
|links| http://arxiv.org/abs/2411.19639v1 |
|updated| 2024-11-29 11:45:21 UTC |
|summary| In recent years model-based reinforcement learning MBRL has emerged as asolution to address sample complexity in multi-agent reinforcement learningMARL by modeling agent-environment dynamics to improve sample efficiency.However most MBRL methods assume complete and continuous observations fromeach agent during the inference stage which can be overly idealistic inpractical applications. A novel model-based MARL approach called RMIO isintroduced to address this limitation specifically designed for scenarioswhere observation is lost in some agent. RMIO leverages the world model toreconstruct missing observations and further reduces reconstruction errorsthrough inter-agent information integration to ensure stable multi-agentdecision-making. Secondly unlike CTCE methods such as MAMBA RMIO adopts theCTDE paradigm in standard environment and enabling limited communication onlywhen agents lack observation data thereby reducing reliance on communication.Additionally RMIO improves asymptotic performance through strategies such asreward smoothing a dual-layer experience replay buffer and an RNN-augmentedpolicy model surpassing previous work. Our experiments conducted in both theSMAC and MaMuJoCo environments demonstrate that RMIO outperforms currentstate-of-the-art approaches in terms of asymptotic convergence performance andpolicy robustness both in standard mission settings and in scenarios involvingobservation loss. |


| Item |Content|
| --- |---|
|idx| 2411.19526v1 |
|title| A Local Information Aggregation based Multi-Agent Reinforcement Learning for Robot Swarm Dynamic Task Allocation |
|authors| Yang LvJinlong LeiPeng Yi
|links| http://arxiv.org/abs/2411.19526v1 |
|updated| 2024-11-29 07:53:05 UTC |
|summary| In this paper we explore how to optimize task allocation for robot swarms indynamic environments emphasizing the necessity of formulating robustflexible and scalable strategies for robot cooperation. We introduce a novelframework using a decentralized partially observable Markov decision processDec_POMDP specifically designed for distributed robot swarm networks. At thecore of our methodology is the Local Information Aggregation Multi-Agent DeepDeterministic Policy Gradient LIA_MADDPG algorithm which merges centralizedtraining with distributed execution CTDE. During the centralized trainingphase a local information aggregation LIA module is meticulously designed togather critical data from neighboring robots enhancing decision-makingefficiency. In the distributed execution phase a strategy improvement methodis proposed to dynamically adjust task allocation based on changing andpartially observable environmental conditions. Our empirical evaluations showthat the LIA module can be seamlessly integrated into various CTDE-based MARLmethods significantly enhancing their performance. Additionally by comparingLIA_MADDPG with six conventional reinforcement learning algorithms and aheuristic algorithm we demonstrate its superior scalability rapid adaptationto environmental changes and ability to maintain both stability andconvergence speed. These results underscore LIA_MADDPGs outstandingperformance and its potential to significantly improve dynamic task allocationin robot swarms through enhanced local collaboration and adaptive strategyexecution. |


