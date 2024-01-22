# cs.CL 

| Item |Content|
| --- |---|
|idx| 2401.10225v1 |
|title| ChatQA: Building GPT-4 Level Conversational QA Models |
|authors| Zihan LiuWei PingRajarshi RoyPeng XuMohammad ShoeybiBryan Catanzaro
|links| http://arxiv.org/abs/2401.10225v1 |
|updated| 2024-01-18 18:59:11 UTC |
|summary| In this work we introduce ChatQA a family of conversational questionanswering QA models that obtain GPT-4 level accuracies. Specifically wepropose a two-stage instruction tuning method that can significantly improvethe zero-shot conversational QA results from large language models LLMs. Tohandle retrieval in conversational QA we fine-tune a dense retriever on amulti-turn QA dataset which provides comparable results to using thestate-of-the-art query rewriting model while largely reducing deployment cost.Notably our ChatQA-70B can outperform GPT-4 in terms of average score on 10conversational QA datasets 54.14 vs. 53.90 without relying on any syntheticdata from OpenAI GPT models. |


| Item |Content|
| --- |---|
|idx| 2401.10208v1 |
|title| MM-Interleaved: Interleaved Image-Text Generative Modeling via Multi-modal Feature Synchronizer |
|authors| Changyao TianXizhou ZhuYuwen XiongWeiyun WangZhe ChenWenhai WangYuntao ChenLewei LuTong LuJie ZhouHongsheng LiYu QiaoJifeng Dai
|links| http://arxiv.org/abs/2401.10208v1 |
|updated| 2024-01-18 18:50:16 UTC |
|summary| Developing generative models for interleaved image-text data has bothresearch and practical value. It requires models to understand the interleavedsequences and subsequently generate images and text. However existing attemptsare limited by the issue that the fixed number of visual tokens cannotefficiently capture image details which is particularly problematic in themulti-image scenarios. To address this this paper presents MM-Interleaved anend-to-end generative model for interleaved image-text data. It introduces amulti-scale and multi-image feature synchronizer module allowing direct accessto fine-grained image features in the previous context during the generationprocess. MM-Interleaved is end-to-end pre-trained on both paired andinterleaved image-text corpora. It is further enhanced through a supervisedfine-tuning phase wherein the model improves its ability to follow complexmulti-modal instructions. Experiments demonstrate the versatility ofMM-Interleaved in recognizing visual details following multi-modal instructionsand generating consistent images following both textual and visual conditions.Code and models are available aturlhttps://github.com/OpenGVLab/MM-Interleaved. |


| Item |Content|
| --- |---|
|idx| 2401.10189v1 |
|title| Chem-FINESE: Validating Fine-Grained Few-shot Entity Extraction through Text Reconstruction |
|authors| Qingyun WangZixuan ZhangHongxiang LiXuan LiuJiawei HanHeng JiHuimin Zhao
|links| http://arxiv.org/abs/2401.10189v1 |
|updated| 2024-01-18 18:20:15 UTC |
|summary| Fine-grained few-shot entity extraction in the chemical domain faces twounique challenges. First compared with entity extraction tasks in the generaldomain sentences from chemical papers usually contain more entities. Moreoverentity extraction models usually have difficulty extracting entities oflong-tailed types. In this paper we propose Chem-FINESE a novelsequence-to-sequence seq2seq based few-shot entity extraction approach toaddress these two challenges. Our Chem-FINESE has two components: a seq2seqentity extractor to extract named entities from the input sentence and aseq2seq self-validation module to reconstruct the original input sentence fromextracted entities. Inspired by the fact that a good entity extraction systemneeds to extract entities faithfully our new self-validation module leveragesentity extraction results to reconstruct the original input sentence. Besideswe design a new contrastive loss to reduce excessive copying during theextraction process. Finally we release ChemNER a new fine-grained chemicalentity extraction dataset that is annotated by domain experts with the ChemNERschema. Experiments in few-shot settings with both ChemNER and CHEMET datasetsshow that our newly proposed framework has contributed up to 8.26 and 6.84absolute F1-score gains respectively. |


| Item |Content|
| --- |---|
|idx| 2401.10186v1 |
|title| Beyond Reference-Based Metrics: Analyzing Behaviors of Open LLMs on Data-to-Text Generation |
|authors| Zdeněk KasnerOndřej Dušek
|links| http://arxiv.org/abs/2401.10186v1 |
|updated| 2024-01-18 18:15:46 UTC |
|summary| We investigate to which extent open large language models LLMs can generatecoherent and relevant text from structured data. To prevent bias frombenchmarks leaked into LLM training data we collect Quintd-1: an ad-hocbenchmark for five data-to-text D2T generation tasks consisting ofstructured data records in standard formats gathered from public APIs. Weleverage reference-free evaluation metrics and LLMs in-context learningcapabilities allowing us to test the models with no human-written references.Our evaluation focuses on annotating semantic accuracy errors on token-levelcombining human annotators and a metric based on GPT-4. Our systematicexamination of the models behavior across domains and tasks suggests thatstate-of-the-art open LLMs with 7B parameters can generate fluent and coherenttext from various standard data formats in zero-shot settings. However we alsoshow that semantic accuracy of the outputs remains a major issue: on ourbenchmark 80 of outputs of open LLMs contain a semantic error according tohuman annotators 91 according to GPT-4. Our code data and model outputsare available at https://d2t-llm.github.io. |


| Item |Content|
| --- |---|
|idx| 2401.10134v1 |
|title| Spatial-Temporal Large Language Model for Traffic Prediction |
|authors| Chenxi LiuSun YangQianxiong XuZhishuai LiCheng LongZiyue LiRui Zhao
|links| http://arxiv.org/abs/2401.10134v1 |
|updated| 2024-01-18 17:03:59 UTC |
|summary| Traffic prediction a critical component for intelligent transportationsystems endeavors to foresee future traffic at specific locations usinghistorical data. Although existing traffic prediction models often emphasizedeveloping complex neural network structures their accuracy has not seenimprovements accordingly. Recently Large Language Models LLMs have shownoutstanding capabilities in time series analysis. Differing from existingmodels LLMs progress mainly through parameter expansion and extensivepre-training while maintaining their fundamental structures. In this paper wepropose a Spatial-Temporal Large Language Model ST-LLM for trafficprediction. Specifically ST-LLM redefines the timesteps at each location astokens and incorporates a spatial-temporal embedding module to learn thespatial location and global temporal representations of tokens. Then theserepresentations are fused to provide each token with unified spatial andtemporal information. Furthermore we propose a novel partially frozenattention strategy of the LLM which is designed to capture spatial-temporaldependencies for traffic prediction. Comprehensive experiments on real trafficdatasets offer evidence that ST-LLM outperforms state-of-the-art models.Notably the ST-LLM also exhibits robust performance in both few-shot andzero-shot prediction scenarios. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2401.10225v1 |
|title| ChatQA: Building GPT-4 Level Conversational QA Models |
|authors| Zihan LiuWei PingRajarshi RoyPeng XuMohammad ShoeybiBryan Catanzaro
|links| http://arxiv.org/abs/2401.10225v1 |
|updated| 2024-01-18 18:59:11 UTC |
|summary| In this work we introduce ChatQA a family of conversational questionanswering QA models that obtain GPT-4 level accuracies. Specifically wepropose a two-stage instruction tuning method that can significantly improvethe zero-shot conversational QA results from large language models LLMs. Tohandle retrieval in conversational QA we fine-tune a dense retriever on amulti-turn QA dataset which provides comparable results to using thestate-of-the-art query rewriting model while largely reducing deployment cost.Notably our ChatQA-70B can outperform GPT-4 in terms of average score on 10conversational QA datasets 54.14 vs. 53.90 without relying on any syntheticdata from OpenAI GPT models. |


| Item |Content|
| --- |---|
|idx| 2401.10222v1 |
|title| Supervised Fine-tuning in turn Improves Visual Foundation Models |
|authors| Xiaohu JiangYixiao GeYuying GeChun YuanYing Shan
|links| http://arxiv.org/abs/2401.10222v1 |
|updated| 2024-01-18 18:58:54 UTC |
|summary| Image-text training like CLIP has dominated the pretraining of visionfoundation models in recent years. Subsequent efforts have been made tointroduce region-level visual learning into CLIPs pretraining but facescalability challenges due to the lack of large-scale region-level datasets.Drawing inspiration from supervised fine-tuning SFT in natural languageprocessing such as instruction tuning we explore the potential of fine-grainedSFT in enhancing the generation of vision foundation models after theirpretraining. Thus a two-stage method ViSFT Vision SFT is proposed to unleashthe fine-grained knowledge of vision foundation models. In ViSFT the visionfoundation model is enhanced by performing visual joint learning on somein-domain tasks and then tested on out-of-domain benchmarks. With updatingusing ViSFT on 8 V100 GPUs in less than 2 days a vision transformer with over4.4B parameters shows improvements across various out-of-domain benchmarksincluding vision and vision-linguistic scenarios. |


| Item |Content|
| --- |---|
|idx| 2401.10207v1 |
|title| Eclectic Rule Extraction for Explainability of Deep Neural Network based Intrusion Detection Systems |
|authors| Jesse AblesNathaniel ChildersWilliam AndersonSudip MittalShahram RahimiIoana BanicescuMaria Seale
|links| http://arxiv.org/abs/2401.10207v1 |
|updated| 2024-01-18 18:45:29 UTC |
|summary| This paper addresses trust issues created from the ubiquity of black boxalgorithms and surrogate explainers in Explainable Intrusion Detection SystemsX-IDS. While Explainable Artificial Intelligence XAI aims to enhancetransparency black box surrogate explainers such as Local InterpretableModel-Agnostic Explanation LIME and SHapley Additive exPlanation SHAP aredifficult to trust. The black box nature of these surrogate explainers makesthe process behind explanation generation opaque and difficult to understand.To avoid this problem one can use transparent white box algorithms such asRule Extraction RE. There are three types of RE algorithms: pedagogicaldecompositional and eclectic. Pedagogical methods offer fast but untrustworthywhite-box explanations while decompositional RE provides trustworthyexplanations with poor scalability. This work explores eclectic ruleextraction which strikes a balance between scalability and trustworthiness. Bycombining techniques from pedagogical and decompositional approaches eclecticrule extraction leverages the advantages of both while mitigating some oftheir drawbacks. The proposed Hybrid X-IDS architecture features eclectic RE asa white box surrogate explainer for black box Deep Neural Networks DNN. Thepresented eclectic RE algorithm extracts human-readable rules from hiddenlayers facilitating explainable and trustworthy rulesets. Evaluations onUNSW-NB15 and CIC-IDS-2017 datasets demonstrate the algorithms ability togenerate rulesets with 99.9 accuracy mimicking DNN outputs. The contributionsof this work include the hybrid X-IDS architecture the eclectic ruleextraction algorithm applicable to intrusion detection datasets and a thoroughanalysis of performance and explainability demonstrating the trade-offsinvolved in rule extraction speed and accuracy. |


| Item |Content|
| --- |---|
|idx| 2401.10189v1 |
|title| Chem-FINESE: Validating Fine-Grained Few-shot Entity Extraction through Text Reconstruction |
|authors| Qingyun WangZixuan ZhangHongxiang LiXuan LiuJiawei HanHeng JiHuimin Zhao
|links| http://arxiv.org/abs/2401.10189v1 |
|updated| 2024-01-18 18:20:15 UTC |
|summary| Fine-grained few-shot entity extraction in the chemical domain faces twounique challenges. First compared with entity extraction tasks in the generaldomain sentences from chemical papers usually contain more entities. Moreoverentity extraction models usually have difficulty extracting entities oflong-tailed types. In this paper we propose Chem-FINESE a novelsequence-to-sequence seq2seq based few-shot entity extraction approach toaddress these two challenges. Our Chem-FINESE has two components: a seq2seqentity extractor to extract named entities from the input sentence and aseq2seq self-validation module to reconstruct the original input sentence fromextracted entities. Inspired by the fact that a good entity extraction systemneeds to extract entities faithfully our new self-validation module leveragesentity extraction results to reconstruct the original input sentence. Besideswe design a new contrastive loss to reduce excessive copying during theextraction process. Finally we release ChemNER a new fine-grained chemicalentity extraction dataset that is annotated by domain experts with the ChemNERschema. Experiments in few-shot settings with both ChemNER and CHEMET datasetsshow that our newly proposed framework has contributed up to 8.26 and 6.84absolute F1-score gains respectively. |


| Item |Content|
| --- |---|
|idx| 2401.10178v1 |
|title| Neural Echos: Depthwise Convolutional Filters Replicate Biological Receptive Fields |
|authors| Zahra BabaieePeyman M. KiasariDaniela RusRadu Grosu
|links| http://arxiv.org/abs/2401.10178v1 |
|updated| 2024-01-18 18:06:22 UTC |
|summary| In this study we present evidence suggesting that depthwise convolutionalkernels are effectively replicating the structural intricacies of thebiological receptive fields observed in the mammalian retina. We provideanalytics of trained kernels from various state-of-the-art modelssubstantiating this evidence. Inspired by this intriguing discovery we proposean initialization scheme that draws inspiration from the biological receptivefields. Experimental analysis of the ImageNet dataset with multiple CNNarchitectures featuring depthwise convolutions reveals a marked enhancement inthe accuracy of the learned model when initialized with biologically derivedweights. This underlies the potential for biologically inspired computationalmodels to further our understanding of vision processing systems and to improvethe efficacy of convolutional networks. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2401.10227v1 |
|title| A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting |
|authors| Wouter Van GansbekeBert De Brabandere
|links| http://arxiv.org/abs/2401.10227v1 |
|updated| 2024-01-18 18:59:19 UTC |
|summary| Panoptic and instance segmentation networks are often trained withspecialized object detection modules complex loss functions and ad-hocpost-processing steps to handle the permutation-invariance of the instancemasks. This work builds upon Stable Diffusion and proposes a latent diffusionapproach for panoptic segmentation resulting in a simple architecture whichomits these complexities. Our training process consists of two steps: 1training a shallow autoencoder to project the segmentation masks to latentspace 2 training a diffusion model to allow image-conditioned sampling inlatent space. The use of a generative model unlocks the exploration of maskcompletion or inpainting which has applications in interactive segmentation.The experimental validation yields promising results for both panopticsegmentation and mask inpainting. While not setting a new state-of-the-art ourmodels simplicity generality and mask completion capability are desirableproperties. |


| Item |Content|
| --- |---|
|idx| 2401.10225v1 |
|title| ChatQA: Building GPT-4 Level Conversational QA Models |
|authors| Zihan LiuWei PingRajarshi RoyPeng XuMohammad ShoeybiBryan Catanzaro
|links| http://arxiv.org/abs/2401.10225v1 |
|updated| 2024-01-18 18:59:11 UTC |
|summary| In this work we introduce ChatQA a family of conversational questionanswering QA models that obtain GPT-4 level accuracies. Specifically wepropose a two-stage instruction tuning method that can significantly improvethe zero-shot conversational QA results from large language models LLMs. Tohandle retrieval in conversational QA we fine-tune a dense retriever on amulti-turn QA dataset which provides comparable results to using thestate-of-the-art query rewriting model while largely reducing deployment cost.Notably our ChatQA-70B can outperform GPT-4 in terms of average score on 10conversational QA datasets 54.14 vs. 53.90 without relying on any syntheticdata from OpenAI GPT models. |


| Item |Content|
| --- |---|
|idx| 2401.10220v1 |
|title| AutoFT: Robust Fine-Tuning by Optimizing Hyperparameters on OOD Data |
|authors| Caroline ChoiYoonho LeeAnnie ChenAllan ZhouAditi RaghunathanChelsea Finn
|links| http://arxiv.org/abs/2401.10220v1 |
|updated| 2024-01-18 18:58:49 UTC |
|summary| Foundation models encode rich representations that can be adapted to adesired task by fine-tuning on task-specific data. However fine-tuning a modelon one particular data distribution often compromises the models originalperformance on other distributions. Current methods for robust fine-tuningutilize hand-crafted regularization techniques to constrain the fine-tuningprocess towards the base foundation model. Yet it is hard to precisely specifywhat characteristics of the foundation model to retain during fine-tuning asthis depends on how the pre-training fine-tuning and evaluation datadistributions relate to each other. We propose AutoFT a data-driven approachfor guiding foundation model fine-tuning. AutoFT optimizes fine-tuninghyperparameters to maximize performance on a small out-of-distribution OODvalidation set. To guide fine-tuning in a granular way AutoFT searches ahighly expressive hyperparameter space that includes weight coefficients formany different losses in addition to learning rate and weight decay values. Weevaluate AutoFT on nine natural distribution shifts which include domain shiftsand subpopulation shifts. Our experiments show that AutoFT significantlyimproves generalization to new OOD data outperforming existing robustfine-tuning methods. Notably AutoFT achieves new state-of-the-art performanceon the WILDS-iWildCam and WILDS-FMoW benchmarks outperforming the previousbest methods by 6.0 and 1.5 respectively. |


| Item |Content|
| --- |---|
|idx| 2401.10216v1 |
|title| Enabling Efficient Equivariant Operations in the Fourier Basis via Gaunt Tensor Products |
|authors| Shengjie LuoTianlang ChenAditi S. Krishnapriyan
|links| http://arxiv.org/abs/2401.10216v1 |
|updated| 2024-01-18 18:57:10 UTC |
|summary| Developing equivariant neural networks for the E3 group plays an importantrole in modeling 3D data across real-world applications. Enforcing thisequivariance primarily involves the tensor products of irreduciblerepresentations irreps. However the computational complexity of suchoperations increases significantly as higher-order tensors are used. In thiswork we propose a systematic approach to substantially accelerate thecomputation of the tensor products of irreps. We mathematically connect thecommonly used Clebsch-Gordan coefficients to the Gaunt coefficients which areintegrals of products of three spherical harmonics. Through Gaunt coefficientsthe tensor product of irreps becomes equivalent to the multiplication betweenspherical functions represented by spherical harmonics. This perspectivefurther allows us to change the basis for the equivariant operations fromspherical harmonics to a 2D Fourier basis. Consequently the multiplicationbetween spherical functions represented by a 2D Fourier basis can beefficiently computed via the convolution theorem and Fast Fourier Transforms.This transformation reduces the complexity of full tensor products of irrepsfrom mathcalOL6 to mathcalOL3 where L is the max degree ofirreps. Leveraging this approach we introduce the Gaunt Tensor Product whichserves as a new method to construct efficient equivariant operations acrossdifferent model architectures. Our experiments on the Open Catalyst Project and3BPA datasets demonstrate both the increased efficiency and improvedperformance of our approach. |


| Item |Content|
| --- |---|
|idx| 2401.10207v1 |
|title| Eclectic Rule Extraction for Explainability of Deep Neural Network based Intrusion Detection Systems |
|authors| Jesse AblesNathaniel ChildersWilliam AndersonSudip MittalShahram RahimiIoana BanicescuMaria Seale
|links| http://arxiv.org/abs/2401.10207v1 |
|updated| 2024-01-18 18:45:29 UTC |
|summary| This paper addresses trust issues created from the ubiquity of black boxalgorithms and surrogate explainers in Explainable Intrusion Detection SystemsX-IDS. While Explainable Artificial Intelligence XAI aims to enhancetransparency black box surrogate explainers such as Local InterpretableModel-Agnostic Explanation LIME and SHapley Additive exPlanation SHAP aredifficult to trust. The black box nature of these surrogate explainers makesthe process behind explanation generation opaque and difficult to understand.To avoid this problem one can use transparent white box algorithms such asRule Extraction RE. There are three types of RE algorithms: pedagogicaldecompositional and eclectic. Pedagogical methods offer fast but untrustworthywhite-box explanations while decompositional RE provides trustworthyexplanations with poor scalability. This work explores eclectic ruleextraction which strikes a balance between scalability and trustworthiness. Bycombining techniques from pedagogical and decompositional approaches eclecticrule extraction leverages the advantages of both while mitigating some oftheir drawbacks. The proposed Hybrid X-IDS architecture features eclectic RE asa white box surrogate explainer for black box Deep Neural Networks DNN. Thepresented eclectic RE algorithm extracts human-readable rules from hiddenlayers facilitating explainable and trustworthy rulesets. Evaluations onUNSW-NB15 and CIC-IDS-2017 datasets demonstrate the algorithms ability togenerate rulesets with 99.9 accuracy mimicking DNN outputs. The contributionsof this work include the hybrid X-IDS architecture the eclectic ruleextraction algorithm applicable to intrusion detection datasets and a thoroughanalysis of performance and explainability demonstrating the trade-offsinvolved in rule extraction speed and accuracy. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2401.10232v1 |
|title| ParaHome: Parameterizing Everyday Home Activities Towards 3D Generative Modeling of Human-Object Interactions |
|authors| Jeonghwan KimJisoo KimJeonghyeon NaHanbyul Joo
|links| http://arxiv.org/abs/2401.10232v1 |
|updated| 2024-01-18 18:59:58 UTC |
|summary| To enable machines to learn how humans interact with the physical world inour daily activities it is crucial to provide rich data that encompasses the3D motion of humans as well as the motion of objects in a learnable 3Drepresentation. Ideally this data should be collected in a natural setupcapturing the authentic dynamic 3D signals during human-object interactions. Toaddress this challenge we introduce the ParaHome system designed to captureand parameterize dynamic 3D movements of humans and objects within a commonhome environment. Our system consists of a multi-view setup with 70synchronized RGB cameras as well as wearable motion capture devices equippedwith an IMU-based body suit and hand motion capture gloves. By leveraging theParaHome system we collect a novel large-scale dataset of human-objectinteraction. Notably our dataset offers key advancement over existing datasetsin three main aspects: 1 capturing 3D body and dexterous hand manipulationmotion alongside 3D object movement within a contextual home environment duringnatural activities 2 encompassing human interaction with multiple objects invarious episodic scenarios with corresponding descriptions in texts 3including articulated objects with multiple parts expressed with parameterizedarticulations. Building upon our dataset we introduce new research tasks aimedat building a generative model for learning and synthesizing human-objectinteractions in a real-world room setting. |


| Item |Content|
| --- |---|
|idx| 2401.10229v1 |
|title| OMG-Seg: Is One Model Good Enough For All Segmentation? |
|authors| Xiangtai LiHaobo YuanWei LiHenghui DingSize WuWenwei ZhangYining LiKai ChenChen Change Loy
|links| http://arxiv.org/abs/2401.10229v1 |
|updated| 2024-01-18 18:59:34 UTC |
|summary| In this work we address various segmentation tasks each traditionallytackled by distinct or partially unified models. We propose OMG-Seg One Modelthat is Good enough to efficiently and effectively handle all the segmentationtasks including image semantic instance and panoptic segmentation as wellas their video counterparts open vocabulary settings prompt-driveninteractive segmentation like SAM and video object segmentation. To ourknowledge this is the first model to handle all these tasks in one model andachieve satisfactory performance. We show that OMG-Seg a transformer-basedencoder-decoder architecture with task-specific queries and outputs cansupport over ten distinct segmentation tasks and yet significantly reducecomputational and parameter overhead across various tasks and datasets. Werigorously evaluate the inter-task influences and correlations duringco-training. Code and models are available at https://github.com/lxtGH/OMG-Seg. |


| Item |Content|
| --- |---|
|idx| 2401.10228v1 |
|title| RAP-SAM: Towards Real-Time All-Purpose Segment Anything |
|authors| Shilin XuHaobo YuanQingyu ShiLu QiJingbo WangYibo YangYining LiKai ChenYunhai TongBernard GhanemXiangtai LiMing-Hsuan Yang
|links| http://arxiv.org/abs/2401.10228v1 |
|updated| 2024-01-18 18:59:30 UTC |
|summary| Advanced by transformer architecture vision foundation models VFMs achieveremarkable progress in performance and generalization ability. Segment AnythingModel SAM is one remarkable model that can achieve generalized segmentation.However most VFMs cannot run in realtime which makes it difficult to transferthem into several products. On the other hand current real-time segmentationmainly has one purpose such as semantic segmentation on the driving scene. Weargue that diverse outputs are needed for real applications. Thus this workexplores a new real-time segmentation setting named all-purpose segmentationin real-time to transfer VFMs in real-time deployment. It contains threedifferent tasks including interactive segmentation panoptic segmentation andvideo segmentation. We aim to use one model to achieve the above tasks inreal-time. We first benchmark several strong baselines. Then we presentReal-Time All Purpose SAM RAP-SAM. It contains an efficient encoder and anefficient decoupled decoder to perform prompt-driven decoding. Moreover wefurther explore different training strategies and tuning methods to boostco-training performance further. Our code and model are available athttps://github.com/xushilin1/RAP-SAM/. |


| Item |Content|
| --- |---|
|idx| 2401.10227v1 |
|title| A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting |
|authors| Wouter Van GansbekeBert De Brabandere
|links| http://arxiv.org/abs/2401.10227v1 |
|updated| 2024-01-18 18:59:19 UTC |
|summary| Panoptic and instance segmentation networks are often trained withspecialized object detection modules complex loss functions and ad-hocpost-processing steps to handle the permutation-invariance of the instancemasks. This work builds upon Stable Diffusion and proposes a latent diffusionapproach for panoptic segmentation resulting in a simple architecture whichomits these complexities. Our training process consists of two steps: 1training a shallow autoencoder to project the segmentation masks to latentspace 2 training a diffusion model to allow image-conditioned sampling inlatent space. The use of a generative model unlocks the exploration of maskcompletion or inpainting which has applications in interactive segmentation.The experimental validation yields promising results for both panopticsegmentation and mask inpainting. While not setting a new state-of-the-art ourmodels simplicity generality and mask completion capability are desirableproperties. |


| Item |Content|
| --- |---|
|idx| 2401.10226v1 |
|title| Towards Language-Driven Video Inpainting via Multimodal Large Language Models |
|authors| Jianzong WuXiangtai LiChenyang SiShangchen ZhouJingkang YangJiangning ZhangYining LiKai ChenYunhai TongZiwei LiuChen Change Loy
|links| http://arxiv.org/abs/2401.10226v1 |
|updated| 2024-01-18 18:59:13 UTC |
|summary| We introduce a new task -- language-driven video inpainting which usesnatural language instructions to guide the inpainting process. This approachovercomes the limitations of traditional video inpainting methods that dependon manually labeled binary masks a process often tedious and labor-intensive.We present the Remove Objects from Videos by Instructions ROVI datasetcontaining 5650 videos and 9091 inpainting results to support training andevaluation for this task. We also propose a novel diffusion-basedlanguage-driven video inpainting framework the first end-to-end baseline forthis task integrating Multimodal Large Language Models to understand andexecute complex language-based inpainting requests effectively. Ourcomprehensive results showcase the datasets versatility and the modelseffectiveness in various language-instructed inpainting scenarios. We will makedatasets code and models publicly available. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2401.10204v1 |
|title| Maximal-Capacity Discrete Memoryless Channel Identification |
|authors| Maximilian EggerRawad BitarAntonia Wachter-ZehDeniz GündüzNir Weinberger
|links| http://arxiv.org/abs/2401.10204v1 |
|updated| 2024-01-18 18:44:10 UTC |
|summary| The problem of identifying the channel with the highest capacity amongseveral discrete memoryless channels DMCs is considered. The problem is castas a pure-exploration multi-armed bandit problem which follows the practicaluse of training sequences to sense the communication channel statistics. Acapacity estimator is proposed and tight confidence bounds on the estimatorerror are derived. Based on this capacity estimator a gap-eliminationalgorithm termed BestChanID is proposed which is oblivious to thecapacity-achieving input distribution and is guaranteed to output the DMC withthe largest capacity with a desired confidence. Furthermore two additionalalgorithms NaiveChanSel and MedianChanEl that output with certain confidence aDMC with capacity close to the maximal are introduced. Each of thosealgorithms is beneficial in a different regime and can be used as a subroutinein BestChanID. The sample complexity of all algorithms is analyzed as afunction of the desired confidence parameter the number of channels and thechannels input and output alphabet sizes. The cost of best channelidentification is shown to scale quadratically with the alphabet size and afundamental lower bound for the required number of channel senses to identifythe best channel with a certain confidence is derived. |


| Item |Content|
| --- |---|
|idx| 2401.09979v1 |
|title| False Discovery Rate Control for Gaussian Graphical Models via Neighborhood Screening |
|authors| Taulant KokaJasin MachkourMichael Muma
|links| http://arxiv.org/abs/2401.09979v1 |
|updated| 2024-01-18 13:46:41 UTC |
|summary| Gaussian graphical models emerge in a wide range of fields. They model thestatistical relationships between variables as a graph where an edge betweentwo variables indicates conditional dependence. Unfortunately well-establishedestimators such as the graphical lasso or neighborhood selection are known tobe susceptible to a high prevalence of false edge detections. False detectionsmay encourage inaccurate or even incorrect scientific interpretations withmajor implications in applications such as biomedicine or healthcare. In thispaper we introduce a nodewise variable selection approach to graph learningand provably control the false discovery rate of the selected edge set at aself-estimated level. A novel fusion method of the individual neighborhoodsoutputs an undirected graph estimate. The proposed method is parameter-free anddoes not require tuning by the user. Benchmarks against competing falsediscovery rate controlling methods in numerical experiments consideringdifferent graph topologies show a significant gain in performance. |


| Item |Content|
| --- |---|
|idx| 2401.09840v1 |
|title| FREED++: Improving RL Agents for Fragment-Based Molecule Generation by Thorough Reproduction |
|authors| Alexander TelepovArtem TsypinKuzma KhrabrovSergey YakukhnovPavel StrashnovPetr ZhilyaevEgor RumiantsevDaniel EzhovManvel AvetisianOlga PopovaArtur Kadurin
|links| http://arxiv.org/abs/2401.09840v1 |
|updated| 2024-01-18 09:54:19 UTC |
|summary| A rational design of new therapeutic drugs aims to find a molecular structurewith desired biological functionality e.g. an ability to activate or suppressa specific protein via binding to it. Molecular docking is a common techniquefor evaluating protein-molecule interactions. Recently Reinforcement LearningRL has emerged as a promising approach to generating molecules with thedocking score DS as a reward. In this work we reproduce scrutinize andimprove the recent RL model for molecule generation called FREEDarXiv:2110.01219. Extensive evaluation of the proposed method reveals severallimitations and challenges despite the outstanding results reported for threetarget proteins. Our contributions include fixing numerous implementation bugsand simplifying the model while increasing its quality significantly extendingexperiments and conducting an accurate comparison with currentstate-of-the-art methods for protein-conditioned molecule generation. We showthat the resulting fixed model is capable of producing molecules with superiordocking scores compared to alternative approaches. |


| Item |Content|
| --- |---|
|idx| 2401.09787v1 |
|title| Querying Easily Flip-flopped Samples for Deep Active Learning |
|authors| Seong Jin ChoGwangsu KimJunghyun LeeJinwoo ShinChang D. Yoo
|links| http://arxiv.org/abs/2401.09787v1 |
|updated| 2024-01-18 08:12:23 UTC |
|summary| Active learning is a machine learning paradigm that aims to improve theperformance of a model by strategically selecting and querying unlabeled data.One effective selection strategy is to base it on the models predictiveuncertainty which can be interpreted as a measure of how informative a sampleis. The samples distance to the decision boundary is a natural measure ofpredictive uncertainty but it is often intractable to compute especially forcomplex decision boundaries formed in multiclass classification tasks. Toaddress this issue this paper proposes the it least disagree metric LDMdefined as the smallest probability of disagreement of the predicted label andan estimator for LDM proven to be asymptotically consistent under mildassumptions. The estimator is computationally efficient and can be easilyimplemented for deep learning models using parameter perturbation. TheLDM-based active learning is performed by querying unlabeled data with thesmallest LDM. Experimental results show that our LDM-based active learningalgorithm obtains state-of-the-art overall performance on all considereddatasets and deep architectures. |


| Item |Content|
| --- |---|
|idx| 2401.09681v1 |
|title| Harnessing Density Ratios for Online Reinforcement Learning |
|authors| Philip AmortilaDylan J. FosterNan JiangAyush SekhariTengyang Xie
|links| http://arxiv.org/abs/2401.09681v1 |
|updated| 2024-01-18 02:21:06 UTC |
|summary| The theories of offline and online reinforcement learning despite havingevolved in parallel have begun to show signs of the possibility for aunification with algorithms and analysis techniques for one setting oftenhaving natural counterparts in the other. However the notion of density ratiomodeling an emerging paradigm in offline RL has been largely absent fromonline RL perhaps for good reason: the very existence and boundedness ofdensity ratios relies on access to an exploratory dataset with good coveragebut the core challenge in online RL is to collect such a dataset without havingone to start. In this work we show -- perhaps surprisingly -- that densityratio-based algorithms have online counterparts. Assuming only the existence ofan exploratory distribution with good coverage a structural condition known ascoverability Xie et al. 2023 we give a new algorithm GLOW that usesdensity ratio realizability and value function realizability to performsample-efficient online exploration. GLOW addresses unbounded density ratiosvia careful use of truncation and combines this with optimism to guideexploration. GLOW is computationally inefficient we complement it with a moreefficient counterpart HyGLOW for the Hybrid RL setting Song et al. 2022wherein online RL is augmented with additional offline data. HyGLOW is derivedas a special case of a more general meta-algorithm that provides a provableblack-box reduction from hybrid RL to offline RL which may be of independentinterest. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2401.10184v1 |
|title| Comparing Traditional and LLM-based Search for Image Geolocation |
|authors| Albatool WazzanStephen MacNeilRichard Souvenir
|links| http://dx.doi.org/10.1145/3627508.3638305 |
|updated| 2024-01-18 18:12:28 UTC |
|summary| Web search engines have long served as indispensable tools for informationretrieval user behavior and query formulation strategies have been wellstudied. The introduction of search engines powered by large language modelsLLMs suggested more conversational search and new types of query strategies.In this paper we compare traditional and LLM-based search for the task ofimage geolocation i.e. determining the location where an image was captured.Our work examines user interactions with a particular focus on queryformulation strategies. In our study 60 participants were assigned eithertraditional or LLM-based search engines as assistants for geolocation.Participants using traditional search more accurately predicted the location ofthe image compared to those using the LLM-based search. Distinct strategiesemerged between users depending on the type of assistant. Participants usingthe LLM-based search issued longer more natural language queries but hadshorter search sessions. When reformulating their search queries traditionalsearch participants tended to add more terms to their initial queries whereasparticipants using the LLM-based search consistently rephrased their initialqueries. |


| Item |Content|
| --- |---|
|idx| 2401.10175v1 |
|title| DualTake: Predicting Takeovers across Mobilities for Future Personalized Mobility Services |
|authors| Zhaobo ZhengKumar AkashTeruhisa Misu
|links| http://dx.doi.org/10.1145/3610978.3640610 |
|updated| 2024-01-18 18:03:07 UTC |
|summary| A hybrid society is expected to emerge in the near future with differentmobilities interacting together including cars micro-mobilities pedestriansand robots. People may utilize multiple types of mobilities in their dailylives. As vehicle automation advances driver modeling flourishes to providepersonalized intelligent services. Thus modeling drivers across mobilitieswould pave the road for future society mobility-as-a-service and it isparticularly interesting to predict driver behaviors in newer mobilities withtraditional mobility data. In this work we present takeover prediction on amicro-mobility with car simulation data.The promising model performancedemonstrates the feasibility of driver modeling across mobilities as the firstin the field. |


| Item |Content|
| --- |---|
|idx| 2401.09937v1 |
|title| From Cash to Cashless: UPI's Impact on Spending Behavior among Indian Users |
|authors| Harshal DevRaj GuptaDhruv Kumar
|links| http://arxiv.org/abs/2401.09937v1 |
|updated| 2024-01-18 12:39:53 UTC |
|summary| The emergence of digital payment systems has transformed how individualsconduct financial transactions offering convenience security and efficiency.One groundbreaking innovation making waves in the Indian financial landscape isthe Unified Payments Interface UPI developed by the National PaymentsCorporation of India NPCI. Existing work has explored how digital paymentsbenefit a countrys economy and GDP. However our study explores how theintroduction of UPI has influenced spending behavior among Indian users on anindividual level. We gathered 235 valid survey responses encompassing diversedemographics and conducted semi-structured interviews with 20 surveyrespondents. Approximately 75 of the survey respondents reported increasedspending due to UPI with only 7 indicating reduced spending. Significantly91.5 of the respondents reported satisfaction with their UPI usage. Also 95.2of the survey respondents found making payments via UPI convenient. Ourresearch also provides suggestions for UPI applications and variousstakeholders to enhance digital payment systems enabling users to makeinformed decisions and fostering responsible financial management. |


| Item |Content|
| --- |---|
|idx| 2401.09896v1 |
|title| Experimental Shake Gesture Detection API for Apple Watch |
|authors| Ezequiel França dos Santos
|links| http://arxiv.org/abs/2401.09896v1 |
|updated| 2024-01-18 11:20:39 UTC |
|summary| In this paper we present the WatchShaker project The project involves anexperimental API that detects the Apple Watchs shake gesturea surprisinglyabsent natively feature Through a simple heuristic leveraging the Apple Watchsaccelerometer data the API discerns not just the occurrence of shake gesturesbut also their direction enhancing the interactivity potential of the deviceDespite the projects simplicity and lack of formal testing it has garneredsignificant attention indicating a genuine interest and need within thedeveloper community for such functionality The WatchShaker project exemplifieshow a minimalistic approach can yield a practical and impactful tool inwearable technology providing a springboard for further research anddevelopment in intuitive gesture recognition |


| Item |Content|
| --- |---|
|idx| 2401.09828v1 |
|title| Enhanced Automated Quality Assessment Network for Interactive Building Segmentation in High-Resolution Remote Sensing Imagery |
|authors| Zhili ZhangXiangyun HuJiabo Xu
|links| http://arxiv.org/abs/2401.09828v1 |
|updated| 2024-01-18 09:42:47 UTC |
|summary| In this research we introduce the enhanced automated quality assessmentnetwork IBS-AQSNet an innovative solution for assessing the quality ofinteractive building segmentation within high-resolution remote sensingimagery. This is a new challenge in segmentation quality assessment and ourproposed IBS-AQSNet allievate this by identifying missed and mistaken segmentareas. First of all to acquire robust image features our method combines arobust pre-trained backbone with a lightweight counterpart for comprehensivefeature extraction from imagery and segmentation results. These features arethen fused through a simple combination of concatenation convolution layersand residual connections. Additionally ISR-AQSNet incorporates a multi-scaledifferential quality assessment decoder proficient in pinpointing areas wheresegmentation result is either missed or mistaken. Experiments on a newly-builtEVLab-BGZ dataset which includes over 39198 buildings demonstrate thesuperiority of the proposed method in automating segmentation qualityassessment thereby setting a new benchmark in the field. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2401.10149v1 |
|title| Multi-Agent Reinforcement Learning for Maritime Operational Technology Cyber Security |
|authors| Alec WilsonRyan MenziesNeela MorarjiDavid FosterMarco Casassa MontEsin TurkbeylerLisa Gralewski
|links| http://arxiv.org/abs/2401.10149v1 |
|updated| 2024-01-18 17:22:22 UTC |
|summary| This paper demonstrates the potential for autonomous cyber defence to beapplied on industrial control systems and provides a baseline environment tofurther explore Multi-Agent Reinforcement Learnings MARL application to thisproblem domain. It introduces a simulation environment IPMSRL of a genericIntegrated Platform Management System IPMS and explores the use of MARL forautonomous cyber defence decision-making on generic maritime based IPMSOperational Technology OT. OT cyber defensive actions are less mature thanthey are for Enterprise IT. This is due to the relatively brittle nature of OTinfrastructure originating from the use of legacy systems design-timeengineering assumptions and lack of full-scale modern security controls. Thereare many obstacles to be tackled across the cyber landscape due to continuallyincreasing cyber-attack sophistication and the limitations of traditionalIT-centric cyber defence solutions. Traditional IT controls are rarely deployedon OT infrastructure and where they are some threats arent fully addressed.In our experiments a shared critic implementation of Multi Agent ProximalPolicy Optimisation MAPPO outperformed Independent Proximal PolicyOptimisation IPPO. MAPPO reached an optimal policy episode outcome mean of1 after 800K timesteps whereas IPPO was only able to reach an episode outcomemean of 0.966 after one million timesteps. Hyperparameter tuning greatlyimproved training performance. Across one million timesteps the tunedhyperparameters reached an optimal policy whereas the default hyperparametersonly managed to win sporadically with most simulations resulting in a draw. Wetested a real-world constraint attack detection alert success and found thatwhen alert success probability is reduced to 0.75 or 0.9 the MARL defenderswere still able to win in over 97.5 or 99.5 of episodes respectively. |


| Item |Content|
| --- |---|
|idx| 2401.09666v1 |
|title| Traffic Smoothing Controllers for Autonomous Vehicles Using Deep Reinforcement Learning and Real-World Trajectory Data |
|authors| Nathan LichtléKathy JangAdit ShahEugene VinitskyJonathan W. LeeAlexandre M. Bayen
|links| http://arxiv.org/abs/2401.09666v1 |
|updated| 2024-01-18 00:50:41 UTC |
|summary| Designing traffic-smoothing cruise controllers that can be deployed ontoautonomous vehicles is a key step towards improving traffic flow reducingcongestion and enhancing fuel efficiency in mixed autonomy traffic. We bypassthe common issue of having to carefully fine-tune a large trafficmicrosimulator by leveraging real-world trajectory data from the I-24 highwayin Tennessee replayed in a one-lane simulation. Using standard deepreinforcement learning methods we train energy-reducing wave-smoothingpolicies. As an input to the agent we observe the speed and distance of onlythe vehicle in front which are local states readily available on most recentvehicles as well as non-local observations about the downstream state of thetraffic. We show that at a low 4 autonomous vehicle penetration rate weachieve significant fuel savings of over 15 on trajectories exhibiting manystop-and-go waves. Finally we analyze the smoothing effect of the controllersand demonstrate robustness to adding lane-changing into the simulation as wellas the removal of downstream information. |


| Item |Content|
| --- |---|
|idx| 2401.09032v1 |
|title| Improved Consensus ADMM for Cooperative Motion Planning of Large-Scale Connected Autonomous Vehicles with Limited Communication |
|authors| Haichao LiuZhenmin HuangZicheng ZhuYulin LiShaojie ShenJun Ma
|links| http://arxiv.org/abs/2401.09032v1 |
|updated| 2024-01-17 07:58:48 UTC |
|summary| This paper investigates a cooperative motion planning problem for large-scaleconnected autonomous vehicles CAVs under limited communications whichaddresses the challenges of high communication and computing resourcerequirements. Our proposed methodology incorporates a parallel optimizationalgorithm with improved consensus ADMM considering a more realistic locallyconnected topology network and time complexity of ON is achieved byexploiting the sparsity in the dual update process. To further enhance thecomputational efficiency we employ a lightweight evolution strategy for thedynamic connectivity graph of CAVs and each sub-problem split from theconsensus ADMM only requires managing a small group of CAVs. The proposedmethod implemented with the receding horizon scheme is validated thoroughlyand comparisons with existing numerical solvers and approaches demonstrate theefficiency of our proposed algorithm. Also simulations on large-scalecooperative driving tasks involving 80 vehicles are performed in thehigh-fidelity CARLA simulator which highlights the remarkable computationalefficiency scalability and effectiveness of our proposed development.Demonstration videos are available athttps://henryhcliu.github.io/icadmm_cmp_carla. |


| Item |Content|
| --- |---|
|idx| 2401.09014v1 |
|title| Data assimilation approach for addressing imperfections in people flow measurement techniques using particle filter |
|authors| Ryo MurataKenji Tanaka
|links| http://arxiv.org/abs/2401.09014v1 |
|updated| 2024-01-17 07:20:15 UTC |
|summary| Understanding and predicting people flow in urban areas is useful fordecision-making in urban planning and marketing strategies. Traditional methodsfor understanding people flow can be divided into measurement-based approachesand simulation-based approaches. Measurement-based approaches have theadvantage of directly capturing actual people flow but they face the challengeof data imperfection. On the other hand simulations can obtain complete dataon a computer but they only consider some of the factors determining humanbehavior leading to a divergence from actual people flow. Both measurement andsimulation methods have unresolved issues and combining the two cancomplementarily overcome them. This paper proposes a method that applies dataassimilation a fusion technique of measurement and simulation to agent-basedsimulation. Data assimilation combines the advantages of both measurement andsimulation contributing to the creation of an environment that can reflectreal people flow while acquiring richer data. The paper verifies theeffectiveness of the proposed method in a virtual environment and demonstratesthe potential of data assimilation to compensate for the three types ofimperfection in people flow measurement techniques. These findings can serve asguidelines for supplementing sparse measurement data in physical environments. |


| Item |Content|
| --- |---|
|idx| 2401.08728v1 |
|title| AgentMixer: Multi-Agent Correlated Policy Factorization |
|authors| Zhiyuan LiWenshuai ZhaoLijun WuJoni Pajarinen
|links| http://arxiv.org/abs/2401.08728v1 |
|updated| 2024-01-16 15:32:41 UTC |
|summary| Centralized training with decentralized execution CTDE is widely employedto stabilize partially observable multi-agent reinforcement learning MARL byutilizing a centralized value function during training. However existingmethods typically assume that agents make decisions based on their localobservations independently which may not lead to a correlated joint policywith sufficient coordination. Inspired by the concept of correlatedequilibrium we propose to introduce a textitstrategy modification toprovide a mechanism for agents to correlate their policies. Specifically wepresent a novel framework AgentMixer which constructs the joint fullyobservable policy as a non-linear combination of individual partiallyobservable policies. To enable decentralized execution one can deriveindividual policies by imitating the joint policy. Unfortunately suchimitation learning can lead to textitasymmetric learning failure caused bythe mismatch between joint policy and individual policy information. Tomitigate this issue we jointly train the joint policy and individual policiesand introduce textitIndividual-Global-Consistency to guarantee modeconsistency between the centralized and decentralized policies. We thentheoretically prove that AgentMixer converges to an epsilon-approximateCorrelated Equilibrium. The strong experimental performance on three MARLbenchmarks demonstrates the effectiveness of our method. |


