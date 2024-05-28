# cs.CL 

| Item |Content|
| --- |---|
|idx| 2405.17430v1 |
|title| Matryoshka Multimodal Models |
|authors| Mu CaiJianwei YangJianfeng GaoYong Jae Lee
|links| http://arxiv.org/abs/2405.17430v1 |
|updated| 2024-05-27 17:59:56 UTC |
|summary| Large Multimodal Models LMMs such as LLaVA have shown strong performance invisual-linguistic reasoning. These models first embed images into a fixed largenumber of visual tokens and then feed them into a Large Language Model LLM.However this design causes an excessive number of tokens for dense visualscenarios such as high-resolution images and videos leading to greatinefficiency. While token pruning/merging methods do exist they produce asingle length output for each image and do not afford flexibility in tradingoff information density v.s. efficiency. Inspired by the concept of MatryoshkaDolls we propose M3: Matryoshka Multimodal Models which learns to representvisual content as nested sets of visual tokens that capture information acrossmultiple coarse-to-fine granularities. Our approach offers several uniquebenefits for LMMs: 1 One can explicitly control the visual granularity pertest instance during inference e.g.  adjusting the number of tokens used torepresent an image based on the anticipated complexity or simplicity of thecontent 2 M3 provides a framework for analyzing the granularity needed forexisting datasets where we find that COCO-style benchmarks only need around 9visual tokens to obtain accuracy similar to that of using all 576 tokens 3Our approach provides a foundation to explore the best trade-off betweenperformance and visual token length at sample level where our investigationreveals that a large gap exists between the oracle upper bound and currentfixed-scale representations. |


| Item |Content|
| --- |---|
|idx| 2405.17428v1 |
|title| NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models |
|authors| Chankyu LeeRajarshi RoyMengyao XuJonathan RaimanMohammad ShoeybiBryan CatanzaroWei Ping
|links| http://arxiv.org/abs/2405.17428v1 |
|updated| 2024-05-27 17:59:45 UTC |
|summary| Decoder-only large language model LLM-based embedding models are beginningto outperform BERT or T5-based embedding models in general-purpose textembedding tasks including dense vector-based retrieval. In this work weintroduce the NV-Embed model with a variety of architectural designs andtraining procedures to significantly enhance the performance of LLM as aversatile embedding model while maintaining its simplicity andreproducibility. For model architecture we propose a latent attention layer toobtain pooled embeddings which consistently improves retrieval and downstreamtask accuracy compared to mean pooling or using the last EOS token embeddingfrom LLMs. To enhance representation learning we remove the causal attentionmask of LLMs during contrastive training. For model training we introduce atwo-stage contrastive instruction-tuning method. It first applies contrastivetraining with instructions on retrieval datasets utilizing in-batch negativesand curated hard negative examples. At stage-2 it blends various non-retrievaldatasets into instruction tuning which not only enhances non-retrieval taskaccuracy but also improves retrieval performance. Combining these techniquesour NV-Embed model using only publicly available data has achieved arecord-high score of 69.32 ranking No. 1 on the Massive Text EmbeddingBenchmark MTEB as of May 24 2024 with 56 tasks encompassing retrievalreranking classification clustering and semantic textual similarity tasks.Notably our model also attains the highest score of 59.36 on 15 retrievaltasks in the MTEB benchmark also known as BEIR. We will open-source the modelat: https://huggingface.co/nvidia/NV-Embed-v1. |


| Item |Content|
| --- |---|
|idx| 2405.17423v1 |
|title| Privacy-Aware Visual Language Models |
|authors| Laurens SamsonNimrod BarazaniSennay GhebreabYuki M. Asano
|links| http://arxiv.org/abs/2405.17423v1 |
|updated| 2024-05-27 17:59:25 UTC |
|summary| This paper aims to advance our understanding of how Visual Language ModelsVLMs handle privacy-sensitive information a crucial concern as thesetechnologies become integral to everyday life. To this end we introduce a newbenchmark PrivBench which contains images from 8 sensitive categories such aspassports or fingerprints. We evaluate 10 state-of-the-art VLMs on thisbenchmark and observe a generally limited understanding of privacyhighlighting a significant area for model improvement. Based on this weintroduce PrivTune a new instruction-tuning dataset aimed at equipping VLMswith knowledge about visual privacy. By tuning two pretrained VLMs TinyLLaVaand MiniGPT-v2 on this small dataset we achieve strong gains in their abilityto recognize sensitive content outperforming even GPT4-V. At the same time weshow that privacy-tuning only minimally affects the VLMs performance onstandard benchmarks such as VQA. Overall this paper lays out a crucialchallenge for making VLMs effective in handling real-world data safely andprovides a simple recipe that takes the first step towards buildingprivacy-aware VLMs. |


| Item |Content|
| --- |---|
|idx| 2405.17402v1 |
|title| THREAD: Thinking Deeper with Recursive Spawning |
|authors| Philip SchroederNathaniel MorganHongyin LuoJames Glass
|links| http://arxiv.org/abs/2405.17402v1 |
|updated| 2024-05-27 17:51:24 UTC |
|summary| Large language models LLMs have shown impressive capabilities acrossdiverse settings but still struggle as the length and complexity of thecontext increases. To address this challenge we propose Thinking Recursivelyand Dynamically ThReaD. THREAD frames model generation as a thread ofexecution that based on the context can run to completion or dynamicallyspawn new threads. By spawning threads can offload work e.g. thinkingretrieving information to child threads which only return tokens needed forthe parent thread to do its work. In effect this enables the model to adaptas needed the amount of intermediate work used to produce tokens. We applyTHREAD in the settings of LLM task solving and question answering where thedynamic threading allows the model to recursively decompose the given task orquestion into progressively simpler sub-problems that can be solved by separatechild threads. We test THREAD implemented using a few-shot learning approachon diverse benchmarks for agent tasks and data-grounded question answering.THREAD achieves state-of-the-art performance with GPT-4 and GPT-3.5 on thesebenchmarks including ALFWorld TextCraft and WebShop along with two newbenchmarks DataCommons QA and MIMIC-III ICU QA. In addition THREADoutperforms existing frameworks by 10 to 50 absolute points with smallermodels including Llama-3-8b and CodeLlama-7b. |


| Item |Content|
| --- |---|
|idx| 2405.17394v1 |
|title| The Expressive Capacity of State Space Models: A Formal Language Perspective |
|authors| Yash SarrofYana VeitsmanMichael Hahn
|links| http://arxiv.org/abs/2405.17394v1 |
|updated| 2024-05-27 17:46:57 UTC |
|summary| Recently recurrent models based on linear state space models SSMs haveshown promising performance in language modeling LM competititve withtransformers. However there is little understanding of the in-principleabilities of such models which could provide useful guidance to the search forbetter LM architectures. We present a comprehensive theoretical study of thecapacity of such SSMs as it compares to that of transformers and traditionalRNNs. We find that SSMs and transformers have overlapping but distinctstrengths. In star-free state tracking SSMs implement straightforward andexact solutions to problems that transformers struggle to represent exactly.They can also model bounded hierarchical structure with optimal memory evenwithout simulating a stack. On the other hand we identify a design choice incurrent SSMs that limits their expressive power. We discuss implications forSSM and LM research and verify results empirically on a recent SSM Mamba. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2405.17430v1 |
|title| Matryoshka Multimodal Models |
|authors| Mu CaiJianwei YangJianfeng GaoYong Jae Lee
|links| http://arxiv.org/abs/2405.17430v1 |
|updated| 2024-05-27 17:59:56 UTC |
|summary| Large Multimodal Models LMMs such as LLaVA have shown strong performance invisual-linguistic reasoning. These models first embed images into a fixed largenumber of visual tokens and then feed them into a Large Language Model LLM.However this design causes an excessive number of tokens for dense visualscenarios such as high-resolution images and videos leading to greatinefficiency. While token pruning/merging methods do exist they produce asingle length output for each image and do not afford flexibility in tradingoff information density v.s. efficiency. Inspired by the concept of MatryoshkaDolls we propose M3: Matryoshka Multimodal Models which learns to representvisual content as nested sets of visual tokens that capture information acrossmultiple coarse-to-fine granularities. Our approach offers several uniquebenefits for LMMs: 1 One can explicitly control the visual granularity pertest instance during inference e.g.  adjusting the number of tokens used torepresent an image based on the anticipated complexity or simplicity of thecontent 2 M3 provides a framework for analyzing the granularity needed forexisting datasets where we find that COCO-style benchmarks only need around 9visual tokens to obtain accuracy similar to that of using all 576 tokens 3Our approach provides a foundation to explore the best trade-off betweenperformance and visual token length at sample level where our investigationreveals that a large gap exists between the oracle upper bound and currentfixed-scale representations. |


| Item |Content|
| --- |---|
|idx| 2405.17429v1 |
|title| GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction |
|authors| Yuanhui HuangWenzhao ZhengYunpeng ZhangJie ZhouJiwen Lu
|links| http://arxiv.org/abs/2405.17429v1 |
|updated| 2024-05-27 17:59:51 UTC |
|summary| 3D semantic occupancy prediction aims to obtain 3D fine-grained geometry andsemantics of the surrounding scene and is an important task for the robustnessof vision-centric autonomous driving. Most existing methods employ dense gridssuch as voxels as scene representations which ignore the sparsity of occupancyand the diversity of object scales and thus lead to unbalanced allocation ofresources. To address this we propose an object-centric representation todescribe 3D scenes with sparse 3D semantic Gaussians where each Gaussianrepresents a flexible region of interest and its semantic features. Weaggregate information from images through the attention mechanism anditeratively refine the properties of 3D Gaussians including positioncovariance and semantics. We then propose an efficient Gaussian-to-voxelsplatting method to generate 3D occupancy predictions which only aggregatesthe neighboring Gaussians for a certain position. We conduct extensiveexperiments on the widely adopted nuScenes and KITTI-360 datasets. Experimentalresults demonstrate that GaussianFormer achieves comparable performance withstate-of-the-art methods with only 17.8 - 24.8 of their memory consumption.Code is available at: https://github.com/huang-yh/GaussianFormer. |


| Item |Content|
| --- |---|
|idx| 2405.17428v1 |
|title| NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models |
|authors| Chankyu LeeRajarshi RoyMengyao XuJonathan RaimanMohammad ShoeybiBryan CatanzaroWei Ping
|links| http://arxiv.org/abs/2405.17428v1 |
|updated| 2024-05-27 17:59:45 UTC |
|summary| Decoder-only large language model LLM-based embedding models are beginningto outperform BERT or T5-based embedding models in general-purpose textembedding tasks including dense vector-based retrieval. In this work weintroduce the NV-Embed model with a variety of architectural designs andtraining procedures to significantly enhance the performance of LLM as aversatile embedding model while maintaining its simplicity andreproducibility. For model architecture we propose a latent attention layer toobtain pooled embeddings which consistently improves retrieval and downstreamtask accuracy compared to mean pooling or using the last EOS token embeddingfrom LLMs. To enhance representation learning we remove the causal attentionmask of LLMs during contrastive training. For model training we introduce atwo-stage contrastive instruction-tuning method. It first applies contrastivetraining with instructions on retrieval datasets utilizing in-batch negativesand curated hard negative examples. At stage-2 it blends various non-retrievaldatasets into instruction tuning which not only enhances non-retrieval taskaccuracy but also improves retrieval performance. Combining these techniquesour NV-Embed model using only publicly available data has achieved arecord-high score of 69.32 ranking No. 1 on the Massive Text EmbeddingBenchmark MTEB as of May 24 2024 with 56 tasks encompassing retrievalreranking classification clustering and semantic textual similarity tasks.Notably our model also attains the highest score of 59.36 on 15 retrievaltasks in the MTEB benchmark also known as BEIR. We will open-source the modelat: https://huggingface.co/nvidia/NV-Embed-v1. |


| Item |Content|
| --- |---|
|idx| 2405.17422v1 |
|title| Hardness-Aware Scene Synthesis for Semi-Supervised 3D Object Detection |
|authors| Shuai ZengWenzhao ZhengJiwen LuHaibin Yan
|links| http://arxiv.org/abs/2405.17422v1 |
|updated| 2024-05-27 17:59:23 UTC |
|summary| 3D object detection aims to recover the 3D information of concerning objectsand serves as the fundamental task of autonomous driving perception. Itsperformance greatly depends on the scale of labeled training data yet it iscostly to obtain high-quality annotations for point cloud data. Whileconventional methods focus on generating pseudo-labels for unlabeled samples assupplements for training the structural nature of 3D point cloud datafacilitates the composition of objects and backgrounds to synthesize realisticscenes. Motivated by this we propose a hardness-aware scene synthesis HASSmethod to generate adaptive synthetic scenes to improve the generalization ofthe detection models. We obtain pseudo-labels for unlabeled objects andgenerate diverse scenes with different compositions of objects and backgrounds.As the scene synthesis is sensitive to the quality of pseudo-labels we furtherpropose a hardness-aware strategy to reduce the effect of low-qualitypseudo-labels and maintain a dynamic pseudo-database to ensure the diversityand quality of synthetic scenes. Extensive experimental results on the widelyused KITTI and Waymo datasets demonstrate the superiority of the proposed HASSmethod which outperforms existing semi-supervised learning methods on 3Dobject detection. Code: https://github.com/wzzheng/HASS. |


| Item |Content|
| --- |---|
|idx| 2405.17419v1 |
|title| MultiOOD: Scaling Out-of-Distribution Detection for Multiple Modalities |
|authors| Hao DongYue ZhaoEleni ChatziOlga Fink
|links| http://arxiv.org/abs/2405.17419v1 |
|updated| 2024-05-27 17:59:02 UTC |
|summary| Detecting out-of-distribution OOD samples is important for deployingmachine learning models in safety-critical applications such as autonomousdriving and robot-assisted surgery. Existing research has mainly focused onunimodal scenarios on image data. However real-world applications areinherently multimodal which makes it essential to leverage information frommultiple modalities to enhance the efficacy of OOD detection. To establish afoundation for more realistic Multimodal OOD Detection we introduce thefirst-of-its-kind benchmark MultiOOD characterized by diverse dataset sizesand varying modality combinations. We first evaluate existing unimodal OODdetection algorithms on MultiOOD observing that the mere inclusion ofadditional modalities yields substantial improvements. This underscores theimportance of utilizing multiple modalities for OOD detection. Based on theobservation of Modality Prediction Discrepancy between in-distribution ID andOOD data and its strong correlation with OOD performance we propose theAgree-to-Disagree A2D algorithm to encourage such discrepancy duringtraining. Moreover we introduce a novel outlier synthesis method NP-Mixwhich explores broader feature spaces by leveraging the information fromnearest neighbor classes and complements A2D to strengthen OOD detectionperformance. Extensive experiments on MultiOOD demonstrate that training withA2D and NP-Mix improves existing OOD detection algorithms by a large margin.Our source code and MultiOOD benchmark are available athttps://github.com/donghao51/MultiOOD. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2405.17430v1 |
|title| Matryoshka Multimodal Models |
|authors| Mu CaiJianwei YangJianfeng GaoYong Jae Lee
|links| http://arxiv.org/abs/2405.17430v1 |
|updated| 2024-05-27 17:59:56 UTC |
|summary| Large Multimodal Models LMMs such as LLaVA have shown strong performance invisual-linguistic reasoning. These models first embed images into a fixed largenumber of visual tokens and then feed them into a Large Language Model LLM.However this design causes an excessive number of tokens for dense visualscenarios such as high-resolution images and videos leading to greatinefficiency. While token pruning/merging methods do exist they produce asingle length output for each image and do not afford flexibility in tradingoff information density v.s. efficiency. Inspired by the concept of MatryoshkaDolls we propose M3: Matryoshka Multimodal Models which learns to representvisual content as nested sets of visual tokens that capture information acrossmultiple coarse-to-fine granularities. Our approach offers several uniquebenefits for LMMs: 1 One can explicitly control the visual granularity pertest instance during inference e.g.  adjusting the number of tokens used torepresent an image based on the anticipated complexity or simplicity of thecontent 2 M3 provides a framework for analyzing the granularity needed forexisting datasets where we find that COCO-style benchmarks only need around 9visual tokens to obtain accuracy similar to that of using all 576 tokens 3Our approach provides a foundation to explore the best trade-off betweenperformance and visual token length at sample level where our investigationreveals that a large gap exists between the oracle upper bound and currentfixed-scale representations. |


| Item |Content|
| --- |---|
|idx| 2405.17428v1 |
|title| NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models |
|authors| Chankyu LeeRajarshi RoyMengyao XuJonathan RaimanMohammad ShoeybiBryan CatanzaroWei Ping
|links| http://arxiv.org/abs/2405.17428v1 |
|updated| 2024-05-27 17:59:45 UTC |
|summary| Decoder-only large language model LLM-based embedding models are beginningto outperform BERT or T5-based embedding models in general-purpose textembedding tasks including dense vector-based retrieval. In this work weintroduce the NV-Embed model with a variety of architectural designs andtraining procedures to significantly enhance the performance of LLM as aversatile embedding model while maintaining its simplicity andreproducibility. For model architecture we propose a latent attention layer toobtain pooled embeddings which consistently improves retrieval and downstreamtask accuracy compared to mean pooling or using the last EOS token embeddingfrom LLMs. To enhance representation learning we remove the causal attentionmask of LLMs during contrastive training. For model training we introduce atwo-stage contrastive instruction-tuning method. It first applies contrastivetraining with instructions on retrieval datasets utilizing in-batch negativesand curated hard negative examples. At stage-2 it blends various non-retrievaldatasets into instruction tuning which not only enhances non-retrieval taskaccuracy but also improves retrieval performance. Combining these techniquesour NV-Embed model using only publicly available data has achieved arecord-high score of 69.32 ranking No. 1 on the Massive Text EmbeddingBenchmark MTEB as of May 24 2024 with 56 tasks encompassing retrievalreranking classification clustering and semantic textual similarity tasks.Notably our model also attains the highest score of 59.36 on 15 retrievaltasks in the MTEB benchmark also known as BEIR. We will open-source the modelat: https://huggingface.co/nvidia/NV-Embed-v1. |


| Item |Content|
| --- |---|
|idx| 2405.17425v1 |
|title| From Neurons to Neutrons: A Case Study in Interpretability |
|authors| Ouail KitouniNiklas NolteVíctor Samuel Pérez-DíazSokratis TrifinopoulosMike Williams
|links| http://arxiv.org/abs/2405.17425v1 |
|updated| 2024-05-27 17:59:35 UTC |
|summary| Mechanistic Interpretability MI promises a path toward fully understandinghow neural networks make their predictions. Prior work demonstrates that evenwhen trained to perform simple arithmetic models can implement a variety ofalgorithms sometimes concurrently depending on initialization andhyperparameters. Does this mean neuron-level interpretability techniques havelimited applicability We argue that high-dimensional neural networks can learnlow-dimensional representations of their training data that are useful beyondsimply making good predictions. Such representations can be understood throughthe mechanistic interpretability lens and provide insights that aresurprisingly faithful to human-derived domain knowledge. This indicates thatsuch approaches to interpretability can be useful for deriving a newunderstanding of a problem from models trained to solve it. As a case study weextract nuclear physics concepts by studying models trained to reproducenuclear data. |


| Item |Content|
| --- |---|
|idx| 2405.17422v1 |
|title| Hardness-Aware Scene Synthesis for Semi-Supervised 3D Object Detection |
|authors| Shuai ZengWenzhao ZhengJiwen LuHaibin Yan
|links| http://arxiv.org/abs/2405.17422v1 |
|updated| 2024-05-27 17:59:23 UTC |
|summary| 3D object detection aims to recover the 3D information of concerning objectsand serves as the fundamental task of autonomous driving perception. Itsperformance greatly depends on the scale of labeled training data yet it iscostly to obtain high-quality annotations for point cloud data. Whileconventional methods focus on generating pseudo-labels for unlabeled samples assupplements for training the structural nature of 3D point cloud datafacilitates the composition of objects and backgrounds to synthesize realisticscenes. Motivated by this we propose a hardness-aware scene synthesis HASSmethod to generate adaptive synthetic scenes to improve the generalization ofthe detection models. We obtain pseudo-labels for unlabeled objects andgenerate diverse scenes with different compositions of objects and backgrounds.As the scene synthesis is sensitive to the quality of pseudo-labels we furtherpropose a hardness-aware strategy to reduce the effect of low-qualitypseudo-labels and maintain a dynamic pseudo-database to ensure the diversityand quality of synthetic scenes. Extensive experimental results on the widelyused KITTI and Waymo datasets demonstrate the superiority of the proposed HASSmethod which outperforms existing semi-supervised learning methods on 3Dobject detection. Code: https://github.com/wzzheng/HASS. |


| Item |Content|
| --- |---|
|idx| 2405.17420v1 |
|title| Survival of the Fittest Representation: A Case Study with Modular Addition |
|authors| Xiaoman Delores DingZifan Carl GuoEric J. MichaudZiming LiuMax Tegmark
|links| http://arxiv.org/abs/2405.17420v1 |
|updated| 2024-05-27 17:59:04 UTC |
|summary| When a neural network can learn multiple distinct algorithms to solve a taskhow does it choose between them during training To approach this questionwe take inspiration from ecology: when multiple species coexist theyeventually reach an equilibrium where some survive while others die out.Analogously we suggest that a neural network at initialization contains manysolutions representations and algorithms which compete with each other underpressure from resource constraints with the fittest ultimately prevailing.To investigate this Survival of the Fittest hypothesis we conduct a case studyon neural networks performing modular addition and find that these networksmultiple circular representations at different Fourier frequencies undergo suchcompetitive dynamics with only a few circles surviving at the end. We findthat the frequencies with high initial signals and gradients the fittestare more likely to survive. By increasing the embedding dimension we alsoobserve more surviving frequencies. Inspired by the Lotka-Volterra equationsdescribing the dynamics between species we find that the dynamics of thecircles can be nicely characterized by a set of linear differential equations.Our results with modular addition show that it is possible to decomposecomplicated representations into simpler components along with their basicinteractions to offer insight on the training dynamics of representations. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2405.17430v1 |
|title| Matryoshka Multimodal Models |
|authors| Mu CaiJianwei YangJianfeng GaoYong Jae Lee
|links| http://arxiv.org/abs/2405.17430v1 |
|updated| 2024-05-27 17:59:56 UTC |
|summary| Large Multimodal Models LMMs such as LLaVA have shown strong performance invisual-linguistic reasoning. These models first embed images into a fixed largenumber of visual tokens and then feed them into a Large Language Model LLM.However this design causes an excessive number of tokens for dense visualscenarios such as high-resolution images and videos leading to greatinefficiency. While token pruning/merging methods do exist they produce asingle length output for each image and do not afford flexibility in tradingoff information density v.s. efficiency. Inspired by the concept of MatryoshkaDolls we propose M3: Matryoshka Multimodal Models which learns to representvisual content as nested sets of visual tokens that capture information acrossmultiple coarse-to-fine granularities. Our approach offers several uniquebenefits for LMMs: 1 One can explicitly control the visual granularity pertest instance during inference e.g.  adjusting the number of tokens used torepresent an image based on the anticipated complexity or simplicity of thecontent 2 M3 provides a framework for analyzing the granularity needed forexisting datasets where we find that COCO-style benchmarks only need around 9visual tokens to obtain accuracy similar to that of using all 576 tokens 3Our approach provides a foundation to explore the best trade-off betweenperformance and visual token length at sample level where our investigationreveals that a large gap exists between the oracle upper bound and currentfixed-scale representations. |


| Item |Content|
| --- |---|
|idx| 2405.17429v1 |
|title| GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction |
|authors| Yuanhui HuangWenzhao ZhengYunpeng ZhangJie ZhouJiwen Lu
|links| http://arxiv.org/abs/2405.17429v1 |
|updated| 2024-05-27 17:59:51 UTC |
|summary| 3D semantic occupancy prediction aims to obtain 3D fine-grained geometry andsemantics of the surrounding scene and is an important task for the robustnessof vision-centric autonomous driving. Most existing methods employ dense gridssuch as voxels as scene representations which ignore the sparsity of occupancyand the diversity of object scales and thus lead to unbalanced allocation ofresources. To address this we propose an object-centric representation todescribe 3D scenes with sparse 3D semantic Gaussians where each Gaussianrepresents a flexible region of interest and its semantic features. Weaggregate information from images through the attention mechanism anditeratively refine the properties of 3D Gaussians including positioncovariance and semantics. We then propose an efficient Gaussian-to-voxelsplatting method to generate 3D occupancy predictions which only aggregatesthe neighboring Gaussians for a certain position. We conduct extensiveexperiments on the widely adopted nuScenes and KITTI-360 datasets. Experimentalresults demonstrate that GaussianFormer achieves comparable performance withstate-of-the-art methods with only 17.8 - 24.8 of their memory consumption.Code is available at: https://github.com/huang-yh/GaussianFormer. |


| Item |Content|
| --- |---|
|idx| 2405.17427v1 |
|title| Reason3D: Searching and Reasoning 3D Segmentation via Large Language Model |
|authors| Kuan-Chih HuangXiangtai LiLu QiShuicheng YanMing-Hsuan Yang
|links| http://arxiv.org/abs/2405.17427v1 |
|updated| 2024-05-27 17:59:41 UTC |
|summary| Recent advancements in multimodal large language models LLMs have showntheir potential in various domains especially concept reasoning. Despite thesedevelopments applications in understanding 3D environments remain limited.This paper introduces Reason3D a novel LLM designed for comprehensive 3Dunderstanding. Reason3D takes point cloud data and text prompts as input toproduce textual responses and segmentation masks facilitating advanced taskslike 3D reasoning segmentation hierarchical searching express referring andquestion answering with detailed mask outputs. Specifically we propose ahierarchical mask decoder to locate small objects within expansive scenes. Thisdecoder initially generates a coarse location estimate covering the objectsgeneral area. This foundational estimation facilitates a detailedcoarse-to-fine segmentation strategy that significantly enhances the precisionof object identification and segmentation. Experiments validate that Reason3Dachieves remarkable results on large-scale ScanNet and Matterport3D datasetsfor 3D express referring 3D question answering and 3D reasoning segmentationtasks. Code and models are available at:https://github.com/KuanchihHuang/Reason3D. |


| Item |Content|
| --- |---|
|idx| 2405.17426v1 |
|title| Benchmarking and Improving Bird's Eye View Perception Robustness in Autonomous Driving |
|authors| Shaoyuan XieLingdong KongWenwei ZhangJiawei RenLiang PanKai ChenZiwei Liu
|links| http://arxiv.org/abs/2405.17426v1 |
|updated| 2024-05-27 17:59:39 UTC |
|summary| Recent advancements in birds eye view BEV representations have shownremarkable promise for in-vehicle 3D perception. However while these methodshave achieved impressive results on standard benchmarks their robustness invaried conditions remains insufficiently assessed. In this study we presentRoboBEV an extensive benchmark suite designed to evaluate the resilience ofBEV algorithms. This suite incorporates a diverse set of camera corruptiontypes each examined over three severity levels. Our benchmarks also considerthe impact of complete sensor failures that occur when using multi-modalmodels. Through RoboBEV we assess 33 state-of-the-art BEV-based perceptionmodels spanning tasks like detection map segmentation depth estimation andoccupancy prediction. Our analyses reveal a noticeable correlation between themodels performance on in-distribution datasets and its resilience toout-of-distribution challenges. Our experimental results also underline theefficacy of strategies like pre-training and depth-free BEV transformations inenhancing robustness against out-of-distribution data. Furthermore we observethat leveraging extensive temporal information significantly improves themodels robustness. Based on our observations we design an effectiverobustness enhancement strategy based on the CLIP model. The insights from thisstudy pave the way for the development of future BEV models that seamlesslycombine accuracy with real-world robustness. |


| Item |Content|
| --- |---|
|idx| 2405.17424v1 |
|title| LARM: Large Auto-Regressive Model for Long-Horizon Embodied Intelligence |
|authors| Zhuoling LiXiaogang XuZhenhua XuSerNam LimHengshuang Zhao
|links| http://arxiv.org/abs/2405.17424v1 |
|updated| 2024-05-27 17:59:32 UTC |
|summary| Due to the need to interact with the real world embodied agents are requiredto possess comprehensive prior knowledge long-horizon planning capability anda swift response speed. Despite recent large language model LLM based agentsachieving promising performance they still exhibit several limitations. Forinstance the output of LLMs is a descriptive sentence which is ambiguous whendetermining specific actions. To address these limitations we introduce thelarge auto-regressive model LARM. LARM leverages both text and multi-viewimages as input and predicts subsequent actions in an auto-regressive manner.To train LARM we develop a novel data format named auto-regressive nodetransmission structure and assemble a corresponding dataset. Adopting atwo-phase training regimen LARM successfully harvests enchanted equipment inMinecraft which demands significantly more complex decision-making chains thanthe highest achievements of prior best methods. Besides the speed of LARM is6.8x faster. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2405.17412v1 |
|title| Towards One Model for Classical Dimensionality Reduction: A Probabilistic Perspective on UMAP and t-SNE |
|authors| Aditya RavuriNeil D. Lawrence
|links| http://arxiv.org/abs/2405.17412v1 |
|updated| 2024-05-27 17:57:12 UTC |
|summary| This paper shows that the dimensionality reduction methods UMAP and t-SNEcan be approximately recast as MAP inference methods corresponding to ageneralized Wishart-based model introduced in ProbDR. This interpretationoffers deeper theoretical insights into these algorithms while introducingtools with which similar dimensionality reduction methods can be studied. |


| Item |Content|
| --- |---|
|idx| 2405.17404v1 |
|title| Spectral Greedy Coresets for Graph Neural Networks |
|authors| Mucong DingYinhan HeJundong LiFurong Huang
|links| http://arxiv.org/abs/2405.17404v1 |
|updated| 2024-05-27 17:52:12 UTC |
|summary| The ubiquity of large-scale graphs in node-classification tasks significantlyhinders the real-world applications of Graph Neural Networks GNNs. Nodesampling graph coarsening and dataset condensation are effective strategiesfor enhancing data efficiency. However owing to the interdependence of graphnodes coreset selection which selects subsets of the data examples has notbeen successfully applied to speed up GNN training on large graphs warrantingspecial treatment. This paper studies graph coresets for GNNs and avoids theinterdependence issue by selecting ego-graphs i.e. neighborhood subgraphsaround a node based on their spectral embeddings. We decompose the coresetselection problem for GNNs into two phases: a coarse selection of widely spreadego graphs and a refined selection to diversify their topologies. We design agreedy algorithm that approximately optimizes both objectives. Our spectralgreedy graph coreset SGGC scales to graphs with millions of nodes obviatesthe need for model pre-training and applies to low-homophily graphs. Extensiveexperiments on ten datasets demonstrate that SGGC outperforms other coresetmethods by a wide margin generalizes well across GNN architectures and ismuch faster than graph condensation. |


| Item |Content|
| --- |---|
|idx| 2405.17401v1 |
|title| RB-Modulation: Training-Free Personalization of Diffusion Models using Stochastic Optimal Control |
|authors| Litu RoutYujia ChenNataniel RuizAbhishek KumarConstantine CaramanisSanjay ShakkottaiWen-Sheng Chu
|links| http://arxiv.org/abs/2405.17401v1 |
|updated| 2024-05-27 17:51:08 UTC |
|summary| We propose Reference-Based Modulation RB-Modulation a new plug-and-playsolution for training-free personalization of diffusion models. Existingtraining-free approaches exhibit difficulties in a style extraction fromreference images in the absence of additional style or content textdescriptions b unwanted content leakage from reference style images and ceffective composition of style and content. RB-Modulation is built on a novelstochastic optimal controller where a style descriptor encodes the desiredattributes through a terminal cost. The resulting drift not only overcomes thedifficulties above but also ensures high fidelity to the reference style andadheres to the given text prompt. We also introduce a cross-attention-basedfeature aggregation scheme that allows RB-Modulation to decouple content andstyle from the reference image. With theoretical justification and empiricalevidence our framework demonstrates precise extraction and control of contentand style in a training-free manner. Further our method allows a seamlesscomposition of content and style which marks a departure from the dependencyon external adapters or ControlNets. |


| Item |Content|
| --- |---|
|idx| 2405.17333v1 |
|title| Conditioning on Time is All You Need for Synthetic Survival Data Generation |
|authors| Mohd AshhadRicardo Henao
|links| http://arxiv.org/abs/2405.17333v1 |
|updated| 2024-05-27 16:34:18 UTC |
|summary| Synthetic data generation holds considerable promise offering avenues toenhance privacy fairness and data accessibility. Despite the availability ofvarious methods for generating synthetic tabular data challenges persistparticularly in specialized applications such as survival analysis. Onesignificant obstacle in survival data generation is censoring which manifestsas not knowing the precise timing of observed target events for certaininstances. Existing methods face difficulties in accurately reproducing thereal distribution of event times for both observed uncensored events andcensored events i.e. the generated event-time distributions do not accuratelymatch the underlying distributions of the real data. So motivated we propose asimple paradigm to produce synthetic survival data by generating covariatesconditioned on event times and censoring indicators thus allowing one toreuse existing conditional generative models for tabular data withoutsignificant computational overhead and without making assumptions about theusually unknown generation mechanism underlying censoring. We evaluate thismethod via extensive experiments on real-world datasets. Our methodologyoutperforms multiple competitive baselines at generating survival data whileimproving the performance of downstream survival models trained on it andtested on real data. |


| Item |Content|
| --- |---|
|idx| 2405.17324v1 |
|title| Leveraging Offline Data in Linear Latent Bandits |
|authors| Chinmaya KausikKevin TanAmbuj Tewari
|links| http://arxiv.org/abs/2405.17324v1 |
|updated| 2024-05-27 16:23:34 UTC |
|summary| Sequential decision-making domains such as recommender systems healthcareand education often have unobserved heterogeneity in the population that can bemodeled using latent bandits - a framework where an unobserved latent statedetermines the model for a trajectory. While the latent bandit framework iscompelling the extent of its generality is unclear. We first address this byestablishing a de Finetti theorem for decision processes and show thattextitevery exchangeable and coherent stateless decision process is alatent bandit. The latent bandit framework lends itself particularly well toonline learning with offline datasets a problem of growing interest insequential decision-making. One can leverage offline latent bandit data tolearn a complex model for each latent state so that an agent can simply learnthe latent state online to act optimally. We focus on a linear model for alatent bandit with d_A-dimensional actions where the latent states lie in anunknown d_K-dimensional subspace for d_K ll d_A. We present SOLD a novelprincipled method to learn this subspace from short offline trajectories withguarantees. We then provide two methods to leverage this subspace online:LOCAL-UCB and ProBALL-UCB. We demonstrate that LOCAL-UCB enjoys tildeOmind_AsqrtT d_KsqrtT1sqrtd_AT/d_KN regret guarantees wherethe effective dimension is lower when the size N of the offline dataset islarger. ProBALL-UCB enjoys a slightly weaker guarantee but is more practicaland computationally efficient. Finally we establish the efficacy of ourmethods using experiments on both synthetic data and real-life movierecommendation data from MovieLens. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2405.17410v1 |
|title| The Peripatetic Hater: Predicting Movement Among Hate Subreddits |
|authors| Daniel HickeyDaniel M. T. FesslerKristina LermanKeith Burghardt
|links| http://arxiv.org/abs/2405.17410v1 |
|updated| 2024-05-27 17:57:05 UTC |
|summary| Many online hate groups exist to disparage others based on race genderidentity sex or other characteristics. The accessibility of these communitiesallows users to join multiple types of hate groups e.g. a racist communityand misogynistic community which calls into question whether theseperipatetic users could be further radicalized compared to users that stay inone type of hate group. However little is known about the dynamics of joiningmultiple types of hate groups nor the effect of these groups on peripateticusers. In this paper we develop a new method to classify hate subreddits andthe identities they disparage which we use to better understand how usersbecome peripatetic join different types of hate subreddits. The hateclassification technique utilizes human-validated LLMs to extract the protectedidentities attacked if any across 168 subreddits. We then clusteridentity-attacking subreddits to discover three broad categories of hate:racist anti-LGBTQ and misogynistic. We show that becoming active in a usersfirst hate subreddit can cause them to become active in additional hatesubreddits of a different category. We also find that users who join additionalhate subreddits especially of a different category become more active in hatesubreddits as a whole and develop a wider hate group lexicon. We are thereforemotivated to train an AI model that we find usefully predicts the hatecategories users will become active in based on post text read and written. Theaccuracy of this model may be partly driven by peripatetic users often usingthe language of hate subreddits they eventually join. Overall these resultshighlight the unique risks associated with hate communities on a social mediaplatform as discussion of alternative targets of hate may lead users to targetmore protected identities. |


| Item |Content|
| --- |---|
|idx| 2405.17279v1 |
|title| Socially-Aware Shared Control Navigation for Assistive Mobile Robots in the Built Environment |
|authors| Yifan XuQianwei WangVineet KamatCarol Menassa
|links| http://arxiv.org/abs/2405.17279v1 |
|updated| 2024-05-27 15:40:34 UTC |
|summary| As the number of Persons with Disabilities PWD particularly those with oneor more physical impairments increases there is an increasing demand forassistive robotic technologies that can support independent mobility in thebuilt environment and reduce the burden on caregivers. Current assistivemobility platforms e.g. robotic wheelchairs often fail to incorporate userpreferences and control leading to reduced trust and efficiency. Existingshared control algorithms do not allow the incorporation of the user controlpreferences inside the navigation framework or the path planning algorithm. Inaddition existing dynamic local planner algorithms for robotic wheelchairs donot take into account the social spaces of people potentially leading suchplatforms to infringe upon these areas and cause discomfort. To address theseconcerns this work introduces a novel socially-aware shared autonomy-basednavigation system for assistive mobile robotic platforms.  Our navigation framework comprises a Global Planner and a Local Planner. Toimplement the Global Planner the proposed approach introduces a novel UserPreference Field UPF theory within its global planning framework explicitlyacknowledging user preferences to adeptly navigate away from congested areas.For the Local Planner we propose a Socially-aware Shared Control-based ModelPredictive Control with Dynamic Control Barrier Function SS-MPC-DCBF toadjust movements in real-time integrating user preferences for safer moreautonomous navigation. Evaluation results show that our Global Planner alignsclosely with user preferences compared to baselines and our Local Plannerdemonstrates enhanced safety and efficiency in dynamic and static scenarios.This integrated approach fosters trust and autonomy crucial for the acceptanceof assistive mobility technologies in the built environment. |


| Item |Content|
| --- |---|
|idx| 2405.17236v1 |
|title| From Text to Blueprint: Leveraging Text-to-Image Tools for Floor Plan Creation |
|authors| Xiaoyu LiJonathan BenjaminXin Zhang
|links| http://arxiv.org/abs/2405.17236v1 |
|updated| 2024-05-27 14:51:33 UTC |
|summary| Artificial intelligence is revolutionizing architecture through text-to-imagesynthesis converting textual descriptions into detailed visualrepresentations. We explore AI-assisted floor plan design focusing ontechnical background practical methods and future directions. Using toolslike Stable Diffusion AI leverages models such as Generative AdversarialNetworks and Variational Autoencoders to generate complex and functionalfloorplans designs. We evaluates these AI models effectiveness in generatingresidential floor plans from text prompts. Through experiments with referenceimages text prompts and sketches we assess the strengths and limitations ofcurrent text-to-image technology in architectural visualization. Architects canuse these AI tools to streamline design processes create multiple designoptions and enhance creativity and collaboration. We highlight AIs potentialto drive smarter more efficient floorplan design contributing to ongoingdiscussions on AI integration in the design profession and its future impact. |


| Item |Content|
| --- |---|
|idx| 2405.17229v1 |
|title| InsigHTable: Insight-driven Hierarchical Table Visualization with Reinforcement Learning |
|authors| Guozheng LiPeng HeXinyu WangRunfei LiChi Harold LiuChuangxin OuDong HeGuoren Wang
|links| http://arxiv.org/abs/2405.17229v1 |
|updated| 2024-05-27 14:47:00 UTC |
|summary| Embedding visual representations within original hierarchical tables canmitigate additional cognitive load stemming from the division of usersattention. The created hierarchical table visualizations can help usersunderstand and explore complex data with multi-level attributes. Howeverbecause of many options available for transforming hierarchical tables andselecting subsets for embedding the design space of hierarchical tablevisualizations becomes vast and the construction process turns out to betedious hindering users from constructing hierarchical table visualizationswith many data insights efficiently. We propose InsigHTable a mixed-initiativeand insight-driven hierarchical table transformation and visualization system.We first define data insights within hierarchical tables which consider thehierarchical structure in the table headers. Since hierarchical tablevisualization construction is a sequential decision-making process InsigHTableintegrates a deep reinforcement learning framework incorporating an auxiliaryrewards mechanism. This mechanism addresses the challenge of sparse rewards inconstructing hierarchical table visualizations. Within the deep reinforcementlearning framework the agent continuously optimizes its decision-makingprocess to create hierarchical table visualizations to uncover more insights bycollaborating with analysts. We demonstrate the usability and effectiveness ofInsigHTable through two case studies and sets of experiments. The resultsvalidate the effectiveness of the deep reinforcement learning framework andshow that InsigHTable can facilitate users to construct hierarchical tablevisualizations and understand underlying data insights. |


| Item |Content|
| --- |---|
|idx| 2405.17217v1 |
|title| Collage is the New Writing: Exploring the Fragmentation of Text and User Interfaces in AI Tools |
|authors| Daniel Buschek
|links| http://dx.doi.org/10.1145/3643834.3660681 |
|updated| 2024-05-27 14:35:17 UTC |
|summary| This essay proposes and explores the concept of Collage for the design of AIwriting tools transferred from avant-garde literature with four facets: 1fragmenting text in writing interfaces 2 juxtaposing voices content vscommand 3 integrating material from multiple sources e.g. textsuggestions and 4 shifting from manual writing to editorial andcompositional decision-making such as selecting and arranging snippets. Theessay then employs Collage as an analytical lens to analyse the user interfacedesign of recent AI writing tools and as a constructive lens to inspire newdesign directions. Finally a critical perspective relates the concerns thatwriters historically expressed through literary collage to AI writing tools. Ina broad view this essay explores how literary concepts can help advance designtheory around AI writing tools. It encourages creators of future writing toolsto engage not only with new technological possibilities but also with pastwriting innovations. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2405.17152v1 |
|title| CoSLight: Co-optimizing Collaborator Selection and Decision-making to Enhance Traffic Signal Control |
|authors| Jingqing RuanZiyue LiHua WeiHaoyuan JiangJiaming LuXuantang XiongHangyu MaoRui Zhao
|links| http://arxiv.org/abs/2405.17152v1 |
|updated| 2024-05-27 13:26:59 UTC |
|summary| Effective multi-intersection collaboration is pivotal forreinforcement-learning-based traffic signal control to alleviate congestion.Existing work mainly chooses neighboring intersections as collaborators.However quite an amount of congestion even some wide-range congestion iscaused by non-neighbors failing to collaborate. To address these issues wepropose to separate the collaborator selection as a second policy to belearned concurrently being updated with the original signal-controllingpolicy. Specifically the selection policy in real-time adaptively selects thebest teammates according to phase- and intersection-level features. Empiricalresults on both synthetic and real-world datasets provide robust validation forthe superiority of our approach offering significant improvements overexisting state-of-the-art methods. The code is available athttps://github.com/AnonymousAccountss/CoSLight. |


| Item |Content|
| --- |---|
|idx| 2405.17017v1 |
|title| Analysis of Multiscale Reinforcement Q-Learning Algorithms for Mean Field Control Games |
|authors| Andrea AngiuliJean-Pierre FouqueMathieu LaurièreMengrui Zhang
|links| http://arxiv.org/abs/2405.17017v1 |
|updated| 2024-05-27 10:01:52 UTC |
|summary| Mean Field Control Games MFCG introduced in Angiuli et al. 2022arepresent competitive games between a large number of large collaborativegroups of agents in the infinite limit of number and size of groups. In thispaper we prove the convergence of a three-timescale Reinforcement Q-LearningRL algorithm to solve MFCG in a model-free approach from the point of view ofrepresentative agents. Our analysis uses a Q-table for finite state and actionspaces updated at each discrete time-step over an infinite horizon. In Angiuliet al. 2023 we proved convergence of two-timescale algorithms for MFG andMFC separately highlighting the need to follow multiple populationdistributions in the MFC case. Here we integrate this feature for MFCG as wellas three rates of update decreasing to zero in the proper ratios. Our techniqueof proof uses a generalization to three timescales of the two-timescaleanalysis in Borkar 1997. We give a simple example satisfying the varioushypothesis made in the proof of convergence and illustrating the performance ofthe algorithm. |


| Item |Content|
| --- |---|
|idx| 2405.16887v1 |
|title| A Large Language Model-based multi-agent manufacturing system for intelligent shopfloor |
|authors| Zhen ZhaoDunbing TangHaihua ZhuZequn ZhangKai ChenChangchun LiuYuchen Ji
|links| http://arxiv.org/abs/2405.16887v1 |
|updated| 2024-05-27 07:10:04 UTC |
|summary| As productivity advances the demand of customers for multi-variety andsmall-batch production is increasing thereby putting forward higherrequirements for manufacturing systems. When production tasks frequent changesdue to this demand traditional manufacturing systems often cannot responsepromptly. The multi-agent manufacturing system is proposed to address thisproblem. However because of technical limitations the negotiation amongagents in this kind of system is realized through predefined heuristic ruleswhich is not intelligent enough to deal with the multi-variety and small batchproduction. To this end a Large Language Model-based LLM-based multi-agentmanufacturing system for intelligent shopfloor is proposed in the presentstudy. This system delineates the diverse agents and defines theircollaborative methods. The roles of the agents encompass Machine Server AgentMSA Bid Inviter Agent BIA Bidder Agent BA Thinking Agent TA andDecision Agent DA. Due to the support of LLMs TA and DA acquire the abilityof analyzing the shopfloor condition and choosing the most suitable machine asopposed to executing a predefined program artificially. The negotiation betweenBAs and BIA is the most crucial step in connecting manufacturing resources.With the support of TA and DA BIA will finalize the distribution of ordersrelying on the information of each machine returned by BA. MSAs bears theresponsibility for connecting the agents with the physical shopfloor. Thissystem aims to distribute and transmit workpieces through the collaboration ofthe agents with these distinct roles distinguishing it from other schedulingapproaches. Comparative experiments were also conducted to validate theperformance of this system. |


| Item |Content|
| --- |---|
|idx| 2405.16854v1 |
|title| Knowing What Not to Do: Leverage Language Model Insights for Action Space Pruning in Multi-agent Reinforcement Learning |
|authors| Zhihao LiuXianliang YangZichuan LiuYifan XiaWei JiangYuanyu ZhangLijuan LiGuoliang FanLei SongBian Jiang
|links| http://arxiv.org/abs/2405.16854v1 |
|updated| 2024-05-27 06:00:24 UTC |
|summary| Multi-agent reinforcement learning MARL is employed to develop autonomousagents that can learn to adopt cooperative or competitive strategies withincomplex environments. However the linear increase in the number of agentsleads to a combinatorial explosion of the action space which may result inalgorithmic instability difficulty in convergence or entrapment in localoptima. While researchers have designed a variety of effective algorithms tocompress the action space these methods also introduce new challenges such asthe need for manually designed prior knowledge or reliance on the structure ofthe problem which diminishes the applicability of these techniques. In thispaper we introduce Evolutionary action SPAce Reduction with KnowledgeeSpark an exploration function generation framework driven by large languagemodels LLMs to boost exploration and prune unnecessary actions in MARL. Usingjust a basic prompt that outlines the overall task and setting eSpark iscapable of generating exploration functions in a zero-shot manner identifyingand pruning redundant or irrelevant state-action pairs and then achievingautonomous improvement from policy feedback. In reinforcement learning tasksinvolving inventory management and traffic light control encompassing a totalof 15 scenarios eSpark consistently outperforms the combined MARL algorithm inall scenarios achieving an average performance gain of 34.4 and 9.9 in thetwo types of tasks respectively. Additionally eSpark has proven to be capableof managing situations with a large number of agents securing a 29.7improvement in scalability challenges that featured over 500 agents. The codecan be found in https://github.com/LiuZhihao2022/eSpark.git. |


| Item |Content|
| --- |---|
|idx| 2405.16751v1 |
|title| LLM-Based Cooperative Agents using Information Relevance and Plan Validation |
|authors| SeungWon SeoJunhyeok LeeSeongRae NohHyeongYeop Kang
|links| http://arxiv.org/abs/2405.16751v1 |
|updated| 2024-05-27 01:47:14 UTC |
|summary| We address the challenge of multi-agent cooperation where agents achieve acommon goal by interacting with a 3D scene and cooperating with decentralizedagents under complex partial observations. This involves managing communicationcosts and optimizing interaction trajectories in dynamic environments. Ourresearch focuses on three primary limitations of existing cooperative agentsystems. Firstly current systems demonstrate inefficiency in managing acquiredinformation through observation resulting in declining planning performance asthe environment becomes more complex with additional objects or goals.Secondly the neglect of false plans in partially observable settings leads tosuboptimal cooperative performance as agents struggle to adapt toenvironmental changes influenced by the unseen actions of other agents. Lastlythe failure to incorporate spatial data into decision-making processesrestricts the agents ability to construct optimized trajectories. To overcomethese limitations we propose the RElevance and Validation-Enhanced CooperativeLanguage Agent REVECA a novel cognitive architecture powered by GPT-3.5.REVECA leverages relevance assessment plan validation and spatial informationto enhance the efficiency and robustness of agent cooperation in dynamic andpartially observable environments while minimizing continuous communicationcosts and effectively managing irrelevant dummy objects. Our extensiveexperiments demonstrate the superiority of REVECA over previous approachesincluding those driven by GPT-4.0. Additionally a user study highlightsREVECAs potential for achieving trustworthy human-AI cooperation. We expectthat REVECA will have significant applications in gaming XR applicationseducational tools and humanoid robots contributing to substantial economiccommercial and academic advancements. |


