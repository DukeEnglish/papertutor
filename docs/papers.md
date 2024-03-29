# cs.CL 

| Item |Content|
| --- |---|
|idx| 2403.19651v1 |
|title| MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions |
|authors| Kai ZhangYi LuanHexiang HuKenton LeeSiyuan QiaoWenhu ChenYu SuMing-Wei Chang
|links| http://arxiv.org/abs/2403.19651v1 |
|updated| 2024-03-28 17:59:20 UTC |
|summary| Image retrieval i.e. finding desired images given a reference imageinherently encompasses rich multi-faceted search intents that are difficult tocapture solely using image-based measures. Recent work leverages textinstructions to allow users to more freely express their search intents.However existing work primarily focuses on image pairs that are visuallysimilar and/or can be characterized by a small set of pre-defined relations.The core thesis of this paper is that text instructions can enable retrievingimages with richer relations beyond visual similarity. To show this weintroduce MagicLens a series of self-supervised image retrieval models thatsupport open-ended instructions. MagicLens is built on a key novel insight:image pairs that naturally occur on the same web pages contain a wide range ofimplicit relations e.g. inside view of and we can bring those implicitrelations explicit by synthesizing instructions via large multimodal modelsLMMs and large language models LLMs. Trained on 36.7M query imageinstruction target image triplets with rich semantic relations mined from theweb MagicLens achieves comparable or better results on eight benchmarks ofvarious image retrieval tasks than prior state-of-the-art SOTA methods.Remarkably it outperforms previous SOTA but with a 50X smaller model size onmultiple benchmarks. Additional human analyses on a 1.4M-image unseen corpusfurther demonstrate the diversity of search intents supported by MagicLens. |


| Item |Content|
| --- |---|
|idx| 2403.19647v1 |
|title| Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models |
|authors| Samuel MarksCan RagerEric J. MichaudYonatan BelinkovDavid BauAaron Mueller
|links| http://arxiv.org/abs/2403.19647v1 |
|updated| 2024-03-28 17:56:07 UTC |
|summary| We introduce methods for discovering and applying sparse feature circuits.These are causally implicated subnetworks of human-interpretable features forexplaining language model behaviors. Circuits identified in prior work consistof polysemantic and difficult-to-interpret units like attention heads orneurons rendering them unsuitable for many downstream applications. Incontrast sparse feature circuits enable detailed understanding ofunanticipated mechanisms. Because they are based on fine-grained units sparsefeature circuits are useful for downstream tasks: We introduce SHIFT where weimprove the generalization of a classifier by ablating features that a humanjudges to be task-irrelevant. Finally we demonstrate an entirely unsupervisedand scalable interpretability pipeline by discovering thousands of sparsefeature circuits for automatically discovered model behaviors. |


| Item |Content|
| --- |---|
|idx| 2403.19634v1 |
|title| Asymmetric and trial-dependent modeling: the contribution of LIA to SdSV Challenge Task 2 |
|authors| Pierre-Michel BousquetMickael Rouvier
|links| http://arxiv.org/abs/2403.19634v1 |
|updated| 2024-03-28 17:49:31 UTC |
|summary| The SdSv challenge Task 2 provided an opportunity to assess efficiency androbustness of modern text-independent speaker verification systems. But it alsomade it possible to test new approaches capable of taking into account themain issues of this challenge duration language .... This paper describesthe contributions of our laboratory to the speaker recognition field. Thesecontributions highlight two other challenges in addition to short-duration andlanguage: the mismatch between enrollment and test data and the one betweensubsets of the evaluation trial dataset. The proposed approaches experimentallyshow their relevance and efficiency on the SdSv evaluation and could be ofinterest in many real-life applications. |


| Item |Content|
| --- |---|
|idx| 2403.19631v1 |
|title| Retrieval-Enhanced Knowledge Editing for Multi-Hop Question Answering in Language Models |
|authors| Yucheng ShiQiaoyu TanXuansheng WuShaochen ZhongKaixiong ZhouNinghao Liu
|links| http://arxiv.org/abs/2403.19631v1 |
|updated| 2024-03-28 17:47:19 UTC |
|summary| Large Language Models LLMs have shown proficiency in question-answeringtasks but often struggle to integrate real-time knowledge updates leading topotentially outdated or inaccurate responses. This problem becomes even morechallenging when dealing with multi-hop questions since they require LLMs toupdate and integrate multiple knowledge pieces relevant to the questions. Totackle the problem we propose the Retrieval-Augmented model Editing RAEframework tailored for multi-hop question answering. RAE first retrieves editedfacts and then refines the language model through in-context learning.Specifically our retrieval approach based on mutual information maximizationleverages the reasoning abilities of LLMs to identify chain facts that naivesimilarity-based searches might miss. Additionally our framework incorporatesa pruning strategy to eliminate redundant information from the retrieved factswhich enhances the editing accuracy and mitigates the hallucination problem.Our framework is supported by theoretical justification for its fact retrievalefficacy. Finally comprehensive evaluation across various LLMs validates RAEsability in providing accurate answers with updated knowledge. |


| Item |Content|
| --- |---|
|idx| 2403.19603v1 |
|title| Semantic Map-based Generation of Navigation Instructions |
|authors| Chengzu LiChao ZhangSimone TeufelRama Sanand DoddipatlaSvetlana Stoyanchev
|links| http://arxiv.org/abs/2403.19603v1 |
|updated| 2024-03-28 17:27:44 UTC |
|summary| We are interested in the generation of navigation instructions either intheir own right or as training material for robotic navigation task. In thispaper we propose a new approach to navigation instruction generation byframing the problem as an image captioning task using semantic maps as visualinput. Conventional approaches employ a sequence of panorama images to generatenavigation instructions. Semantic maps abstract away from visual details andfuse the information in multiple panorama images into a single top-downrepresentation thereby reducing computational complexity to process the input.We present a benchmark dataset for instruction generation using semantic mapspropose an initial model and ask human subjects to manually assess the qualityof generated instructions. Our initial investigations show promise in usingsemantic maps for instruction generation instead of a sequence of panoramaimages but there is vast scope for improvement. We release the code for datapreparation and model training at https://github.com/chengzu-li/VLGen. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2403.19652v1 |
|title| InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction |
|authors| Sirui XuZiyin WangYu-Xiong WangLiang-Yan Gui
|links| http://arxiv.org/abs/2403.19652v1 |
|updated| 2024-03-28 17:59:30 UTC |
|summary| Text-conditioned human motion generation has experienced significantadvancements with diffusion models trained on extensive motion capture data andcorresponding textual annotations. However extending such success to 3Ddynamic human-object interaction HOI generation faces notable challengesprimarily due to the lack of large-scale interaction data and comprehensivedescriptions that align with these interactions. This paper takes theinitiative and showcases the potential of generating human-object interactionswithout direct training on text-interaction pair data. Our key insight inachieving this is that interaction semantics and dynamics can be decoupled.Being unable to learn interaction semantics through supervised training weinstead leverage pre-trained large models synergizing knowledge from a largelanguage model and a text-to-motion model. While such knowledge offershigh-level control over interaction semantics it cannot grasp the intricaciesof low-level interaction dynamics. To overcome this issue we further introducea world model designed to comprehend simple physics modeling how human actionsinfluence object motion. By integrating these components our novel frameworkInterDreamer is able to generate text-aligned 3D HOI sequences in a zero-shotmanner. We apply InterDreamer to the BEHAVE and CHAIRS datasets and ourcomprehensive experimental analysis demonstrates its capability to generaterealistic and coherent interaction sequences that seamlessly align with thetext directives. |


| Item |Content|
| --- |---|
|idx| 2403.19651v1 |
|title| MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions |
|authors| Kai ZhangYi LuanHexiang HuKenton LeeSiyuan QiaoWenhu ChenYu SuMing-Wei Chang
|links| http://arxiv.org/abs/2403.19651v1 |
|updated| 2024-03-28 17:59:20 UTC |
|summary| Image retrieval i.e. finding desired images given a reference imageinherently encompasses rich multi-faceted search intents that are difficult tocapture solely using image-based measures. Recent work leverages textinstructions to allow users to more freely express their search intents.However existing work primarily focuses on image pairs that are visuallysimilar and/or can be characterized by a small set of pre-defined relations.The core thesis of this paper is that text instructions can enable retrievingimages with richer relations beyond visual similarity. To show this weintroduce MagicLens a series of self-supervised image retrieval models thatsupport open-ended instructions. MagicLens is built on a key novel insight:image pairs that naturally occur on the same web pages contain a wide range ofimplicit relations e.g. inside view of and we can bring those implicitrelations explicit by synthesizing instructions via large multimodal modelsLMMs and large language models LLMs. Trained on 36.7M query imageinstruction target image triplets with rich semantic relations mined from theweb MagicLens achieves comparable or better results on eight benchmarks ofvarious image retrieval tasks than prior state-of-the-art SOTA methods.Remarkably it outperforms previous SOTA but with a 50X smaller model size onmultiple benchmarks. Additional human analyses on a 1.4M-image unseen corpusfurther demonstrate the diversity of search intents supported by MagicLens. |


| Item |Content|
| --- |---|
|idx| 2403.19648v1 |
|title| Human-compatible driving partners through data-regularized self-play reinforcement learning |
|authors| Daphne CornelisseEugene Vinitsky
|links| http://arxiv.org/abs/2403.19648v1 |
|updated| 2024-03-28 17:56:56 UTC |
|summary| A central challenge for autonomous vehicles is coordinating with humans.Therefore incorporating realistic human agents is essential for scalabletraining and evaluation of autonomous driving systems in simulation. Simulationagents are typically developed by imitating large-scale high-quality datasetsof human driving. However pure imitation learning agents empirically have highcollision rates when executed in a multi-agent closed-loop setting. To buildagents that are realistic and effective in closed-loop settings we proposeHuman-Regularized PPO HR-PPO a multi-agent algorithm where agents aretrained through self-play with a small penalty for deviating from a humanreference policy. In contrast to prior work our approach is RL-first and onlyuses 30 minutes of imperfect human demonstrations. We evaluate agents in alarge set of multi-agent traffic scenes. Results show our HR-PPO agents arehighly effective in achieving goals with a success rate of 93 an off-roadrate of 3.5 and a collision rate of 3. At the same time the agents drive ina human-like manner as measured by their similarity to existing human drivinglogs. We also find that HR-PPO agents show considerable improvements on proxymeasures for coordination with human driving particularly in highlyinteractive scenarios. We open-source our code and trained agents athttps://github.com/Emerge-Lab/nocturne_lab and provide demonstrations of agentbehaviors at https://sites.google.com/view/driving-partners. |


| Item |Content|
| --- |---|
|idx| 2403.19647v1 |
|title| Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models |
|authors| Samuel MarksCan RagerEric J. MichaudYonatan BelinkovDavid BauAaron Mueller
|links| http://arxiv.org/abs/2403.19647v1 |
|updated| 2024-03-28 17:56:07 UTC |
|summary| We introduce methods for discovering and applying sparse feature circuits.These are causally implicated subnetworks of human-interpretable features forexplaining language model behaviors. Circuits identified in prior work consistof polysemantic and difficult-to-interpret units like attention heads orneurons rendering them unsuitable for many downstream applications. Incontrast sparse feature circuits enable detailed understanding ofunanticipated mechanisms. Because they are based on fine-grained units sparsefeature circuits are useful for downstream tasks: We introduce SHIFT where weimprove the generalization of a classifier by ablating features that a humanjudges to be task-irrelevant. Finally we demonstrate an entirely unsupervisedand scalable interpretability pipeline by discovering thousands of sparsefeature circuits for automatically discovered model behaviors. |


| Item |Content|
| --- |---|
|idx| 2403.19631v1 |
|title| Retrieval-Enhanced Knowledge Editing for Multi-Hop Question Answering in Language Models |
|authors| Yucheng ShiQiaoyu TanXuansheng WuShaochen ZhongKaixiong ZhouNinghao Liu
|links| http://arxiv.org/abs/2403.19631v1 |
|updated| 2024-03-28 17:47:19 UTC |
|summary| Large Language Models LLMs have shown proficiency in question-answeringtasks but often struggle to integrate real-time knowledge updates leading topotentially outdated or inaccurate responses. This problem becomes even morechallenging when dealing with multi-hop questions since they require LLMs toupdate and integrate multiple knowledge pieces relevant to the questions. Totackle the problem we propose the Retrieval-Augmented model Editing RAEframework tailored for multi-hop question answering. RAE first retrieves editedfacts and then refines the language model through in-context learning.Specifically our retrieval approach based on mutual information maximizationleverages the reasoning abilities of LLMs to identify chain facts that naivesimilarity-based searches might miss. Additionally our framework incorporatesa pruning strategy to eliminate redundant information from the retrieved factswhich enhances the editing accuracy and mitigates the hallucination problem.Our framework is supported by theoretical justification for its fact retrievalefficacy. Finally comprehensive evaluation across various LLMs validates RAEsability in providing accurate answers with updated knowledge. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2403.19648v1 |
|title| Human-compatible driving partners through data-regularized self-play reinforcement learning |
|authors| Daphne CornelisseEugene Vinitsky
|links| http://arxiv.org/abs/2403.19648v1 |
|updated| 2024-03-28 17:56:56 UTC |
|summary| A central challenge for autonomous vehicles is coordinating with humans.Therefore incorporating realistic human agents is essential for scalabletraining and evaluation of autonomous driving systems in simulation. Simulationagents are typically developed by imitating large-scale high-quality datasetsof human driving. However pure imitation learning agents empirically have highcollision rates when executed in a multi-agent closed-loop setting. To buildagents that are realistic and effective in closed-loop settings we proposeHuman-Regularized PPO HR-PPO a multi-agent algorithm where agents aretrained through self-play with a small penalty for deviating from a humanreference policy. In contrast to prior work our approach is RL-first and onlyuses 30 minutes of imperfect human demonstrations. We evaluate agents in alarge set of multi-agent traffic scenes. Results show our HR-PPO agents arehighly effective in achieving goals with a success rate of 93 an off-roadrate of 3.5 and a collision rate of 3. At the same time the agents drive ina human-like manner as measured by their similarity to existing human drivinglogs. We also find that HR-PPO agents show considerable improvements on proxymeasures for coordination with human driving particularly in highlyinteractive scenarios. We open-source our code and trained agents athttps://github.com/Emerge-Lab/nocturne_lab and provide demonstrations of agentbehaviors at https://sites.google.com/view/driving-partners. |


| Item |Content|
| --- |---|
|idx| 2403.19647v1 |
|title| Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models |
|authors| Samuel MarksCan RagerEric J. MichaudYonatan BelinkovDavid BauAaron Mueller
|links| http://arxiv.org/abs/2403.19647v1 |
|updated| 2024-03-28 17:56:07 UTC |
|summary| We introduce methods for discovering and applying sparse feature circuits.These are causally implicated subnetworks of human-interpretable features forexplaining language model behaviors. Circuits identified in prior work consistof polysemantic and difficult-to-interpret units like attention heads orneurons rendering them unsuitable for many downstream applications. Incontrast sparse feature circuits enable detailed understanding ofunanticipated mechanisms. Because they are based on fine-grained units sparsefeature circuits are useful for downstream tasks: We introduce SHIFT where weimprove the generalization of a classifier by ablating features that a humanjudges to be task-irrelevant. Finally we demonstrate an entirely unsupervisedand scalable interpretability pipeline by discovering thousands of sparsefeature circuits for automatically discovered model behaviors. |


| Item |Content|
| --- |---|
|idx| 2403.19631v1 |
|title| Retrieval-Enhanced Knowledge Editing for Multi-Hop Question Answering in Language Models |
|authors| Yucheng ShiQiaoyu TanXuansheng WuShaochen ZhongKaixiong ZhouNinghao Liu
|links| http://arxiv.org/abs/2403.19631v1 |
|updated| 2024-03-28 17:47:19 UTC |
|summary| Large Language Models LLMs have shown proficiency in question-answeringtasks but often struggle to integrate real-time knowledge updates leading topotentially outdated or inaccurate responses. This problem becomes even morechallenging when dealing with multi-hop questions since they require LLMs toupdate and integrate multiple knowledge pieces relevant to the questions. Totackle the problem we propose the Retrieval-Augmented model Editing RAEframework tailored for multi-hop question answering. RAE first retrieves editedfacts and then refines the language model through in-context learning.Specifically our retrieval approach based on mutual information maximizationleverages the reasoning abilities of LLMs to identify chain facts that naivesimilarity-based searches might miss. Additionally our framework incorporatesa pruning strategy to eliminate redundant information from the retrieved factswhich enhances the editing accuracy and mitigates the hallucination problem.Our framework is supported by theoretical justification for its fact retrievalefficacy. Finally comprehensive evaluation across various LLMs validates RAEsability in providing accurate answers with updated knowledge. |


| Item |Content|
| --- |---|
|idx| 2403.19629v1 |
|title| Metric Learning from Limited Pairwise Preference Comparisons |
|authors| Zhi WangGeelon SoRamya Korlakai Vinayak
|links| http://arxiv.org/abs/2403.19629v1 |
|updated| 2024-03-28 17:46:25 UTC |
|summary| We study metric learning from preference comparisons under the ideal pointmodel in which a user prefers an item over another if it is closer to theirlatent ideal item. These items are embedded into mathbbRd equipped withan unknown Mahalanobis distance shared across users. While recent work showsthat it is possible to simultaneously recover the metric and ideal items givenmathcalOd pairwise comparisons per user in practice we often have alimited budget of od comparisons. We study whether the metric can still berecovered even though it is known that learning individual ideal items is nowno longer possible. We show that in general od comparisons reveals noinformation about the metric even with infinitely many users. However whencomparisons are made over items that exhibit low-dimensional structure eachuser can contribute to learning the metric restricted to a low-dimensionalsubspace so that the metric can be jointly identified. We present adivide-and-conquer approach that achieves this and provide theoreticalrecovery guarantees and empirical validation. |


| Item |Content|
| --- |---|
|idx| 2403.19625v1 |
|title| Top-$k$ Classification and Cardinality-Aware Prediction |
|authors| Anqi MaoMehryar MohriYutao Zhong
|links| http://arxiv.org/abs/2403.19625v1 |
|updated| 2024-03-28 17:45:03 UTC |
|summary| We present a detailed study of top-k classification the task of predictingthe k most probable classes for an input extending beyond single-classprediction. We demonstrate that several prevalent surrogate loss functions inmulti-class classification such as comp-sum and constrained losses aresupported by H-consistency bounds with respect to the top-k loss. Thesebounds guarantee consistency in relation to the hypothesis set H providingstronger guarantees than Bayes-consistency due to their non-asymptotic andhypothesis-set specific nature. To address the trade-off between accuracy andcardinality k we further introduce cardinality-aware loss functions throughinstance-dependent cost-sensitive learning. For these functions we derivecost-sensitive comp-sum and constrained surrogate losses establishing theirH-consistency bounds and Bayes-consistency. Minimizing these losses leads tonew cardinality-aware algorithms for top-k classification. We report theresults of extensive experiments on CIFAR-100 ImageNet CIFAR-10 and SVHNdatasets demonstrating the effectiveness and benefit of these algorithms. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2403.19655v1 |
|title| GaussianCube: Structuring Gaussian Splatting using Optimal Transport for 3D Generative Modeling |
|authors| Bowen ZhangYiji ChengJiaolong YangChunyu WangFeng ZhaoYansong TangDong ChenBaining Guo
|links| http://arxiv.org/abs/2403.19655v1 |
|updated| 2024-03-28 17:59:50 UTC |
|summary| 3D Gaussian Splatting GS have achieved considerable improvement over NeuralRadiance Fields in terms of 3D fitting fidelity and rendering speed. Howeverthis unstructured representation with scattered Gaussians poses a significantchallenge for generative modeling. To address the problem we introduceGaussianCube a structured GS representation that is both powerful andefficient for generative modeling. We achieve this by first proposing amodified densification-constrained GS fitting algorithm which can yieldhigh-quality fitting results using a fixed number of free Gaussians and thenre-arranging the Gaussians into a predefined voxel grid via Optimal Transport.The structured grid representation allows us to use standard 3D U-Net as ourbackbone in diffusion generative modeling without elaborate designs. Extensiveexperiments conducted on ShapeNet and OmniObject3D show that our model achievesstate-of-the-art generation results both qualitatively and quantitativelyunderscoring the potential of GaussianCube as a powerful and versatile 3Drepresentation. |


| Item |Content|
| --- |---|
|idx| 2403.19654v1 |
|title| RSMamba: Remote Sensing Image Classification with State Space Model |
|authors| Keyan ChenBowen ChenChenyang LiuWenyuan LiZhengxia ZouZhenwei Shi
|links| http://arxiv.org/abs/2403.19654v1 |
|updated| 2024-03-28 17:59:49 UTC |
|summary| Remote sensing image classification forms the foundation of variousunderstanding tasks serving a crucial function in remote sensing imageinterpretation. The recent advancements of Convolutional Neural Networks CNNsand Transformers have markedly enhanced classification accuracy. Nonethelessremote sensing scene classification remains a significant challenge especiallygiven the complexity and diversity of remote sensing scenarios and thevariability of spatiotemporal resolutions. The capacity for whole-imageunderstanding can provide more precise semantic cues for scene discrimination.In this paper we introduce RSMamba a novel architecture for remote sensingimage classification. RSMamba is based on the State Space Model SSM andincorporates an efficient hardware-aware design known as the Mamba. Itintegrates the advantages of both a global receptive field and linear modelingcomplexity. To overcome the limitation of the vanilla Mamba which can onlymodel causal sequences and is not adaptable to two-dimensional image data wepropose a dynamic multi-path activation mechanism to augment Mambas capacityto model non-causal data. Notably RSMamba maintains the inherent modelingmechanism of the vanilla Mamba yet exhibits superior performance acrossmultiple remote sensing image classification datasets. This indicates thatRSMamba holds significant potential to function as the backbone of futurevisual foundation models. The code will be available aturlhttps://github.com/KyanChen/RSMamba. |


| Item |Content|
| --- |---|
|idx| 2403.19653v1 |
|title| Detecting Image Attribution for Text-to-Image Diffusion Models in RGB and Beyond |
|authors| Katherine XuLingzhi ZhangJianbo Shi
|links| http://arxiv.org/abs/2403.19653v1 |
|updated| 2024-03-28 17:59:42 UTC |
|summary| Modern text-to-image T2I diffusion models can generate images withremarkable realism and creativity. These advancements have sparked research infake image detection and attribution yet prior studies have not fully exploredthe practical and scientific dimensions of this task. In addition toattributing images to 12 state-of-the-art T2I generators we provide extensiveanalyses on what inference stage hyperparameters and image modifications arediscernible. Our experiments reveal that initialization seeds are highlydetectable along with other subtle variations in the image generation processto some extent. We further investigate what visual traces are leveraged inimage attribution by perturbing high-frequency details and employing mid-levelrepresentations of image style and structure. Notably altering high-frequencyinformation causes only slight reductions in accuracy and training anattributor on style representations outperforms training on RGB images. Ouranalyses underscore that fake images are detectable and attributable at variouslevels of visual granularity than previously explored. |


| Item |Content|
| --- |---|
|idx| 2403.19652v1 |
|title| InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction |
|authors| Sirui XuZiyin WangYu-Xiong WangLiang-Yan Gui
|links| http://arxiv.org/abs/2403.19652v1 |
|updated| 2024-03-28 17:59:30 UTC |
|summary| Text-conditioned human motion generation has experienced significantadvancements with diffusion models trained on extensive motion capture data andcorresponding textual annotations. However extending such success to 3Ddynamic human-object interaction HOI generation faces notable challengesprimarily due to the lack of large-scale interaction data and comprehensivedescriptions that align with these interactions. This paper takes theinitiative and showcases the potential of generating human-object interactionswithout direct training on text-interaction pair data. Our key insight inachieving this is that interaction semantics and dynamics can be decoupled.Being unable to learn interaction semantics through supervised training weinstead leverage pre-trained large models synergizing knowledge from a largelanguage model and a text-to-motion model. While such knowledge offershigh-level control over interaction semantics it cannot grasp the intricaciesof low-level interaction dynamics. To overcome this issue we further introducea world model designed to comprehend simple physics modeling how human actionsinfluence object motion. By integrating these components our novel frameworkInterDreamer is able to generate text-aligned 3D HOI sequences in a zero-shotmanner. We apply InterDreamer to the BEHAVE and CHAIRS datasets and ourcomprehensive experimental analysis demonstrates its capability to generaterealistic and coherent interaction sequences that seamlessly align with thetext directives. |


| Item |Content|
| --- |---|
|idx| 2403.19651v1 |
|title| MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions |
|authors| Kai ZhangYi LuanHexiang HuKenton LeeSiyuan QiaoWenhu ChenYu SuMing-Wei Chang
|links| http://arxiv.org/abs/2403.19651v1 |
|updated| 2024-03-28 17:59:20 UTC |
|summary| Image retrieval i.e. finding desired images given a reference imageinherently encompasses rich multi-faceted search intents that are difficult tocapture solely using image-based measures. Recent work leverages textinstructions to allow users to more freely express their search intents.However existing work primarily focuses on image pairs that are visuallysimilar and/or can be characterized by a small set of pre-defined relations.The core thesis of this paper is that text instructions can enable retrievingimages with richer relations beyond visual similarity. To show this weintroduce MagicLens a series of self-supervised image retrieval models thatsupport open-ended instructions. MagicLens is built on a key novel insight:image pairs that naturally occur on the same web pages contain a wide range ofimplicit relations e.g. inside view of and we can bring those implicitrelations explicit by synthesizing instructions via large multimodal modelsLMMs and large language models LLMs. Trained on 36.7M query imageinstruction target image triplets with rich semantic relations mined from theweb MagicLens achieves comparable or better results on eight benchmarks ofvarious image retrieval tasks than prior state-of-the-art SOTA methods.Remarkably it outperforms previous SOTA but with a 50X smaller model size onmultiple benchmarks. Additional human analyses on a 1.4M-image unseen corpusfurther demonstrate the diversity of search intents supported by MagicLens. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2403.19629v1 |
|title| Metric Learning from Limited Pairwise Preference Comparisons |
|authors| Zhi WangGeelon SoRamya Korlakai Vinayak
|links| http://arxiv.org/abs/2403.19629v1 |
|updated| 2024-03-28 17:46:25 UTC |
|summary| We study metric learning from preference comparisons under the ideal pointmodel in which a user prefers an item over another if it is closer to theirlatent ideal item. These items are embedded into mathbbRd equipped withan unknown Mahalanobis distance shared across users. While recent work showsthat it is possible to simultaneously recover the metric and ideal items givenmathcalOd pairwise comparisons per user in practice we often have alimited budget of od comparisons. We study whether the metric can still berecovered even though it is known that learning individual ideal items is nowno longer possible. We show that in general od comparisons reveals noinformation about the metric even with infinitely many users. However whencomparisons are made over items that exhibit low-dimensional structure eachuser can contribute to learning the metric restricted to a low-dimensionalsubspace so that the metric can be jointly identified. We present adivide-and-conquer approach that achieves this and provide theoreticalrecovery guarantees and empirical validation. |


| Item |Content|
| --- |---|
|idx| 2403.19625v1 |
|title| Top-$k$ Classification and Cardinality-Aware Prediction |
|authors| Anqi MaoMehryar MohriYutao Zhong
|links| http://arxiv.org/abs/2403.19625v1 |
|updated| 2024-03-28 17:45:03 UTC |
|summary| We present a detailed study of top-k classification the task of predictingthe k most probable classes for an input extending beyond single-classprediction. We demonstrate that several prevalent surrogate loss functions inmulti-class classification such as comp-sum and constrained losses aresupported by H-consistency bounds with respect to the top-k loss. Thesebounds guarantee consistency in relation to the hypothesis set H providingstronger guarantees than Bayes-consistency due to their non-asymptotic andhypothesis-set specific nature. To address the trade-off between accuracy andcardinality k we further introduce cardinality-aware loss functions throughinstance-dependent cost-sensitive learning. For these functions we derivecost-sensitive comp-sum and constrained surrogate losses establishing theirH-consistency bounds and Bayes-consistency. Minimizing these losses leads tonew cardinality-aware algorithms for top-k classification. We report theresults of extensive experiments on CIFAR-100 ImageNet CIFAR-10 and SVHNdatasets demonstrating the effectiveness and benefit of these algorithms. |


| Item |Content|
| --- |---|
|idx| 2403.19587v1 |
|title| Taming the Interactive Particle Langevin Algorithm -- the superlinear case |
|authors| Tim JohnstonNikolaos MakrasSotirios Sabanis
|links| http://arxiv.org/abs/2403.19587v1 |
|updated| 2024-03-28 17:11:25 UTC |
|summary| Recent advances in stochastic optimization have yielded the interactiveparticle Langevin algorithm IPLA which leverages the notion of interactingparticle systems IPS to efficiently sample from approximate posteriordensities. This becomes particularly crucial within the framework ofExpectation-Maximization EM where the E-step is computationally challengingor even intractable. Although prior research has focused on scenarios involvingconvex cases with gradients of log densities that grow at most linearly ourwork extends this framework to include polynomial growth. Taming techniques areemployed to produce an explicit discretization scheme that yields a new classof stable under such non-linearities algorithms which are called tamedinteractive particle Langevin algorithms tIPLA. We obtain non-asymptoticconvergence error estimates in Wasserstein-2 distance for the new class underan optimal rate. |


| Item |Content|
| --- |---|
|idx| 2403.19516v1 |
|title| Maximum Likelihood Estimation on Stochastic Blockmodels for Directed Graph Clustering |
|authors| Mihai CucuringuXiaowen DongNing Zhang
|links| http://arxiv.org/abs/2403.19516v1 |
|updated| 2024-03-28 15:47:13 UTC |
|summary| This paper studies the directed graph clustering problem through the lens ofstatistics where we formulate clustering as estimating underlying communitiesin the directed stochastic block model DSBM. We conduct the maximumlikelihood estimation MLE on the DSBM and thereby ascertain the most probablecommunity assignment given the observed graph structure. In addition to thestatistical point of view we further establish the equivalence between thisMLE formulation and a novel flow optimization heuristic which jointlyconsiders two important directed graph statistics: edge density and edgeorientation. Building on this new formulation of directed clustering weintroduce two efficient and interpretable directed clustering algorithms aspectral clustering algorithm and a semidefinite programming based clusteringalgorithm. We provide a theoretical upper bound on the number of misclusteredvertices of the spectral clustering algorithm using tools from matrixperturbation theory. We compare both quantitatively and qualitatively ourproposed algorithms with existing directed clustering methods on both syntheticand real-world data thus providing further ground to our theoreticalcontributions. |


| Item |Content|
| --- |---|
|idx| 2403.19500v1 |
|title| Tensor Network-Constrained Kernel Machines as Gaussian Processes |
|authors| Frederiek WeselKim Batselier
|links| http://arxiv.org/abs/2403.19500v1 |
|updated| 2024-03-28 15:29:30 UTC |
|summary| Tensor Networks TNs have recently been used to speed up kernel machines byconstraining the model weights yielding exponential computational and storagesavings. In this paper we prove that the outputs of Canonical PolyadicDecomposition CPD and Tensor Train TT-constrained kernel machines recover aGaussian Process GP which we fully characterize when placing i.i.d. priorsover their parameters. We analyze the convergence of both CPD andTT-constrained models and show how TT yields models exhibiting more GPbehavior compared to CPD for the same number of model parameters. Weempirically observe this behavior in two numerical experiments where werespectively analyze the convergence to the GP and the performance atprediction. We thereby establish a connection between TN-constrained kernelmachines and GPs. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2403.19620v1 |
|title| Collaborative Interactive Evolution of Art in the Latent Space of Deep Generative Models |
|authors| Ole HallAnil Yaman
|links| http://arxiv.org/abs/2403.19620v1 |
|updated| 2024-03-28 17:40:15 UTC |
|summary| Generative Adversarial Networks GANs have shown great success in generatinghigh quality images and are thus used as one of the main approaches to generateart images. However usually the image generation process involves samplingfrom the latent space of the learned art representations allowing littlecontrol over the output. In this work we first employ GANs that are trained toproduce creative images using an architecture known as Creative AdversarialNetworks CANs then we employ an evolutionary approach to navigate withinthe latent space of the models to discover images. We use automatic aestheticand collaborative interactive human evaluation metrics to assess the generatedimages. In the human interactive evaluation case we propose a collaborativeevaluation based on the assessments of several participants. Furthermore wealso experiment with an intelligent mutation operator that aims to improve thequality of the images through local search based on an aesthetic measure. Weevaluate the effectiveness of this approach by comparing the results producedby the automatic and collaborative interactive evolution. The results show thatthe proposed approach can generate highly attractive art images when theevolution is guided by collaborative human feedback. |


| Item |Content|
| --- |---|
|idx| 2403.19560v1 |
|title| Exploring Communication Dynamics: Eye-tracking Analysis in Pair Programming of Computer Science Education |
|authors| Wunmin JangHong GaoTilman MichaeliEnkelejda Kasneci
|links| http://dx.doi.org/10.1145/3649902.3653942 |
|updated| 2024-03-28 16:44:20 UTC |
|summary| Pair programming is widely recognized as an effective educational tool incomputer science that promotes collaborative learning and mirrors real-worldwork dynamics. However communication breakdowns within pairs significantlychallenge this learning process. In this study we use eye-tracking datarecorded during pair programming sessions to study communication dynamicsbetween various pair programming roles across different student expert andmixed group cohorts containing 19 participants. By combining eye-tracking dataanalysis with focus group interviews and questionnaires we provide insightsinto communications multifaceted nature in pair programming. Our findingshighlight distinct eye-tracking patterns indicating changes in communicationskills across group compositions with participants prioritizing codeexploration over communication especially during challenging tasks. Furtherstudents showed a preference for pairing with experts emphasizing theimportance of understanding group formation in pair programming scenarios.These insights emphasize the importance of understanding group dynamics andenhancing communication skills through pair programming for successful outcomesin computer science education. |


| Item |Content|
| --- |---|
|idx| 2403.19506v1 |
|title| LLMs as Academic Reading Companions: Extending HCI Through Synthetic Personae |
|authors| Celia ChenAlex Leitch
|links| http://arxiv.org/abs/2403.19506v1 |
|updated| 2024-03-28 15:37:10 UTC |
|summary| This position paper argues that large language models LLMs constitutepromising yet underutilized academic reading companions capable of enhancinglearning. We detail an exploratory study examining Claude.ai from Anthropic anLLM-based interactive assistant that helps students comprehend complexqualitative literature content. The study compares quantitative survey data andqualitative interviews assessing outcomes between a control group and anexperimental group leveraging Claude.ai over a semester across two graduatecourses. Initial findings demonstrate tangible improvements in readingcomprehension and engagement among participants using the AI agent versusunsupported independent study. However there is potential for overreliance andethical considerations that warrant continued investigation. By documenting anearly integration of an LLM reading companion into an educational context thiswork contributes pragmatic insights to guide development of synthetic personaesupporting learning. Broader impacts compel policy and industry actions touphold responsible design in order to maximize benefits of AI integration whileprioritizing student wellbeing. |


| Item |Content|
| --- |---|
|idx| 2403.19475v1 |
|title| A theoretical framework for the design and analysis of computational thinking problems in education |
|authors| Giorgia AdorniAlberto PiattiEngin BumbacherLucio NegriniFrancesco MondadaDorit AssafFrancesca MangiliLuca Gambardella
|links| http://arxiv.org/abs/2403.19475v1 |
|updated| 2024-03-28 15:02:28 UTC |
|summary| The field of computational thinking education has grown in recent years asresearchers and educators have sought to develop and assess studentscomputational thinking abilities. While much of the research in this area hasfocused on defining computational thinking the competencies it involves andhow to assess them in teaching and learning contexts this work takes adifferent approach. We provide a more situated perspective on computationalthinking focusing on the types of problems that require computational thinkingskills to be solved and the features that support these processes. We develop aframework for analysing existing computational thinking problems in aneducational context. We conduct a comprehensive literature review to identifyprototypical activities from areas where computational thinking is typicallypursued in education. We identify the main components and characteristics ofthese activities along with their influence on activating computationalthinking competencies. The framework provides a catalogue of computationalthinking skills that can be used to understand the relationship between problemfeatures and competencies activated. This study contributes to the field ofcomputational thinking education by offering a tool for evaluating and revisingexisting problems to activate specific skills and for assisting in designingnew problems that target the development of particular competencies. Theresults of this study may be of interest to researchers and educators workingin computational thinking education. |


| Item |Content|
| --- |---|
|idx| 2403.19436v1 |
|title| "At the end of the day, I am accountable": Gig Workers' Self-Tracking for Multi-Dimensional Accountability Management |
|authors| Rie HeleneHernandezQiurong SongYubo KouXinning Gui
|links| http://arxiv.org/abs/2403.19436v1 |
|updated| 2024-03-28 14:04:30 UTC |
|summary| Tracking is inherent in and central to the gig economy. Platforms track gigworkers performance through metrics such as acceptance rate and punctualitywhile gig workers themselves engage in self-tracking. Although prior researchhas extensively examined how gig platforms track workers through metrics --with some studies briefly acknowledging the phenomenon of self-tracking amongworkers -- there is a dearth of studies that explore how and why gig workerstrack themselves. To address this we conducted 25 semi-structured interviewsrevealing how gig workers self-tracking to manage accountabilities tothemselves and external entities across three identities: the holistic selfthe entrepreneurial self and the platformized self. We connect our findings toneoliberalism through which we contextualize gig workers self-accountabilityand the invisible labor of self-tracking. We further discuss how self-trackingmitigates information and power asymmetries in gig work and offer designimplications to support gig workers multi-dimensional self-tracking. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2403.19648v1 |
|title| Human-compatible driving partners through data-regularized self-play reinforcement learning |
|authors| Daphne CornelisseEugene Vinitsky
|links| http://arxiv.org/abs/2403.19648v1 |
|updated| 2024-03-28 17:56:56 UTC |
|summary| A central challenge for autonomous vehicles is coordinating with humans.Therefore incorporating realistic human agents is essential for scalabletraining and evaluation of autonomous driving systems in simulation. Simulationagents are typically developed by imitating large-scale high-quality datasetsof human driving. However pure imitation learning agents empirically have highcollision rates when executed in a multi-agent closed-loop setting. To buildagents that are realistic and effective in closed-loop settings we proposeHuman-Regularized PPO HR-PPO a multi-agent algorithm where agents aretrained through self-play with a small penalty for deviating from a humanreference policy. In contrast to prior work our approach is RL-first and onlyuses 30 minutes of imperfect human demonstrations. We evaluate agents in alarge set of multi-agent traffic scenes. Results show our HR-PPO agents arehighly effective in achieving goals with a success rate of 93 an off-roadrate of 3.5 and a collision rate of 3. At the same time the agents drive ina human-like manner as measured by their similarity to existing human drivinglogs. We also find that HR-PPO agents show considerable improvements on proxymeasures for coordination with human driving particularly in highlyinteractive scenarios. We open-source our code and trained agents athttps://github.com/Emerge-Lab/nocturne_lab and provide demonstrations of agentbehaviors at https://sites.google.com/view/driving-partners. |


| Item |Content|
| --- |---|
|idx| 2403.19375v1 |
|title| Multi-Agent Team Access Monitoring: Environments that Benefit from Target Information Sharing |
|authors| Andrew DudashScott JamesRyan Rubel
|links| http://arxiv.org/abs/2403.19375v1 |
|updated| 2024-03-28 12:37:11 UTC |
|summary| Robotic access monitoring of multiple target areas has applications includingcheckpoint enforcement surveillance and containment of fire and flood hazards.Monitoring access for a single target region has been successfully modeled as aminimum-cut problem. We generalize this model to support multiple target areasusing two approaches: iterating on individual targets and examining thecollections of targets holistically. Through simulation we measure theperformance of each approach on different scenarios. |


| Item |Content|
| --- |---|
|idx| 2403.19253v1 |
|title| Inferring Latent Temporal Sparse Coordination Graph for Multi-Agent Reinforcement Learning |
|authors| Wei DuanJie LuJunyu Xuan
|links| http://arxiv.org/abs/2403.19253v1 |
|updated| 2024-03-28 09:20:15 UTC |
|summary| Effective agent coordination is crucial in cooperative Multi-AgentReinforcement Learning MARL. While agent cooperation can be represented bygraph structures prevailing graph learning methods in MARL are limited. Theyrely solely on one-step observations neglecting crucial historicalexperiences leading to deficient graphs that foster redundant or detrimentalinformation exchanges. Additionally high computational demands for action-paircalculations in dense graphs impede scalability. To address these challengeswe propose inferring a Latent Temporal Sparse Coordination Graph LTS-CG forMARL. The LTS-CG leverages agents historical observations to calculate anagent-pair probability matrix where a sparse graph is sampled from and usedfor knowledge exchange between agents thereby simultaneously capturing agentdependencies and relation uncertainty. The computational complexity of thisprocedure is only related to the number of agents. This graph learning processis further augmented by two innovative characteristics: Predict-Future whichenables agents to foresee upcoming observations and Infer-Present ensuring athorough grasp of the environmental context from limited data. These featuresallow LTS-CG to construct temporal graphs from historical and real-timeinformation promoting knowledge exchange during policy learning and effectivecollaboration. Graph learning and agent training occur simultaneously in anend-to-end manner. Our demonstrated results on the StarCraft II benchmarkunderscore LTS-CGs superior performance. |


| Item |Content|
| --- |---|
|idx| 2403.18985v1 |
|title| Robustness and Visual Explanation for Black Box Image, Video, and ECG Signal Classification with Reinforcement Learning |
|authors| Soumyendu SarkarAshwin Ramesh BabuSajad MousaviVineet GundechaAvisek NaugSahand Ghorbanpour
|links| http://dx.doi.org/10.1609/aaai.v38i21.30579 |
|updated| 2024-03-27 20:07:39 UTC |
|summary| We present a generic Reinforcement Learning RL framework optimized forcrafting adversarial attacks on different model types spanning from ECG signalanalysis 1D image classification 2D and video classification 3D. Theframework focuses on identifying sensitive regions and inducingmisclassifications with minimal distortions and various distortion types. Thenovel RL method outperforms state-of-the-art methods for all threeapplications proving its efficiency. Our RL approach produces superiorlocalization masks enhancing interpretability for image classification and ECGanalysis models. For applications such as ECG analysis our platform highlightscritical ECG segments for clinicians while ensuring resilience againstprevalent distortions. This comprehensive tool aims to bolster both resiliencewith adversarial training and transparency across varied applications and datatypes. |


| Item |Content|
| --- |---|
|idx| 2403.18591v1 |
|title| Safety Verification of Wait-Only Non-Blocking Broadcast Protocols |
|authors| Lucie GuillouArnaud SangnierNathalie Sznajder
|links| http://arxiv.org/abs/2403.18591v1 |
|updated| 2024-03-27 14:17:33 UTC |
|summary| We study networks of processes that all execute the same finite protocol andcommunicate synchronously in two different ways: a process can broadcast onemessage to all other processes or send it to at most one other process. In bothcases if no process can receive the message it will still be sent. Weestablish a precise complexity class for two coverability problems with aparameterised number of processes: the state coverability problem and theconfiguration coverability problem. It is already known that these problems areAckermann-hard but decidable in the general case. We show that when theprotocol is Wait-Only i.e. it has no state from which a process can send andreceive messages the complexity drops to P and PSPACE respectively. |


