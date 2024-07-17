# cs.CL 

| Item |Content|
| --- |---|
|idx| 2407.11969v1 |
|title| Does Refusal Training in LLMs Generalize to the Past Tense? |
|authors| Maksym AndriushchenkoNicolas Flammarion
|links| http://arxiv.org/abs/2407.11969v1 |
|updated| 2024-07-16 17:59:55 UTC |
|summary| Refusal training is widely used to prevent LLMs from generating harmfulundesirable or illegal outputs. We reveal a curious generalization gap in thecurrent refusal training approaches: simply reformulating a harmful request inthe past tense e.g. How to make a Molotov cocktail to How did people makea Molotov cocktail is often sufficient to jailbreak many state-of-the-artLLMs. We systematically evaluate this method on Llama-3 8B GPT-3.5 TurboGemma-2 9B Phi-3-Mini GPT-4o and R2D2 models using GPT-3.5 Turbo as areformulation model. For example the success rate of this simple attack onGPT-4o increases from 1 using direct requests to 88 using 20 past tensereformulation attempts on harmful requests from JailbreakBench with GPT-4 as ajailbreak judge. Interestingly we also find that reformulations in the futuretense are less effective suggesting that refusal guardrails tend to considerpast historical questions more benign than hypothetical future questions.Moreover our experiments on fine-tuning GPT-3.5 Turbo show that defendingagainst past reformulations is feasible when past tense examples are explicitlyincluded in the fine-tuning data. Overall our findings highlight that thewidely used alignment techniques -- such as SFT RLHF and adversarial training-- employed to align the studied models can be brittle and do not alwaysgeneralize as intended. We provide code and jailbreak artifacts athttps://github.com/tml-epfl/llm-past-tense. |


| Item |Content|
| --- |---|
|idx| 2407.11963v1 |
|title| NeedleBench: Can LLMs Do Retrieval and Reasoning in 1 Million Context Window? |
|authors| Mo LiSongyang ZhangYunxin LiuKai Chen
|links| http://arxiv.org/abs/2407.11963v1 |
|updated| 2024-07-16 17:59:06 UTC |
|summary| In evaluating the long-context capabilities of large language models LLMsidentifying content relevant to a users query from original long documents isa crucial prerequisite for any LLM to answer questions based on long text. Wepresent NeedleBench a framework consisting of a series of progressively morechallenging tasks for assessing bilingual long-context capabilities spanningmultiple length intervals 4k 8k 32k 128k 200k 1000k and beyond anddifferent depth ranges allowing the strategic insertion of critical datapoints in different text depth zones to rigorously test the retrieval andreasoning capabilities of models in diverse contexts. We use the NeedleBenchframework to assess how well the leading open-source models can identify keyinformation relevant to the question and apply that information to reasoning inbilingual long texts. Furthermore we propose the Ancestral Trace ChallengeATC to mimic the complexity of logical reasoning challenges that are likelyto be present in real-world long-context tasks providing a simple method forevaluating LLMs in dealing with complex long-context situations. Our resultssuggest that current LLMs have significant room for improvement in practicallong-context applications as they struggle with the complexity of logicalreasoning challenges that are likely to be present in real-world long-contexttasks. All codes and resources are available at OpenCompass:https://github.com/open-compass/opencompass. |


| Item |Content|
| --- |---|
|idx| 2407.11948v1 |
|title| Rethinking Transformer-based Multi-document Summarization: An Empirical Investigation |
|authors| Congbo MaWei Emma ZhangDileepa PitawelaHaojie ZhuangYanfeng Shu
|links| http://arxiv.org/abs/2407.11948v1 |
|updated| 2024-07-16 17:42:37 UTC |
|summary| The utilization of Transformer-based models prospers the growth ofmulti-document summarization MDS. Given the huge impact and widespreadadoption of Transformer-based models in various natural language processingtasks investigating their performance and behaviors in the context of MDSbecomes crucial for advancing the field and enhancing the quality of summary.To thoroughly examine the behaviours of Transformer-based MDS models thispaper presents five empirical studies on 1 measuring the impact of documentboundary separators quantitatively 2 exploring the effectiveness ofdifferent mainstream Transformer structures 3 examining the sensitivity ofthe encoder and decoder 4 discussing different training strategies and 5discovering the repetition in a summary generation. The experimental results onprevalent MDS datasets and eleven evaluation metrics show the influence ofdocument boundary separators the granularity of different level features anddifferent model training strategies. The results also reveal that the decoderexhibits greater sensitivity to noises compared to the encoder. Thisunderscores the important role played by the decoder suggesting a potentialdirection for future research in MDS. Furthermore the experimental resultsindicate that the repetition problem in the generated summaries hascorrelations with the high uncertainty scores. |


| Item |Content|
| --- |---|
|idx| 2407.11930v1 |
|title| Fine-grained Hallucination Detection and Mitigation in Long-form Question Answering |
|authors| Rachneet SachdevaYixiao SongMohit IyyerIryna Gurevych
|links| http://arxiv.org/abs/2407.11930v1 |
|updated| 2024-07-16 17:23:16 UTC |
|summary| Long-form question answering LFQA aims to provide thorough and in-depthanswers to complex questions enhancing comprehension. However such detailedresponses are prone to hallucinations and factual inconsistencies challengingtheir faithful evaluation. This work introduces HaluQuestQA the firsthallucination dataset with localized error annotations for human-written andmodel-generated LFQA answers. HaluQuestQA comprises 698 QA pairs with 4.7kspan-level error annotations for five different error types by expertannotators along with preference judgments. Using our collected data wethoroughly analyze the shortcomings of long-form answers and find that theylack comprehensiveness and provide unhelpful references. We train an automaticfeedback model on this dataset that predicts error spans with incompleteinformation and provides associated explanations. Finally we propose aprompt-based approach Error-informed refinement that uses signals from thelearned feedback model to refine generated answers which we show reduceshallucination and improves answer quality. Furthermore humans find answersgenerated by our approach comprehensive and highly prefer them 84 over thebaseline answers. |


| Item |Content|
| --- |---|
|idx| 2407.11919v1 |
|title| What's Wrong? Refining Meeting Summaries with LLM Feedback |
|authors| Frederic KirsteinTerry RuasBela Gipp
|links| http://arxiv.org/abs/2407.11919v1 |
|updated| 2024-07-16 17:10:16 UTC |
|summary| Meeting summarization has become a critical task since digital encountershave become a common practice. Large language models LLMs show greatpotential in summarization offering enhanced coherence and contextunderstanding compared to traditional methods. However they still struggle tomaintain relevance and avoid hallucination. We introduce a multi-LLM correctionapproach for meeting summarization using a two-phase process that mimics thehuman review process: mistake identification and summary refinement. We releaseQMSum Mistake a dataset of 200 automatically generated meeting summariesannotated by humans on nine error types including structural omission andirrelevance errors. Our experiments show that these errors can be identifiedwith high accuracy by an LLM. We transform identified mistakes into actionablefeedback to improve the quality of a given summary measured by relevanceinformativeness conciseness and coherence. This post-hoc refinementeffectively improves summary quality by leveraging multiple LLMs to validateoutput quality. Our multi-LLM approach for meeting summarization showspotential for similar complex text generation tasks requiring robustnessaction planning and discussion towards a goal. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2407.11969v1 |
|title| Does Refusal Training in LLMs Generalize to the Past Tense? |
|authors| Maksym AndriushchenkoNicolas Flammarion
|links| http://arxiv.org/abs/2407.11969v1 |
|updated| 2024-07-16 17:59:55 UTC |
|summary| Refusal training is widely used to prevent LLMs from generating harmfulundesirable or illegal outputs. We reveal a curious generalization gap in thecurrent refusal training approaches: simply reformulating a harmful request inthe past tense e.g. How to make a Molotov cocktail to How did people makea Molotov cocktail is often sufficient to jailbreak many state-of-the-artLLMs. We systematically evaluate this method on Llama-3 8B GPT-3.5 TurboGemma-2 9B Phi-3-Mini GPT-4o and R2D2 models using GPT-3.5 Turbo as areformulation model. For example the success rate of this simple attack onGPT-4o increases from 1 using direct requests to 88 using 20 past tensereformulation attempts on harmful requests from JailbreakBench with GPT-4 as ajailbreak judge. Interestingly we also find that reformulations in the futuretense are less effective suggesting that refusal guardrails tend to considerpast historical questions more benign than hypothetical future questions.Moreover our experiments on fine-tuning GPT-3.5 Turbo show that defendingagainst past reformulations is feasible when past tense examples are explicitlyincluded in the fine-tuning data. Overall our findings highlight that thewidely used alignment techniques -- such as SFT RLHF and adversarial training-- employed to align the studied models can be brittle and do not alwaysgeneralize as intended. We provide code and jailbreak artifacts athttps://github.com/tml-epfl/llm-past-tense. |


| Item |Content|
| --- |---|
|idx| 2407.11966v1 |
|title| Efficient Training with Denoised Neural Weights |
|authors| Yifan GongZheng ZhanYanyu LiYerlan IdelbayevAndrey ZharkovKfir AbermanSergey TulyakovYanzhi WangJian Ren
|links| http://arxiv.org/abs/2407.11966v1 |
|updated| 2024-07-16 17:59:42 UTC |
|summary| Good weight initialization serves as an effective measure to reduce thetraining cost of a deep neural network DNN model. The choice of how toinitialize parameters is challenging and may require manual tuning which canbe time-consuming and prone to human error. To overcome such limitations thiswork takes a novel step towards building a weight generator to synthesize theneural weights for initialization. We use the image-to-image translation taskwith generative adversarial networks GANs as an example due to the ease ofcollecting model weights spanning a wide range. Specifically we first collecta dataset with various image editing concepts and their corresponding trainedweights which are later used for the training of the weight generator. Toaddress the different characteristics among layers and the substantial numberof weights to be predicted we divide the weights into equal-sized blocks andassign each block an index. Subsequently a diffusion model is trained withsuch a dataset using both text conditions of the concept and the block indexes.By initializing the image translation model with the denoised weights predictedby our diffusion model the training requires only 43.3 seconds. Compared totraining from scratch i.e. Pix2pix we achieve a 15x training timeacceleration for a new concept while obtaining even better image generationquality. |


| Item |Content|
| --- |---|
|idx| 2407.11962v1 |
|title| Motion-Oriented Compositional Neural Radiance Fields for Monocular Dynamic Human Modeling |
|authors| Jaehyeok KimDongyoon WeeDan Xu
|links| http://arxiv.org/abs/2407.11962v1 |
|updated| 2024-07-16 17:59:01 UTC |
|summary| This paper introduces Motion-oriented Compositional Neural Radiance FieldsMoCo-NeRF a framework designed to perform free-viewpoint rendering ofmonocular human videos via novel non-rigid motion modeling approach. In thecontext of dynamic clothed humans complex cloth dynamics generate non-rigidmotions that are intrinsically distinct from skeletal articulations andcritically important for the rendering quality. The conventional approachmodels non-rigid motions as spatial 3D deviations in addition to skeletaltransformations. However it is either time-consuming or challenging to achieveoptimal quality due to its high learning complexity without a directsupervision. To target this problem we propose a novel approach of modelingnon-rigid motions as radiance residual fields to benefit from more direct colorsupervision in the rendering and utilize the rigid radiance fields as a priorto reduce the complexity of the learning process. Our approach utilizes asingle multiresolution hash encoding MHE to concurrently learn the canonicalT-pose representation from rigid skeletal motions and the radiance residualfield for non-rigid motions. Additionally to further improve both trainingefficiency and usability we extend MoCo-NeRF to support simultaneous trainingof multiple subjects within a single framework thanks to our effective designfor modeling non-rigid motions. This scalability is achieved through theintegration of a global MHE and learnable identity codes in addition tomultiple local MHEs. We present extensive results on ZJU-MoCap and MonoCapclearly demonstrating state-of-the-art performance in both single- andmulti-subject settings. The code and model will be made publicly available atthe project page: https://stevejaehyeok.github.io/publications/moco-nerf. |


| Item |Content|
| --- |---|
|idx| 2407.11948v1 |
|title| Rethinking Transformer-based Multi-document Summarization: An Empirical Investigation |
|authors| Congbo MaWei Emma ZhangDileepa PitawelaHaojie ZhuangYanfeng Shu
|links| http://arxiv.org/abs/2407.11948v1 |
|updated| 2024-07-16 17:42:37 UTC |
|summary| The utilization of Transformer-based models prospers the growth ofmulti-document summarization MDS. Given the huge impact and widespreadadoption of Transformer-based models in various natural language processingtasks investigating their performance and behaviors in the context of MDSbecomes crucial for advancing the field and enhancing the quality of summary.To thoroughly examine the behaviours of Transformer-based MDS models thispaper presents five empirical studies on 1 measuring the impact of documentboundary separators quantitatively 2 exploring the effectiveness ofdifferent mainstream Transformer structures 3 examining the sensitivity ofthe encoder and decoder 4 discussing different training strategies and 5discovering the repetition in a summary generation. The experimental results onprevalent MDS datasets and eleven evaluation metrics show the influence ofdocument boundary separators the granularity of different level features anddifferent model training strategies. The results also reveal that the decoderexhibits greater sensitivity to noises compared to the encoder. Thisunderscores the important role played by the decoder suggesting a potentialdirection for future research in MDS. Furthermore the experimental resultsindicate that the repetition problem in the generated summaries hascorrelations with the high uncertainty scores. |


| Item |Content|
| --- |---|
|idx| 2407.11928v1 |
|title| Tackling Oversmoothing in GNN via Graph Sparsification: A Truss-based Approach |
|authors| Tanvir HossainKhaled Mohammed SaifuddinMuhammad Ifte Khairul IslamFarhan TanvirEsra Akbas
|links| http://arxiv.org/abs/2407.11928v1 |
|updated| 2024-07-16 17:21:36 UTC |
|summary| Graph Neural Network GNN achieves great success for node-level andgraph-level tasks via encoding meaningful topological structures of networks invarious domains ranging from social to biological networks. However repeatedaggregation operations lead to excessive mixing of node representationsparticularly in dense regions with multiple GNN layers resulting in nearlyindistinguishable embeddings. This phenomenon leads to the oversmoothingproblem that hampers downstream graph analytics tasks. To overcome this issuewe propose a novel and flexible truss-based graph sparsification model thatprunes edges from dense regions of the graph. Pruning redundant edges in denseregions helps to prevent the aggregation of excessive neighborhood informationduring hierarchical message passing and pooling in GNN models. We then utilizeour sparsification model in the state-of-the-art baseline GNNs and poolingmodels such as GIN SAGPool GMT DiffPool MinCutPool HGP-SL DMonPool andAdamGNN. Extensive experiments on different real-world datasets show that ourmodel significantly improves the performance of the baseline GNN models in thegraph classification task. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2407.11969v1 |
|title| Does Refusal Training in LLMs Generalize to the Past Tense? |
|authors| Maksym AndriushchenkoNicolas Flammarion
|links| http://arxiv.org/abs/2407.11969v1 |
|updated| 2024-07-16 17:59:55 UTC |
|summary| Refusal training is widely used to prevent LLMs from generating harmfulundesirable or illegal outputs. We reveal a curious generalization gap in thecurrent refusal training approaches: simply reformulating a harmful request inthe past tense e.g. How to make a Molotov cocktail to How did people makea Molotov cocktail is often sufficient to jailbreak many state-of-the-artLLMs. We systematically evaluate this method on Llama-3 8B GPT-3.5 TurboGemma-2 9B Phi-3-Mini GPT-4o and R2D2 models using GPT-3.5 Turbo as areformulation model. For example the success rate of this simple attack onGPT-4o increases from 1 using direct requests to 88 using 20 past tensereformulation attempts on harmful requests from JailbreakBench with GPT-4 as ajailbreak judge. Interestingly we also find that reformulations in the futuretense are less effective suggesting that refusal guardrails tend to considerpast historical questions more benign than hypothetical future questions.Moreover our experiments on fine-tuning GPT-3.5 Turbo show that defendingagainst past reformulations is feasible when past tense examples are explicitlyincluded in the fine-tuning data. Overall our findings highlight that thewidely used alignment techniques -- such as SFT RLHF and adversarial training-- employed to align the studied models can be brittle and do not alwaysgeneralize as intended. We provide code and jailbreak artifacts athttps://github.com/tml-epfl/llm-past-tense. |


| Item |Content|
| --- |---|
|idx| 2407.11966v1 |
|title| Efficient Training with Denoised Neural Weights |
|authors| Yifan GongZheng ZhanYanyu LiYerlan IdelbayevAndrey ZharkovKfir AbermanSergey TulyakovYanzhi WangJian Ren
|links| http://arxiv.org/abs/2407.11966v1 |
|updated| 2024-07-16 17:59:42 UTC |
|summary| Good weight initialization serves as an effective measure to reduce thetraining cost of a deep neural network DNN model. The choice of how toinitialize parameters is challenging and may require manual tuning which canbe time-consuming and prone to human error. To overcome such limitations thiswork takes a novel step towards building a weight generator to synthesize theneural weights for initialization. We use the image-to-image translation taskwith generative adversarial networks GANs as an example due to the ease ofcollecting model weights spanning a wide range. Specifically we first collecta dataset with various image editing concepts and their corresponding trainedweights which are later used for the training of the weight generator. Toaddress the different characteristics among layers and the substantial numberof weights to be predicted we divide the weights into equal-sized blocks andassign each block an index. Subsequently a diffusion model is trained withsuch a dataset using both text conditions of the concept and the block indexes.By initializing the image translation model with the denoised weights predictedby our diffusion model the training requires only 43.3 seconds. Compared totraining from scratch i.e. Pix2pix we achieve a 15x training timeacceleration for a new concept while obtaining even better image generationquality. |


| Item |Content|
| --- |---|
|idx| 2407.11962v1 |
|title| Motion-Oriented Compositional Neural Radiance Fields for Monocular Dynamic Human Modeling |
|authors| Jaehyeok KimDongyoon WeeDan Xu
|links| http://arxiv.org/abs/2407.11962v1 |
|updated| 2024-07-16 17:59:01 UTC |
|summary| This paper introduces Motion-oriented Compositional Neural Radiance FieldsMoCo-NeRF a framework designed to perform free-viewpoint rendering ofmonocular human videos via novel non-rigid motion modeling approach. In thecontext of dynamic clothed humans complex cloth dynamics generate non-rigidmotions that are intrinsically distinct from skeletal articulations andcritically important for the rendering quality. The conventional approachmodels non-rigid motions as spatial 3D deviations in addition to skeletaltransformations. However it is either time-consuming or challenging to achieveoptimal quality due to its high learning complexity without a directsupervision. To target this problem we propose a novel approach of modelingnon-rigid motions as radiance residual fields to benefit from more direct colorsupervision in the rendering and utilize the rigid radiance fields as a priorto reduce the complexity of the learning process. Our approach utilizes asingle multiresolution hash encoding MHE to concurrently learn the canonicalT-pose representation from rigid skeletal motions and the radiance residualfield for non-rigid motions. Additionally to further improve both trainingefficiency and usability we extend MoCo-NeRF to support simultaneous trainingof multiple subjects within a single framework thanks to our effective designfor modeling non-rigid motions. This scalability is achieved through theintegration of a global MHE and learnable identity codes in addition tomultiple local MHEs. We present extensive results on ZJU-MoCap and MonoCapclearly demonstrating state-of-the-art performance in both single- andmulti-subject settings. The code and model will be made publicly available atthe project page: https://stevejaehyeok.github.io/publications/moco-nerf. |


| Item |Content|
| --- |---|
|idx| 2407.11942v1 |
|title| Context-Guided Diffusion for Out-of-Distribution Molecular and Protein Design |
|authors| Leo KlarnerTim G. J. RudnerGarrett M. MorrisCharlotte M. DeaneYee Whye Teh
|links| http://arxiv.org/abs/2407.11942v1 |
|updated| 2024-07-16 17:34:00 UTC |
|summary| Generative models have the potential to accelerate key steps in the discoveryof novel molecular therapeutics and materials. Diffusion models have recentlyemerged as a powerful approach excelling at unconditional sample generationand with data-driven guidance conditional generation within their trainingdomain. Reliably sampling from high-value regions beyond the training datahowever remains an open challenge -- with current methods predominantlyfocusing on modifying the diffusion process itself. In this paper we developcontext-guided diffusion CGD a simple plug-and-play method that leveragesunlabeled data and smoothness constraints to improve the out-of-distributiongeneralization of guided diffusion models. We demonstrate that this approachleads to substantial performance gains across various settings includingcontinuous discrete and graph-structured diffusion processes withapplications across drug discovery materials science and protein design. |


| Item |Content|
| --- |---|
|idx| 2407.11933v1 |
|title| Fairly Accurate: Optimizing Accuracy Parity in Fair Target-Group Detection |
|authors| Soumyajit GuptaVenelin KovatchevMaria De-ArteagaMatthew Lease
|links| http://arxiv.org/abs/2407.11933v1 |
|updated| 2024-07-16 17:23:41 UTC |
|summary| In algorithmic toxicity detection pipelines it is important to identifywhich demographic groups are the subject of a post a task commonly known astextittarget group detection. While accurate detection is clearlyimportant we further advocate a fairness objective: to provide equalprotection to all groups who may be targeted. To this end we adopttextitAccuracy Parity AP -- balanced detection accuracy across groups --as our fairness objective. However in order to align model training with ourAP fairness objective we require an equivalent loss function. Moreover forgradient-based models such as neural networks this loss function needs to bedifferentiable. Because no such loss function exists today for AP we proposeemphGroup Accuracy Parity GAP: the first differentiable loss functionhaving a one-on-one mapping to AP. We empirically show that GAP addressesdisparate impact on groups for target detection. Furthermore because a singlepost often targets multiple groups in practice we also provide a mathematicalextension of GAP to larger multi-group settings something typically requiringheuristics in prior work. Our findings show that by optimizing AP GAP bettermitigates bias in comparison with other commonly employed loss functions. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2407.11966v1 |
|title| Efficient Training with Denoised Neural Weights |
|authors| Yifan GongZheng ZhanYanyu LiYerlan IdelbayevAndrey ZharkovKfir AbermanSergey TulyakovYanzhi WangJian Ren
|links| http://arxiv.org/abs/2407.11966v1 |
|updated| 2024-07-16 17:59:42 UTC |
|summary| Good weight initialization serves as an effective measure to reduce thetraining cost of a deep neural network DNN model. The choice of how toinitialize parameters is challenging and may require manual tuning which canbe time-consuming and prone to human error. To overcome such limitations thiswork takes a novel step towards building a weight generator to synthesize theneural weights for initialization. We use the image-to-image translation taskwith generative adversarial networks GANs as an example due to the ease ofcollecting model weights spanning a wide range. Specifically we first collecta dataset with various image editing concepts and their corresponding trainedweights which are later used for the training of the weight generator. Toaddress the different characteristics among layers and the substantial numberof weights to be predicted we divide the weights into equal-sized blocks andassign each block an index. Subsequently a diffusion model is trained withsuch a dataset using both text conditions of the concept and the block indexes.By initializing the image translation model with the denoised weights predictedby our diffusion model the training requires only 43.3 seconds. Compared totraining from scratch i.e. Pix2pix we achieve a 15x training timeacceleration for a new concept while obtaining even better image generationquality. |


| Item |Content|
| --- |---|
|idx| 2407.11965v1 |
|title| UrbanWorld: An Urban World Model for 3D City Generation |
|authors| Yu ShangJiansheng ChenHangyu FanJingtao DingJie FengYong Li
|links| http://arxiv.org/abs/2407.11965v1 |
|updated| 2024-07-16 17:59:29 UTC |
|summary| Cities as the most fundamental environment of human life encompass diversephysical elements such as buildings roads and vegetation with complexinterconnection. Crafting realistic interactive 3D urban environments plays acrucial role in constructing AI agents capable of perceiving decision-makingand acting like humans in real-world environments. However creatinghigh-fidelity 3D urban environments usually entails extensive manual labor fromdesigners involving intricate detailing and accurate representation of complexurban features. Therefore how to accomplish this in an automatical way remainsa longstanding challenge. Toward this problem we propose UrbanWorld the firstgenerative urban world model that can automatically create a customizedrealistic and interactive 3D urban world with flexible control conditions.UrbanWorld incorporates four key stages in the automatical crafting pipeline:3D layout generation from openly accessible OSM data urban scene planning anddesigning with a powerful urban multimodal large language model Urban MLLMcontrollable urban asset rendering with advanced 3D diffusion techniques andfinally the MLLM-assisted scene refinement. The crafted high-fidelity 3D urbanenvironments enable realistic feedback and interactions for general AI andmachine perceptual systems in simulations. We are working on contributingUrbanWorld as an open-source and versatile platform for evaluating andimproving AI abilities in perception decision-making and interaction inrealistic urban environments. |


| Item |Content|
| --- |---|
|idx| 2407.11962v1 |
|title| Motion-Oriented Compositional Neural Radiance Fields for Monocular Dynamic Human Modeling |
|authors| Jaehyeok KimDongyoon WeeDan Xu
|links| http://arxiv.org/abs/2407.11962v1 |
|updated| 2024-07-16 17:59:01 UTC |
|summary| This paper introduces Motion-oriented Compositional Neural Radiance FieldsMoCo-NeRF a framework designed to perform free-viewpoint rendering ofmonocular human videos via novel non-rigid motion modeling approach. In thecontext of dynamic clothed humans complex cloth dynamics generate non-rigidmotions that are intrinsically distinct from skeletal articulations andcritically important for the rendering quality. The conventional approachmodels non-rigid motions as spatial 3D deviations in addition to skeletaltransformations. However it is either time-consuming or challenging to achieveoptimal quality due to its high learning complexity without a directsupervision. To target this problem we propose a novel approach of modelingnon-rigid motions as radiance residual fields to benefit from more direct colorsupervision in the rendering and utilize the rigid radiance fields as a priorto reduce the complexity of the learning process. Our approach utilizes asingle multiresolution hash encoding MHE to concurrently learn the canonicalT-pose representation from rigid skeletal motions and the radiance residualfield for non-rigid motions. Additionally to further improve both trainingefficiency and usability we extend MoCo-NeRF to support simultaneous trainingof multiple subjects within a single framework thanks to our effective designfor modeling non-rigid motions. This scalability is achieved through theintegration of a global MHE and learnable identity codes in addition tomultiple local MHEs. We present extensive results on ZJU-MoCap and MonoCapclearly demonstrating state-of-the-art performance in both single- andmulti-subject settings. The code and model will be made publicly available atthe project page: https://stevejaehyeok.github.io/publications/moco-nerf. |


| Item |Content|
| --- |---|
|idx| 2407.11954v1 |
|title| Gated Temporal Diffusion for Stochastic Long-Term Dense Anticipation |
|authors| Olga ZatsarynnaEmad BahramiYazan Abu FarhaGianpiero FrancescaJuergen Gall
|links| http://arxiv.org/abs/2407.11954v1 |
|updated| 2024-07-16 17:48:05 UTC |
|summary| Long-term action anticipation has become an important task for manyapplications such as autonomous driving and human-robot interaction. Unlikeshort-term anticipation predicting more actions into the future imposes a realchallenge with the increasing uncertainty in longer horizons. While there hasbeen a significant progress in predicting more actions into the future most ofthe proposed methods address the task in a deterministic setup and ignore theunderlying uncertainty. In this paper we propose a novel Gated TemporalDiffusion GTD network that models the uncertainty of both the observation andthe future predictions. As generator we introduce a Gated Anticipation NetworkGTAN to model both observed and unobserved frames of a video in a mutualrepresentation. On the one hand using a mutual representation for past andfuture allows us to jointly model ambiguities in the observation and futurewhile on the other hand GTAN can by design treat the observed and unobservedparts differently and steer the information flow between them. Our modelachieves state-of-the-art results on the Breakfast Assembly101 and 50Saladsdatasets in both stochastic and deterministic settings. Code:https://github.com/olga-zats/GTDA . |


| Item |Content|
| --- |---|
|idx| 2407.11950v1 |
|title| Temporally Consistent Stereo Matching |
|authors| Jiaxi ZengChengtang YaoYuwei WuYunde Jia
|links| http://arxiv.org/abs/2407.11950v1 |
|updated| 2024-07-16 17:44:34 UTC |
|summary| Stereo matching provides depth estimation from binocular images fordownstream applications. These applications mostly take video streams as inputand require temporally consistent depth maps. However existing methods mainlyfocus on the estimation at the single-frame level. This commonly leads totemporally inconsistent results especially in ill-posed regions. In thispaper we aim to leverage temporal information to improve the temporalconsistency accuracy and efficiency of stereo matching. To achieve this weformulate video stereo matching as a process of temporal disparity completionfollowed by continuous iterative refinements. Specifically we first projectthe disparity of the previous timestamp to the current viewpoint obtaining asemi-dense disparity map. Then we complete this map through a disparitycompletion module to obtain a well-initialized disparity map. The statefeatures from the current completion module and from the past refinement arefused together providing a temporally coherent state for subsequentrefinement. Based on this coherent state we introduce a dual-space refinementmodule to iteratively refine the initialized result in both disparity anddisparity gradient spaces improving estimations in ill-posed regions.Extensive experiments demonstrate that our method effectively alleviatestemporal inconsistency while enhancing both accuracy and efficiency. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2407.11942v1 |
|title| Context-Guided Diffusion for Out-of-Distribution Molecular and Protein Design |
|authors| Leo KlarnerTim G. J. RudnerGarrett M. MorrisCharlotte M. DeaneYee Whye Teh
|links| http://arxiv.org/abs/2407.11942v1 |
|updated| 2024-07-16 17:34:00 UTC |
|summary| Generative models have the potential to accelerate key steps in the discoveryof novel molecular therapeutics and materials. Diffusion models have recentlyemerged as a powerful approach excelling at unconditional sample generationand with data-driven guidance conditional generation within their trainingdomain. Reliably sampling from high-value regions beyond the training datahowever remains an open challenge -- with current methods predominantlyfocusing on modifying the diffusion process itself. In this paper we developcontext-guided diffusion CGD a simple plug-and-play method that leveragesunlabeled data and smoothness constraints to improve the out-of-distributiongeneralization of guided diffusion models. We demonstrate that this approachleads to substantial performance gains across various settings includingcontinuous discrete and graph-structured diffusion processes withapplications across drug discovery materials science and protein design. |


| Item |Content|
| --- |---|
|idx| 2407.11932v1 |
|title| Impossibility of latent inner product recovery via rate distortion |
|authors| Cheng MaoShenduo Zhang
|links| http://arxiv.org/abs/2407.11932v1 |
|updated| 2024-07-16 17:23:29 UTC |
|summary| In this largely expository note we present an impossibility result for innerproduct recovery in a random geometric graph or latent space model using therate-distortion theory. More precisely suppose that we observe a graph A onn vertices with average edge density p generated from Gaussian or sphericallatent locations z_1 dots z_n in mathbbRd associated with the nvertices. It is of interest to estimate the inner products langle z_i z_jrangle which represent the geometry of the latent points. We prove that it isimpossible to recover the inner products if d gtrsim n hp where hp isthe binary entropy function. This matches the condition required for positiveresults on inner product recovery in the literature. The proof follows thewell-established rate-distortion theory with the main technical ingredientbeing a lower bound on the rate-distortion function of the Wishart distributionwhich is interesting in its own right. |


| Item |Content|
| --- |---|
|idx| 2407.11927v1 |
|title| Bayesian Causal Forests for Longitudinal Data: Assessing the Impact of Part-Time Work on Growth in High School Mathematics Achievement |
|authors| Nathan McJamesAnn O'SheaAndrew Parnell
|links| http://arxiv.org/abs/2407.11927v1 |
|updated| 2024-07-16 17:18:33 UTC |
|summary| Modelling growth in student achievement is a significant challenge in thefield of education. Understanding how interventions or experiences such aspart-time work can influence this growth is also important. Traditional methodslike difference-in-differences are effective for estimating causal effects fromlongitudinal data. Meanwhile Bayesian non-parametric methods have recentlybecome popular for estimating causal effects from single time pointobservational studies. However there remains a scarcity of methods capable ofcombining the strengths of these two approaches to flexibly estimateheterogeneous causal effects from longitudinal data. Motivated by two waves ofdata from the High School Longitudinal Study the NCES most recentlongitudinal study which tracks a representative sample of over 20000 studentsin the US our study introduces a longitudinal extension of Bayesian CausalForests. This model allows for the flexible identification of both individualgrowth in mathematical ability and the effects of participation in part-timework. Simulation studies demonstrate the predictive performance and reliableuncertainty quantification of the proposed model. Results reveal the negativeimpact of part time work for most students but hint at potential benefits forthose students with an initially low sense of school belonging. Clear signs ofa widening achievement gap between students with high and low academicachievement are also identified. Potential policy implications are discussedalong with promising areas for future research. |


| Item |Content|
| --- |---|
|idx| 2407.11917v1 |
|title| Global Optimisation of Black-Box Functions with Generative Models in the Wasserstein Space |
|authors| Tigran RamazyanMikhail HushchynDenis Derkach
|links| http://arxiv.org/abs/2407.11917v1 |
|updated| 2024-07-16 17:09:47 UTC |
|summary| We propose a new uncertainty estimator for gradient-free optimisation ofblack-box simulators using deep generative surrogate models. Optimisation ofthese simulators is especially challenging for stochastic simulators and higherdimensions. To address these issues we utilise a deep generative surrogateapproach to model the black box response for the entire parameter space. Wethen leverage this knowledge to estimate the proposed uncertainty based on theWasserstein distance - the Wasserstein uncertainty. This approach is employedin a posterior agnostic gradient-free optimisation algorithm that minimisesregret over the entire parameter space. A series of tests were conducted todemonstrate that our method is more robust to the shape of both the black boxfunction and the stochastic response of the black box than state-of-the-artmethods such as efficient global optimisation with a deep Gaussian processsurrogate. |


| Item |Content|
| --- |---|
|idx| 2407.11901v1 |
|title| Combining Wasserstein-1 and Wasserstein-2 proximals: robust manifold learning via well-posed generative flows |
|authors| Hyemin GuMarkos A. KatsoulakisLuc Rey-BelletBenjamin J. Zhang
|links| http://arxiv.org/abs/2407.11901v1 |
|updated| 2024-07-16 16:34:31 UTC |
|summary| We formulate well-posed continuous-time generative flows for learningdistributions that are supported on low-dimensional manifolds throughWasserstein proximal regularizations of f-divergences. Wasserstein-1 proximaloperators regularize f-divergences so that singular distributions can becompared. Meanwhile Wasserstein-2 proximal operators regularize the paths ofthe generative flows by adding an optimal transport cost i.e. a kineticenergy penalization. Via mean-field game theory we show that the combinationof the two proximals is critical for formulating well-posed generative flows.Generative flows can be analyzed through optimality conditions of a mean-fieldgame MFG a system of a backward Hamilton-Jacobi HJ and a forwardcontinuity partial differential equations PDEs whose solution characterizesthe optimal generative flow. For learning distributions that are supported onlow-dimensional manifolds the MFG theory shows that the Wasserstein-1proximal which addresses the HJ terminal condition and the Wasserstein-2proximal which addresses the HJ dynamics are both necessary for thecorresponding backward-forward PDE system to be well-defined and have a uniquesolution with provably linear flow trajectories. This implies that thecorresponding generative flow is also unique and can therefore be learned in arobust manner even for learning high-dimensional distributions supported onlow-dimensional manifolds. The generative flows are learned through adversarialtraining of continuous-time flows which bypasses the need for reversesimulation. We demonstrate the efficacy of our approach for generatinghigh-dimensional images without the need to resort to autoencoders orspecialized architectures. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2407.11837v1 |
|title| The Patchkeeper: An Integrated Wearable Electronic Stethoscope with Multiple Sensors |
|authors| Hongwei LiZoran RadivojevicMichael S. Eggleston
|links| http://arxiv.org/abs/2407.11837v1 |
|updated| 2024-07-16 15:22:10 UTC |
|summary| Many parts of human body generate internal sound during biological processeswhich are rich sources of information for understanding health and wellbeing.Despite a long history of development and usage of stethoscopes there is stilla lack of proper tools for recording internal body sound together withcomplementary sensors for long term monitoring. In this paper we show ourdevelopment of a wearable electronic stethoscope coined Patchkeeper PK thatcan be used for internal body sound recording over long periods of time.Patchkeeper also integrates several state-of-the-art biological sensorsincluding electrocardiogram ECG photoplethysmography PPG and inertialmeasurement unit IMU sensors. As a wearable device Patchkeeper can be placedon various parts of the body to collect sound from particular organs includingheart lung stomach and joints etc. We show in this paper that several vitalsignals can be recorded simultaneously with high quality. As Patchkeeper can beoperated directly by the user e.g. without involving health careprofessionals we believe it could be a useful tool for telemedicine and remotediagnostics. |


| Item |Content|
| --- |---|
|idx| 2407.11823v1 |
|title| Harmonizing Safety and Speed: A Human-Algorithm Approach to Enhance the FDA's Medical Device Clearance Policy |
|authors| Mohammad ZhalechianSoroush SaghafianOmar Robles
|links| http://arxiv.org/abs/2407.11823v1 |
|updated| 2024-07-16 15:11:29 UTC |
|summary| The United States Food and Drug Administrations FDAs PremarketNotification 510K pathway allows manufacturers to gain approval for a medicaldevice by demonstrating its substantial equivalence to another legally marketeddevice. However the inherent ambiguity of this regulatory procedure has led tohigh recall rates for many devices cleared through this pathway. This trend hasraised significant concerns regarding the efficacy of the FDAs currentapproach prompting a reassessment of the 510K regulatory framework. In thispaper we develop a combined human-algorithm approach to assist the FDA inimproving its 510k medical device clearance process by reducing the risk ofpotential recalls and the workload imposed on the FDA. We first develop machinelearning methods to estimate the risk of recall of 510k medical devices basedon the information available at the time of submission. We then propose adata-driven clearance policy that recommends acceptance rejection or deferralto FDAs committees for in-depth evaluation. We conduct an empirical studyusing a unique large-scale dataset of over 31000 medical devices and 12000national and international manufacturers from over 65 countries that weassembled based on data sources from the FDA and Centers for Medicare andMedicaid Service CMS. A conservative evaluation of our proposed policy basedon this data shows a 38.9 improvement in the recall rate and a 43.0 reductionin the FDAs workload. Our analyses also indicate that implementing our policycould result in significant annual cost-savings ranging between 2.4 billionand 2.7 billion which highlights the value of using a holistic anddata-driven approach to improve the FDAs current 510K medical deviceevaluation pathway. |


| Item |Content|
| --- |---|
|idx| 2407.11748v1 |
|title| Ubiquitous Metadata: Design and Fabrication of Embedded Markers for Real-World Object Identification and Interaction |
|authors| Mustafa Doga Dogan
|links| http://arxiv.org/abs/2407.11748v1 |
|updated| 2024-07-16 14:14:52 UTC |
|summary| The convergence of the physical and digital realms has ushered in a new eraof immersive experiences and seamless interactions. As the boundaries betweenthe real world and virtual environments blur and result in a mixed realitythere arises a need for robust and efficient methods to connect physicalobjects with their virtual counterparts. In this thesis we present a novelapproach to bridging this gap through the design fabrication and detection ofembedded machine-readable markers.  We categorize the proposed marking approaches into three distinct categories:natural markers structural markers and internal markers. Natural markerssuch as those used in SensiCut are inherent fingerprints of objects repurposedas machine-readable identifiers while structural markers such as StructCodeand G-ID leverage the structural artifacts in objects that emerge during thefabrication process itself. Internal markers such as InfraredTag andBrightMarker are embedded inside fabricated objects using specializedmaterials. Leveraging a combination of methods from computer vision machinelearning computational imaging and material science the presented approachesoffer robust and versatile solutions for object identification tracking andinteraction.  These markers seamlessly integrated into real-world objects effectivelycommunicate an objects identity origin function and interactionfunctioning as gateways to ubiquitous metadata - a concept where metadata isembedded into physical objects similar to metadata in digital files. Acrossthe different chapters we demonstrate the applications of the presentedmethods in diverse domains including product design manufacturing retaillogistics education entertainment security and sustainability. |


| Item |Content|
| --- |---|
|idx| 2407.11671v1 |
|title| A Comparative Analysis of Interactive Reinforcement Learning Algorithms in Warehouse Robot Grid Based Environment |
|authors| Arunabh Bora
|links| http://arxiv.org/abs/2407.11671v1 |
|updated| 2024-07-16 12:41:49 UTC |
|summary| The field of warehouse robotics is currently in high demand with majortechnology and logistics companies making significant investments in theseadvanced systems. Training robots to operate in such complex environments ischallenging often requiring human supervision for adaptation and learning.Interactive reinforcement learning IRL is a key training methodology inhuman-computer interaction. This paper presents a comparative study of two IRLalgorithms: Q-learning and SARSA both trained in a virtualgrid-simulation-based warehouse environment. To maintain consistent feedbackrewards and avoid bias feedback was provided by the same individual throughoutthe study. |


| Item |Content|
| --- |---|
|idx| 2407.11625v1 |
|title| Beware of Validation by Eye: Visual Validation of Linear Trends in Scatterplots |
|authors| Daniel BraunRemco ChangMichael GleicherTatiana von Landesberger
|links| http://arxiv.org/abs/2407.11625v1 |
|updated| 2024-07-16 11:41:24 UTC |
|summary| Visual validation of regression models in scatterplots is a common practicefor assessing model quality yet its efficacy remains unquantified. Weconducted two empirical experiments to investigate individuals ability tovisually validate linear regression models linear trends and to examine theimpact of common visualization designs on validation quality. The firstexperiment showed that the level of accuracy for visual estimation of slopei.e. fitting a line to data is higher than for visual validation of slopei.e. accepting a shown line. Notably we found bias toward slopes that aretoo steep in both cases. This lead to novel insights that participantsnaturally assessed regression with orthogonal distances between the points andthe line i.e. ODR regression rather than the common vertical distances OLSregression. In the second experiment we investigated whether incorporatingcommon designs for regression visualization error lines bounding boxes andconfidence intervals would improve visual validation. Even though error linesreduced validation bias results failed to show the desired improvements inaccuracy for any design. Overall our findings suggest caution in using visualmodel validation for linear trends in scatterplots. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2407.11889v1 |
|title| Map of Elections |
|authors| Stanisław Szufa
|links| http://arxiv.org/abs/2407.11889v1 |
|updated| 2024-07-16 16:18:29 UTC |
|summary| Our main contribution is the introduction of the map of elections framework.A map of elections consists of three main elements: 1 a dataset of electionsi.e. collections of ordinal votes over given sets of candidates 2 a wayof measuring similarities between these elections and 3 a representation ofthe elections in the 2D Euclidean space as points so that the more similar twoelections are the closer are their points. In our maps we mostly focus ondatasets of synthetic elections but we also show an example of a map overreal-life ones. To measure similarities we would have preferred to use e.g.the isomorphic swap distance but this is infeasible due to its highcomputational complexity. Hence we propose polynomial-time computablepositionwise distance and use it instead. Regarding the representations in 2DEuclidean space we mostly use the Kamada-Kawai algorithm but we also show twoalternatives.  We develop the necessary theoretical results to form our maps and argueexperimentally that they are accurate and credible. Further we show howcoloring the elections in a map according to various criteria helps inanalyzing results of a number of experiments. In particular we show coloringsaccording to the scores of winning candidates or committees running times ofILP-based winner determination algorithms and approximation ratios achieved byparticular algorithms. |


| Item |Content|
| --- |---|
|idx| 2407.11592v1 |
|title| Learning to Imitate Spatial Organization in Multi-robot Systems |
|authors| Ayomide O. AgunloyeSarvapali D. RamchurnMohammad D. Soorati
|links| http://arxiv.org/abs/2407.11592v1 |
|updated| 2024-07-16 10:50:39 UTC |
|summary| Understanding collective behavior and how it evolves is important to ensurethat robot swarms can be trusted in a shared environment. One way to understandthe behavior of the swarm is through collective behavior reconstruction usingprior demonstrations. Existing approaches often require access to the swarmcontroller which may not be available. We reconstruct collective behaviors indistinct swarm scenarios involving shared environments without using swarmcontroller information. We achieve this by transforming prior demonstrationsinto features that sufficiently describe multi-agent interactions beforebehavior reconstruction with multi-agent generative adversarial imitationlearning MA-GAIL. We show that our approach outperforms existing algorithmsin all investigated swarm scenarios and can be used to observe and reconstructa swarms behavior for further analysis and testing which might be impracticalor undesirable on the original robot swarm. |


| Item |Content|
| --- |---|
|idx| 2407.11330v1 |
|title| Navigating the swarm: Deep neural networks command emergent behaviours |
|authors| Dongjo KimJeongsu LeeHo-Young Kim
|links| http://arxiv.org/abs/2407.11330v1 |
|updated| 2024-07-16 02:46:11 UTC |
|summary| Interacting individuals in complex systems often give rise to coherent motionexhibiting coordinated global structures. Such phenomena are ubiquitouslyobserved in nature from cell migration bacterial swarms animal and insectgroups and even human societies. Primary mechanisms responsible for theemergence of collective behavior have been extensively identified includinglocal alignments based on average or relative velocity non-local pairwiserepulsive-attractive interactions such as distance-based potentials interplaybetween local and non-local interactions and cognitive-based inhomogeneousinteractions. However discovering how to adapt these mechanisms to modulateemergent behaviours remains elusive. Here we demonstrate that it is possibleto generate coordinated structures in collective behavior at desired momentswith intended global patterns by fine-tuning an inter-agent interaction rule.Our strategy employs deep neural networks obeying the laws of dynamics tofind interaction rules that command desired collective structures. Thedecomposition of interaction rules into distancing and aligning forcesexpressed by polynomial series facilitates the training of neural networks topropose desired interaction models. Presented examples include altering themean radius and size of clusters in vortical swarms timing of transitions fromrandom to ordered states and continuously shifting between typical modes ofcollective motions. This strategy can even be leveraged to superimposecollective modes resulting in hitherto unexplored but highly practical hybridcollective patterns such as protective security formations. Our findingsreveal innovative strategies for creating and controlling collective motionpaving the way for new applications in robotic swarm operations active matterorganisation and for the uncovering of obscure interaction rules in biologicalsystems. |


| Item |Content|
| --- |---|
|idx| 2407.11250v1 |
|title| Conditions for Altruistic Perversity in Two-Strategy Population Games |
|authors| Colton HillPhilip N. BrownKeith Paarporn
|links| http://arxiv.org/abs/2407.11250v1 |
|updated| 2024-07-15 21:35:31 UTC |
|summary| Self-interested behavior from individuals can collectively lead to poorsocietal outcomes. These outcomes can seemingly be improved through the actionsof altruistic agents which benefit other agents in the system. However it isknown in specific contexts that altruistic agents can actually induce worseoutcomes compared to a fully selfish population -- a phenomenon we termaltruistic perversity. This paper provides a holistic investigation into thenecessary conditions that give rise to altruistic perversity. In particular westudy the class of two-strategy population games where one sub-population isaltruistic and the other is selfish. We find that a population game can admitaltruistic perversity only if the associated social welfare function is convexand the altruistic population is sufficiently large. Our results are a firststep in establishing a connection between properties of nominal agentinteractions and the potential impacts from altruistic behaviors. |


| Item |Content|
| --- |---|
|idx| 2407.11170v1 |
|title| Time Shift Governor for Constrained Control of Spacecraft Orbit and Attitude Relative Motion in Bicircular Restricted Four-Body Problem |
|authors| Taehyeun KimIlya KolmanovskyAnouck Girard
|links| http://arxiv.org/abs/2407.11170v1 |
|updated| 2024-07-15 18:45:04 UTC |
|summary| This paper considers constrained spacecraft rendezvous and docking RVD inthe setting of the Bicircular Restricted Four-Body Problem BCR4BP whileaccounting for attitude dynamics. We consider Line of Sight LoS coneconstraints thrust limits thrust direction limits and approach velocityconstraints during RVD missions in a near rectilinear halo orbit NRHO in theSun-Earth-Moon system. To enforce the constraints the Time Shift GovernorTSG which uses a time-shifted Chief spacecraft trajectory as a targetreference for the Deputy spacecraft is employed. The time shift is graduallyreduced to zero so that the virtual target gradually evolves towards the Chiefspacecraft as time goes by and the RVD mission objective can be achieved.Numerical simulation results are reported to validate the proposed controlmethod. |


