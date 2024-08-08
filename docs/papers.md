# cs.CL 

| Item |Content|
| --- |---|
|idx| 2408.03936v1 |
|title| SLIM-RAFT: A Novel Fine-Tuning Approach to Improve Cross-Linguistic Performance for Mercosur Common Nomenclature |
|authors| Vinícius Di OliveiraYuri Façanha BezerraLi WeigangPedro Carvalho BromVictor Rafael R. Celestino
|links| http://arxiv.org/abs/2408.03936v1 |
|updated| 2024-08-07 17:54:21 UTC |
|summary| Natural language processing NLP has seen significant advancements with theadvent of large language models LLMs. However substantial improvements arestill needed for languages other than English especially for specific domainslike the applications of Mercosur Common Nomenclature NCM a BrazilianHarmonized System HS. To address this gap this study uses TeenyTineLLaMA afoundational Portuguese LLM as an LLM source to implement the NCM applicationprocessing. Additionally a simplified Retrieval-Augmented Fine-Tuning RAFTtechnique termed SLIM-RAFT is proposed for task-specific fine-tuning of LLMs.This approach retains the chain-of-thought CoT methodology for promptdevelopment in a more concise and streamlined manner utilizing brief andfocused documents for training. The proposed model demonstrates an efficientand cost-effective alternative for fine-tuning smaller LLMs significantlyoutperforming TeenyTineLLaMA and ChatGPT-4 in the same task. Although theresearch focuses on NCM applications the methodology can be easily adapted forHS applications worldwide. |


| Item |Content|
| --- |---|
|idx| 2408.03934v1 |
|title| From Words to Worth: Newborn Article Impact Prediction with LLM |
|authors| Penghai ZhaoQinghua XingKairan DouJinyu TianYing TaiJian YangMing-Ming ChengXiang Li
|links| http://arxiv.org/abs/2408.03934v1 |
|updated| 2024-08-07 17:52:02 UTC |
|summary| As the academic landscape expands the challenge of efficiently identifyingpotentially high-impact articles among the vast number of newly published worksbecomes critical. This paper introduces a promising approach leveraging thecapabilities of fine-tuned LLMs to predict the future impact of newbornarticles solely based on titles and abstracts. Moving beyond traditionalmethods heavily reliant on external information the proposed method discernsthe shared semantic features of highly impactful papers from a large collectionof title-abstract and potential impact pairs. These semantic features arefurther utilized to regress an improved metric TNCSI_SP which has beenendowed with value field and time normalization properties. Additionally acomprehensive dataset has been constructed and released for fine-tuning theLLM containing over 12000 entries with corresponding titles abstracts andTNCSI_SP. The quantitative results with an NDCG20 of 0.901 demonstrate thatthe proposed approach achieves state-of-the-art performance in predicting theimpact of newborn articles when compared to competitive counterparts. Finallywe demonstrate a real-world application for predicting the impact of newbornjournal articles to demonstrate its noteworthy practical value. Overall ourfindings challenge existing paradigms and propose a shift towards a morecontent-focused prediction of academic impact offering new insights forassessing newborn article impact. |


| Item |Content|
| --- |---|
|idx| 2408.03910v1 |
|title| CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases |
|authors| Xiangyan LiuBo LanZhiyuan HuYang LiuZhicheng ZhangWenmeng ZhouFei WangMichael Shieh
|links| http://arxiv.org/abs/2408.03910v1 |
|updated| 2024-08-07 17:13:59 UTC |
|summary| Large Language Models LLMs excel in stand-alone code tasks like HumanEvaland MBPP but struggle with handling entire code repositories. This challengehas prompted research on enhancing LLM-codebase interaction at a repositoryscale. Current solutions rely on similarity-based retrieval or manual tools andAPIs each with notable drawbacks. Similarity-based retrieval often has lowrecall in complex tasks while manual tools and APIs are typicallytask-specific and require expert knowledge reducing their generalizabilityacross diverse code tasks and real-world applications. To mitigate theselimitations we introduce framework a system that integrates LLM agents withgraph database interfaces extracted from code repositories. By leveraging thestructural properties of graph databases and the flexibility of the graph querylanguage framework enables the LLM agent to construct and execute queriesallowing for precise code structure-aware context retrieval and codenavigation. We assess framework using three benchmarks: CrossCodeEvalSWE-bench and EvoCodeBench. Additionally we develop five real-world codingapplications. With a unified graph database schema framework demonstratescompetitive performance and potential in both academic and real-worldenvironments showcasing its versatility and efficacy in software engineering.Our application demo:https://github.com/modelscope/modelscope-agent/tree/master/apps/codexgraph_agent. |


| Item |Content|
| --- |---|
|idx| 2408.03907v1 |
|title| Decoding Biases: Automated Methods and LLM Judges for Gender Bias Detection in Language Models |
|authors| Shachi H KumarSaurav SahaySahisnu MazumderEda OkurRamesh ManuvinakurikeNicole BeckageHsuan SuHung-yi LeeLama Nachman
|links| http://arxiv.org/abs/2408.03907v1 |
|updated| 2024-08-07 17:11:34 UTC |
|summary| Large Language Models LLMs have excelled at language understanding andgenerating human-level text. However even with supervised training and humanalignment these LLMs are susceptible to adversarial attacks where malicioususers can prompt the model to generate undesirable text. LLMs also inherentlyencode potential biases that can cause various harmful effects duringinteractions. Bias evaluation metrics lack standards as well as consensus andexisting methods often rely on human-generated templates and annotations whichare expensive and labor intensive. In this work we train models toautomatically create adversarial prompts to elicit biased responses from targetLLMs. We present LLM- based bias evaluation metrics and also analyze severalexisting automatic evaluation methods and metrics. We analyze the variousnuances of model responses identify the strengths and weaknesses of modelfamilies and assess where evaluation methods fall short. We compare thesemetrics to human evaluation and validate that the LLM-as-a-Judge metric alignswith human judgement on bias in response generation. |


| Item |Content|
| --- |---|
|idx| 2408.03900v1 |
|title| Speech-MASSIVE: A Multilingual Speech Dataset for SLU and Beyond |
|authors| Beomseok LeeIoan CalapodescuMarco GaidoMatteo NegriLaurent Besacier
|links| http://arxiv.org/abs/2408.03900v1 |
|updated| 2024-08-07 16:55:28 UTC |
|summary| We present Speech-MASSIVE a multilingual Spoken Language Understanding SLUdataset comprising the speech counterpart for a portion of the MASSIVE textualcorpus. Speech-MASSIVE covers 12 languages from different families and inheritsfrom MASSIVE the annotations for the intent prediction and slot-filling tasks.Our extension is prompted by the scarcity of massively multilingual SLUdatasets and the growing need for versatile speech datasets to assessfoundation models LLMs speech encoders across languages and tasks. Weprovide a multimodal multitask multilingual dataset and report SLU baselinesusing both cascaded and end-to-end architectures in various training scenarioszero-shot few-shot and full fine-tune. Furthermore we demonstrate thesuitability of Speech-MASSIVE for benchmarking other tasks such as speechtranscription language identification and speech translation. The datasetmodels and code are publicly available at:https://github.com/hlt-mt/Speech-MASSIVE |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2408.03936v1 |
|title| SLIM-RAFT: A Novel Fine-Tuning Approach to Improve Cross-Linguistic Performance for Mercosur Common Nomenclature |
|authors| Vinícius Di OliveiraYuri Façanha BezerraLi WeigangPedro Carvalho BromVictor Rafael R. Celestino
|links| http://arxiv.org/abs/2408.03936v1 |
|updated| 2024-08-07 17:54:21 UTC |
|summary| Natural language processing NLP has seen significant advancements with theadvent of large language models LLMs. However substantial improvements arestill needed for languages other than English especially for specific domainslike the applications of Mercosur Common Nomenclature NCM a BrazilianHarmonized System HS. To address this gap this study uses TeenyTineLLaMA afoundational Portuguese LLM as an LLM source to implement the NCM applicationprocessing. Additionally a simplified Retrieval-Augmented Fine-Tuning RAFTtechnique termed SLIM-RAFT is proposed for task-specific fine-tuning of LLMs.This approach retains the chain-of-thought CoT methodology for promptdevelopment in a more concise and streamlined manner utilizing brief andfocused documents for training. The proposed model demonstrates an efficientand cost-effective alternative for fine-tuning smaller LLMs significantlyoutperforming TeenyTineLLaMA and ChatGPT-4 in the same task. Although theresearch focuses on NCM applications the methodology can be easily adapted forHS applications worldwide. |


| Item |Content|
| --- |---|
|idx| 2408.03910v1 |
|title| CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases |
|authors| Xiangyan LiuBo LanZhiyuan HuYang LiuZhicheng ZhangWenmeng ZhouFei WangMichael Shieh
|links| http://arxiv.org/abs/2408.03910v1 |
|updated| 2024-08-07 17:13:59 UTC |
|summary| Large Language Models LLMs excel in stand-alone code tasks like HumanEvaland MBPP but struggle with handling entire code repositories. This challengehas prompted research on enhancing LLM-codebase interaction at a repositoryscale. Current solutions rely on similarity-based retrieval or manual tools andAPIs each with notable drawbacks. Similarity-based retrieval often has lowrecall in complex tasks while manual tools and APIs are typicallytask-specific and require expert knowledge reducing their generalizabilityacross diverse code tasks and real-world applications. To mitigate theselimitations we introduce framework a system that integrates LLM agents withgraph database interfaces extracted from code repositories. By leveraging thestructural properties of graph databases and the flexibility of the graph querylanguage framework enables the LLM agent to construct and execute queriesallowing for precise code structure-aware context retrieval and codenavigation. We assess framework using three benchmarks: CrossCodeEvalSWE-bench and EvoCodeBench. Additionally we develop five real-world codingapplications. With a unified graph database schema framework demonstratescompetitive performance and potential in both academic and real-worldenvironments showcasing its versatility and efficacy in software engineering.Our application demo:https://github.com/modelscope/modelscope-agent/tree/master/apps/codexgraph_agent. |


| Item |Content|
| --- |---|
|idx| 2408.03909v1 |
|title| LaFA: Latent Feature Attacks on Non-negative Matrix Factorization |
|authors| Minh VuBen NebgenErik SkauGeigh ZollicofferJuan CastorenaKim RasmussenBoian AlexandrovManish Bhattarai
|links| http://arxiv.org/abs/2408.03909v1 |
|updated| 2024-08-07 17:13:46 UTC |
|summary| As Machine Learning ML applications rapidly grow concerns aboutadversarial attacks compromising their reliability have gained significantattention. One unsupervised ML method known for its resilience to such attacksis Non-negative Matrix Factorization NMF an algorithm that decomposes inputdata into lower-dimensional latent features. However the introduction ofpowerful computational tools such as Pytorch enables the computation ofgradients of the latent features with respect to the original data raisingconcerns about NMFs reliability. Interestingly naively deriving theadversarial loss for NMF as in the case of ML would result in thereconstruction loss which can be shown theoretically to be an ineffectiveattacking objective. In this work we introduce a novel class of attacks in NMFtermed Latent Feature Attacks LaFA which aim to manipulate the latentfeatures produced by the NMF process. Our method utilizes the Feature ErrorFE loss directly on the latent features. By employing FE loss we generateperturbations in the original data that significantly affect the extractedlatent features revealing vulnerabilities akin to those found in other MLtechniques. To handle large peak-memory overhead from gradient back-propagationin FE attacks we develop a method based on implicit differentiation whichenables their scaling to larger datasets. We validate NMF vulnerabilities andFE attacks effectiveness through extensive experiments on synthetic andreal-world data. |


| Item |Content|
| --- |---|
|idx| 2408.03907v1 |
|title| Decoding Biases: Automated Methods and LLM Judges for Gender Bias Detection in Language Models |
|authors| Shachi H KumarSaurav SahaySahisnu MazumderEda OkurRamesh ManuvinakurikeNicole BeckageHsuan SuHung-yi LeeLama Nachman
|links| http://arxiv.org/abs/2408.03907v1 |
|updated| 2024-08-07 17:11:34 UTC |
|summary| Large Language Models LLMs have excelled at language understanding andgenerating human-level text. However even with supervised training and humanalignment these LLMs are susceptible to adversarial attacks where malicioususers can prompt the model to generate undesirable text. LLMs also inherentlyencode potential biases that can cause various harmful effects duringinteractions. Bias evaluation metrics lack standards as well as consensus andexisting methods often rely on human-generated templates and annotations whichare expensive and labor intensive. In this work we train models toautomatically create adversarial prompts to elicit biased responses from targetLLMs. We present LLM- based bias evaluation metrics and also analyze severalexisting automatic evaluation methods and metrics. We analyze the variousnuances of model responses identify the strengths and weaknesses of modelfamilies and assess where evaluation methods fall short. We compare thesemetrics to human evaluation and validate that the LLM-as-a-Judge metric alignswith human judgement on bias in response generation. |


| Item |Content|
| --- |---|
|idx| 2408.03904v1 |
|title| Lightweight Video Denoising Using a Classic Bayesian Backbone |
|authors| Clément BledFrançois Pitié
|links| http://arxiv.org/abs/2408.03904v1 |
|updated| 2024-08-07 17:08:46 UTC |
|summary| In recent years state-of-the-art image and video denoising networks havebecome increasingly large requiring millions of trainable parameters toachieve best-in-class performance. Improved denoising quality has come at thecost of denoising speed where modern transformer networks are far slower torun than smaller denoising networks such as FastDVDnet and classic Bayesiandenoisers such as the Wiener filter.  In this paper we implement a hybrid Wiener filter which leverages smallancillary networks to increase the original denoiser performance whileretaining fast denoising speeds. These networks are used to refine the Wienercoring estimate optimise windowing functions and estimate the unknown noiseprofile. Using these methods we outperform several popular denoisers andremain within 0.2 dB on average of the popular VRT transformer. Our methodwas found to be over x10 faster than the transformer method with a far lowerparameter cost. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2408.03936v1 |
|title| SLIM-RAFT: A Novel Fine-Tuning Approach to Improve Cross-Linguistic Performance for Mercosur Common Nomenclature |
|authors| Vinícius Di OliveiraYuri Façanha BezerraLi WeigangPedro Carvalho BromVictor Rafael R. Celestino
|links| http://arxiv.org/abs/2408.03936v1 |
|updated| 2024-08-07 17:54:21 UTC |
|summary| Natural language processing NLP has seen significant advancements with theadvent of large language models LLMs. However substantial improvements arestill needed for languages other than English especially for specific domainslike the applications of Mercosur Common Nomenclature NCM a BrazilianHarmonized System HS. To address this gap this study uses TeenyTineLLaMA afoundational Portuguese LLM as an LLM source to implement the NCM applicationprocessing. Additionally a simplified Retrieval-Augmented Fine-Tuning RAFTtechnique termed SLIM-RAFT is proposed for task-specific fine-tuning of LLMs.This approach retains the chain-of-thought CoT methodology for promptdevelopment in a more concise and streamlined manner utilizing brief andfocused documents for training. The proposed model demonstrates an efficientand cost-effective alternative for fine-tuning smaller LLMs significantlyoutperforming TeenyTineLLaMA and ChatGPT-4 in the same task. Although theresearch focuses on NCM applications the methodology can be easily adapted forHS applications worldwide. |


| Item |Content|
| --- |---|
|idx| 2408.03915v1 |
|title| Hard to Explain: On the Computational Hardness of In-Distribution Model Interpretation |
|authors| Guy AmirShahaf BassanGuy Katz
|links| http://arxiv.org/abs/2408.03915v1 |
|updated| 2024-08-07 17:20:52 UTC |
|summary| The ability to interpret Machine Learning ML models is becomingincreasingly essential. However despite significant progress in the fieldthere remains a lack of rigorous characterization regarding the innateinterpretability of different models. In an attempt to bridge this gap recentwork has demonstrated that it is possible to formally assess interpretabilityby studying the computational complexity of explaining the decisions of variousmodels. In this setting if explanations for a particular model can be obtainedefficiently the model is considered interpretable since it can be explainedeasily. However if generating explanations over an ML model iscomputationally intractable it is considered uninterpretable. Prior researchidentified two key factors that influence the complexity of interpreting an MLmodel: i the type of the model e.g. neural networks decision trees etc.and ii the form of explanation e.g. contrastive explanations Shapleyvalues etc.. In this work we claim that a third important factor must alsobe considered for this analysis -- the underlying distribution over which theexplanation is obtained. Considering the underlying distribution is key inavoiding explanations that are socially misaligned i.e. convey informationthat is biased and unhelpful to users. We demonstrate the significant influenceof the underlying distribution on the resulting overall interpretationcomplexity in two settings: i prediction models paired with an externalout-of-distribution OOD detector and ii prediction models designed toinherently generate socially aligned explanations. Our findings prove that theexpressiveness of the distribution can significantly influence the overallcomplexity of interpretation and identify essential prerequisites that a modelmust possess to generate socially aligned explanations. |


| Item |Content|
| --- |---|
|idx| 2408.03913v1 |
|title| AdapMTL: Adaptive Pruning Framework for Multitask Learning Model |
|authors| Mingcan XiangSteven Jiaxun TangQizheng YangHui GuanTongping Liu
|links| http://dx.doi.org/10.1145/3664647.3681426 |
|updated| 2024-08-07 17:19:15 UTC |
|summary| In the domain of multimedia and multimodal processing the efficient handlingof diverse data streams such as images video and sensor data is paramount.Model compression and multitask learning MTL are crucial in this fieldoffering the potential to address the resource-intensive demands of processingand interpreting multiple forms of media simultaneously. However effectivelycompressing a multitask model presents significant challenges due to thecomplexities of balancing sparsity allocation and accuracy performance acrossmultiple tasks. To tackle these challenges we propose AdapMTL an adaptivepruning framework for MTL models. AdapMTL leverages multiple learnable softthresholds independently assigned to the shared backbone and the task-specificheads to capture the nuances in different components sensitivity to pruning.During training it co-optimizes the soft thresholds and MTL model weights toautomatically determine the suitable sparsity level at each component toachieve both high task accuracy and high overall sparsity. It furtherincorporates an adaptive weighting mechanism that dynamically adjusts theimportance of task-specific losses based on each tasks robustness to pruning.We demonstrate the effectiveness of AdapMTL through comprehensive experimentson popular multitask datasets namely NYU-v2 and Tiny-Taskonomy with differentarchitectures showcasing superior performance compared to state-of-the-artpruning methods. |


| Item |Content|
| --- |---|
|idx| 2408.03909v1 |
|title| LaFA: Latent Feature Attacks on Non-negative Matrix Factorization |
|authors| Minh VuBen NebgenErik SkauGeigh ZollicofferJuan CastorenaKim RasmussenBoian AlexandrovManish Bhattarai
|links| http://arxiv.org/abs/2408.03909v1 |
|updated| 2024-08-07 17:13:46 UTC |
|summary| As Machine Learning ML applications rapidly grow concerns aboutadversarial attacks compromising their reliability have gained significantattention. One unsupervised ML method known for its resilience to such attacksis Non-negative Matrix Factorization NMF an algorithm that decomposes inputdata into lower-dimensional latent features. However the introduction ofpowerful computational tools such as Pytorch enables the computation ofgradients of the latent features with respect to the original data raisingconcerns about NMFs reliability. Interestingly naively deriving theadversarial loss for NMF as in the case of ML would result in thereconstruction loss which can be shown theoretically to be an ineffectiveattacking objective. In this work we introduce a novel class of attacks in NMFtermed Latent Feature Attacks LaFA which aim to manipulate the latentfeatures produced by the NMF process. Our method utilizes the Feature ErrorFE loss directly on the latent features. By employing FE loss we generateperturbations in the original data that significantly affect the extractedlatent features revealing vulnerabilities akin to those found in other MLtechniques. To handle large peak-memory overhead from gradient back-propagationin FE attacks we develop a method based on implicit differentiation whichenables their scaling to larger datasets. We validate NMF vulnerabilities andFE attacks effectiveness through extensive experiments on synthetic andreal-world data. |


| Item |Content|
| --- |---|
|idx| 2408.03877v1 |
|title| Knowledge Probing for Graph Representation Learning |
|authors| Mingyu ZhaoXingyu HuangZiyu LyuYanlin WangLixin CuiLu Bai
|links| http://arxiv.org/abs/2408.03877v1 |
|updated| 2024-08-07 16:27:45 UTC |
|summary| Graph learning methods have been extensively applied in diverse applicationareas. However what kind of inherent graph properties e.g. graph proximitygraph structural information has been encoded into graph representationlearning for downstream tasks is still under-explored. In this paper wepropose a novel graph probing framework GraphProbe to investigate andinterpret whether the family of graph learning methods has encoded differentlevels of knowledge in graph representation learning. Based on the intrinsicproperties of graphs we design three probes to systematically investigate thegraph representation learning process from different perspectives respectivelythe node-wise level the path-wise level and the structural level. Weconstruct a thorough evaluation benchmark with nine representative graphlearning methods from random walk based approaches basic graph neural networksand self-supervised graph methods and probe them on six benchmark datasets fornode classification link prediction and graph classification. The experimentalevaluation verify that GraphProbe can estimate the capability of graphrepresentation learning. Remaking results have been concluded: GCN andWeightedGCN methods are relatively versatile methods achieving better resultswith respect to different tasks. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2408.03940v1 |
|title| How Well Can Vision Language Models See Image Details? |
|authors| Chenhui GouAbdulwahab FelembanFaizan Farooq KhanDeyao ZhuJianfei CaiHamid RezatofighiMohamed Elhoseiny
|links| http://arxiv.org/abs/2408.03940v1 |
|updated| 2024-08-07 17:59:40 UTC |
|summary| Large Language Model-based Vision-Language Models LLM-based VLMs havedemonstrated impressive results in various vision-language understanding tasks.However how well these VLMs can see image detail beyond the semantic levelremains unclear. In our study we introduce a pixel value prediction task PVPto explore How Well Can Vision Language Models See Image Details and toassist VLMs in perceiving more details. Typically these models comprise afrozen CLIP visual encoder a large language model and a connecting module.After fine-tuning VLMs on the PVP task we find: 1 existing VLMs struggle topredict precise pixel values by only fine-tuning the connection module and LLMand 2 prediction precision is significantly improved when the vision encoderis also adapted. Additionally our research reveals that incorporating pixelvalue prediction as one of the VLM pre-training tasks and vision encoderadaptation markedly boosts VLM performance on downstream image-languageunderstanding tasks requiring detailed image perception such as referringimage segmentation with an average 10.19 cIoU improvement and video gamedecision making with average score improvements of 80.34 and 70.54 on twogames respectively. |


| Item |Content|
| --- |---|
|idx| 2408.03923v1 |
|title| Fast Sprite Decomposition from Animated Graphics |
|authors| Tomoyuki SuzukiKotaro KikuchiKota Yamaguchi
|links| http://arxiv.org/abs/2408.03923v1 |
|updated| 2024-08-07 17:30:59 UTC |
|summary| This paper presents an approach to decomposing animated graphics intosprites a set of basic elements or layers. Our approach builds on theoptimization of sprite parameters to fit the raster video. For efficiency weassume static textures for sprites to reduce the search space while preventingartifacts using a texture prior model. To further speed up the optimization weintroduce the initialization of the sprite parameters utilizing a pre-trainedvideo object segmentation model and user input of single frame annotations. Forour study we construct the Crello Animation dataset from an online designservice and define quantitative metrics to measure the quality of the extractedsprites. Experiments show that our method significantly outperforms baselinesfor similar decomposition tasks in terms of the quality/efficiency tradeoff. |


| Item |Content|
| --- |---|
|idx| 2408.03922v1 |
|title| FMiFood: Multi-modal Contrastive Learning for Food Image Classification |
|authors| Xinyue PanJiangpeng HeFengqing Zhu
|links| http://arxiv.org/abs/2408.03922v1 |
|updated| 2024-08-07 17:29:19 UTC |
|summary| Food image classification is the fundamental step in image-based dietaryassessment which aims to estimate participants nutrient intake from eatingoccasion images. A common challenge of food images is the intra-class diversityand inter-class similarity which can significantly hinder classificationperformance. To address this issue we introduce a novel multi-modalcontrastive learning framework called FMiFood which learns more discriminativefeatures by integrating additional contextual information such as foodcategory text descriptions to enhance classification accuracy. Specificallywe propose a flexible matching technique that improves the similarity matchingbetween text and image embeddings to focus on multiple key information.Furthermore we incorporate the classification objectives into the frameworkand explore the use of GPT-4 to enrich the text descriptions and provide moredetailed context. Our method demonstrates improved performance on both theUPMC-101 and VFN datasets compared to existing methods. |


| Item |Content|
| --- |---|
|idx| 2408.03913v1 |
|title| AdapMTL: Adaptive Pruning Framework for Multitask Learning Model |
|authors| Mingcan XiangSteven Jiaxun TangQizheng YangHui GuanTongping Liu
|links| http://dx.doi.org/10.1145/3664647.3681426 |
|updated| 2024-08-07 17:19:15 UTC |
|summary| In the domain of multimedia and multimodal processing the efficient handlingof diverse data streams such as images video and sensor data is paramount.Model compression and multitask learning MTL are crucial in this fieldoffering the potential to address the resource-intensive demands of processingand interpreting multiple forms of media simultaneously. However effectivelycompressing a multitask model presents significant challenges due to thecomplexities of balancing sparsity allocation and accuracy performance acrossmultiple tasks. To tackle these challenges we propose AdapMTL an adaptivepruning framework for MTL models. AdapMTL leverages multiple learnable softthresholds independently assigned to the shared backbone and the task-specificheads to capture the nuances in different components sensitivity to pruning.During training it co-optimizes the soft thresholds and MTL model weights toautomatically determine the suitable sparsity level at each component toachieve both high task accuracy and high overall sparsity. It furtherincorporates an adaptive weighting mechanism that dynamically adjusts theimportance of task-specific losses based on each tasks robustness to pruning.We demonstrate the effectiveness of AdapMTL through comprehensive experimentson popular multitask datasets namely NYU-v2 and Tiny-Taskonomy with differentarchitectures showcasing superior performance compared to state-of-the-artpruning methods. |


| Item |Content|
| --- |---|
|idx| 2408.03904v1 |
|title| Lightweight Video Denoising Using a Classic Bayesian Backbone |
|authors| Clément BledFrançois Pitié
|links| http://arxiv.org/abs/2408.03904v1 |
|updated| 2024-08-07 17:08:46 UTC |
|summary| In recent years state-of-the-art image and video denoising networks havebecome increasingly large requiring millions of trainable parameters toachieve best-in-class performance. Improved denoising quality has come at thecost of denoising speed where modern transformer networks are far slower torun than smaller denoising networks such as FastDVDnet and classic Bayesiandenoisers such as the Wiener filter.  In this paper we implement a hybrid Wiener filter which leverages smallancillary networks to increase the original denoiser performance whileretaining fast denoising speeds. These networks are used to refine the Wienercoring estimate optimise windowing functions and estimate the unknown noiseprofile. Using these methods we outperform several popular denoisers andremain within 0.2 dB on average of the popular VRT transformer. Our methodwas found to be over x10 faster than the transformer method with a far lowerparameter cost. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2408.03769v1 |
|title| Nadaraya-Watson kernel smoothing as a random energy model |
|authors| Jacob A. Zavatone-VethCengiz Pehlevan
|links| http://arxiv.org/abs/2408.03769v1 |
|updated| 2024-08-07 13:43:21 UTC |
|summary| We investigate the behavior of the Nadaraya-Watson kernel smoothing estimatorin high dimensions using its relationship to the random energy model and todense associative memories. |


| Item |Content|
| --- |---|
|idx| 2408.03746v1 |
|title| Flexible Bayesian Last Layer Models Using Implicit Priors and Diffusion Posterior Sampling |
|authors| Jian XuZhiqi LinShigui LiMin ChenJunmei YangDelu ZengJohn Paisley
|links| http://arxiv.org/abs/2408.03746v1 |
|updated| 2024-08-07 12:59:58 UTC |
|summary| Bayesian Last Layer BLL models focus solely on uncertainty in the outputlayer of neural networks demonstrating comparable performance to more complexBayesian models. However the use of Gaussian priors for last layer weights inBayesian Last Layer BLL models limits their expressive capacity when facedwith non-Gaussian outlier-rich or high-dimensional datasets. To address thisshortfall we introduce a novel approach that combines diffusion techniques andimplicit priors for variational learning of Bayesian last layer weights. Thismethod leverages implicit distributions for modeling weight priors in BLLcoupled with diffusion samplers for approximating true posterior predictionsthereby establishing a comprehensive Bayesian prior and posterior estimationstrategy. By delivering an explicit and computationally efficient variationallower bound our method aims to augment the expressive abilities of BLL modelsenhancing model accuracy calibration and out-of-distribution detectionproficiency. Through detailed exploration and experimental validation Weshowcase the methods potential for improving predictive accuracy anduncertainty quantification while ensuring computational efficiency. |


| Item |Content|
| --- |---|
|idx| 2408.03733v1 |
|title| Bayes-optimal learning of an extensive-width neural network from quadratically many samples |
|authors| Antoine MaillardEmanuele TroianiSimon MartinFlorent KrzakalaLenka Zdeborová
|links| http://arxiv.org/abs/2408.03733v1 |
|updated| 2024-08-07 12:41:56 UTC |
|summary| We consider the problem of learning a target function corresponding to asingle hidden layer neural network with a quadratic activation function afterthe first layer and random weights. We consider the asymptotic limit where theinput dimension and the network width are proportionally large. Recent workCui  al 23 established that linear regression provides Bayes-optimal testerror to learn such a function when the number of available samples is onlylinear in the dimension. That work stressed the open challenge of theoreticallyanalyzing the optimal test error in the more interesting regime where thenumber of samples is quadratic in the dimension. In this paper we solve thischallenge for quadratic activations and derive a closed-form expression for theBayes-optimal test error. We also provide an algorithm that we call GAMP-RIEwhich combines approximate message passing with rotationally invariant matrixdenoising and that asymptotically achieves the optimal performance.Technically our result is enabled by establishing a link with recent works onoptimal denoising of extensive-rank matrices and on the ellipsoid fittingproblem. We further show empirically that in the absence of noiserandomly-initialized gradient descent seems to sample the space of weightsleading to zero training loss and averaging over initialization leads to atest error equal to the Bayes-optimal one. |


| Item |Content|
| --- |---|
|idx| 2408.03626v1 |
|title| On the choice of the non-trainable internal weights in random feature maps |
|authors| Pinak MandalGeorg A. Gottwald
|links| http://arxiv.org/abs/2408.03626v1 |
|updated| 2024-08-07 08:37:23 UTC |
|summary| The computationally cheap machine learning architecture of random featuremaps can be viewed as a single-layer feedforward network in which the weightsof the hidden layer are random but fixed and only the outer weights are learnedvia linear regression. The internal weights are typically chosen from aprescribed distribution. The choice of the internal weights significantlyimpacts the accuracy of random feature maps. We address here the task of how tobest select the internal weights. In particular we consider the forecastingproblem whereby random feature maps are used to learn a one-step propagator mapfor a dynamical system. We provide a computationally cheap hit-and-runalgorithm to select good internal weights which lead to good forecasting skill.We show that the number of good features is the main factor controlling theforecasting skill of random feature maps and acts as an effective featuredimension. Lastly we compare random feature maps with single-layer feedforwardneural networks in which the internal weights are now learned using gradientdescent. We find that random feature maps have superior forecastingcapabilities whilst having several orders of magnitude lower computationalcost. |


| Item |Content|
| --- |---|
|idx| 2408.03590v1 |
|title| Sensitivity analysis using the Metamodel of Optimal Prognosis |
|authors| Thomas MostJohannes Will
|links| http://arxiv.org/abs/2408.03590v1 |
|updated| 2024-08-07 07:09:06 UTC |
|summary| In real case applications within the virtual prototyping process it is notalways possible to reduce the complexity of the physical models and to obtainnumerical models which can be solved quickly. Usually every single numericalsimulation takes hours or even days. Although the progresses in numericalmethods and high performance computing in such cases it is not possible toexplore various model configurations hence efficient surrogate models arerequired. Generally the available meta-model techniques show several advantagesand disadvantages depending on the investigated problem. In this paper wepresent an automatic approach for the selection of the optimal suitablemeta-model for the actual problem. Together with an automatic reduction of thevariable space using advanced filter techniques an efficient approximation isenabled also for high dimensional problems. This filter techniques enable areduction of the high dimensional variable space to a much smaller subspacewhere meta-model-based sensitivity analyses are carried out to assess theinfluence of important variables and to identify the optimal subspace withcorresponding surrogate model which enables the most accurate probabilisticanalysis. For this purpose we investigate variance-based and moment-freesensitivity measures in combination with advanced meta-models as moving leastsquares and kriging. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2408.03889v1 |
|title| The State of Reproducibility Stamps for Visualization Research Papers |
|authors| Tobias Isenberg
|links| http://arxiv.org/abs/2408.03889v1 |
|updated| 2024-08-07 16:40:03 UTC |
|summary| I analyze the evolution of papers certified by the Graphics ReplicabilityStamp Initiative GRSI to be reproducible with a specific focus on the subsetof publications that address visualization-related topics. With this analysis Ishow that while the number of papers is increasing overall and within thevisualization field we still have to improve quite a bit to escape thereplication crisis. I base my analysis on the data published by the GRSI aswell as publication data for the different venues in visualization and lists ofjournal papers that have been presented at visualization-focused conferences. Ialso analyze the differences between the involved journals as well as thepercentage of reproducible papers in the different presentation venues.Furthermore I look at the authors of the publications and in particulartheir affiliation countries to see where most reproducible papers come from.Finally I discuss potential reasons for the low reproducibility numbers andsuggest possible ways to overcome these obstacles. This paper is reproducibleitself with source code and data available fromgithub.com/tobiasisenberg/Visualization-Reproducibility as well as a free papercopy and all supplemental materials at osf.io/mvnbj. |


| Item |Content|
| --- |---|
|idx| 2408.03876v1 |
|title| From Data to Story: Towards Automatic Animated Data Video Creation with LLM-based Multi-Agent Systems |
|authors| Leixian ShenHaotian LiYun WangHuamin Qu
|links| http://arxiv.org/abs/2408.03876v1 |
|updated| 2024-08-07 16:25:39 UTC |
|summary| Creating data stories from raw data is challenging due to humans limitedattention spans and the need for specialized skills. Recent advancements inlarge language models LLMs offer great opportunities to develop systems withautonomous agents to streamline the data storytelling workflow. Thoughmulti-agent systems have benefits such as fully realizing LLM potentials withdecomposed tasks for individual agents designing such systems also faceschallenges in task decomposition performance optimization for sub-tasks andworkflow design. To better understand these issues we develop Data Directoran LLM-based multi-agent system designed to automate the creation of animateddata videos a representative genre of data stories. Data Director interpretsraw data breaks down tasks designs agent roles to make informed decisionsautomatically and seamlessly integrates diverse components of data videos. Acase study demonstrates Data Directors effectiveness in generating datavideos. Throughout development we have derived lessons learned from addressingchallenges guiding further advancements in autonomous agents for datastorytelling. We also shed light on future directions for global optimizationhuman-in-the-loop design and the application of advanced multi-modal LLMs. |


| Item |Content|
| --- |---|
|idx| 2408.03845v1 |
|title| ImageSI: Semantic Interaction for Deep Learning Image Projections |
|authors| Jiayue LinRebecca FaustChris North
|links| http://arxiv.org/abs/2408.03845v1 |
|updated| 2024-08-07 15:40:05 UTC |
|summary| Semantic interaction SI in Dimension Reduction DR of images allows usersto incorporate feedback through direct manipulation of the 2D positions ofimages. Through interaction users specify a set of pairwise relationships thatthe DR should aim to capture. Existing methods for images incorporate feedbackinto the DR through feature weights on abstract embedding features. However ifthe original embedding features do not suitably capture the users task thenthe DR cannot either. We propose ImageSI an SI method for image DR thatincorporates user feedback directly into the image model to update theunderlying embeddings rather than weighting them. In doing so ImageSI ensuresthat the embeddings suitably capture the features necessary for the task sothat the DR can subsequently organize images using those features. We presenttwo variations of ImageSI using different loss functions - ImageSI_MDS_Inversewhich prioritizes the explicit pairwise relationships from the interaction andImageSI_Triplet which prioritizes clustering using the interaction to definegroups of images. Finally we present a usage scenario and a simulation basedevaluation to demonstrate the utility of ImageSI and compare it to currentmethods. |


| Item |Content|
| --- |---|
|idx| 2408.03827v1 |
|title| Automated Code Fix Suggestions for Accessibility Issues in Mobile Apps |
|authors| Forough MehralianTitus BarikJeff NicholsAmanda Swearngin
|links| http://arxiv.org/abs/2408.03827v1 |
|updated| 2024-08-07 15:06:07 UTC |
|summary| Accessibility is crucial for inclusive app usability yet developers oftenstruggle to identify and fix app accessibility issues due to a lack ofawareness expertise and inadequate tools. Current accessibility testing toolscan identify accessibility issues but may not always provide guidance on how toaddress them. We introduce FixAlly an automated tool designed to suggestsource code fixes for accessibility issues detected by automated accessibilityscanners. FixAlly employs a multi-agent LLM architecture to generate fixstrategies localize issues within the source code and propose codemodification suggestions to fix the accessibility issue. Our empirical studydemonstrates FixAllys capability in suggesting fixes that resolve issues foundby accessibility scanners -- with an effectiveness of 77 in generatingplausible fix suggestions -- and our survey of 12 iOS developers finds theywould be willing to accept 69.4 of evaluated fix suggestions. |


| Item |Content|
| --- |---|
|idx| 2408.03819v1 |
|title| Leveraging Variation Theory in Counterfactual Data Augmentation for Optimized Active Learning |
|authors| Simret Araya GebreegziabherKuangshi AiZheng ZhangElena L. GlassmanToby Jia-Jun Li
|links| http://arxiv.org/abs/2408.03819v1 |
|updated| 2024-08-07 14:55:04 UTC |
|summary| Active Learning AL allows models to learn interactively from user feedback.This paper introduces a counterfactual data augmentation approach to ALparticularly addressing the selection of datapoints for user querying apivotal concern in enhancing data efficiency. Our approach is inspired byVariation Theory a theory of human concept learning that emphasizes theessential features of a concept by focusing on what stays the same and whatchanges. Instead of just querying with existing datapoints our approachsynthesizes artificial datapoints that highlight potential key similarities anddifferences among labels using a neuro-symbolic pipeline combining largelanguage models LLMs and rule-based models. Through an experiment in theexample domain of text classification we show that our approach achievessignificantly higher performance when there are fewer annotated data. As theannotated training data gets larger the impact of the generated data starts todiminish showing its capability to address the cold start problem in AL. Thisresearch sheds light on integrating theories of human learning into theoptimization of AL. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2408.03692v1 |
|title| Asynchronous Credit Assignment Framework for Multi-Agent Reinforcement Learning |
|authors| Yongheng LiangHejun WuHaitao WangHao Cai
|links| http://arxiv.org/abs/2408.03692v1 |
|updated| 2024-08-07 11:13:26 UTC |
|summary| Credit assignment is a core problem that distinguishes agents marginalcontributions for optimizing cooperative strategies in multi-agentreinforcement learning MARL. Current credit assignment methods usually assumesynchronous decision-making among agents. However a prerequisite for manyrealistic cooperative tasks is asynchronous decision-making by agents withoutwaiting for others to avoid disastrous consequences. To address this issue wepropose an asynchronous credit assignment framework with a problem model calledADEX-POMDP and a multiplicative value decomposition MVD algorithm. ADEX-POMDPis an asynchronous problem model with extra virtual agents for a decentralizedpartially observable markov decision process. We prove that ADEX-POMDPpreserves both the task equilibrium and the algorithm convergence. MVD utilizesmultiplicative interaction to efficiently capture the interactions ofasynchronous decisions and we theoretically demonstrate its advantages inhandling asynchronous tasks. Experimental results show that on two asynchronousdecision-making benchmarks Overcooked and POAC MVD not only consistentlyoutperforms state-of-the-art MARL methods but also provides theinterpretability for asynchronous cooperation. |


| Item |Content|
| --- |---|
|idx| 2408.03405v1 |
|title| Combining Diverse Information for Coordinated Action: Stochastic Bandit Algorithms for Heterogeneous Agents |
|authors| Lucia GordonEsther RolfMilind Tambe
|links| http://arxiv.org/abs/2408.03405v1 |
|updated| 2024-08-06 18:56:29 UTC |
|summary| Stochastic multi-agent multi-armed bandits typically assume that the rewardsfrom each arm follow a fixed distribution regardless of which agent pulls thearm. However in many real-world settings rewards can depend on thesensitivity of each agent to their environment. In medical screening diseasedetection rates can vary by test type in preference matching rewards candepend on user preferences and in environmental sensing observation qualitycan vary across sensors. Since past work does not specify how to allocateagents of heterogeneous but known sensitivity of these types in a stochasticbandit setting we introduce a UCB-style algorithm Min-Width which aggregatesinformation from diverse agents. In doing so we address the joint challengesof i aggregating the rewards which follow different distributions for eachagent-arm pair and ii coordinating the assignments of agents to arms.Min-Width facilitates efficient collaboration among heterogeneous agentsexploiting the known structure in the agents reward functions to weight theirrewards accordingly. We analyze the regret of Min-Width and conductpseudo-synthetic and fully synthetic experiments to study the performance ofdifferent levels of information sharing. Our results confirm that the gains tomodeling agent heterogeneity tend to be greater when the sensitivities are morevaried across agents while combining more information does not always improveperformance. |


| Item |Content|
| --- |---|
|idx| 2408.02845v1 |
|title| Heterogeneous graph attention network improves cancer multiomics integration |
|authors| Sina TabakhiCharlotte VandermeulenIan SudberyHaiping Lu
|links| http://arxiv.org/abs/2408.02845v1 |
|updated| 2024-08-05 22:01:13 UTC |
|summary| The increase in high-dimensional multiomics data demands advanced integrationmodels to capture the complexity of human diseases. Graph-based deep learningintegration models despite their promise struggle with small patient cohortsand high-dimensional features often applying independent feature selectionwithout modeling relationships among omics. Furthermore conventionalgraph-based omics models focus on homogeneous graphs lacking multiple types ofnodes and edges to capture diverse structures. We introduce a HeterogeneousGraph ATtention network for omics integration HeteroGATomics to improvecancer diagnosis. HeteroGATomics performs joint feature selection through amulti-agent system creating dedicated networks of feature and patientsimilarity for each omic modality. These networks are then combined into oneheterogeneous graph for learning holistic omic-specific representations andintegrating predictions across modalities. Experiments on three cancermultiomics datasets demonstrate HeteroGATomics superior performance in cancerdiagnosis. Moreover HeteroGATomics enhances interpretability by identifyingimportant biomarkers contributing to the diagnosis outcomes. |


| Item |Content|
| --- |---|
|idx| 2408.02768v1 |
|title| Assessing the Effects of Container Handling Strategies on Enhancing Freight Throughput |
|authors| Sarita RattanakunuprakarnMingzhou JinMustafa Can CamurXueping Li
|links| http://arxiv.org/abs/2408.02768v1 |
|updated| 2024-08-05 18:38:27 UTC |
|summary| As global supply chains and freight volumes grow the U.S. faces escalatingtransportation demands. The heavy reliance on road transport coupled with theunderutilization of the railway system results in congested highwaysprolonged transportation times higher costs and increased carbon emissions.Californias San Pedro Port Complex SPPC the nations busiest incurs asignificant share of these challenges. We utilize an agent-based simulation toreplicate real-world scenarios focusing on the intricacies of interactions ina modified intermodal inbound freight system for the SPPC. This involvesrelocating container classification to potential warehouses in CaliforniaUtah Arizona and Nevada rather than exclusively at port areas. Our primaryaim is to evaluate the proposed systems efficiency considering cost andfreight throughput while also examining the effects of workforce shortages.Computational analysis suggests that strategically installing intermodalcapabilities in select warehouses can reduce transportation costs boostthroughput and foster resour |


| Item |Content|
| --- |---|
|idx| 2408.02248v1 |
|title| ReDel: A Toolkit for LLM-Powered Recursive Multi-Agent Systems |
|authors| Andrew ZhuLiam DuganChris Callison-Burch
|links| http://arxiv.org/abs/2408.02248v1 |
|updated| 2024-08-05 05:43:23 UTC |
|summary| Recently there has been increasing interest in using Large Language ModelsLLMs to construct complex multi-agent systems to perform tasks such ascompiling literature reviews drafting consumer reports and planningvacations. Many tools and libraries exist for helping create such systemshowever none support recursive multi-agent systems -- where the modelsthemselves flexibly decide when to delegate tasks and how to organize theirdelegation structure. In this work we introduce ReDel: a toolkit for recursivemulti-agent systems that supports custom tool-use delegation schemesevent-based logging and interactive replay in an easy-to-use web interface. Weshow that using ReDel we are able to achieve significant performance gains onagentic benchmarks and easily identify potential areas of improvements throughthe visualization and debugging tools. Our code documentation and PyPIpackage are open-source and free to use under the MIT license. |


