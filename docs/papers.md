# cs.CL 

| Item |Content|
| --- |---|
|idx| 2411.04118v1 |
|title| Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress? |
|authors| Daniel P. JeongSaurabh GargZachary C. LiptonMichael Oberst
|links| http://arxiv.org/abs/2411.04118v1 |
|updated| 2024-11-06 18:51:02 UTC |
|summary| Several recent works seek to develop foundation models specifically formedical applications adapting general-purpose large language models LLMs andvision-language models VLMs via continued pretraining on publicly availablebiomedical corpora. These works typically claim that such domain-adaptivepretraining DAPT improves performance on downstream medical tasks such asanswering medical licensing exam questions. In this paper we compare sevenpublic medical LLMs and two VLMs against their corresponding base modelsarriving at a different conclusion: all medical VLMs and nearly all medicalLLMs fail to consistently improve over their base models in the zero-/few-shotprompting regime for medical question-answering QA tasks. For instanceacross the tasks and model pairs we consider in the 3-shot setting medicalLLMs only outperform their base models in 12.1 of cases reach a statisticaltie in 49.8 of cases and are significantly worse than their base models inthe remaining 38.2 of cases. Our conclusions are based on i comparing eachmedical model head-to-head directly against the corresponding base model iioptimizing the prompts for each model separately and iii accounting forstatistical uncertainty in comparisons. While these basic practices are notconsistently adopted in the literature our ablations show that theysubstantially impact conclusions. Our findings suggest that state-of-the-artgeneral-domain models may already exhibit strong medical knowledge andreasoning capabilities and offer recommendations to strengthen the conclusionsof future studies. |


| Item |Content|
| --- |---|
|idx| 2411.04109v1 |
|title| Self-Consistency Preference Optimization |
|authors| Archiki PrasadWeizhe YuanRichard Yuanzhe PangJing XuMaryam Fazel-ZarandiMohit BansalSainbayar SukhbaatarJason WestonJane Yu
|links| http://arxiv.org/abs/2411.04109v1 |
|updated| 2024-11-06 18:36:22 UTC |
|summary| Self-alignment whereby models learn to improve themselves without humanannotation is a rapidly growing research area. However existing techniquesoften fail to improve complex reasoning tasks due to the difficulty ofassigning correct rewards. An orthogonal approach that is known to improvecorrectness is self-consistency a method applied at inference time based onmultiple sampling in order to find the most consistent answer. In this work weextend the self-consistency concept to help train models. We thus introduceself-consistency preference optimization ScPO which iteratively trainsconsistent answers to be preferred over inconsistent ones on unsupervised newproblems. We show ScPO leads to large improvements over conventional rewardmodel training on reasoning tasks such as GSM8K and MATH closing the gap withsupervised training with gold answers or preferences and that combining ScPOwith standard supervised learning improves results even further. On ZebraLogicScPO finetunes Llama-3 8B to be superior to Llama-3 70B Gemma-2 27B andClaude-3 Haiku. |


| Item |Content|
| --- |---|
|idx| 2411.04105v2 |
|title| How Transformers Solve Propositional Logic Problems: A Mechanistic Analysis |
|authors| Guan Zhe HongNishanth DikkalaEnming LuoCyrus RashtchianXin WangRina Panigrahy
|links| http://arxiv.org/abs/2411.04105v2 |
|updated| 2024-11-07 03:50:19 UTC |
|summary| Large language models LLMs have shown amazing performance on tasks thatrequire planning and reasoning. Motivated by this we investigate the internalmechanisms that underpin a networks ability to perform complex logicalreasoning. We first construct a synthetic propositional logic problem thatserves as a concrete test-bed for network training and evaluation. Cruciallythis problem demands nontrivial planning to solve but we can train a smalltransformer to achieve perfect accuracy. Building on our set-up we then pursuean understanding of precisely how a three-layer transformer trained fromscratch solves this problem. We are able to identify certain planning andreasoning circuits in the network that necessitate cooperation between theattention blocks to implement the desired logic. To expand our findings wethen study a larger model Mistral 7B. Using activation patching wecharacterize internal components that are critical in solving our logicproblem. Overall our work systemically uncovers novel aspects of small andlarge transformers and continues the study of how they plan and reason. |


| Item |Content|
| --- |---|
|idx| 2411.04093v1 |
|title| Summarization of Opinionated Political Documents with Varied Perspectives |
|authors| Nicholas DeasKathleen McKeown
|links| http://arxiv.org/abs/2411.04093v1 |
|updated| 2024-11-06 18:14:48 UTC |
|summary| Global partisan hostility and polarization has increased and thispolarization is heightened around presidential elections. Models capable ofgenerating accurate summaries of diverse perspectives can help reduce suchpolarization by exposing users to alternative perspectives. In this work weintroduce a novel dataset and task for independently summarizing each politicalperspective in a set of passages from opinionated news articles. For this taskwe propose a framework for evaluating different dimensions of perspectivesummary performance. We benchmark 10 models of varying sizes and architecturesthrough both automatic and human evaluation. While recent models like GPT-4operform well on this task we find that all models struggle to generatesummaries faithful to the intended perspective. Our analysis of summariesfocuses on how extraction behavior depends on the features of the inputdocuments. |


| Item |Content|
| --- |---|
|idx| 2411.04090v2 |
|title| A Collaborative Content Moderation Framework for Toxicity Detection based on Conformalized Estimates of Annotation Disagreement |
|authors| Guillermo Villate-CastilloJavier Del SerBorja Sanz
|links| http://arxiv.org/abs/2411.04090v2 |
|updated| 2024-11-07 07:12:45 UTC |
|summary| Content moderation typically combines the efforts of human moderators andmachine learning models. However these systems often rely on data wheresignificant disagreement occurs during moderation reflecting the subjectivenature of toxicity perception. Rather than dismissing this disagreement asnoise we interpret it as a valuable signal that highlights the inherentambiguity of the contentan insight missed when only the majority label isconsidered. In this work we introduce a novel content moderation frameworkthat emphasizes the importance of capturing annotation disagreement. Ourapproach uses multitask learning where toxicity classification serves as theprimary task and annotation disagreement is addressed as an auxiliary task.Additionally we leverage uncertainty estimation techniques specificallyConformal Prediction to account for both the ambiguity in comment annotationsand the models inherent uncertainty in predicting toxicity anddisagreement.The framework also allows moderators to adjust thresholds forannotation disagreement offering flexibility in determining when ambiguityshould trigger a review. We demonstrate that our joint approach enhances modelperformance calibration and uncertainty estimation while offering greaterparameter efficiency and improving the review process in comparison tosingle-task methods. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2411.04118v1 |
|title| Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress? |
|authors| Daniel P. JeongSaurabh GargZachary C. LiptonMichael Oberst
|links| http://arxiv.org/abs/2411.04118v1 |
|updated| 2024-11-06 18:51:02 UTC |
|summary| Several recent works seek to develop foundation models specifically formedical applications adapting general-purpose large language models LLMs andvision-language models VLMs via continued pretraining on publicly availablebiomedical corpora. These works typically claim that such domain-adaptivepretraining DAPT improves performance on downstream medical tasks such asanswering medical licensing exam questions. In this paper we compare sevenpublic medical LLMs and two VLMs against their corresponding base modelsarriving at a different conclusion: all medical VLMs and nearly all medicalLLMs fail to consistently improve over their base models in the zero-/few-shotprompting regime for medical question-answering QA tasks. For instanceacross the tasks and model pairs we consider in the 3-shot setting medicalLLMs only outperform their base models in 12.1 of cases reach a statisticaltie in 49.8 of cases and are significantly worse than their base models inthe remaining 38.2 of cases. Our conclusions are based on i comparing eachmedical model head-to-head directly against the corresponding base model iioptimizing the prompts for each model separately and iii accounting forstatistical uncertainty in comparisons. While these basic practices are notconsistently adopted in the literature our ablations show that theysubstantially impact conclusions. Our findings suggest that state-of-the-artgeneral-domain models may already exhibit strong medical knowledge andreasoning capabilities and offer recommendations to strengthen the conclusionsof future studies. |


| Item |Content|
| --- |---|
|idx| 2411.04112v1 |
|title| Fed-EC: Bandwidth-Efficient Clustering-Based Federated Learning For Autonomous Visual Robot Navigation |
|authors| Shreya GummadiMateus V. GasparinoDeepak VasishtGirish Chowdhary
|links| http://arxiv.org/abs/2411.04112v1 |
|updated| 2024-11-06 18:44:09 UTC |
|summary| Centralized learning requires data to be aggregated at a central serverwhich poses significant challenges in terms of data privacy and bandwidthconsumption. Federated learning presents a compelling alternative howevervanilla federated learning methods deployed in robotics aim to learn a singleglobal model across robots that works ideally for all. But in practice onemodel may not be well suited for robots deployed in various environments. Thispaper proposes Federated-EmbedCluster Fed-EC a clustering-based federatedlearning framework that is deployed with vision based autonomous robotnavigation in diverse outdoor environments. The framework addresses the keyfederated learning challenge of deteriorating model performance of a singleglobal model due to the presence of non-IID data across real-world robots.Extensive real-world experiments validate that Fed-EC reduces the communicationsize by 23x for each robot while matching the performance of centralizedlearning for goal-oriented navigation and outperforms local learning. Fed-ECcan transfer previously learnt models to new robots that join the cluster. |


| Item |Content|
| --- |---|
|idx| 2411.04109v1 |
|title| Self-Consistency Preference Optimization |
|authors| Archiki PrasadWeizhe YuanRichard Yuanzhe PangJing XuMaryam Fazel-ZarandiMohit BansalSainbayar SukhbaatarJason WestonJane Yu
|links| http://arxiv.org/abs/2411.04109v1 |
|updated| 2024-11-06 18:36:22 UTC |
|summary| Self-alignment whereby models learn to improve themselves without humanannotation is a rapidly growing research area. However existing techniquesoften fail to improve complex reasoning tasks due to the difficulty ofassigning correct rewards. An orthogonal approach that is known to improvecorrectness is self-consistency a method applied at inference time based onmultiple sampling in order to find the most consistent answer. In this work weextend the self-consistency concept to help train models. We thus introduceself-consistency preference optimization ScPO which iteratively trainsconsistent answers to be preferred over inconsistent ones on unsupervised newproblems. We show ScPO leads to large improvements over conventional rewardmodel training on reasoning tasks such as GSM8K and MATH closing the gap withsupervised training with gold answers or preferences and that combining ScPOwith standard supervised learning improves results even further. On ZebraLogicScPO finetunes Llama-3 8B to be superior to Llama-3 70B Gemma-2 27B andClaude-3 Haiku. |


| Item |Content|
| --- |---|
|idx| 2411.04105v2 |
|title| How Transformers Solve Propositional Logic Problems: A Mechanistic Analysis |
|authors| Guan Zhe HongNishanth DikkalaEnming LuoCyrus RashtchianXin WangRina Panigrahy
|links| http://arxiv.org/abs/2411.04105v2 |
|updated| 2024-11-07 03:50:19 UTC |
|summary| Large language models LLMs have shown amazing performance on tasks thatrequire planning and reasoning. Motivated by this we investigate the internalmechanisms that underpin a networks ability to perform complex logicalreasoning. We first construct a synthetic propositional logic problem thatserves as a concrete test-bed for network training and evaluation. Cruciallythis problem demands nontrivial planning to solve but we can train a smalltransformer to achieve perfect accuracy. Building on our set-up we then pursuean understanding of precisely how a three-layer transformer trained fromscratch solves this problem. We are able to identify certain planning andreasoning circuits in the network that necessitate cooperation between theattention blocks to implement the desired logic. To expand our findings wethen study a larger model Mistral 7B. Using activation patching wecharacterize internal components that are critical in solving our logicproblem. Overall our work systemically uncovers novel aspects of small andlarge transformers and continues the study of how they plan and reason. |


| Item |Content|
| --- |---|
|idx| 2411.04097v1 |
|title| RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models |
|authors| Maya VarmaJean-Benoit DelbrouckZhihong ChenAkshay ChaudhariCurtis Langlotz
|links| http://arxiv.org/abs/2411.04097v1 |
|updated| 2024-11-06 18:25:00 UTC |
|summary| Fine-tuned vision-language models VLMs often capture spurious correlationsbetween image features and textual attributes resulting in degraded zero-shotperformance at test time. Existing approaches for addressing spuriouscorrelations i primarily operate at the global image-level rather thanintervening directly on fine-grained image features and ii are predominantlydesigned for unimodal settings. In this work we present RaVL which takes afine-grained perspective on VLM robustness by discovering and mitigatingspurious correlations using local image features rather than operating at theglobal image level. Given a fine-tuned VLM RaVL first discovers spuriouscorrelations by leveraging a region-level clustering approach to identifyprecise image features contributing to zero-shot classification errors. ThenRaVL mitigates the identified spurious correlation with a novel region-awareloss function that enables the VLM to focus on relevant regions and ignorespurious relationships during fine-tuning. We evaluate RaVL on 654 VLMs withvarious model architectures data domains and learned spurious correlations.Our results show that RaVL accurately discovers 191 improvement over theclosest baseline and mitigates 8.2 improvement on worst-group imageclassification accuracy spurious correlations. Qualitative evaluations ongeneral-domain and medical-domain VLMs confirm our findings. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2411.04118v1 |
|title| Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress? |
|authors| Daniel P. JeongSaurabh GargZachary C. LiptonMichael Oberst
|links| http://arxiv.org/abs/2411.04118v1 |
|updated| 2024-11-06 18:51:02 UTC |
|summary| Several recent works seek to develop foundation models specifically formedical applications adapting general-purpose large language models LLMs andvision-language models VLMs via continued pretraining on publicly availablebiomedical corpora. These works typically claim that such domain-adaptivepretraining DAPT improves performance on downstream medical tasks such asanswering medical licensing exam questions. In this paper we compare sevenpublic medical LLMs and two VLMs against their corresponding base modelsarriving at a different conclusion: all medical VLMs and nearly all medicalLLMs fail to consistently improve over their base models in the zero-/few-shotprompting regime for medical question-answering QA tasks. For instanceacross the tasks and model pairs we consider in the 3-shot setting medicalLLMs only outperform their base models in 12.1 of cases reach a statisticaltie in 49.8 of cases and are significantly worse than their base models inthe remaining 38.2 of cases. Our conclusions are based on i comparing eachmedical model head-to-head directly against the corresponding base model iioptimizing the prompts for each model separately and iii accounting forstatistical uncertainty in comparisons. While these basic practices are notconsistently adopted in the literature our ablations show that theysubstantially impact conclusions. Our findings suggest that state-of-the-artgeneral-domain models may already exhibit strong medical knowledge andreasoning capabilities and offer recommendations to strengthen the conclusionsof future studies. |


| Item |Content|
| --- |---|
|idx| 2411.04109v1 |
|title| Self-Consistency Preference Optimization |
|authors| Archiki PrasadWeizhe YuanRichard Yuanzhe PangJing XuMaryam Fazel-ZarandiMohit BansalSainbayar SukhbaatarJason WestonJane Yu
|links| http://arxiv.org/abs/2411.04109v1 |
|updated| 2024-11-06 18:36:22 UTC |
|summary| Self-alignment whereby models learn to improve themselves without humanannotation is a rapidly growing research area. However existing techniquesoften fail to improve complex reasoning tasks due to the difficulty ofassigning correct rewards. An orthogonal approach that is known to improvecorrectness is self-consistency a method applied at inference time based onmultiple sampling in order to find the most consistent answer. In this work weextend the self-consistency concept to help train models. We thus introduceself-consistency preference optimization ScPO which iteratively trainsconsistent answers to be preferred over inconsistent ones on unsupervised newproblems. We show ScPO leads to large improvements over conventional rewardmodel training on reasoning tasks such as GSM8K and MATH closing the gap withsupervised training with gold answers or preferences and that combining ScPOwith standard supervised learning improves results even further. On ZebraLogicScPO finetunes Llama-3 8B to be superior to Llama-3 70B Gemma-2 27B andClaude-3 Haiku. |


| Item |Content|
| --- |---|
|idx| 2411.04108v1 |
|title| Weighted Sobolev Approximation Rates for Neural Networks on Unbounded Domains |
|authors| Ahmed AbdeljawadThomas Dittrich
|links| http://arxiv.org/abs/2411.04108v1 |
|updated| 2024-11-06 18:36:21 UTC |
|summary| In this work we consider the approximation capabilities of shallow neuralnetworks in weighted Sobolev spaces for functions in the spectral Barron space.The existing literature already covers several cases in which the spectralBarron space can be approximated well i.e. without curse of dimensionalityby shallow networks and several different classes of activation function. Thelimitations of the existing results are mostly on the error measures that wereconsidered in which the results are restricted to Sobolev spaces over abounded domain. We will here treat two cases that extend upon the existingresults. Namely we treat the case with bounded domain and Muckenhoupt weightsand the case where the domain is allowed to be unbounded and the weights arerequired to decay. We first present embedding results for the more generalweighted Fourier-Lebesgue spaces in the weighted Sobolev spaces and then weestablish asymptotic approximation rates for shallow neural networks that comewithout curse of dimensionality. |


| Item |Content|
| --- |---|
|idx| 2411.04106v1 |
|title| A Comparative Study of Deep Reinforcement Learning for Crop Production Management |
|authors| Joseph BalderasDong ChenYanbo HuangLi WangRen-Cang Li
|links| http://arxiv.org/abs/2411.04106v1 |
|updated| 2024-11-06 18:35:51 UTC |
|summary| Crop production management is essential for optimizing yield and minimizing afields environmental impact to crop fields yet it remains challenging due tothe complex and stochastic processes involved. Recently researchers haveturned to machine learning to address these complexities. Specificallyreinforcement learning RL a cutting-edge approach designed to learn optimaldecision-making strategies through trial and error in dynamic environments hasemerged as a promising tool for developing adaptive crop management policies.RL models aim to optimize long-term rewards by continuously interacting withthe environment making them well-suited for tackling the uncertainties andvariability inherent in crop management. Studies have shown that RL cangenerate crop management policies that compete with and even outperformexpert-designed policies within simulation-based crop models. In the gym-DSSATcrop model environment one of the most widely used simulators for cropmanagement proximal policy optimization PPO and deep Q-networks DQN haveshown promising results. However these methods have not yet beensystematically evaluated under identical conditions. In this study weevaluated PPO and DQN against static baseline policies across three differentRL tasks fertilization irrigation and mixed management provided by thegym-DSSAT environment. To ensure a fair comparison we used consistent defaultparameters identical reward functions and the same environment settings. Ourresults indicate that PPO outperforms DQN in fertilization and irrigationtasks while DQN excels in the mixed management task. This comparative analysisprovides critical insights into the strengths and limitations of each approachadvancing the development of more effective RL-based crop managementstrategies. |


| Item |Content|
| --- |---|
|idx| 2411.04105v2 |
|title| How Transformers Solve Propositional Logic Problems: A Mechanistic Analysis |
|authors| Guan Zhe HongNishanth DikkalaEnming LuoCyrus RashtchianXin WangRina Panigrahy
|links| http://arxiv.org/abs/2411.04105v2 |
|updated| 2024-11-07 03:50:19 UTC |
|summary| Large language models LLMs have shown amazing performance on tasks thatrequire planning and reasoning. Motivated by this we investigate the internalmechanisms that underpin a networks ability to perform complex logicalreasoning. We first construct a synthetic propositional logic problem thatserves as a concrete test-bed for network training and evaluation. Cruciallythis problem demands nontrivial planning to solve but we can train a smalltransformer to achieve perfect accuracy. Building on our set-up we then pursuean understanding of precisely how a three-layer transformer trained fromscratch solves this problem. We are able to identify certain planning andreasoning circuits in the network that necessitate cooperation between theattention blocks to implement the desired logic. To expand our findings wethen study a larger model Mistral 7B. Using activation patching wecharacterize internal components that are critical in solving our logicproblem. Overall our work systemically uncovers novel aspects of small andlarge transformers and continues the study of how they plan and reason. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2411.04125v1 |
|title| Community Forensics: Using Thousands of Generators to Train Fake Image Detectors |
|authors| Jeongsoo ParkAndrew Owens
|links| http://arxiv.org/abs/2411.04125v1 |
|updated| 2024-11-06 18:59:41 UTC |
|summary| One of the key challenges of detecting AI-generated images is spotting imagesthat have been created by previously unseen generative models. We argue thatthe limited diversity of the training data is a major obstacle to addressingthis problem and we propose a new dataset that is significantly larger andmore diverse than prior work. As part of creating this dataset wesystematically download thousands of text-to-image latent diffusion models andsample images from them. We also collect images from dozens of popular opensource and commercial models. The resulting dataset contains 2.7M images thathave been sampled from 4803 different models. These images collectively capturea wide range of scene content generator architectures and image processingsettings. Using this dataset we study the generalization abilities of fakeimage detectors. Our experiments suggest that detection performance improves asthe number of models in the training set increases even when these models havesimilar architectures. We also find that detection performance improves as thediversity of the models increases and that our trained detectors generalizebetter than those trained on other datasets. |


| Item |Content|
| --- |---|
|idx| 2411.04112v1 |
|title| Fed-EC: Bandwidth-Efficient Clustering-Based Federated Learning For Autonomous Visual Robot Navigation |
|authors| Shreya GummadiMateus V. GasparinoDeepak VasishtGirish Chowdhary
|links| http://arxiv.org/abs/2411.04112v1 |
|updated| 2024-11-06 18:44:09 UTC |
|summary| Centralized learning requires data to be aggregated at a central serverwhich poses significant challenges in terms of data privacy and bandwidthconsumption. Federated learning presents a compelling alternative howevervanilla federated learning methods deployed in robotics aim to learn a singleglobal model across robots that works ideally for all. But in practice onemodel may not be well suited for robots deployed in various environments. Thispaper proposes Federated-EmbedCluster Fed-EC a clustering-based federatedlearning framework that is deployed with vision based autonomous robotnavigation in diverse outdoor environments. The framework addresses the keyfederated learning challenge of deteriorating model performance of a singleglobal model due to the presence of non-IID data across real-world robots.Extensive real-world experiments validate that Fed-EC reduces the communicationsize by 23x for each robot while matching the performance of centralizedlearning for goal-oriented navigation and outperforms local learning. Fed-ECcan transfer previously learnt models to new robots that join the cluster. |


| Item |Content|
| --- |---|
|idx| 2411.04097v1 |
|title| RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models |
|authors| Maya VarmaJean-Benoit DelbrouckZhihong ChenAkshay ChaudhariCurtis Langlotz
|links| http://arxiv.org/abs/2411.04097v1 |
|updated| 2024-11-06 18:25:00 UTC |
|summary| Fine-tuned vision-language models VLMs often capture spurious correlationsbetween image features and textual attributes resulting in degraded zero-shotperformance at test time. Existing approaches for addressing spuriouscorrelations i primarily operate at the global image-level rather thanintervening directly on fine-grained image features and ii are predominantlydesigned for unimodal settings. In this work we present RaVL which takes afine-grained perspective on VLM robustness by discovering and mitigatingspurious correlations using local image features rather than operating at theglobal image level. Given a fine-tuned VLM RaVL first discovers spuriouscorrelations by leveraging a region-level clustering approach to identifyprecise image features contributing to zero-shot classification errors. ThenRaVL mitigates the identified spurious correlation with a novel region-awareloss function that enables the VLM to focus on relevant regions and ignorespurious relationships during fine-tuning. We evaluate RaVL on 654 VLMs withvarious model architectures data domains and learned spurious correlations.Our results show that RaVL accurately discovers 191 improvement over theclosest baseline and mitigates 8.2 improvement on worst-group imageclassification accuracy spurious correlations. Qualitative evaluations ongeneral-domain and medical-domain VLMs confirm our findings. |


| Item |Content|
| --- |---|
|idx| 2411.04079v1 |
|title| Textual Decomposition Then Sub-motion-space Scattering for Open-Vocabulary Motion Generation |
|authors| Ke FanJiangning ZhangRan YiJingyu GongYabiao WangYating WangXin TanChengjie WangLizhuang Ma
|links| http://arxiv.org/abs/2411.04079v1 |
|updated| 2024-11-06 17:57:43 UTC |
|summary| Text-to-motion generation is a crucial task in computer vision whichgenerates the target 3D motion by the given text. The existing annotateddatasets are limited in scale resulting in most existing methods overfittingto the small datasets and unable to generalize to the motions of the opendomain. Some methods attempt to solve the open-vocabulary motion generationproblem by aligning to the CLIP space or using the Pretrain-then-Finetuningparadigm. However the current annotated datasets limited scale only allowsthem to achieve mapping from sub-text-space to sub-motion-space instead ofmapping between full-text-space and full-motion-space full mapping which isthe key to attaining open-vocabulary motion generation. To this end this paperproposes to leverage the atomic motion simple body part motions over a shorttime period as an intermediate representation and leverage two orderlycoupled steps i.e. Textual Decomposition and Sub-motion-space Scattering toaddress the full mapping problem. For Textual Decomposition we design afine-grained description conversion algorithm and combine it with thegeneralization ability of a large language model to convert any given motiontext into atomic texts. Sub-motion-space Scattering learns the compositionalprocess from atomic motions to the target motions to make the learnedsub-motion-space scattered to form the full-motion-space. For a given motion ofthe open domain it transforms the extrapolation into interpolation and therebysignificantly improves generalization. Our network DSO-Net combines textualdecomposition and sub-motion-space scattering to solve theopen-vocabulary motion generation. Extensive experiments demonstrate that ourDSO-Net achieves significant improvements over the state-of-the-art methods onopen-vocabulary motion generation. Code is available athttps://vankouf.github.io/DSONet/. |


| Item |Content|
| --- |---|
|idx| 2411.04077v1 |
|title| H-POPE: Hierarchical Polling-based Probing Evaluation of Hallucinations in Large Vision-Language Models |
|authors| Nhi PhamMichael Schott
|links| http://arxiv.org/abs/2411.04077v1 |
|updated| 2024-11-06 17:55:37 UTC |
|summary| By leveraging both texts and images large vision language models LVLMshave shown significant progress in various multi-modal tasks. Neverthelessthese models often suffer from hallucinations e.g. they exhibitinconsistencies between the visual input and the textual output. To addressthis we propose H-POPE a coarse-to-fine-grained benchmark that systematicallyassesses hallucination in object existence and attributes. Our evaluation showsthat models are prone to hallucinations on object existence and even more soon fine-grained attributes. We further investigate whether these models rely onvisual input to formulate the output texts. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2411.04108v1 |
|title| Weighted Sobolev Approximation Rates for Neural Networks on Unbounded Domains |
|authors| Ahmed AbdeljawadThomas Dittrich
|links| http://arxiv.org/abs/2411.04108v1 |
|updated| 2024-11-06 18:36:21 UTC |
|summary| In this work we consider the approximation capabilities of shallow neuralnetworks in weighted Sobolev spaces for functions in the spectral Barron space.The existing literature already covers several cases in which the spectralBarron space can be approximated well i.e. without curse of dimensionalityby shallow networks and several different classes of activation function. Thelimitations of the existing results are mostly on the error measures that wereconsidered in which the results are restricted to Sobolev spaces over abounded domain. We will here treat two cases that extend upon the existingresults. Namely we treat the case with bounded domain and Muckenhoupt weightsand the case where the domain is allowed to be unbounded and the weights arerequired to decay. We first present embedding results for the more generalweighted Fourier-Lebesgue spaces in the weighted Sobolev spaces and then weestablish asymptotic approximation rates for shallow neural networks that comewithout curse of dimensionality. |


| Item |Content|
| --- |---|
|idx| 2411.04054v1 |
|title| Partial Structure Discovery is Sufficient for No-regret Learning in Causal Bandits |
|authors| Muhammad Qasim ElahiMahsa GhasemiMurat Kocaoglu
|links| http://arxiv.org/abs/2411.04054v1 |
|updated| 2024-11-06 16:59:11 UTC |
|summary| Causal knowledge about the relationships among decision variables and areward variable in a bandit setting can accelerate the learning of an optimaldecision. Current works often assume the causal graph is known which may notalways be available a priori. Motivated by this challenge we focus on thecausal bandit problem in scenarios where the underlying causal graph is unknownand may include latent confounders. While intervention on the parents of thereward node is optimal in the absence of latent confounders this is notnecessarily the case in general. Instead one must consider a set of possiblyoptimal arms/interventions each being a special subset of the ancestors of thereward node making causal discovery beyond the parents of the reward nodeessential. For regret minimization we identify that discovering the fullcausal structure is unnecessary however no existing work provides thenecessary and sufficient components of the causal graph. We formallycharacterize the set of necessary and sufficient latent confounders one needsto detect or learn to ensure that all possibly optimal arms are identifiedcorrectly. We also propose a randomized algorithm for learning the causal graphwith a limited number of samples providing a sample complexity guarantee forany desired confidence level. In the causal bandit setup we propose atwo-stage approach. In the first stage we learn the induced subgraph onancestors of the reward along with a necessary and sufficient subset of latentconfounders to construct the set of possibly optimal arms. The regret incurredduring this phase scales polynomially with respect to the number of nodes inthe causal graph. The second phase involves the application of a standardbandit algorithm such as the UCB algorithm. We also establish a regret boundfor our two-phase approach which is sublinear in the number of rounds. |


| Item |Content|
| --- |---|
|idx| 2411.03936v1 |
|title| GUIDE-VAE: Advancing Data Generation with User Information and Pattern Dictionaries |
|authors| Kutay BölatSimon Tindemans
|links| http://arxiv.org/abs/2411.03936v1 |
|updated| 2024-11-06 14:11:46 UTC |
|summary| Generative modelling of multi-user datasets has become prominent in scienceand engineering. Generating a data point for a given user requires employinguser information and conventional generative models including variationalautoencoders VAEs often ignore that. This paper introduces GUIDE-VAE anovel conditional generative model that leverages user embeddings to generateuser-guided data. By allowing the model to benefit from shared patterns acrossusers GUIDE-VAE enhances performance in multi-user settings even undersignificant data imbalance. In addition to integrating user informationGUIDE-VAE incorporates a pattern dictionary-based covariance composition PDCCto improve the realism of generated samples by capturing complex featuredependencies. While user embeddings drive performance gains PDCC addressescommon issues such as noise and over-smoothing typically seen in VAEs.  The proposed GUIDE-VAE was evaluated on a multi-user smart meter datasetcharacterized by substantial data imbalance across users. Quantitative resultsshow that GUIDE-VAE performs effectively in both synthetic data generation andmissing record imputation tasks while qualitative evaluations reveal thatGUIDE-VAE produces more plausible and less noisy data. These results establishGUIDE-VAE as a promising tool for controlled realistic data generation inmulti-user datasets with potential applications across various domainsrequiring user-informed modelling. |


| Item |Content|
| --- |---|
|idx| 2411.03932v1 |
|title| Improved Regret of Linear Ensemble Sampling |
|authors| Harin LeeMin-hwan Oh
|links| http://arxiv.org/abs/2411.03932v1 |
|updated| 2024-11-06 14:09:11 UTC |
|summary| In this work we close the fundamental gap of theory and practice byproviding an improved regret bound for linear ensemble sampling. We prove thatwith an ensemble size logarithmic in T linear ensemble sampling can achievea frequentist regret bound of tildemathcalOd3/2sqrtT matchingstate-of-the-art results for randomized linear bandit algorithms where d andT are the dimension of the parameter and the time horizon respectively. Ourapproach introduces a general regret analysis framework for linear banditalgorithms. Additionally we reveal a significant relationship between linearensemble sampling and Linear Perturbed-History Exploration LinPHE showingthat LinPHE is a special case of linear ensemble sampling when the ensemblesize equals T. This insight allows us to derive a new regret bound oftildemathcalOd3/2sqrtT for LinPHE independent of the number ofarms. Our contributions advance the theoretical foundation of ensemblesampling bringing its regret bounds in line with the best known bounds forother randomized exploration algorithms. |


| Item |Content|
| --- |---|
|idx| 2411.03810v1 |
|title| Hybrid Transfer Reinforcement Learning: Provable Sample Efficiency from Shifted-Dynamics Data |
|authors| Chengrui QuLaixi ShiKishan PanagantiPengcheng YouAdam Wierman
|links| http://arxiv.org/abs/2411.03810v1 |
|updated| 2024-11-06 10:14:46 UTC |
|summary| Online Reinforcement learning RL typically requires high-stakes onlineinteraction data to learn a policy for a target task. This prompts interest inleveraging historical data to improve sample efficiency. The historical datamay come from outdated or related source environments with different dynamics.It remains unclear how to effectively use such data in the target task toprovably enhance learning and sample efficiency. To address this we propose ahybrid transfer RL HTRL setting where an agent learns in a targetenvironment while accessing offline data from a source environment with shifteddynamics. We show that -- without information on the dynamics shift -- generalshifted-dynamics data even with subtle shifts does not reduce samplecomplexity in the target environment. However with prior information on thedegree of the dynamics shift we design HySRL a transfer algorithm thatachieves problem-dependent sample complexity and outperforms pure online RL.Finally our experimental results demonstrate that HySRL surpassesstate-of-the-art online RL baseline. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2411.04090v2 |
|title| A Collaborative Content Moderation Framework for Toxicity Detection based on Conformalized Estimates of Annotation Disagreement |
|authors| Guillermo Villate-CastilloJavier Del SerBorja Sanz
|links| http://arxiv.org/abs/2411.04090v2 |
|updated| 2024-11-07 07:12:45 UTC |
|summary| Content moderation typically combines the efforts of human moderators andmachine learning models. However these systems often rely on data wheresignificant disagreement occurs during moderation reflecting the subjectivenature of toxicity perception. Rather than dismissing this disagreement asnoise we interpret it as a valuable signal that highlights the inherentambiguity of the contentan insight missed when only the majority label isconsidered. In this work we introduce a novel content moderation frameworkthat emphasizes the importance of capturing annotation disagreement. Ourapproach uses multitask learning where toxicity classification serves as theprimary task and annotation disagreement is addressed as an auxiliary task.Additionally we leverage uncertainty estimation techniques specificallyConformal Prediction to account for both the ambiguity in comment annotationsand the models inherent uncertainty in predicting toxicity anddisagreement.The framework also allows moderators to adjust thresholds forannotation disagreement offering flexibility in determining when ambiguityshould trigger a review. We demonstrate that our joint approach enhances modelperformance calibration and uncertainty estimation while offering greaterparameter efficiency and improving the review process in comparison tosingle-task methods. |


| Item |Content|
| --- |---|
|idx| 2411.04037v2 |
|title| Taming Toxicity or Fueling It? The Great Ban`s Role in Shifting Toxic User Behavior and Engagement |
|authors| Lorenzo CimaBenedetta TessaStefano CresciAmaury TrujilloMarco Avvenuti
|links| http://arxiv.org/abs/2411.04037v2 |
|updated| 2024-11-07 08:26:32 UTC |
|summary| In todays online environments users experience harm and abuse on a dailybasis. Therefore content moderation is crucial to ensure their safety andwell-being. However the effectiveness of many moderation interventions isstill uncertain. We evaluate the effectiveness of The Great Ban one of thelargest deplatforming interventions carried out by Reddit that affected almost2000 communities. We analyze 53M comments shared by nearly 34K usersproviding in-depth results on both the intended and unintended consequences ofthis ban. We found that 15.6 of the moderated users abandoned the platformwhile the remaining ones decreased their overall toxicity by 4.1. Nonethelessa subset of those users increased their toxicity by 70 after the intervention.In any case increases in toxicity did not lead to marked increases in activityor engagement meaning that the most toxic users had overall a limited impact.Our findings bring to light new insights on the effectiveness of deplatforming.Furthermore they also contribute to informing future content moderationstrategies. |


| Item |Content|
| --- |---|
|idx| 2411.03885v1 |
|title| Disability data futures: Achievable imaginaries for AI and disability data justice |
|authors| Denis Newman-GriffisBonnielin SwenorRupa ValdezGillian Mason
|links| http://arxiv.org/abs/2411.03885v1 |
|updated| 2024-11-06 13:04:29 UTC |
|summary| Data are the medium through which individuals identities and experiences arefiltered in contemporary states and systems and AI is increasingly the layermediating between people data and decisions. The history of data and AI isoften one of disability exclusion oppression and the reduction of disabledexperience left unchallenged the current proliferation of AI and data systemsthus risks further automating ableism behind the veneer of algorithmicneutrality. However exclusionary histories do not preclude inclusive futuresand disability-led visions can chart new paths for collective action to achievefutures founded in disability justice. This chapter brings together fouracademics and disability advocates working at the nexus of disability dataand AI to describe achievable imaginaries for artificial intelligence anddisability data justice. Reflecting diverse contexts disciplinaryperspectives and personal experiences we draw out the shape actors andgoals of imagined future systems where data and AI support movement towardsdisability justice. |


| Item |Content|
| --- |---|
|idx| 2411.03827v1 |
|title| DesignMinds: Enhancing Video-Based Design Ideation with Vision-Language Model and Context-Injected Large Language Model |
|authors| Tianhao HeAndrija StankovicEvangelos NiforatosGerd Kortuem
|links| http://arxiv.org/abs/2411.03827v1 |
|updated| 2024-11-06 11:00:44 UTC |
|summary| Ideation is a critical component of video-based design VBD where videosserve as the primary medium for design exploration and inspiration. Theemergence of generative AI offers considerable potential to enhance thisprocess by streamlining video analysis and facilitating idea generation. Inthis paper we present DesignMinds a prototype that integrates astate-of-the-art Vision-Language Model VLM with a context-enhanced LargeLanguage Model LLM to support ideation in VBD. To evaluate DesignMinds weconducted a between-subject study with 35 design practitioners comparing itsperformance to a baseline condition. Our results demonstrate that DesignMindssignificantly enhances the flexibility and originality of ideation while alsoincreasing task engagement. Importantly the introduction of this technologydid not negatively impact user experience technology acceptance or usability. |


| Item |Content|
| --- |---|
|idx| 2411.03817v1 |
|title| From Novice to Expert: LLM Agent Policy Optimization via Step-wise Reinforcement Learning |
|authors| Zhirui DengZhicheng DouYutao ZhuJi-Rong WenRuibin XiongMang WangWeipeng Chen
|links| http://arxiv.org/abs/2411.03817v1 |
|updated| 2024-11-06 10:35:11 UTC |
|summary| The outstanding capabilities of large language models LLMs render them acrucial component in various autonomous agent systems. While traditionalmethods depend on the inherent knowledge of LLMs without fine-tuning morerecent approaches have shifted toward the reinforcement learning strategy tofurther enhance agents ability to solve complex interactive tasks withenvironments and tools. However previous approaches are constrained by thesparse reward issue where existing datasets solely provide a final scalarreward for each multi-step reasoning chain potentially leading toineffectiveness and inefficiency in policy learning. In this paper weintroduce StepAgent which utilizes step-wise reward to optimize the agentsreinforcement learning process. Inheriting the spirit of novice-to-experttheory we first compare the actions of the expert and the agent toautomatically generate intermediate rewards for fine-grained optimization.Additionally we propose implicit-reward and inverse reinforcement learningtechniques to facilitate agent reflection and policy adjustment. Furthertheoretical analysis demonstrates that the action distribution of the agent canconverge toward the expert action distribution over multiple training cycles.Experimental results across various datasets indicate that StepAgentoutperforms existing baseline methods. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2411.04073v1 |
|title| Rescheduling after vehicle failures in the multi-depot rural postman problem with rechargeable and reusable vehicles |
|authors| Eashwar SathyamurthyJeffrey W. HerrmannShapour Azarm
|links| http://arxiv.org/abs/2411.04073v1 |
|updated| 2024-11-06 17:50:32 UTC |
|summary| We present a centralized auction algorithm to solve the Multi-Depot RuralPostman Problem with Rechargeable and Reusable Vehicles MD-RPP-RRV focusingon rescheduling arc routing after vehicle failures. The problem involvesfinding heuristically obtained best feasible routes for multiple rechargeableand reusable vehicles with capacity constraints capable of performing multipletrips from multiple depots with the possibility of vehicle failures. Ouralgorithm auctions the failed trips to active non-failed vehicles throughlocal auctioning modifying initial routes to handle dynamic vehicle failuresefficiently. When a failure occurs the algorithm searches for the best activevehicle to perform the failed trip and inserts the trip into that vehiclesroute which avoids a complete rescheduling and reduces the computationaleffort. We compare the algorithms solutions against offline optimal solutionsobtained from solving a Mixed Integer Linear Programming MILP formulationusing the Gurobi solver this formulation assumes that perfect informationabout the vehicle failures and failure times is given. The results demonstratethat the centralized auction algorithm produces solutions that are in somecases near optimal moreover the execution time for the proposed approach ismuch more consistent and is for some instances orders of magnitude less thanthe execution time of the Gurobi solver. The theoretical analysis provides anupper bound for the competitive ratio and computational complexity of ouralgorithm offering a formal performance guarantee in dynamic failurescenarios. |


| Item |Content|
| --- |---|
|idx| 2411.03865v1 |
|title| AdaSociety: An Adaptive Environment with Social Structures for Multi-Agent Decision-Making |
|authors| Yizhe HuangXingbo WangHao LiuFanqi KongAoyang QinMin TangXiaoxi WangSong-Chun ZhuMingjie BiSiyuan QiXue Feng
|links| http://arxiv.org/abs/2411.03865v1 |
|updated| 2024-11-06 12:19:01 UTC |
|summary| Traditional interactive environments limit agents intelligence growth withfixed tasks. Recently single-agent environments address this by generating newtasks based on agent actions enhancing task diversity. We consider thedecision-making problem in multi-agent settings where tasks are furtherinfluenced by social connections affecting rewards and information access.However existing multi-agent environments lack a combination of adaptivephysical surroundings and social connections hindering the learning ofintelligent behaviors. To address this we introduce AdaSociety a customizablemulti-agent environment featuring expanding state and action spaces alongsideexplicit and alterable social structures. As agents progress the environmentadaptively generates new tasks with social structures for agents to undertake.In AdaSociety we develop three mini-games showcasing distinct socialstructures and tasks. Initial results demonstrate that specific socialstructures can promote both individual and collective benefits though currentreinforcement learning and LLM-based algorithms show limited effectiveness inleveraging social structures to enhance performance. Overall AdaSociety servesas a valuable research platform for exploring intelligence in diverse physicaland social settings. The code is available athttps://github.com/bigai-ai/AdaSociety. |


| Item |Content|
| --- |---|
|idx| 2411.03603v1 |
|title| CPEG: Leveraging Consistency Policy with Consensus Guidance for Multi-agent Exploration |
|authors| Yuqian FuYuanheng ZhuHaoran LiZijie ZhaoJiajun ChaiDongbin Zhao
|links| http://arxiv.org/abs/2411.03603v1 |
|updated| 2024-11-06 01:40:21 UTC |
|summary| Efficient exploration is crucial in cooperative multi-agent reinforcementlearning MARL especially in sparse-reward settings. However due to thereliance on the unimodal policy existing methods are prone to falling into thelocal optima hindering the effective exploration of better policies.Furthermore tackling multi-agent tasks in complex environments requirescooperation during exploration posing substantial challenges for MARL methods.To address these issues we propose a Consistency Policy with consEnsusGuidance CPEG with two primary components: a introducing a multimodalpolicy to enhance exploration capabilities and b sharing the consensus amongagents to foster agent cooperation. For component a CPEG incorporates aConsistency model as the policy leveraging its multimodal nature andstochastic characteristics to facilitate exploration. Regarding component bCPEG introduces a Consensus Learner to deduce the consensus on the global statefrom local observations. This consensus then serves as a guidance for theConsistency Policy promoting cooperation among agents. The proposed method isevaluated in multi-agent particle environments MPE and multi-agent MuJoCoMAMuJoCo and empirical results indicate that CPEG not only achievesimprovements in sparse-reward settings but also matches the performance ofbaselines in dense-reward environments. |


| Item |Content|
| --- |---|
|idx| 2411.03519v1 |
|title| AI Metropolis: Scaling Large Language Model-based Multi-Agent Simulation with Out-of-order Execution |
|authors| Zhiqiang XieHao KangYing ShengTushar KrishnaKayvon FatahalianChristos Kozyrakis
|links| http://arxiv.org/abs/2411.03519v1 |
|updated| 2024-11-05 21:54:14 UTC |
|summary| With more advanced natural language understanding and reasoning capabilitieslarge language model LLM-powered agents are increasingly developed insimulated environments to perform complex tasks interact with other agentsand exhibit emergent behaviors relevant to social science and gaming. Howevercurrent multi-agent simulations frequently suffer from inefficiencies due tothe limited parallelism caused by false dependencies resulting in performancebottlenecks. In this paper we introduce AI Metropolis a simulation enginethat improves the efficiency of LLM agent simulations by incorporatingout-of-order execution scheduling. By dynamically tracking real dependenciesbetween agents AI Metropolis minimizes false dependencies enhancingparallelism and enabling efficient hardware utilization. Our evaluationsdemonstrate that AI Metropolis achieves speedups from 1.3x to 4.15x overstandard parallel simulation with global synchronization approaching optimalperformance as the number of agents increases. |


| Item |Content|
| --- |---|
|idx| 2411.03284v1 |
|title| SMoA: Improving Multi-agent Large Language Models with Sparse Mixture-of-Agents |
|authors| Dawei LiZhen TanPeijia QianYifan LiKumar Satvik ChaudharyLijie HuJiayi Shen
|links| http://arxiv.org/abs/2411.03284v1 |
|updated| 2024-11-05 17:33:39 UTC |
|summary| While multi-agent systems have been shown to significantly enhance theperformance of Large Language Models LLMs across various tasks andapplications the dense interaction between scaling agents potentially hamperstheir efficiency and diversity. To address these challenges we drawinspiration from the sparse mixture-of-agents SMoE and propose a sparsemixture-of-agents SMoA framework to improve the efficiency and diversity ofmulti-agent LLMs. Unlike completely connected structures SMoA introduces novelResponse Selection and Early Stopping mechanisms to sparsify information flowsamong individual LLM agents striking a balance between performance andefficiency. Additionally inspired by the expert diversity principle in SMoEframeworks for workload balance between experts we assign distinct roledescriptions to each LLM agent fostering diverse and divergent thinking.Extensive experiments on reasoning alignment and fairness benchmarksdemonstrate that SMoA achieves performance comparable to traditionalmixture-of-agents approaches but with significantly lower computational costs.Further analysis reveals that SMoA is more stable has a greater capacity toscale and offers considerable potential through hyper-parameter optimization.Code and data will be available at: https://github.com/David-Li0406/SMoA. |


