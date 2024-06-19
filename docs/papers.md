# cs.CL 

| Item |Content|
| --- |---|
|idx| 2406.12845v1 |
|title| Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts |
|authors| Haoxiang WangWei XiongTengyang XieHan ZhaoTong Zhang
|links| http://arxiv.org/abs/2406.12845v1 |
|updated| 2024-06-18 17:58:28 UTC |
|summary| Reinforcement learning from human feedback RLHF has emerged as the primarymethod for aligning large language models LLMs with human preferences. TheRLHF process typically starts by training a reward model RM using humanpreference data. Conventional RMs are trained on pairwise responses to the sameuser request with relative ratings indicating which response humans prefer.The trained RM serves as a proxy for human preferences. However due to theblack-box nature of RMs their outputs lack interpretability as humans cannotintuitively understand why an RM thinks a response is good or not. As RMs actas human preference proxies we believe they should be human-interpretable toensure that their internal decision processes are consistent with humanpreferences and to prevent reward hacking in LLM alignment. To build RMs withinterpretable preferences we propose a two-stage approach: i train anAbsolute-Rating Multi-Objective Reward Model ArmoRM with multi-dimensionalabsolute-rating data each dimension corresponding to a human-interpretableobjective e.g. honesty verbosity safety ii employ a Mixture-of-ExpertsMoE strategy with a gating network that automatically selects the mostsuitable reward objectives based on the context. We efficiently trained anArmoRM with Llama-3 8B and a gating network consisting of a shallow MLP on topof the ArmoRM. Our trained model ArmoRM-Llama3-8B obtains state-of-the-artperformance on RewardBench a benchmark evaluating RMs for language modeling.Notably the performance of our model surpasses the LLM-as-a-judge method withGPT-4 judges by a margin and approaches the performance of the much largerNemotron-4 340B reward model. |


| Item |Content|
| --- |---|
|idx| 2406.12832v1 |
|title| LaMDA: Large Model Fine-Tuning via Spectrally Decomposed Low-Dimensional Adaptation |
|authors| Seyedarmin AziziSouvik KunduMassoud Pedram
|links| http://arxiv.org/abs/2406.12832v1 |
|updated| 2024-06-18 17:52:59 UTC |
|summary| Low-rank adaptation LoRA has become the default approach to fine-tune largelanguage models LLMs due to its significant reduction in trainableparameters. However trainable parameter demand for LoRA increases withincreasing model embedding dimensions leading to high compute costs.Additionally its backward updates require storing high-dimensionalintermediate activations and optimizer states demanding high peak GPU memory.In this paper we introduce large model fine-tuning via spectrally decomposedlow-dimensional adaptation LaMDA a novel approach to fine-tuning largelanguage models which leverages low-dimensional adaptation to achievesignificant reductions in trainable parameters and peak GPU memory footprint.LaMDA freezes a first projection matrix PMA in the adaptation path whileintroducing a low-dimensional trainable square matrix resulting in substantialreductions in trainable parameters and peak GPU memory usage. LaMDA graduallyfreezes a second projection matrix PMB during the early fine-tuning stagesreducing the compute cost associated with weight updates to enhance parameterefficiency further. We also present an enhancement LaMDA incorporating alite-weight adaptive rank allocation for the LoRA path via normalizedspectrum analysis of pre-trained model weights. We evaluate LaMDA/LaMDAacross various tasks including natural language understanding with the GLUEbenchmark text summarization natural language generation and complexreasoning on different LLMs. Results show that LaMDA matches or surpasses theperformance of existing alternatives while requiring up to 17.7x fewerparameter updates and up to 1.32x lower peak GPU memory usage duringfine-tuning. Code will be publicly available. |


| Item |Content|
| --- |---|
|idx| 2406.12830v1 |
|title| What Are the Odds? Language Models Are Capable of Probabilistic Reasoning |
|authors| Akshay ParuchuriJake GarrisonShun LiaoJohn HernandezJacob SunshineTim AlthoffXin LiuDaniel McDuff
|links| http://arxiv.org/abs/2406.12830v1 |
|updated| 2024-06-18 17:51:24 UTC |
|summary| Language models LM are capable of remarkably complex linguistic taskshowever numerical reasoning is an area in which they frequently struggle. Animportant but rarely evaluated form of reasoning is understanding probabilitydistributions. In this paper we focus on evaluating the probabilisticreasoning capabilities of LMs using idealized and real-world statisticaldistributions. We perform a systematic evaluation of state-of-the-art LMs onthree tasks: estimating percentiles drawing samples and calculatingprobabilities. We evaluate three ways to provide context to LMs 1 anchoringexamples from within a distribution or family of distributions 2 real-worldcontext 3 summary statistics on which to base a Normal approximation. Modelscan make inferences about distributions and can be further aided by theincorporation of real-world context example shots and simplified assumptionseven if these assumptions are incorrect or misspecified. To conduct this workwe developed a comprehensive benchmark distribution dataset with associatedquestion-answer pairs that we will release publicly. |


| Item |Content|
| --- |---|
|idx| 2406.12824v1 |
|title| From RAGs to rich parameters: Probing how language models utilize external knowledge over parametric information for factual queries |
|authors| Hitesh WadhwaRahul SeetharamanSomyaa AggarwalReshmi GhoshSamyadeep BasuSoundararajan SrinivasanWenlong ZhaoShreyas ChaudhariEhsan Aghazadeh
|links| http://arxiv.org/abs/2406.12824v1 |
|updated| 2024-06-18 17:46:08 UTC |
|summary| Retrieval Augmented Generation RAG enriches the ability of language modelsto reason using external context to augment responses for a given user prompt.This approach has risen in popularity due to practical applications in variousapplications of language models in search question/answering and chat-bots.However the exact nature of how this approach works isnt clearly understood.In this paper we mechanistically examine the RAG pipeline to highlight thatlanguage models take shortcut and have a strong bias towards utilizing only thecontext information to answer the question while relying minimally on theirparametric memory. We probe this mechanistic behavior in language models with:i Causal Mediation Analysis to show that the parametric memory is minimallyutilized when answering a question and ii Attention Contributions andKnockouts to show that the last token residual stream do not get enriched fromthe subject token in the question but gets enriched from other informativetokens in the context. We find this pronounced shortcut behaviour true acrossboth LLaMa and Phi family of models. |


| Item |Content|
| --- |---|
|idx| 2406.12822v1 |
|title| Is It Good Data for Multilingual Instruction Tuning or Just Bad Multilingual Evaluation for Large Language Models? |
|authors| Pinzhen ChenSimon YuZhicheng GuoBarry Haddow
|links| http://arxiv.org/abs/2406.12822v1 |
|updated| 2024-06-18 17:43:47 UTC |
|summary| Large language models particularly multilingual ones are designed claimedand expected to cater to native speakers of varied languages. We hypothesisethat the current practices of fine-tuning and evaluating these models maymismatch this intention owing to a heavy reliance on translation which canintroduce translation artefacts and defects. It remains unknown whether thenature of the instruction data has an impact on the model output on the otherhand it remains questionable whether translated test sets can capture suchnuances. Due to the often coupled practices of using translated data in bothstages such imperfections could have been overlooked. This work investigatesthese issues by using controlled native or translated data during instructiontuning and evaluation stages and observing model results. Experiments on eightbase models and eight different benchmarks reveal that native or generationbenchmarks display a notable difference between native and translatedinstruction data especially when model performance is high whereas other typesof test sets cannot. Finally we demonstrate that regularization is beneficialto bridging this gap on structured but not generative tasks. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2406.12844v1 |
|title| Synergizing Foundation Models and Federated Learning: A Survey |
|authors| Shenghui LiFanghua YeMeng FangJiaxu ZhaoYun-Hin ChanEdith C. -H. NgaiThiemo Voigt
|links| http://arxiv.org/abs/2406.12844v1 |
|updated| 2024-06-18 17:58:09 UTC |
|summary| The recent development of Foundation Models FMs represented by largelanguage models vision transformers and multimodal models has been making asignificant impact on both academia and industry. Compared with small-scalemodels FMs have a much stronger demand for high-volume data during thepre-training phase. Although general FMs can be pre-trained on data collectedfrom open sources such as the Internet domain-specific FMs need proprietarydata posing a practical challenge regarding the amount of data available dueto privacy concerns. Federated Learning FL is a collaborative learningparadigm that breaks the barrier of data availability from differentparticipants. Therefore it provides a promising solution to customize andadapt FMs to a wide range of domain-specific tasks using distributed datasetswhilst preserving privacy. This survey paper discusses the potentials andchallenges of synergizing FL and FMs and summarizes core techniques futuredirections and applications. A periodically updated paper collection on FM-FLis available at https://github.com/lishenghui/awesome-fm-fl. |


| Item |Content|
| --- |---|
|idx| 2406.12843v1 |
|title| Can Go AIs be adversarially robust? |
|authors| Tom TsengEuan McLeanKellin PelrineTony T. WangAdam Gleave
|links| http://arxiv.org/abs/2406.12843v1 |
|updated| 2024-06-18 17:57:49 UTC |
|summary| Prior work found that superhuman Go AIs like KataGo can be defeated by simpleadversarial strategies. In this paper we study if simple defenses can improveKataGos worst-case performance. We test three natural defenses: adversarialtraining on hand-constructed positions iterated adversarial training andchanging the network architecture. We find that some of these defenses are ableto protect against previously discovered attacks. Unfortunately we also findthat none of these defenses are able to withstand adaptive attacks. Inparticular we are able to train new adversaries that reliably defeat ourdefended agents by causing them to blunder in ways humans would not. Ourresults suggest that building robust AI systems is challenging even in narrowdomains such as Go. For interactive examples of attacks and a link to ourcodebase see https://goattack.far.ai. |


| Item |Content|
| --- |---|
|idx| 2406.12841v1 |
|title| Demystifying Higher-Order Graph Neural Networks |
|authors| Maciej BestaFlorian ScheidlLukas GianinazziShachar KlaimanJürgen MüllerTorsten Hoefler
|links| http://arxiv.org/abs/2406.12841v1 |
|updated| 2024-06-18 17:57:11 UTC |
|summary| Higher-order graph neural networks HOGNNs are an important class of GNNmodels that harness polyadic relations between vertices beyond plain edges.They have been used to eliminate issues such as over-smoothing orover-squashing to significantly enhance the accuracy of GNN predictions toimprove the expressiveness of GNN architectures and for numerous other goals.A plethora of HOGNN models have been introduced and they come with diverseneural architectures and even with different notions of what thehigher-order means. This richness makes it very challenging to appropriatelyanalyze and compare HOGNN models and to decide in what scenario to usespecific ones. To alleviate this we first design an in-depth taxonomy and ablueprint for HOGNNs. This facilitates designing models that maximizeperformance. Then we use our taxonomy to analyze and compare the availableHOGNN models. The outcomes of our analysis are synthesized in a set of insightsthat help to select the most beneficial GNN model in a given scenario and acomprehensive list of challenges and opportunities for further research intomore powerful HOGNNs. |


| Item |Content|
| --- |---|
|idx| 2406.12835v1 |
|title| Influence Maximization via Graph Neural Bandits |
|authors| Yuting FengVincent Y. F. TanBogdan Cautis
|links| http://arxiv.org/abs/2406.12835v1 |
|updated| 2024-06-18 17:54:33 UTC |
|summary| We consider a ubiquitous scenario in the study of Influence MaximizationIM in which there is limited knowledge about the topology of the diffusionnetwork. We set the IM problem in a multi-round diffusion campaign aiming tomaximize the number of distinct users that are influenced. Leveraging thecapability of bandit algorithms to effectively balance the objectives ofexploration and exploitation as well as the expressivity of neural networksour study explores the application of neural bandit algorithms to the IMproblem. We propose the framework IM-GNB Influence Maximization with GraphNeural Bandits where we provide an estimate of the users probabilities ofbeing influenced by influencers also known as diffusion seeds. This initialestimate forms the basis for constructing both an exploitation graph and anexploration one. Subsequently IM-GNB handles the exploration-exploitationtradeoff by selecting seed nodes in real-time using Graph ConvolutionalNetworks GCN in which the pre-estimated graphs are employed to refine theinfluencers estimated rewards in each contextual setting. Through extensiveexperiments on two large real-world datasets we demonstrate the effectivenessof IM-GNB compared with other baseline methods significantly improving thespread outcome of such diffusion campaigns when the underlying network isunknown. |


| Item |Content|
| --- |---|
|idx| 2406.12832v1 |
|title| LaMDA: Large Model Fine-Tuning via Spectrally Decomposed Low-Dimensional Adaptation |
|authors| Seyedarmin AziziSouvik KunduMassoud Pedram
|links| http://arxiv.org/abs/2406.12832v1 |
|updated| 2024-06-18 17:52:59 UTC |
|summary| Low-rank adaptation LoRA has become the default approach to fine-tune largelanguage models LLMs due to its significant reduction in trainableparameters. However trainable parameter demand for LoRA increases withincreasing model embedding dimensions leading to high compute costs.Additionally its backward updates require storing high-dimensionalintermediate activations and optimizer states demanding high peak GPU memory.In this paper we introduce large model fine-tuning via spectrally decomposedlow-dimensional adaptation LaMDA a novel approach to fine-tuning largelanguage models which leverages low-dimensional adaptation to achievesignificant reductions in trainable parameters and peak GPU memory footprint.LaMDA freezes a first projection matrix PMA in the adaptation path whileintroducing a low-dimensional trainable square matrix resulting in substantialreductions in trainable parameters and peak GPU memory usage. LaMDA graduallyfreezes a second projection matrix PMB during the early fine-tuning stagesreducing the compute cost associated with weight updates to enhance parameterefficiency further. We also present an enhancement LaMDA incorporating alite-weight adaptive rank allocation for the LoRA path via normalizedspectrum analysis of pre-trained model weights. We evaluate LaMDA/LaMDAacross various tasks including natural language understanding with the GLUEbenchmark text summarization natural language generation and complexreasoning on different LLMs. Results show that LaMDA matches or surpasses theperformance of existing alternatives while requiring up to 17.7x fewerparameter updates and up to 1.32x lower peak GPU memory usage duringfine-tuning. Code will be publicly available. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2406.12845v1 |
|title| Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts |
|authors| Haoxiang WangWei XiongTengyang XieHan ZhaoTong Zhang
|links| http://arxiv.org/abs/2406.12845v1 |
|updated| 2024-06-18 17:58:28 UTC |
|summary| Reinforcement learning from human feedback RLHF has emerged as the primarymethod for aligning large language models LLMs with human preferences. TheRLHF process typically starts by training a reward model RM using humanpreference data. Conventional RMs are trained on pairwise responses to the sameuser request with relative ratings indicating which response humans prefer.The trained RM serves as a proxy for human preferences. However due to theblack-box nature of RMs their outputs lack interpretability as humans cannotintuitively understand why an RM thinks a response is good or not. As RMs actas human preference proxies we believe they should be human-interpretable toensure that their internal decision processes are consistent with humanpreferences and to prevent reward hacking in LLM alignment. To build RMs withinterpretable preferences we propose a two-stage approach: i train anAbsolute-Rating Multi-Objective Reward Model ArmoRM with multi-dimensionalabsolute-rating data each dimension corresponding to a human-interpretableobjective e.g. honesty verbosity safety ii employ a Mixture-of-ExpertsMoE strategy with a gating network that automatically selects the mostsuitable reward objectives based on the context. We efficiently trained anArmoRM with Llama-3 8B and a gating network consisting of a shallow MLP on topof the ArmoRM. Our trained model ArmoRM-Llama3-8B obtains state-of-the-artperformance on RewardBench a benchmark evaluating RMs for language modeling.Notably the performance of our model surpasses the LLM-as-a-judge method withGPT-4 judges by a margin and approaches the performance of the much largerNemotron-4 340B reward model. |


| Item |Content|
| --- |---|
|idx| 2406.12844v1 |
|title| Synergizing Foundation Models and Federated Learning: A Survey |
|authors| Shenghui LiFanghua YeMeng FangJiaxu ZhaoYun-Hin ChanEdith C. -H. NgaiThiemo Voigt
|links| http://arxiv.org/abs/2406.12844v1 |
|updated| 2024-06-18 17:58:09 UTC |
|summary| The recent development of Foundation Models FMs represented by largelanguage models vision transformers and multimodal models has been making asignificant impact on both academia and industry. Compared with small-scalemodels FMs have a much stronger demand for high-volume data during thepre-training phase. Although general FMs can be pre-trained on data collectedfrom open sources such as the Internet domain-specific FMs need proprietarydata posing a practical challenge regarding the amount of data available dueto privacy concerns. Federated Learning FL is a collaborative learningparadigm that breaks the barrier of data availability from differentparticipants. Therefore it provides a promising solution to customize andadapt FMs to a wide range of domain-specific tasks using distributed datasetswhilst preserving privacy. This survey paper discusses the potentials andchallenges of synergizing FL and FMs and summarizes core techniques futuredirections and applications. A periodically updated paper collection on FM-FLis available at https://github.com/lishenghui/awesome-fm-fl. |


| Item |Content|
| --- |---|
|idx| 2406.12843v1 |
|title| Can Go AIs be adversarially robust? |
|authors| Tom TsengEuan McLeanKellin PelrineTony T. WangAdam Gleave
|links| http://arxiv.org/abs/2406.12843v1 |
|updated| 2024-06-18 17:57:49 UTC |
|summary| Prior work found that superhuman Go AIs like KataGo can be defeated by simpleadversarial strategies. In this paper we study if simple defenses can improveKataGos worst-case performance. We test three natural defenses: adversarialtraining on hand-constructed positions iterated adversarial training andchanging the network architecture. We find that some of these defenses are ableto protect against previously discovered attacks. Unfortunately we also findthat none of these defenses are able to withstand adaptive attacks. Inparticular we are able to train new adversaries that reliably defeat ourdefended agents by causing them to blunder in ways humans would not. Ourresults suggest that building robust AI systems is challenging even in narrowdomains such as Go. For interactive examples of attacks and a link to ourcodebase see https://goattack.far.ai. |


| Item |Content|
| --- |---|
|idx| 2406.12841v1 |
|title| Demystifying Higher-Order Graph Neural Networks |
|authors| Maciej BestaFlorian ScheidlLukas GianinazziShachar KlaimanJürgen MüllerTorsten Hoefler
|links| http://arxiv.org/abs/2406.12841v1 |
|updated| 2024-06-18 17:57:11 UTC |
|summary| Higher-order graph neural networks HOGNNs are an important class of GNNmodels that harness polyadic relations between vertices beyond plain edges.They have been used to eliminate issues such as over-smoothing orover-squashing to significantly enhance the accuracy of GNN predictions toimprove the expressiveness of GNN architectures and for numerous other goals.A plethora of HOGNN models have been introduced and they come with diverseneural architectures and even with different notions of what thehigher-order means. This richness makes it very challenging to appropriatelyanalyze and compare HOGNN models and to decide in what scenario to usespecific ones. To alleviate this we first design an in-depth taxonomy and ablueprint for HOGNNs. This facilitates designing models that maximizeperformance. Then we use our taxonomy to analyze and compare the availableHOGNN models. The outcomes of our analysis are synthesized in a set of insightsthat help to select the most beneficial GNN model in a given scenario and acomprehensive list of challenges and opportunities for further research intomore powerful HOGNNs. |


| Item |Content|
| --- |---|
|idx| 2406.12839v1 |
|title| Evaluating the design space of diffusion-based generative models |
|authors| Yuqing WangYe HeMolei Tao
|links| http://arxiv.org/abs/2406.12839v1 |
|updated| 2024-06-18 17:56:10 UTC |
|summary| Most existing theoretical investigations of the accuracy of diffusion modelsalbeit significant assume the score function has been approximated to acertain accuracy and then use this a priori bound to control the error ofgeneration. This article instead provides a first quantitative understanding ofthe whole generation process i.e. both training and sampling. More preciselyit conducts a non-asymptotic convergence analysis of denoising score matchingunder gradient descent. In addition a refined sampling error analysis forvariance exploding models is also provided. The combination of these tworesults yields a full error analysis which elucidates again but this timetheoretically how to design the training and sampling processes for effectivegeneration. For instance our theory implies a preference toward noisedistribution and loss weighting that qualitatively agree with the ones used inKarras et al. 2022. It also provides some perspectives on why the time andvariance schedule used in Karras et al. 2022 could be better tuned than thepioneering version in Song et al. 2020. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2406.12849v1 |
|title| Depth Anywhere: Enhancing 360 Monocular Depth Estimation via Perspective Distillation and Unlabeled Data Augmentation |
|authors| Ning-Hsu WangYu-Lun Liu
|links| http://arxiv.org/abs/2406.12849v1 |
|updated| 2024-06-18 17:59:31 UTC |
|summary| Accurately estimating depth in 360-degree imagery is crucial for virtualreality autonomous navigation and immersive media applications. Existingdepth estimation methods designed for perspective-view imagery fail whenapplied to 360-degree images due to different camera projections anddistortions whereas 360-degree methods perform inferior due to the lack oflabeled data pairs. We propose a new depth estimation framework that utilizesunlabeled 360-degree data effectively. Our approach uses state-of-the-artperspective depth estimation models as teacher models to generate pseudo labelsthrough a six-face cube projection technique enabling efficient labeling ofdepth in 360-degree images. This method leverages the increasing availabilityof large datasets. Our approach includes two main stages: offline maskgeneration for invalid regions and an online semi-supervised joint trainingregime. We tested our approach on benchmark datasets such as Matterport3D andStanford2D3D showing significant improvements in depth estimation accuracyparticularly in zero-shot scenarios. Our proposed training pipeline can enhanceany 360 monocular depth estimator and demonstrates effective knowledge transferacross different camera projections and data types. See our project page forresults: https://albert100121.github.io/Depth-Anywhere/ |


| Item |Content|
| --- |---|
|idx| 2406.12847v1 |
|title| ChangeViT: Unleashing Plain Vision Transformers for Change Detection |
|authors| Duowang ZhuXiaohu HuangHaiyan HuangZhenfeng ShaoQimin Cheng
|links| http://arxiv.org/abs/2406.12847v1 |
|updated| 2024-06-18 17:59:08 UTC |
|summary| Change detection in remote sensing images is essential for trackingenvironmental changes on the Earths surface. Despite the success of visiontransformers ViTs as backbones in numerous computer vision applications theyremain underutilized in change detection where convolutional neural networksCNNs continue to dominate due to their powerful feature extractioncapabilities. In this paper our study uncovers ViTs unique advantage indiscerning large-scale changes a capability where CNNs fall short.Capitalizing on this insight we introduce ChangeViT a framework that adopts aplain ViT backbone to enhance the performance of large-scale changes. Thisframework is supplemented by a detail-capture module that generates detailedspatial features and a feature injector that efficiently integratesfine-grained spatial information into high-level semantic learning. The featureintegration ensures that ChangeViT excels in both detecting large-scale changesand capturing fine-grained details providing comprehensive change detectionacross diverse scales. Without bells and whistles ChangeViT achievesstate-of-the-art performance on three popular high-resolution datasets i.e.LEVIR-CD WHU-CD and CLCD and one low-resolution dataset i.e. OSCD whichunderscores the unleashed potential of plain ViTs for change detection.Furthermore thorough quantitative and qualitative analyses validate theefficacy of the introduced modules solidifying the effectiveness of ourapproach. The source code is available athttps://github.com/zhuduowang/ChangeViT. |


| Item |Content|
| --- |---|
|idx| 2406.12846v1 |
|title| DrVideo: Document Retrieval Based Long Video Understanding |
|authors| Ziyu MaChenhui GouHengcan ShiBin SunShutao LiHamid RezatofighiJianfei Cai
|links| http://arxiv.org/abs/2406.12846v1 |
|updated| 2024-06-18 17:59:03 UTC |
|summary| Existing methods for long video understanding primarily focus on videos onlylasting tens of seconds with limited exploration of techniques for handlinglonger videos. The increased number of frames in longer videos presents twomain challenges: difficulty in locating key information and performinglong-range reasoning. Thus we propose DrVideo a document-retrieval-basedsystem designed for long video understanding. Our key idea is to convert thelong-video understanding problem into a long-document understanding task so asto effectively leverage the power of large language models. SpecificallyDrVideo transforms a long video into a text-based long document to initiallyretrieve key frames and augment the information of these frames which is usedthis as the systems starting point. It then employs an agent-based iterativeloop to continuously search for missing information augment relevant data andprovide final predictions in a chain-of-thought manner once sufficientquestion-related information is gathered. Extensive experiments on long videobenchmarks confirm the effectiveness of our method. DrVideo outperformsexisting state-of-the-art methods with 3.8 accuracy on EgoSchema benchmark 3minutes 17.9 in MovieChat-1K break mode 38.0 in MovieChat-1K global mode10 minutes and 30.2 on the LLama-Vid QA dataset over 60 minutes. |


| Item |Content|
| --- |---|
|idx| 2406.12837v1 |
|title| LayerMerge: Neural Network Depth Compression through Layer Pruning and Merging |
|authors| Jinuk KimMarwa El HalabiMingi JiHyun Oh Song
|links| http://arxiv.org/abs/2406.12837v1 |
|updated| 2024-06-18 17:55:15 UTC |
|summary| Recent works show that reducing the number of layers in a convolutionalneural network can enhance efficiency while maintaining the performance of thenetwork. Existing depth compression methods remove redundant non-linearactivation functions and merge the consecutive convolution layers into a singlelayer. However these methods suffer from a critical drawback the kernel sizeof the merged layers becomes larger significantly undermining the latencyreduction gained from reducing the depth of the network. We show that thisproblem can be addressed by jointly pruning convolution layers and activationfunctions. To this end we propose LayerMerge a novel depth compression methodthat selects which activation layers and convolution layers to remove toachieve a desired inference speed-up while minimizing performance loss. Sincethe corresponding selection problem involves an exponential search space weformulate a novel surrogate optimization problem and efficiently solve it viadynamic programming. Empirical results demonstrate that our method consistentlyoutperforms existing depth compression and layer pruning methods on variousnetwork architectures both on image classification and generation tasks. Werelease the code at https://github.com/snu-mllab/LayerMerge. |


| Item |Content|
| --- |---|
|idx| 2406.12834v1 |
|title| GroPrompt: Efficient Grounded Prompting and Adaptation for Referring Video Object Segmentation |
|authors| Ci-Siang LinI-Jieh LiuMin-Hung ChenChien-Yi WangSifei LiuYu-Chiang Frank Wang
|links| http://arxiv.org/abs/2406.12834v1 |
|updated| 2024-06-18 17:54:17 UTC |
|summary| Referring Video Object Segmentation RVOS aims to segment the objectreferred to by the query sentence throughout the entire video. Most existingmethods require end-to-end training with dense mask annotations which could becomputation-consuming and less scalable. In this work we aim to efficientlyadapt foundation segmentation models for addressing RVOS from weak supervisionwith the proposed Grounded Prompting GroPrompt framework. More specificallywe propose Text-Aware Prompt Contrastive Learning TAP-CL to enhance theassociation between the position prompts and the referring sentences with onlybox supervisions including Text-Contrastive Prompt Learning TextCon andModality-Contrastive Prompt Learning ModalCon at frame level and video levelrespectively. With the proposed TAP-CL our GroPrompt framework can generatetemporal-consistent yet text-aware position prompts describing locations andmovements for the referred object from the video. The experimental results inthe standard RVOS benchmarks Ref-YouTube-VOS Ref-DAVIS17 A2D-Sentences andJHMDB-Sentences demonstrate the competitive performance of our proposedGroPrompt framework given only bounding box weak supervisions. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2406.12843v1 |
|title| Can Go AIs be adversarially robust? |
|authors| Tom TsengEuan McLeanKellin PelrineTony T. WangAdam Gleave
|links| http://arxiv.org/abs/2406.12843v1 |
|updated| 2024-06-18 17:57:49 UTC |
|summary| Prior work found that superhuman Go AIs like KataGo can be defeated by simpleadversarial strategies. In this paper we study if simple defenses can improveKataGos worst-case performance. We test three natural defenses: adversarialtraining on hand-constructed positions iterated adversarial training andchanging the network architecture. We find that some of these defenses are ableto protect against previously discovered attacks. Unfortunately we also findthat none of these defenses are able to withstand adaptive attacks. Inparticular we are able to train new adversaries that reliably defeat ourdefended agents by causing them to blunder in ways humans would not. Ourresults suggest that building robust AI systems is challenging even in narrowdomains such as Go. For interactive examples of attacks and a link to ourcodebase see https://goattack.far.ai. |


| Item |Content|
| --- |---|
|idx| 2406.12839v1 |
|title| Evaluating the design space of diffusion-based generative models |
|authors| Yuqing WangYe HeMolei Tao
|links| http://arxiv.org/abs/2406.12839v1 |
|updated| 2024-06-18 17:56:10 UTC |
|summary| Most existing theoretical investigations of the accuracy of diffusion modelsalbeit significant assume the score function has been approximated to acertain accuracy and then use this a priori bound to control the error ofgeneration. This article instead provides a first quantitative understanding ofthe whole generation process i.e. both training and sampling. More preciselyit conducts a non-asymptotic convergence analysis of denoising score matchingunder gradient descent. In addition a refined sampling error analysis forvariance exploding models is also provided. The combination of these tworesults yields a full error analysis which elucidates again but this timetheoretically how to design the training and sampling processes for effectivegeneration. For instance our theory implies a preference toward noisedistribution and loss weighting that qualitatively agree with the ones used inKarras et al. 2022. It also provides some perspectives on why the time andvariance schedule used in Karras et al. 2022 could be better tuned than thepioneering version in Song et al. 2020. |


| Item |Content|
| --- |---|
|idx| 2406.12815v1 |
|title| Privacy Preserving Federated Learning in Medical Imaging with Uncertainty Estimation |
|authors| Nikolas KoutsoubisYasin YilmazRavi P. RamachandranMatthew SchabathGhulam Rasool
|links| http://arxiv.org/abs/2406.12815v1 |
|updated| 2024-06-18 17:35:52 UTC |
|summary| Machine learning ML and Artificial Intelligence AI have fueled remarkableadvancements particularly in healthcare. Within medical imaging ML modelshold the promise of improving disease diagnoses treatment planning andpost-treatment monitoring. Various computer vision tasks like imageclassification object detection and image segmentation are poised to becomeroutine in clinical analysis. However privacy concerns surrounding patientdata hinder the assembly of large training datasets needed for developing andtraining accurate robust and generalizable models. Federated Learning FLemerges as a compelling solution enabling organizations to collaborate on MLmodel training by sharing model training information gradients rather thandata e.g. medical images. FLs distributed learning framework facilitatesinter-institutional collaboration while preserving patient privacy. HoweverFL while robust in privacy preservation faces several challenges. Sensitiveinformation can still be gleaned from shared gradients that are passed onbetween organizations during model training. Additionally in medical imagingquantifying model confidenceuncertainty accurately is crucial due to the noiseand artifacts present in the data. Uncertainty estimation in FL encountersunique hurdles due to data heterogeneity across organizations. This paperoffers a comprehensive review of FL privacy preservation and uncertaintyestimation with a focus on medical imaging. Alongside a survey of currentresearch we identify gaps in the field and suggest future directions for FLresearch to enhance privacy and address noisy medical imaging data challenges. |


| Item |Content|
| --- |---|
|idx| 2406.12764v1 |
|title| Quasi-Bayes meets Vines |
|authors| David HukYuanhe ZhangMark SteelRitabrata Dutta
|links| http://arxiv.org/abs/2406.12764v1 |
|updated| 2024-06-18 16:31:02 UTC |
|summary| Recently proposed quasi-Bayesian QB methods initiated a new era in Bayesiancomputation by directly constructing the Bayesian predictive distributionthrough recursion removing the need for expensive computations involved insampling the Bayesian posterior distribution. This has proved to bedata-efficient for univariate predictions but extensions to multipledimensions rely on a conditional decomposition resulting from predefinedassumptions on the kernel of the Dirichlet Process Mixture Model which is theimplicit nonparametric model used. Here we propose a different way to extendQuasi-Bayesian prediction to high dimensions through the use of Sklars theoremby decomposing the predictive distribution into one-dimensional predictivemarginals and a high-dimensional copula. Thus we use the efficient recursiveQB construction for the one-dimensional marginals and model the dependenceusing highly expressive vine copulas. Further we tune hyperparameters usingrobust divergences eg. energy score and show that our proposed Quasi-BayesianVine QB-Vine is a fully non-parametric density estimator with emphananalytical form and convergence rate independent of the dimension of data insome situations. Our experiments illustrate that the QB-Vine is appropriate forhigh dimensional distributions sim64 needs very few samples to trainsim200 and outperforms state-of-the-art methods with analytical forms fordensity estimation and supervised tasks by a considerable margin. |


| Item |Content|
| --- |---|
|idx| 2406.12763v1 |
|title| Implicit Bias of Mirror Flow on Separable Data |
|authors| Scott PesmeRadu-Alexandru DragomirNicolas Flammarion
|links| http://arxiv.org/abs/2406.12763v1 |
|updated| 2024-06-18 16:30:51 UTC |
|summary| We examine the continuous-time counterpart of mirror descent namely mirrorflow on classification problems which are linearly separable. Such problemsare minimised at infinity and have many possible solutions we study whichsolution is preferred by the algorithm depending on the mirror potential. Forexponential tailed losses and under mild assumptions on the potential we showthat the iterates converge in direction towards a phi_infty-maximum marginclassifier. The function phi_infty is the textithorizon function ofthe mirror potential and characterises its shape at infinity. When thepotential is separable a simple formula allows to compute this function. Weanalyse several examples of potentials and provide numerical experimentshighlighting our results. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2406.12801v1 |
|title| "A Lot of Moving Parts": A Case Study of Open-Source Hardware Design Collaboration in the Thingiverse Community |
|authors| Kathy ChengShurui ZhouAlison Olechowski
|links| http://arxiv.org/abs/2406.12801v1 |
|updated| 2024-06-18 17:13:40 UTC |
|summary| Open-source is a decentralized and collaborative method of development thatencourages open contribution from an extensive and undefined network ofindividuals. Although commonly associated with software development OSS theopen-source model extends to hardware development forming the basis ofopen-source hardware development OSH. Compared to OSS OSH is relativelynascent lacking adequate tooling support from existing platforms and bestpractices for efficient collaboration. Taking a necessary step towardsimproving OSH collaboration we conduct a detailed case study of DrawBot asuccessful OSH project that remarkably fostered a long-term collaboration onThingiverse - a platform not explicitly intended for complex collaborativedesign. Through analyzing comment threads and design changes over the course ofthe project we found how collaboration occurred the challenges faced and howthe DrawBot community managed to overcome these obstacles. Beyond offering adetailed account of collaboration practices and challenges our workcontributes best practices design implications and practical implications forOSH project maintainers platform builders and researchers respectively. Withthese insights and our publicly available dataset we aim to foster moreeffective and efficient collaborative design in OSH projects. |


| Item |Content|
| --- |---|
|idx| 2406.12787v1 |
|title| Generating Educational Materials with Different Levels of Readability using LLMs |
|authors| Chieh-Yang HuangJing WeiTing-Hao 'Kenneth' Huang
|links| http://arxiv.org/abs/2406.12787v1 |
|updated| 2024-06-18 16:55:10 UTC |
|summary| This study introduces the leveled-text generation task aiming to rewriteeducational materials to specific readability levels while preserving meaning.We assess the capability of GPT-3.5 LLaMA-2 70B and Mixtral 8x7B to generatecontent at various readability levels through zero-shot and few-shot prompting.Evaluating 100 processed educational materials reveals that few-shot promptingsignificantly improves performance in readability manipulation and informationpreservation. LLaMA-2 70B performs better in achieving the desired difficultyrange while GPT-3.5 maintains original meaning. However manual inspectionhighlights concerns such as misinformation introduction and inconsistent editdistribution. These findings emphasize the need for further research to ensurethe quality of generated educational content. |


| Item |Content|
| --- |---|
|idx| 2406.12762v1 |
|title| Unsupervised explainable activity prediction in competitive Nordic Walking from experimental data |
|authors| Silvia García-MéndezFrancisco de Arriba-PérezFrancisco J. González-CastañoJavier Vales-Alonso
|links| http://dx.doi.org/10.1109/MCE.2024.3387019 |
|updated| 2024-06-18 16:29:07 UTC |
|summary| Artificial Intelligence AI has found application in Human ActivityRecognition HAR in competitive sports. To date most Machine Learning MLapproaches for HAR have relied on offline batch training imposing highercomputational and tagging burdens compared to online processing unsupervisedapproaches. Additionally the decisions behind traditional ML predictors areopaque and require human interpretation. In this work we apply an onlineprocessing unsupervised clustering approach based on low-cost wearable InertialMeasurement Units IMUs. The outcomes generated by the system allow for theautomatic expansion of limited tagging available e.g. by referees withinthose clusters producing pertinent information for the explainableclassification stage. Specifically our work focuses on achieving automaticexplainability for predictions related to athletes activities distinguishingbetween correct incorrect and cheating practices in Nordic Walking. Theproposed solution achieved performance metrics of close to 100  on average. |


| Item |Content|
| --- |---|
|idx| 2406.12692v1 |
|title| MAGIC: Generating Self-Correction Guideline for In-Context Text-to-SQL |
|authors| Arian AskariChristian PoelitzXinye Tang
|links| http://arxiv.org/abs/2406.12692v1 |
|updated| 2024-06-18 15:06:06 UTC |
|summary| Self-correction in text-to-SQL is the process of prompting large languagemodel LLM to revise its previously incorrectly generated SQL and commonlyrelies on manually crafted self-correction guidelines by human experts that arenot only labor-intensive to produce but also limited by the human ability inidentifying all potential error patterns in LLM responses. We introduce MAGICa novel multi-agent method that automates the creation of the self-correctionguideline. MAGIC uses three specialized agents: a manager a correction and afeedback agent. These agents collaborate on the failures of an LLM-based methodon the training set to iteratively generate and refine a self-correctionguideline tailored to LLM mistakes mirroring human processes but without humaninvolvement. Our extensive experiments show that MAGICs guideline outperformsexpert humans created ones. We empirically find out that the guidelineproduced by MAGIC enhance the interpretability of the corrections madeproviding insights in analyzing the reason behind the failures and successes ofLLMs in self-correction. We make all agent interactions publicly available tothe research community to foster further research in this area offering asynthetic dataset for future explorations into automatic self-correctionguideline generation. |


| Item |Content|
| --- |---|
|idx| 2406.12651v1 |
|title| Transforming Surgical Interventions with Embodied Intelligence for Ultrasound Robotics |
|authors| Huan XuJinlin WuGuanglin CaoZhen ChenZhen LeiHongbin Liu
|links| http://arxiv.org/abs/2406.12651v1 |
|updated| 2024-06-18 14:22:16 UTC |
|summary| Ultrasonography has revolutionized non-invasive diagnostic methodologiessignificantly enhancing patient outcomes across various medical domains.Despite its advancements integrating ultrasound technology with roboticsystems for automated scans presents challenges including limited commandunderstanding and dynamic execution capabilities. To address these challengesthis paper introduces a novel Ultrasound Embodied Intelligence system thatsynergistically combines ultrasound robots with large language models LLMsand domain-specific knowledge augmentation enhancing ultrasound robotsintelligence and operational efficiency. Our approach employs a dual strategy:firstly integrating LLMs with ultrasound robots to interpret doctors verbalinstructions into precise motion planning through a comprehensive understandingof ultrasound domain knowledge including APIs and operational manualssecondly incorporating a dynamic execution mechanism allowing for real-timeadjustments to scanning plans based on patient movements or procedural errors.We demonstrate the effectiveness of our system through extensive experimentsincluding ablation studies and comparisons across various models showcasingsignificant improvements in executing medical procedures from verbal commands.Our findings suggest that the proposed system improves the efficiency andquality of ultrasound scans and paves the way for further advancements inautonomous medical scanning technologies with the potential to transformnon-invasive diagnostics and streamline medical workflows. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2406.12526v1 |
|title| On the Convergence of Tâtonnement for Linear Fisher Markets |
|authors| Tianlong NanYuan GaoChristian Kroer
|links| http://arxiv.org/abs/2406.12526v1 |
|updated| 2024-06-18 11:54:01 UTC |
|summary| Tatonnement is a simple intuitive market process where prices areiteratively adjusted based on the difference between demand and supply. Manyvariants under different market assumptions have been studied and shown toconverge to a market equilibrium in some cases at a fast rate. However theclassical case of linear Fisher markets have long eluded the analyses and itremains unclear whether tatonnement converges in this case. We show that fora sufficiently small step size the prices given by the tatonnement processare guaranteed to converge to equilibrium prices up to a small approximationradius that depends on the stepsize. To achieve this we consider the dualEisenberg-Gale convex program in the price space view tatonnement assubgradient descent on this convex program and utilize novel last-iterateconvergence results for subgradient descent under error bound conditions. Indoing so we show that the convex program satisfies a particular error boundcondition the quadratic growth condition and that the price sequencegenerated by tatonnement is bounded above and away from zero. We also showthat a similar convergence result holds for tatonnement in quasi-linearFisher markets. Numerical experiments are conducted to demonstrate that thetheoretical linear convergence aligns with empirical observations. |


| Item |Content|
| --- |---|
|idx| 2406.12302v1 |
|title| A Step Towards a Universal Method for Modeling and Implementing Cross-Organizational Business Processes |
|authors| Gerhard ZeislerTim Tobias BraunauerAlbert FleischmannRobert Singer
|links| http://arxiv.org/abs/2406.12302v1 |
|updated| 2024-06-18 06:19:44 UTC |
|summary| The widely adopted Business Process Model and Notation BPMN is acornerstone of industry standards for business process modeling. However itsambiguous execution semantics often result in inconsistent interpretationsdepending on the software used for implementation. In response the ProcessSpecification Language PASS provides formally defined semantics to overcomethese interpretational challenges. Despite its clear advantages PASS has notreached the same level of industry penetration as BPMN.  This feasibility study proposes using PASS as an intermediary framework totranslate and execute BPMN models. It describes the development of a prototypetranslator that converts specific BPMN elements into a format compatible withPASS. These models are then transformed into source code and executed in abespoke workflow environment marking a departure from traditional BPMNimplementations.  Our findings suggest that integrating PASS enhances compatibility acrossdifferent modeling and execution tools and offers a more robust methodology forimplementing business processes across organizations. This study lays thegroundwork for more accurate and unified business process model executionspotentially transforming industry standards for process modeling and execution. |


| Item |Content|
| --- |---|
|idx| 2406.11938v1 |
|title| Tracking the perspectives of interacting language models |
|authors| Hayden HelmBrandon DuderstadtYoungser ParkCarey E. Priebe
|links| http://arxiv.org/abs/2406.11938v1 |
|updated| 2024-06-17 17:20:16 UTC |
|summary| Large language models LLMs are capable of producing high qualityinformation at unprecedented rates. As these models continue to entrenchthemselves in society the content they produce will become increasinglypervasive in databases that are in turn incorporated into the pre-trainingdata fine-tuning data retrieval data etc. of other language models. In thispaper we formalize the idea of a communication network of LLMs and introduce amethod for representing the perspective of individual models within acollection of LLMs. Given these tools we systematically study informationdiffusion in the communication network of LLMs in various simulated settings. |


| Item |Content|
| --- |---|
|idx| 2406.11709v1 |
|title| Instruct, Not Assist: LLM-based Multi-Turn Planning and Hierarchical Questioning for Socratic Code Debugging |
|authors| Priyanka KarguptaIshika AgarwalDilek Hakkani-TurJiawei Han
|links| http://arxiv.org/abs/2406.11709v1 |
|updated| 2024-06-17 16:28:21 UTC |
|summary| Socratic questioning is an effective teaching strategy encouraging criticalthinking and problem-solving. The conversational capabilities of large languagemodels LLMs show great potential for providing scalable real-time studentguidance. However current LLMs often give away solutions directly making themineffective instructors. We tackle this issue in the code debugging domain withTreeInstruct an Instructor agent guided by a novel state space-based planningalgorithm. TreeInstruct asks probing questions to help students independentlyidentify and resolve errors. It estimates a students conceptual andsyntactical knowledge to dynamically construct a question tree based on theirresponses and current knowledge state effectively addressing both independentand dependent mistakes concurrently in a multi-turn interaction setting. Inaddition to using an existing single-bug debugging benchmark we construct amore challenging multi-bug dataset of 150 coding problems incorrect solutionsand bug fixes -- all carefully constructed and annotated by experts. Extensiveevaluation shows TreeInstructs state-of-the-art performance on both datasetsproving it to be a more effective instructor than baselines. Furthermore areal-world case study with five students of varying skill levels furtherdemonstrates TreeInstructs ability to guide students to debug their codeefficiently with minimal turns and highly Socratic questioning. |


| Item |Content|
| --- |---|
|idx| 2406.11496v1 |
|title| Decentralized Collaborative Pricing and Shunting for Multiple EV Charging Stations Based on Multi-Agent Reinforcement Learning |
|authors| Tianhao BuHang LiGuojie Li
|links| http://arxiv.org/abs/2406.11496v1 |
|updated| 2024-06-17 12:59:30 UTC |
|summary| The extraordinary electric vehicle EV popularization in the recent yearshas facilitated research studies in alleviating EV energy charging demand.Previous studies primarily focused on the optimizations over charging stationsCS profit and EV users cost savings through charge/discharge schedulingevents. In this work the random behaviors of EVs are considered with EV userspreferences over multi-CS characteristics modelled to imitate the potential CSselection disequilibrium. A price scheduling strategy under decentralizedcollaborative framework is proposed to achieve EV shunting in a multi-CSenvironment while minimizing the charging cost through multi agentreinforcement learning. The proposed problem is formulated as a Markov DecisionProcess MDP with uncertain transition probability. |


