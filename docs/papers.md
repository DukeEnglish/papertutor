# cs.CL 

| Item |Content|
| --- |---|
|idx| 2402.07899v1 |
|title| A systematic investigation of learnability from single child linguistic input |
|authors| Yulu QinWentao WangBrenden M. Lake
|links| http://arxiv.org/abs/2402.07899v1 |
|updated| 2024-02-12 18:58:58 UTC |
|summary| Language models LMs have demonstrated remarkable proficiency in generatinglinguistically coherent text sparking discussions about their relevance tounderstanding human language learnability. However a significant gap existsbetween the training data for these models and the linguistic input a childreceives. LMs are typically trained on data that is orders of magnitude largerand fundamentally different from child-directed speech Warstadt and Bowman2022 Warstadt et al. 2023 Frank 2023a. Addressing this discrepancy ourresearch focuses on training LMs on subsets of a single childs linguisticinput. Previously Wang Vong Kim and Lake 2023 found that LMs trained inthis setting can form syntactic and semantic word clusters and developsensitivity to certain linguistic phenomena but they only considered LSTMs andsimpler neural networks trained from just one single-child dataset. Here toexamine the robustness of learnability from single-child input wesystematically train six different model architectures on five datasets 3single-child and 2 baselines. We find that the models trained on single-childdatasets showed consistent results that matched with previous workunderscoring the robustness of forming meaningful syntactic and semanticrepresentations from a subset of a childs linguistic input. |


| Item |Content|
| --- |---|
|idx| 2402.07896v2 |
|title| Suppressing Pink Elephants with Direct Principle Feedback |
|authors| Louis CastricatoNathan LileSuraj AnandHailey SchoelkopfSiddharth VermaStella Biderman
|links| http://arxiv.org/abs/2402.07896v2 |
|updated| 2024-02-13 18:44:11 UTC |
|summary| Existing methods for controlling language models such as RLHF andConstitutional AI involve determining which LLM behaviors are desirable andtraining them into a language model. However in many cases it is desirablefor LLMs to be controllable at inference time so that they can be used inmultiple contexts with diverse needs. We illustrate this with the Pink ElephantProblem: instructing an LLM to avoid discussing a certain entity a PinkElephant and instead discuss a preferred entity Grey Elephant. Weapply a novel simplification of Constitutional AI Direct Principle Feedbackwhich skips the ranking of responses and uses DPO directly on critiques andrevisions. Our results show that after DPF fine-tuning on our synthetic PinkElephants dataset our 13B fine-tuned LLaMA 2 model significantly outperformsLlama-2-13B-Chat and a prompted baseline and performs as well as GPT-4 in onour curated test set assessing the Pink Elephant Problem. |


| Item |Content|
| --- |---|
|idx| 2402.07891v1 |
|title| Label-Efficient Model Selection for Text Generation |
|authors| Shir Ashury-TahanBenjamin SznajderLeshem ChoshenLiat Ein-DorEyal ShnarchAriel Gera
|links| http://arxiv.org/abs/2402.07891v1 |
|updated| 2024-02-12 18:54:02 UTC |
|summary| Model selection for a given target task can be costly as it may entailextensive annotation of the quality of outputs of different models. Weintroduce DiffUse an efficient method to make an informed decision betweencandidate text generation models. DiffUse reduces the required amount ofpreference annotations thus saving valuable time and resources in performingevaluation. DiffUse intelligently selects instances by clustering embeddingsthat represent the semantic differences between model outputs. Thus it is ableto identify a subset of examples that are more informative for preferencedecisions. Our method is model-agnostic and can be applied to any textgeneration model. Moreover we propose a practical iterative approach fordynamically determining how many instances to annotate. In a series ofexperiments over hundreds of model pairs we demonstrate that DiffUse candramatically reduce the required number of annotations -- by up to 75 -- whilemaintaining high evaluation reliability. |


| Item |Content|
| --- |---|
|idx| 2402.07876v1 |
|title| Policy Improvement using Language Feedback Models |
|authors| Victor ZhongDipendra MisraXingdi YuanMarc-Alexandre Côté
|links| http://arxiv.org/abs/2402.07876v1 |
|updated| 2024-02-12 18:41:34 UTC |
|summary| We introduce Language Feedback Models LFMs that identify desirablebehaviour - actions that help achieve tasks specified in the instruction - forimitation learning in instruction following. To train LFMs we obtain feedbackfrom Large Language Models LLMs on visual trajectories verbalized to languagedescriptions. First by using LFMs to identify desirable behaviour to imitatewe improve in task-completion rate over strong behavioural cloning baselines onthree distinct language grounding environments Touchdown ScienceWorld andALFWorld. Second LFMs outperform using LLMs as experts to directly predictactions when controlling for the number of LLM output tokens. Third LFMsgeneralize to unseen environments improving task-completion rate by 3.5-12.0through one round of adaptation. Finally LFM can be modified to providehuman-interpretable feedback without performance loss allowing humanverification of desirable behaviour for imitation learning. |


| Item |Content|
| --- |---|
|idx| 2402.07871v1 |
|title| Scaling Laws for Fine-Grained Mixture of Experts |
|authors| Jakub KrajewskiJan LudziejewskiKamil AdamczewskiMaciej PióroMichał KrutulSzymon AntoniakKamil CiebieraKrystian KrólTomasz OdrzygóźdźPiotr SankowskiMarek CyganSebastian Jaszczur
|links| http://arxiv.org/abs/2402.07871v1 |
|updated| 2024-02-12 18:33:47 UTC |
|summary| Mixture of Experts MoE models have emerged as a primary solution forreducing the computational cost of Large Language Models. In this work weanalyze their scaling properties incorporating an expanded range of variables.Specifically we introduce a new hyperparameter granularity whose adjustmentenables precise control over the size of the experts. Building on this weestablish scaling laws for fine-grained MoE taking into account the number oftraining tokens model size and granularity. Leveraging these laws we derivethe optimal training configuration for a given computational budget. Ourfindings not only show that MoE models consistently outperform denseTransformers but also highlight that the efficiency gap between dense and MoEmodels widens as we scale up the model size and training budget. Furthermorewe demonstrate that the common practice of setting the size of experts in MoEto mirror the feed-forward layer is not optimal at almost any computationalbudget. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2402.07901v1 |
|title| FAST: Factorizable Attention for Speeding up Transformers |
|authors| Armin GeramiMonte HooverPranav S. DulepetRamani Duraiswami
|links| http://arxiv.org/abs/2402.07901v1 |
|updated| 2024-02-12 18:59:39 UTC |
|summary| Motivated by the factorization inherent in the original fast multipole methodand the improved fast Gauss transform we introduce a factorable form ofattention that operates efficiently in high dimensions. This approach reducesthe computational and memory complexity of the attention mechanism intransformers from ON2 to ON. In comparison to previous attempts ourwork presents a linearly scaled attention mechanism that maintains the fullrepresentation of the attention matrix without compromising on sparsificationand incorporates the all-to-all relationship between tokens. We explore theproperties of our new attention metric and conduct tests in various standardsettings. Results indicate that our attention mechanism has a robustperformance and holds significant promise for diverse applications whereself-attention is used. |


| Item |Content|
| --- |---|
|idx| 2402.07895v1 |
|title| Detection of Spider Mites on Labrador Beans through Machine Learning Approaches Using Custom Datasets |
|authors| Violet LiuJason ChenAns QureshiMahla Nejati
|links| http://arxiv.org/abs/2402.07895v1 |
|updated| 2024-02-12 18:57:06 UTC |
|summary| Amidst growing food production demands early plant disease detection isessential to safeguard crops this study proposes a visual machine learningapproach for plant disease detection harnessing RGB and NIR data collected inreal-world conditions through a JAI FS-1600D-10GE camera to build an RGBNdataset. A two-stage early plant disease detection model with YOLOv8 and asequential CNN was used to train on a dataset with partial labels which showeda 3.6 increase in mAP compared to a single-stage end-to-end segmentationmodel. The sequential CNN model achieved 90.62 validation accuracy utilisingRGBN data. An average of 6.25 validation accuracy increase is found using RGBNin classification compared to RGB using ResNet15 and the sequential CNN models.Further research and dataset improvements are needed to meet food productiondemands. |


| Item |Content|
| --- |---|
|idx| 2402.07890v1 |
|title| MAIDCRL: Semi-centralized Multi-Agent Influence Dense-CNN Reinforcement Learning |
|authors| Ayesha Siddika NipuSiming LiuAnthony Harris
|links| http://dx.doi.org/10.1109/CoG51982.2022.9893711 |
|updated| 2024-02-12 18:53:20 UTC |
|summary| Distributed decision-making in multi-agent systems presents difficultchallenges for interactive behavior learning in both cooperative andcompetitive systems. To mitigate this complexity MAIDRL presents asemi-centralized Dense Reinforcement Learning algorithm enhanced by agentinfluence maps AIMs for learning effective multi-agent control on StarCraftMulti-Agent Challenge SMAC scenarios. In this paper we extend the DenseNetin MAIDRL and introduce semi-centralized Multi-Agent Dense-CNN ReinforcementLearning MAIDCRL by incorporating convolutional layers into the deep modelarchitecture and evaluate the performance on both homogeneous andheterogeneous scenarios. The results show that the CNN-enabled MAIDCRLsignificantly improved the learning performance and achieved a faster learningrate compared to the existing MAIDRL especially on more complicatedheterogeneous SMAC scenarios. We further investigate the stability androbustness of our model. The statistics reflect that our model not onlyachieves higher winning rate in all the given scenarios but also boosts theagents learning process in fine-grained decision-making. |


| Item |Content|
| --- |---|
|idx| 2402.07877v1 |
|title| WildfireGPT: Tailored Large Language Model for Wildfire Analysis |
|authors| Yangxinyu XieTanwi MallickJoshua David BergersonJohn K. HutchisonDuane R. VernerJordan BranhamM. Ross AlexanderRobert B. RossYan FengLeslie-Anne LevyWeijie Su
|links| http://arxiv.org/abs/2402.07877v1 |
|updated| 2024-02-12 18:41:55 UTC |
|summary| The recent advancement of large language models LLMs represents atransformational capability at the frontier of artificial intelligence AI andmachine learning ML. However LLMs are generalized models trained onextensive text corpus and often struggle to provide context-specificinformation particularly in areas requiring specialized knowledge such aswildfire details within the broader context of climate change. Fordecision-makers and policymakers focused on wildfire resilience and adaptationit is crucial to obtain responses that are not only precise but alsodomain-specific rather than generic. To that end we developed WildfireGPT aprototype LLM agent designed to transform user queries into actionable insightson wildfire risks. We enrich WildfireGPT by providing additional context suchas climate projections and scientific literature to ensure its information iscurrent relevant and scientifically accurate. This enables WildfireGPT to bean effective tool for delivering detailed user-specific insights on wildfirerisks to support a diverse set of end users including researchers engineersurban planners emergency managers and infrastructure operators. |


| Item |Content|
| --- |---|
|idx| 2402.07876v1 |
|title| Policy Improvement using Language Feedback Models |
|authors| Victor ZhongDipendra MisraXingdi YuanMarc-Alexandre Côté
|links| http://arxiv.org/abs/2402.07876v1 |
|updated| 2024-02-12 18:41:34 UTC |
|summary| We introduce Language Feedback Models LFMs that identify desirablebehaviour - actions that help achieve tasks specified in the instruction - forimitation learning in instruction following. To train LFMs we obtain feedbackfrom Large Language Models LLMs on visual trajectories verbalized to languagedescriptions. First by using LFMs to identify desirable behaviour to imitatewe improve in task-completion rate over strong behavioural cloning baselines onthree distinct language grounding environments Touchdown ScienceWorld andALFWorld. Second LFMs outperform using LLMs as experts to directly predictactions when controlling for the number of LLM output tokens. Third LFMsgeneralize to unseen environments improving task-completion rate by 3.5-12.0through one round of adaptation. Finally LFM can be modified to providehuman-interpretable feedback without performance loss allowing humanverification of desirable behaviour for imitation learning. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2402.07901v1 |
|title| FAST: Factorizable Attention for Speeding up Transformers |
|authors| Armin GeramiMonte HooverPranav S. DulepetRamani Duraiswami
|links| http://arxiv.org/abs/2402.07901v1 |
|updated| 2024-02-12 18:59:39 UTC |
|summary| Motivated by the factorization inherent in the original fast multipole methodand the improved fast Gauss transform we introduce a factorable form ofattention that operates efficiently in high dimensions. This approach reducesthe computational and memory complexity of the attention mechanism intransformers from ON2 to ON. In comparison to previous attempts ourwork presents a linearly scaled attention mechanism that maintains the fullrepresentation of the attention matrix without compromising on sparsificationand incorporates the all-to-all relationship between tokens. We explore theproperties of our new attention metric and conduct tests in various standardsettings. Results indicate that our attention mechanism has a robustperformance and holds significant promise for diverse applications whereself-attention is used. |


| Item |Content|
| --- |---|
|idx| 2402.07899v1 |
|title| A systematic investigation of learnability from single child linguistic input |
|authors| Yulu QinWentao WangBrenden M. Lake
|links| http://arxiv.org/abs/2402.07899v1 |
|updated| 2024-02-12 18:58:58 UTC |
|summary| Language models LMs have demonstrated remarkable proficiency in generatinglinguistically coherent text sparking discussions about their relevance tounderstanding human language learnability. However a significant gap existsbetween the training data for these models and the linguistic input a childreceives. LMs are typically trained on data that is orders of magnitude largerand fundamentally different from child-directed speech Warstadt and Bowman2022 Warstadt et al. 2023 Frank 2023a. Addressing this discrepancy ourresearch focuses on training LMs on subsets of a single childs linguisticinput. Previously Wang Vong Kim and Lake 2023 found that LMs trained inthis setting can form syntactic and semantic word clusters and developsensitivity to certain linguistic phenomena but they only considered LSTMs andsimpler neural networks trained from just one single-child dataset. Here toexamine the robustness of learnability from single-child input wesystematically train six different model architectures on five datasets 3single-child and 2 baselines. We find that the models trained on single-childdatasets showed consistent results that matched with previous workunderscoring the robustness of forming meaningful syntactic and semanticrepresentations from a subset of a childs linguistic input. |


| Item |Content|
| --- |---|
|idx| 2402.07891v1 |
|title| Label-Efficient Model Selection for Text Generation |
|authors| Shir Ashury-TahanBenjamin SznajderLeshem ChoshenLiat Ein-DorEyal ShnarchAriel Gera
|links| http://arxiv.org/abs/2402.07891v1 |
|updated| 2024-02-12 18:54:02 UTC |
|summary| Model selection for a given target task can be costly as it may entailextensive annotation of the quality of outputs of different models. Weintroduce DiffUse an efficient method to make an informed decision betweencandidate text generation models. DiffUse reduces the required amount ofpreference annotations thus saving valuable time and resources in performingevaluation. DiffUse intelligently selects instances by clustering embeddingsthat represent the semantic differences between model outputs. Thus it is ableto identify a subset of examples that are more informative for preferencedecisions. Our method is model-agnostic and can be applied to any textgeneration model. Moreover we propose a practical iterative approach fordynamically determining how many instances to annotate. In a series ofexperiments over hundreds of model pairs we demonstrate that DiffUse candramatically reduce the required number of annotations -- by up to 75 -- whilemaintaining high evaluation reliability. |


| Item |Content|
| --- |---|
|idx| 2402.07890v1 |
|title| MAIDCRL: Semi-centralized Multi-Agent Influence Dense-CNN Reinforcement Learning |
|authors| Ayesha Siddika NipuSiming LiuAnthony Harris
|links| http://dx.doi.org/10.1109/CoG51982.2022.9893711 |
|updated| 2024-02-12 18:53:20 UTC |
|summary| Distributed decision-making in multi-agent systems presents difficultchallenges for interactive behavior learning in both cooperative andcompetitive systems. To mitigate this complexity MAIDRL presents asemi-centralized Dense Reinforcement Learning algorithm enhanced by agentinfluence maps AIMs for learning effective multi-agent control on StarCraftMulti-Agent Challenge SMAC scenarios. In this paper we extend the DenseNetin MAIDRL and introduce semi-centralized Multi-Agent Dense-CNN ReinforcementLearning MAIDCRL by incorporating convolutional layers into the deep modelarchitecture and evaluate the performance on both homogeneous andheterogeneous scenarios. The results show that the CNN-enabled MAIDCRLsignificantly improved the learning performance and achieved a faster learningrate compared to the existing MAIDRL especially on more complicatedheterogeneous SMAC scenarios. We further investigate the stability androbustness of our model. The statistics reflect that our model not onlyachieves higher winning rate in all the given scenarios but also boosts theagents learning process in fine-grained decision-making. |


| Item |Content|
| --- |---|
|idx| 2402.07878v1 |
|title| Using Graph Theory for Improving Machine Learning-based Detection of Cyber Attacks |
|authors| Giacomo ZonneveldLorenzo PrincipiMarco Baldi
|links| http://arxiv.org/abs/2402.07878v1 |
|updated| 2024-02-12 18:44:02 UTC |
|summary| Early detection of network intrusions and cyber threats is one of the mainpillars of cybersecurity. One of the most effective approaches for this purposeis to analyze network traffic with the help of artificial intelligencealgorithms with the aim of detecting the possible presence of an attacker bydistinguishing it from a legitimate user. This is commonly done by collectingthe traffic exchanged between terminals in a network and analyzing it on aper-packet or per-connection basis. In this paper we propose instead toperform pre-processing of network traffic under analysis with the aim ofextracting some new metrics on which we can perform more efficient detectionand overcome some limitations of classical approaches. These new metrics arebased on graph theory and consider the network as a whole rather thanfocusing on individual packets or connections. Our approach is validatedthrough experiments performed on publicly available data sets from which itresults that it can not only overcome some of the limitations of classicalapproaches but also achieve a better detection capability of cyber threats. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2402.07900v2 |
|title| Wavefront Randomization Improves Deconvolution |
|authors| Amit KohliAnastasios N. AngelopoulosLaura Waller
|links| http://arxiv.org/abs/2402.07900v2 |
|updated| 2024-02-13 03:00:54 UTC |
|summary| The performance of an imaging system is limited by optical aberrations whichcause blurriness in the resulting image. Digital correction techniques such asdeconvolution have limited ability to correct the blur since some spatialfrequencies in the scene are not measured adequately i.e. zeros of thesystem transfer function. We prove that the addition of a random mask to animaging system removes its dependence on aberrations reducing the likelihoodof zeros in the transfer function and consequently decreasing the sensitivityto noise during deconvolution. In simulation we show that this strategyimproves image quality over a range of aberration types aberration strengthsand signal-to-noise ratios. |


| Item |Content|
| --- |---|
|idx| 2402.07895v1 |
|title| Detection of Spider Mites on Labrador Beans through Machine Learning Approaches Using Custom Datasets |
|authors| Violet LiuJason ChenAns QureshiMahla Nejati
|links| http://arxiv.org/abs/2402.07895v1 |
|updated| 2024-02-12 18:57:06 UTC |
|summary| Amidst growing food production demands early plant disease detection isessential to safeguard crops this study proposes a visual machine learningapproach for plant disease detection harnessing RGB and NIR data collected inreal-world conditions through a JAI FS-1600D-10GE camera to build an RGBNdataset. A two-stage early plant disease detection model with YOLOv8 and asequential CNN was used to train on a dataset with partial labels which showeda 3.6 increase in mAP compared to a single-stage end-to-end segmentationmodel. The sequential CNN model achieved 90.62 validation accuracy utilisingRGBN data. An average of 6.25 validation accuracy increase is found using RGBNin classification compared to RGB using ResNet15 and the sequential CNN models.Further research and dataset improvements are needed to meet food productiondemands. |


| Item |Content|
| --- |---|
|idx| 2402.07894v1 |
|title| MODIPHY: Multimodal Obscured Detection for IoT using PHantom Convolution-Enabled Faster YOLO |
|authors| Shubhabrata MukherjeeCory BeardZhu Li
|links| http://arxiv.org/abs/2402.07894v1 |
|updated| 2024-02-12 18:56:53 UTC |
|summary| Low-light conditions and occluded scenarios impede object detection inreal-world Internet of Things IoT applications like autonomous vehicles andsecurity systems. While advanced machine learning models strive for accuracytheir computational demands clash with the limitations of resource-constraineddevices hampering real-time performance. In our current research we tacklethis challenge by introducing YOLO Phantom one of the smallest YOLO modelsever conceived. YOLO Phantom utilizes the novel Phantom Convolution blockachieving comparable accuracy to the latest YOLOv8n model while simultaneouslyreducing both parameters and model size by 43 resulting in a significant 19reduction in Giga Floating Point Operations GFLOPs. YOLO Phantom leveragestransfer learning on our multimodal RGB-infrared dataset to address low-lightand occlusion issues equipping it with robust vision under adverse conditions.Its real-world efficacy is demonstrated on an IoT platform with advancedlow-light and RGB cameras seamlessly connecting to an AWS-based notificationendpoint for efficient real-time object detection. Benchmarks reveal asubstantial boost of 17 and 14 in frames per second FPS for thermal and RGBdetection respectively compared to the baseline YOLOv8n model. For communitycontribution both the code and the multimodal dataset are available on GitHub. |


| Item |Content|
| --- |---|
|idx| 2402.07872v1 |
|title| PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs |
|authors| Soroush NasirianyFei XiaWenhao YuTed XiaoJacky LiangIshita DasguptaAnnie XieDanny DriessAyzaan WahidZhuo XuQuan VuongTingnan ZhangTsang-Wei Edward LeeKuang-Huei LeePeng XuSean KirmaniYuke ZhuAndy ZengKarol HausmanNicolas HeessChelsea FinnSergey LevineBrian Ichter
|links| http://arxiv.org/abs/2402.07872v1 |
|updated| 2024-02-12 18:33:47 UTC |
|summary| Vision language models VLMs have shown impressive capabilities across avariety of tasks from logical reasoning to visual understanding. This opensthe door to richer interaction with the world for example robotic control.However VLMs produce only textual outputs while robotic control and otherspatial tasks require outputting continuous coordinates actions ortrajectories. How can we enable VLMs to handle such settings withoutfine-tuning on task-specific data  In this paper we propose a novel visual prompting approach for VLMs that wecall Prompting with Iterative Visual Optimization PIVOT which casts tasks asiterative visual question answering. In each iteration the image is annotatedwith a visual representation of proposals that the VLM can refer to e.g.candidate robot actions localizations or trajectories. The VLM then selectsthe best ones for the task. These proposals are iteratively refined allowingthe VLM to eventually zero in on the best available answer. We investigatePIVOT on real-world robotic navigation real-world manipulation from imagesinstruction following in simulation and additional spatial inference taskssuch as localization. We find perhaps surprisingly that our approach enableszero-shot control of robotic systems without any robot training datanavigation in a variety of environments and other capabilities. Althoughcurrent performance is far from perfect our work highlights potentials andlimitations of this new regime and shows a promising approach forInternet-Scale VLMs in robotic and spatial reasoning domains. Website:pivot-prompt.github.io and HuggingFace:https://huggingface.co/spaces/pivot-prompt/pivot-prompt-demo. |


| Item |Content|
| --- |---|
|idx| 2402.07865v1 |
|title| Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models |
|authors| Siddharth KaramchetiSuraj NairAshwin BalakrishnaPercy LiangThomas KollarDorsa Sadigh
|links| http://arxiv.org/abs/2402.07865v1 |
|updated| 2024-02-12 18:21:14 UTC |
|summary| Visually-conditioned language models VLMs have seen growing adoption inapplications such as visual dialogue scene understanding and robotic taskplanning adoption that has fueled a wealth of new models such as LLaVaInstructBLIP and PaLI-3. Despite the volume of new releases key designdecisions around image preprocessing architecture and optimization areunder-explored making it challenging to understand what factors account formodel performance - a challenge further complicated by the lack of objectiveconsistent evaluations. To address these gaps we first compile a suite ofstandardized evaluations spanning visual question answering objectlocalization from language and targeted challenge sets that probe propertiessuch as hallucination evaluations that provide calibrated fine-grainedinsight into a VLMs capabilities. Second we rigorously investigate VLMs alongkey design axes including pretrained visual representations and quantifyingthe tradeoffs of using base vs. instruct-tuned language models amongst others.We couple our analysis with three resource contributions: 1 a unifiedframework for evaluating VLMs 2 optimized flexible code for VLM trainingand 3 checkpoints for all models including a family of VLMs at the 7-13Bscale that strictly outperform InstructBLIP and LLaVa v1.5 thestate-of-the-art in open-source VLMs. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2402.07875v1 |
|title| Implicit Bias of Policy Gradient in Linear Quadratic Control: Extrapolation to Unseen Initial States |
|authors| Noam RazinYotam AlexanderEdo Cohen-KarlikRaja GiryesAmir GlobersonNadav Cohen
|links| http://arxiv.org/abs/2402.07875v1 |
|updated| 2024-02-12 18:41:31 UTC |
|summary| In modern machine learning models can often fit training data in numerousways some of which perform well on unseen test data while others do not.Remarkably in such cases gradient descent frequently exhibits an implicit biasthat leads to excellent performance on unseen data. This implicit bias wasextensively studied in supervised learning but is far less understood inoptimal control reinforcement learning. There learning a controller appliedto a system via gradient descent is known as policy gradient and a question ofprime importance is the extent to which a learned controller extrapolates tounseen initial states. This paper theoretically studies the implicit bias ofpolicy gradient in terms of extrapolation to unseen initial states. Focusing onthe fundamental Linear Quadratic Regulator LQR problem we establish that theextent of extrapolation depends on the degree of exploration induced by thesystem when commencing from initial states included in training. Experimentscorroborate our theory and demonstrate its conclusions on problems beyond LQRwhere systems are non-linear and controllers are neural networks. Wehypothesize that real-world optimal control may be greatly improved bydeveloping methods for informed selection of initial states to train on. |


| Item |Content|
| --- |---|
|idx| 2402.07846v1 |
|title| Generative Modeling of Discrete Joint Distributions by E-Geodesic Flow Matching on Assignment Manifolds |
|authors| Bastian BollDaniel Gonzalez-AlvaradoChristoph Schnörr
|links| http://arxiv.org/abs/2402.07846v1 |
|updated| 2024-02-12 17:56:52 UTC |
|summary| This paper introduces a novel generative model for discrete distributionsbased on continuous normalizing flows on the submanifold of factorizingdiscrete measures. Integration of the flow gradually assigns categories andavoids issues of discretizing the latent continuous model like rounding sampletruncation etc. General non-factorizing discrete distributions capable ofrepresenting complex statistical dependencies of structured discrete data canbe approximated by embedding the submanifold into a the meta-simplex of alljoint discrete distributions and data-driven averaging. Efficient training ofthe generative model is demonstrated by matching the flow of geodesics offactorizing discrete distributions. Various experiments underline theapproachs broad applicability. |


| Item |Content|
| --- |---|
|idx| 2402.07821v1 |
|title| On Computationally Efficient Multi-Class Calibration |
|authors| Parikshit GopalanLunjia HuGuy N. Rothblum
|links| http://arxiv.org/abs/2402.07821v1 |
|updated| 2024-02-12 17:25:23 UTC |
|summary| Consider a multi-class labelling problem where the labels can take values ink and a predictor predicts a distribution over the labels. In this workwe study the following foundational question: Are there notions of multi-classcalibration that give strong guarantees of meaningful predictions and can beachieved in time and sample complexities polynomial in k Prior notions ofcalibration exhibit a tradeoff between computational efficiency andexpressivity: they either suffer from having sample complexity exponential ink or needing to solve computationally intractable problems or give ratherweak guarantees.  Our main contribution is a notion of calibration that achieves all thesedesiderata: we formulate a robust notion of projected smooth calibration formulti-class predictions and give new recalibration algorithms for efficientlycalibrating predictors under this definition with complexity polynomial in k.Projected smooth calibration gives strong guarantees for all downstreamdecision makers who want to use the predictor for binary classificationproblems of the form: does the label belong to a subset T subseteq k: e.g.is this an image of an animal It ensures that the probabilities predicted bysumming the probabilities assigned to labels in T are close to some perfectlycalibrated binary predictor for that task. We also show that naturalstrengthenings of our definition are computationally hard to achieve: they runinto information theoretic barriers or computational intractability. Underlyingboth our upper and lower bounds is a tight connection that we prove betweenmulti-class calibration and the well-studied problem of agnostic learning inthe standard binary prediction setting. |


| Item |Content|
| --- |---|
|idx| 2402.07802v1 |
|title| Towards a mathematical theory for consistency training in diffusion models |
|authors| Gen LiZhihan HuangYuting Wei
|links| http://arxiv.org/abs/2402.07802v1 |
|updated| 2024-02-12 17:07:02 UTC |
|summary| Consistency models which were proposed to mitigate the high computationaloverhead during the sampling phase of diffusion models facilitate single-stepsampling while attaining state-of-the-art empirical performance. Whenintegrated into the training phase consistency models attempt to train asequence of consistency functions capable of mapping any point at any time stepof the diffusion process to its starting point. Despite the empirical successa comprehensive theoretical understanding of consistency training remainselusive. This paper takes a first step towards establishing theoreticalunderpinnings for consistency models. We demonstrate that in order to generatesamples within varepsilon proximity to the target in distribution measuredby some Wasserstein metric it suffices for the number of steps in consistencylearning to exceed the order of d5/2/varepsilon with d the datadimension. Our theory offers rigorous insights into the validity and efficacyof consistency models illuminating their utility in downstream inferencetasks. |


| Item |Content|
| --- |---|
|idx| 2402.07793v1 |
|title| Tuning-Free Stochastic Optimization |
|authors| Ahmed KhaledChi Jin
|links| http://arxiv.org/abs/2402.07793v1 |
|updated| 2024-02-12 16:59:06 UTC |
|summary| Large-scale machine learning problems make the cost of hyperparameter tuningever more prohibitive. This creates a need for algorithms that can tunethemselves on-the-fly. We formalize the notion of tuning-free algorithms thatcan match the performance of optimally-tuned optimization algorithms up topolylogarithmic factors given only loose hints on the relevant problemparameters. We consider in particular algorithms that can match optimally-tunedStochastic Gradient Descent SGD. When the domain of optimization is boundedwe show tuning-free matching of SGD is possible and achieved by severalexisting algorithms. We prove that for the task of minimizing a convex andsmooth or Lipschitz function over an unbounded domain tuning-free optimizationis impossible. We discuss conditions under which tuning-free optimization ispossible even over unbounded domains. In particular we show that the recentlyproposed DoG and DoWG algorithms are tuning-free when the noise distribution issufficiently well-behaved. For the task of finding a stationary point of asmooth and potentially nonconvex function we give a variant of SGD thatmatches the best-known high-probability convergence rate for tuned SGD at onlyan additional polylogarithmic cost. However we also give an impossibilityresult that shows no algorithm can hope to match the optimal expectedconvergence rate for tuned SGD with high probability. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2402.07897v1 |
|title| A holographic mobile-based application for practicing pronunciation of basic English vocabulary for Spanish speaking children |
|authors| R. CerezoV. CalderonC. Romero
|links| http://dx.doi.org/10.1016/j.ijhcs.2018.11.009 |
|updated| 2024-02-12 18:58:01 UTC |
|summary| This paper describes a holographic mobile-based application designed to helpSpanish-speaking children to practice the pronunciation of basic Englishvocabulary words. The mastery of vocabulary is a fundamental step when learninga language but is often perceived as boring. Producing the correctpronunciation is frequently regarded as the most difficult and complex skillfor new learners of English. In order to address these problems this researchtakes advantage of the power of multi-channel stimuli sound image andinteraction in a mobilebased hologram application in order to motivatestudents and improve their experience of practicing. We adapted theprize-winning HolograFX game and developed a new mobile application to helppractice English pronunciation. A 3D holographic robot that acts as a virtualteacher interacts via voice with the children. To test the tool we carried outan experiment with 70 Spanish pre-school children divided into three classesthe control group using traditional methods such as images in books and on theblackboard and two experimental groups using our drills and practice software.One experimental group used the mobile application without the holographic gameand the other experimental group used the application with the holographicgame. We performed pre-test and post-test performance assessments asatisfaction survey and emotion analysis. The results are very promising. Theyshow that the use of the holographic mobile-based application had a significantimpact on the childrens motivation. It also improved their performancecompared to traditional methods used in the classroom. |


| Item |Content|
| --- |---|
|idx| 2402.07864v1 |
|title| Cruising Queer HCI on the DL: A Literature Review of LGBTQ+ People in HCI |
|authors| Jordan TaylorEllen SimpsonAnh-Ton TranJed BrubakerSarah FoxHaiyi Zhu
|links| http://arxiv.org/abs/2402.07864v1 |
|updated| 2024-02-12 18:20:07 UTC |
|summary| LGBTQ people have received increased attention in HCI research parallelinga greater emphasis on social justice in recent years. However there has notbeen a systematic review of how LGBTQ people are researched or discussed inHCI. In this work we review all research mentioning LGBTQ people across theHCI venues of CHI CSCW DIS and TOCHI. Since 2014 we find a linear growth inthe number of papers substantially about LGBTQ people and an exponentialincrease in the number of mentions. Research about LGBTQ people tends tocenter experiences of being politicized outside the norm stigmatized orhighly vulnerable. LGBTQ people are typically mentioned as a marginalizedgroup or an area of future research. We identify gaps and opportunities for 1research about and 2 the discussion of LGBTQ in HCI and provide a dataset tofacilitate future Queer HCI research. |


| Item |Content|
| --- |---|
|idx| 2402.07687v1 |
|title| Privacy-Preserving Gaze Data Streaming in Immersive Interactive Virtual Reality: Robustness and User Experience |
|authors| Ethan WilsonAzim IbragimovMichael J. ProulxSai Deep TetaliKevin ButlerEakta Jain
|links| http://arxiv.org/abs/2402.07687v1 |
|updated| 2024-02-12 14:53:12 UTC |
|summary| Eye tracking is routinely being incorporated into virtual reality VRsystems. Prior research has shown that eye tracking data can be used forre-identification attacks. The state of our knowledge about currently existingprivacy mechanisms is limited to privacy-utility trade-off curves based ondata-centric metrics of utility such as prediction error and black-box threatmodels. We propose that for interactive VR applications it is essential toconsider user-centric notions of utility and a variety of threat models. Wedevelop a methodology to evaluate real-time privacy mechanisms for interactiveVR applications that incorporate subjective user experience and taskperformance metrics. We evaluate selected privacy mechanisms using thismethodology and find that re-identification accuracy can be decreased to as lowas 14 while maintaining a high usability score and reasonable taskperformance. Finally we elucidate three threat scenarios black-box black-boxwith exemplars and white-box and assess how well the different privacymechanisms hold up to these adversarial scenarios. This work advances the stateof the art in VR privacy by providing a methodology for end-to-end assessmentof the risk of re-identification attacks and potential mitigating solutions. |


| Item |Content|
| --- |---|
|idx| 2402.07632v1 |
|title| Overconfident and Unconfident AI Hinder Human-AI Collaboration |
|authors| Jingshu LiYitian YangYi-chieh Lee
|links| http://arxiv.org/abs/2402.07632v1 |
|updated| 2024-02-12 13:16:30 UTC |
|summary| As artificial intelligence AI advances human-AI collaboration has becomeincreasingly prevalent across both professional and everyday settings. In suchcollaboration AI can express its confidence level about its performanceserving as a crucial indicator for humans to evaluate AIs suggestions.However AI may exhibit overconfidence or underconfidence--its expressedconfidence is higher or lower than its actual performance--which may leadhumans to mistakenly evaluate AI advice. Our study investigates the influencesof AIs overconfidence and underconfidence on human trust their acceptance ofAI suggestions and collaboration outcomes. Our study reveal that disclosing AIconfidence levels and performance feedback facilitates better recognition of AIconfidence misalignments. However participants tend to withhold their trust asperceiving such misalignments leading to a rejection of AI suggestions andsubsequently poorer performance in collaborative tasks. Conversely withoutsuch information participants struggle to identify misalignments resulting ineither the neglect of correct AI advice or the following of incorrect AIsuggestions adversely affecting collaboration. This study offers valuableinsights for enhancing human-AI collaboration by underscoring the importance ofaligning AIs expressed confidence with its actual performance and thenecessity of calibrating human trust towards AI confidence. |


| Item |Content|
| --- |---|
|idx| 2402.07540v1 |
|title| PKG API: A Tool for Personal Knowledge Graph Management |
|authors| Nolwenn BernardIvica KostricWeronika ŁajewskaKrisztian BalogPetra GaluščákováVinay SettyMartin G. Skjæveland
|links| http://arxiv.org/abs/2402.07540v1 |
|updated| 2024-02-12 10:09:16 UTC |
|summary| Personal knowledge graphs PKGs offer individuals a way to store andconsolidate their fragmented personal data in a central place improvingservice personalization while maintaining full user control. Despite theirpotential practical PKG implementations with user-friendly interfaces remainscarce. This work addresses this gap by proposing a complete solution torepresent manage and interface with PKGs. Our approach includes 1 auser-facing PKG Client enabling end-users to administer their personal dataeasily via natural language statements and 2 a service-oriented PKG API. Totackle the complexity of representing these statements within a PKG we presentan RDF-based PKG vocabulary that supports this along with properties foraccess rights and provenance. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2402.07752v1 |
|title| Mixed Q-Functionals: Advancing Value-Based Methods in Cooperative MARL with Continuous Action Domains |
|authors| Yasin FindikS. Reza Ahmadzadeh
|links| http://arxiv.org/abs/2402.07752v1 |
|updated| 2024-02-12 16:21:50 UTC |
|summary| Tackling multi-agent learning problems efficiently is a challenging task incontinuous action domains. While value-based algorithms excel in sampleefficiency when applied to discrete action domains they are usuallyinefficient when dealing with continuous actions. Policy-based algorithms onthe other hand attempt to address this challenge by leveraging critic networksfor guiding the learning process and stabilizing the gradient estimation. Thelimitations in the estimation of true return and falling into local optima inthese methods result in inefficient and often sub-optimal policies. In thispaper we diverge from the trend of further enhancing critic networks andfocus on improving the effectiveness of value-based methods in multi-agentcontinuous domains by concurrently evaluating numerous actions. We propose anovel multi-agent value-based algorithm Mixed Q-Functionals MQF inspiredfrom the idea of Q-Functionals that enables agents to transform their statesinto basis functions. Our algorithm fosters collaboration among agents bymixing their action-values. We evaluate the efficacy of our algorithm in sixcooperative multi-agent scenarios. Our empirical findings reveal that MQFoutperforms four variants of Deep Deterministic Policy Gradient through rapidaction evaluation and increased sample efficiency. |


| Item |Content|
| --- |---|
|idx| 2402.07547v1 |
|title| Ensuring trustworthy and ethical behaviour in intelligent logical agents |
|authors| Stefania Costantini
|links| http://dx.doi.org/10.1093/logcom/exab091 |
|updated| 2024-02-12 10:19:17 UTC |
|summary| Autonomous Intelligent Agents are employed in many applications upon whichthe life and welfare of living beings and vital social functions may depend.Therefore agents should be trustworthy. A priori certification techniquesi.e. techniques applied prior to systems deployment can be useful but arenot sufficient for agents that evolve and thus modify their epistemic andbelief state and for open Multi-Agent Systems where heterogeneous agents canjoin or leave the system at any stage of its operation. In this paper wepropose/refine/extend dynamic runtime logic-based self-checking techniquesdevised in order to be able to ensure agents trustworthy and ethicalbehaviour. |


| Item |Content|
| --- |---|
|idx| 2402.07462v2 |
|title| A Hormetic Approach to the Value-Loading Problem: Preventing the Paperclip Apocalypse? |
|authors| Nathan I. N. HenryMangor PedersenMatt WilliamsJamin L. B. MartinLiesje Donkin
|links| http://arxiv.org/abs/2402.07462v2 |
|updated| 2024-02-13 05:21:40 UTC |
|summary| The value-loading problem is a significant challenge for researchers aimingto create artificial intelligence AI systems that align with human values andpreferences. This problem requires a method to define and regulate safe andoptimal limits of AI behaviors. In this work we propose HALO HormeticALignment via Opponent processes a regulatory paradigm that uses hormeticanalysis to regulate the behavioral patterns of AI. Behavioral hormesis is aphenomenon where low frequencies of a behavior have beneficial effects whilehigh frequencies are harmful. By modeling behaviors as allostatic opponentprocesses we can use either Behavioral Frequency Response Analysis BFRA orBehavioral Count Response Analysis BCRA to quantify the hormetic limits ofrepeatable behaviors. We demonstrate how HALO can solve the paperclipmaximizer scenario a thought experiment where an unregulated AI tasked withmaking paperclips could end up converting all matter in the universe intopaperclips. Our approach may be used to help create an evolving database ofvalues based on the hedonic calculus of repeatable behaviors with decreasingmarginal utility. This positions HALO as a promising solution for thevalue-loading problem which involves embedding human-aligned values into an AIsystem and the weak-to-strong generalization problem which explores whetherweak models can supervise stronger models as they become more intelligent.Hence HALO opens several research avenues that may lead to the development ofa computational value system that allows an AI algorithm to learn whether thedecisions it makes are right or wrong. |


| Item |Content|
| --- |---|
|idx| 2402.07437v1 |
|title| Learning Optimal Tax Design in Nonatomic Congestion Games |
|authors| Qiwen CuiMaryam FazelSimon S. Du
|links| http://arxiv.org/abs/2402.07437v1 |
|updated| 2024-02-12 06:32:53 UTC |
|summary| We study how to learn the optimal tax design to maximize the efficiency innonatomic congestion games. It is known that self-interested behavior among theplayers can damage the systems efficiency. Tax mechanisms is a common methodto alleviate this issue and induce socially optimal behavior. In this work wetake the initial step for learning the optimal tax that can minimize the socialcost with emphequilibrium feedback i.e. the tax designer can only observethe equilibrium state under the enforced tax. Existing algorithms are notapplicable due to the exponentially large tax function space nonexistence ofthe gradient and nonconvexity of the objective. To tackle these challengesour algorithm leverages several novel components: 1 piece-wise linear tax toapproximate the optimal tax 2 an extra linear term to guarantee a stronglyconvex potential function 3 efficient subroutine to find the boundarytax. The algorithm can find an epsilon-optimal tax with ObetaF2/epsilon sample complexity where beta is the smoothness of the costfunction and F is the number of facilities. |


| Item |Content|
| --- |---|
|idx| 2402.07404v1 |
|title| Enhancing Multi-Criteria Decision Analysis with AI: Integrating Analytic Hierarchy Process and GPT-4 for Automated Decision Support |
|authors| Igor SvobodaDmytro Lande
|links| http://arxiv.org/abs/2402.07404v1 |
|updated| 2024-02-12 04:47:38 UTC |
|summary| Our study presents a new framework that incorporates the Analytic HierarchyProcess AHP and Generative Pre-trained Transformer 4 GPT-4 large languagemodel LLM bringing novel approaches to cybersecurity Multiple-criteriaDecision Making MCDA. By utilizing the capabilities of GPT-4 autonomousagents as virtual experts we automate the decision-making process enhancingboth efficiency and reliability. This new approach focuses on leveraging LLMsfor sophisticated decision analysis highlighting the synergy betweentraditional decision-making models and cutting-edge AI technologies. Ourinnovative methodology demonstrates significant advancements in using AI-drivenagents for complex decision-making scenarios highlighting the importance of AIin strategic cybersecurity applications. The findings reveal the transformativepotential of combining AHP and LLMs establishing a new paradigm forintelligent decision support systems in cybersecurity and beyond. |


