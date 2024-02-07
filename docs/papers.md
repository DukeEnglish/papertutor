# cs.CL 

| Item |Content|
| --- |---|
|idx| 2402.03303v1 |
|title| Nevermind: Instruction Override and Moderation in Large Language Models |
|authors| Edward Kim
|links| http://arxiv.org/abs/2402.03303v1 |
|updated| 2024-02-05 18:58:19 UTC |
|summary| Given the impressive capabilities of recent Large Language Models LLMs weinvestigate and benchmark the most popular proprietary and different sized opensource models on the task of explicit instruction following in conflictingsituations e.g. overrides. These include the ability of the model to overridethe knowledge within the weights of the model the ability to override ormoderate extracted knowledge in the prompt and lastly the ability to performa full jailbreak. Experimentation performed suggest several key findings toimprove instruction following - larger models perform the best in followinginstructions that override internal and contextual instructions and areobedient even to a fault. When scaling to longer contexts via rope scaling asignificant buffer needs to be maintained from the edge of the perplexity cliffin order to maintain instruction following capabilities. Finally we observeimproving instruction following and subsequently instructionoverrides/jailbreaks is fundamentally at odds with the ability of a languagemodel to follow given safety filters or guidelines. Thus we postulate the mosteffective approach for safe trustworthy AI should be dealt external to the LLMitself. |


| Item |Content|
| --- |---|
|idx| 2402.03300v2 |
|title| DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
|authors| Zhihong ShaoPeiyi WangQihao ZhuRunxin XuJunxiao SongMingchuan ZhangY. K. LiY. WuDaya Guo
|links| http://arxiv.org/abs/2402.03300v2 |
|updated| 2024-02-06 18:39:38 UTC |
|summary| Mathematical reasoning poses a significant challenge for language models dueto its complex and structured nature. In this paper we introduce DeepSeekMath7B which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120Bmath-related tokens sourced from Common Crawl together with natural languageand code data. DeepSeekMath 7B has achieved an impressive score of 51.7 on thecompetition-level MATH benchmark without relying on external toolkits andvoting techniques approaching the performance level of Gemini-Ultra and GPT-4.Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9 on MATH.The mathematical reasoning capability of DeepSeekMath is attributed to two keyfactors: First we harness the significant potential of publicly available webdata through a meticulously engineered data selection pipeline. Second weintroduce Group Relative Policy Optimization GRPO a variant of ProximalPolicy Optimization PPO that enhances mathematical reasoning abilities whileconcurrently optimizing the memory usage of PPO. |


| Item |Content|
| --- |---|
|idx| 2402.03299v1 |
|title| GUARD: Role-playing to Generate Natural-language Jailbreakings to Test Guideline Adherence of Large Language Models |
|authors| Haibo JinRuoxi ChenAndy ZhouJinyin ChenYang ZhangHaohan Wang
|links| http://arxiv.org/abs/2402.03299v1 |
|updated| 2024-02-05 18:54:43 UTC |
|summary| The discovery of jailbreaks to bypass safety filters of Large LanguageModels LLMs and harmful responses have encouraged the community to implementsafety measures. One major safety measure is to proactively test the LLMs withjailbreaks prior to the release. Therefore such testing will require a methodthat can generate jailbreaks massively and efficiently. In this paper wefollow a novel yet intuitive strategy to generate jailbreaks in the style ofthe human generation. We propose a role-playing system that assigns fourdifferent roles to the user LLMs to collaborate on new jailbreaks. Furthermorewe collect existing jailbreaks and split them into different independentcharacteristics using clustering frequency and semantic patterns sentence bysentence. We organize these characteristics into a knowledge graph making themmore accessible and easier to retrieve. Our system of different roles willleverage this knowledge graph to generate new jailbreaks which have provedeffective in inducing LLMs to generate unethical or guideline-violatingresponses. In addition we also pioneer a setting in our system that willautomatically follow the government-issued guidelines to generate jailbreaks totest whether LLMs follow the guidelines accordingly. We refer to our system asGUARD Guideline Upholding through Adaptive Role-play Diagnostics. We haveempirically validated the effectiveness of GUARD on three cutting-edgeopen-sourced LLMs Vicuna-13B LongChat-7B and Llama-2-7B as well as awidely-utilized commercial LLM ChatGPT. Moreover our work extends to therealm of vision language models MiniGPT-v2 and Gemini Vision Pro showcasingGUARDs versatility and contributing valuable insights for the development ofsafer more reliable LLM-based applications across diverse modalities. |


| Item |Content|
| --- |---|
|idx| 2402.03284v1 |
|title| Deal, or no deal (or who knows)? Forecasting Uncertainty in Conversations using Large Language Models |
|authors| Anthony SiciliaHyunwoo KimKhyathi Raghavi ChanduMalihe AlikhaniJack Hessel
|links| http://arxiv.org/abs/2402.03284v1 |
|updated| 2024-02-05 18:39:47 UTC |
|summary| Effective interlocutors account for the uncertain goals beliefs andemotions of others. But even the best human conversationalist cannot perfectlyanticipate the trajectory of a dialogue. How well can language models representinherent uncertainty in conversations We propose FortUne Dial an expansion ofthe long-standing conversation forecasting task: instead of just accuracyevaluation is conducted with uncertainty-aware metrics effectively enablingabstention on individual instances. We study two ways in which language modelspotentially represent outcome uncertainty internally using scores anddirectly using tokens and propose fine-tuning strategies to improvecalibration of both representations. Experiments on eight difficult negotiationcorpora demonstrate that our proposed fine-tuning strategies a traditionalsupervision strategy and an off-policy reinforcement learning strategy cancalibrate smaller open-source models to compete with pre-trained models 10xtheir size. |


| Item |Content|
| --- |---|
|idx| 2402.03271v1 |
|title| Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models |
|authors| Zhiyuan HuChumin LiuXidong FengYilun ZhaoSee-Kiong NgAnh Tuan LuuJunxian HePang Wei KohBryan Hooi
|links| http://arxiv.org/abs/2402.03271v1 |
|updated| 2024-02-05 18:28:44 UTC |
|summary| In the face of uncertainty the ability to seek information is of fundamentalimportance. In many practical applications such as medical diagnosis andtroubleshooting the information needed to solve the task is not initiallygiven and has to be actively sought by asking follow-up questions forexample a doctor asking a patient for more details about their symptoms. Inthis work we introduce Uncertainty of Thoughts UoT an algorithm to augmentlarge language models with the ability to actively seek information by askingeffective questions. UoT combines 1 an uncertainty-aware simulation approachwhich enables the model to simulate possible future scenarios and how likelythey are to occur 2 uncertainty-based rewards motivated by information gainwhich incentivizes the model to seek information and 3 a reward propagationscheme to select the optimal question to ask in a way that maximizes theexpected reward. In experiments on medical diagnosis troubleshooting and the20 Questions game UoT achieves an average performance improvement of 57.8in the rate of successful task completion across multiple LLMs compared withdirect prompting and also improves efficiency i.e. the number of questionsneeded to complete the task. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2402.03311v1 |
|title| HASSOD: Hierarchical Adaptive Self-Supervised Object Detection |
|authors| Shengcao CaoDhiraj JoshiLiang-Yan GuiYu-Xiong Wang
|links| http://arxiv.org/abs/2402.03311v1 |
|updated| 2024-02-05 18:59:41 UTC |
|summary| The human visual perception system demonstrates exceptional capabilities inlearning without explicit supervision and understanding the part-to-wholecomposition of objects. Drawing inspiration from these two abilities wepropose Hierarchical Adaptive Self-Supervised Object Detection HASSOD anovel approach that learns to detect objects and understand their compositionswithout human supervision. HASSOD employs a hierarchical adaptive clusteringstrategy to group regions into object masks based on self-supervised visualrepresentations adaptively determining the number of objects per image.Furthermore HASSOD identifies the hierarchical levels of objects in terms ofcomposition by analyzing coverage relations between masks and constructingtree structures. This additional self-supervised learning task leads toimproved detection performance and enhanced interpretability. Lastly weabandon the inefficient multi-round self-training process utilized in priormethods and instead adapt the Mean Teacher framework from semi-supervisedlearning which leads to a smoother and more efficient training process.Through extensive experiments on prevalent image datasets we demonstrate thesuperiority of HASSOD over existing methods thereby advancing the state of theart in self-supervised object detection. Notably we improve Mask AR from 20.2to 22.5 on LVIS and from 17.0 to 26.0 on SA-1B. Project page:https://HASSOD-NeurIPS23.github.io. |


| Item |Content|
| --- |---|
|idx| 2402.03310v1 |
|title| V-IRL: Grounding Virtual Intelligence in Real Life |
|authors| Jihan YangRunyu DingEllis BrownXiaojuan QiSaining Xie
|links| http://arxiv.org/abs/2402.03310v1 |
|updated| 2024-02-05 18:59:36 UTC |
|summary| There is a sensory gulf between the Earth that humans inhabit and the digitalrealms in which modern AI agents are created. To develop AI agents that cansense think and act as flexibly as humans in real-world settings it isimperative to bridge the realism gap between the digital and physical worlds.How can we embody agents in an environment as rich and diverse as the one weinhabit without the constraints imposed by real hardware and control Towardsthis end we introduce V-IRL: a platform that enables agents to scalablyinteract with the real world in a virtual yet realistic environment. Ourplatform serves as a playground for developing agents that can accomplishvarious practical tasks and as a vast testbed for measuring progress incapabilities spanning perception decision-making and interaction withreal-world data across the entire globe. |


| Item |Content|
| --- |---|
|idx| 2402.03305v1 |
|title| Do Diffusion Models Learn Semantically Meaningful and Efficient Representations? |
|authors| Qiyao LiangZiming LiuIla Fiete
|links| http://arxiv.org/abs/2402.03305v1 |
|updated| 2024-02-05 18:58:38 UTC |
|summary| Diffusion models are capable of impressive feats of image generation withuncommon juxtapositions such as astronauts riding horses on the moon withproperly placed shadows. These outputs indicate the ability to performcompositional generalization but how do the models do so We performcontrolled experiments on conditional DDPMs learning to generate 2D sphericalGaussian bumps centered at specified x- and y-positions. Our results showthat the emergence of semantically meaningful latent representations is key toachieving high performance. En route to successful performance over learningthe model traverses three distinct phases of latent representations: phase Ano latent structure phase B a 2D manifold of disordered states and phaseC a 2D ordered manifold. Corresponding to each of these phases we identifyqualitatively different generation behaviors: 1 multiple bumps are generated2 one bump is generated but at inaccurate x and y locations 3 a bump isgenerated at the correct x and y location. Furthermore we show that evenunder imbalanced datasets where features x- versus y-positions arerepresented with skewed frequencies the learning process for x and y iscoupled rather than factorized demonstrating that simple vanilla-flavoreddiffusion models cannot learn efficient representations in which localizationin x and y are factorized into separate 1D tasks. These findings suggestthe need for future work to find inductive biases that will push generativemodels to discover and exploit factorizable independent structures in theirinputs which will be required to vault these models into more data-efficientregimes. |


| Item |Content|
| --- |---|
|idx| 2402.03303v1 |
|title| Nevermind: Instruction Override and Moderation in Large Language Models |
|authors| Edward Kim
|links| http://arxiv.org/abs/2402.03303v1 |
|updated| 2024-02-05 18:58:19 UTC |
|summary| Given the impressive capabilities of recent Large Language Models LLMs weinvestigate and benchmark the most popular proprietary and different sized opensource models on the task of explicit instruction following in conflictingsituations e.g. overrides. These include the ability of the model to overridethe knowledge within the weights of the model the ability to override ormoderate extracted knowledge in the prompt and lastly the ability to performa full jailbreak. Experimentation performed suggest several key findings toimprove instruction following - larger models perform the best in followinginstructions that override internal and contextual instructions and areobedient even to a fault. When scaling to longer contexts via rope scaling asignificant buffer needs to be maintained from the edge of the perplexity cliffin order to maintain instruction following capabilities. Finally we observeimproving instruction following and subsequently instructionoverrides/jailbreaks is fundamentally at odds with the ability of a languagemodel to follow given safety filters or guidelines. Thus we postulate the mosteffective approach for safe trustworthy AI should be dealt external to the LLMitself. |


| Item |Content|
| --- |---|
|idx| 2402.03300v2 |
|title| DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
|authors| Zhihong ShaoPeiyi WangQihao ZhuRunxin XuJunxiao SongMingchuan ZhangY. K. LiY. WuDaya Guo
|links| http://arxiv.org/abs/2402.03300v2 |
|updated| 2024-02-06 18:39:38 UTC |
|summary| Mathematical reasoning poses a significant challenge for language models dueto its complex and structured nature. In this paper we introduce DeepSeekMath7B which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120Bmath-related tokens sourced from Common Crawl together with natural languageand code data. DeepSeekMath 7B has achieved an impressive score of 51.7 on thecompetition-level MATH benchmark without relying on external toolkits andvoting techniques approaching the performance level of Gemini-Ultra and GPT-4.Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9 on MATH.The mathematical reasoning capability of DeepSeekMath is attributed to two keyfactors: First we harness the significant potential of publicly available webdata through a meticulously engineered data selection pipeline. Second weintroduce Group Relative Policy Optimization GRPO a variant of ProximalPolicy Optimization PPO that enhances mathematical reasoning abilities whileconcurrently optimizing the memory usage of PPO. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2402.03312v1 |
|title| Test-Time Adaptation for Depth Completion |
|authors| Hyoungseob ParkAnjali GuptaAlex Wong
|links| http://arxiv.org/abs/2402.03312v1 |
|updated| 2024-02-05 18:59:52 UTC |
|summary| It is common to observe performance degradation when transferring modelstrained on some source datasets to target testing data due to a domain gapbetween them. Existing methods for bridging this gap such as domain adaptationDA may require the source data on which the model was trained often notavailable while others i.e. source-free DA require many passes through thetesting data. We propose an online test-time adaptation method for depthcompletion the task of inferring a dense depth map from a single image andassociated sparse depth map that closes the performance gap in a single pass.We first present a study on how the domain shift in each data modality affectsmodel performance. Based on our observations that the sparse depth modalityexhibits a much smaller covariate shift than the image we design an embeddingmodule trained in the source domain that preserves a mapping from featuresencoding only sparse depth to those encoding image and sparse depth. Duringtest time sparse depth features are projected using this map as a proxy forsource domain features and are used as guidance to train a set of auxiliaryparameters i.e. adaptation layer to align image and sparse depth featuresfrom the target test domain to that of the source domain. We evaluate ourmethod on indoor and outdoor scenarios and show that it improves over baselinesby an average of 21.1. |


| Item |Content|
| --- |---|
|idx| 2402.03311v1 |
|title| HASSOD: Hierarchical Adaptive Self-Supervised Object Detection |
|authors| Shengcao CaoDhiraj JoshiLiang-Yan GuiYu-Xiong Wang
|links| http://arxiv.org/abs/2402.03311v1 |
|updated| 2024-02-05 18:59:41 UTC |
|summary| The human visual perception system demonstrates exceptional capabilities inlearning without explicit supervision and understanding the part-to-wholecomposition of objects. Drawing inspiration from these two abilities wepropose Hierarchical Adaptive Self-Supervised Object Detection HASSOD anovel approach that learns to detect objects and understand their compositionswithout human supervision. HASSOD employs a hierarchical adaptive clusteringstrategy to group regions into object masks based on self-supervised visualrepresentations adaptively determining the number of objects per image.Furthermore HASSOD identifies the hierarchical levels of objects in terms ofcomposition by analyzing coverage relations between masks and constructingtree structures. This additional self-supervised learning task leads toimproved detection performance and enhanced interpretability. Lastly weabandon the inefficient multi-round self-training process utilized in priormethods and instead adapt the Mean Teacher framework from semi-supervisedlearning which leads to a smoother and more efficient training process.Through extensive experiments on prevalent image datasets we demonstrate thesuperiority of HASSOD over existing methods thereby advancing the state of theart in self-supervised object detection. Notably we improve Mask AR from 20.2to 22.5 on LVIS and from 17.0 to 26.0 on SA-1B. Project page:https://HASSOD-NeurIPS23.github.io. |


| Item |Content|
| --- |---|
|idx| 2402.03309v1 |
|title| AONeuS: A Neural Rendering Framework for Acoustic-Optical Sensor Fusion |
|authors| Mohamad QadriKevin ZhangAkshay HindujaMichael KaessAdithya PediredlaChristopher A. Metzler
|links| http://arxiv.org/abs/2402.03309v1 |
|updated| 2024-02-05 18:59:31 UTC |
|summary| Underwater perception and 3D surface reconstruction are challenging problemswith broad applications in construction security marine archaeology andenvironmental monitoring. Treacherous operating conditions fragilesurroundings and limited navigation control often dictate that submersiblesrestrict their range of motion and thus the baseline over which they cancapture measurements. In the context of 3D scene reconstruction it iswell-known that smaller baselines make reconstruction more challenging. Ourwork develops a physics-based multimodal acoustic-optical neural surfacereconstruction framework AONeuS capable of effectively integratinghigh-resolution RGB measurements with low-resolution depth-resolved imagingsonar measurements. By fusing these complementary modalities our framework canreconstruct accurate high-resolution 3D surfaces from measurements capturedover heavily-restricted baselines. Through extensive simulations and in-labexperiments we demonstrate that AONeuS dramatically outperforms recentRGB-only and sonar-only inverse-differentiable-rendering--based surfacereconstruction methods. A website visualizing the results of our paper islocated at this address: https://aoneus.github.io/ |


| Item |Content|
| --- |---|
|idx| 2402.03305v1 |
|title| Do Diffusion Models Learn Semantically Meaningful and Efficient Representations? |
|authors| Qiyao LiangZiming LiuIla Fiete
|links| http://arxiv.org/abs/2402.03305v1 |
|updated| 2024-02-05 18:58:38 UTC |
|summary| Diffusion models are capable of impressive feats of image generation withuncommon juxtapositions such as astronauts riding horses on the moon withproperly placed shadows. These outputs indicate the ability to performcompositional generalization but how do the models do so We performcontrolled experiments on conditional DDPMs learning to generate 2D sphericalGaussian bumps centered at specified x- and y-positions. Our results showthat the emergence of semantically meaningful latent representations is key toachieving high performance. En route to successful performance over learningthe model traverses three distinct phases of latent representations: phase Ano latent structure phase B a 2D manifold of disordered states and phaseC a 2D ordered manifold. Corresponding to each of these phases we identifyqualitatively different generation behaviors: 1 multiple bumps are generated2 one bump is generated but at inaccurate x and y locations 3 a bump isgenerated at the correct x and y location. Furthermore we show that evenunder imbalanced datasets where features x- versus y-positions arerepresented with skewed frequencies the learning process for x and y iscoupled rather than factorized demonstrating that simple vanilla-flavoreddiffusion models cannot learn efficient representations in which localizationin x and y are factorized into separate 1D tasks. These findings suggestthe need for future work to find inductive biases that will push generativemodels to discover and exploit factorizable independent structures in theirinputs which will be required to vault these models into more data-efficientregimes. |


| Item |Content|
| --- |---|
|idx| 2402.03303v1 |
|title| Nevermind: Instruction Override and Moderation in Large Language Models |
|authors| Edward Kim
|links| http://arxiv.org/abs/2402.03303v1 |
|updated| 2024-02-05 18:58:19 UTC |
|summary| Given the impressive capabilities of recent Large Language Models LLMs weinvestigate and benchmark the most popular proprietary and different sized opensource models on the task of explicit instruction following in conflictingsituations e.g. overrides. These include the ability of the model to overridethe knowledge within the weights of the model the ability to override ormoderate extracted knowledge in the prompt and lastly the ability to performa full jailbreak. Experimentation performed suggest several key findings toimprove instruction following - larger models perform the best in followinginstructions that override internal and contextual instructions and areobedient even to a fault. When scaling to longer contexts via rope scaling asignificant buffer needs to be maintained from the edge of the perplexity cliffin order to maintain instruction following capabilities. Finally we observeimproving instruction following and subsequently instructionoverrides/jailbreaks is fundamentally at odds with the ability of a languagemodel to follow given safety filters or guidelines. Thus we postulate the mosteffective approach for safe trustworthy AI should be dealt external to the LLMitself. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2402.03312v1 |
|title| Test-Time Adaptation for Depth Completion |
|authors| Hyoungseob ParkAnjali GuptaAlex Wong
|links| http://arxiv.org/abs/2402.03312v1 |
|updated| 2024-02-05 18:59:52 UTC |
|summary| It is common to observe performance degradation when transferring modelstrained on some source datasets to target testing data due to a domain gapbetween them. Existing methods for bridging this gap such as domain adaptationDA may require the source data on which the model was trained often notavailable while others i.e. source-free DA require many passes through thetesting data. We propose an online test-time adaptation method for depthcompletion the task of inferring a dense depth map from a single image andassociated sparse depth map that closes the performance gap in a single pass.We first present a study on how the domain shift in each data modality affectsmodel performance. Based on our observations that the sparse depth modalityexhibits a much smaller covariate shift than the image we design an embeddingmodule trained in the source domain that preserves a mapping from featuresencoding only sparse depth to those encoding image and sparse depth. Duringtest time sparse depth features are projected using this map as a proxy forsource domain features and are used as guidance to train a set of auxiliaryparameters i.e. adaptation layer to align image and sparse depth featuresfrom the target test domain to that of the source domain. We evaluate ourmethod on indoor and outdoor scenarios and show that it improves over baselinesby an average of 21.1. |


| Item |Content|
| --- |---|
|idx| 2402.03311v1 |
|title| HASSOD: Hierarchical Adaptive Self-Supervised Object Detection |
|authors| Shengcao CaoDhiraj JoshiLiang-Yan GuiYu-Xiong Wang
|links| http://arxiv.org/abs/2402.03311v1 |
|updated| 2024-02-05 18:59:41 UTC |
|summary| The human visual perception system demonstrates exceptional capabilities inlearning without explicit supervision and understanding the part-to-wholecomposition of objects. Drawing inspiration from these two abilities wepropose Hierarchical Adaptive Self-Supervised Object Detection HASSOD anovel approach that learns to detect objects and understand their compositionswithout human supervision. HASSOD employs a hierarchical adaptive clusteringstrategy to group regions into object masks based on self-supervised visualrepresentations adaptively determining the number of objects per image.Furthermore HASSOD identifies the hierarchical levels of objects in terms ofcomposition by analyzing coverage relations between masks and constructingtree structures. This additional self-supervised learning task leads toimproved detection performance and enhanced interpretability. Lastly weabandon the inefficient multi-round self-training process utilized in priormethods and instead adapt the Mean Teacher framework from semi-supervisedlearning which leads to a smoother and more efficient training process.Through extensive experiments on prevalent image datasets we demonstrate thesuperiority of HASSOD over existing methods thereby advancing the state of theart in self-supervised object detection. Notably we improve Mask AR from 20.2to 22.5 on LVIS and from 17.0 to 26.0 on SA-1B. Project page:https://HASSOD-NeurIPS23.github.io. |


| Item |Content|
| --- |---|
|idx| 2402.03310v1 |
|title| V-IRL: Grounding Virtual Intelligence in Real Life |
|authors| Jihan YangRunyu DingEllis BrownXiaojuan QiSaining Xie
|links| http://arxiv.org/abs/2402.03310v1 |
|updated| 2024-02-05 18:59:36 UTC |
|summary| There is a sensory gulf between the Earth that humans inhabit and the digitalrealms in which modern AI agents are created. To develop AI agents that cansense think and act as flexibly as humans in real-world settings it isimperative to bridge the realism gap between the digital and physical worlds.How can we embody agents in an environment as rich and diverse as the one weinhabit without the constraints imposed by real hardware and control Towardsthis end we introduce V-IRL: a platform that enables agents to scalablyinteract with the real world in a virtual yet realistic environment. Ourplatform serves as a playground for developing agents that can accomplishvarious practical tasks and as a vast testbed for measuring progress incapabilities spanning perception decision-making and interaction withreal-world data across the entire globe. |


| Item |Content|
| --- |---|
|idx| 2402.03309v1 |
|title| AONeuS: A Neural Rendering Framework for Acoustic-Optical Sensor Fusion |
|authors| Mohamad QadriKevin ZhangAkshay HindujaMichael KaessAdithya PediredlaChristopher A. Metzler
|links| http://arxiv.org/abs/2402.03309v1 |
|updated| 2024-02-05 18:59:31 UTC |
|summary| Underwater perception and 3D surface reconstruction are challenging problemswith broad applications in construction security marine archaeology andenvironmental monitoring. Treacherous operating conditions fragilesurroundings and limited navigation control often dictate that submersiblesrestrict their range of motion and thus the baseline over which they cancapture measurements. In the context of 3D scene reconstruction it iswell-known that smaller baselines make reconstruction more challenging. Ourwork develops a physics-based multimodal acoustic-optical neural surfacereconstruction framework AONeuS capable of effectively integratinghigh-resolution RGB measurements with low-resolution depth-resolved imagingsonar measurements. By fusing these complementary modalities our framework canreconstruct accurate high-resolution 3D surfaces from measurements capturedover heavily-restricted baselines. Through extensive simulations and in-labexperiments we demonstrate that AONeuS dramatically outperforms recentRGB-only and sonar-only inverse-differentiable-rendering--based surfacereconstruction methods. A website visualizing the results of our paper islocated at this address: https://aoneus.github.io/ |


| Item |Content|
| --- |---|
|idx| 2402.03307v1 |
|title| 4D Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes |
|authors| Yuanxing DuanFangyin WeiQiyu DaiYuhang HeWenzheng ChenBaoquan Chen
|links| http://arxiv.org/abs/2402.03307v1 |
|updated| 2024-02-05 18:59:04 UTC |
|summary| We consider the problem of novel view synthesis NVS for dynamic scenes.Recent neural approaches have accomplished exceptional NVS results for static3D scenes but extensions to 4D time-varying scenes remain non-trivial. Priorefforts often encode dynamics by learning a canonical space plus implicit orexplicit deformation fields which struggle in challenging scenarios likesudden movements or capturing high-fidelity renderings. In this paper weintroduce 4D Gaussian Splatting 4DGS a novel method that represents dynamicscenes with anisotropic 4D XYZT Gaussians inspired by the success of 3DGaussian Splatting in static scenes. We model dynamics at each timestamp bytemporally slicing the 4D Gaussians which naturally compose dynamic 3DGaussians and can be seamlessly projected into images. As an explicitspatial-temporal representation 4DGS demonstrates powerful capabilities formodeling complicated dynamics and fine details especially for scenes withabrupt motions. We further implement our temporal slicing and splattingtechniques in a highly optimized CUDA acceleration framework achievingreal-time inference rendering speeds of up to 277 FPS on an RTX 3090 GPU and583 FPS on an RTX 4090 GPU. Rigorous evaluations on scenes with diverse motionsshowcase the superior efficiency and effectiveness of 4DGS which consistentlyoutperforms existing methods both quantitatively and qualitatively. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2402.03295v1 |
|title| Ginger: An Efficient Curvature Approximation with Linear Complexity for General Neural Networks |
|authors| Yongchang HaoYanshuai CaoLili Mou
|links| http://arxiv.org/abs/2402.03295v1 |
|updated| 2024-02-05 18:51:17 UTC |
|summary| Second-order optimization approaches like the generalized Gauss-Newton methodare considered more powerful as they utilize the curvature information of theobjective function with preconditioning matrices. Albeit offering temptingtheoretical benefits they are not easily applicable to modern deep learning.The major reason is due to the quadratic memory and cubic time complexity tocompute the inverse of the matrix. These requirements are infeasible even withstate-of-the-art hardware. In this work we propose Ginger aneigendecomposition for the inverse of the generalized Gauss-Newton matrix. Ourmethod enjoys efficient linear memory and time complexity for each iteration.Instead of approximating the conditioning matrix we directly maintain itsinverse to make the approximation more accurate. We provide the convergenceresult of Ginger for non-convex objectives. Our experiments on different taskswith different model architectures verify the effectiveness of our method. Ourcode is publicly available. |


| Item |Content|
| --- |---|
|idx| 2402.03293v1 |
|title| Flora: Low-Rank Adapters Are Secretly Gradient Compressors |
|authors| Yongchang HaoYanshuai CaoLili Mou
|links| http://arxiv.org/abs/2402.03293v1 |
|updated| 2024-02-05 18:50:39 UTC |
|summary| Despite large neural networks demonstrating remarkable abilities to completedifferent tasks they require excessive memory usage to store the optimizationstates for training. To alleviate this the low-rank adaptation LoRA isproposed to reduce the optimization states by training fewer parameters.However LoRA restricts overall weight update matrices to be low-rank limitingthe model performance. In this work we investigate the dynamics of LoRA andidentify that it can be approximated by a random projection. Based on thisobservation we propose Flora which is able to achieve high-rank updates byresampling the projection matrices while enjoying the sublinear spacecomplexity of optimization states. We conduct experiments across differenttasks and model architectures to verify the effectiveness of our approach. |


| Item |Content|
| --- |---|
|idx| 2402.03282v1 |
|title| A Framework for Partially Observed Reward-States in RLHF |
|authors| Chinmaya KausikMirco MuttiAldo PacchianoAmbuj Tewari
|links| http://arxiv.org/abs/2402.03282v1 |
|updated| 2024-02-05 18:38:55 UTC |
|summary| The study of reinforcement learning from human feedback RLHF has gainedprominence in recent years due to its role in the development of LLMs.Neuroscience research shows that human responses to stimuli are known to dependon partially-observed internal states. Unfortunately current models of RLHFdo not take take this into consideration. Moreover most RLHF models do notaccount for intermediate feedback which is gaining importance in empiricalwork and can help improve both sample complexity and alignment. To addressthese limitations we model RLHF as reinforcement learning with partiallyobserved reward-states PORRL. We show reductions from the the two dominantforms of human feedback in RLHF - cardinal and dueling feedback to PORRL. Forcardinal feedback we develop generic statistically efficient algorithms andinstantiate them to present POR-UCRL and POR-UCBVI. For dueling feedback weshow that a naive reduction to cardinal feedback fails to achieve sublineardueling regret. We then present the first explicit reduction that convertsguarantees for cardinal regret to dueling regret. We show that our models andguarantees in both settings generalize and extend existing ones. Finally weidentify a recursive structure on our model that could improve the statisticaland computational tractability of PORRL giving examples from past work on RLHFas well as learning perfect reward machines which PORRL subsumes. |


| Item |Content|
| --- |---|
|idx| 2402.03256v1 |
|title| Learning Best-in-Class Policies for the Predict-then-Optimize Framework |
|authors| Michael HuangVishal Gupta
|links| http://arxiv.org/abs/2402.03256v1 |
|updated| 2024-02-05 18:14:28 UTC |
|summary| We propose a novel family of decision-aware surrogate losses calledPerturbation Gradient PG losses for the predict-then-optimize framework.These losses directly approximate the downstream decision loss and can beoptimized using off-the-shelf gradient-based methods. Importantly unlikeexisting surrogate losses the approximation error of our PG losses vanishes asthe number of samples grows. This implies that optimizing our surrogate lossyields a best-in-class policy asymptotically even in misspecified settings.This is the first such result in misspecified settings and we provide numericalevidence confirming our PG losses substantively outperform existing proposalswhen the underlying model is misspecified and the noise is not centrallysymmetric. Insofar as misspecification is commonplace in practice -- especiallywhen we might prefer a simpler more interpretable model -- PG losses offer anovel theoretically justified method for computationally tractabledecision-aware learning. |


| Item |Content|
| --- |---|
|idx| 2402.03254v1 |
|title| Minimum Description Length and Generalization Guarantees for Representation Learning |
|authors| Milad SefidgaranAbdellatif ZaidiPiotr Krasnowski
|links| http://arxiv.org/abs/2402.03254v1 |
|updated| 2024-02-05 18:12:28 UTC |
|summary| A major challenge in designing efficient statistical supervised learningalgorithms is finding representations that perform well not only on availabletraining samples but also on unseen data. While the study of representationlearning has spurred much interest most existing such approaches areheuristic and very little is known about theoretical generalizationguarantees.  In this paper we establish a compressibility framework that allows us toderive upper bounds on the generalization error of a representation learningalgorithm in terms of the Minimum Description Length MDL of the labels orthe latent variables representations. Rather than the mutual informationbetween the encoders input and the representation which is often believed toreflect the algorithms generalization capability in the related literature butin fact falls short of doing so our new bounds involve the multi-letterrelative entropy between the distribution of the representations or labels ofthe training and test sets and a fixed prior. In particular these new boundsreflect the structure of the encoder and are not vacuous for deterministicalgorithms. Our compressibility approach which is information-theoretic innature builds upon that of Blum-Langford for PAC-MDL bounds and introduces twoessential ingredients: block-coding and lossy-compression. The latter allowsour approach to subsume the so-called geometrical compressibility as a specialcase. To the best knowledge of the authors the established generalizationbounds are the first of their kind for Information Bottleneck IB typeencoders and representation learning. Finally we partly exploit thetheoretical results by introducing a new data-dependent prior. Numericalsimulations illustrate the advantages of well-chosen such priors over classicalpriors used in IB. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2402.03291v1 |
|title| Knowledge Acquisition and Integration with Expert-in-the-loop |
|authors| Sajjadur RahmanFrederick ChoiHannah KimDan ZhangEstevam Hruschka
|links| http://arxiv.org/abs/2402.03291v1 |
|updated| 2024-02-05 18:49:55 UTC |
|summary| Constructing and serving knowledge graphs KGs is an iterative andhuman-centered process involving on-demand programming and analysis. In thispaper we present Kyurem a programmable and interactive widget library thatfacilitates human-in-the-loop knowledge acquisition and integration to enablecontinuous curation a knowledge graph KG. Kyurem provides a seamlessenvironment within computational notebooks where data scientists explore a KGto identify opportunities for acquiring new knowledge and verifyrecommendations provided by AI agents for integrating the acquired knowledge inthe KG. We refined Kyurem through participatory design and conducted casestudies in a real-world setting for evaluation. The case-studies show thatintroduction of Kyurem within an existing HR knowledge graph construction andserving platform improved the user experience of the experts and helpederadicate inefficiencies related to knowledge acquisition and integration tasks |


| Item |Content|
| --- |---|
|idx| 2402.03279v1 |
|title| Stepping into the Right Shoes: The Effects of User-Matched Avatar Ethnicity and Gender on Sense of Embodiment in Virtual Reality |
|authors| Tiffany D. DoCamille IsabellaRyan P. McMahan
|links| http://arxiv.org/abs/2402.03279v1 |
|updated| 2024-02-05 18:36:11 UTC |
|summary| In many consumer virtual reality VR applications users embody predefinedcharacters that offer minimal customization options frequently emphasizingstorytelling over user choice. We explore whether matching a users physicalcharacteristics specifically ethnicity and gender with their virtualself-avatar affects their sense of embodiment in VR. We conducted a 2 x 2within-subjects experiment n32 with a diverse user population to explore theimpact of matching or not matching a users self-avatar to their ethnicity andgender on their sense of embodiment. Our results indicate that matching theethnicity of the user and their self-avatar significantly enhances sense ofembodiment regardless of gender extending across various aspects includingappearance response and ownership. We also found that matching gendersignificantly enhanced ownership suggesting that this aspect is influenced bymatching both ethnicity and gender. Interestingly we found that matchingethnicity specifically affects self-location while matching gender specificallyaffects ones body ownership. |


| Item |Content|
| --- |---|
|idx| 2402.03259v1 |
|title| Meeting Bridges: Designing Information Artifacts that Bridge from Synchronous Meetings to Asynchronous Collaboration |
|authors| Ruotong WangLin QiuJustin CranshawAmy X. Zhang
|links| http://arxiv.org/abs/2402.03259v1 |
|updated| 2024-02-05 18:17:15 UTC |
|summary| A recent surge in remote meetings has led to complaints of Zoom fatigueand collaboration overload negatively impacting worker productivity andwell-being. One way to alleviate the burden of meetings is to de-emphasizetheir synchronous participation by shifting work to and enabling sensemakingduring post-meeting asynchronous activities. Towards this goal we propose thedesign concept of meeting bridges or information artifacts that canencapsulate meeting information towards bridging to and facilitatingpost-meeting activities. Through 13 interviews and a survey of 198 informationworkers we learn how people use online meeting information after meetings areover finding five main uses: as an archive as task reminders to onboard orsupport inclusion for group sensemaking and as a launching point forfollow-on collaboration. However we also find that current common meetingartifacts such as notes and recordings present challenges in serving asmeeting bridges. After conducting co-design sessions with 16 participants wedistill key principles for the design of meeting bridges to optimally supportasynchronous collaboration goals. Overall our findings point to theopportunity of designing information artifacts that not only support users toaccess but also continue to transform and engage in meeting informationpost-meeting. |


| Item |Content|
| --- |---|
|idx| 2402.03255v1 |
|title| Security Advice for Parents and Children About Content Filtering and Circumvention as Found on YouTube and TikTok |
|authors| Ran ElgedawyJohn SadikAnuj GautamTrinity BissahoyoChristopher ChildressJacob LeonardClay ShubertScott Ruoti
|links| http://arxiv.org/abs/2402.03255v1 |
|updated| 2024-02-05 18:12:33 UTC |
|summary| In todays digital age concerns about online security and privacy havebecome paramount. However addressing these issues can be difficult especiallywithin the context of family relationships wherein parents and children mayhave conflicting interests. In this environment parents and children may turnto online security advice to determine how to proceed. In this paper weexamine the advice available to parents and children regarding contentfiltering and circumvention as found on YouTube and TikTok. In an analysis of839 videos returned from queries on these topics we found that half n399provide relevant advice. Our results show that of these videos roughlythree-quarters are accurate with the remaining one-fourth containing factuallyincorrect advice. We find that videos targeting children are both more likelyto be incorrect and actionable than videos targeting parents leaving childrenat increased risk of taking harmful action. Moreover we find that while advicevideos targeting parents will occasionally discuss the ethics of contentfiltering and device monitoring including recommendations to respectchildrens autonomy no such discussion of the ethics or risks of circumventingcontent filtering is given to children leaving them unaware of any risks thatmay be involved with doing so. Ultimately our research indicates thatvideo-based social media sites are already effective sources of security advicepropagation and that the public would benefit from security researchers andpractitioners engaging more with these platforms both for the creation ofcontent and of tools designed to help with more effective filtering. |


| Item |Content|
| --- |---|
|idx| 2402.03116v1 |
|title| Feature-Action Design Patterns for Storytelling Visualizations with Time Series Data |
|authors| Saiful KhanScott JonesBenjamin BachJaehoon ChaMin ChenJulie MeikleJonathan C RobertsJeyan ThiyagalingamJo WoodPanagiotis D. Ritsos
|links| http://arxiv.org/abs/2402.03116v1 |
|updated| 2024-02-05 15:45:59 UTC |
|summary| We present a method to create storytelling visualization with time seriesdata. Many personal decisions nowadays rely on access to dynamic dataregularly as we have seen during the COVID-19 pandemic. It is thus desirableto construct storytelling visualization for dynamic data that is selected by anindividual for a specific context. Because of the need to tell data-dependentstories predefined storyboards based on known data cannot accommodate dynamicdata easily nor scale up to many different individuals and contexts. Motivatedinitially by the need to communicate time series data during the COVID-19pandemic we developed a novel computer-assisted method for meta-authoring ofstories which enables the design of storyboards that include feature-actionpatterns in anticipation of potential features that may appear in dynamicallyarrived or selected data. In addition to meta-storyboards involving COVID-19data we also present storyboards for telling stories about progress in amachine learning workflow. Our approach is complementary to traditional methodsfor authoring storytelling visualization and provides an efficient means toconstruct data-dependent storyboards for different data-streams of similarcontexts. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2402.03048v1 |
|title| Cooperative Learning with Gaussian Processes for Euler-Lagrange Systems Tracking Control under Switching Topologies |
|authors| Zewen YangSongbo DongArmin LedererXiaobing DaiSiyu ChenStefan SosnowskiGeorges HattabSandra Hirche
|links| http://arxiv.org/abs/2402.03048v1 |
|updated| 2024-02-05 14:33:52 UTC |
|summary| This work presents an innovative learning-based approach to tackle thetracking control problem of Euler-Lagrange multi-agent systems with partiallyunknown dynamics operating under switching communication topologies. Theapproach leverages a correlation-aware cooperative algorithm framework builtupon Gaussian process regression which adeptly captures inter-agentcorrelations for uncertainty predictions. A standout feature is its exceptionalefficiency in deriving the aggregation weights achieved by circumventing thecomputationally intensive posterior variance calculations. Through Lyapunovstability analysis the distributed control law ensures bounded tracking errorswith high probability. Simulation experiments validate the protocols efficacyin effectively managing complex scenarios establishing it as a promisingsolution for robust tracking control in multi-agent systems characterized byuncertain dynamics and dynamic communication structures. |


| Item |Content|
| --- |---|
|idx| 2402.02896v1 |
|title| LLM Agents in Interaction: Measuring Personality Consistency and Linguistic Alignment in Interacting Populations of Large Language Models |
|authors| Ivar FrischMario Giulianelli
|links| http://arxiv.org/abs/2402.02896v1 |
|updated| 2024-02-05 11:05:20 UTC |
|summary| While both agent interaction and personalisation are vibrant topics inresearch on large language models LLMs there has been limited focus on theeffect of language interaction on the behaviour of persona-conditioned LLMagents. Such an endeavour is important to ensure that agents remain consistentto their assigned traits yet are able to engage in open naturalisticdialogues. In our experiments we condition GPT-3.5 on personality profilesthrough prompting and create a two-group population of LLM agents using asimple variability-inducing sampling algorithm. We then administer personalitytests and submit the agents to a collaborative writing task finding thatdifferent profiles exhibit different degrees of personality consistency andlinguistic alignment to their conversational partners. Our study seeks to laythe groundwork for better understanding of dialogue-based interaction betweenLLMs and highlights the need for new approaches to crafting robust morehuman-like LLM personas for interactive environments. |


| Item |Content|
| --- |---|
|idx| 2402.02468v1 |
|title| Fast Peer Adaptation with Context-aware Exploration |
|authors| Long MaYuanfei WangFangwei ZhongSong-Chun ZhuYizhou Wang
|links| http://arxiv.org/abs/2402.02468v1 |
|updated| 2024-02-04 13:02:27 UTC |
|summary| Fast adapting to unknown peers partners or opponents with differentstrategies is a key challenge in multi-agent games. To do so it is crucial forthe agent to efficiently probe and identify the peers strategy as this is theprerequisite for carrying out the best response in adaptation. However it isdifficult to explore the strategies of unknown peers especially when the gamesare partially observable and have a long horizon. In this paper we propose apeer identification reward which rewards the learning agent based on how wellit can identify the behavior pattern of the peer over the historical contextsuch as the observation over multiple episodes. This reward motivates the agentto learn a context-aware policy for effective exploration and fast adaptationi.e. to actively seek and collect informative feedback from peers whenuncertain about their policies and to exploit the context to perform the bestresponse when confident. We evaluate our method on diverse testbeds thatinvolve competitive Kuhn Poker cooperative PO-Overcooked or mixedPredator-Prey-W games with peer agents. We demonstrate that our methodinduces more active exploration behavior achieving faster adaptation andbetter outcomes than existing methods. |


| Item |Content|
| --- |---|
|idx| 2402.02097v1 |
|title| Settling Decentralized Multi-Agent Coordinated Exploration by Novelty Sharing |
|authors| Haobin JiangZiluo DingZongqing Lu
|links| http://arxiv.org/abs/2402.02097v1 |
|updated| 2024-02-03 09:35:25 UTC |
|summary| Exploration in decentralized cooperative multi-agent reinforcement learningfaces two challenges. One is that the novelty of global states is unavailablewhile the novelty of local observations is biased. The other is how agents canexplore in a coordinated way. To address these challenges we propose MACE asimple yet effective multi-agent coordinated exploration method. Bycommunicating only local novelty agents can take into account other agentslocal novelty to approximate the global novelty. Further we newly introduceweighted mutual information to measure the influence of one agents action onother agents accumulated novelty. We convert it as an intrinsic reward inhindsight to encourage agents to exert more influence on other agentsexploration and boost coordinated exploration. Empirically we show that MACEachieves superior performance in three multi-agent environments with sparserewards. |


| Item |Content|
| --- |---|
|idx| 2402.01968v1 |
|title| A Survey on Context-Aware Multi-Agent Systems: Techniques, Challenges and Future Directions |
|authors| Hung DuSrikanth ThudumuRajesh VasaKon Mouzakis
|links| http://arxiv.org/abs/2402.01968v1 |
|updated| 2024-02-03 00:27:22 UTC |
|summary| Research interest in autonomous agents is on the rise as an emerging topic.The notable achievements of Large Language Models LLMs have demonstrated theconsiderable potential to attain human-like intelligence in autonomous agents.However the challenge lies in enabling these agents to learn reason andnavigate uncertainties in dynamic environments. Context awareness emerges as apivotal element in fortifying multi-agent systems when dealing with dynamicsituations. Despite existing research focusing on both context-aware systemsand multi-agent systems there is a lack of comprehensive surveys outliningtechniques for integrating context-aware systems with multi-agent systems. Toaddress this gap this survey provides a comprehensive overview ofstate-of-the-art context-aware multi-agent systems. First we outline theproperties of both context-aware systems and multi-agent systems thatfacilitate integration between these systems. Subsequently we propose ageneral process for context-aware systems with each phase of the processencompassing diverse approaches drawn from various application domains such ascollision avoidance in autonomous driving disaster relief management utilitymanagement supply chain management human-AI interaction and others. Finallywe discuss the existing challenges of context-aware multi-agent systems andprovide future research directions in this field. |


