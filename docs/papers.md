# cs.CL 

| Item |Content|
| --- |---|
|idx| 2408.12599v1 |
|title| Controllable Text Generation for Large Language Models: A Survey |
|authors| Xun LiangHanyu WangYezhaohui WangShichao SongJiawei YangSimin NiuJie HuDan LiuShunyu YaoFeiyu XiongZhiyu Li
|links| http://arxiv.org/abs/2408.12599v1 |
|updated| 2024-08-22 17:59:04 UTC |
|summary| In Natural Language Processing NLP Large Language Models LLMs havedemonstrated high text generation quality. However in real-world applicationsLLMs must meet increasingly complex requirements. Beyond avoiding misleading orinappropriate content LLMs are also expected to cater to specific user needssuch as imitating particular writing styles or generating text with poeticrichness. These varied demands have driven the development of Controllable TextGeneration CTG techniques which ensure that outputs adhere to predefinedcontrol conditions--such as safety sentiment thematic consistency andlinguistic style--while maintaining high standards of helpfulness fluency anddiversity.  This paper systematically reviews the latest advancements in CTG for LLMsoffering a comprehensive definition of its core concepts and clarifying therequirements for control conditions and text quality. We categorize CTG tasksinto two primary types: content control and attribute control. The key methodsare discussed including model retraining fine-tuning reinforcement learningprompt engineering latent space manipulation and decoding-time intervention.We analyze each methods characteristics advantages and limitationsproviding nuanced insights for achieving generation control. Additionally wereview CTG evaluation methods summarize its applications across domains andaddress key challenges in current research including reduced fluency andpracticality. We also propose several appeals such as placing greater emphasison real-world applications in future research. This paper aims to offervaluable guidance to researchers and developers in the field. Our referencelist and Chinese version are open-sourced athttps://github.com/IAAR-Shanghai/CTGSurvey. |


| Item |Content|
| --- |---|
|idx| 2408.12579v1 |
|title| RuleAlign: Making Large Language Models Better Physicians with Diagnostic Rule Alignment |
|authors| Xiaohan WangXiaoyan YangYuqi ZhuYue ShenJian WangPeng WeiLei LiangJinjie GuHuajun ChenNingyu Zhang
|links| http://arxiv.org/abs/2408.12579v1 |
|updated| 2024-08-22 17:44:40 UTC |
|summary| Large Language Models LLMs like GPT-4 MedPaLM-2 and Med-Gemini achieveperformance competitively with human experts across various medical benchmarks.However they still face challenges in making professional diagnoses akin tophysicians particularly in efficiently gathering patient information andreasoning the final diagnosis. To this end we introduce the RuleAlignframework designed to align LLMs with specific diagnostic rules. We develop amedical dialogue dataset comprising rule-based communications between patientsand physicians and design an alignment learning approach through preferencelearning. Experimental results demonstrate the effectiveness of the proposedapproach. We hope that our work can serve as an inspiration for exploring thepotential of LLMs as AI physicians. |


| Item |Content|
| --- |---|
|idx| 2408.12574v1 |
|title| MuMA-ToM: Multi-modal Multi-Agent Theory of Mind |
|authors| Haojun ShiSuyu YeXinyu FangChuanyang JinLayla IsikYen-Ling KuoTianmin Shu
|links| http://arxiv.org/abs/2408.12574v1 |
|updated| 2024-08-22 17:41:45 UTC |
|summary| Understanding peoples social interactions in complex real-world scenariosoften relies on intricate mental reasoning. To truly understand how and whypeople interact with one another we must infer the underlying mental statesthat give rise to the social interactions i.e. Theory of Mind reasoning inmulti-agent interactions. Additionally social interactions are oftenmulti-modal -- we can watch peoples actions hear their conversations and/orread about their past behaviors. For AI systems to successfully and safelyinteract with people in real-world environments they also need to understandpeoples mental states as well as their inferences about each others mentalstates based on multi-modal information about their interactions. For this weintroduce MuMA-ToM a Multi-modal Multi-Agent Theory of Mind benchmark.MuMA-ToM is the first multi-modal Theory of Mind benchmark that evaluatesmental reasoning in embodied multi-agent interactions. In MuMA-ToM we providevideo and text descriptions of peoples multi-modal behavior in realistichousehold environments. Based on the context we then ask questions aboutpeoples goals beliefs and beliefs about others goals. We validated MuMA-ToMin a human experiment and provided a human baseline. We also proposed a novelmulti-modal multi-agent ToM model LIMP Language model-based InverseMulti-agent Planning. Our experimental results show that LIMP significantlyoutperforms state-of-the-art methods including large multi-modal models e.g.GPT-4o Gemini-1.5 Pro and a recent multi-modal ToM model BIP-ALM. |


| Item |Content|
| --- |---|
|idx| 2408.12570v1 |
|title| Jamba-1.5: Hybrid Transformer-Mamba Models at Scale |
|authors| Jamba TeamBarak LenzAlan AraziAmir BergmanAvshalom ManevichBarak PelegBen AviramChen AlmagorClara FridmanDan PadnosDaniel GissinDaniel JannaiDor MuhlgayDor ZimbergEdden M GerberElad DolevEran KrakovskyErez SafahiErez SchwartzGal CohenGal ShachafHaim RozenblumHofit BataIdo BlassInbal MagarItay DalmedigosJhonathan OsinJulie FadlonMaria RozmanMatan DanosMichael GokhmanMor ZusmanNaama GidronNir RatnerNoam GatNoam RozenOded FriedOhad LeshnoOmer AntvergOmri AbendOpher LieberOr DaganOrit CohaviRaz AlonRo'i BelsonRoi CohenRom GiladRoman GlozmanShahar LevShaked MeiromTal DelbariTal NessTomer AsidaTom Ben GalTom BraudeUriya PumerantzYehoshua CohenYonatan BelinkovYuval GlobersonYuval Peleg LevyYoav Shoham
|links| http://arxiv.org/abs/2408.12570v1 |
|updated| 2024-08-22 17:38:59 UTC |
|summary| We present Jamba-1.5 new instruction-tuned large language models based onour Jamba architecture. Jamba is a hybrid Transformer-Mamba mixture of expertsarchitecture providing high throughput and low memory usage across contextlengths while retaining the same or better quality as Transformer models. Werelease two model sizes: Jamba-1.5-Large with 94B active parameters andJamba-1.5-Mini with 12B active parameters. Both models are fine-tuned for avariety of conversational and instruction-following capabilties and have aneffective context length of 256K tokens the largest amongst open-weightmodels. To support cost-effective inference we introduce ExpertsInt8 a novelquantization technique that allows fitting Jamba-1.5-Large on a machine with 880GB GPUs when processing 256K-token contexts without loss of quality. Whenevaluated on a battery of academic and chatbot benchmarks Jamba-1.5 modelsachieve excellent results while providing high throughput and outperformingother open-weight models on long-context benchmarks. The model weights for bothsizes are publicly available under the Jamba Open Model License and we releaseExpertsInt8 as open source. |


| Item |Content|
| --- |---|
|idx| 2408.12547v1 |
|title| Towards Evaluating and Building Versatile Large Language Models for Medicine |
|authors| Chaoyi WuPengcheng QiuJinxin LiuHongfei GuNa LiYa ZhangYanfeng WangWeidi Xie
|links| http://arxiv.org/abs/2408.12547v1 |
|updated| 2024-08-22 17:01:34 UTC |
|summary| In this study we present MedS-Bench a comprehensive benchmark designed toevaluate the performance of large language models LLMs in clinical contexts.Unlike existing benchmarks that focus on multiple-choice question answeringMedS-Bench spans 11 high-level clinical tasks including clinical reportsummarization treatment recommendations diagnosis named entity recognitionand medical concept explanation among others. We evaluated six leading LLMse.g. MEDITRON Mistral InternLM 2 Llama 3 GPT-4 and Claude-3.5 usingfew-shot prompting and found that even the most sophisticated models strugglewith these complex tasks. To address these limitations we developed MedS-Insa large-scale instruction tuning dataset for medicine. MedS-Ins comprises 58medically oriented language corpora totaling 13.5 million samples across 122tasks. To demonstrate the datasets utility we conducted a proof-of-conceptexperiment by performing instruction tuning on a lightweight open-sourcemedical language model. The resulting model MMedIns-Llama 3 significantlyoutperformed existing models across nearly all clinical tasks. To promotefurther advancements in the application of LLMs to clinical challenges we havemade the MedS-Ins dataset fully accessible and invite the research community tocontribute to its expansion.Additionally we have launched a dynamicleaderboard for MedS-Bench which we plan to regularly update the test set totrack progress and enhance the adaptation of general LLMs to the medicaldomain. Leaderboard: https://henrychur.github.io/MedS-Bench/. Github:https://github.com/MAGIC-AI4Med/MedS-Ins. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2408.12598v1 |
|title| ND-SDF: Learning Normal Deflection Fields for High-Fidelity Indoor Reconstruction |
|authors| Ziyu TangWeicai YeYifan WangDi HuangHujun BaoTong HeGuofeng Zhang
|links| http://arxiv.org/abs/2408.12598v1 |
|updated| 2024-08-22 17:59:01 UTC |
|summary| Neural implicit reconstruction via volume rendering has demonstrated itseffectiveness in recovering dense 3D surfaces. However it is non-trivial tosimultaneously recover meticulous geometry and preserve smoothness acrossregions with differing characteristics. To address this issue previous methodstypically employ geometric priors which are often constrained by theperformance of the prior models. In this paper we propose ND-SDF which learnsa Normal Ddeflection field to represent the angular deviation between the scenenormal and the prior normal. Unlike previous methods that uniformly applygeometric priors on all samples introducing significant bias in accuracy ourproposed normal deflection field dynamically learns and adapts the utilizationof samples based on their specific characteristics thereby improving both theaccuracy and effectiveness of the model. Our method not only obtains smoothweakly textured regions such as walls and floors but also preserves thegeometric details of complex structures. In addition we introduce a novel raysampling strategy based on the deflection angle to facilitate the unbiasedrendering process which significantly improves the quality and accuracy ofintricate surfaces especially on thin structures. Consistent improvements onvarious challenging datasets demonstrate the superiority of our method. |


| Item |Content|
| --- |---|
|idx| 2408.12591v1 |
|title| Differentiable Logic Programming for Distant Supervision |
|authors| Akihiro TakemuraKatsumi Inoue
|links| http://arxiv.org/abs/2408.12591v1 |
|updated| 2024-08-22 17:55:52 UTC |
|summary| We introduce a new method for integrating neural networks with logicprogramming in Neural-Symbolic AI NeSy aimed at learning with distantsupervision in which direct labels are unavailable. Unlike prior methods ourapproach does not depend on symbolic solvers for reasoning about missinglabels. Instead it evaluates logical implications and constraints in adifferentiable manner by embedding both neural network outputs and logicprograms into matrices. This method facilitates more efficient learning underdistant supervision. We evaluated our approach against existing methods whilemaintaining a constant volume of training data. The findings indicate that ourmethod not only matches or exceeds the accuracy of other methods across varioustasks but also speeds up the learning process. These results highlight thepotential of our approach to enhance both accuracy and learning efficiency inNeSy applications. |


| Item |Content|
| --- |---|
|idx| 2408.12590v1 |
|title| xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations |
|authors| Can QinCongying XiaKrithika RamakrishnanMichael RyooLifu TuYihao FengManli ShuHonglu ZhouAnas AwadallaJun WangSenthil PurushwalkamLe XueYingbo ZhouHuan WangSilvio SavareseJuan Carlos NieblesZeyuan ChenRan XuCaiming Xiong
|links| http://arxiv.org/abs/2408.12590v1 |
|updated| 2024-08-22 17:55:22 UTC |
|summary| We present xGen-VideoSyn-1 a text-to-video T2V generation model capable ofproducing realistic scenes from textual descriptions. Building on recentadvancements such as OpenAIs Sora we explore the latent diffusion modelLDM architecture and introduce a video variational autoencoder VidVAE.VidVAE compresses video data both spatially and temporally significantlyreducing the length of visual tokens and the computational demands associatedwith generating long-sequence videos. To further address the computationalcosts we propose a divide-and-merge strategy that maintains temporalconsistency across video segments. Our Diffusion Transformer DiT modelincorporates spatial and temporal self-attention layers enabling robustgeneralization across different timeframes and aspect ratios. We have devised adata processing pipeline from the very beginning and collected over 13Mhigh-quality video-text pairs. The pipeline includes multiple steps such asclipping text detection motion estimation aesthetics scoring and densecaptioning based on our in-house video-LLM model. Training the VidVAE and DiTmodels required approximately 40 and 642 H100 days respectively. Our modelsupports over 14-second 720p video generation in an end-to-end way anddemonstrates competitive performance against state-of-the-art T2V models. |


| Item |Content|
| --- |---|
|idx| 2408.12581v1 |
|title| Identifying the Best Arm in the Presence of Global Environment Shifts |
|authors| Phurinut SrisawadJuergen BrankeLong Tran-Thanh
|links| http://arxiv.org/abs/2408.12581v1 |
|updated| 2024-08-22 17:47:01 UTC |
|summary| This paper formulates a new Best-Arm Identification problem in thenon-stationary stochastic bandits setting where the means of all arms areshifted in the same way due to a global influence of the environment. The aimis to identify the unique best arm across environmental change given a fixedtotal budget. While this setting can be regarded as a special case ofAdversarial Bandits or Corrupted Bandits we demonstrate that existingsolutions tailored to those settings do not fully utilise the nature of thisglobal influence and thus do not work well in practice despite theirtheoretical guarantees. To overcome this issue in this paper we develop anovel selection policy that is consistent and robust in dealing with globalenvironmental shifts. We then propose an allocation policy LinLUCB whichexploits information about global shifts across all arms in each environment.Empirical tests depict a significant improvement in our policies against otherexisting methods. |


| Item |Content|
| --- |---|
|idx| 2408.12579v1 |
|title| RuleAlign: Making Large Language Models Better Physicians with Diagnostic Rule Alignment |
|authors| Xiaohan WangXiaoyan YangYuqi ZhuYue ShenJian WangPeng WeiLei LiangJinjie GuHuajun ChenNingyu Zhang
|links| http://arxiv.org/abs/2408.12579v1 |
|updated| 2024-08-22 17:44:40 UTC |
|summary| Large Language Models LLMs like GPT-4 MedPaLM-2 and Med-Gemini achieveperformance competitively with human experts across various medical benchmarks.However they still face challenges in making professional diagnoses akin tophysicians particularly in efficiently gathering patient information andreasoning the final diagnosis. To this end we introduce the RuleAlignframework designed to align LLMs with specific diagnostic rules. We develop amedical dialogue dataset comprising rule-based communications between patientsand physicians and design an alignment learning approach through preferencelearning. Experimental results demonstrate the effectiveness of the proposedapproach. We hope that our work can serve as an inspiration for exploring thepotential of LLMs as AI physicians. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2408.12594v1 |
|title| Non-Homophilic Graph Pre-Training and Prompt Learning |
|authors| Xingtong YuJie ZhangYuan FangRenhe Jiang
|links| http://arxiv.org/abs/2408.12594v1 |
|updated| 2024-08-22 17:57:31 UTC |
|summary| Graphs are ubiquitous for modeling complex relationships between objectsacross various fields. Graph neural networks GNNs have become a mainstreamtechnique for graph-based applications but their performance heavily relies onabundant labeled data. To reduce labeling requirement pre-training and promptlearning has become a popular alternative. However most existing promptmethods do not differentiate homophilic and heterophilic characteristics ofreal-world graphs. In particular many real-world graphs are non-homophilicnot strictly or uniformly homophilic with mixing homophilic and heterophilicpatterns exhibiting varying non-homophilic characteristics across graphs andnodes. In this paper we propose ProNoG a novel pre-training and promptlearning framework for such non-homophilic graphs. First we analyze existinggraph pre-training methods providing theoretical insights into the choice ofpre-training tasks. Second recognizing that each node exhibits uniquenon-homophilic characteristics we propose a conditional network tocharacterize the node-specific patterns in downstream tasks. Finally wethoroughly evaluate and analyze ProNoG through extensive experiments on tenpublic datasets. |


| Item |Content|
| --- |---|
|idx| 2408.12581v1 |
|title| Identifying the Best Arm in the Presence of Global Environment Shifts |
|authors| Phurinut SrisawadJuergen BrankeLong Tran-Thanh
|links| http://arxiv.org/abs/2408.12581v1 |
|updated| 2024-08-22 17:47:01 UTC |
|summary| This paper formulates a new Best-Arm Identification problem in thenon-stationary stochastic bandits setting where the means of all arms areshifted in the same way due to a global influence of the environment. The aimis to identify the unique best arm across environmental change given a fixedtotal budget. While this setting can be regarded as a special case ofAdversarial Bandits or Corrupted Bandits we demonstrate that existingsolutions tailored to those settings do not fully utilise the nature of thisglobal influence and thus do not work well in practice despite theirtheoretical guarantees. To overcome this issue in this paper we develop anovel selection policy that is consistent and robust in dealing with globalenvironmental shifts. We then propose an allocation policy LinLUCB whichexploits information about global shifts across all arms in each environment.Empirical tests depict a significant improvement in our policies against otherexisting methods. |


| Item |Content|
| --- |---|
|idx| 2408.12579v1 |
|title| RuleAlign: Making Large Language Models Better Physicians with Diagnostic Rule Alignment |
|authors| Xiaohan WangXiaoyan YangYuqi ZhuYue ShenJian WangPeng WeiLei LiangJinjie GuHuajun ChenNingyu Zhang
|links| http://arxiv.org/abs/2408.12579v1 |
|updated| 2024-08-22 17:44:40 UTC |
|summary| Large Language Models LLMs like GPT-4 MedPaLM-2 and Med-Gemini achieveperformance competitively with human experts across various medical benchmarks.However they still face challenges in making professional diagnoses akin tophysicians particularly in efficiently gathering patient information andreasoning the final diagnosis. To this end we introduce the RuleAlignframework designed to align LLMs with specific diagnostic rules. We develop amedical dialogue dataset comprising rule-based communications between patientsand physicians and design an alignment learning approach through preferencelearning. Experimental results demonstrate the effectiveness of the proposedapproach. We hope that our work can serve as an inspiration for exploring thepotential of LLMs as AI physicians. |


| Item |Content|
| --- |---|
|idx| 2408.12578v1 |
|title| A Percolation Model of Emergence: Analyzing Transformers Trained on a Formal Language |
|authors| Ekdeep Singh LubanaKyogo KawaguchiRobert P. DickHidenori Tanaka
|links| http://arxiv.org/abs/2408.12578v1 |
|updated| 2024-08-22 17:44:22 UTC |
|summary| Increase in data size or compute can lead to sudden learning of specificcapabilities by a neural network -- a phenomenon often called emergence.Beyond scientific understanding establishing the causal factors underlyingsuch emergent capabilities is crucial to enable risk regulation frameworks forAI. In this work we seek inspiration from study of emergent properties inother fields and propose a phenomenological definition for the concept in thecontext of neural networks. Our definition implicates the acquisition ofspecific structures underlying the data-generating process as a cause of suddenperformance growth for specific narrower tasks. We empirically investigatethis definition by proposing an experimental system grounded in acontext-sensitive formal language and find that Transformers trained to performtasks on top of strings from this language indeed exhibit emergentcapabilities. Specifically we show that once the languages underlying grammarand context-sensitivity inducing structures are learned by the modelperformance on narrower tasks suddenly begins to improve. We then analogize ournetworks learning dynamics with the process of percolation on a bipartitegraph establishing a formal phase transition model that predicts the shift inthe point of emergence observed in experiment when changing the data structure.Overall our experimental and theoretical frameworks yield a step towardsbetter defining characterizing and predicting emergence in neural networks. |


| Item |Content|
| --- |---|
|idx| 2408.12574v1 |
|title| MuMA-ToM: Multi-modal Multi-Agent Theory of Mind |
|authors| Haojun ShiSuyu YeXinyu FangChuanyang JinLayla IsikYen-Ling KuoTianmin Shu
|links| http://arxiv.org/abs/2408.12574v1 |
|updated| 2024-08-22 17:41:45 UTC |
|summary| Understanding peoples social interactions in complex real-world scenariosoften relies on intricate mental reasoning. To truly understand how and whypeople interact with one another we must infer the underlying mental statesthat give rise to the social interactions i.e. Theory of Mind reasoning inmulti-agent interactions. Additionally social interactions are oftenmulti-modal -- we can watch peoples actions hear their conversations and/orread about their past behaviors. For AI systems to successfully and safelyinteract with people in real-world environments they also need to understandpeoples mental states as well as their inferences about each others mentalstates based on multi-modal information about their interactions. For this weintroduce MuMA-ToM a Multi-modal Multi-Agent Theory of Mind benchmark.MuMA-ToM is the first multi-modal Theory of Mind benchmark that evaluatesmental reasoning in embodied multi-agent interactions. In MuMA-ToM we providevideo and text descriptions of peoples multi-modal behavior in realistichousehold environments. Based on the context we then ask questions aboutpeoples goals beliefs and beliefs about others goals. We validated MuMA-ToMin a human experiment and provided a human baseline. We also proposed a novelmulti-modal multi-agent ToM model LIMP Language model-based InverseMulti-agent Planning. Our experimental results show that LIMP significantlyoutperforms state-of-the-art methods including large multi-modal models e.g.GPT-4o Gemini-1.5 Pro and a recent multi-modal ToM model BIP-ALM. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2408.12601v1 |
|title| DreamCinema: Cinematic Transfer with Free Camera and 3D Character |
|authors| Weiliang ChenFangfu LiuDiankun WuHaowen SunHaixu SongYueqi Duan
|links| http://arxiv.org/abs/2408.12601v1 |
|updated| 2024-08-22 17:59:44 UTC |
|summary| We are living in a flourishing era of digital media where everyone has thepotential to become a personal filmmaker. Current research on cinematictransfer empowers filmmakers to reproduce and manipulate the visual elementse.g. cinematography and character behaviors from classic shots. Howevercharacters in the reimagined films still rely on manual crafting whichinvolves significant technical complexity and high costs making itunattainable for ordinary users. Furthermore their estimated cinematographylacks smoothness due to inadequate capturing of inter-frame motion and modelingof physical trajectories. Fortunately the remarkable success of 2D and 3D AIGChas opened up the possibility of efficiently generating characters tailored tousers needs diversifying cinematography. In this paper we proposeDreamCinema a novel cinematic transfer framework that pioneers generative AIinto the film production paradigm aiming at facilitating user-friendly filmcreation. Specifically we first extract cinematic elements i.e. human andcamera pose and optimize the camera trajectory. Then we apply a charactergenerator to efficiently create 3D high-quality characters with a humanstructure prior. Finally we develop a structure-guided motion transferstrategy to incorporate generated characters into film creation and transfer itvia 3D graphics engines smoothly. Extensive experiments demonstrate theeffectiveness of our method for creating high-quality films with free cameraand 3D characters. |


| Item |Content|
| --- |---|
|idx| 2408.12598v1 |
|title| ND-SDF: Learning Normal Deflection Fields for High-Fidelity Indoor Reconstruction |
|authors| Ziyu TangWeicai YeYifan WangDi HuangHujun BaoTong HeGuofeng Zhang
|links| http://arxiv.org/abs/2408.12598v1 |
|updated| 2024-08-22 17:59:01 UTC |
|summary| Neural implicit reconstruction via volume rendering has demonstrated itseffectiveness in recovering dense 3D surfaces. However it is non-trivial tosimultaneously recover meticulous geometry and preserve smoothness acrossregions with differing characteristics. To address this issue previous methodstypically employ geometric priors which are often constrained by theperformance of the prior models. In this paper we propose ND-SDF which learnsa Normal Ddeflection field to represent the angular deviation between the scenenormal and the prior normal. Unlike previous methods that uniformly applygeometric priors on all samples introducing significant bias in accuracy ourproposed normal deflection field dynamically learns and adapts the utilizationof samples based on their specific characteristics thereby improving both theaccuracy and effectiveness of the model. Our method not only obtains smoothweakly textured regions such as walls and floors but also preserves thegeometric details of complex structures. In addition we introduce a novel raysampling strategy based on the deflection angle to facilitate the unbiasedrendering process which significantly improves the quality and accuracy ofintricate surfaces especially on thin structures. Consistent improvements onvarious challenging datasets demonstrate the superiority of our method. |


| Item |Content|
| --- |---|
|idx| 2408.12593v1 |
|title| Automating Deformable Gasket Assembly |
|authors| Simeon AdebolaTara SadjadpourKarim El-RefaiWill PanitchZehan MaRoy LinTianshuang QiuShreya GantiCharlotte LeJaimyn DrakeKen Goldberg
|links| http://arxiv.org/abs/2408.12593v1 |
|updated| 2024-08-22 17:57:03 UTC |
|summary| In Gasket Assembly a deformable gasket must be aligned and pressed into anarrow channel. This task is common for sealing surfaces in the manufacturingof automobiles appliances electronics and other products. Gasket Assembly isa long-horizon high-precision task and the gasket must align with the channeland be fully pressed in to achieve a secure fit. To compare approaches wepresent 4 methods for Gasket Assembly: one policy from deep imitation learningand three procedural algorithms. We evaluate these methods with 100 physicaltrials. Results suggest that the Binary algorithm succeeds in 10/10 on thestraight channel whereas the learned policy based on 250 human teleoperateddemonstrations succeeds in 8/10 trials and is significantly slower. Code CADmodels videos and data can be found athttps://berkeleyautomation.github.io/robot-gasket/ |


| Item |Content|
| --- |---|
|idx| 2408.12590v1 |
|title| xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations |
|authors| Can QinCongying XiaKrithika RamakrishnanMichael RyooLifu TuYihao FengManli ShuHonglu ZhouAnas AwadallaJun WangSenthil PurushwalkamLe XueYingbo ZhouHuan WangSilvio SavareseJuan Carlos NieblesZeyuan ChenRan XuCaiming Xiong
|links| http://arxiv.org/abs/2408.12590v1 |
|updated| 2024-08-22 17:55:22 UTC |
|summary| We present xGen-VideoSyn-1 a text-to-video T2V generation model capable ofproducing realistic scenes from textual descriptions. Building on recentadvancements such as OpenAIs Sora we explore the latent diffusion modelLDM architecture and introduce a video variational autoencoder VidVAE.VidVAE compresses video data both spatially and temporally significantlyreducing the length of visual tokens and the computational demands associatedwith generating long-sequence videos. To further address the computationalcosts we propose a divide-and-merge strategy that maintains temporalconsistency across video segments. Our Diffusion Transformer DiT modelincorporates spatial and temporal self-attention layers enabling robustgeneralization across different timeframes and aspect ratios. We have devised adata processing pipeline from the very beginning and collected over 13Mhigh-quality video-text pairs. The pipeline includes multiple steps such asclipping text detection motion estimation aesthetics scoring and densecaptioning based on our in-house video-LLM model. Training the VidVAE and DiTmodels required approximately 40 and 642 H100 days respectively. Our modelsupports over 14-second 720p video generation in an end-to-end way anddemonstrates competitive performance against state-of-the-art T2V models. |


| Item |Content|
| --- |---|
|idx| 2408.12588v1 |
|title| Real-Time Video Generation with Pyramid Attention Broadcast |
|authors| Xuanlei ZhaoXiaolong JinKai WangYang You
|links| http://arxiv.org/abs/2408.12588v1 |
|updated| 2024-08-22 17:54:21 UTC |
|summary| We present Pyramid Attention Broadcast PAB a real-time high quality andtraining-free approach for DiT-based video generation. Our method is founded onthe observation that attention difference in the diffusion process exhibits aU-shaped pattern indicating significant redundancy. We mitigate this bybroadcasting attention outputs to subsequent steps in a pyramid style. Itapplies different broadcast strategies to each attention based on theirvariance for best efficiency. We further introduce broadcast sequence parallelfor more efficient distributed inference. PAB demonstrates superior resultsacross three models compared to baselines achieving real-time generation forup to 720p videos. We anticipate that our simple yet effective method willserve as a robust baseline and facilitate future research and application forvideo generation. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2408.12564v1 |
|title| Factor Adjusted Spectral Clustering for Mixture Models |
|authors| Shange TangSoham JanaJianqing Fan
|links| http://arxiv.org/abs/2408.12564v1 |
|updated| 2024-08-22 17:31:21 UTC |
|summary| This paper studies a factor modeling-based approach for clusteringhigh-dimensional data generated from a mixture of strongly correlatedvariables. Statistical modeling with correlated structures pervades modernapplications in economics finance genomics wireless sensing etc. withfactor modeling being one of the popular techniques for explaining the commondependence. Standard techniques for clustering high-dimensional data e.g.naive spectral clustering often fail to yield insightful results as theirperformances heavily depend on the mixture components having a weaklycorrelated structure. To address the clustering problem in the presence of alatent factor model we propose the Factor Adjusted Spectral Clustering FASCalgorithm which uses an additional data denoising step via eliminating thefactor component to cope with the data dependency. We prove this methodachieves an exponentially low mislabeling rate with respect to the signal tonoise ratio under a general set of assumptions. Our assumption bridges manyclassical factor models in the literature such as the pervasive factor modelthe weak factor model and the sparse factor model. The FASC algorithm is alsocomputationally efficient requiring only near-linear sample complexity withrespect to the data dimension. We also show the applicability of the FASCalgorithm with real data experiments and numerical studies and establish thatFASC provides significant results in many cases where traditional spectralclustering fails. |


| Item |Content|
| --- |---|
|idx| 2408.12353v1 |
|title| Distributed quasi-Newton robust estimation under differential privacy |
|authors| Chuhan WangLixing ZhuXuehu Zhu
|links| http://arxiv.org/abs/2408.12353v1 |
|updated| 2024-08-22 12:51:28 UTC |
|summary| For distributed computing with Byzantine machines under Privacy ProtectionPP constraints this paper develops a robust PP distributed quasi-Newtonestimation which only requires the node machines to transmit five vectors tothe central processor with high asymptotic relative efficiency. Compared withthe gradient descent strategy which requires more rounds of transmission andthe Newton iteration strategy which requires the entire Hessian matrix to betransmitted the novel quasi-Newton iteration has advantages in reducingprivacy budgeting and transmission cost. Moreover our PP algorithm does notdepend on the boundedness of gradients and second-order derivatives. Whengradients and second-order derivatives follow sub-exponential distributions weoffer a mechanism that can ensure PP with a sufficiently high probability.Furthermore this novel estimator can achieve the optimal convergence rate andthe asymptotic normality. The numerical studies on synthetic and real data setsevaluate the performance of the proposed algorithm. |


| Item |Content|
| --- |---|
|idx| 2408.12332v1 |
|title| Simplifying Random Forests' Probabilistic Forecasts |
|authors| Nils KosterFabian Krüger
|links| http://arxiv.org/abs/2408.12332v1 |
|updated| 2024-08-22 12:20:17 UTC |
|summary| Since their introduction by Breiman Random Forests RFs have proven to beuseful for both classification and regression tasks. The RF prediction of apreviously unseen observation can be represented as a weighted sum of alltraining sample observations. This nearest-neighbor-type representation isuseful among other things for constructing forecast distributionsMeinshausen 2006. In this paper we consider simplifying RF-based forecastdistributions by sparsifying them. That is we focus on a small subset ofnearest neighbors while setting the remaining weights to zero. Thissparsification step greatly improves the interpretability of RF predictions. Itcan be applied to any forecasting task without re-training existing RF models.In empirical experiments we document that the simplified predictions can besimilar to or exceed the original ones in terms of forecasting performance. Weexplore the statistical sources of this finding via a stylized analytical modelof RFs. The model suggests that simplification is particularly promising if theunknown true forecast distribution contains many small weights that areestimated imprecisely. |


| Item |Content|
| --- |---|
|idx| 2408.12319v1 |
|title| Neural-ANOVA: Model Decomposition for Interpretable Machine Learning |
|authors| Steffen LimmerSteffen UdluftClemens Otte
|links| http://arxiv.org/abs/2408.12319v1 |
|updated| 2024-08-22 11:55:43 UTC |
|summary| The analysis of variance ANOVA decomposition offers a systematic method tounderstand the interaction effects that contribute to a specific decisionoutput. In this paper we introduce Neural-ANOVA an approach to decomposeneural networks into glassbox models using the ANOVA decomposition. Ourapproach formulates a learning problem which enables rapid and closed-formevaluation of integrals over subspaces that appear in the calculation of theANOVA decomposition. Finally we conduct numerical experiments to illustratethe advantages of enhanced interpretability and model validation by adecomposition of the learned interaction effects. |


| Item |Content|
| --- |---|
|idx| 2408.12288v1 |
|title| Demystifying Functional Random Forests: Novel Explainability Tools for Model Transparency in High-Dimensional Spaces |
|authors| Fabrizio MaturoAnnamaria Porreca
|links| http://arxiv.org/abs/2408.12288v1 |
|updated| 2024-08-22 10:52:32 UTC |
|summary| The advent of big data has raised significant challenges in analysinghigh-dimensional datasets across various domains such as medicine ecology andeconomics. Functional Data Analysis FDA has proven to be a robust frameworkfor addressing these challenges enabling the transformation ofhigh-dimensional data into functional forms that capture intricate temporal andspatial patterns. However despite advancements in functional classificationmethods and very high performance demonstrated by combining FDA and ensemblemethods a critical gap persists in the literature concerning the transparencyand interpretability of black-box models e.g. Functional Random Forests FRF.In response to this need this paper introduces a novel suite of explainabilitytools to illuminate the inner mechanisms of FRF. We propose using FunctionalPartial Dependence Plots FPDPs Functional Principal Component FPCProbability Heatmaps various model-specific and model-agnostic FPCsimportance metrics and the FPC Internal-External Importance and ExplainedVariance Bubble Plot. These tools collectively enhance the transparency of FRFmodels by providing a detailed analysis of how individual FPCs contribute tomodel predictions. By applying these methods to an ECG dataset we demonstratethe effectiveness of these tools in revealing critical patterns and improvingthe explainability of FRF. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2408.12579v1 |
|title| RuleAlign: Making Large Language Models Better Physicians with Diagnostic Rule Alignment |
|authors| Xiaohan WangXiaoyan YangYuqi ZhuYue ShenJian WangPeng WeiLei LiangJinjie GuHuajun ChenNingyu Zhang
|links| http://arxiv.org/abs/2408.12579v1 |
|updated| 2024-08-22 17:44:40 UTC |
|summary| Large Language Models LLMs like GPT-4 MedPaLM-2 and Med-Gemini achieveperformance competitively with human experts across various medical benchmarks.However they still face challenges in making professional diagnoses akin tophysicians particularly in efficiently gathering patient information andreasoning the final diagnosis. To this end we introduce the RuleAlignframework designed to align LLMs with specific diagnostic rules. We develop amedical dialogue dataset comprising rule-based communications between patientsand physicians and design an alignment learning approach through preferencelearning. Experimental results demonstrate the effectiveness of the proposedapproach. We hope that our work can serve as an inspiration for exploring thepotential of LLMs as AI physicians. |


| Item |Content|
| --- |---|
|idx| 2408.12500v1 |
|title| WhisperMask: A Noise Suppressive Mask-Type Microphone for Whisper Speech |
|authors| Hirotaka HirakiShusuke KanazawaTakahiro MiuraManabu YoshidaMasaaki MochimaruJun Rekimoto
|links| http://dx.doi.org/10.1145/3652920.3652925 |
|updated| 2024-08-22 15:51:07 UTC |
|summary| Whispering is a common privacy-preserving technique in voice-basedinteractions but its effectiveness is limited in noisy environments. Inconventional hardware- and software-based noise reduction approaches isolatingwhispered speech from ambient noise and other speech sounds remains achallenge. We thus propose WhisperMask a mask-type microphone featuring alarge diaphragm with low sensitivity making the wearers voice significantlylouder than the background noise. We evaluated WhisperMask using three keymetrics: signal-to-noise ratio quality of recorded voices and speechrecognition rate. Across all metrics WhisperMask consistently outperformedtraditional noise-suppressing microphones and software-based solutions.Notably WhisperMask showed a 30 higher recognition accuracy for whisperedspeech recorded in an environment with 80 dB background noise compared with thepin microphone and earbuds. Furthermore while a denoiser decreased thewhispered speech recognition rate of these two microphones by approximately 20at 30-60 dB noise WhisperMask maintained a high performance even withoutdenoising surpassing the other microphones performances by a significantmargin.WhisperMasks design renders the wearers voice as the dominant inputand effectively suppresses background noise without relying on signalprocessing. This device allows for reliable voice interactions such as phonecalls and voice commands in a wide range of noisy real-world scenarios whilepreserving user privacy. |


| Item |Content|
| --- |---|
|idx| 2408.12463v1 |
|title| Smartphone-based Eye Tracking System using Edge Intelligence and Model Optimisation |
|authors| Nishan GunawardenaGough Yumu LuiJeewani Anupama GinigeBahman Javadi
|links| http://arxiv.org/abs/2408.12463v1 |
|updated| 2024-08-22 15:04:59 UTC |
|summary| A significant limitation of current smartphone-based eye-tracking algorithmsis their low accuracy when applied to video-type visual stimuli as they aretypically trained on static images. Also the increasing demand for real-timeinteractive applications like games VR and AR on smartphones requiresovercoming the limitations posed by resource constraints such as limitedcomputational power battery life and network bandwidth. Therefore wedeveloped two new smartphone eye-tracking techniques for video-type visuals bycombining Convolutional Neural Networks CNN with two different RecurrentNeural Networks RNN namely Long Short Term Memory LSTM and Gated RecurrentUnit GRU. Our CNNLSTM and CNNGRU models achieved an average Root MeanSquare Error of 0.955cm and 1.091cm respectively. To address the computationalconstraints of smartphones we developed an edge intelligence architecture toenhance the performance of smartphone-based eye tracking. We applied variousoptimisation methods like quantisation and pruning to deep learning models forbetter energy CPU and memory usage on edge devices focusing on real-timeprocessing. Using model quantisation the model inference time in the CNNLSTMand CNNGRU models was reduced by 21.72 and 19.50 respectively on edgedevices. |


| Item |Content|
| --- |---|
|idx| 2408.12428v1 |
|title| VR4UrbanDev: An Immersive Virtual Reality Experience for Energy Data Visualization |
|authors| Saeed SafikhaniGeorg Arbesser-RastburgAnna SchreuerJürgen Suschek-BergerHermann EdtmayerJohanna Pirker
|links| http://arxiv.org/abs/2408.12428v1 |
|updated| 2024-08-22 14:22:05 UTC |
|summary| In this demonstration paper we present our interactive virtual reality VRexperience which has been designed to facilitate interaction withenergy-related information. This experience consists of two main modes: theworld in miniature for large-scale and first-person for real-world scalevisualizations. Additionally we presented our approach to potential targetgroups in interviews. The results of these interviews can help developers forfuture implementation considering the requirements of each group. |


| Item |Content|
| --- |---|
|idx| 2408.12365v1 |
|title| Enhancing Uncertainty Communication in Time Series Predictions: Insights and Recommendations |
|authors| Apoorva KaragappaPawandeep Kaur BetzJonas GilgMoritz ZeumerAndreas GerndtBernhard Preim
|links| http://arxiv.org/abs/2408.12365v1 |
|updated| 2024-08-22 13:03:55 UTC |
|summary| As the world increasingly relies on mathematical models for forecasts indifferent areas effective communication of uncertainty in time seriespredictions is important for informed decision making. This study explores howusers estimate probabilistic uncertainty in time series predictions underdifferent variants of line charts depicting uncertainty. It examines the roleof individual characteristics and the influence of user-reported metrics onuncertainty estimations. By addressing these aspects this paper aims toenhance the understanding of uncertainty visualization and for improvingcommunication in time series forecast visualizations and the design ofprediction data dashboards.As the world increasingly relies on mathematicalmodels for forecasts in different areas effective communication of uncertaintyin time series predictions is important for informed decision making. Thisstudy explores how users estimate probabilistic uncertainty in time seriespredictions under different variants of line charts depicting uncertainty. Itexamines the role of individual characteristics and the influence ofuser-reported metrics on uncertainty estimations. By addressing these aspectsthis paper aims to enhance the understanding of uncertainty visualization andfor improving communication in time series forecast visualizations and thedesign of prediction data dashboards. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2408.12496v1 |
|title| MEDCO: Medical Education Copilots Based on A Multi-Agent Framework |
|authors| Hao WeiJianing QiuHaibao YuWu Yuan
|links| http://arxiv.org/abs/2408.12496v1 |
|updated| 2024-08-22 15:41:58 UTC |
|summary| Large language models LLMs have had a significant impact on diverseresearch domains including medicine and healthcare. However the potential ofLLMs as copilots in medical education remains underexplored. CurrentAI-assisted educational tools are limited by their solitary learning approachand inability to simulate the multi-disciplinary and interactive nature ofactual medical training. To address these limitations we propose MEDCOMedical EDucation COpilots a novel multi-agent-based copilot systemspecially developed to emulate real-world medical training environments. MEDCOincorporates three primary agents: an agentic patient an expert doctor and aradiologist facilitating a multi-modal and interactive learning environment.Our framework emphasizes the learning of proficient question-asking skillsmulti-disciplinary collaboration and peer discussions between students. Ourexperiments show that simulated virtual students who underwent training withMEDCO not only achieved substantial performance enhancements comparable tothose of advanced models but also demonstrated human-like learning behaviorsand improvements coupled with an increase in the number of learning samples.This work contributes to medical education by introducing a copilot thatimplements an interactive and collaborative learning approach. It also providesvaluable insights into the effectiveness of AI-integrated training paradigms. |


| Item |Content|
| --- |---|
|idx| 2408.12112v1 |
|title| Balancing Act: Prioritization Strategies for LLM-Designed Restless Bandit Rewards |
|authors| Shresth VermaNiclas BoehmerLingkai KongMilind Tambe
|links| http://arxiv.org/abs/2408.12112v1 |
|updated| 2024-08-22 03:54:08 UTC |
|summary| LLMs are increasingly used to design reward functions based on humanpreferences in Reinforcement Learning RL. We focus on LLM-designed rewardsfor Restless Multi-Armed Bandits a framework for allocating limited resourcesamong agents. In applications such as public health this approach empowersgrassroots health workers to tailor automated allocation decisions to communityneeds. In the presence of multiple agents altering the reward function basedon human preferences can impact subpopulations very differently leading tocomplex tradeoffs and a multi-objective resource allocation problem. We are thefirst to present a principled method termed Social Choice Language Model fordealing with these tradeoffs for LLM-designed rewards for multiagent plannersin general and restless bandits in particular. The novel part of our model is atransparent and configurable selection component called an adjudicatorexternal to the LLM that controls complex tradeoffs via a user-selected socialwelfare function. Our experiments demonstrate that our model reliably selectsmore effective aligned and balanced reward functions compared to purelyLLM-based approaches. |


| Item |Content|
| --- |---|
|idx| 2408.12038v1 |
|title| Empirical Equilibria in Agent-based Economic systems with Learning agents |
|authors| Kshama DwarakanathSvitlana VyetrenkoTucker Balch
|links| http://arxiv.org/abs/2408.12038v1 |
|updated| 2024-08-21 23:47:46 UTC |
|summary| We present an agent-based simulator for economic systems with heterogeneoushouseholds firms central bank and government agents. These agents interactto define production consumption and monetary flow. Each agent type hasdistinct objectives such as households seeking utility from consumption andthe central bank targeting inflation and production. We define this multi-agenteconomic system using an OpenAI Gym-style environment enabling agents tooptimize their objectives through reinforcement learning. Standard multi-agentreinforcement learning MARL schemes like independent learning enable agentsto learn concurrently but do not address whether the resulting strategies areat equilibrium. This study integrates the Policy Space Response Oracle PSROalgorithm which has shown superior performance over independent MARL in gameswith homogeneous agents with economic agent-based modeling. We use PSRO todevelop agent policies approximating Nash equilibria of the empirical economicgame thereby linking to economic equilibria. Our results demonstrate that PSROstrategies achieve lower regret values than independent MARL strategies in oureconomic system with four agent types. This work aims to bridge artificialintelligence economics and empirical game theory towards future research. |


| Item |Content|
| --- |---|
|idx| 2408.11772v1 |
|title| VIRIS: Simulating indoor airborne transmission combining architectural design and people movement |
|authors| Yidan XueWassim JabiThomas E. WoolleyKaterina Kaouri
|links| http://arxiv.org/abs/2408.11772v1 |
|updated| 2024-08-21 16:54:22 UTC |
|summary| A Viral Infection Risk Indoor Simulator VIRIS has been developed to quicklyassess and compare mitigations for airborne disease spread. This agent-basedsimulator combines people movement in an indoor space viral transmissionmodelling and detailed architectural design and it is powered by topologicpyan open-source Python library. VIRIS generates very fast predictions of theviral concentration and the spatiotemporal infection risk for individuals asthey move through a given space. The simulator is validated with data from acourtroom superspreader event. A sensitivity study for unknown parameter valuesis also performed. We compare several non-pharmaceutical interventions NPIsissued in UK government guidance for two indoor settings: a care home and asupermarket. Additionally we have developed the user-friendly VIRIS web appthat allows quick exploration of diverse scenarios of interest andvisualisation allowing policymakers architects and space managers to easilydesign or assess infection risk in an indoor space. |


| Item |Content|
| --- |---|
|idx| 2408.11751v1 |
|title| Bayesian Optimization Framework for Efficient Fleet Design in Autonomous Multi-Robot Exploration |
|authors| David Molina ConchaJiping LiHaoran YinKyeonghyeon ParkHyun-Rok LeeTaesik LeeDhruv SirohiChi-Guhn Lee
|links| http://arxiv.org/abs/2408.11751v1 |
|updated| 2024-08-21 16:22:51 UTC |
|summary| This study addresses the challenge of fleet design optimization in thecontext of heterogeneous multi-robot fleets aiming to obtain feasible designsthat balance performance and costs. In the domain of autonomous multi-robotexploration reinforcement learning agents play a central role offeringadaptability to complex terrains and facilitating collaboration among robots.However modifying the fleet composition results in changes in the learnedbehavior and training multi-robot systems using multi-agent reinforcementlearning is expensive. Therefore an exhaustive evaluation of each potentialfleet design is infeasible. To tackle these hurdles we introduce BayesianOptimization for Fleet Design BOFD a framework leveraging multi-objectiveBayesian Optimization to explore fleets on the Pareto front of performance andcost while accounting for uncertainty in the design space. Moreover weestablish a sub-linear bound for cumulative regret supporting BOFDsrobustness and efficacy. Extensive benchmark experiments in synthetic andsimulated environments demonstrate the superiority of our framework overstate-of-the-art methods achieving efficient fleet designs with minimal fleetevaluations. |


