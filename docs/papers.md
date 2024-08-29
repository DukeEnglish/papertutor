# cs.CL 

| Item |Content|
| --- |---|
|idx| 2408.15992v1 |
|title| CoGen: Learning from Feedback with Coupled Comprehension and Generation |
|authors| Mustafa Omer GulYoav Artzi
|links| http://arxiv.org/abs/2408.15992v1 |
|updated| 2024-08-28 17:58:39 UTC |
|summary| Systems with both language comprehension and generation capabilities canbenefit from the tight connection between the two. This work studies couplingcomprehension and generation with focus on continually learning frominteraction with users. We propose techniques to tightly integrate the twocapabilities for both learning and inference. We situate our studies intwo-player reference games and deploy various models for thousands ofinteractions with human users while learning from interaction feedbacksignals. We show dramatic improvements in performance over time withcomprehension-generation coupling leading to performance improvements up to 26in absolute terms and up to 17 higher accuracies compared to a non-coupledsystem. Our analysis also shows coupling has substantial qualitative impact onthe systems language making it significantly more human-like. |


| Item |Content|
| --- |---|
|idx| 2408.15971v1 |
|title| BattleAgentBench: A Benchmark for Evaluating Cooperation and Competition Capabilities of Language Models in Multi-Agent Systems |
|authors| Wei WangDan ZhangTao FengBoyan WangJie Tang
|links| http://arxiv.org/abs/2408.15971v1 |
|updated| 2024-08-28 17:43:55 UTC |
|summary| Large Language Models LLMs are becoming increasingly powerful and capableof handling complex tasks e.g. building single agents and multi-agentsystems. Compared to single agents multi-agent systems have higherrequirements for the collaboration capabilities of language models. Manybenchmarks are proposed to evaluate their collaborative abilities. Howeverthese benchmarks lack fine-grained evaluations of LLM collaborativecapabilities. Additionally multi-agent collaborative and competitive scenariosare ignored in existing works. To address these two problems we propose abenchmark called BattleAgentBench which defines seven sub-stages of threevarying difficulty levels and conducts a fine-grained evaluation of languagemodels in terms of single-agent scenario navigation capabilities paired-agenttask execution abilities and multi-agent collaboration and competitioncapabilities. We conducted extensive evaluations on leading four closed-sourceand seven open-source models. Experimental results indicate that API-basedmodels perform excellently on simple tasks but open-source small modelsstruggle with simple tasks. Regarding difficult tasks that requirecollaborative and competitive abilities although API-based models havedemonstrated some collaborative capabilities there is still enormous room forimprovement. |


| Item |Content|
| --- |---|
|idx| 2408.15966v1 |
|title| More Text, Less Point: Towards 3D Data-Efficient Point-Language Understanding |
|authors| Yuan TangXu HanXianzhi LiQiao YuJinfeng XuYixue HaoLong HuMin Chen
|links| http://arxiv.org/abs/2408.15966v1 |
|updated| 2024-08-28 17:38:44 UTC |
|summary| Enabling Large Language Models LLMs to comprehend the 3D physical worldremains a significant challenge. Due to the lack of large-scale 3D-text pairdatasets the success of LLMs has yet to be replicated in 3D understanding. Inthis paper we rethink this issue and propose a new task: 3D Data-EfficientPoint-Language Understanding. The goal is to enable LLMs to achieve robust 3Dobject understanding with minimal 3D point cloud and text data pairs. Toaddress this task we introduce GreenPLM which leverages more text data tocompensate for the lack of 3D data. First inspired by using CLIP to alignimages and text we utilize a pre-trained point cloud-text encoder to map the3D point cloud space to the text space. This mapping leaves us to seamlesslyconnect the text space with LLMs. Once the point-text-LLM connection isestablished we further enhance text-LLM alignment by expanding theintermediate text space thereby reducing the reliance on 3D point cloud data.Specifically we generate 6M free-text descriptions of 3D objects and design athree-stage training strategy to help LLMs better explore the intrinsicconnections between different modalities. To achieve efficient modalityalignment we design a zero-parameter cross-attention module for token pooling.Extensive experimental results show that GreenPLM requires only 12 of the 3Dtraining data used by existing state-of-the-art models to achieve superior 3Dunderstanding. Remarkably GreenPLM also achieves competitive performance usingtext-only data. The code and weights are available at:https://github.com/TangYuan96/GreenPLM. |


| Item |Content|
| --- |---|
|idx| 2408.15915v1 |
|title| Leveraging Open Knowledge for Advancing Task Expertise in Large Language Models |
|authors| Yuncheng YangYulei QinTong WuZihan XuGang LiPengcheng GuoHang ShaoYucheng ShiKe LiXing SunJie YangYun Gu
|links| http://arxiv.org/abs/2408.15915v1 |
|updated| 2024-08-28 16:28:07 UTC |
|summary| The cultivation of expertise for large language models LLMs to solve tasksof specific areas often requires special-purpose tuning with calibratedbehaviors on the expected stable outputs. To avoid huge cost brought by manualpreparation of instruction datasets and training resources up to hundreds ofhours the exploitation of open knowledge including a wealth of low rankadaptation LoRA models and instruction datasets serves as a good startingpoint. However existing methods on model and data selection focus on theperformance of general-purpose capabilities while neglecting the knowledge gapexposed in domain-specific deployment. In the present study we propose tobridge such gap by introducing few human-annotated samples i.e. K-shot foradvancing task expertise of LLMs with open knowledge. Specifically we developan efficient and scalable pipeline to cost-efficiently produce task expertswhere K-shot data intervene in selecting the most promising expert candidatesand the task-relevant instructions. A mixture-of-expert MoE system is builtto make the best use of individual-yet-complementary knowledge between multipleexperts. We unveil the two keys to the success of a MoE system 1 the abidanceby K-shot and 2 the insistence on diversity. For the former we ensure thatmodels that truly possess problem-solving abilities on K-shot are selectedrather than those blind guessers. Besides during data selection instructionsthat share task-relevant contexts with K-shot are prioritized. For the latterwe highlight the diversity of constituting experts and that of the fine-tuninginstructions throughout the model and data selection process. Extensiveexperimental results confirm the superiority of our approach over existingmethods on utilization of open knowledge across various tasks. Codes and modelswill be released later. |


| Item |Content|
| --- |---|
|idx| 2408.15903v1 |
|title| LLM-Based Multi-Hop Question Answering with Knowledge Graph Integration in Evolving Environments |
|authors| Ruirui ChenWeifeng JiangChengwei QinIshaan Singh RawalCheston TanDongkyu ChoiBo XiongBo Ai
|links| http://arxiv.org/abs/2408.15903v1 |
|updated| 2024-08-28 16:15:45 UTC |
|summary| The rapid obsolescence of information in Large Language Models LLMs hasdriven the development of various techniques to incorporate new facts. Howeverexisting methods for knowledge editing still face difficulties with multi-hopquestions that require accurate fact identification and sequential logicalreasoning particularly among numerous fact updates. To tackle thesechallenges this paper introduces Graph Memory-based Editing for Large LanguageModels GMeLLo a straitforward and effective method that merges the explicitknowledge representation of Knowledge Graphs KGs with the linguisticflexibility of LLMs. Beyond merely leveraging LLMs for question answeringGMeLLo employs these models to convert free-form language into structuredqueries and fact triples facilitating seamless interaction with KGs for rapidupdates and precise multi-hop reasoning. Our results show that GMeLLosignificantly surpasses current state-of-the-art knowledge editing methods inthe multi-hop question answering benchmark MQuAKE especially in scenarioswith extensive knowledge edits. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2408.15998v1 |
|title| Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders |
|authors| Min ShiFuxiao LiuShihao WangShijia LiaoSubhashree RadhakrishnanDe-An HuangHongxu YinKaran SapraYaser YacoobHumphrey ShiBryan CatanzaroAndrew TaoJan KautzZhiding YuGuilin Liu
|links| http://arxiv.org/abs/2408.15998v1 |
|updated| 2024-08-28 17:59:31 UTC |
|summary| The ability to accurately interpret complex visual information is a crucialtopic of multimodal large language models MLLMs. Recent work indicates thatenhanced visual perception significantly reduces hallucinations and improvesperformance on resolution-sensitive tasks such as optical characterrecognition and document analysis. A number of recent MLLMs achieve this goalusing a mixture of vision encoders. Despite their success there is a lack ofsystematic comparisons and detailed ablation studies addressing criticalaspects such as expert selection and the integration of multiple visionexperts. This study provides an extensive exploration of the design space forMLLMs using a mixture of vision encoders and resolutions. Our findings revealseveral underlying principles common to various existing strategies leading toa streamlined yet effective design approach. We discover that simplyconcatenating visual tokens from a set of complementary vision encoders is aseffective as more complex mixing architectures or strategies. We additionallyintroduce Pre-Alignment to bridge the gap between vision-focused encoders andlanguage tokens enhancing model coherence. The resulting family of MLLMsEagle surpasses other leading open-source models on major MLLM benchmarks.Models and code: https://github.com/NVlabs/Eagle |


| Item |Content|
| --- |---|
|idx| 2408.15997v1 |
|title| Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need |
|authors| Sijia PengYun XiongYangyong ZhuZhiqiang Shen
|links| http://arxiv.org/abs/2408.15997v1 |
|updated| 2024-08-28 17:59:27 UTC |
|summary| Time series forecasting requires balancing short-term and long-termdependencies for accurate predictions. Existing methods mainly focus onlong-term dependency modeling neglecting the complexities of short-termdynamics which may hinder performance. Transformers are superior in modelinglong-term dependencies but are criticized for their quadratic computationalcost. Mamba provides a near-linear alternative but is reported less effectivein time series longterm forecasting due to potential information loss. Currentarchitectures fall short in offering both high efficiency and strongperformance for long-term dependency modeling. To address these challenges weintroduce Mixture of Universals MoU a versatile model to capture bothshort-term and long-term dependencies for enhancing performance in time seriesforecasting. MoU is composed of two novel designs: Mixture of FeatureExtractors MoF an adaptive method designed to improve time series patchrepresentations for short-term dependency and Mixture of Architectures MoAwhich hierarchically integrates Mamba FeedForward Convolution andSelf-Attention architectures in a specialized order to model long-termdependency from a hybrid perspective. The proposed approach achievesstate-of-the-art performance while maintaining relatively low computationalcosts. Extensive experiments on seven real-world datasets demonstrate thesuperiority of MoU. Code is available at https://github.com/lunaaa95/mou/. |


| Item |Content|
| --- |---|
|idx| 2408.15996v1 |
|title| Spatio-Temporal Context Prompting for Zero-Shot Action Detection |
|authors| Wei-Jhe HuangMin-Hung ChenShang-Hong Lai
|links| http://arxiv.org/abs/2408.15996v1 |
|updated| 2024-08-28 17:59:05 UTC |
|summary| Spatio-temporal action detection encompasses the tasks of localizing andclassifying individual actions within a video. Recent works aim to enhance thisprocess by incorporating interaction modeling which captures the relationshipbetween people and their surrounding context. However these approaches haveprimarily focused on fully-supervised learning and the current limitation liesin the lack of generalization capability to recognize unseen action categories.In this paper we aim to adapt the pretrained image-language models to detectunseen actions. To this end we propose a method which can effectively leveragethe rich knowledge of visual-language models to perform Person-ContextInteraction. Meanwhile our Context Prompting module will utilize contextualinformation to prompt labels thereby enhancing the generation of morerepresentative text features. Moreover to address the challenge of recognizingdistinct actions by multiple people at the same timestamp we design theInterest Token Spotting mechanism which employs pretrained visual knowledge tofind each persons interest context tokens and then these tokens will be usedfor prompting to generate text features tailored to each individual. Toevaluate the ability to detect unseen actions we propose a comprehensivebenchmark on J-HMDB UCF101-24 and AVA datasets. The experiments show that ourmethod achieves superior results compared to previous approaches and can befurther extended to multi-action videos bringing it closer to real-worldapplications. The code and data can be found inhttps://webber2933.github.io/ST-CLIP-project-page. |


| Item |Content|
| --- |---|
|idx| 2408.15992v1 |
|title| CoGen: Learning from Feedback with Coupled Comprehension and Generation |
|authors| Mustafa Omer GulYoav Artzi
|links| http://arxiv.org/abs/2408.15992v1 |
|updated| 2024-08-28 17:58:39 UTC |
|summary| Systems with both language comprehension and generation capabilities canbenefit from the tight connection between the two. This work studies couplingcomprehension and generation with focus on continually learning frominteraction with users. We propose techniques to tightly integrate the twocapabilities for both learning and inference. We situate our studies intwo-player reference games and deploy various models for thousands ofinteractions with human users while learning from interaction feedbacksignals. We show dramatic improvements in performance over time withcomprehension-generation coupling leading to performance improvements up to 26in absolute terms and up to 17 higher accuracies compared to a non-coupledsystem. Our analysis also shows coupling has substantial qualitative impact onthe systems language making it significantly more human-like. |


| Item |Content|
| --- |---|
|idx| 2408.15980v1 |
|title| In-Context Imitation Learning via Next-Token Prediction |
|authors| Letian FuHuang HuangGaurav DattaLawrence Yunliang ChenWilliam Chung-Ho PanitchFangchen LiuHui LiKen Goldberg
|links| http://arxiv.org/abs/2408.15980v1 |
|updated| 2024-08-28 17:50:19 UTC |
|summary| We explore how to enhance next-token prediction models to perform in-contextimitation learning on a real robot where the robot executes new tasks byinterpreting contextual information provided during the input phase withoutupdating its underlying policy parameters. We propose In-Context RobotTransformer ICRT a causal transformer that performs autoregressiveprediction on sensorimotor trajectories without relying on any linguistic dataor reward function. This formulation enables flexible and training-freeexecution of new tasks at test time achieved by prompting the model withsensorimotor trajectories of the new task composing of image observationsactions and states tuples collected through human teleoperation. Experimentswith a Franka Emika robot demonstrate that the ICRT can adapt to new tasksspecified by prompts even in environment configurations that differ from boththe prompt and the training data. In a multitask environment setup ICRTsignificantly outperforms current state-of-the-art next-token prediction modelsin robotics on generalizing to unseen tasks. Code checkpoints and data areavailable on https://icrt.dev/ |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2408.15999v1 |
|title| Q-MRS: A Deep Learning Framework for Quantitative Magnetic Resonance Spectra Analysis |
|authors| Christopher J. WuLawrence S. KegelesJia Guo
|links| http://arxiv.org/abs/2408.15999v1 |
|updated| 2024-08-28 18:05:53 UTC |
|summary| Magnetic resonance spectroscopy MRS is an established technique forstudying tissue metabolism particularly in central nervous system disorders.While powerful and versatile MRS is often limited by challenges associatedwith data quality processing and quantification. Existing MRS quantificationmethods face difficulties in balancing model complexity and reproducibilityduring spectral modeling often falling into the trap of eitheroversimplification or over-parameterization. To address these limitations thisstudy introduces a deep learning DL framework that employs transfer learningin which the model is pre-trained on simulated datasets before it undergoesfine-tuning on in vivo data. The proposed framework showed promisingperformance when applied to the Philips dataset from the BIG GABA repositoryand represents an exciting advancement in MRS data analysis. |


| Item |Content|
| --- |---|
|idx| 2408.15998v1 |
|title| Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders |
|authors| Min ShiFuxiao LiuShihao WangShijia LiaoSubhashree RadhakrishnanDe-An HuangHongxu YinKaran SapraYaser YacoobHumphrey ShiBryan CatanzaroAndrew TaoJan KautzZhiding YuGuilin Liu
|links| http://arxiv.org/abs/2408.15998v1 |
|updated| 2024-08-28 17:59:31 UTC |
|summary| The ability to accurately interpret complex visual information is a crucialtopic of multimodal large language models MLLMs. Recent work indicates thatenhanced visual perception significantly reduces hallucinations and improvesperformance on resolution-sensitive tasks such as optical characterrecognition and document analysis. A number of recent MLLMs achieve this goalusing a mixture of vision encoders. Despite their success there is a lack ofsystematic comparisons and detailed ablation studies addressing criticalaspects such as expert selection and the integration of multiple visionexperts. This study provides an extensive exploration of the design space forMLLMs using a mixture of vision encoders and resolutions. Our findings revealseveral underlying principles common to various existing strategies leading toa streamlined yet effective design approach. We discover that simplyconcatenating visual tokens from a set of complementary vision encoders is aseffective as more complex mixing architectures or strategies. We additionallyintroduce Pre-Alignment to bridge the gap between vision-focused encoders andlanguage tokens enhancing model coherence. The resulting family of MLLMsEagle surpasses other leading open-source models on major MLLM benchmarks.Models and code: https://github.com/NVlabs/Eagle |


| Item |Content|
| --- |---|
|idx| 2408.15997v1 |
|title| Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need |
|authors| Sijia PengYun XiongYangyong ZhuZhiqiang Shen
|links| http://arxiv.org/abs/2408.15997v1 |
|updated| 2024-08-28 17:59:27 UTC |
|summary| Time series forecasting requires balancing short-term and long-termdependencies for accurate predictions. Existing methods mainly focus onlong-term dependency modeling neglecting the complexities of short-termdynamics which may hinder performance. Transformers are superior in modelinglong-term dependencies but are criticized for their quadratic computationalcost. Mamba provides a near-linear alternative but is reported less effectivein time series longterm forecasting due to potential information loss. Currentarchitectures fall short in offering both high efficiency and strongperformance for long-term dependency modeling. To address these challenges weintroduce Mixture of Universals MoU a versatile model to capture bothshort-term and long-term dependencies for enhancing performance in time seriesforecasting. MoU is composed of two novel designs: Mixture of FeatureExtractors MoF an adaptive method designed to improve time series patchrepresentations for short-term dependency and Mixture of Architectures MoAwhich hierarchically integrates Mamba FeedForward Convolution andSelf-Attention architectures in a specialized order to model long-termdependency from a hybrid perspective. The proposed approach achievesstate-of-the-art performance while maintaining relatively low computationalcosts. Extensive experiments on seven real-world datasets demonstrate thesuperiority of MoU. Code is available at https://github.com/lunaaa95/mou/. |


| Item |Content|
| --- |---|
|idx| 2408.15993v1 |
|title| ClimDetect: A Benchmark Dataset for Climate Change Detection and Attribution |
|authors| Sungduk YuBrian L. WhiteAnahita BhiwandiwallaMusashi HinckMatthew Lyle OlsonTung NguyenVasudev Lal
|links| http://arxiv.org/abs/2408.15993v1 |
|updated| 2024-08-28 17:58:53 UTC |
|summary| Detecting and attributing temperature increases due to climate change iscrucial for understanding global warming and guiding adaptation strategies. Thecomplexity of distinguishing human-induced climate signals from naturalvariability has challenged traditional detection and attribution DAapproaches which seek to identify specific fingerprints in climate responsevariables. Deep learning offers potential for discerning these complex patternsin expansive spatial datasets. However lack of standard protocols has hinderedconsistent comparisons across studies. We introduce ClimDetect a standardizeddataset of over 816k daily climate snapshots designed to enhance modelaccuracy in identifying climate change signals. ClimDetect integrates variousinput and target variables used in past research ensuring comparability andconsistency. We also explore the application of vision transformers ViT toclimate data a novel and modernizing approach in this context. Our open-accessdata and code serve as a benchmark for advancing climate science throughimproved model evaluations. ClimDetect is publicly accessible via Huggingfacedataet respository at: https://huggingface.co/datasets/ClimDetect/ClimDetect. |


| Item |Content|
| --- |---|
|idx| 2408.15992v1 |
|title| CoGen: Learning from Feedback with Coupled Comprehension and Generation |
|authors| Mustafa Omer GulYoav Artzi
|links| http://arxiv.org/abs/2408.15992v1 |
|updated| 2024-08-28 17:58:39 UTC |
|summary| Systems with both language comprehension and generation capabilities canbenefit from the tight connection between the two. This work studies couplingcomprehension and generation with focus on continually learning frominteraction with users. We propose techniques to tightly integrate the twocapabilities for both learning and inference. We situate our studies intwo-player reference games and deploy various models for thousands ofinteractions with human users while learning from interaction feedbacksignals. We show dramatic improvements in performance over time withcomprehension-generation coupling leading to performance improvements up to 26in absolute terms and up to 17 higher accuracies compared to a non-coupledsystem. Our analysis also shows coupling has substantial qualitative impact onthe systems language making it significantly more human-like. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2408.15998v1 |
|title| Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders |
|authors| Min ShiFuxiao LiuShihao WangShijia LiaoSubhashree RadhakrishnanDe-An HuangHongxu YinKaran SapraYaser YacoobHumphrey ShiBryan CatanzaroAndrew TaoJan KautzZhiding YuGuilin Liu
|links| http://arxiv.org/abs/2408.15998v1 |
|updated| 2024-08-28 17:59:31 UTC |
|summary| The ability to accurately interpret complex visual information is a crucialtopic of multimodal large language models MLLMs. Recent work indicates thatenhanced visual perception significantly reduces hallucinations and improvesperformance on resolution-sensitive tasks such as optical characterrecognition and document analysis. A number of recent MLLMs achieve this goalusing a mixture of vision encoders. Despite their success there is a lack ofsystematic comparisons and detailed ablation studies addressing criticalaspects such as expert selection and the integration of multiple visionexperts. This study provides an extensive exploration of the design space forMLLMs using a mixture of vision encoders and resolutions. Our findings revealseveral underlying principles common to various existing strategies leading toa streamlined yet effective design approach. We discover that simplyconcatenating visual tokens from a set of complementary vision encoders is aseffective as more complex mixing architectures or strategies. We additionallyintroduce Pre-Alignment to bridge the gap between vision-focused encoders andlanguage tokens enhancing model coherence. The resulting family of MLLMsEagle surpasses other leading open-source models on major MLLM benchmarks.Models and code: https://github.com/NVlabs/Eagle |


| Item |Content|
| --- |---|
|idx| 2408.15996v1 |
|title| Spatio-Temporal Context Prompting for Zero-Shot Action Detection |
|authors| Wei-Jhe HuangMin-Hung ChenShang-Hong Lai
|links| http://arxiv.org/abs/2408.15996v1 |
|updated| 2024-08-28 17:59:05 UTC |
|summary| Spatio-temporal action detection encompasses the tasks of localizing andclassifying individual actions within a video. Recent works aim to enhance thisprocess by incorporating interaction modeling which captures the relationshipbetween people and their surrounding context. However these approaches haveprimarily focused on fully-supervised learning and the current limitation liesin the lack of generalization capability to recognize unseen action categories.In this paper we aim to adapt the pretrained image-language models to detectunseen actions. To this end we propose a method which can effectively leveragethe rich knowledge of visual-language models to perform Person-ContextInteraction. Meanwhile our Context Prompting module will utilize contextualinformation to prompt labels thereby enhancing the generation of morerepresentative text features. Moreover to address the challenge of recognizingdistinct actions by multiple people at the same timestamp we design theInterest Token Spotting mechanism which employs pretrained visual knowledge tofind each persons interest context tokens and then these tokens will be usedfor prompting to generate text features tailored to each individual. Toevaluate the ability to detect unseen actions we propose a comprehensivebenchmark on J-HMDB UCF101-24 and AVA datasets. The experiments show that ourmethod achieves superior results compared to previous approaches and can befurther extended to multi-action videos bringing it closer to real-worldapplications. The code and data can be found inhttps://webber2933.github.io/ST-CLIP-project-page. |


| Item |Content|
| --- |---|
|idx| 2408.15995v1 |
|title| TEDRA: Text-based Editing of Dynamic and Photoreal Actors |
|authors| Basavaraj SunagadHeming ZhuMohit MendirattaAdam KortylewskiChristian TheobaltMarc Habermann
|links| http://arxiv.org/abs/2408.15995v1 |
|updated| 2024-08-28 17:59:02 UTC |
|summary| Over the past years significant progress has been made in creatingphotorealistic and drivable 3D avatars solely from videos of real humans.However a core remaining challenge is the fine-grained and user-friendlyediting of clothing styles by means of textual descriptions. To this end wepresent TEDRA the first method allowing text-based edits of an avatar whichmaintains the avatars high fidelity space-time coherency as well asdynamics and enables skeletal pose and view control. We begin by training amodel to create a controllable and high-fidelity digital replica of the realactor. Next we personalize a pretrained generative diffusion model byfine-tuning it on various frames of the real character captured from differentcamera angles ensuring the digital representation faithfully captures thedynamics and movements of the real person. This two-stage process lays thefoundation for our approach to dynamic human avatar editing. Utilizing thispersonalized diffusion model we modify the dynamic avatar based on a providedtext prompt using our Personalized Normal Aligned Score Distillation SamplingPNA-SDS within a model-based guidance framework. Additionally we propose atime step annealing strategy to ensure high-quality edits. Our resultsdemonstrate a clear improvement over prior work in functionality and visualquality. |


| Item |Content|
| --- |---|
|idx| 2408.15994v1 |
|title| Perceive-IR: Learning to Perceive Degradation Better for All-in-One Image Restoration |
|authors| Xu ZhangJiaqi MaGuoli WangQian ZhangHuan ZhangLefei Zhang
|links| http://arxiv.org/abs/2408.15994v1 |
|updated| 2024-08-28 17:58:54 UTC |
|summary| The limitations of task-specific and general image restoration methods forspecific degradation have prompted the development of all-in-one imagerestoration techniques. However the diversity of patterns among multipledegradation along with the significant uncertainties in mapping betweendegraded images of different severities and their corresponding undistortedversions pose significant challenges to the all-in-one restoration tasks. Toaddress these challenges we propose Perceive-IR an all-in-one image restorerdesigned to achieve fine-grained quality control that enables restored imagesto more closely resemble their undistorted counterparts regardless of the typeor severity of degradation. Specifically Perceive-IR contains two stages: 1prompt learning stage and 2 restoration stage. In the prompt learning stagewe leverage prompt learning to acquire a fine-grained quality perceiver capableof distinguishing three-tier quality levels by constraining the prompt-imagesimilarity in the CLIP perception space. Subsequently this quality perceiverand difficulty-adaptive perceptual loss are integrated as a quality-awarelearning strategy to realize fine-grained quality control in restoration stage.For the restoration stage a semantic guidance module SGM and compact featureextraction CFE are proposed to further promote the restoration process byutilizing the robust semantic information from the pre-trained large scalevision models and distinguishing degradation-specific features. Extensiveexperiments demonstrate that our Perceive-IR outperforms state-of-the-artmethods in all-in-one image restoration tasks and exhibit superiorgeneralization ability when dealing with unseen tasks. |


| Item |Content|
| --- |---|
|idx| 2408.15993v1 |
|title| ClimDetect: A Benchmark Dataset for Climate Change Detection and Attribution |
|authors| Sungduk YuBrian L. WhiteAnahita BhiwandiwallaMusashi HinckMatthew Lyle OlsonTung NguyenVasudev Lal
|links| http://arxiv.org/abs/2408.15993v1 |
|updated| 2024-08-28 17:58:53 UTC |
|summary| Detecting and attributing temperature increases due to climate change iscrucial for understanding global warming and guiding adaptation strategies. Thecomplexity of distinguishing human-induced climate signals from naturalvariability has challenged traditional detection and attribution DAapproaches which seek to identify specific fingerprints in climate responsevariables. Deep learning offers potential for discerning these complex patternsin expansive spatial datasets. However lack of standard protocols has hinderedconsistent comparisons across studies. We introduce ClimDetect a standardizeddataset of over 816k daily climate snapshots designed to enhance modelaccuracy in identifying climate change signals. ClimDetect integrates variousinput and target variables used in past research ensuring comparability andconsistency. We also explore the application of vision transformers ViT toclimate data a novel and modernizing approach in this context. Our open-accessdata and code serve as a benchmark for advancing climate science throughimproved model evaluations. ClimDetect is publicly accessible via Huggingfacedataet respository at: https://huggingface.co/datasets/ClimDetect/ClimDetect. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2408.15923v1 |
|title| Generalized Naive Bayes |
|authors| Edith Alice KovácsAnna OrszágDániel PfeiferAndrás Benczúr
|links| http://arxiv.org/abs/2408.15923v1 |
|updated| 2024-08-28 16:36:18 UTC |
|summary| In this paper we introduce the so-called Generalized Naive Bayes structure asan extension of the Naive Bayes structure. We give a new greedy algorithm thatfinds a good fitting Generalized Naive Bayes GNB probability distribution. Weprove that this fits the data at least as well as the probability distributiondetermined by the classical Naive Bayes NB. Then under a not veryrestrictive condition we give a second algorithm for which we can prove thatit finds the optimal GNB probability distribution i.e. best fitting structurein the sense of KL divergence. Both algorithms are constructed to maximize theinformation content and aim to minimize redundancy. Based on these algorithmsnew methods for feature selection are introduced. We discuss the similaritiesand differences to other related algorithms in terms of structure methodologyand complexity. Experimental results show that the algorithms introducedoutperform the related algorithms in many cases. |


| Item |Content|
| --- |---|
|idx| 2408.15784v1 |
|title| Implicit Regularization Paths of Weighted Neural Representations |
|authors| Jin-Hong DuPratik Patil
|links| http://arxiv.org/abs/2408.15784v1 |
|updated| 2024-08-28 13:26:36 UTC |
|summary| We study the implicit regularization effects induced by observationweighting of pretrained features. For weight and feature matrices of boundedoperator norms that are infinitesimally free with respect to normalized tracefunctionals we derive equivalence paths connecting different weightingmatrices and ridge regularization levels. Specifically we show that ridgeestimators trained on weighted features along the same path are asymptoticallyequivalent when evaluated against test vectors of bounded norms. These pathscan be interpreted as matching the effective degrees of freedom of ridgeestimators fitted with weighted features. For the special case of subsamplingwithout replacement our results apply to independently sampled random featuresand kernel features and confirm recent conjectures Conjectures 7 and 8 of theauthors on the existence of such paths in Patil et al. We also present anadditive risk decomposition for ensembles of weighted estimators and show thatthe risks are equivalent along the paths when the ensemble size goes toinfinity. As a practical consequence of the path equivalences we develop anefficient cross-validation method for tuning and apply it to subsampledpretrained representations across several models e.g. ResNet-50 and datasetse.g. CIFAR-100. |


| Item |Content|
| --- |---|
|idx| 2408.15495v1 |
|title| Remove Symmetries to Control Model Expressivity |
|authors| Liu ZiyinYizhou XuIsaac Chuang
|links| http://arxiv.org/abs/2408.15495v1 |
|updated| 2024-08-28 02:45:41 UTC |
|summary| When symmetry is present in the loss function the model is likely to betrapped in a low-capacity state that is sometimes known as a collapse. Beingtrapped in these low-capacity states can be a major obstacle to training acrossmany scenarios where deep learning technology is applied. We first prove twoconcrete mechanisms through which symmetries lead to reduced capacities andignored features during training. We then propose a simple and theoreticallyjustified algorithm syre to remove almost all symmetry-induced low-capacitystates in neural networks. The proposed method is shown to improve the trainingof neural networks in scenarios when this type of entrapment is especially aconcern. A remarkable merit of the proposed method is that it is model-agnosticand does not require any knowledge of the symmetry. |


| Item |Content|
| --- |---|
|idx| 2408.15458v1 |
|title| PersonalizedUS: Interpretable Breast Cancer Risk Assessment with Local Coverage Uncertainty Quantification |
|authors| Alek FröhlichThiago RamosGustavo CabelloIsabela BuzattoRafael IzbickiDaniel Tiezzi
|links| http://arxiv.org/abs/2408.15458v1 |
|updated| 2024-08-28 00:47:55 UTC |
|summary| Correctly assessing the malignancy of breast lesions identified duringultrasound examinations is crucial for effective clinical decision-making.However the current golden standard relies on manual BI-RADS scoring byclinicians often leading to unnecessary biopsies and a significant mentalhealth burden on patients and their families. In this paper we introducePersonalizedUS an interpretable machine learning system that leverages recentadvances in conformal prediction to provide precise and personalized riskestimates with local coverage guarantees and sensitivity specificity andpredictive values above 0.9 across various threshold levels. In particular weidentify meaningful lesion subgroups where distribution-free model-agnosticconditional coverage holds with approximately 90 of our prediction setscontaining only the ground truth in most lesion subgroups thus explicitlycharacterizing for which patients the model is most suitably applied. Moreoverwe make available a curated tabular dataset of 1936 biopsied breast lesionsfrom a recent observational multicenter study and benchmark the performance ofseveral state-of-the-art learning algorithms. We also report a successful casestudy of the deployed system in the same multicenter context. Concrete clinicalbenefits include up to a 65 reduction in requested biopsies among BI-RADS 4aand 4b lesions with minimal to no missed cancer cases. |


| Item |Content|
| --- |---|
|idx| 2408.15356v1 |
|title| Optimal level set estimation for non-parametric tournament and crowdsourcing problems |
|authors| Maximilian GrafAlexandra CarpentierNicolas Verzelen
|links| http://arxiv.org/abs/2408.15356v1 |
|updated| 2024-08-27 18:28:31 UTC |
|summary| Motivated by crowdsourcing we consider a problem where we partially observethe correctness of the answers of n experts on d questions. In this paperwe assume that both the experts and the questions can be ordered namely thatthe matrix M containing the probability that expert i answers correctly toquestion j is bi-isotonic up to a permutation of it rows and columns. Whennd this also encompasses the strongly stochastic transitive SST modelfrom the tournament literature. Here we focus on the relevant problem ofdeciphering small entries of M from large entries of M which is key incrowdsourcing for efficient allocation of workers to questions. More preciselywe aim at recovering a or several level set p of the matrix up to aprecision h namely recovering resp. the sets of positions ij in Msuch that M_ijph and M_ijp-h. We consider as a loss measure thenumber of misclassified entries. As our main result we construct an efficientpolynomial-time algorithm that turns out to be minimax optimal for thisclassification problem. This heavily contrasts with existing literature in theSST model where for the stronger reconstruction lossstatistical-computational gaps have been conjectured. More generally thisshades light on the nature of statistical-computational gaps for permutationsmodels. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2408.15906v1 |
|title| Exploring the potential of AI in nurturing learner empathy, prosocial values and environmental stewardship |
|authors| Kenneth Y T LimMinh Anh Nguyen DucMinh Tuan Nguyen Thien
|links| http://arxiv.org/abs/2408.15906v1 |
|updated| 2024-08-28 16:19:55 UTC |
|summary| With Artificial Intelligence AI becoming a powerful tool for educationZawacki-Richter et al. 2019 this chapter describes the concept of combininggenerative and traditional AI citizen-science physiological neuroergonomicwearables and environmental sensors into activities for learners to understandtheir own well-being and emotional states better with a view to developingempathy and environmental stewardship. Alongside bespoke and affordablewearables DIY EEG headsets and biometric wristbands interpretable AI anddata science are used for learners to explore how the environment affects themphysiologically and mentally in authentic environments. For examplerelationships between environmental changes e.g. poorer air quality and theirwell-being e.g. cognitive functioning can be discovered. This is particularlycrucial as relevant knowledge can influence the way people treat theenvironment as suggested by the disciplines of environmental neuroscience andenvironmental psychology Doell et al. 2023. Yet according to Palme andSalvati there have been relatively few studies on the relationships betweenmicroclimates and human health and emotions Palme and Salvati 2021. Asanthropogenic environmental pollution is becoming a prevalent problem ourresearch also aims to leverage on generative AI to introduce hypotheticalscenarios of the environment as emotionally strong stimuli of relevance to thelearners. This would provoke an emotional response for them to learn abouttheir own physiological and neurological responses using neuro-physiologicaldata. Ultimately we hope to establish a bidirectional understanding of howthe environment affects humans physiologically and mentally after which togain insights as to how AI can be used to effectively foster empathypro-environmental attitudes and stewardship. |


| Item |Content|
| --- |---|
|idx| 2408.15682v1 |
|title| A quantitative model of takeover request time budget for conditionally automated driving |
|authors| Foghor TanshiDirk Söffker
|links| http://dx.doi.org/10.1109/ACCESS.2023.0322000 |
|updated| 2024-08-28 10:12:44 UTC |
|summary| In conditional automation the automated driving system assumes full controland only issues a takeover request to a human driver to resume driving incritical situations. Previous studies have concluded that the time budgetrequired by drivers to resume driving after a takeover request varies withsituations and different takeover variables. However no comprehensivegeneralized approaches for estimating in advance the time budget required bydrivers to takeover have been provided. In this contribution fixed 7 s andvariable time budgets 6 s 5 s and 4 s with and without visual imageryassistance were investigated for suitability in three takeover scenarios usingperformance measures such as average lateral displacement. The results indicatethat 7 s is suitable for two of the studied scenarios based on theircharacteristics. Using the obtained results and known relations betweentakeover variables a mathematical formula for estimating takeover request timebudget is proposed. The proposed formula integrates individual stimulusresponse time driving experience scenario specific requirements and allowsincreased safety for takeover maneuvers. Furthermore the visual imageryresulted in increased takeover time which invariably increases the time budget.Thus the time demand of the visualized information if applicable such asvisual imagery should be included in the time budget. |


| Item |Content|
| --- |---|
|idx| 2408.15618v1 |
|title| Super-intelligent society for the silver segment: Ethics in design |
|authors| Jaana LeikasRebekah RousiHannu VilpponenPertti Saariluoma
|links| http://arxiv.org/abs/2408.15618v1 |
|updated| 2024-08-28 08:18:33 UTC |
|summary| A super-intelligent AI- society should be based on inclusion so that allmembers of society can equally benefit from the possibilities new technologiesoffer in everyday life. At present the digital society is overwhelming manypeople a large group of whom are older adults whose quality of life has beenundermined in many respects by their difficulties in using digital technology.However this silver segment should be kept involved as active users of digitalservices and contribute to the functioning and development of asuper-intelligent AI-enabled society. The paper calls for action-orienteddesign thinking that considers the challenge to improve the quality of lifewith an emphasis on ethical design and ethical impact assessment. |


| Item |Content|
| --- |---|
|idx| 2408.15543v1 |
|title| An Investigation of Warning Erroneous Chat Translations in Cross-lingual Communication |
|authors| Yunmeng LiJun SuzukiMakoto MorishitaKaori AbeKentaro Inui
|links| http://dx.doi.org/10.18653/v1/2023.ijcnlp-srw.2 |
|updated| 2024-08-28 05:36:25 UTC |
|summary| The complexities of chats pose significant challenges for machine translationmodels. Recognizing the need for a precise evaluation metric to address theissues of chat translation this study introduces Multidimensional QualityMetrics for Chat Translation MQM-Chat. Through the experiments of five modelsusing MQM-Chat we observed that all models generated certain fundamentalerrors while each of them has different shortcomings such as omission overlycorrecting ambiguous source content and buzzword issues resulting in the lossof stylized information. Our findings underscore the effectiveness of MQM-Chatin evaluating chat translation emphasizing the importance of stylized contentand dialogue consistency for future studies. |


| Item |Content|
| --- |---|
|idx| 2408.15367v1 |
|title| How Much is too Much: Exploring the Effect of Verbal Route Description Length on Indoor Navigation |
|authors| Fathima Nourin NPradip PramanickChayan Sarkar
|links| http://arxiv.org/abs/2408.15367v1 |
|updated| 2024-08-27 19:04:07 UTC |
|summary| Navigating through a new indoor environment can be stressful. Recently manyplaces have deployed robots to assist visitors. One of the features of suchrobots is escorting the visitors to their desired destination within theenvironment but this is neither scalable nor necessary for every visitor.Instead a robot assistant could be deployed at a strategic location to providewayfinding instructions. This not only increases the user experience but can behelpful in many time-critical scenarios e.g. escorting someone to theirboarding gate at an airport. However delivering route descriptions verballyposes a challenge. If the description is too verbose people may struggle torecall all the information while overly brief descriptions may be simplyunhelpful. This article focuses on studying the optimal length of verbal routedescriptions that are effective for reaching the destination and easy forpeople to recall. This work proposes a theoretical framework that links routesegments to chunks in working memory. Based on this framework an experiment isdesigned and conducted to examine the effects of route descriptions ofdifferent lengths on navigational performance. The results revealed intriguingpatterns suggesting an ideal length of four route segments. This study lays afoundation for future research exploring the relationship between routedescription lengths working memory capacity and navigational performance inindoor environments. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2408.15725v1 |
|title| Different Facets for Different Experts: A Framework for Streamlining The Integration of Qualitative Insights into ABM Development |
|authors| Vivek NallurPedram AghaeiGraham Finlay
|links| http://arxiv.org/abs/2408.15725v1 |
|updated| 2024-08-28 11:43:14 UTC |
|summary| A key problem in agent-based simulation is that integrating qualitativeinsights from multiple discipline experts is extremely hard. In mostsimulations agent capabilities and corresponding behaviour needs to beprogrammed into the agent. We report on the architecture of a tool thatdisconnects the programmed functions of the agent from the acquisition ofcapability and displayed behaviour. This allows multiple different domainexperts to represent qualitative insights without the need for code to bechanged. It also allows a continuous integration or even change ofqualitative behaviour processes as more insights are gained. The consequentbehaviour observed in the model is both more faithful to the experts insightas well as able to be contrasted against other models representing otherinsights. |


| Item |Content|
| --- |---|
|idx| 2408.15538v1 |
|title| TrafficGamer: Reliable and Flexible Traffic Simulation for Safety-Critical Scenarios with Game-Theoretic Oracles |
|authors| Guanren QiaoGuorui QuanJiawei YuShujun JiaGuiliang Liu
|links| http://arxiv.org/abs/2408.15538v1 |
|updated| 2024-08-28 05:11:16 UTC |
|summary| While modern Autonomous Vehicle AV systems can develop reliable drivingpolicies under regular traffic conditions they frequently struggle withsafety-critical traffic scenarios. This difficulty primarily arises from therarity of such scenarios in driving datasets and the complexities associatedwith predictive modeling among multiple vehicles. To support the testing andrefinement of AV policies simulating safety-critical traffic events is anessential challenge to be addressed. In this work we introduce TrafficGamerwhich facilitates game-theoretic traffic simulation by viewing common roaddriving as a multi-agent game. In evaluating the empirical performance acrossvarious real-world datasets TrafficGamer ensures both fidelity andexploitability of the simulated scenarios guaranteeing that they not onlystatically align with real-world traffic distribution but also efficientlycapture equilibriums for representing safety-critical scenarios involvingmultiple agents. Additionally the results demonstrate that TrafficGamerexhibits highly flexible simulation across various contexts. Specifically wedemonstrate that the generated scenarios can dynamically adapt to equilibriumsof varying tightness by configuring risk-sensitive constraints duringoptimization. To the best of our knowledge TrafficGamer is the first simulatorcapable of generating diverse traffic scenarios involving multiple agents. Wehave provided a demo webpage for the project athttps://qiaoguanren.github.io/trafficgamer-demo/. |


| Item |Content|
| --- |---|
|idx| 2408.15449v1 |
|title| Graph Attention Inference of Network Topology in Multi-Agent Systems |
|authors| Akshay KolliReza AzadehKshitj Jerath
|links| http://arxiv.org/abs/2408.15449v1 |
|updated| 2024-08-27 23:58:51 UTC |
|summary| Accurately identifying the underlying graph structures of multi-agent systemsremains a difficult challenge. Our work introduces a novel machinelearning-based solution that leverages the attention mechanism to predictfuture states of multi-agent systems by learning node representations. Thegraph structure is then inferred from the strength of the attention values.This approach is applied to both linear consensus dynamics and the non-lineardynamics of Kuramoto oscillators resulting in implicit learning the graph bylearning good agent representations. Our results demonstrate that the presenteddata-driven graph attention machine learning model can identify the networktopology in multi-agent systems even when the underlying dynamic model is notknown as evidenced by the F1 scores achieved in the link prediction. |


| Item |Content|
| --- |---|
|idx| 2408.14948v1 |
|title| Decentralized Unlabeled Multi-agent Pathfinding Via Target And Priority Swapping (With Supplementary) |
|authors| Stepan DergachevKonstantin Yakovlev
|links| http://arxiv.org/abs/2408.14948v1 |
|updated| 2024-08-27 10:45:57 UTC |
|summary| In this paper we study a challenging variant of the multi-agent pathfindingproblem MAPF when a set of agents must reach a set of goal locations but itdoes not matter which agent reaches a specific goal - Anonymous MAPF AMAPF.Current optimal and suboptimal AMAPF solvers rely on the existence of acentralized controller which is in charge of both target assignment andpathfinding. We extend the state of the art and present the first AMAPF solvercapable of solving the problem at hand in a fully decentralized fashion wheneach agent makes decisions individually and relies only on the localcommunication with the others. The core of our method is a priority and targetswapping procedure tailored to produce consistent goal assignments i.e. makingsure that no two agents are heading towards the same goal. Coupled with anestablished rule-based path planning we end up with a TP-SWAP an efficientand flexible approach to solve decentralized AMAPF. On the theoretical side weprove that TP-SWAP is complete i.e. TP-SWAP guarantees that each target willbe reached by some agent. Empirically we evaluate TP-SWAP across a wide rangeof setups and compare it to both centralized and decentralized baselines.Indeed TP-SWAP outperforms the fully-decentralized competitor and can evenoutperform the semi-decentralized one i.e. the one relying on the initialconsistent goal assignment in terms of flowtime a widespread cost objectivein MAPF |


| Item |Content|
| --- |---|
|idx| 2408.14527v1 |
|title| Multi-Agent Path Finding with Real Robot Dynamics and Interdependent Tasks for Automated Warehouses |
|authors| Vassilissa Lehoux-LebacqueTomi SilanderChristelle LoiodiceSeungjoon LeeAlbert WangSofia Michel
|links| http://arxiv.org/abs/2408.14527v1 |
|updated| 2024-08-26 15:13:38 UTC |
|summary| Multi-Agent Path Finding MAPF is an important optimization problemunderlying the deployment of robots in automated warehouses and factories.Despite the large body of work on this topic most approaches make heavysimplifications both on the environment and the agents which make theresulting algorithms impractical for real-life scenarios. In this paper weconsider a realistic problem of online order delivery in a warehouse where afleet of robots bring the products belonging to each order from shelves toworkstations. This creates a stream of inter-dependent pickup and deliverytasks and the associated MAPF problem consists of computing realisticcollision-free robot trajectories fulfilling these tasks. To solve this MAPFproblem we propose an extension of the standard Prioritized Planning algorithmto deal with the inter-dependent tasks Interleaved Prioritized Planning and anovel Via-Point Star VP algorithm to compute an optimal dynamics-compliantrobot trajectory to visit a sequence of goal locations while avoiding movingobstacles. We prove the completeness of our approach and evaluate it insimulation as well as in a real warehouse. |


