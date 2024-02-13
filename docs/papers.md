# cs.CL 

| Item |Content|
| --- |---|
|idx| 2402.06627v1 |
|title| Feedback Loops With Language Models Drive In-Context Reward Hacking |
|authors| Alexander PanErik JonesMeena JagadeesanJacob Steinhardt
|links| http://arxiv.org/abs/2402.06627v1 |
|updated| 2024-02-09 18:59:29 UTC |
|summary| Language models influence the external world: they query APIs that read andwrite to web pages generate content that shapes human behavior and run systemcommands as autonomous agents. These interactions form feedback loops: LLMoutputs affect the world which in turn affect subsequent LLM outputs. In thiswork we show that feedback loops can cause in-context reward hacking ICRHwhere the LLM at test-time optimizes a potentially implicit objective butcreates negative side effects in the process. For example consider an LLMagent deployed to increase Twitter engagement the LLM may retrieve itsprevious tweets into the context window and make them more controversialincreasing engagement but also toxicity. We identify and study two processesthat lead to ICRH: output-refinement and policy-refinement. For theseprocesses evaluations on static datasets are insufficient -- they miss thefeedback effects and thus cannot capture the most harmful behavior. Inresponse we provide three recommendations for evaluation to capture moreinstances of ICRH. As AI development accelerates the effects of feedback loopswill proliferate increasing the need to understand their role in shaping LLMbehavior. |


| Item |Content|
| --- |---|
|idx| 2402.06625v1 |
|title| Understanding the Effects of Iterative Prompting on Truthfulness |
|authors| Satyapriya KrishnaChirag AgarwalHimabindu Lakkaraju
|links| http://arxiv.org/abs/2402.06625v1 |
|updated| 2024-02-09 18:57:08 UTC |
|summary| The development of Large Language Models LLMs has notably transformednumerous sectors offering impressive text generation capabilities. Yet thereliability and truthfulness of these models remain pressing concerns. To thisend we investigate iterative prompting a strategy hypothesized to refine LLMresponses assessing its impact on LLM truthfulness an area which has not beenthoroughly explored. Our extensive experiments delve into the intricacies ofiterative prompting variants examining their influence on the accuracy andcalibration of model responses. Our findings reveal that naive promptingmethods significantly undermine truthfulness leading to exacerbatedcalibration errors. In response to these challenges we introduce severalprompting variants designed to address the identified issues. These variantsdemonstrate marked improvements over existing baselines signaling a promisingdirection for future research. Our work provides a nuanced understanding ofiterative prompting and introduces novel approaches to enhance the truthfulnessof LLMs thereby contributing to the development of more accurate andtrustworthy AI systems. |


| Item |Content|
| --- |---|
|idx| 2402.06619v1 |
|title| Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning |
|authors| Shivalika SinghFreddie VargusDaniel DsouzaBörje F. KarlssonAbinaya MahendiranWei-Yin KoHerumb ShandilyaJay PatelDeividas MataciunasLaura OMahonyMike ZhangRamith HettiarachchiJoseph WilsonMarina MachadoLuisa Souza MouraDominik KrzemińskiHakimeh FadaeiIrem ErgünIfeoma OkohAisha AlaagibOshan MudannayakeZaid AlyafeaiVu Minh ChienSebastian RuderSurya GuthikondaEmad A. AlghamdiSebastian GehrmannNiklas MuennighoffMax BartoloJulia KreutzerAhmet ÜstünMarzieh FadaeeSara Hooker
|links| http://arxiv.org/abs/2402.06619v1 |
|updated| 2024-02-09 18:51:49 UTC |
|summary| Datasets are foundational to many breakthroughs in modern artificialintelligence. Many recent achievements in the space of natural languageprocessing NLP can be attributed to the finetuning of pre-trained models on adiverse set of tasks that enables a large language model LLM to respond toinstructions. Instruction fine-tuning IFT requires specifically constructedand annotated datasets. However existing datasets are almost all in theEnglish language. In this work our primary goal is to bridge the language gapby building a human-curated instruction-following dataset spanning 65languages. We worked with fluent speakers of languages from around the world tocollect natural instances of instructions and completions. Furthermore wecreate the most extensive multilingual collection to date comprising 513million instances through templating and translating existing datasets across114 languages. In total we contribute four key resources: we develop andopen-source the Aya Annotation Platform the Aya Dataset the Aya Collectionand the Aya Evaluation Suite. The Aya initiative also serves as a valuable casestudy in participatory research involving collaborators from 119 countries. Wesee this as a valuable framework for future research collaborations that aim tobridge gaps in resources. |


| Item |Content|
| --- |---|
|idx| 2402.06617v1 |
|title| FaBERT: Pre-training BERT on Persian Blogs |
|authors| Mostafa MasumiSeyed Soroush MajdMehrnoush ShamsfardHamid Beigy
|links| http://arxiv.org/abs/2402.06617v1 |
|updated| 2024-02-09 18:50:51 UTC |
|summary| We introduce FaBERT a Persian BERT-base model pre-trained on the HmBlogscorpus encompassing both informal and formal Persian texts. FaBERT is designedto excel in traditional Natural Language Understanding NLU tasks addressingthe intricacies of diverse sentence structures and linguistic styles prevalentin the Persian language. In our comprehensive evaluation of FaBERT on 12datasets in various downstream tasks encompassing Sentiment Analysis SANamed Entity Recognition NER Natural Language Inference NLI QuestionAnswering QA and Question Paraphrasing QP it consistently demonstratedimproved performance all achieved within a compact model size. The findingshighlight the importance of utilizing diverse and cleaned corpora such asHmBlogs to enhance the performance of language models like BERT in PersianNatural Language Processing NLP applications. FaBERT is openly accessible athttps://huggingface.co/sbunlp/fabert |


| Item |Content|
| --- |---|
|idx| 2402.06608v1 |
|title| TIC: Translate-Infer-Compile for accurate 'text to plan' using LLMs and logical intermediate representations |
|authors| Sudhir AgarwalAnu Sreepathy
|links| http://arxiv.org/abs/2402.06608v1 |
|updated| 2024-02-09 18:39:13 UTC |
|summary| We study the problem of generating plans for given natural language planningtask requests. On one hand LLMs excel at natural language processing but donot perform well on planning. On the other hand classical planning tools excelat planning tasks but require input in a structured language such as thePlanning Domain Definition Language PDDL. We leverage the strengths of boththe techniques by using an LLM for generating the PDDL representation taskPDDL of planning task requests followed by using a classical planner forcomputing a plan. Unlike previous approaches that use LLMs for generating taskPDDLs directly our approach comprises of a translate: using an LLM only forgenerating a logically interpretable intermediate representation of naturallanguage task descriptions b infer: deriving additional logically dependentinformation from the intermediate representation using a logic reasonercurrently Answer Set Programming solver and c compile: generating thetarget task PDDL from the base and inferred information. We observe that usingan LLM to only output the intermediate representation significantly reduces LLMerrors. Consequently TIC approach achieves for at least one LLM highaccuracy on task PDDL generation for all seven domains of our evaluationdataset. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2402.06627v1 |
|title| Feedback Loops With Language Models Drive In-Context Reward Hacking |
|authors| Alexander PanErik JonesMeena JagadeesanJacob Steinhardt
|links| http://arxiv.org/abs/2402.06627v1 |
|updated| 2024-02-09 18:59:29 UTC |
|summary| Language models influence the external world: they query APIs that read andwrite to web pages generate content that shapes human behavior and run systemcommands as autonomous agents. These interactions form feedback loops: LLMoutputs affect the world which in turn affect subsequent LLM outputs. In thiswork we show that feedback loops can cause in-context reward hacking ICRHwhere the LLM at test-time optimizes a potentially implicit objective butcreates negative side effects in the process. For example consider an LLMagent deployed to increase Twitter engagement the LLM may retrieve itsprevious tweets into the context window and make them more controversialincreasing engagement but also toxicity. We identify and study two processesthat lead to ICRH: output-refinement and policy-refinement. For theseprocesses evaluations on static datasets are insufficient -- they miss thefeedback effects and thus cannot capture the most harmful behavior. Inresponse we provide three recommendations for evaluation to capture moreinstances of ICRH. As AI development accelerates the effects of feedback loopswill proliferate increasing the need to understand their role in shaping LLMbehavior. |


| Item |Content|
| --- |---|
|idx| 2402.06619v1 |
|title| Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning |
|authors| Shivalika SinghFreddie VargusDaniel DsouzaBörje F. KarlssonAbinaya MahendiranWei-Yin KoHerumb ShandilyaJay PatelDeividas MataciunasLaura OMahonyMike ZhangRamith HettiarachchiJoseph WilsonMarina MachadoLuisa Souza MouraDominik KrzemińskiHakimeh FadaeiIrem ErgünIfeoma OkohAisha AlaagibOshan MudannayakeZaid AlyafeaiVu Minh ChienSebastian RuderSurya GuthikondaEmad A. AlghamdiSebastian GehrmannNiklas MuennighoffMax BartoloJulia KreutzerAhmet ÜstünMarzieh FadaeeSara Hooker
|links| http://arxiv.org/abs/2402.06619v1 |
|updated| 2024-02-09 18:51:49 UTC |
|summary| Datasets are foundational to many breakthroughs in modern artificialintelligence. Many recent achievements in the space of natural languageprocessing NLP can be attributed to the finetuning of pre-trained models on adiverse set of tasks that enables a large language model LLM to respond toinstructions. Instruction fine-tuning IFT requires specifically constructedand annotated datasets. However existing datasets are almost all in theEnglish language. In this work our primary goal is to bridge the language gapby building a human-curated instruction-following dataset spanning 65languages. We worked with fluent speakers of languages from around the world tocollect natural instances of instructions and completions. Furthermore wecreate the most extensive multilingual collection to date comprising 513million instances through templating and translating existing datasets across114 languages. In total we contribute four key resources: we develop andopen-source the Aya Annotation Platform the Aya Dataset the Aya Collectionand the Aya Evaluation Suite. The Aya initiative also serves as a valuable casestudy in participatory research involving collaborators from 119 countries. Wesee this as a valuable framework for future research collaborations that aim tobridge gaps in resources. |


| Item |Content|
| --- |---|
|idx| 2402.06608v1 |
|title| TIC: Translate-Infer-Compile for accurate 'text to plan' using LLMs and logical intermediate representations |
|authors| Sudhir AgarwalAnu Sreepathy
|links| http://arxiv.org/abs/2402.06608v1 |
|updated| 2024-02-09 18:39:13 UTC |
|summary| We study the problem of generating plans for given natural language planningtask requests. On one hand LLMs excel at natural language processing but donot perform well on planning. On the other hand classical planning tools excelat planning tasks but require input in a structured language such as thePlanning Domain Definition Language PDDL. We leverage the strengths of boththe techniques by using an LLM for generating the PDDL representation taskPDDL of planning task requests followed by using a classical planner forcomputing a plan. Unlike previous approaches that use LLMs for generating taskPDDLs directly our approach comprises of a translate: using an LLM only forgenerating a logically interpretable intermediate representation of naturallanguage task descriptions b infer: deriving additional logically dependentinformation from the intermediate representation using a logic reasonercurrently Answer Set Programming solver and c compile: generating thetarget task PDDL from the base and inferred information. We observe that usingan LLM to only output the intermediate representation significantly reduces LLMerrors. Consequently TIC approach achieves for at least one LLM highaccuracy on task PDDL generation for all seven domains of our evaluationdataset. |


| Item |Content|
| --- |---|
|idx| 2402.06606v1 |
|title| RQP-SGD: Differential Private Machine Learning through Noisy SGD and Randomized Quantization |
|authors| Ce FengParv Venkitasubramaniam
|links| http://arxiv.org/abs/2402.06606v1 |
|updated| 2024-02-09 18:34:08 UTC |
|summary| The rise of IoT devices has prompted the demand for deploying machinelearning at-the-edge with real-time efficient and secure data processing. Inthis context implementing machine learning ML models with real-valued weightparameters can prove to be impractical particularly for large models and thereis a need to train models with quantized discrete weights. At the same timethese low-dimensional models also need to preserve privacy of the underlyingdataset. In this work we present RQP-SGD a new approach forprivacy-preserving quantization to train machine learning models for low-memoryML-at-the-edge. This approach combines differentially private stochasticgradient descent DP-SGD with randomized quantization providing a measurableprivacy guarantee in machine learning. In particular we study the utilityconvergence of implementing RQP-SGD on ML tasks with convex objectives andquantization constraints and demonstrate its efficacy over deterministicquantization. Through experiments conducted on two datasets we show thepractical effectiveness of RQP-SGD. |


| Item |Content|
| --- |---|
|idx| 2402.06599v1 |
|title| On the Out-Of-Distribution Generalization of Multimodal Large Language Models |
|authors| Xingxuan ZhangJiansheng LiWenjing ChuJunjia HaiRenzhe XuYuqing YangShikai GuanJiazheng XuPeng Cui
|links| http://arxiv.org/abs/2402.06599v1 |
|updated| 2024-02-09 18:21:51 UTC |
|summary| We investigate the generalization boundaries of current Multimodal LargeLanguage Models MLLMs via comprehensive evaluation under out-of-distributionscenarios and domain-specific tasks. We evaluate their zero-shot generalizationacross synthetic images real-world distributional shifts and specializeddatasets like medical and molecular imagery. Empirical results indicate thatMLLMs struggle with generalization beyond common training domains limitingtheir direct application without adaptation. To understand the cause ofunreliable performance we analyze three hypotheses: semanticmisinterpretation visual feature extraction insufficiency and mappingdeficiency. Results identify mapping deficiency as the primary hurdle. Toaddress this problem we show that in-context learning ICL can significantlyenhance MLLMs generalization opening new avenues for overcominggeneralization barriers. We further explore the robustness of ICL underdistribution shifts and show its vulnerability to domain shifts label shiftsand spurious correlation shifts between in-context examples and test data. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2402.06627v1 |
|title| Feedback Loops With Language Models Drive In-Context Reward Hacking |
|authors| Alexander PanErik JonesMeena JagadeesanJacob Steinhardt
|links| http://arxiv.org/abs/2402.06627v1 |
|updated| 2024-02-09 18:59:29 UTC |
|summary| Language models influence the external world: they query APIs that read andwrite to web pages generate content that shapes human behavior and run systemcommands as autonomous agents. These interactions form feedback loops: LLMoutputs affect the world which in turn affect subsequent LLM outputs. In thiswork we show that feedback loops can cause in-context reward hacking ICRHwhere the LLM at test-time optimizes a potentially implicit objective butcreates negative side effects in the process. For example consider an LLMagent deployed to increase Twitter engagement the LLM may retrieve itsprevious tweets into the context window and make them more controversialincreasing engagement but also toxicity. We identify and study two processesthat lead to ICRH: output-refinement and policy-refinement. For theseprocesses evaluations on static datasets are insufficient -- they miss thefeedback effects and thus cannot capture the most harmful behavior. Inresponse we provide three recommendations for evaluation to capture moreinstances of ICRH. As AI development accelerates the effects of feedback loopswill proliferate increasing the need to understand their role in shaping LLMbehavior. |


| Item |Content|
| --- |---|
|idx| 2402.06614v1 |
|title| The Complexity of Sequential Prediction in Dynamical Systems |
|authors| Vinod RamanUnique SubediAmbuj Tewari
|links| http://arxiv.org/abs/2402.06614v1 |
|updated| 2024-02-09 18:45:00 UTC |
|summary| We study the problem of learning to predict the next state of a dynamicalsystem when the underlying evolution function is unknown. Unlike previous workwe place no parametric assumptions on the dynamical system and study theproblem from a learning theory perspective. We define new combinatorialmeasures and dimensions and show that they quantify the optimal mistake andregret bounds in the realizable and agnostic setting respectively. |


| Item |Content|
| --- |---|
|idx| 2402.06606v1 |
|title| RQP-SGD: Differential Private Machine Learning through Noisy SGD and Randomized Quantization |
|authors| Ce FengParv Venkitasubramaniam
|links| http://arxiv.org/abs/2402.06606v1 |
|updated| 2024-02-09 18:34:08 UTC |
|summary| The rise of IoT devices has prompted the demand for deploying machinelearning at-the-edge with real-time efficient and secure data processing. Inthis context implementing machine learning ML models with real-valued weightparameters can prove to be impractical particularly for large models and thereis a need to train models with quantized discrete weights. At the same timethese low-dimensional models also need to preserve privacy of the underlyingdataset. In this work we present RQP-SGD a new approach forprivacy-preserving quantization to train machine learning models for low-memoryML-at-the-edge. This approach combines differentially private stochasticgradient descent DP-SGD with randomized quantization providing a measurableprivacy guarantee in machine learning. In particular we study the utilityconvergence of implementing RQP-SGD on ML tasks with convex objectives andquantization constraints and demonstrate its efficacy over deterministicquantization. Through experiments conducted on two datasets we show thepractical effectiveness of RQP-SGD. |


| Item |Content|
| --- |---|
|idx| 2402.06590v1 |
|title| Predictive representations: building blocks of intelligence |
|authors| Wilka CarvalhoMomchil S. TomovWilliam de CothiCaswell BarrySamuel J. Gershman
|links| http://arxiv.org/abs/2402.06590v1 |
|updated| 2024-02-09 18:10:38 UTC |
|summary| Adaptive behavior often requires predicting future events. The theory ofreinforcement learning prescribes what kinds of predictive representations areuseful and how to compute them. This paper integrates these theoretical ideaswith work on cognition and neuroscience. We pay special attention to thesuccessor representation SR and its generalizations which have been widelyapplied both as engineering tools and models of brain function. Thisconvergence suggests that particular kinds of predictive representations mayfunction as versatile building blocks of intelligence. |


| Item |Content|
| --- |---|
|idx| 2402.06581v1 |
|title| More than the Sum of Its Parts: Ensembling Backbone Networks for Few-Shot Segmentation |
|authors| Nico CatalanoAlessandro MaranelliAgnese ChiattiMatteo Matteucci
|links| http://arxiv.org/abs/2402.06581v1 |
|updated| 2024-02-09 18:01:15 UTC |
|summary| Semantic segmentation is a key prerequisite to robust image understanding forapplications in acrlongai and Robotics. acrlongfss in particularconcerns the extension and optimization of traditional segmentation methods inchallenging conditions where limited training examples are available. Apredominant approach in acrlongfss is to rely on a single backbone forvisual feature extraction. Choosing which backbone to leverage is a decidingfactor contributing to the overall performance. In this work we interrogate onwhether fusing features from different backbones can improve the ability ofacrlongfss models to capture richer visual features. To tackle thisquestion we propose and compare two ensembling techniques-Independent Votingand Feature Fusion. Among the available acrlongfss methods we implement theproposed ensembling techniques on PANet. The module dedicated to predictingsegmentation masks from the backbone embeddings in PANet avoids trainableparameters creating a controlled in vitro setting for isolating the impactof different ensembling strategies. Leveraging the complementary strengths ofdifferent backbones our approach outperforms the original single-backbonePANet across standard benchmarks even in challenging one-shot learningscenarios. Specifically it achieved a performance improvement of 7.37 onPASCAL-5textsuperscripti and of 10.68 on COCO-20textsuperscripti inthe top-performing scenario where three backbones are combined. These resultstogether with the qualitative inspection of the predicted subject maskssuggest that relying on multiple backbones in PANet leads to a morecomprehensive feature representation thus expediting the successfulapplication of acrlongfss methods in challenging data-scarce environments. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2402.06611v1 |
|title| Image-based Deep Learning for the time-dependent prediction of fresh concrete properties |
|authors| Max MeyerAmadeus LangerMax MehltretterDries BeyerMax CoenenTobias SchackMichael HaistChristian Heipke
|links| http://arxiv.org/abs/2402.06611v1 |
|updated| 2024-02-09 18:42:30 UTC |
|summary| Increasing the degree of digitisation and automation in the concreteproduction process can play a crucial role in reducing the CO_2 emissionsthat are associated with the production of concrete. In this paper a method ispresented that makes it possible to predict the properties of fresh concreteduring the mixing process based on stereoscopic image sequences of theconcretes flow behaviour. A Convolutional Neural Network CNN is used for theprediction which receives the images supported by information on the mixdesign as input. In addition the network receives temporal information in theform of the time difference between the time at which the images are taken andthe time at which the reference values of the concretes are carried out. Withthis temporal information the network implicitly learns the time-dependentbehaviour of the concretes properties. The network predicts the slump flowdiameter the yield stress and the plastic viscosity. The time-dependentprediction potentially opens up the pathway to determine the temporaldevelopment of the fresh concrete properties already during mixing. Thisprovides a huge advantage for the concrete industry. As a resultcountermeasures can be taken in a timely manner. It is shown that an approachbased on depth and optical flow images supported by information of the mixdesign achieves the best results. |


| Item |Content|
| --- |---|
|idx| 2402.06599v1 |
|title| On the Out-Of-Distribution Generalization of Multimodal Large Language Models |
|authors| Xingxuan ZhangJiansheng LiWenjing ChuJunjia HaiRenzhe XuYuqing YangShikai GuanJiazheng XuPeng Cui
|links| http://arxiv.org/abs/2402.06599v1 |
|updated| 2024-02-09 18:21:51 UTC |
|summary| We investigate the generalization boundaries of current Multimodal LargeLanguage Models MLLMs via comprehensive evaluation under out-of-distributionscenarios and domain-specific tasks. We evaluate their zero-shot generalizationacross synthetic images real-world distributional shifts and specializeddatasets like medical and molecular imagery. Empirical results indicate thatMLLMs struggle with generalization beyond common training domains limitingtheir direct application without adaptation. To understand the cause ofunreliable performance we analyze three hypotheses: semanticmisinterpretation visual feature extraction insufficiency and mappingdeficiency. Results identify mapping deficiency as the primary hurdle. Toaddress this problem we show that in-context learning ICL can significantlyenhance MLLMs generalization opening new avenues for overcominggeneralization barriers. We further explore the robustness of ICL underdistribution shifts and show its vulnerability to domain shifts label shiftsand spurious correlation shifts between in-context examples and test data. |


| Item |Content|
| --- |---|
|idx| 2402.06581v1 |
|title| More than the Sum of Its Parts: Ensembling Backbone Networks for Few-Shot Segmentation |
|authors| Nico CatalanoAlessandro MaranelliAgnese ChiattiMatteo Matteucci
|links| http://arxiv.org/abs/2402.06581v1 |
|updated| 2024-02-09 18:01:15 UTC |
|summary| Semantic segmentation is a key prerequisite to robust image understanding forapplications in acrlongai and Robotics. acrlongfss in particularconcerns the extension and optimization of traditional segmentation methods inchallenging conditions where limited training examples are available. Apredominant approach in acrlongfss is to rely on a single backbone forvisual feature extraction. Choosing which backbone to leverage is a decidingfactor contributing to the overall performance. In this work we interrogate onwhether fusing features from different backbones can improve the ability ofacrlongfss models to capture richer visual features. To tackle thisquestion we propose and compare two ensembling techniques-Independent Votingand Feature Fusion. Among the available acrlongfss methods we implement theproposed ensembling techniques on PANet. The module dedicated to predictingsegmentation masks from the backbone embeddings in PANet avoids trainableparameters creating a controlled in vitro setting for isolating the impactof different ensembling strategies. Leveraging the complementary strengths ofdifferent backbones our approach outperforms the original single-backbonePANet across standard benchmarks even in challenging one-shot learningscenarios. Specifically it achieved a performance improvement of 7.37 onPASCAL-5textsuperscripti and of 10.68 on COCO-20textsuperscripti inthe top-performing scenario where three backbones are combined. These resultstogether with the qualitative inspection of the predicted subject maskssuggest that relying on multiple backbones in PANet leads to a morecomprehensive feature representation thus expediting the successfulapplication of acrlongfss methods in challenging data-scarce environments. |


| Item |Content|
| --- |---|
|idx| 2402.06560v1 |
|title| Video Annotator: A framework for efficiently building video classifiers using vision-language models and active learning |
|authors| Amir ZiaiAneesh Vartakavi
|links| http://arxiv.org/abs/2402.06560v1 |
|updated| 2024-02-09 17:19:05 UTC |
|summary| High-quality and consistent annotations are fundamental to the successfuldevelopment of robust machine learning models. Traditional data annotationmethods are resource-intensive and inefficient often leading to a reliance onthird-party annotators who are not the domain experts. Hard samples which areusually the most informative for model training tend to be difficult to labelaccurately and consistently without business context. These can ariseunpredictably during the annotation process requiring a variable number ofiterations and rounds of feedback leading to unforeseen expenses and timecommitments to guarantee quality.  We posit that more direct involvement of domain experts using ahuman-in-the-loop system can resolve many of these practical challenges. Wepropose a novel framework we call Video Annotator VA for annotatingmanaging and iterating on video classification datasets. Our approach offers anew paradigm for an end-user-centered model development process enhancing theefficiency usability and effectiveness of video classifiers. Uniquely VAallows for a continuous annotation process seamlessly integrating datacollection and model training.  We leverage the zero-shot capabilities of vision-language foundation modelscombined with active learning techniques and demonstrate that VA enables theefficient creation of high-quality models. VA achieves a median 6.8 pointimprovement in Average Precision relative to the most competitive baselineacross a wide-ranging assortment of tasks. We release a dataset with 153klabels across 56 video understanding tasks annotated by three professionalvideo editors using VA and also release code to replicate our experiments at:http://github.com/netflix/videoannotator. |


| Item |Content|
| --- |---|
|idx| 2402.06539v1 |
|title| Hybridnet for depth estimation and semantic segmentation |
|authors| Dalila Sánchez-EscobedoXiao LinJosep R. CasasMontse Pardàs
|links| http://dx.doi.org/10.1109/ICASSP.2018.8462433 |
|updated| 2024-02-09 16:52:45 UTC |
|summary| Semantic segmentation and depth estimation are two important tasks in thearea of image processing. Traditionally these two tasks are addressed in anindependent manner. However for those applications where geometric andsemantic information is required such as robotics or autonomousnavigationdepth or semantic segmentation alone are not sufficient. In thispaper depth estimation and semantic segmentation are addressed together from asingle input image through a hybrid convolutional network. Different from thestate of the art methods where features are extracted by a sole featureextraction network for both tasks the proposed HybridNet improves the featuresextraction by separating the relevant features for one task from those whichare relevant for both. Experimental results demonstrate that HybridNet resultsare comparable with the state of the art methods as well as the single taskmethods that HybridNet is based on. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2402.06614v1 |
|title| The Complexity of Sequential Prediction in Dynamical Systems |
|authors| Vinod RamanUnique SubediAmbuj Tewari
|links| http://arxiv.org/abs/2402.06614v1 |
|updated| 2024-02-09 18:45:00 UTC |
|summary| We study the problem of learning to predict the next state of a dynamicalsystem when the underlying evolution function is unknown. Unlike previous workwe place no parametric assumptions on the dynamical system and study theproblem from a learning theory perspective. We define new combinatorialmeasures and dimensions and show that they quantify the optimal mistake andregret bounds in the realizable and agnostic setting respectively. |


| Item |Content|
| --- |---|
|idx| 2402.06578v1 |
|title| On the Universality of Coupling-based Normalizing Flows |
|authors| Felix DraxlerStefan WahlChristoph SchnörrUllrich Köthe
|links| http://arxiv.org/abs/2402.06578v1 |
|updated| 2024-02-09 17:51:43 UTC |
|summary| We present a novel theoretical framework for understanding the expressivepower of coupling-based normalizing flows such as RealNVP. Despite theirprevalence in scientific applications a comprehensive understanding ofcoupling flows remains elusive due to their restricted architectures. Existingtheorems fall short as they require the use of arbitrarily ill-conditionedneural networks limiting practical applicability. Additionally we demonstratethat these constructions inherently lead to volume-preserving flows a propertywhich we show to be a fundamental constraint for expressivity. We propose a newdistributional universality theorem for coupling-based normalizing flows whichovercomes several limitations of prior work. Our results support the generalwisdom that the coupling architecture is expressive and provide a nuanced viewfor choosing the expressivity of coupling functions bridging a gap betweenempirical results and theoretical understanding. |


| Item |Content|
| --- |---|
|idx| 2402.06535v1 |
|title| Bandit Convex Optimisation |
|authors| Tor Lattimore
|links| http://arxiv.org/abs/2402.06535v1 |
|updated| 2024-02-09 16:49:13 UTC |
|summary| Bandit convex optimisation is a fundamental framework for studyingzeroth-order convex optimisation. These notes cover the many tools used forthis problem including cutting plane methods interior point methodscontinuous exponential weights gradient descent and online Newton step. Thenuances between the many assumptions and setups are explained. Although thereis not much truly new here some existing tools are applied in novel ways toobtain new algorithms. A few bounds are improved in minor ways. |


| Item |Content|
| --- |---|
|idx| 2402.06525v1 |
|title| Flexible infinite-width graph convolutional networks and the importance of representation learning |
|authors| Ben AnsonEdward MilsomLaurence Aitchison
|links| http://arxiv.org/abs/2402.06525v1 |
|updated| 2024-02-09 16:37:08 UTC |
|summary| A common theoretical approach to understanding neural networks is to take aninfinite-width limit at which point the outputs become Gaussian process GPdistributed. This is known as a neural network Gaussian process NNGP.However the NNGP kernel is fixed and tunable only through a small number ofhyperparameters eliminating any possibility of representation learning. Thiscontrasts with finite-width NNs which are often believed to perform wellprecisely because they are able to learn representations. Thus in simplifyingNNs to make them theoretically tractable NNGPs may eliminate precisely whatmakes them work well representation learning. This motivated us to understandwhether representation learning is necessary in a range of graph classificationtasks. We develop a precise tool for this task the graph convolutional deepkernel machine. This is very similar to an NNGP in that it is an infinitewidth limit and uses kernels but comes with a knob to control the amount ofrepresentation learning. We found that representation learning is necessary inthe sense that it gives dramatic performance improvements in graphclassification tasks and heterophilous node classification tasks but not inhomophilous node classification tasks. |


| Item |Content|
| --- |---|
|idx| 2402.06461v1 |
|title| Sequential Flow Matching for Generative Modeling |
|authors| Jongmin YoonJuho Lee
|links| http://arxiv.org/abs/2402.06461v1 |
|updated| 2024-02-09 15:09:38 UTC |
|summary| Straightening the probability flow of the continuous-time generative modelssuch as diffusion models or flow-based models is the key to fast samplingthrough the numerical solvers existing methods learn a linear path by directlygenerating the probability path the joint distribution between the noise anddata distribution. One key reason for the slow sampling speed of the ODE-basedsolvers that simulate these generative models is the global truncation error ofthe ODE solver caused by the high curvature of the ODE trajectory whichexplodes the truncation error of the numerical solvers in the low-NFE regime.To address this challenge We propose a novel method called SeqRF a learningtechnique that straightens the probability flow to reduce the global truncationerror and hence enable acceleration of sampling and improve the synthesisquality. In both theoretical and empirical studies we first observe thestraightening property of our SeqRF. Through empirical evaluations via SeqRFover flow-based generative models We achieve surpassing results on CIFAR-10CelebA-64 times 64 and LSUN-Church datasets. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2402.06596v1 |
|title| Understanding the Weakness of Large Language Model Agents within a Complex Android Environment |
|authors| Mingzhe XingRongkai ZhangHui XueQi ChenFan YangZhen Xiao
|links| http://arxiv.org/abs/2402.06596v1 |
|updated| 2024-02-09 18:19:25 UTC |
|summary| Large language models LLMs have empowered intelligent agents to executeintricate tasks within domain-specific software such as browsers and games.However when applied to general-purpose software systems like operatingsystems LLM agents face three primary challenges. Firstly the action space isvast and dynamic posing difficulties for LLM agents to maintain an up-to-dateunderstanding and deliver accurate responses. Secondly real-world tasks oftenrequire inter-application cooperation demanding farsighted planning from LLMagents. Thirdly agents need to identify optimal solutions aligning with userconstraints such as security concerns and preferences. These challengesmotivate AndroidArena an environment and benchmark designed to evaluate LLMagents on a modern operating system. To address high-cost of manpower wedesign a scalable and semi-automated method to construct the benchmark. In thetask evaluation AndroidArena incorporates accurate and adaptive metrics toaddress the issue of non-unique solutions. Our findings reveal that evenstate-of-the-art LLM agents struggle in cross-APP scenarios and adhering tospecific constraints. Additionally we identify a lack of four keycapabilities i.e. understanding reasoning exploration and reflection asprimary reasons for the failure of LLM agents. Furthermore we provideempirical analysis on the failure of reflection and improve the success rateby 27 with our proposed exploration strategy. This work is the first topresent valuable insights in understanding fine-grained weakness of LLM agentsand offers a path forward for future research in this area. Environmentbenchmark and evaluation code for AndroidArena are released athttps://github.com/AndroidArenaAgent/AndroidArena. |


| Item |Content|
| --- |---|
|idx| 2402.06563v1 |
|title| What is Hiding in Medicine's Dark Matter? Learning with Missing Data in Medical Practices |
|authors| Neslihan SuzenEvgeny M. MirkesDamian RolandJeremy LevesleyAlexander N. GorbanTim J. Coats
|links| http://dx.doi.org/10.1109/BigData59044.2023.10386194 |
|updated| 2024-02-09 17:27:35 UTC |
|summary| Electronic patient records EPRs produce a wealth of data but containsignificant missing information. Understanding and handling this missing datais an important part of clinical data analysis and if left unaddressed couldresult in bias in analysis and distortion in critical conclusions. Missing datamay be linked to health care professional practice patterns and imputation ofmissing data can increase the validity of clinical decisions. This studyfocuses on statistical approaches for understanding and interpreting themissing data and machine learning based clinical data imputation using a singlecentres paediatric emergency data and the data from UKs largest clinicalaudit for traumatic injury database TARN. In the study of 56961 data pointsrelated to initial vital signs and observations taken on children presenting toan Emergency Department we have shown that missing data are likely to benon-random and how these are linked to health care professional practicepatterns. We have then examined 79 TARN fields with missing values for 5791trauma cases. Singular Value Decomposition SVD and k-Nearest Neighbour kNNbased missing data imputation methods are used and imputation results againstthe original dataset are compared and statistically tested. We have concludedthat the 1NN imputer is the best imputation which indicates a usual pattern ofclinical decision making: find the most similar patients and take theirattributes as imputation. |


| Item |Content|
| --- |---|
|idx| 2402.06501v1 |
|title| Scalable Interactive Machine Learning for Future Command and Control |
|authors| Anna MadisonEllen NovosellerVinicius G. GoecksBenjamin T. FilesNicholas WaytowichAlfred YuVernon J. LawhernSteven ThurmanChristopher KelshawKaleb McDowell
|links| http://arxiv.org/abs/2402.06501v1 |
|updated| 2024-02-09 16:11:04 UTC |
|summary| Future warfare will require Command and Control C2 personnel to makedecisions at shrinking timescales in complex and potentially ill-definedsituations. Given the need for robust decision-making processes anddecision-support tools integration of artificial and human intelligence holdsthe potential to revolutionize the C2 operations process to ensure adaptabilityand efficiency in rapidly changing operational environments. We propose toleverage recent promising breakthroughs in interactive machine learning inwhich humans can cooperate with machine learning algorithms to guide machinelearning algorithm behavior. This paper identifies several gaps instate-of-the-art science and technology that future work should address toextend these approaches to function in complex C2 contexts. In particular wedescribe three research focus areas that together aim to enable scalableinteractive machine learning SIML: 1 developing human-AI interactionalgorithms to enable planning in complex dynamic situations 2 fosteringresilient human-AI teams through optimizing roles configurations and trustand 3 scaling algorithms and human-AI teams for flexibility across a range ofpotential contexts and situations. |


| Item |Content|
| --- |---|
|idx| 2402.06472v1 |
|title| "When He Feels Cold, He Goes to the Seahorse"-Blending Generative AI into Multimaterial Storymaking for Family Expressive Arts Therapy |
|authors| Di LiuHanqing ZhouPengcheng An
|links| http://dx.doi.org/10.1145/3613904.3642852 |
|updated| 2024-02-09 15:25:36 UTC |
|summary| Storymaking as an integrative form of expressive arts therapy is aneffective means to foster family communication. Yet the integration ofgenerative AI as expressive materials in therapeutic storymaking remainsunderexplored. And there is a lack of HCI implications on how to supportfamilies and therapists in this context. Addressing this our study involvedfive weeks of storymaking sessions with seven families guided by a professionaltherapist. In these sessions the families used both traditional art-makingmaterials and image-based generative AI to create and evolve their familystories. Via the rich empirical data and commentaries from four experttherapists we contextualize how families creatively melded AI and traditionalexpressive materials to externalize their ideas and feelings. Through the lensof Expressive Therapies Continuum ETC we characterize the therapeuticimplications of AI as expressive materials. Desirable interaction qualities tosupport children parents and therapists are distilled for future HCIresearch. |


| Item |Content|
| --- |---|
|idx| 2402.06421v1 |
|title| What's in People's Digital File Collections? |
|authors| Jesse David DinneenCharles-Antoine Julien
|links| http://dx.doi.org/10.1002/pra2.64 |
|updated| 2024-02-09 14:09:03 UTC |
|summary| Thoughtfully designing services and rigorously testing software to supportpersonal information management PIM requires understanding the relevantcollections but relatively little is known about what people keep in theirfile collections especially personal collections. Complementing recent work onthe structure of 348 file collections we examine those collections contentshow much content is duplicated and how collections used for personal mattersdiffer from those used for study and work. Though all collections contain manyimages some intuitively common file types are surprisingly scarce. Personalcollections contain more audio than others knowledge workers collectionscontain more text documents but far fewer folders and IT collections exhibitunusual traits. Collection duplication is correlated to collections structuraltraits but surprisingly not to collection age. We discuss our findings inlight of prior works and provide implications for various kinds of informationresearch. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2402.06576v2 |
|title| Value-based Resource Matching with Fairness Criteria: Application to Agricultural Water Trading |
|authors| Abhijin AdigaYohai TrabelsiTanvir FerdousiMadhav MaratheS. S. RaviSamarth SwarupAnil Kumar VullikantiMandy L. WilsonSarit KrausReetwika BasuSupriya SavalkarMatthew YourekMichael BradyKirti RajagopalanJonathan Yoder
|links| http://arxiv.org/abs/2402.06576v2 |
|updated| 2024-02-12 02:17:04 UTC |
|summary| Optimal allocation of agricultural water in the event of droughts is animportant global problem. In addressing this problem many aspects includingthe welfare of farmers the economy and the environment must be considered.Under this backdrop our work focuses on several resource-matching problemsaccounting for agents with multi-crop portfolios geographic constraints andfairness. First we address a matching problem where the goal is to maximize awelfare function in two-sided markets where buyers requirements and sellerssupplies are represented by value functions that assign prices or costs tospecified volumes of water. For the setting where the value functions satisfycertain monotonicity properties we present an efficient algorithm thatmaximizes a social welfare function. When there are minimum water requirementconstraints we present a randomized algorithm which ensures that theconstraints are satisfied in expectation. For a single seller--multiple buyerssetting with fairness constraints we design an efficient algorithm thatmaximizes the minimum level of satisfaction of any buyer. We also presentcomputational complexity results that highlight the limits on thegeneralizability of our results. We evaluate the algorithms developed in ourwork with experiments on both real-world and synthetic data sets with respectto drought severity value functions and seniority of agents. |


| Item |Content|
| --- |---|
|idx| 2402.06359v1 |
|title| Modelling Human Values for AI Reasoning |
|authors| Nardine OsmanMark d'Inverno
|links| http://arxiv.org/abs/2402.06359v1 |
|updated| 2024-02-09 12:08:49 UTC |
|summary| One of todays most significant societal challenges is building AI systemswhose behaviour or the behaviour it enables within communities of interactingagents human and artificial aligns with human values. To address thischallenge we detail a formal model of human values for their explicitcomputational representation. To our knowledge this has not been attempted asyet which is surprising given the growing volume of research integratingvalues within AI. Taking as our starting point the wealth of researchinvestigating the nature of human values from social psychology over the lastfew decades we set out to provide such a formal model. We show how this modelcan provide the foundational apparatus for AI-based reasoning over values anddemonstrate its applicability in real-world use cases. We illustrate how ourmodel captures the key ideas from social psychology research and propose aroadmap for future integrated and interdisciplinary research into humanvalues in AI. The ability to automatically reason over values not only helpsaddress the value alignment problem but also facilitates the design of AIsystems that can support individuals and communities in making more informedvalue-aligned decisions. More and more individuals and organisations aremotivated to understand their values more explicitly and explore whether theirbehaviours and attitudes properly reflect them. Our work on modelling humanvalues will enable AI systems to be designed and deployed to meet this growingneed. |


| Item |Content|
| --- |---|
|idx| 2402.06228v1 |
|title| Towards participatory multi-modeling for policy support across domains and scales: a systematic procedure for integral multi-model design |
|authors| Vittorio NespecaRick QuaxMarcel G. M. Olde RikkertHubert P. L. M. KorziliusVincent A. W. J. MarchauSophie HadijsotiriouTom OreelJannie CoenenHeiman WertheimAlexey VoinovEtiënne A. J. A. RouwetteVítor V. Vasconcelos
|links| http://arxiv.org/abs/2402.06228v1 |
|updated| 2024-02-09 07:35:40 UTC |
|summary| Policymaking for complex challenges such as pandemics necessitates theconsideration of intricate implications across multiple domains and scales.Computational models can support policymaking but a single model is ofteninsufficient for such multidomain and scale challenges. Multi-models comprisingseveral interacting computational models at different scales or relying ondifferent modeling paradigms offer a potential solution. Such multi-models canbe assembled from existing computational models i.e. integrated modeling orbe designed conceptually as a whole before their computational implementationi.e. integral modeling. Integral modeling is particularly valuable for novelpolicy problems such as those faced in the early stages of a pandemic whererelevant models may be unavailable or lack standard documentation. Designingsuch multi-models through an integral approach is however a complex taskrequiring the collaboration of modelers and experts from various domains. Inthis collaborative effort modelers must precisely define the domain knowledgeneeded from experts and establish a systematic procedure for translating suchknowledge into a multi-model. Yet these requirements and systematic proceduresare currently lacking for multi-models that are both multiscale andmulti-paradigm. We address this challenge by introducing a procedure fordeveloping multi-models with an integral approach based on clearly defineddomain knowledge requirements derived from literature. We illustrate thisprocedure using the case of school closure policies in the Netherlands duringthe COVID-19 pandemic revealing their potential implications in the short andlong term and across the healthcare and educational domains. The requirementsand procedure provided in this article advance the application of integralmulti-modeling for policy support in multiscale and multidomain contexts. |


| Item |Content|
| --- |---|
|idx| 2402.06176v1 |
|title| Cooperative Nonlinear Guidance Strategies for Guaranteed Pursuit-Evasion |
|authors| Saurabh KumarShashi Ranjan KumarAbhinav Sinha
|links| http://arxiv.org/abs/2402.06176v1 |
|updated| 2024-02-09 04:24:23 UTC |
|summary| This paper addresses the pursuit-evasion problem involving three agents -- apurser an evader and a defender. We develop cooperative guidance laws for theevader-defender team that guarantee that the defender intercepts the pursuerbefore it reaches the vicinity of the evader. Unlike heuristic methods optimalcontrol differential game formulation and recently proposed time-constrainedguidance techniques we propose a geometric solution to safeguard the evaderfrom the pursuers incoming threat. The proposed strategy is computationallyefficient and expected to be scalable as the number of agents increases.Another alluring feature of the proposed strategy is that the evader-defenderteam does not require the knowledge of the pursuers strategy and that thepursuers interception is guaranteed from arbitrary initial engagementgeometries. We further show that the necessary error variables for theevader-defender team vanish within a time that can be exactly prescribed priorto the three-body engagement. Finally we demonstrate the efficacy of theproposed cooperative defense strategy via simulation in diverse engagementscenarios. |


| Item |Content|
| --- |---|
|idx| 2402.06127v1 |
|title| CityFlowER: An Efficient and Realistic Traffic Simulator with Embedded Machine Learning Models |
|authors| Longchao DaChen ChuWeinan ZhangHua Wei
|links| http://arxiv.org/abs/2402.06127v1 |
|updated| 2024-02-09 01:19:41 UTC |
|summary| Traffic simulation is an essential tool for transportation infrastructureplanning intelligent traffic control policy learning and traffic flowanalysis. Its effectiveness relies heavily on the realism of the simulatorsused. Traditional traffic simulators such as SUMO and CityFlow are oftenlimited by their reliance on rule-based models with hyperparameters thatoversimplify driving behaviors resulting in unrealistic simulations. Toenhance realism some simulators have provided Application ProgrammingInterfaces APIs to interact with Machine Learning ML models which learnfrom observed data and offer more sophisticated driving behavior models.However this approach faces challenges in scalability and time efficiency asvehicle numbers increase. Addressing these limitations we introduceCityFlowER an advancement over the existing CityFlow simulator designed forefficient and realistic city-wide traffic simulation. CityFlowER innovativelypre-embeds ML models within the simulator eliminating the need for externalAPI interactions and enabling faster data computation. This approach allows fora blend of rule-based and ML behavior models for individual vehicles offeringunparalleled flexibility and efficiency particularly in large-scalesimulations. We provide detailed comparisons with existing simulatorsimplementation insights and comprehensive experiments to demonstrateCityFlowERs superiority in terms of realism efficiency and adaptability. |


