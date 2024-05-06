# cs.CL 

| Item |Content|
| --- |---|
|idx| 2405.02287v1 |
|title| Vibe-Eval: A hard evaluation suite for measuring progress of multimodal language models |
|authors| Piotr PadlewskiMax BainMatthew HendersonZhongkai ZhuNishant RelanHai PhamDonovan OngKaloyan AleksievAitor OrmazabalSamuel PhuaEthan YeoEugenie LamprechtQi LiuYuqi WangEric ChenDeyu FuLei LiChe ZhengCyprien de Masson d'AutumeDani YogatamaMikel ArtetxeYi Tay
|links| http://arxiv.org/abs/2405.02287v1 |
|updated| 2024-05-03 17:59:55 UTC |
|summary| We introduce Vibe-Eval: a new open benchmark and framework for evaluatingmultimodal chat models. Vibe-Eval consists of 269 visual understanding promptsincluding 100 of hard difficulty complete with gold-standard responsesauthored by experts. Vibe-Eval is open-ended and challenging with dualobjectives: i vibe checking multimodal chat models for day-to-day tasks andii rigorously testing and probing the capabilities of present frontiermodels. Notably our hard set contains 50 questions that all frontier modelsanswer incorrectly. We explore the nuances of designing evaluating andranking models on ultra challenging prompts. We also discuss trade-offs betweenhuman and automatic evaluation and show that automatic model evaluation usingReka Core roughly correlates to human judgment. We offer free API access forthe purpose of lightweight evaluation and plan to conduct formal humanevaluations for public models that perform well on the Vibe-Evals automaticscores. We release the evaluation code and data seehttps://github.com/reka-ai/reka-vibe-eval |


| Item |Content|
| --- |---|
|idx| 2405.02267v1 |
|title| Structural Pruning of Pre-trained Language Models via Neural Architecture Search |
|authors| Aaron KleinJacek GolebiowskiXingchen MaValerio PerroneCedric Archambeau
|links| http://arxiv.org/abs/2405.02267v1 |
|updated| 2024-05-03 17:34:57 UTC |
|summary| Pre-trained language models PLM for example BERT or RoBERTa mark thestate-of-the-art for natural language understanding task when fine-tuned onlabeled data. However their large size poses challenges in deploying them forinference in real-world applications due to significant GPU memoryrequirements and high inference latency. This paper explores neuralarchitecture search NAS for structural pruning to find sub-parts of thefine-tuned network that optimally trade-off efficiency for example in terms ofmodel size or latency and generalization performance. We also show how we canutilize more recently developed two-stage weight-sharing NAS approaches in thissetting to accelerate the search process. Unlike traditional pruning methodswith fixed thresholds we propose to adopt a multi-objective approach thatidentifies the Pareto optimal set of sub-networks allowing for a more flexibleand automated compression process. |


| Item |Content|
| --- |---|
|idx| 2405.02228v1 |
|title| REASONS: A benchmark for REtrieval and Automated citationS Of scieNtific Sentences using Public and Proprietary LLMs |
|authors| Deepa TilwaniYash SaxenaAli MohammadiEdward RaffAmit ShethSrinivasan ParthasarathyManas Gaur
|links| http://arxiv.org/abs/2405.02228v1 |
|updated| 2024-05-03 16:38:51 UTC |
|summary| Automatic citation generation for sentences in a document or report isparamount for intelligence analysts cybersecurity news agencies andeducation personnel. In this research we investigate whether large languagemodels LLMs are capable of generating references based on two forms ofsentence queries: a Direct Queries LLMs are asked to provide author names ofthe given research article and b Indirect Queries LLMs are asked to providethe title of a mentioned article when given a sentence from a differentarticle. To demonstrate where LLM stands in this task we introduce a largedataset called REASONS comprising abstracts of the 12 most popular domains ofscientific research on arXiv. From around 20K research articles we make thefollowing deductions on public and proprietary LLMs: a State-of-the-artoften called anthropomorphic GPT-4 and GPT-3.5 suffers from high passpercentage PP to minimize the hallucination rate HR. When tested withPerplexity.ai 7B they unexpectedly made more errors b Augmenting relevantmetadata lowered the PP and gave the lowest HR c Advance retrieval-augmentedgeneration RAG using Mistral demonstrates consistent and robust citationsupport on indirect queries and matched performance to GPT-3.5 and GPT-4. TheHR across all domains and models decreased by an average of 41.93 and the PPwas reduced to 0 in most cases. In terms of generation quality the average F1Score and BLEU were 68.09 and 57.51 respectively d Testing withadversarial samples showed that LLMs including the Advance RAG Mistralstruggle to understand context but the extent of this issue was small inMistral and GPT-4-Preview. Our study con tributes valuable insights into thereliability of RAG for automated citation generation tasks. |


| Item |Content|
| --- |---|
|idx| 2405.02195v1 |
|title| Impact of emoji exclusion on the performance of Arabic sarcasm detection models |
|authors| Ghalyah H. AleryaniWael DeabesKhaled AlbishreAlaa E. Abdel-Hakim
|links| http://arxiv.org/abs/2405.02195v1 |
|updated| 2024-05-03 15:51:02 UTC |
|summary| The complex challenge of detecting sarcasm in Arabic speech on social mediais increased by the language diversity and the nature of sarcastic expressions.There is a significant gap in the capability of existing models to effectivelyinterpret sarcasm in Arabic which mandates the necessity for moresophisticated and precise detection methods. In this paper we investigate theimpact of a fundamental preprocessing component on sarcasm speech detection.While emojis play a crucial role in mitigating the absence effect of bodylanguage and facial expressions in modern communication their impact onautomated text analysis particularly in sarcasm detection remainsunderexplored. We investigate the impact of emoji exclusion from datasets onthe performance of sarcasm detection models in social media content for Arabicas a vocabulary-super rich language. This investigation includes the adaptationand enhancement of AraBERT pre-training models specifically by excludingemojis to improve sarcasm detection capabilities. We use AraBERT pre-trainingto refine the specified models demonstrating that the removal of emojis cansignificantly boost the accuracy of sarcasm detection. This approachfacilitates a more refined interpretation of language eliminating thepotential confusion introduced by non-textual elements. The evaluated AraBERTmodels through the focused strategy of emoji removal adeptly navigate thecomplexities of Arabic sarcasm. This study establishes new benchmarks in Arabicnatural language processing and presents valuable insights for social mediaplatforms. |


| Item |Content|
| --- |---|
|idx| 2405.02178v1 |
|title| Assessing and Verifying Task Utility in LLM-Powered Applications |
|authors| Negar ArabzadehSiging HuoNikhil MehtaQinqyun WuChi WangAhmed AwadallahCharles L. A. ClarkeJulia Kiseleva
|links| http://arxiv.org/abs/2405.02178v1 |
|updated| 2024-05-03 15:26:27 UTC |
|summary| The rapid development of Large Language Models LLMs has led to a surge inapplications that facilitate collaboration among multiple agents assistinghumans in their daily tasks. However a significant gap remains in assessing towhat extent LLM-powered applications genuinely enhance user experience and taskexecution efficiency. This highlights the need to verify utility of LLM-poweredapplications particularly by ensuring alignment between the applicationsfunctionality and end-user needs. We introduce AgentEval a novel frameworkdesigned to simplify the utility verification process by automaticallyproposing a set of criteria tailored to the unique purpose of any givenapplication. This allows for a comprehensive assessment quantifying theutility of an application against the suggested criteria. We present acomprehensive analysis of the effectiveness and robustness of AgentEval for twoopen source datasets including Math Problem solving and ALFWorld House-holdrelated tasks. For reproducibility purposes we make the data code and all thelogs publicly available at https://bit.ly/3w3yKcS . |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2405.02287v1 |
|title| Vibe-Eval: A hard evaluation suite for measuring progress of multimodal language models |
|authors| Piotr PadlewskiMax BainMatthew HendersonZhongkai ZhuNishant RelanHai PhamDonovan OngKaloyan AleksievAitor OrmazabalSamuel PhuaEthan YeoEugenie LamprechtQi LiuYuqi WangEric ChenDeyu FuLei LiChe ZhengCyprien de Masson d'AutumeDani YogatamaMikel ArtetxeYi Tay
|links| http://arxiv.org/abs/2405.02287v1 |
|updated| 2024-05-03 17:59:55 UTC |
|summary| We introduce Vibe-Eval: a new open benchmark and framework for evaluatingmultimodal chat models. Vibe-Eval consists of 269 visual understanding promptsincluding 100 of hard difficulty complete with gold-standard responsesauthored by experts. Vibe-Eval is open-ended and challenging with dualobjectives: i vibe checking multimodal chat models for day-to-day tasks andii rigorously testing and probing the capabilities of present frontiermodels. Notably our hard set contains 50 questions that all frontier modelsanswer incorrectly. We explore the nuances of designing evaluating andranking models on ultra challenging prompts. We also discuss trade-offs betweenhuman and automatic evaluation and show that automatic model evaluation usingReka Core roughly correlates to human judgment. We offer free API access forthe purpose of lightweight evaluation and plan to conduct formal humanevaluations for public models that perform well on the Vibe-Evals automaticscores. We release the evaluation code and data seehttps://github.com/reka-ai/reka-vibe-eval |


| Item |Content|
| --- |---|
|idx| 2405.02246v1 |
|title| What matters when building vision-language models? |
|authors| Hugo LaurençonLéo TronchonMatthieu CordVictor Sanh
|links| http://arxiv.org/abs/2405.02246v1 |
|updated| 2024-05-03 17:00:00 UTC |
|summary| The growing interest in vision-language models VLMs has been driven byimprovements in large language models and vision transformers. Despite theabundance of literature on this subject we observe that critical decisionsregarding the design of VLMs are often not justified. We argue that theseunsupported decisions impede progress in the field by making it difficult toidentify which choices improve model performance. To address this issue weconduct extensive experiments around pre-trained models architecture choicedata and training methods. Our consolidation of findings includes thedevelopment of Idefics2 an efficient foundational VLM of 8 billion parameters.Idefics2 achieves state-of-the-art performance within its size category acrossvarious multimodal benchmarks and is often on par with models four times itssize. We release the model base instructed and chat along with the datasetscreated for its training. |


| Item |Content|
| --- |---|
|idx| 2405.02228v1 |
|title| REASONS: A benchmark for REtrieval and Automated citationS Of scieNtific Sentences using Public and Proprietary LLMs |
|authors| Deepa TilwaniYash SaxenaAli MohammadiEdward RaffAmit ShethSrinivasan ParthasarathyManas Gaur
|links| http://arxiv.org/abs/2405.02228v1 |
|updated| 2024-05-03 16:38:51 UTC |
|summary| Automatic citation generation for sentences in a document or report isparamount for intelligence analysts cybersecurity news agencies andeducation personnel. In this research we investigate whether large languagemodels LLMs are capable of generating references based on two forms ofsentence queries: a Direct Queries LLMs are asked to provide author names ofthe given research article and b Indirect Queries LLMs are asked to providethe title of a mentioned article when given a sentence from a differentarticle. To demonstrate where LLM stands in this task we introduce a largedataset called REASONS comprising abstracts of the 12 most popular domains ofscientific research on arXiv. From around 20K research articles we make thefollowing deductions on public and proprietary LLMs: a State-of-the-artoften called anthropomorphic GPT-4 and GPT-3.5 suffers from high passpercentage PP to minimize the hallucination rate HR. When tested withPerplexity.ai 7B they unexpectedly made more errors b Augmenting relevantmetadata lowered the PP and gave the lowest HR c Advance retrieval-augmentedgeneration RAG using Mistral demonstrates consistent and robust citationsupport on indirect queries and matched performance to GPT-3.5 and GPT-4. TheHR across all domains and models decreased by an average of 41.93 and the PPwas reduced to 0 in most cases. In terms of generation quality the average F1Score and BLEU were 68.09 and 57.51 respectively d Testing withadversarial samples showed that LLMs including the Advance RAG Mistralstruggle to understand context but the extent of this issue was small inMistral and GPT-4-Preview. Our study con tributes valuable insights into thereliability of RAG for automated citation generation tasks. |


| Item |Content|
| --- |---|
|idx| 2405.02225v1 |
|title| Fair Risk Control: A Generalized Framework for Calibrating Multi-group Fairness Risks |
|authors| Lujing ZhangAaron RothLinjun Zhang
|links| http://arxiv.org/abs/2405.02225v1 |
|updated| 2024-05-03 16:32:09 UTC |
|summary| This paper introduces a framework for post-processing machine learning modelsso that their predictions satisfy multi-group fairness guarantees. Based on thecelebrated notion of multicalibration we introduce mathbfsmathcalGalpha-GMC Generalized Multi-Dimensional Multicalibration formulti-dimensional mappings mathbfs constraint set mathcalG and apre-specified threshold level alpha. We propose associated algorithms toachieve this notion in general settings. This framework is then applied todiverse scenarios encompassing different fairness concerns including falsenegative rate control in image segmentation prediction set conditionaluncertainty quantification in hierarchical classification and de-biased textgeneration in language models. We conduct numerical studies on several datasetsand tasks. |


| Item |Content|
| --- |---|
|idx| 2405.02213v1 |
|title| Automatic Programming: Large Language Models and Beyond |
|authors| Michael R. LyuBaishakhi RayAbhik RoychoudhuryShin Hwei TanPatanamon Thongtanunam
|links| http://arxiv.org/abs/2405.02213v1 |
|updated| 2024-05-03 16:19:24 UTC |
|summary| Automatic programming has seen increasing popularity due to the emergence oftools like GitHub Copilot which rely on Large Language Models LLMs. At thesame time automatically generated code faces challenges during deployment dueto concerns around quality and trust. In this article we study automatedcoding in a general sense and study the concerns around code quality securityand related issues of programmer responsibility. These are key issues fororganizations while deciding on the usage of automatically generated code. Wediscuss how advances in software engineering such as program repair andanalysis can enable automatic programming. We conclude with a forward lookingview focusing on the programming environment of the near future whereprogrammers may need to switch to different roles to fully utilize the power ofautomatic programming. Automated repair of automatically generated programsfrom LLMs can help produce higher assurance code from LLMs along withevidence of assurance |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2405.02267v1 |
|title| Structural Pruning of Pre-trained Language Models via Neural Architecture Search |
|authors| Aaron KleinJacek GolebiowskiXingchen MaValerio PerroneCedric Archambeau
|links| http://arxiv.org/abs/2405.02267v1 |
|updated| 2024-05-03 17:34:57 UTC |
|summary| Pre-trained language models PLM for example BERT or RoBERTa mark thestate-of-the-art for natural language understanding task when fine-tuned onlabeled data. However their large size poses challenges in deploying them forinference in real-world applications due to significant GPU memoryrequirements and high inference latency. This paper explores neuralarchitecture search NAS for structural pruning to find sub-parts of thefine-tuned network that optimally trade-off efficiency for example in terms ofmodel size or latency and generalization performance. We also show how we canutilize more recently developed two-stage weight-sharing NAS approaches in thissetting to accelerate the search process. Unlike traditional pruning methodswith fixed thresholds we propose to adopt a multi-objective approach thatidentifies the Pareto optimal set of sub-networks allowing for a more flexibleand automated compression process. |


| Item |Content|
| --- |---|
|idx| 2405.02240v1 |
|title| Subgraph2vec: A random walk-based algorithm for embedding knowledge graphs |
|authors| Elika BozorgiSaber SoleimaniSakher Khalil AlqaiidiHamid Reza ArabniaKrzysztof Kochut
|links| http://arxiv.org/abs/2405.02240v1 |
|updated| 2024-05-03 16:51:18 UTC |
|summary| Graph is an important data representation which occurs naturally in the realworld applications citegoyal2018graph. Therefore analyzing graphs providesusers with better insights in different areas such as anomaly detectioncitema2021comprehensive decision making citefan2023graph clusteringcitetsitsulin2023graph classification citewang2021mixup and etc.However most of these methods require high levels of computational time andspace. We can use other ways like embedding to reduce these costs. Knowledgegraph KG embedding is a technique that aims to achieve the vectorrepresentation of a KG. It represents entities and relations of a KG in alow-dimensional space while maintaining the semantic meanings of them. Thereare different methods for embedding graphs including random walk-based methodssuch as node2vec metapath2vec and regpattern2vec. However most of thesemethods bias the walks based on a rigid pattern usually hard-coded in thealgorithm. In this work we introduce textitsubgraph2vec for embedding KGswhere walks are run inside a user-defined subgraph. We use this embedding forlink prediction and prove our method has better performance in most cases incomparison with the previous ones. |


| Item |Content|
| --- |---|
|idx| 2405.02235v1 |
|title| Learning Optimal Deterministic Policies with Stochastic Policy Gradients |
|authors| Alessandro MontenegroMarco MussiAlberto Maria MetelliMatteo Papini
|links| http://arxiv.org/abs/2405.02235v1 |
|updated| 2024-05-03 16:45:15 UTC |
|summary| Policy gradient PG methods are successful approaches to deal withcontinuous reinforcement learning RL problems. They learn stochasticparametric hyperpolicies by either exploring in the space of actions or inthe space of parameters. Stochastic controllers however are often undesirablefrom a practical perspective because of their lack of robustness safety andtraceability. In common practice stochastic hyperpolicies are learned onlyto deploy their deterministic version. In this paper we make a step towardsthe theoretical understanding of this practice. After introducing a novelframework for modeling this scenario we study the global convergence to thebest deterministic policy under weak gradient domination assumptions. Thenwe illustrate how to tune the exploration level used for learning to optimizethe trade-off between the sample complexity and the performance of the deployeddeterministic policy. Finally we quantitatively compare action-based andparameter-based exploration giving a formal guise to intuitive results. |


| Item |Content|
| --- |---|
|idx| 2405.02225v1 |
|title| Fair Risk Control: A Generalized Framework for Calibrating Multi-group Fairness Risks |
|authors| Lujing ZhangAaron RothLinjun Zhang
|links| http://arxiv.org/abs/2405.02225v1 |
|updated| 2024-05-03 16:32:09 UTC |
|summary| This paper introduces a framework for post-processing machine learning modelsso that their predictions satisfy multi-group fairness guarantees. Based on thecelebrated notion of multicalibration we introduce mathbfsmathcalGalpha-GMC Generalized Multi-Dimensional Multicalibration formulti-dimensional mappings mathbfs constraint set mathcalG and apre-specified threshold level alpha. We propose associated algorithms toachieve this notion in general settings. This framework is then applied todiverse scenarios encompassing different fairness concerns including falsenegative rate control in image segmentation prediction set conditionaluncertainty quantification in hierarchical classification and de-biased textgeneration in language models. We conduct numerical studies on several datasetsand tasks. |


| Item |Content|
| --- |---|
|idx| 2405.02221v1 |
|title| Discretization Error of Fourier Neural Operators |
|authors| Samuel LanthalerAndrew M. StuartMargaret Trautner
|links| http://arxiv.org/abs/2405.02221v1 |
|updated| 2024-05-03 16:28:05 UTC |
|summary| Operator learning is a variant of machine learning that is designed toapproximate maps between function spaces from data. The Fourier Neural OperatorFNO is a common model architecture used for operator learning. The FNOcombines pointwise linear and nonlinear operations in physical space withpointwise linear operations in Fourier space leading to a parameterized mapacting between function spaces. Although FNOs formally involve convolutions offunctions on a continuum in practice the computations are performed on adiscretized grid allowing efficient implementation via the FFT. In this paperthe aliasing error that results from such a discretization is quantified andalgebraic rates of convergence in terms of the grid resolution are obtained asa function of the regularity of the input. Numerical experiments that validatethe theory and describe model stability are performed. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2405.02287v1 |
|title| Vibe-Eval: A hard evaluation suite for measuring progress of multimodal language models |
|authors| Piotr PadlewskiMax BainMatthew HendersonZhongkai ZhuNishant RelanHai PhamDonovan OngKaloyan AleksievAitor OrmazabalSamuel PhuaEthan YeoEugenie LamprechtQi LiuYuqi WangEric ChenDeyu FuLei LiChe ZhengCyprien de Masson d'AutumeDani YogatamaMikel ArtetxeYi Tay
|links| http://arxiv.org/abs/2405.02287v1 |
|updated| 2024-05-03 17:59:55 UTC |
|summary| We introduce Vibe-Eval: a new open benchmark and framework for evaluatingmultimodal chat models. Vibe-Eval consists of 269 visual understanding promptsincluding 100 of hard difficulty complete with gold-standard responsesauthored by experts. Vibe-Eval is open-ended and challenging with dualobjectives: i vibe checking multimodal chat models for day-to-day tasks andii rigorously testing and probing the capabilities of present frontiermodels. Notably our hard set contains 50 questions that all frontier modelsanswer incorrectly. We explore the nuances of designing evaluating andranking models on ultra challenging prompts. We also discuss trade-offs betweenhuman and automatic evaluation and show that automatic model evaluation usingReka Core roughly correlates to human judgment. We offer free API access forthe purpose of lightweight evaluation and plan to conduct formal humanevaluations for public models that perform well on the Vibe-Evals automaticscores. We release the evaluation code and data seehttps://github.com/reka-ai/reka-vibe-eval |


| Item |Content|
| --- |---|
|idx| 2405.02280v1 |
|title| DreamScene4D: Dynamic Multi-Object Scene Generation from Monocular Videos |
|authors| Wen-Hsuan ChuLei KeKaterina Fragkiadaki
|links| http://arxiv.org/abs/2405.02280v1 |
|updated| 2024-05-03 17:55:34 UTC |
|summary| Existing VLMs can track in-the-wild 2D video objects while current generativemodels provide powerful visual priors for synthesizing novel views for thehighly under-constrained 2D-to-3D object lifting. Building upon this excitingprogress we present DreamScene4D the first approach that can generatethree-dimensional dynamic scenes of multiple objects from monocular in-the-wildvideos with large object motion across occlusions and novel viewpoints. Our keyinsight is to design a decompose-then-recompose scheme to factorize both thewhole video scene and each objects 3D motion. We first decompose the videoscene by using open-vocabulary mask trackers and an adapted image diffusionmodel to segment track and amodally complete the objects and background inthe video. Each object track is mapped to a set of 3D Gaussians that deform andmove in space and time. We also factorize the observed motion into multiplecomponents to handle fast motion. The camera motion can be inferred byre-rendering the background to match the video frames. For the object motionwe first model the object-centric deformation of the objects by leveragingrendering losses and multi-view generative priors in an object-centric framethen optimize object-centric to world-frame transformations by comparing therendered outputs against the perceived pixel and optical flow. Finally werecompose the background and objects and optimize for relative object scalesusing monocular depth prediction guidance. We show extensive results on thechallenging DAVIS Kubric and self-captured videos detail some limitationsand provide future directions. Besides 4D scene generation our results showthat DreamScene4D enables accurate 2D point motion tracking by projecting theinferred 3D trajectories to 2D while never explicitly trained to do so. |


| Item |Content|
| --- |---|
|idx| 2405.02266v1 |
|title| On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning? |
|authors| Maxime ZanellaIsmail Ben Ayed
|links| http://arxiv.org/abs/2405.02266v1 |
|updated| 2024-05-03 17:34:02 UTC |
|summary| The development of large vision-language models notably CLIP has catalyzedresearch into effective adaptation techniques with a particular focus on softprompt tuning. Conjointly test-time augmentation which utilizes multipleaugmented views of a single image to enhance zero-shot generalization isemerging as a significant area of interest. This has predominantly directedresearch efforts toward test-time prompt tuning. In contrast we introduce arobust MeanShift for Test-time Augmentation MTA which surpasses prompt-basedmethods without requiring this intensive training procedure. This positions MTAas an ideal solution for both standalone and API-based applications.Additionally our method does not rely on ad hoc rules e.g. confidencethreshold used in some previous test-time augmentation techniques to filterthe augmented views. Instead MTA incorporates a quality assessment variablefor each view directly into its optimization process termed as the inliernessscore. This score is jointly optimized with a density mode seeking processleading to an efficient training- and hyperparameter-free approach. Weextensively benchmark our method on 15 datasets and demonstrate MTAssuperiority and computational efficiency. Deployed easily as plug-and-playmodule on top of zero-shot models and state-of-the-art few-shot methods MTAshows systematic and consistent improvements. |


| Item |Content|
| --- |---|
|idx| 2405.02246v1 |
|title| What matters when building vision-language models? |
|authors| Hugo LaurençonLéo TronchonMatthieu CordVictor Sanh
|links| http://arxiv.org/abs/2405.02246v1 |
|updated| 2024-05-03 17:00:00 UTC |
|summary| The growing interest in vision-language models VLMs has been driven byimprovements in large language models and vision transformers. Despite theabundance of literature on this subject we observe that critical decisionsregarding the design of VLMs are often not justified. We argue that theseunsupported decisions impede progress in the field by making it difficult toidentify which choices improve model performance. To address this issue weconduct extensive experiments around pre-trained models architecture choicedata and training methods. Our consolidation of findings includes thedevelopment of Idefics2 an efficient foundational VLM of 8 billion parameters.Idefics2 achieves state-of-the-art performance within its size category acrossvarious multimodal benchmarks and is often on par with models four times itssize. We release the model base instructed and chat along with the datasetscreated for its training. |


| Item |Content|
| --- |---|
|idx| 2405.02220v1 |
|title| Designed Dithering Sign Activation for Binary Neural Networks |
|authors| Brayan MonroyJuan EstupiñanTatiana Gelvez-BarreraJorge BaccaHenry Arguello
|links| http://arxiv.org/abs/2405.02220v1 |
|updated| 2024-05-03 16:27:39 UTC |
|summary| Binary Neural Networks emerged as a cost-effective and energy-efficientsolution for computer vision tasks by binarizing either network weights oractivations. However common binary activations such as the Sign activationfunction abruptly binarize the values with a single threshold losingfine-grained details in the feature outputs. This work proposes an activationthat applies multiple thresholds following dithering principles shifting theSign activation function for each pixel according to a spatially periodicthreshold kernel. Unlike literature methods the shifting is defined jointlyfor a set of adjacent pixels taking advantage of spatial correlations.Experiments over the classification task demonstrate the effectiveness of thedesigned dithering Sign activation function as an alternative activation forbinary neural networks without increasing the computational cost. FurtherDeSign balances the preservation of details with the efficiency of binaryoperations. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2405.02225v1 |
|title| Fair Risk Control: A Generalized Framework for Calibrating Multi-group Fairness Risks |
|authors| Lujing ZhangAaron RothLinjun Zhang
|links| http://arxiv.org/abs/2405.02225v1 |
|updated| 2024-05-03 16:32:09 UTC |
|summary| This paper introduces a framework for post-processing machine learning modelsso that their predictions satisfy multi-group fairness guarantees. Based on thecelebrated notion of multicalibration we introduce mathbfsmathcalGalpha-GMC Generalized Multi-Dimensional Multicalibration formulti-dimensional mappings mathbfs constraint set mathcalG and apre-specified threshold level alpha. We propose associated algorithms toachieve this notion in general settings. This framework is then applied todiverse scenarios encompassing different fairness concerns including falsenegative rate control in image segmentation prediction set conditionaluncertainty quantification in hierarchical classification and de-biased textgeneration in language models. We conduct numerical studies on several datasetsand tasks. |


| Item |Content|
| --- |---|
|idx| 2405.02200v1 |
|title| Position Paper: Rethinking Empirical Research in Machine Learning: Addressing Epistemic and Methodological Challenges of Experimentation |
|authors| Moritz HerrmannF. Julian D. LangeKatharina EggenspergerGiuseppe CasalicchioMarcel WeverMatthias FeurerDavid RügamerEyke HüllermeierAnne-Laure BoulesteixBernd Bischl
|links| http://arxiv.org/abs/2405.02200v1 |
|updated| 2024-05-03 15:57:22 UTC |
|summary| We warn against a common but incomplete understanding of empirical researchin machine learning ML that leads to non-replicable results makes findingsunreliable and threatens to undermine progress in the field. To overcome thisalarming situation we call for more awareness of the plurality of ways ofgaining knowledge experimentally but also of some epistemic limitations. Inparticular we argue most current empirical ML research is fashioned asconfirmatory research while it should rather be considered exploratory. |


| Item |Content|
| --- |---|
|idx| 2405.02188v1 |
|title| Optimistic Regret Bounds for Online Learning in Adversarial Markov Decision Processes |
|authors| Sang Bin MoonAbolfazl Hashemi
|links| http://arxiv.org/abs/2405.02188v1 |
|updated| 2024-05-03 15:44:31 UTC |
|summary| The Adversarial Markov Decision Process AMDP is a learning framework thatdeals with unknown and varying tasks in decision-making applications likerobotics and recommendation systems. A major limitation of the AMDP formalismhowever is pessimistic regret analysis results in the sense that although thecost function can change from one episode to the next the evolution in manysettings is not adversarial. To address this we introduce and study a newvariant of AMDP which aims to minimize regret while utilizing a set of costpredictors. For this setting we develop a new policy search method thatachieves a sublinear optimistic regret with high probability that is a regretbound which gracefully degrades with the estimation power of the costpredictors. Establishing such optimistic regret bounds is nontrivial given thati as we demonstrate the existing importance-weighted cost estimators cannotestablish optimistic bounds and ii the feedback model of AMDP is differentand more realistic than the existing optimistic online learning works. Ourresult in particular hinges upon developing a novel optimistically biasedcost estimator that leverages cost predictors and enables a high-probabilityregret analysis without imposing restrictive assumptions. We further discusspractical extensions of the proposed scheme and demonstrate its efficacynumerically. |


| Item |Content|
| --- |---|
|idx| 2405.02183v1 |
|title| Metalearners for Ranking Treatment Effects |
|authors| Toon VanderschuerenWouter VerbekeFelipe MoraesHugo Manuel Proença
|links| http://arxiv.org/abs/2405.02183v1 |
|updated| 2024-05-03 15:31:18 UTC |
|summary| Efficiently allocating treatments with a budget constraint constitutes animportant challenge across various domains. In marketing for example the useof promotions to target potential customers and boost conversions is limited bythe available budget. While much research focuses on estimating causal effectsthere is relatively limited work on learning to allocate treatments whileconsidering the operational context. Existing methods for uplift modeling orcausal inference primarily estimate treatment effects without considering howthis relates to a profit maximizing allocation policy that respects budgetconstraints. The potential downside of using these methods is that theresulting predictive model is not aligned with the operational context.Therefore prediction errors are propagated to the optimization of the budgetallocation problem subsequently leading to a suboptimal allocation policy. Wepropose an alternative approach based on learning to rank. Our proposedmethodology directly learns an allocation policy by prioritizing instances interms of their incremental profit. We propose an efficient sampling procedurefor the optimization of the ranking model to scale our methodology tolarge-scale data sets. Theoretically we show how learning to rank can maximizethe area under a policys incremental profit curve. Empirically we validateour methodology and show its effectiveness in practice through a series ofexperiments on both synthetic and real-world data. |


| Item |Content|
| --- |---|
|idx| 2405.02140v1 |
|title| An Information Theoretic Perspective on Conformal Prediction |
|authors| Alvaro H. C. CorreiaFabio Valerio MassoliChristos LouizosArash Behboodi
|links| http://arxiv.org/abs/2405.02140v1 |
|updated| 2024-05-03 14:43:07 UTC |
|summary| Conformal Prediction CP is a distribution-free uncertainty estimationframework that constructs prediction sets guaranteed to contain the true answerwith a user-specified probability. Intuitively the size of the prediction setencodes a general notion of uncertainty with larger sets associated withhigher degrees of uncertainty. In this work we leverage information theory toconnect conformal prediction to other notions of uncertainty. More preciselywe prove three different ways to upper bound the intrinsic uncertainty asdescribed by the conditional entropy of the target variable given the inputsby combining CP with information theoretical inequalities. Moreover wedemonstrate two direct and useful applications of such connection betweenconformal prediction and information theory: i more principled and effectiveconformal training objectives that generalize previous approaches and enableend-to-end training of machine learning models from scratch and ii a naturalmechanism to incorporate side information into conformal prediction. Weempirically validate both applications in centralized and federated learningsettings showing our theoretical results translate to lower inefficiencyaverage prediction set size for popular CP methods. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2405.02260v1 |
|title| Leveraging Large Language Models to Enhance Domain Expert Inclusion in Data Science Workflows |
|authors| Jasmine Y. ShihVishal MohantyYannis KatsisHariharan Subramonyam
|links| http://arxiv.org/abs/2405.02260v1 |
|updated| 2024-05-03 17:22:15 UTC |
|summary| Domain experts can play a crucial role in guiding data scientists to optimizemachine learning models while ensuring contextual relevance for downstream use.However in current workflows such collaboration is challenging due todiffering expertise abstract documentation practices and lack of access andvisibility into low-level implementation artifacts. To address these challengesand enable domain expert participation we introduce CellSync a collaborationframework comprising 1 a Jupyter Notebook extension that continuously trackschanges to dataframes and model metrics and 2 a Large Language Model poweredvisualization dashboard that makes those changes interpretable to domainexperts. Through CellSyncs cell-level dataset visualization with codesummaries domain experts can interactively examine how individual data andmodeling operations impact different data segments. The chat features enabledata-centric conversations and targeted feedback to data scientists. Ourpreliminary evaluation shows that CellSync provides transparency and promotescritical discussions about the intents and implications of data operations. |


| Item |Content|
| --- |---|
|idx| 2405.02229v1 |
|title| On the Utility of External Agent Intention Predictor for Human-AI Coordination |
|authors| Chenxu WangZilong ChenAngelo CangelosiHuaping Liu
|links| http://arxiv.org/abs/2405.02229v1 |
|updated| 2024-05-03 16:39:20 UTC |
|summary| Reaching a consensus on the team plans is vital to human-AI coordination.Although previous studies provide approaches through communications in variousways it could still be hard to coordinate when the AI has no explainable planto communicate. To cover this gap we suggest incorporating external models toassist humans in understanding the intentions of AI agents. In this paper wepropose a two-stage paradigm that first trains a Theory of Mind ToM modelfrom collected offline trajectories of the target agent and utilizes the modelin the process of human-AI collaboration by real-timely displaying the futureaction predictions of the target agent. Such a paradigm leaves the AI agent asa black box and thus is available for improving any agents. To test ourparadigm we further implement a transformer-based predictor as the ToM modeland develop an extended online human-AI collaboration platform for experiments.The comprehensive experimental results verify that human-AI teams can achievebetter performance with the help of our model. A user assessment attached tothe experiment further demonstrates that our paradigm can significantly enhancethe situational awareness of humans. Our study presents the potential toaugment the ability of humans via external assistance in human-AIcollaboration which may further inspire future research. |


| Item |Content|
| --- |---|
|idx| 2405.02173v1 |
|title| Task Synthesis for Elementary Visual Programming in XLogoOnline Environment |
|authors| Chao WenAhana GhoshJacqueline StaubAdish Singla
|links| http://arxiv.org/abs/2405.02173v1 |
|updated| 2024-05-03 15:22:46 UTC |
|summary| In recent years the XLogoOnline programming platform has gained popularityamong novice learners. It integrates the Logo programming language with visualprogramming providing a visual interface for learning computing concepts.However XLogoOnline offers only a limited set of tasks which are inadequatefor learners to master the computing concepts that require sufficient practice.To address this we introduce XLogoSyn a novel technique for synthesizinghigh-quality tasks for varying difficulty levels. Given a reference taskXLogoSyn can generate practice tasks at varying difficulty levels that cater tothe varied needs and abilities of different learners. XLogoSyn achieves this bycombining symbolic execution and constraint satisfaction techniques. Our expertstudy demonstrates the effectiveness of XLogoSyn. We have also deployedsynthesized practice tasks into XLogoOnline highlighting the educationalbenefits of these synthesized practice tasks. |


| Item |Content|
| --- |---|
|idx| 2405.02045v1 |
|title| Are We in The Zone? Exploring The Features and Method of Detecting Simultaneous Flow Experiences Based on EEG Signals |
|authors| Baiqiao ZhangXiangxian LiYunfan ZhouJuan LiuWeiying LiuChao ZhouYulong Bian
|links| http://arxiv.org/abs/2405.02045v1 |
|updated| 2024-05-03 12:22:35 UTC |
|summary| When executing interdependent personal tasks for the teams purposesimultaneous individual flowsimultaneous flow is the antecedent condition ofachieving shared team flow. Detecting simultaneous flow helps betterunderstanding the status of team members which is thus important foroptimizing multi-user interaction systems. However there is currently a lackexploration on objective features and methods for detecting simultaneous flow.Based on brain mechanism of flow in teamwork and previous studies onelectroencephalogram EEG-based individual flow detection this study aims toexplore the significant EEG features related to simultaneous flow as well aseffective detection methods based on EEG signals. First a two-playersimultaneous flow task is designed based on which we construct the firstmulti-EEG signals dataset of simultaneous flow. Then we explore the potentialEEG signal features that may be related to individual and simultaneous flow andvalidate their effectiveness in simultaneous flow detection with variousmachine learning models. The results show that 1 the inter-brain synchronyfeatures are relevant to simultaneous flow due to enhancing the modelsperformance in detecting different types of simultaneous flow 2 the featuresfrom the frontal lobe area seem to be given priority attention when detectingsimultaneous flows 3 Random Forests performed best in binary classificationwhile Neural Network and Deep Neural Network3 performed best in ternaryclassification. |


| Item |Content|
| --- |---|
|idx| 2405.02016v1 |
|title| Adversarial Botometer: Adversarial Analysis for Social Bot Detection |
|authors| Shaghayegh NajariDavood RafieeMostafa SalehiReza Farahbakhsh
|links| http://arxiv.org/abs/2405.02016v1 |
|updated| 2024-05-03 11:28:21 UTC |
|summary| Social bots play a significant role in many online social networks OSN asthey imitate human behavior. This fact raises difficult questions about theircapabilities and potential risks. Given the recent advances in Generative AIGenAI social bots are capable of producing highly realistic and complexcontent that mimics human creativity. As the malicious social bots emerge todeceive people with their unrealistic content identifying them anddistinguishing the content they produce has become an actual challenge fornumerous social platforms. Several approaches to this problem have already beenproposed in the literature but the proposed solutions have not been widelyevaluated. To address this issue we evaluate the behavior of a text-based botdetector in a competitive environment where some scenarios are proposed:textitFirst the tug-of-war between a bot and a bot detector is examined. Itis interesting to analyze which party is more likely to prevail and whichcircumstances influence these expectations. In this regard we model theproblem as a synthetic adversarial game in which a conversational bot and a botdetector are engaged in strategic online interactions. textitSecond the botdetection model is evaluated under attack examples generated by a social botto this end we poison the dataset with attack examples and evaluate the modelperformance under this condition. textitFinally to investigate the impactof the dataset a cross-domain analysis is performed. Through our comprehensiveevaluation of different categories of social bots using two benchmark datasetswe were able to demonstrate some achivement that could be utilized in futureworks. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2405.02229v1 |
|title| On the Utility of External Agent Intention Predictor for Human-AI Coordination |
|authors| Chenxu WangZilong ChenAngelo CangelosiHuaping Liu
|links| http://arxiv.org/abs/2405.02229v1 |
|updated| 2024-05-03 16:39:20 UTC |
|summary| Reaching a consensus on the team plans is vital to human-AI coordination.Although previous studies provide approaches through communications in variousways it could still be hard to coordinate when the AI has no explainable planto communicate. To cover this gap we suggest incorporating external models toassist humans in understanding the intentions of AI agents. In this paper wepropose a two-stage paradigm that first trains a Theory of Mind ToM modelfrom collected offline trajectories of the target agent and utilizes the modelin the process of human-AI collaboration by real-timely displaying the futureaction predictions of the target agent. Such a paradigm leaves the AI agent asa black box and thus is available for improving any agents. To test ourparadigm we further implement a transformer-based predictor as the ToM modeland develop an extended online human-AI collaboration platform for experiments.The comprehensive experimental results verify that human-AI teams can achievebetter performance with the help of our model. A user assessment attached tothe experiment further demonstrates that our paradigm can significantly enhancethe situational awareness of humans. Our study presents the potential toaugment the ability of humans via external assistance in human-AIcollaboration which may further inspire future research. |


| Item |Content|
| --- |---|
|idx| 2405.02198v1 |
|title| The Cambridge RoboMaster: An Agile Multi-Robot Research Platform |
|authors| Jan BlumenkampAjay ShankarMatteo BettiniJoshua BirdAmanda Prorok
|links| http://arxiv.org/abs/2405.02198v1 |
|updated| 2024-05-03 15:54:20 UTC |
|summary| Compact robotic platforms with powerful compute and actuation capabilitiesare key enablers for practical real-world deployments of multi-agent research.This article introduces a tightly integrated hardware control and simulationsoftware stack on a fleet of holonomic ground robot platforms designed withthis motivation. Our robots a fleet of customised DJI Robomaster S1 vehiclesoffer a balance between small robots that do not possess sufficient compute oractuation capabilities and larger robots that are unsuitable for indoormulti-robot tests. They run a modular ROS2-based optimal estimation and controlstack for full onboard autonomy contain ad-hoc peer-to-peer communicationinfrastructure and can zero-shot run multi-agent reinforcement learning MARLpolicies trained in our vectorized multi-agent simulation framework. We presentan in-depth review of other platforms currently available showcase newexperimental validation of our systems capabilities and introduce casestudies that highlight the versatility and reliabilty of our system as atestbed for a wide range of research demonstrations. Our system as well assupplementary material is available online:https://proroklab.github.io/cambridge-robomaster |


| Item |Content|
| --- |---|
|idx| 2405.02161v1 |
|title| Simulating the economic impact of rationality through reinforcement learning and agent-based modelling |
|authors| Simone BrusatinTommaso PadoanAndrea ColettaDomenico Delli GattiAldo Glielmo
|links| http://arxiv.org/abs/2405.02161v1 |
|updated| 2024-05-03 15:08:25 UTC |
|summary| Agent-based models ABMs are simulation models used in economics to overcomesome of the limitations of traditional frameworks based on general equilibriumassumptions. However agents within an ABM follow predetermined not fullyrational behavioural rules which can be cumbersome to design and difficult tojustify. Here we leverage multi-agent reinforcement learning RL to expand thecapabilities of ABMs with the introduction of fully rational agents that learntheir policy by interacting with the environment and maximising a rewardfunction. Specifically we propose a Rational macro ABM R-MABM framework byextending a paradigmatic macro ABM from the economic literature. We show thatgradually substituting ABM firms in the model with RL agents trained tomaximise profits allows for a thorough study of the impact of rationality onthe economy. We find that RL agents spontaneously learn three distinctstrategies for maximising profits with the optimal strategy depending on thelevel of market competition and rationality. We also find that RL agents withindependent policies and without the ability to communicate with each otherspontaneously learn to segregate into different strategic groups thusincreasing market power and overall profits. Finally we find that a higherdegree of rationality in the economy always improves the macroeconomicenvironment as measured by total output depending on the specific rationalpolicy this can come at the cost of higher instability. Our R-MABM frameworkis general it allows for stable multi-agent learning and represents aprincipled and robust direction to extend existing economic simulators. |


| Item |Content|
| --- |---|
|idx| 2405.02133v1 |
|title| Learning from Evolution: Improving Collective Decision-Making Mechanisms using Insights from Evolutionary Robotics |
|authors| Tanja Katharina Kaiser
|links| http://dx.doi.org/10.1145/3638529.3653988 |
|updated| 2024-05-03 14:37:17 UTC |
|summary| Collective decision-making enables multi-robot systems to act autonomously inreal-world environments. Existing collective decision-making mechanisms sufferfrom the so-called speed versus accuracy trade-off or rely on high complexitye.g. by including global communication. Recent work has shown that moreefficient collective decision-making mechanisms based on artificial neuralnetworks can be generated using methods from evolutionary computation. A majordrawback of these decision-making neural networks is their limitedinterpretability. Analyzing evolved decision-making mechanisms can help usimprove the efficiency of hand-coded decision-making mechanisms whilemaintaining a higher interpretability. In this paper we analyze evolvedcollective decision-making mechanisms in detail and hand-code two newdecision-making mechanisms based on the insights gained. In benchmarkexperiments we show that the newly implemented collective decision-makingmechanisms are more efficient than the state-of-the-art collectivedecision-making mechanisms voter model and majority rule. |


| Item |Content|
| --- |---|
|idx| 2405.01870v1 |
|title| Detecting and Deterring Manipulation in a Cognitive Hierarchy |
|authors| Nitay AlonLion SchulzJoseph M. BarnbyJeffrey S. RosenscheinPeter Dayan
|links| http://arxiv.org/abs/2405.01870v1 |
|updated| 2024-05-03 05:53:09 UTC |
|summary| Social agents with finitely nested opponent models are vulnerable tomanipulation by agents with deeper reasoning and more sophisticated opponentmodelling. This imbalance rooted in logic and the theory of recursivemodelling frameworks cannot be solved directly. We propose a computationalframework aleph-IPOMDP augmenting model-based RL agents Bayesianinference with an anomaly detection algorithm and an out-of-belief policy. Ourmechanism allows agents to realize they are being deceived even if they cannotunderstand how and to deter opponents via a credible threat. We test thisframework in both a mixed-motive and zero-sum game. Our results show thealeph mechanisms effectiveness leading to more equitable outcomes and lessexploitation by more sophisticated agents. We discuss implications for AIsafety cybersecurity cognitive science and psychiatry. |


