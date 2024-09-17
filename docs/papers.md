# cs.CL 

| Item |Content|
| --- |---|
|idx| 2409.09030v1 |
|title| Agents in Software Engineering: Survey, Landscape, and Vision |
|authors| Yanxian HuangWanjun ZhongEnsheng ShiMin YangJiachi ChenHui LiYuchi MaQianxiang WangZibin ZhengYanlin Wang
|links| http://arxiv.org/abs/2409.09030v1 |
|updated| 2024-09-13 17:55:58 UTC |
|summary| In recent years Large Language Models LLMs have achieved remarkablesuccess and have been widely used in various downstream tasks especially inthe tasks of the software engineering SE field. We find that many studiescombining LLMs with SE have employed the concept of agents either explicitly orimplicitly. However there is a lack of an in-depth survey to sort out thedevelopment context of existing works analyze how existing works combine theLLM-based agent technologies to optimize various tasks and clarify theframework of LLM-based agents in SE. In this paper we conduct the first surveyof the studies on combining LLM-based agents with SE and present a framework ofLLM-based agents in SE which includes three key modules: perception memoryand action. We also summarize the current challenges in combining the twofields and propose future opportunities in response to existing challenges. Wemaintain a GitHub repository of the related papers at:https://github.com/DeepSoftwareAnalytics/Awesome-Agent4SE. |


| Item |Content|
| --- |---|
|idx| 2409.09013v1 |
|title| AI-LieDar: Examine the Trade-off Between Utility and Truthfulness in LLM Agents |
|authors| Zhe SuXuhui ZhouSanketh RangrejiAnubha KabraJulia MendelsohnFaeze BrahmanMaarten Sap
|links| http://arxiv.org/abs/2409.09013v1 |
|updated| 2024-09-13 17:41:12 UTC |
|summary| To be safely and successfully deployed LLMs must simultaneously satisfytruthfulness and utility goals. Yet often these two goals compete e.g. an AIagent assisting a used car salesman selling a car with flaws partly due toambiguous or misleading user instructions. We propose AI-LieDar a framework tostudy how LLM-based agents navigate scenarios with utility-truthfulnessconflicts in a multi-turn interactive setting. We design a set of realisticscenarios where language agents are instructed to achieve goals that are inconflict with being truthful during a multi-turn conversation with simulatedhuman agents. To evaluate the truthfulness at large scale we develop atruthfulness detector inspired by psychological literature to assess theagents responses. Our experiment demonstrates that all models are truthfulless than 50 of the time although truthfulness and goal achievement utilityrates vary across models. We further test the steerability of LLMs towardstruthfulness finding that models follow malicious instructions to deceive andeven truth-steered models can still lie. These findings reveal the complexnature of truthfulness in LLMs and underscore the importance of furtherresearch to ensure the safe and reliable deployment of LLMs and AI agents. |


| Item |Content|
| --- |---|
|idx| 2409.09009v1 |
|title| Optimizing Rare Word Accuracy in Direct Speech Translation with a Retrieval-and-Demonstration Approach |
|authors| Siqi LiDanni LiuJan Niehues
|links| http://arxiv.org/abs/2409.09009v1 |
|updated| 2024-09-13 17:38:03 UTC |
|summary| Direct speech translation ST models often struggle with rare words.Incorrect translation of these words can have severe consequences impactingtranslation quality and user trust. While rare word translation is inherentlychallenging for neural models due to sparse learning signals real-worldscenarios often allow access to translations of past recordings on similartopics. To leverage these valuable resources we propose aretrieval-and-demonstration approach to enhance rare word translation accuracyin direct ST models. First we adapt existing ST models to incorporateretrieved examples for rare word translation which allows the model to benefitfrom prepended examples similar to in-context learning. We then develop across-modal speech-to-speech speech-to-text text-to-text retriever tolocate suitable examples. We demonstrate that standard ST models can beeffectively adapted to leverage examples for rare word translation improvingrare word translation accuracy over the baseline by 17.6 with gold examplesand 8.5 with retrieved examples. Moreover our speech-to-speech retrievalapproach outperforms other modalities and exhibits higher robustness to unseenspeakers. Our code is publicly availablehttps://github.com/SiqiLii/Retrieve-and-Demonstration-ST. |


| Item |Content|
| --- |---|
|idx| 2409.09001v1 |
|title| E2MoCase: A Dataset for Emotional, Event and Moral Observations in News Articles on High-impact Legal Cases |
|authors| Candida M. GrecoLorenzo ZangariDavide PiccaAndrea Tagarelli
|links| http://arxiv.org/abs/2409.09001v1 |
|updated| 2024-09-13 17:31:09 UTC |
|summary| The way media reports on legal cases can significantly shape public opinionoften embedding subtle biases that influence societal views on justice andmorality. Analyzing these biases requires a holistic approach that captures theemotional tone moral framing and specific events within the narratives. Inthis work we introduce E2MoCase a novel dataset designed to facilitate theintegrated analysis of emotions moral values and events within legalnarratives and media coverage. By leveraging advanced models for emotiondetection moral value identification and event extraction E2MoCase offers amulti-dimensional perspective on how legal cases are portrayed in newsarticles. |


| Item |Content|
| --- |---|
|idx| 2409.08963v1 |
|title| Safeguarding Decentralized Social Media: LLM Agents for Automating Community Rule Compliance |
|authors| Lucio La CavaAndrea Tagarelli
|links| http://arxiv.org/abs/2409.08963v1 |
|updated| 2024-09-13 16:29:25 UTC |
|summary| Ensuring content compliance with community guidelines is crucial formaintaining healthy online social environments. However traditionalhuman-based compliance checking struggles with scaling due to the increasingvolume of user-generated content and a limited number of moderators. Recentadvancements in Natural Language Understanding demonstrated by Large LanguageModels unlock new opportunities for automated content compliance verification.This work evaluates six AI-agents built on Open-LLMs for automated rulecompliance checking in Decentralized Social Networks a challenging environmentdue to heterogeneous community scopes and rules. Analyzing over 50000 postsfrom hundreds of Mastodon servers we find that AI-agents effectively detectnon-compliant content grasp linguistic subtleties and adapt to diversecommunity contexts. Most agents also show high inter-rater reliability andconsistency in score justification and suggestions for compliance. Human-basedevaluation with domain experts confirmed the agents reliability andusefulness rendering them promising tools for semi-automated orhuman-in-the-loop content moderation systems. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2409.09032v1 |
|title| The unknotting number, hard unknot diagrams, and reinforcement learning |
|authors| Taylor ApplebaumSam BlackwellAlex DaviesThomas EdlichAndrás JuhászMarc LackenbyNenad TomaševDaniel Zheng
|links| http://arxiv.org/abs/2409.09032v1 |
|updated| 2024-09-13 17:59:52 UTC |
|summary| We have developed a reinforcement learning agent that often finds a minimalsequence of unknotting crossing changes for a knot diagram with up to 200crossings hence giving an upper bound on the unknotting number. We have usedthis to determine the unknotting number of 57k knots. We took diagrams ofconnected sums of such knots with oppositely signed signatures where thesummands were overlaid. The agent has found examples where several of thecrossing changes in an unknotting collection of crossings result in hyperbolicknots. Based on this we have shown that given knots K and K that satisfysome mild assumptions there is a diagram of their connected sum and uK uK unknotting crossings such that changing any one of them results in aprime knot. As a by-product we have obtained a dataset of 2.6 million distincthard unknot diagrams most of them under 35 crossings. Assuming the additivityof the unknotting number we have determined the unknotting number of 43 atmost 12-crossing knots for which the unknotting number is unknown. |


| Item |Content|
| --- |---|
|idx| 2409.09030v1 |
|title| Agents in Software Engineering: Survey, Landscape, and Vision |
|authors| Yanxian HuangWanjun ZhongEnsheng ShiMin YangJiachi ChenHui LiYuchi MaQianxiang WangZibin ZhengYanlin Wang
|links| http://arxiv.org/abs/2409.09030v1 |
|updated| 2024-09-13 17:55:58 UTC |
|summary| In recent years Large Language Models LLMs have achieved remarkablesuccess and have been widely used in various downstream tasks especially inthe tasks of the software engineering SE field. We find that many studiescombining LLMs with SE have employed the concept of agents either explicitly orimplicitly. However there is a lack of an in-depth survey to sort out thedevelopment context of existing works analyze how existing works combine theLLM-based agent technologies to optimize various tasks and clarify theframework of LLM-based agents in SE. In this paper we conduct the first surveyof the studies on combining LLM-based agents with SE and present a framework ofLLM-based agents in SE which includes three key modules: perception memoryand action. We also summarize the current challenges in combining the twofields and propose future opportunities in response to existing challenges. Wemaintain a GitHub repository of the related papers at:https://github.com/DeepSoftwareAnalytics/Awesome-Agent4SE. |


| Item |Content|
| --- |---|
|idx| 2409.09026v1 |
|title| Towards Leveraging Contrastively Pretrained Neural Audio Embeddings for Recommender Tasks |
|authors| Florian GrötschlaLuca SträssleLuca A. LanzendörferRoger Wattenhofer
|links| http://arxiv.org/abs/2409.09026v1 |
|updated| 2024-09-13 17:53:06 UTC |
|summary| Music recommender systems frequently utilize network-based models to capturerelationships between music pieces artists and users. Although theserelationships provide valuable insights for predictions new music pieces orartists often face the cold-start problem due to insufficient initialinformation. To address this one can extract content-based informationdirectly from the music to enhance collaborative-filtering-based methods. Whileprevious approaches have relied on hand-crafted audio features for thispurpose we explore the use of contrastively pretrained neural audio embeddingmodels which offer a richer and more nuanced representation of music. Ourexperiments demonstrate that neural embeddings particularly those generatedwith the Contrastive Language-Audio Pretraining CLAP model present apromising approach to enhancing music recommendation tasks within graph-basedframeworks. |


| Item |Content|
| --- |---|
|idx| 2409.09013v1 |
|title| AI-LieDar: Examine the Trade-off Between Utility and Truthfulness in LLM Agents |
|authors| Zhe SuXuhui ZhouSanketh RangrejiAnubha KabraJulia MendelsohnFaeze BrahmanMaarten Sap
|links| http://arxiv.org/abs/2409.09013v1 |
|updated| 2024-09-13 17:41:12 UTC |
|summary| To be safely and successfully deployed LLMs must simultaneously satisfytruthfulness and utility goals. Yet often these two goals compete e.g. an AIagent assisting a used car salesman selling a car with flaws partly due toambiguous or misleading user instructions. We propose AI-LieDar a framework tostudy how LLM-based agents navigate scenarios with utility-truthfulnessconflicts in a multi-turn interactive setting. We design a set of realisticscenarios where language agents are instructed to achieve goals that are inconflict with being truthful during a multi-turn conversation with simulatedhuman agents. To evaluate the truthfulness at large scale we develop atruthfulness detector inspired by psychological literature to assess theagents responses. Our experiment demonstrates that all models are truthfulless than 50 of the time although truthfulness and goal achievement utilityrates vary across models. We further test the steerability of LLMs towardstruthfulness finding that models follow malicious instructions to deceive andeven truth-steered models can still lie. These findings reveal the complexnature of truthfulness in LLMs and underscore the importance of furtherresearch to ensure the safe and reliable deployment of LLMs and AI agents. |


| Item |Content|
| --- |---|
|idx| 2409.09011v1 |
|title| VAE Explainer: Supplement Learning Variational Autoencoders with Interactive Visualization |
|authors| Donald BertucciAlex Endert
|links| http://arxiv.org/abs/2409.09011v1 |
|updated| 2024-09-13 17:40:01 UTC |
|summary| Variational Autoencoders are widespread in Machine Learning but aretypically explained with dense math notation or static code examples. Thispaper presents VAE Explainer an interactive Variational Autoencoder running inthe browser to supplement existing static documentation e.g. Keras CodeExamples. VAE Explainer adds interactions to the VAE summary with interactivemodel inputs latent space and output. VAE Explainer connects the high-levelunderstanding with the implementation: annotated code and a live computationalgraph. The VAE Explainer interactive visualization is live athttps://xnought.github.io/vae-explainer and the code is open source athttps://github.com/xnought/vae-explainer. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2409.09032v1 |
|title| The unknotting number, hard unknot diagrams, and reinforcement learning |
|authors| Taylor ApplebaumSam BlackwellAlex DaviesThomas EdlichAndrás JuhászMarc LackenbyNenad TomaševDaniel Zheng
|links| http://arxiv.org/abs/2409.09032v1 |
|updated| 2024-09-13 17:59:52 UTC |
|summary| We have developed a reinforcement learning agent that often finds a minimalsequence of unknotting crossing changes for a knot diagram with up to 200crossings hence giving an upper bound on the unknotting number. We have usedthis to determine the unknotting number of 57k knots. We took diagrams ofconnected sums of such knots with oppositely signed signatures where thesummands were overlaid. The agent has found examples where several of thecrossing changes in an unknotting collection of crossings result in hyperbolicknots. Based on this we have shown that given knots K and K that satisfysome mild assumptions there is a diagram of their connected sum and uK uK unknotting crossings such that changing any one of them results in aprime knot. As a by-product we have obtained a dataset of 2.6 million distincthard unknot diagrams most of them under 35 crossings. Assuming the additivityof the unknotting number we have determined the unknotting number of 43 atmost 12-crossing knots for which the unknotting number is unknown. |


| Item |Content|
| --- |---|
|idx| 2409.09021v1 |
|title| INN-PAR: Invertible Neural Network for PPG to ABP Reconstruction |
|authors| Soumitra KunduGargi PandaSaumik BhattacharyaAurobinda RoutrayRajlakshmi Guha
|links| http://arxiv.org/abs/2409.09021v1 |
|updated| 2024-09-13 17:48:48 UTC |
|summary| Non-invasive and continuous blood pressure BP monitoring is essential forthe early prevention of many cardiovascular diseases. Estimating arterial bloodpressure ABP from photoplethysmography PPG has emerged as a promisingsolution. However existing deep learning approaches for PPG-to-ABPreconstruction PAR encounter certain information loss impacting theprecision of the reconstructed signal. To overcome this limitation weintroduce an invertible neural network for PPG to ABP reconstruction INN-PARwhich employs a series of invertible blocks to jointly learn the mappingbetween PPG and its gradient with the ABP signal and its gradient. INN-PARefficiently captures both forward and inverse mappings simultaneously therebypreventing information loss. By integrating signal gradients into the learningprocess INN-PAR enhances the networks ability to capture essentialhigh-frequency details leading to more accurate signal reconstruction.Moreover we propose a multi-scale convolution module MSCM within theinvertible block enabling the model to learn features across multiple scaleseffectively. We have experimented on two benchmark datasets which show thatINN-PAR significantly outperforms the state-of-the-art methods in both waveformreconstruction and BP measurement accuracy. |


| Item |Content|
| --- |---|
|idx| 2409.09018v1 |
|title| An Efficient and Streaming Audio Visual Active Speaker Detection System |
|authors| Arnav KunduYanzi JinMohammad SekhavatMax HortonDanny TormoenDevang Naik
|links| http://arxiv.org/abs/2409.09018v1 |
|updated| 2024-09-13 17:45:53 UTC |
|summary| This paper delves into the challenging task of Active Speaker DetectionASD where the system needs to determine in real-time whether a person isspeaking or not in a series of video frames. While previous works have madesignificant strides in improving network architectures and learning effectiverepresentations for ASD a critical gap exists in the exploration of real-timesystem deployment. Existing models often suffer from high latency and memoryusage rendering them impractical for immediate applications. To bridge thisgap we present two scenarios that address the key challenges posed byreal-time constraints. First we introduce a method to limit the number offuture context frames utilized by the ASD model. By doing so we alleviate theneed for processing the entire sequence of future frames before a decision ismade significantly reducing latency. Second we propose a more stringentconstraint that limits the total number of past frames the model can accessduring inference. This tackles the persistent memory issues associated withrunning streaming ASD systems. Beyond these theoretical frameworks we conductextensive experiments to validate our approach. Our results demonstrate thatconstrained transformer models can achieve performance comparable to or evenbetter than state-of-the-art recurrent models such as uni-directional GRUswith a significantly reduced number of context frames. Moreover we shed lighton the temporal memory requirements of ASD systems revealing that larger pastcontext has a more profound impact on accuracy than future context. Whenprofiling on a CPU we find that our efficient architecture is memory bound bythe amount of past context it can use and that the compute cost is negligibleas compared to the memory cost. |


| Item |Content|
| --- |---|
|idx| 2409.09011v1 |
|title| VAE Explainer: Supplement Learning Variational Autoencoders with Interactive Visualization |
|authors| Donald BertucciAlex Endert
|links| http://arxiv.org/abs/2409.09011v1 |
|updated| 2024-09-13 17:40:01 UTC |
|summary| Variational Autoencoders are widespread in Machine Learning but aretypically explained with dense math notation or static code examples. Thispaper presents VAE Explainer an interactive Variational Autoencoder running inthe browser to supplement existing static documentation e.g. Keras CodeExamples. VAE Explainer adds interactions to the VAE summary with interactivemodel inputs latent space and output. VAE Explainer connects the high-levelunderstanding with the implementation: annotated code and a live computationalgraph. The VAE Explainer interactive visualization is live athttps://xnought.github.io/vae-explainer and the code is open source athttps://github.com/xnought/vae-explainer. |


| Item |Content|
| --- |---|
|idx| 2409.09007v1 |
|title| SGFormer: Single-Layer Graph Transformers with Approximation-Free Linear Complexity |
|authors| Qitian WuKai YangHengrui ZhangDavid WipfJunchi Yan
|links| http://arxiv.org/abs/2409.09007v1 |
|updated| 2024-09-13 17:37:34 UTC |
|summary| Learning representations on large graphs is a long-standing challenge due tothe inter-dependence nature. Transformers recently have shown promisingperformance on small graphs thanks to its global attention for capturingall-pair interactions beyond observed structures. Existing approaches tend toinherit the spirit of Transformers in language and vision tasks and embracecomplicated architectures by stacking deep attention-based propagation layers.In this paper we attempt to evaluate the necessity of adopting multi-layerattentions in Transformers on graphs which considerably restricts theefficiency. Specifically we analyze a generic hybrid propagation layercomprised of all-pair attention and graph-based propagation and show thatmulti-layer propagation can be reduced to one-layer propagation with the samecapability for representation learning. It suggests a new technical path forbuilding powerful and efficient Transformers on graphs particularly throughsimplifying model architectures without sacrificing expressiveness. Asexemplified by this work we propose a Simplified Single-layer GraphTransformers SGFormer whose main component is a single-layer globalattention that scales linearly w.r.t. graph sizes and requires none of anyapproximation for accommodating all-pair interactions. Empirically SGFormersuccessfully scales to the web-scale graph ogbn-papers100M yieldingorders-of-magnitude inference acceleration over peer Transformers onmedium-sized graphs and demonstrates competitiveness with limited labeleddata. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2409.09018v1 |
|title| An Efficient and Streaming Audio Visual Active Speaker Detection System |
|authors| Arnav KunduYanzi JinMohammad SekhavatMax HortonDanny TormoenDevang Naik
|links| http://arxiv.org/abs/2409.09018v1 |
|updated| 2024-09-13 17:45:53 UTC |
|summary| This paper delves into the challenging task of Active Speaker DetectionASD where the system needs to determine in real-time whether a person isspeaking or not in a series of video frames. While previous works have madesignificant strides in improving network architectures and learning effectiverepresentations for ASD a critical gap exists in the exploration of real-timesystem deployment. Existing models often suffer from high latency and memoryusage rendering them impractical for immediate applications. To bridge thisgap we present two scenarios that address the key challenges posed byreal-time constraints. First we introduce a method to limit the number offuture context frames utilized by the ASD model. By doing so we alleviate theneed for processing the entire sequence of future frames before a decision ismade significantly reducing latency. Second we propose a more stringentconstraint that limits the total number of past frames the model can accessduring inference. This tackles the persistent memory issues associated withrunning streaming ASD systems. Beyond these theoretical frameworks we conductextensive experiments to validate our approach. Our results demonstrate thatconstrained transformer models can achieve performance comparable to or evenbetter than state-of-the-art recurrent models such as uni-directional GRUswith a significantly reduced number of context frames. Moreover we shed lighton the temporal memory requirements of ASD systems revealing that larger pastcontext has a more profound impact on accuracy than future context. Whenprofiling on a CPU we find that our efficient architecture is memory bound bythe amount of past context it can use and that the compute cost is negligibleas compared to the memory cost. |


| Item |Content|
| --- |---|
|idx| 2409.08953v1 |
|title| Pushing the boundaries of event subsampling in event-based video classification using CNNs |
|authors| Hesam AraghiJan van GemertNergis Tomen
|links| http://arxiv.org/abs/2409.08953v1 |
|updated| 2024-09-13 16:14:45 UTC |
|summary| Event cameras offer low-power visual sensing capabilities ideal foredge-device applications. However their high event rate driven by hightemporal details can be restrictive in terms of bandwidth and computationalresources. In edge AI applications determining the minimum amount of eventsfor specific tasks can allow reducing the event rate to improve bandwidthmemory and processing efficiency. In this paper we study the effect of eventsubsampling on the accuracy of event data classification using convolutionalneural network CNN models. Surprisingly across various datasets the numberof events per video can be reduced by an order of magnitude with little drop inaccuracy revealing the extent to which we can push the boundaries in accuracyvs. event rate trade-off. Additionally we also find that lower classificationaccuracy in high subsampling rates is not solely attributable to informationloss due to the subsampling of the events but that the training of CNNs can bechallenging in highly subsampled scenarios where the sensitivity tohyperparameters increases. We quantify training instability across multipleevent-based classification datasets using a novel metric for evaluating thehyperparameter sensitivity of CNNs in different subsampling settings. Finallywe analyze the weight gradients of the network to gain insight into thisinstability. |


| Item |Content|
| --- |---|
|idx| 2409.08947v1 |
|title| A Diffusion Approach to Radiance Field Relighting using Multi-Illumination Synthesis |
|authors| Yohan Poirier-GinterAlban GauthierJulien PhillipJean-Francois LalondeGeorge Drettakis
|links| http://dx.doi.org/10.1111/cgf.15147 |
|updated| 2024-09-13 16:07:25 UTC |
|summary| Relighting radiance fields is severely underconstrained for multi-view datawhich is most often captured under a single illumination condition It isespecially hard for full scenes containing multiple objects. We introduce amethod to create relightable radiance fields using such single-illuminationdata by exploiting priors extracted from 2D image diffusion models. We firstfine-tune a 2D diffusion model on a multi-illumination dataset conditioned bylight direction allowing us to augment a single-illumination capture into arealistic -- but possibly inconsistent -- multi-illumination dataset fromdirectly defined light directions. We use this augmented data to create arelightable radiance field represented by 3D Gaussian splats. To allow directcontrol of light direction for low-frequency lighting we represent appearancewith a multi-layer perceptron parameterized on light direction. To enforcemulti-view consistency and overcome inaccuracies we optimize a per-imageauxiliary feature vector. We show results on synthetic and real multi-view dataunder single illumination demonstrating that our method successfully exploits2D diffusion model priors to allow realistic 3D relighting for complete scenes.Project sitehttps://repo-sam.inria.fr/fungraph/generative-radiance-field-relighting/ |


| Item |Content|
| --- |---|
|idx| 2409.08943v1 |
|title| Pushing Joint Image Denoising and Classification to the Edge |
|authors| Thomas C MarkhorstJan C van GemertOsman S Kayhan
|links| http://arxiv.org/abs/2409.08943v1 |
|updated| 2024-09-13 16:01:27 UTC |
|summary| In this paper we jointly combine image classification and image denoisingaiming to enhance human perception of noisy images captured by edge deviceslike low-light security cameras. In such settings it is important to retainthe ability of humans to verify the automatic classification decision and thusjointly denoise the image to enhance human perception. Since edge devices havelittle computational power we explicitly optimize for efficiency by proposinga novel architecture that integrates the two tasks. Additionally we alter aNeural Architecture Search NAS method which searches for classifiers tosearch for the integrated model while optimizing for a target latencyclassification accuracy and denoising performance. The NAS architecturesoutperform our manually designed alternatives in both denoising andclassification offering a significant improvement to human perception. Ourapproach empowers users to construct architectures tailored to domains likemedical imaging surveillance systems and industrial inspections. |


| Item |Content|
| --- |---|
|idx| 2409.08926v1 |
|title| ClearDepth: Enhanced Stereo Perception of Transparent Objects for Robotic Manipulation |
|authors| Kaixin BaiHuajian ZengLei ZhangYiwen LiuHongli XuZhaopeng ChenJianwei Zhang
|links| http://arxiv.org/abs/2409.08926v1 |
|updated| 2024-09-13 15:44:38 UTC |
|summary| Transparent object depth perception poses a challenge in everyday life andlogistics primarily due to the inability of standard 3D sensors to accuratelycapture depth on transparent or reflective surfaces. This limitationsignificantly affects depth map and point cloud-reliant applicationsespecially in robotic manipulation. We developed a vision transformer-basedalgorithm for stereo depth recovery of transparent objects. This approach iscomplemented by an innovative feature post-fusion module which enhances theaccuracy of depth recovery by structural features in images. To address thehigh costs associated with dataset collection for stereo camera-basedperception of transparent objects our method incorporates a parameter-aligneddomain-adaptive and physically realistic Sim2Real simulation for efficientdata generation accelerated by AI algorithm. Our experimental resultsdemonstrate the models exceptional Sim2Real generalizability in real-worldscenarios enabling precise depth mapping of transparent objects to assist inrobotic manipulation. Project details are available athttps://sites.google.com/view/cleardepth/ . |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2409.09003v2 |
|title| Model-independent variable selection via the rule-based variable priority |
|authors| Min LuHemant Ishwaran
|links| http://arxiv.org/abs/2409.09003v2 |
|updated| 2024-09-16 17:34:26 UTC |
|summary| While achieving high prediction accuracy is a fundamental goal in machinelearning an equally important task is finding a small number of features withhigh explanatory power. One popular selection technique is permutationimportance which assesses a variables impact by measuring the change inprediction error after permuting the variable. However this can be problematicdue to the need to create artificial data a problem shared by other methods aswell. Another problem is that variable selection methods can be limited bybeing model-specific. We introduce a new model-independent approach VariablePriority VarPro which works by utilizing rules without the need to generateartificial data or evaluate prediction error. The method is relatively easy touse requiring only the calculation of sample averages of simple statisticsand can be applied to many data settings including regression classificationand survival. We investigate the asymptotic properties of VarPro and showamong other things that VarPro has a consistent filtering property for noisevariables. Empirical studies using synthetic and real-world data show themethod achieves a balanced performance and compares favorably to manystate-of-the-art procedures currently used for variable selection. |


| Item |Content|
| --- |---|
|idx| 2409.08954v1 |
|title| A Bayesian Approach to Clustering via the Proper Bayesian Bootstrap: the Bayesian Bagged Clustering (BBC) algorithm |
|authors| Federico Maria QuettiSilvia FiginiElena ballante
|links| http://arxiv.org/abs/2409.08954v1 |
|updated| 2024-09-13 16:14:54 UTC |
|summary| The paper presents a novel approach for unsupervised techniques in the fieldof clustering. A new method is proposed to enhance existing literature modelsusing the proper Bayesian bootstrap to improve results in terms of robustnessand interpretability. Our approach is organized in two steps: k-meansclustering is used for prior elicitation then proper Bayesian bootstrap isapplied as resampling method in an ensemble clustering approach. Results areanalyzed introducing measures of uncertainty based on Shannon entropy. Theproposal provides clear indication on the optimal number of clusters as wellas a better representation of the clustered data. Empirical results areprovided on simulated data showing the methodological and empirical advancesobtained. |


| Item |Content|
| --- |---|
|idx| 2409.08925v1 |
|title| Multi forests: Variable importance for multi-class outcomes |
|authors| Roman HornungAlexander Hapfelmeier
|links| http://arxiv.org/abs/2409.08925v1 |
|updated| 2024-09-13 15:40:29 UTC |
|summary| In prediction tasks with multi-class outcomes identifying covariatesspecifically associated with one or more outcome classes can be important.Conventional variable importance measures VIMs from random forests RFslike permutation and Gini importance focus on overall predictive performanceor node purity without differentiating between the classes. Therefore theycan be expected to fail to distinguish class-associated covariates fromcovariates that only distinguish between groups of classes. We introduce a VIMcalled multi-class VIM tailored for identifying exclusively class-associatedcovariates via a novel RF variant called multi forests MuFs. The trees inMuFs use both multi-way and binary splitting. The multi-way splits generatechild nodes for each class using a split criterion that evaluates how wellthese nodes represent their respective classes. This setup forms the basis ofthe multi-class VIM which measures the discriminatory ability of the splitsperformed in the respective covariates with regard to this split criterion.Alongside the multi-class VIM we introduce a second VIM the discriminatoryVIM. This measure based on the binary splits assesses the strength of thegeneral influence of the covariates irrespective of theirclass-associatedness. Simulation studies demonstrate that the multi-class VIMspecifically ranks class-associated covariates highly unlike conventional VIMswhich also rank other types of covariates highly. Analyses of 121 datasetsreveal that MuFs often have slightly lower predictive performance compared toconventional RFs. This is however not a limiting factor given the algorithmsprimary purpose of calculating the multi-class VIM. |


| Item |Content|
| --- |---|
|idx| 2409.08917v1 |
|title| Latent Space Score-based Diffusion Model for Probabilistic Multivariate Time Series Imputation |
|authors| Guojun LiangNajmeh AbiriAtiye Sadat HashemiJens LundströmStefan ByttnerPrayag Tiwari
|links| http://arxiv.org/abs/2409.08917v1 |
|updated| 2024-09-13 15:32:26 UTC |
|summary| Accurate imputation is essential for the reliability and success ofdownstream tasks. Recently diffusion models have attracted great attention inthis field. However these models neglect the latent distribution in alower-dimensional space derived from the observed data which limits thegenerative capacity of the diffusion model. Additionally dealing with theoriginal missing data without labels becomes particularly problematic. Toaddress these issues we propose the Latent Space Score-Based Diffusion ModelLSSDM for probabilistic multivariate time series imputation. Observed valuesare projected onto low-dimensional latent space and coarse values of themissing data are reconstructed without knowing their ground truth values bythis unsupervised learning approach. Finally the reconstructed values are fedinto a conditional diffusion model to obtain the precise imputed values of thetime series. In this way LSSDM not only possesses the power to identify thelatent distribution but also seamlessly integrates the diffusion model toobtain the high-fidelity imputed values and assess the uncertainty of thedataset. Experimental results demonstrate that LSSDM achieves superiorimputation performance while also providing a better explanation anduncertainty analysis of the imputation mechanism. The website of the code istextithttps://github.com/gorgen2020/LSSDM_imputation. |


| Item |Content|
| --- |---|
|idx| 2409.08861v1 |
|title| Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control |
|authors| Carles Domingo-EnrichMichal DrozdzalBrian KarrerRicky T. Q. Chen
|links| http://arxiv.org/abs/2409.08861v1 |
|updated| 2024-09-13 14:22:14 UTC |
|summary| Dynamical generative models that produce samples through an iterativeprocess such as Flow Matching and denoising diffusion models have seenwidespread use but there has not been many theoretically-sound methods forimproving these models with reward fine-tuning. In this work we cast rewardfine-tuning as stochastic optimal control SOC. Critically we prove that avery specific memoryless noise schedule must be enforced during fine-tuning inorder to account for the dependency between the noise variable and thegenerated samples. We also propose a new algorithm named Adjoint Matching whichoutperforms existing SOC algorithms by casting SOC problems as a regressionproblem. We find that our approach significantly improves over existing methodsfor reward fine-tuning achieving better consistency realism andgeneralization to unseen human preference reward models while retaining samplediversity. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2409.09021v1 |
|title| INN-PAR: Invertible Neural Network for PPG to ABP Reconstruction |
|authors| Soumitra KunduGargi PandaSaumik BhattacharyaAurobinda RoutrayRajlakshmi Guha
|links| http://arxiv.org/abs/2409.09021v1 |
|updated| 2024-09-13 17:48:48 UTC |
|summary| Non-invasive and continuous blood pressure BP monitoring is essential forthe early prevention of many cardiovascular diseases. Estimating arterial bloodpressure ABP from photoplethysmography PPG has emerged as a promisingsolution. However existing deep learning approaches for PPG-to-ABPreconstruction PAR encounter certain information loss impacting theprecision of the reconstructed signal. To overcome this limitation weintroduce an invertible neural network for PPG to ABP reconstruction INN-PARwhich employs a series of invertible blocks to jointly learn the mappingbetween PPG and its gradient with the ABP signal and its gradient. INN-PARefficiently captures both forward and inverse mappings simultaneously therebypreventing information loss. By integrating signal gradients into the learningprocess INN-PAR enhances the networks ability to capture essentialhigh-frequency details leading to more accurate signal reconstruction.Moreover we propose a multi-scale convolution module MSCM within theinvertible block enabling the model to learn features across multiple scaleseffectively. We have experimented on two benchmark datasets which show thatINN-PAR significantly outperforms the state-of-the-art methods in both waveformreconstruction and BP measurement accuracy. |


| Item |Content|
| --- |---|
|idx| 2409.09018v1 |
|title| An Efficient and Streaming Audio Visual Active Speaker Detection System |
|authors| Arnav KunduYanzi JinMohammad SekhavatMax HortonDanny TormoenDevang Naik
|links| http://arxiv.org/abs/2409.09018v1 |
|updated| 2024-09-13 17:45:53 UTC |
|summary| This paper delves into the challenging task of Active Speaker DetectionASD where the system needs to determine in real-time whether a person isspeaking or not in a series of video frames. While previous works have madesignificant strides in improving network architectures and learning effectiverepresentations for ASD a critical gap exists in the exploration of real-timesystem deployment. Existing models often suffer from high latency and memoryusage rendering them impractical for immediate applications. To bridge thisgap we present two scenarios that address the key challenges posed byreal-time constraints. First we introduce a method to limit the number offuture context frames utilized by the ASD model. By doing so we alleviate theneed for processing the entire sequence of future frames before a decision ismade significantly reducing latency. Second we propose a more stringentconstraint that limits the total number of past frames the model can accessduring inference. This tackles the persistent memory issues associated withrunning streaming ASD systems. Beyond these theoretical frameworks we conductextensive experiments to validate our approach. Our results demonstrate thatconstrained transformer models can achieve performance comparable to or evenbetter than state-of-the-art recurrent models such as uni-directional GRUswith a significantly reduced number of context frames. Moreover we shed lighton the temporal memory requirements of ASD systems revealing that larger pastcontext has a more profound impact on accuracy than future context. Whenprofiling on a CPU we find that our efficient architecture is memory bound bythe amount of past context it can use and that the compute cost is negligibleas compared to the memory cost. |


| Item |Content|
| --- |---|
|idx| 2409.09011v1 |
|title| VAE Explainer: Supplement Learning Variational Autoencoders with Interactive Visualization |
|authors| Donald BertucciAlex Endert
|links| http://arxiv.org/abs/2409.09011v1 |
|updated| 2024-09-13 17:40:01 UTC |
|summary| Variational Autoencoders are widespread in Machine Learning but aretypically explained with dense math notation or static code examples. Thispaper presents VAE Explainer an interactive Variational Autoencoder running inthe browser to supplement existing static documentation e.g. Keras CodeExamples. VAE Explainer adds interactions to the VAE summary with interactivemodel inputs latent space and output. VAE Explainer connects the high-levelunderstanding with the implementation: annotated code and a live computationalgraph. The VAE Explainer interactive visualization is live athttps://xnought.github.io/vae-explainer and the code is open source athttps://github.com/xnought/vae-explainer. |


| Item |Content|
| --- |---|
|idx| 2409.08980v1 |
|title| Predicting Trust In Autonomous Vehicles: Modeling Young Adult Psychosocial Traits, Risk-Benefit Attitudes, And Driving Factors With Machine Learning |
|authors| Robert KaufmanEmi LeeManas Satish BedmuthaDavid KirshNadir Weibel
|links| http://arxiv.org/abs/2409.08980v1 |
|updated| 2024-09-13 16:52:24 UTC |
|summary| Low trust remains a significant barrier to Autonomous Vehicle AV adoption.To design trustworthy AVs we need to better understand the individual traitsattitudes and experiences that impact peoples trust judgements. We usemachine learning to understand the most important factors that contribute toyoung adult trust based on a comprehensive set of personal factors gathered viasurvey n  1457. Factors ranged from psychosocial and cognitive attributes todriving style experiences and perceived AV risks and benefits. Using theexplainable AI technique SHAP we found that perceptions of AV risks andbenefits attitudes toward feasibility and usability institutional trustprior experience and a persons mental model are the most importantpredictors. Surprisingly psychosocial and many technology- anddriving-specific factors were not strong predictors. Results highlight theimportance of individual differences for designing trustworthy AVs for diversegroups and lead to key implications for future design and research. |


| Item |Content|
| --- |---|
|idx| 2409.08967v1 |
|title| Modeling Rational Adaptation of Visual Search to Hierarchical Structures |
|authors| Saku SourulahtiChristian P JanssenJussi PP Jokinen
|links| http://arxiv.org/abs/2409.08967v1 |
|updated| 2024-09-13 16:33:18 UTC |
|summary| Efficient attention deployment in visual search is limited by human visualmemory yet this limitation can be offset by exploiting the environmentsstructure. This paper introduces a computational cognitive model that simulateshow the human visual system uses visual hierarchies to prevent refixations insequential attention deployment. The model adopts computational rationalitypositing behaviors as adaptations to cognitive constraints and environmentalstructures. In contrast to earlier models that predict search performance forhierarchical information our model does not include predefined assumptionsabout particular search strategies. Instead our models search strategyemerges as a result of adapting to the environment through reinforcementlearning algorithms. In an experiment with human participants we test themodels prediction that structured environments reduce visual search timescompared to random tasks. Our models predictions correspond well with humansearch performance across various set sizes for both structured andunstructured visual layouts. Our work improves understanding of the adaptivenature of visual search in hierarchically structured environments and informsthe design of optimized search spaces. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2409.08811v1 |
|title| Mutual Theory of Mind in Human-AI Collaboration: An Empirical Study with LLM-driven AI Agents in a Real-time Shared Workspace Task |
|authors| Shao ZhangXihuai WangWenhao ZhangYongshan ChenLandi GaoDakuo WangWeinan ZhangXinbing WangYing Wen
|links| http://arxiv.org/abs/2409.08811v1 |
|updated| 2024-09-13 13:19:48 UTC |
|summary| Theory of Mind ToM significantly impacts human collaboration andcommunication as a crucial capability to understand others. When AI agents withToM capability collaborate with humans Mutual Theory of Mind MToM arises insuch human-AI teams HATs. The MToM process which involves interactivecommunication and ToM-based strategy adjustment affects the teams performanceand collaboration process. To explore the MToM process we conducted amixed-design experiment using a large language model-driven AI agent with ToMand communication modules in a real-time shared-workspace task. We find thatthe agents ToM capability does not significantly impact team performance butenhances human understanding of the agent and the feeling of being understood.Most participants in our study believe verbal communication increases humanburden and the results show that bidirectional communication leads to lowerHAT performance. We discuss the results implications for designing AI agentsthat collaborate with humans in real-time shared workspace tasks. |


| Item |Content|
| --- |---|
|idx| 2409.08404v1 |
|title| Simultaneous Topology Estimation and Synchronization of Dynamical Networks with Time-varying Topology |
|authors| Nana WangEsteban RestrepoDimos V. Dimarogonas
|links| http://arxiv.org/abs/2409.08404v1 |
|updated| 2024-09-12 21:32:48 UTC |
|summary| We propose an adaptive control strategy for the simultaneous estimation oftopology and synchronization in complex dynamical networks with unknowntime-varying topology. Our approach transforms the problem of time-varyingtopology estimation into a problem of estimating the time-varying weights of acomplete graph utilizing an edge-agreement framework. We introduce twoauxiliary networks: one that satisfies the persistent excitation condition tofacilitate topology estimation while the other a uniform-deltapersistently exciting network ensures the boundedness of both weightestimation and synchronization errors assuming bounded time-varying weightsand their derivatives. A relevant numerical example shows the efficiency of ourmethods. |


| Item |Content|
| --- |---|
|idx| 2409.08386v1 |
|title| Self-Supervised Inference of Agents in Trustless Environments |
|authors| Vladyslav LarinIvan NikitinAlexander Firsov
|links| http://arxiv.org/abs/2409.08386v1 |
|updated| 2024-09-12 20:32:07 UTC |
|summary| In this paper we propose a novel approach where agents can form swarms toproduce high-quality responses effectively. This is accomplished by utilizingagents capable of data inference and ranking which can be effectivelyimplemented using LLMs as response classifiers. We assess existing approachesfor trustless agent inference define our methodology estimate practicalparameters and model various types of malicious agent attacks. Our methodleverages the collective intelligence of swarms ensuring robust and efficientdecentralized AI inference with better accuracy security and reliability. Weshow that our approach is an order of magnitude faster than other trustlessinference strategies reaching less than 125 ms validation latency. |


| Item |Content|
| --- |---|
|idx| 2409.08145v1 |
|title| Inertial Coordination Games |
|authors| Andrew KohRicky LiKei Uzui
|links| http://arxiv.org/abs/2409.08145v1 |
|updated| 2024-09-12 15:37:36 UTC |
|summary| We analyze inertial coordination games: dynamic coordination games with anendogenously changing state that depends on i a persistent fundamental thatplayers privately learn about and ii past play. We give a tightcharacterization of how the speed of learning shapes equilibrium dynamics: therisk-dominant action is selected in the limit if and only if learning is slowsuch that posterior precisions grow sub-quadratically. This generalizes resultsfrom static global games and endows them with an alternate learning foundation.Conversely when learning is fast equilibrium dynamics exhibit persistence andlimit play is shaped by initial play. Whenever the risk dominant equilibrium isselected the path of play undergoes a sudden transition when signals areprecise and a gradual transition when signals are noisy. |


| Item |Content|
| --- |---|
|idx| 2409.07932v1 |
|title| Reinforcement Learning Discovers Efficient Decentralized Graph Path Search Strategies |
|authors| Alexei PisacaneVictor-Alexandru DarvariuMirco Musolesi
|links| http://arxiv.org/abs/2409.07932v1 |
|updated| 2024-09-12 10:56:38 UTC |
|summary| Graph path search is a classic computer science problem that has beenrecently approached with Reinforcement Learning RL due to its potential tooutperform prior methods. Existing RL techniques typically assume a global viewof the network which is not suitable for large-scale dynamic andprivacy-sensitive settings. An area of particular interest is search in socialnetworks due to its numerous applications. Inspired by seminal work inexperimental sociology which showed that decentralized yet efficient search ispossible in social networks we frame the problem as a collaborative taskbetween multiple agents equipped with a limited local view of the network. Wepropose a multi-agent approach for graph path search that successfullyleverages both homophily and structural heterogeneity. Our experiments carriedout over synthetic and real-world social networks demonstrate that our modelsignificantly outperforms learned and heuristic baselines. Furthermore ourresults show that meaningful embeddings for graph navigation can be constructedusing reward-driven learning. |


