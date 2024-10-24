# cs.CL 

| Item |Content|
| --- |---|
|idx| 2410.17251v1 |
|title| Altogether: Image Captioning via Re-aligning Alt-text |
|authors| Hu XuPo-Yao HuangXiaoqing Ellen TanChing-Feng YehJacob KahnChristine JouGargi GhoshOmer LevyLuke ZettlemoyerWen-tau YihShang-Wen LiSaining XieChristoph Feichtenhofer
|links| http://arxiv.org/abs/2410.17251v1 |
|updated| 2024-10-22 17:59:57 UTC |
|summary| This paper focuses on creating synthetic data to improve the quality of imagecaptions. Existing works typically have two shortcomings. First they captionimages from scratch ignoring existing alt-text metadata and second lacktransparency if the captioners training data e.g. GPT is unknown. In thispaper we study a principled approach Altogether based on the key idea to editand re-align existing alt-texts associated with the images. To generatetraining data we perform human annotation where annotators start with theexisting alt-text and re-align it to the image content in multiple roundsconsequently constructing captions with rich visual concepts. This differs fromprior work that carries out human annotation as a one-time description tasksolely based on images and annotator knowledge. We train a captioner on thisdata that generalizes the process of re-aligning alt-texts at scale. Ourresults show our Altogether approach leads to richer image captions that alsoimprove text-to-image generation and zero-shot image classification tasks. |


| Item |Content|
| --- |---|
|idx| 2410.17250v1 |
|title| JMMMU: A Japanese Massive Multi-discipline Multimodal Understanding Benchmark for Culture-aware Evaluation |
|authors| Shota OnoharaAtsuyuki MiyaiYuki ImajukuKazuki EgashiraJeonghun BaekXiang YueGraham NeubigKiyoharu Aizawa
|links| http://arxiv.org/abs/2410.17250v1 |
|updated| 2024-10-22 17:59:56 UTC |
|summary| Accelerating research on Large Multimodal Models LMMs in non-Englishlanguages is crucial for enhancing user experiences across broader populations.In this paper we introduce JMMMU Japanese MMMU the first large-scaleJapanese benchmark designed to evaluate LMMs on expert-level tasks based on theJapanese cultural context. To facilitate comprehensive culture-awareevaluation JMMMU features two complementary subsets: i culture-agnostic CAsubset where the culture-independent subjects e.g. Math are selected andtranslated into Japanese enabling one-to-one comparison with its Englishcounterpart MMMU and ii culture-specific CS subset comprising newlycrafted subjects that reflect Japanese cultural context. Using the CA subsetwe observe performance drop in many LMMs when evaluated in Japanese which ispurely attributable to language variation. Using the CS subset we reveal theirinadequate Japanese cultural understanding. Further by combining both subsetswe identify that some LMMs perform well on the CA subset but not on the CSsubset exposing a shallow understanding of the Japanese language that lacksdepth in cultural understanding. We hope this work will not only help advanceLMM performance in Japanese but also serve as a guideline to createhigh-standard culturally diverse benchmarks for multilingual LMM development.The project page is https://mmmu-japanese-benchmark.github.io/JMMMU/. |


| Item |Content|
| --- |---|
|idx| 2410.17247v1 |
|title| PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction |
|authors| Long XingQidong HuangXiaoyi DongJiajie LuPan ZhangYuhang ZangYuhang CaoConghui HeJiaqi WangFeng WuDahua Lin
|links| http://arxiv.org/abs/2410.17247v1 |
|updated| 2024-10-22 17:59:53 UTC |
|summary| In large vision-language models LVLMs images serve as inputs that carry awealth of information. As the idiom A picture is worth a thousand wordsimplies representing a single image in current LVLMs can require hundreds oreven thousands of tokens. This results in significant computational costswhich grow quadratically as input image resolution increases thereby severelyimpacting the efficiency of both training and inference. Previous approacheshave attempted to reduce the number of image tokens either before or within theearly layers of LVLMs. However these strategies inevitably result in the lossof crucial image information ultimately diminishing model performance. Toaddress this challenge we conduct an empirical study revealing that all visualtokens are necessary for LVLMs in the shallow layers and token redundancyprogressively increases in the deeper layers of the model. To this end wepropose PyramidDrop a visual redundancy reduction strategy for LVLMs to boosttheir efficiency in both training and inference with neglectable performanceloss. Specifically we partition the LVLM into several stages and drop part ofthe image tokens at the end of each stage with a pre-defined ratio creatingpyramid-like visual tokens across model layers. The dropping is based on alightweight similarity calculation with a negligible time overhead. Extensiveexperiments demonstrate that PyramidDrop can achieve a 40 training time and55 inference FLOPs acceleration of LLaVA-NeXT with comparable performance.Besides the PyramidDrop could also serve as a plug-and-play strategy forinference acceleration without training with better performance and lowerinference cost than counterparts. We hope that the insights and approachintroduced by PyramidDrop will inspire future research to further investigatethe role of image tokens in LVLMs. |


| Item |Content|
| --- |---|
|idx| 2410.17245v1 |
|title| Towards Reliable Evaluation of Behavior Steering Interventions in LLMs |
|authors| Itamar PresLaura RuisEkdeep Singh LubanaDavid Krueger
|links| http://arxiv.org/abs/2410.17245v1 |
|updated| 2024-10-22 17:59:39 UTC |
|summary| Representation engineering methods have recently shown promise for enablingefficient steering of model behavior. However evaluation pipelines for thesemethods have primarily relied on subjective demonstrations instead ofquantitative objective metrics. We aim to take a step towards addressing thisissue by advocating for four properties missing from current evaluations: icontexts sufficiently similar to downstream tasks should be used for assessingintervention quality ii model likelihoods should be accounted for iiievaluations should allow for standardized comparisons across different targetbehaviors and iv baseline comparisons should be offered. We introduce anevaluation pipeline grounded in these criteria offering both a quantitativeand visual analysis of how effectively a given method works. We use thispipeline to evaluate two representation engineering methods on how effectivelythey can steer behaviors such as truthfulness and corrigibility finding thatsome interventions are less effective than previously reported. |


| Item |Content|
| --- |---|
|idx| 2410.17238v1 |
|title| SELA: Tree-Search Enhanced LLM Agents for Automated Machine Learning |
|authors| Yizhou ChiYizhang LinSirui HongDuyi PanYaying FeiGuanghao MeiBangbang LiuTianqi PangJacky KwokCeyao ZhangBang LiuChenglin Wu
|links| http://arxiv.org/abs/2410.17238v1 |
|updated| 2024-10-22 17:56:08 UTC |
|summary| Automated Machine Learning AutoML approaches encompass traditional methodsthat optimize fixed pipelines for model selection and ensembling as well asnewer LLM-based frameworks that autonomously build pipelines. While LLM-basedagents have shown promise in automating machine learning tasks they oftengenerate low-diversity and suboptimal code even after multiple iterations. Toovercome these limitations we introduce Tree-Search Enhanced LLM AgentsSELA an innovative agent-based system that leverages Monte Carlo Tree SearchMCTS to optimize the AutoML process. By representing pipeline configurationsas trees our framework enables agents to conduct experiments intelligently anditeratively refine their strategies facilitating a more effective explorationof the machine learning solution space. This novel approach allows SELA todiscover optimal pathways based on experimental feedback improving the overallquality of the solutions. In an extensive evaluation across 20 machine learningdatasets we compare the performance of traditional and agent-based AutoMLmethods demonstrating that SELA achieves a win rate of 65 to 80 against eachbaseline across all datasets. These results underscore the significantpotential of agent-based strategies in AutoML offering a fresh perspective ontackling complex machine learning challenges. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2410.17250v1 |
|title| JMMMU: A Japanese Massive Multi-discipline Multimodal Understanding Benchmark for Culture-aware Evaluation |
|authors| Shota OnoharaAtsuyuki MiyaiYuki ImajukuKazuki EgashiraJeonghun BaekXiang YueGraham NeubigKiyoharu Aizawa
|links| http://arxiv.org/abs/2410.17250v1 |
|updated| 2024-10-22 17:59:56 UTC |
|summary| Accelerating research on Large Multimodal Models LMMs in non-Englishlanguages is crucial for enhancing user experiences across broader populations.In this paper we introduce JMMMU Japanese MMMU the first large-scaleJapanese benchmark designed to evaluate LMMs on expert-level tasks based on theJapanese cultural context. To facilitate comprehensive culture-awareevaluation JMMMU features two complementary subsets: i culture-agnostic CAsubset where the culture-independent subjects e.g. Math are selected andtranslated into Japanese enabling one-to-one comparison with its Englishcounterpart MMMU and ii culture-specific CS subset comprising newlycrafted subjects that reflect Japanese cultural context. Using the CA subsetwe observe performance drop in many LMMs when evaluated in Japanese which ispurely attributable to language variation. Using the CS subset we reveal theirinadequate Japanese cultural understanding. Further by combining both subsetswe identify that some LMMs perform well on the CA subset but not on the CSsubset exposing a shallow understanding of the Japanese language that lacksdepth in cultural understanding. We hope this work will not only help advanceLMM performance in Japanese but also serve as a guideline to createhigh-standard culturally diverse benchmarks for multilingual LMM development.The project page is https://mmmu-japanese-benchmark.github.io/JMMMU/. |


| Item |Content|
| --- |---|
|idx| 2410.17248v1 |
|title| HyperspectralViTs: Fast and Accurate methane detection on-board satellites |
|authors| Vít RůžičkaAndrew Markham
|links| http://arxiv.org/abs/2410.17248v1 |
|updated| 2024-10-22 17:59:55 UTC |
|summary| On-board processing of hyperspectral data with machine learning models wouldenable unprecedented amount of autonomy for a wide range of tasks for examplemethane detection or mineral identification. Methane is the second mostimportant greenhouse gas contributor to climate change and its automateddetection on-board of satellites using machine learning models would allow forearly warning system and could enable new capabilities such as automatedscheduling inside constellations of satellites. Classical methods for methanedetection suffer from high false positive rates and previous deep learningmodels exhibit prohibitive computational requirements. We propose fast andaccurate machine learning architectures which support end-to-end training withdata of high spectral dimension. We evaluate our models on two tasks related tohyperspectral data processing - methane leak detection and mineralidentification. With our proposed general architectures we improve the F1score of the previous methane detection state-of-the-art models by more than27 on a newly created synthetic dataset and by almost 13 on the previouslyreleased large benchmark dataset. We also demonstrate that training models onthe synthetic dataset improves performance of models finetuned on the datasetof real events by 6.9 in F1 score in contrast with training from scratch. On anewly created dataset for mineral identification our models provide 3.5improvement in the F1 score in contrast to the default versions of the models.With our proposed models we improve the inference speed by 85.19 in contrastto previous classical and deep learning approaches by removing the dependencyon classically computed features. Namely one capture from the EMIT sensor canbe processed in only 30 seconds on a realistic proxy hardware used on theION-SCV 004 satellite. |


| Item |Content|
| --- |---|
|idx| 2410.17246v1 |
|title| Learning Precise, Contact-Rich Manipulation through Uncalibrated Tactile Skins |
|authors| Venkatesh PattabiramanYifeng CaoSiddhant HaldarLerrel PintoRaunaq Bhirangi
|links| http://arxiv.org/abs/2410.17246v1 |
|updated| 2024-10-22 17:59:49 UTC |
|summary| While visuomotor policy learning has advanced robotic manipulation preciselyexecuting contact-rich tasks remains challenging due to the limitations ofvision in reasoning about physical interactions. To address this recent workhas sought to integrate tactile sensing into policy learning. However manyexisting approaches rely on optical tactile sensors that are either restrictedto recognition tasks or require complex dimensionality reduction steps forpolicy learning. In this work we explore learning policies with magnetic skinsensors which are inherently low-dimensional highly sensitive andinexpensive to integrate with robotic platforms. To leverage these sensorseffectively we present the Visuo-Skin ViSk framework a simple approach thatuses a transformer-based policy and treats skin sensor data as additionaltokens alongside visual information. Evaluated on four complex real-world tasksinvolving credit card swiping plug insertion USB insertion and bookshelfretrieval ViSk significantly outperforms both vision-only and optical tactilesensing based policies. Further analysis reveals that combining tactile andvisual modalities enhances policy performance and spatial generalizationachieving an average improvement of 27.5 across tasks.https://visuoskin.github.io/ |


| Item |Content|
| --- |---|
|idx| 2410.17245v1 |
|title| Towards Reliable Evaluation of Behavior Steering Interventions in LLMs |
|authors| Itamar PresLaura RuisEkdeep Singh LubanaDavid Krueger
|links| http://arxiv.org/abs/2410.17245v1 |
|updated| 2024-10-22 17:59:39 UTC |
|summary| Representation engineering methods have recently shown promise for enablingefficient steering of model behavior. However evaluation pipelines for thesemethods have primarily relied on subjective demonstrations instead ofquantitative objective metrics. We aim to take a step towards addressing thisissue by advocating for four properties missing from current evaluations: icontexts sufficiently similar to downstream tasks should be used for assessingintervention quality ii model likelihoods should be accounted for iiievaluations should allow for standardized comparisons across different targetbehaviors and iv baseline comparisons should be offered. We introduce anevaluation pipeline grounded in these criteria offering both a quantitativeand visual analysis of how effectively a given method works. We use thispipeline to evaluate two representation engineering methods on how effectivelythey can steer behaviors such as truthfulness and corrigibility finding thatsome interventions are less effective than previously reported. |


| Item |Content|
| --- |---|
|idx| 2410.17238v1 |
|title| SELA: Tree-Search Enhanced LLM Agents for Automated Machine Learning |
|authors| Yizhou ChiYizhang LinSirui HongDuyi PanYaying FeiGuanghao MeiBangbang LiuTianqi PangJacky KwokCeyao ZhangBang LiuChenglin Wu
|links| http://arxiv.org/abs/2410.17238v1 |
|updated| 2024-10-22 17:56:08 UTC |
|summary| Automated Machine Learning AutoML approaches encompass traditional methodsthat optimize fixed pipelines for model selection and ensembling as well asnewer LLM-based frameworks that autonomously build pipelines. While LLM-basedagents have shown promise in automating machine learning tasks they oftengenerate low-diversity and suboptimal code even after multiple iterations. Toovercome these limitations we introduce Tree-Search Enhanced LLM AgentsSELA an innovative agent-based system that leverages Monte Carlo Tree SearchMCTS to optimize the AutoML process. By representing pipeline configurationsas trees our framework enables agents to conduct experiments intelligently anditeratively refine their strategies facilitating a more effective explorationof the machine learning solution space. This novel approach allows SELA todiscover optimal pathways based on experimental feedback improving the overallquality of the solutions. In an extensive evaluation across 20 machine learningdatasets we compare the performance of traditional and agent-based AutoMLmethods demonstrating that SELA achieves a win rate of 65 to 80 against eachbaseline across all datasets. These results underscore the significantpotential of agent-based strategies in AutoML offering a fresh perspective ontackling complex machine learning challenges. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2410.17242v1 |
|title| LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias |
|authors| Haian JinHanwen JiangHao TanKai ZhangSai BiTianyuan ZhangFujun LuanNoah SnavelyZexiang Xu
|links| http://arxiv.org/abs/2410.17242v1 |
|updated| 2024-10-22 17:58:28 UTC |
|summary| We propose the Large View Synthesis Model LVSM a novel transformer-basedapproach for scalable and generalizable novel view synthesis from sparse-viewinputs. We introduce two architectures: 1 an encoder-decoder LVSM whichencodes input image tokens into a fixed number of 1D latent tokens functioningas a fully learned scene representation and decodes novel-view images fromthem and 2 a decoder-only LVSM which directly maps input images tonovel-view outputs completely eliminating intermediate scene representations.Both models bypass the 3D inductive biases used in previous methods -- from 3Drepresentations e.g. NeRF 3DGS to network designs e.g. epipolarprojections plane sweeps -- addressing novel view synthesis with a fullydata-driven approach. While the encoder-decoder model offers faster inferencedue to its independent latent representation the decoder-only LVSM achievessuperior quality scalability and zero-shot generalization outperformingprevious state-of-the-art methods by 1.5 to 3.5 dB PSNR. Comprehensiveevaluations across multiple datasets demonstrate that both LVSM variantsachieve state-of-the-art novel view synthesis quality. Notably our modelssurpass all previous methods even with reduced computational resources 1-2GPUs. Please see our website for more details:https://haian-jin.github.io/projects/LVSM/ . |


| Item |Content|
| --- |---|
|idx| 2410.17238v1 |
|title| SELA: Tree-Search Enhanced LLM Agents for Automated Machine Learning |
|authors| Yizhou ChiYizhang LinSirui HongDuyi PanYaying FeiGuanghao MeiBangbang LiuTianqi PangJacky KwokCeyao ZhangBang LiuChenglin Wu
|links| http://arxiv.org/abs/2410.17238v1 |
|updated| 2024-10-22 17:56:08 UTC |
|summary| Automated Machine Learning AutoML approaches encompass traditional methodsthat optimize fixed pipelines for model selection and ensembling as well asnewer LLM-based frameworks that autonomously build pipelines. While LLM-basedagents have shown promise in automating machine learning tasks they oftengenerate low-diversity and suboptimal code even after multiple iterations. Toovercome these limitations we introduce Tree-Search Enhanced LLM AgentsSELA an innovative agent-based system that leverages Monte Carlo Tree SearchMCTS to optimize the AutoML process. By representing pipeline configurationsas trees our framework enables agents to conduct experiments intelligently anditeratively refine their strategies facilitating a more effective explorationof the machine learning solution space. This novel approach allows SELA todiscover optimal pathways based on experimental feedback improving the overallquality of the solutions. In an extensive evaluation across 20 machine learningdatasets we compare the performance of traditional and agent-based AutoMLmethods demonstrating that SELA achieves a win rate of 65 to 80 against eachbaseline across all datasets. These results underscore the significantpotential of agent-based strategies in AutoML offering a fresh perspective ontackling complex machine learning challenges. |


| Item |Content|
| --- |---|
|idx| 2410.17234v1 |
|title| Fine-Tuning Large Language Models to Appropriately Abstain with Semantic Entropy |
|authors| Benedict Aaron TjandraMuhammed RazzakJannik KossenKunal HandaYarin Gal
|links| http://arxiv.org/abs/2410.17234v1 |
|updated| 2024-10-22 17:54:03 UTC |
|summary| Large Language Models LLMs are known to hallucinate whereby they generateplausible but inaccurate text. This phenomenon poses significant risks incritical applications such as medicine or law necessitating robusthallucination mitigation strategies. While recent works have proposedfine-tuning methods to teach LLMs to abstain from answering questions beyondtheir knowledge or capabilities these methods rely on the existence ofground-truth labels or are limited to short-form responses. To address theselimitations we propose fine-tuning using semantic entropy an uncertaintymeasure derived from introspection into the model which does not requireexternal labels. We demonstrate that our approach matches or outperforms modelsfine-tuned using prior work and achieves strong performance for both short andlong-form generations on a range of datasets. |


| Item |Content|
| --- |---|
|idx| 2410.17233v1 |
|title| Few-shot In-Context Preference Learning Using Large Language Models |
|authors| Chao YuHong LuJiaxuan GaoQixin TanXinting YangYu WangYi WuEugene Vinitsky
|links| http://arxiv.org/abs/2410.17233v1 |
|updated| 2024-10-22 17:53:34 UTC |
|summary| Designing reward functions is a core component of reinforcement learning butcan be challenging for truly complex behavior. Reinforcement Learning fromHuman Feedback RLHF has been used to alleviate this challenge by replacing ahand-coded reward function with a reward function learned from preferences.However it can be exceedingly inefficient to learn these rewards as they areoften learned tabula rasa. We investigate whether Large Language Models LLMscan reduce this query inefficiency by converting an iterative series of humanpreferences into code representing the rewards. We propose In-ContextPreference Learning ICPL a method that uses the grounding of an LLM toaccelerate learning reward functions from preferences. ICPL takes theenvironment context and task description synthesizes a set of rewardfunctions and then repeatedly updates the reward functions using humanrankings of videos of the resultant policies. Using synthetic preferences wedemonstrate that ICPL is orders of magnitude more efficient than RLHF and iseven competitive with methods that use ground-truth reward functions instead ofpreferences. Finally we perform a series of human preference-learning trialsand observe that ICPL extends beyond synthetic settings and can workeffectively with humans-in-the-loop. Additional information and videos areprovided at https://sites.google.com/view/few-shot-icpl/home. |


| Item |Content|
| --- |---|
|idx| 2410.17230v1 |
|title| Optimal Robust Estimation under Local and Global Corruptions: Stronger Adversary and Smaller Error |
|authors| Thanasis PittasAnkit Pensia
|links| http://arxiv.org/abs/2410.17230v1 |
|updated| 2024-10-22 17:51:23 UTC |
|summary| Algorithmic robust statistics has traditionally focused on the contaminationmodel where a small fraction of the samples are arbitrarily corrupted. Weconsider a recent contamination model that combines two kinds of corruptions:i small fraction of arbitrary outliers as in classical robust statisticsand ii local perturbations where samples may undergo bounded shifts onaverage. While each noise model is well understood individually the combinedcontamination model poses new algorithmic challenges with only partial resultsknown. Existing efficient algorithms are limited in two ways: i they workonly for a weak notion of local perturbations and ii they obtain suboptimalerror for isotropic subgaussian distributions among others. The latterlimitation led NGS24 COLT24 to hypothesize that improving the error mightin fact be computationally hard. Perhaps surprisingly we show thatinformation theoretically optimal error can indeed be achieved in polynomialtime under an even emphstronger local perturbation model thesliced-Wasserstein metric as opposed to the Wasserstein metric. Notably ouranalysis reveals that the entire family of stability-based robust meanestimators continues to work optimally in a black-box manner for the combinedcontamination model. This generalization is particularly useful in real-worldscenarios where the specific form of data corruption is not known in advance.We also present efficient algorithms for distribution learning and principalcomponent analysis in the combined contamination model. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2410.17251v1 |
|title| Altogether: Image Captioning via Re-aligning Alt-text |
|authors| Hu XuPo-Yao HuangXiaoqing Ellen TanChing-Feng YehJacob KahnChristine JouGargi GhoshOmer LevyLuke ZettlemoyerWen-tau YihShang-Wen LiSaining XieChristoph Feichtenhofer
|links| http://arxiv.org/abs/2410.17251v1 |
|updated| 2024-10-22 17:59:57 UTC |
|summary| This paper focuses on creating synthetic data to improve the quality of imagecaptions. Existing works typically have two shortcomings. First they captionimages from scratch ignoring existing alt-text metadata and second lacktransparency if the captioners training data e.g. GPT is unknown. In thispaper we study a principled approach Altogether based on the key idea to editand re-align existing alt-texts associated with the images. To generatetraining data we perform human annotation where annotators start with theexisting alt-text and re-align it to the image content in multiple roundsconsequently constructing captions with rich visual concepts. This differs fromprior work that carries out human annotation as a one-time description tasksolely based on images and annotator knowledge. We train a captioner on thisdata that generalizes the process of re-aligning alt-texts at scale. Ourresults show our Altogether approach leads to richer image captions that alsoimprove text-to-image generation and zero-shot image classification tasks. |


| Item |Content|
| --- |---|
|idx| 2410.17249v1 |
|title| SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes |
|authors| Cheng-De FanChen-Wei ChangYi-Ruei LiuJie-Ying LeeJiun-Long HuangYu-Chee TsengYu-Lun Liu
|links| http://arxiv.org/abs/2410.17249v1 |
|updated| 2024-10-22 17:59:56 UTC |
|summary| We present SpectroMotion a novel approach that combines 3D GaussianSplatting 3DGS with physically-based rendering PBR and deformation fieldsto reconstruct dynamic specular scenes. Previous methods extending 3DGS tomodel dynamic scenes have struggled to accurately represent specular surfaces.Our method addresses this limitation by introducing a residual correctiontechnique for accurate surface normal computation during deformationcomplemented by a deformable environment map that adapts to time-varyinglighting conditions. We implement a coarse-to-fine training strategy thatsignificantly enhances both scene geometry and specular color prediction. Wedemonstrate that our model outperforms prior methods for view synthesis ofscenes containing dynamic specular objects and that it is the only existing3DGS method capable of synthesizing photorealistic real-world dynamic specularscenes outperforming state-of-the-art methods in rendering complex dynamicand specular scenes. |


| Item |Content|
| --- |---|
|idx| 2410.17250v1 |
|title| JMMMU: A Japanese Massive Multi-discipline Multimodal Understanding Benchmark for Culture-aware Evaluation |
|authors| Shota OnoharaAtsuyuki MiyaiYuki ImajukuKazuki EgashiraJeonghun BaekXiang YueGraham NeubigKiyoharu Aizawa
|links| http://arxiv.org/abs/2410.17250v1 |
|updated| 2024-10-22 17:59:56 UTC |
|summary| Accelerating research on Large Multimodal Models LMMs in non-Englishlanguages is crucial for enhancing user experiences across broader populations.In this paper we introduce JMMMU Japanese MMMU the first large-scaleJapanese benchmark designed to evaluate LMMs on expert-level tasks based on theJapanese cultural context. To facilitate comprehensive culture-awareevaluation JMMMU features two complementary subsets: i culture-agnostic CAsubset where the culture-independent subjects e.g. Math are selected andtranslated into Japanese enabling one-to-one comparison with its Englishcounterpart MMMU and ii culture-specific CS subset comprising newlycrafted subjects that reflect Japanese cultural context. Using the CA subsetwe observe performance drop in many LMMs when evaluated in Japanese which ispurely attributable to language variation. Using the CS subset we reveal theirinadequate Japanese cultural understanding. Further by combining both subsetswe identify that some LMMs perform well on the CA subset but not on the CSsubset exposing a shallow understanding of the Japanese language that lacksdepth in cultural understanding. We hope this work will not only help advanceLMM performance in Japanese but also serve as a guideline to createhigh-standard culturally diverse benchmarks for multilingual LMM development.The project page is https://mmmu-japanese-benchmark.github.io/JMMMU/. |


| Item |Content|
| --- |---|
|idx| 2410.17247v1 |
|title| PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction |
|authors| Long XingQidong HuangXiaoyi DongJiajie LuPan ZhangYuhang ZangYuhang CaoConghui HeJiaqi WangFeng WuDahua Lin
|links| http://arxiv.org/abs/2410.17247v1 |
|updated| 2024-10-22 17:59:53 UTC |
|summary| In large vision-language models LVLMs images serve as inputs that carry awealth of information. As the idiom A picture is worth a thousand wordsimplies representing a single image in current LVLMs can require hundreds oreven thousands of tokens. This results in significant computational costswhich grow quadratically as input image resolution increases thereby severelyimpacting the efficiency of both training and inference. Previous approacheshave attempted to reduce the number of image tokens either before or within theearly layers of LVLMs. However these strategies inevitably result in the lossof crucial image information ultimately diminishing model performance. Toaddress this challenge we conduct an empirical study revealing that all visualtokens are necessary for LVLMs in the shallow layers and token redundancyprogressively increases in the deeper layers of the model. To this end wepropose PyramidDrop a visual redundancy reduction strategy for LVLMs to boosttheir efficiency in both training and inference with neglectable performanceloss. Specifically we partition the LVLM into several stages and drop part ofthe image tokens at the end of each stage with a pre-defined ratio creatingpyramid-like visual tokens across model layers. The dropping is based on alightweight similarity calculation with a negligible time overhead. Extensiveexperiments demonstrate that PyramidDrop can achieve a 40 training time and55 inference FLOPs acceleration of LLaVA-NeXT with comparable performance.Besides the PyramidDrop could also serve as a plug-and-play strategy forinference acceleration without training with better performance and lowerinference cost than counterparts. We hope that the insights and approachintroduced by PyramidDrop will inspire future research to further investigatethe role of image tokens in LVLMs. |


| Item |Content|
| --- |---|
|idx| 2410.17243v1 |
|title| Breaking the Memory Barrier: Near Infinite Batch Size Scaling for Contrastive Loss |
|authors| Zesen ChengHang ZhangKehan LiSicong LengZhiqiang HuFei WuDeli ZhaoXin LiLidong Bing
|links| http://arxiv.org/abs/2410.17243v1 |
|updated| 2024-10-22 17:59:30 UTC |
|summary| Contrastive loss is a powerful approach for representation learning wherelarger batch sizes enhance performance by providing more negative samples tobetter distinguish between similar and dissimilar data. However scaling batchsizes is constrained by the quadratic growth in GPU memory consumptionprimarily due to the full instantiation of the similarity matrix. To addressthis we propose a tile-based computation strategy that partitions thecontrastive loss calculation into arbitrary small blocks avoiding fullmaterialization of the similarity matrix. Furthermore we introduce amulti-level tiling strategy to leverage the hierarchical structure ofdistributed systems employing ring-based communication at the GPU level tooptimize synchronization and fused kernels at the CUDA core level to reduce I/Ooverhead. Experimental results show that the proposed method scales batch sizesto unprecedented levels. For instance it enables contrastive training of aCLIP-ViT-L/14 model with a batch size of 4M or 12M using 8 or 32 A800 80GBwithout sacrificing any accuracy. Compared to SOTA memory-efficient solutionsit achieves a two-order-of-magnitude reduction in memory while maintainingcomparable speed. The code will be made publicly available. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2410.17230v1 |
|title| Optimal Robust Estimation under Local and Global Corruptions: Stronger Adversary and Smaller Error |
|authors| Thanasis PittasAnkit Pensia
|links| http://arxiv.org/abs/2410.17230v1 |
|updated| 2024-10-22 17:51:23 UTC |
|summary| Algorithmic robust statistics has traditionally focused on the contaminationmodel where a small fraction of the samples are arbitrarily corrupted. Weconsider a recent contamination model that combines two kinds of corruptions:i small fraction of arbitrary outliers as in classical robust statisticsand ii local perturbations where samples may undergo bounded shifts onaverage. While each noise model is well understood individually the combinedcontamination model poses new algorithmic challenges with only partial resultsknown. Existing efficient algorithms are limited in two ways: i they workonly for a weak notion of local perturbations and ii they obtain suboptimalerror for isotropic subgaussian distributions among others. The latterlimitation led NGS24 COLT24 to hypothesize that improving the error mightin fact be computationally hard. Perhaps surprisingly we show thatinformation theoretically optimal error can indeed be achieved in polynomialtime under an even emphstronger local perturbation model thesliced-Wasserstein metric as opposed to the Wasserstein metric. Notably ouranalysis reveals that the entire family of stability-based robust meanestimators continues to work optimally in a black-box manner for the combinedcontamination model. This generalization is particularly useful in real-worldscenarios where the specific form of data corruption is not known in advance.We also present efficient algorithms for distribution learning and principalcomponent analysis in the combined contamination model. |


| Item |Content|
| --- |---|
|idx| 2410.17147v1 |
|title| Covariance estimation using Markov chain Monte Carlo |
|authors| Yunbum KookMatthew S. Zhang
|links| http://arxiv.org/abs/2410.17147v1 |
|updated| 2024-10-22 16:27:29 UTC |
|summary| We investigate the complexity of covariance matrix estimation for Gibbsdistributions based on dependent samples from a Markov chain. We show that whenpi satisfies a Poincare inequality and the chain possesses a spectral gapwe can achieve similar sample complexity using MCMC as compared to an estimatorconstructed using i.i.d. samples with potentially much better querycomplexity. As an application of our methods we show improvements for thequery complexity in both constrained and unconstrained settings for concreteinstances of MCMC. In particular we provide guarantees regarding isotropicrounding procedures for sampling uniformly on convex bodies. |


| Item |Content|
| --- |---|
|idx| 2410.17128v2 |
|title| Understanding Transfer Learning via Mean-field Analysis |
|authors| Gholamali AminianŁukasz SzpruchSamuel N. Cohen
|links| http://arxiv.org/abs/2410.17128v2 |
|updated| 2024-10-23 06:51:54 UTC |
|summary| We propose a novel framework for exploring generalization errors of transferlearning through the lens of differential calculus on the space of probabilitymeasures. In particular we consider two main transfer learning scenariosalpha-ERM and fine-tuning with the KL-regularized empirical riskminimization and establish generic conditions under which the generalizationerror and the population risk convergence rates for these scenarios arestudied. Based on our theoretical results we show the benefits of transferlearning with a one-hidden-layer neural network in the mean-field regime undersome suitable integrability and regularity assumptions on the loss andactivation functions. |


| Item |Content|
| --- |---|
|idx| 2410.17055v2 |
|title| Optimal Design for Reward Modeling in RLHF |
|authors| Antoine ScheidEtienne BoursierAlain DurmusMichael I. JordanPierre MénardEric MoulinesMichal Valko
|links| http://arxiv.org/abs/2410.17055v2 |
|updated| 2024-10-23 12:55:39 UTC |
|summary| Reinforcement Learning from Human Feedback RLHF has become a popularapproach to align language models LMs with human preferences. This methodinvolves collecting a large dataset of human pairwise preferences acrossvarious text generations and using it to infer implicitly or explicitly areward model. Numerous methods have been proposed to learn the reward model andalign a LM with it. However the costly process of collecting human preferenceshas received little attention and could benefit from theoretical insights. Thispaper addresses this issue and aims to formalize the reward training model inRLHF. We frame the selection of an effective dataset as a simple regretminimization task using a linear contextual dueling bandit method. Given thepotentially large number of arms this approach is more coherent than thebest-arm identification setting. We then propose an offline framework forsolving this problem. Under appropriate assumptions - linearity of the rewardmodel in the embedding space and boundedness of the reward parameter - wederive bounds on the simple regret. Finally we provide a lower bound thatmatches our upper bound up to constant and logarithmic terms. To our knowledgethis is the first theoretical contribution in this area to provide an offlineapproach as well as worst-case guarantees. |


| Item |Content|
| --- |---|
|idx| 2410.16901v1 |
|title| Bayes without Underfitting: Fully Correlated Deep Learning Posteriors via Alternating Projections |
|authors| Marco MianiHrittik RoySøren Hauberg
|links| http://arxiv.org/abs/2410.16901v1 |
|updated| 2024-10-22 11:15:07 UTC |
|summary| Bayesian deep learning all too often underfits so that the Bayesianprediction is less accurate than a simple point estimate. Uncertaintyquantification then comes at the cost of accuracy. For linearized models thenull space of the generalized Gauss-Newton matrix corresponds to parametersthat preserve the training predictions of the point estimate. We propose tobuild Bayesian approximations in this null space thereby guaranteeing that theBayesian predictive does not underfit. We suggest a matrix-free algorithm forprojecting onto this null space which scales linearly with the number ofparameters and quadratically with the number of output dimensions. We furtherpropose an approximation that only scales linearly with parameters to make themethod applicable to generative models. An extensive empirical evaluation showsthat the approach scales to large models including vision transformers with 28million parameters. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2410.17142v1 |
|title| Coniferest: a complete active anomaly detection framework |
|authors| M. V. KornilovV. S. KorolevK. L. MalanchevA. D. LavrukhinaE. RusseilT. A. SemenikhinE. GanglerE. E. O. IshidaM. V. PruzhinskayaA. A. VolnovaS. Sreejith
|links| http://arxiv.org/abs/2410.17142v1 |
|updated| 2024-10-22 16:19:13 UTC |
|summary| We present coniferest an open source generic purpose active anomalydetection framework written in Python. The package design and implementedalgorithms are described. Currently static outlier detection analysis issupported via the Isolation forest algorithm. Moreover Active AnomalyDiscovery AAD and Pineforest algorithms are available to tackle activeanomaly detection problems. The algorithms and package performance areevaluated on a series of synthetic datasets. We also describe a few successcases which resulted from applying the package to real astronomical data inactive anomaly detection tasks within the SNAD project. |


| Item |Content|
| --- |---|
|idx| 2410.17114v1 |
|title| Lunar Subterra: a Self-Integrative Unit with an Automated Drilling System |
|authors| Anthony SfeirAsya PetkovaSabine ChaayaKarina ChichovaMarta RossiAnna VockAlessandro MosutAkshayanivasini Ramasamy SaravanarajValentina SuminiTommy Nilsson
|links| http://arxiv.org/abs/2410.17114v1 |
|updated| 2024-10-22 15:40:44 UTC |
|summary| As humans venture deeper into space the need for a lunar settlement housingthe first group of settlers grows steadily. By means of new technologies suchas in situ resource utilisation ISRU as well as computational design thisgoal can be implemented in present years. Providing the first arrivals with animmediate underground habitat safe from radiation and other environmentalconstraints is of crucial importance to initialise a prolonged mission on theMoon. The projects proposal revolves around the idea of establishing a basewhich provides an immediately habitable space with the possibility for futureexpansion. Advanced construction methods and sustainable practices lay thegroundwork for a permanent human presence predominantly based on ISRU. Thispaper outlines a two-phase initiative aimed at the foundation of the LunarSubterra followed by an extension of the habitat above ground. Following ourcollaboration with the PoliSpace Sparc Student Association group a VirtualReality VR reproduction of the proposed habitat enabled quick iterativetesting of the habitable space with the use of a Meta Quest 2 headset. This notonly allowed an evaluation of the environment and its impact on human residentsbut also eradicated the need for tangible models to conceptualise the ideaenabling rapid user-centred design and implementation in the future of spaceexploration. |


| Item |Content|
| --- |---|
|idx| 2410.17099v1 |
|title| Human-LLM Hybrid Text Answer Aggregation for Crowd Annotations |
|authors| Jiyi Li
|links| http://arxiv.org/abs/2410.17099v1 |
|updated| 2024-10-22 15:22:58 UTC |
|summary| The quality is a crucial issue for crowd annotations. Answer aggregation isan important type of solution. The aggregated answers estimated from multiplecrowd answers to the same instance are the eventually collected annotationsrather than the individual crowd answers themselves. Recently the capabilityof Large Language Models LLMs on data annotation tasks has attracted interestfrom researchers. Most of the existing studies mainly focus on the averageperformance of individual crowd workers several recent works studied thescenarios of aggregation on categorical labels and LLMs used as label creators.However the scenario of aggregation on text answers and the role of LLMs asaggregators are not yet well-studied. In this paper we investigate thecapability of LLMs as aggregators in the scenario of close-ended crowd textanswer aggregation. We propose a human-LLM hybrid text answer aggregationmethod with a Creator-Aggregator Multi-Stage CAMS crowdsourcing framework. Wemake the experiments based on public crowdsourcing datasets. The results showthe effectiveness of our approach based on the collaboration of crowd workersand LLMs. |


| Item |Content|
| --- |---|
|idx| 2410.16959v1 |
|title| Evaluation of a Data Annotation Platform for Large, Time-Series Datasets in Intensive Care: Mixed Methods Study |
|authors| Marceli WacRaul Santos-RodriguezChris McWilliamsChristopher Bourdeaux
|links| http://arxiv.org/abs/2410.16959v1 |
|updated| 2024-10-22 12:38:56 UTC |
|summary| Intensive Care Units are complex data-rich environments where critically illpatients are treated using variety of clinical equipment. The data collectedusing this equipment can be used clinical staff to gain insight into thecondition of the patients and provide adequate treatment but it also providesample opportunity for applications in machine learning and data science. Whilethis data can frequently be used directly complex problems may requireadditional annotations to provide context and meaning before it could be usedto train the machine learning models. Annotating time-series datasets inclinical setting is a complex problem due to a large volume and complexity ofthe data time-consuming nature of the process and the fact that clinicianstime is in both high demand and short supply. In this study we present anevaluation of a bespoke tool designed to annotate large clinical time-seriesdatasets with staff from intensive care units. The software incorporates twomodes for annotation: by annotating individual admissions and by generatingrulesets which are applied to the entire dataset. Our study was split into twostages focusing on individual and semi-automated annotation and included 28annotators across both stages who utilised 50 clinical parameters to guidetheir annotations. We experienced significant challenges in recruitment andengagement of the participants in the annotation activities and developedinterventions which improved the participation over the course of the study.During the semi-automated annotation we observed preferences for differentparameter types measured vs. observed as well as relative agreement ofparticipants across shared admissions to the decision-tree model trained usingtheir rulesets. |


| Item |Content|
| --- |---|
|idx| 2410.16668v1 |
|title| Satori: Towards Proactive AR Assistant with Belief-Desire-Intention User Modeling |
|authors| Chenyi LiGuande WuGromit Yeuk-Yin ChanDishita G TurakhiaSonia Castelo QuispeDong LiLeslie WelchClaudio SilvaJing Qian
|links| http://arxiv.org/abs/2410.16668v1 |
|updated| 2024-10-22 03:53:46 UTC |
|summary| Augmented Reality assistance are increasingly popular for supporting userswith tasks like assembly and cooking. However current practice typicallyprovide reactive responses initialized from user requests lackingconsideration of rich contextual and user-specific information. To address thislimitation we propose a novel AR assistance system Satori that models bothuser states and environmental contexts to deliver proactive guidance. Oursystem combines the Belief-Desire-Intention BDI model with a state-of-the-artmulti-modal large language model LLM to infer contextually appropriateguidance. The design is informed by two formative studies involving twelveexperts. A sixteen within-subject study find that Satori achieves performancecomparable to an designer-created Wizard-of-Oz WoZ system without relying onmanual configurations or heuristics thereby enhancing generalizabilityreusability and opening up new possibilities for AR assistance. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2410.17221v1 |
|title| Scalable spectral representations for network multiagent control |
|authors| Zhaolin RenRunyuZhangBo DaiNa Li
|links| http://arxiv.org/abs/2410.17221v1 |
|updated| 2024-10-22 17:45:45 UTC |
|summary| Network Markov Decision Processes MDPs a popular model for multi-agentcontrol pose a significant challenge to efficient learning due to theexponential growth of the global state-action space with the number of agents.In this work utilizing the exponential decay property of network dynamics wefirst derive scalable spectral local representations for network MDPs whichinduces a network linear subspace for the local Q-function of each agent.Building on these local spectral representations we design a scalablealgorithmic framework for continuous state-action network MDPs and provideend-to-end guarantees for the convergence of our algorithm. Empirically wevalidate the effectiveness of our scalable representation-based approach on twobenchmark problems and demonstrate the advantages of our approach over genericfunction approximation approaches to representing the local Q-functions. |


| Item |Content|
| --- |---|
|idx| 2410.17068v1 |
|title| Delay-Constrained Grant-Free Random Access in MIMO Systems: Distributed Pilot Allocation and Power Control |
|authors| Jianan BaiZheng ChenErik. G. Larsson
|links| http://arxiv.org/abs/2410.17068v1 |
|updated| 2024-10-22 14:47:08 UTC |
|summary| We study a delay-constrained grant-free random access system with amulti-antenna base station. The users randomly generate data packets withexpiration deadlines which are then transmitted from data queues on a first-infirst-out basis. To deliver a packet a user needs to succeed in both randomaccess phase sending a pilot without collision and data transmission phaseachieving a required data rate with imperfect channel information before thepacket expires. We develop a distributed cross-layer policy that allows theusers to dynamically and independently choose their pilots and transmit powersto achieve a high effective sum throughput with fairness consideration. Ourpolicy design involves three key components: 1 a proxy of the instantaneousdata rate that depends only on macroscopic environment variables andtransmission decisions considering pilot collisions and imperfect channelestimation 2 a quantitative instantaneous measure of fairness within eachcommunication round and 3 a deep learning-based multi-agent controlframework with centralized training and distributed execution. The proposedframework benefits from an accurate differentiable objective function fortraining thereby achieving a higher sample efficiency compared with aconventional application of model-free multi-agent reinforcement learningalgorithms. The performance of the proposed approach is verified by simulationsunder highly dynamic and heterogeneous scenarios. |


| Item |Content|
| --- |---|
|idx| 2410.16686v1 |
|title| SERN: Simulation-Enhanced Realistic Navigation for Multi-Agent Robotic Systems in Contested Environments |
|authors| Jumman HossainEmon DeySnehalraj ChughMasud AhmedMS AnwarAbu-Zaher FarideeJason HoppesTheron TroutAnjon BasakRafidh ChowdhuryRishabh MistryHyun KimJade FreemanNiranjan SuriAdrienne RaglinCarl BusartTimothy GregoryAnuradha RaviNirmalya Roy
|links| http://arxiv.org/abs/2410.16686v1 |
|updated| 2024-10-22 04:35:57 UTC |
|summary| The increasing deployment of autonomous systems in complex environmentsnecessitates efficient communication and task completion among multiple agents.This paper presents SERN Simulation-Enhanced Realistic Navigation a novelframework integrating virtual and physical environments for real-timecollaborative decision-making in multi-robot systems. SERN addresses keychallenges in asset deployment and coordination through a bi-directionalcommunication framework using the AuroraXR ROS Bridge. Our approach advancesthe SOTA through accurate real-world representation in virtual environmentsusing Unity high-fidelity simulator synchronization of physical and virtualrobot movements efficient ROS data distribution between remote locations andintegration of SOTA semantic segmentation for enhanced environmentalperception. Our evaluations show a 15 to 24 improvement in latency and up toa 15 increase in processing efficiency compared to traditional ROS setups.Real-world and virtual simulation experiments with multiple robots demonstratesynchronization accuracy achieving less than 5 cm positional error and under2-degree rotational error. These results highlight SERNs potential to enhancesituational awareness and multi-agent coordination in diverse contestedenvironments. |


| Item |Content|
| --- |---|
|idx| 2410.16629v1 |
|title| Cutting Through the Confusion and Hype: Understanding the True Potential of Generative AI |
|authors| Ante ProdanJo-An OcchipintiRehez AhlipGoran UjdurHarris A. EyreKyle GoosenLuke PenzaMark Heffernan
|links| http://arxiv.org/abs/2410.16629v1 |
|updated| 2024-10-22 02:18:44 UTC |
|summary| This paper explores the nuanced landscape of generative AI genAIparticularly focusing on neural network-based models like Large Language ModelsLLMs. While genAI garners both optimistic enthusiasm and sceptical criticismthis work seeks to provide a balanced examination of its capabilitieslimitations and the profound impact it may have on societal functions andpersonal interactions. The first section demystifies language-based genAIthrough detailed discussions on how LLMs learn their computational needsdistinguishing features from supporting technologies and the inherentlimitations in their accuracy and reliability. Real-world examples illustratethe practical applications and implications of these technologies. The latterpart of the paper adopts a systems perspective evaluating how the integrationof LLMs with existing technologies can enhance productivity and addressemerging concerns. It highlights the need for significant investment tounderstand the implications of recent advancements advocating for awell-informed dialogue to ethically and responsibly integrate genAI intodiverse sectors. The paper concludes with prospective developments andrecommendations emphasizing a forward-looking approach to harnessing genAIspotential while mitigating its risks. |


| Item |Content|
| --- |---|
|idx| 2410.16600v1 |
|title| Convex Markov Games: A Framework for Fairness, Imitation, and Creativity in Multi-Agent Learning |
|authors| Ian GempAndreas HauptLuke MarrisSiqi LiuGeorgios Piliouras
|links| http://arxiv.org/abs/2410.16600v1 |
|updated| 2024-10-22 00:55:04 UTC |
|summary| Expert imitation behavioral diversity and fairness preferences give rise topreferences in sequential decision making domains that do not decomposeadditively across time. We introduce the class of convex Markov games thatallow general convex preferences over occupancy measures. Despite infinite timehorizon and strictly higher generality than Markov games pure strategy Nashequilibria exist under strict convexity. Furthermore equilibria can beapproximated efficiently by performing gradient descent on an upper bound ofexploitability. Our experiments imitate human choices in ultimatum gamesreveal novel solutions to the repeated prisoners dilemma and find fairsolutions in a repeated asymmetric coordination game. In the prisonersdilemma our algorithm finds a policy profile that deviates from observed humanplay only slightly yet achieves higher per-player utility while also beingthree orders of magnitude less exploitable. |


