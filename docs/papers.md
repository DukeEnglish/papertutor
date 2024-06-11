# cs.CL 

| Item |Content|
| --- |---|
|idx| 2406.06496v1 |
|title| Direct Preference Optimization for Suppressing Hallucinated Prior Exams in Radiology Report Generation |
|authors| Oishi BanerjeeHong-Yu ZhouSubathra AdithanStephen KwakKay WuPranav Rajpurkar
|links| http://arxiv.org/abs/2406.06496v1 |
|updated| 2024-06-10 17:31:36 UTC |
|summary| Recent advances in generative vision-language models VLMs have excitingpotential implications for AI in radiology yet VLMs are also known to producehallucinations nonsensical text and other unwanted behaviors that can wasteclinicians time and cause patient harm. Drawing on recent work on directpreference optimization DPO we propose a simple method for modifying thebehavior of pretrained VLMs performing radiology report generation bysuppressing unwanted types of generations. We apply our method to theprevention of hallucinations of prior exams addressing a long-establishedproblem behavior in models performing chest X-ray report generation. Across ourexperiments we find that DPO fine-tuning achieves a 3.2-4.8x reduction inlines hallucinating prior exams while maintaining model performance on clinicalaccuracy metrics. Our work is to the best of our knowledge the first work toapply DPO to medical VLMs providing a data- and compute- efficient way tosuppress problem behaviors while maintaining overall clinical accuracy. |


| Item |Content|
| --- |---|
|idx| 2406.06485v1 |
|title| Can Language Models Serve as Text-Based World Simulators? |
|authors| Ruoyao WangGraham ToddZiang XiaoXingdi YuanMarc-Alexandre CôtéPeter ClarkPeter Jansen
|links| http://arxiv.org/abs/2406.06485v1 |
|updated| 2024-06-10 17:24:44 UTC |
|summary| Virtual environments play a key role in benchmarking advances in complexplanning and decision-making tasks but are expensive and complicated to buildby hand. Can current language models themselves serve as world simulatorscorrectly predicting how actions change different world states thus bypassingthe need for extensive manual coding Our goal is to answer this question inthe context of text-based simulators. Our approach is to build and use a newbenchmark called ByteSized32-State-Prediction containing a dataset of textgame state transitions and accompanying game tasks. We use this to directlyquantify for the first time how well LLMs can serve as text-based worldsimulators. We test GPT-4 on this dataset and find that despite its impressiveperformance it is still an unreliable world simulator without furtherinnovations. This work thus contributes both new insights into current LLMscapabilities and weaknesses as well as a novel benchmark to track futureprogress as new models appear. |


| Item |Content|
| --- |---|
|idx| 2406.06484v1 |
|title| Parallelizing Linear Transformers with the Delta Rule over Sequence Length |
|authors| Songlin YangBailin WangYu ZhangYikang ShenYoon Kim
|links| http://arxiv.org/abs/2406.06484v1 |
|updated| 2024-06-10 17:24:42 UTC |
|summary| Transformers with linear attention i.e. linear transformers andstate-space models have recently been suggested as a viable linear-timealternative to transformers with softmax attention. However these models stillunderperform transformers especially on tasks that require in-contextretrieval. While more expressive variants of linear transformers which replacethe additive outer-product update in linear transformers with the delta rulehave been found to be more effective at associative recall existing algorithmsfor training such models do not parallelize over sequence length and are thusinefficient to train on modern hardware. This work describes ahardware-efficient algorithm for training linear transformers with the deltarule which exploits a memory-efficient representation for computing productsof Householder matrices. This algorithm allows us to scale up DeltaNet tostandard language modeling settings. We train a 1.3B model for 100B tokens andfind that it outperforms recent linear-time baselines such as Mamba and GLA interms of perplexity and zero-shot performance on downstream tasks including ontasks that focus on recall. We also experiment with two hybrid models whichcombine DeltaNet layers with 1 sliding-window attention layers every otherlayer or 2 two global attention layers and find that these hybrid modelsoutperform strong transformer baselines. |


| Item |Content|
| --- |---|
|idx| 2406.06474v1 |
|title| Towards a Personal Health Large Language Model |
|authors| Justin CosentinoAnastasiya BelyaevaXin LiuNicholas A. FurlotteZhun YangChace LeeErik SchenckYojan PatelJian CuiLogan Douglas SchneiderRobby BryantRyan G. GomesAllen JiangRoy LeeYun LiuJavier PerezJameson K. RogersCathy SpeedShyam TailorMegan WalkerJeffrey YuTim AlthoffConor HeneghanJohn HernandezMark MalhotraLeor SternYossi MatiasGreg S. CorradoShwetak PatelShravya ShettyJiening ZhanShruthi PrabhakaraDaniel McDuffCory Y. McLean
|links| http://arxiv.org/abs/2406.06474v1 |
|updated| 2024-06-10 17:16:49 UTC |
|summary| In health most large language model LLM research has focused on clinicaltasks. However mobile and wearable devices which are rarely integrated intosuch tasks provide rich longitudinal data for personal health monitoring.Here we present Personal Health Large Language Model PH-LLM fine-tuned fromGemini for understanding and reasoning over numerical time-series personalhealth data. We created and curated three datasets that test 1 production ofpersonalized insights and recommendations from sleep patterns physicalactivity and physiological responses 2 expert domain knowledge and 3prediction of self-reported sleep outcomes. For the first task we designed 857case studies in collaboration with domain experts to assess real-worldscenarios in sleep and fitness. Through comprehensive evaluation ofdomain-specific rubrics we observed that Gemini Ultra 1.0 and PH-LLM are notstatistically different from expert performance in fitness and while expertsremain superior for sleep fine-tuning PH-LLM provided significant improvementsin using relevant domain knowledge and personalizing information for sleepinsights. We evaluated PH-LLM domain knowledge using multiple choice sleepmedicine and fitness examinations. PH-LLM achieved 79 on sleep and 88 onfitness exceeding average scores from a sample of human experts. Finally wetrained PH-LLM to predict self-reported sleep quality outcomes from textual andmultimodal encoding representations of wearable data and demonstrate thatmultimodal encoding is required to match performance of specializeddiscriminative models. Although further development and evaluation arenecessary in the safety-critical personal health domain these resultsdemonstrate both the broad knowledge and capabilities of Gemini models and thebenefit of contextualizing physiological data for personal health applicationsas done with PH-LLM. |


| Item |Content|
| --- |---|
|idx| 2406.06469v1 |
|title| Husky: A Unified, Open-Source Language Agent for Multi-Step Reasoning |
|authors| Joongwon KimBhargavi ParanjapeTushar KhotHannaneh Hajishirzi
|links| http://arxiv.org/abs/2406.06469v1 |
|updated| 2024-06-10 17:07:25 UTC |
|summary| Language agents perform complex tasks by using tools to execute each stepprecisely. However most existing agents are based on proprietary models ordesigned to target specific tasks such as mathematics or multi-hop questionanswering. We introduce Husky a holistic open-source language agent thatlearns to reason over a unified action space to address a diverse set ofcomplex tasks involving numerical tabular and knowledge-based reasoning.Husky iterates between two stages: 1 generating the next action to taketowards solving a given task and 2 executing the action using expert modelsand updating the current solution state. We identify a thorough ontology ofactions for addressing complex tasks and curate high-quality data to trainexpert models for executing these actions. Our experiments show that Huskyoutperforms prior language agents across 14 evaluation datasets. Moreover weintroduce HuskyQA a new evaluation set which stress tests language agents formixed-tool reasoning with a focus on retrieving missing knowledge andperforming numerical reasoning. Despite using 7B models Husky matches or evenexceeds frontier LMs such as GPT-4 on these tasks showcasing the efficacy ofour holistic approach in addressing complex reasoning problems. Our code andmodels are available at https://github.com/agent-husky/Husky-v1. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2406.06527v1 |
|title| IllumiNeRF: 3D Relighting without Inverse Rendering |
|authors| Xiaoming ZhaoPratul P. SrinivasanDor VerbinKeunhong ParkRicardo Martin BruallaPhilipp Henzler
|links| http://arxiv.org/abs/2406.06527v1 |
|updated| 2024-06-10 17:59:59 UTC |
|summary| Existing methods for relightable view synthesis -- using a set of images ofan object under unknown lighting to recover a 3D representation that can berendered from novel viewpoints under a target illumination -- are based oninverse rendering and attempt to disentangle the object geometry materialsand lighting that explain the input images. Furthermore this typicallyinvolves optimization through differentiable Monte Carlo rendering which isbrittle and computationally-expensive. In this work we propose a simplerapproach: we first relight each input image using an image diffusion modelconditioned on lighting and then reconstruct a Neural Radiance Field NeRFwith these relit images from which we render novel views under the targetlighting. We demonstrate that this strategy is surprisingly competitive andachieves state-of-the-art results on multiple relighting benchmarks. Please seeour project page at https://illuminerf.github.io/. |


| Item |Content|
| --- |---|
|idx| 2406.06520v1 |
|title| Decentralized Personalized Federated Learning |
|authors| Salma KharratMarco CaniniSamuel Horvath
|links| http://arxiv.org/abs/2406.06520v1 |
|updated| 2024-06-10 17:58:48 UTC |
|summary| This work tackles the challenges of data heterogeneity and communicationlimitations in decentralized federated learning. We focus on creating acollaboration graph that guides each client in selecting suitable collaboratorsfor training personalized models that leverage their local data effectively.Our approach addresses these issues through a novel communication-efficientstrategy that enhances resource efficiency. Unlike traditional methods ourformulation identifies collaborators at a granular level by consideringcombinatorial relations of clients enhancing personalization while minimizingcommunication overhead. We achieve this through a bi-level optimizationframework that employs a constrained greedy algorithm resulting in aresource-efficient collaboration graph for personalized learning. Extensiveevaluation against various baselines across diverse datasets demonstrates thesuperiority of our method named DPFL. DPFL consistently outperforms otherapproaches showcasing its effectiveness in handling real-world dataheterogeneity minimizing communication overhead enhancing resourceefficiency and building personalized models in decentralized federatedlearning scenarios. |


| Item |Content|
| --- |---|
|idx| 2406.06512v1 |
|title| Merlin: A Vision Language Foundation Model for 3D Computed Tomography |
|authors| Louis BlankemeierJoseph Paul CohenAshwin KumarDave Van VeenSyed Jamal Safdar GardeziMagdalini PaschaliZhihong ChenJean-Benoit DelbrouckEduardo ReisCesar TruytsChristian BluethgenMalte Engmann Kjeldskov JensenSophie OstmeierMaya VarmaJeya Maria Jose ValanarasuZhongnan FangZepeng HuoZaid NabulsiDiego ArdilaWei-Hung WengEdson Amaro JuniorNeera AhujaJason FriesNigam H. ShahAndrew JohnstonRobert D. BoutinAndrew WentlandCurtis P. LanglotzJason HomSergios GatidisAkshay S. Chaudhari
|links| http://arxiv.org/abs/2406.06512v1 |
|updated| 2024-06-10 17:53:01 UTC |
|summary| Over 85 million computed tomography CT scans are performed annually in theUS of which approximately one quarter focus on the abdomen. Given the currentradiologist shortage there is a large impetus to use artificial intelligenceto alleviate the burden of interpreting these complex imaging studies. Priorstate-of-the-art approaches for automated medical image interpretation leveragevision language models VLMs. However current medical VLMs are generallylimited to 2D images and short reports and do not leverage electronic healthrecord EHR data for supervision. We introduce Merlin - a 3D VLM that we trainusing paired CT scans 6 million images from 15331 CTs EHR diagnosis codes1.8 million codes and radiology reports 6 million tokens. We evaluateMerlin on 6 task types and 752 individual tasks. The non-adaptedoff-the-shelf tasks include zero-shot findings classification 31 findingsphenotype classification 692 phenotypes and zero-shot cross-modal retrievalimage to findings and image to impressions while model adapted tasks include5-year disease prediction 6 diseases radiology report generation and 3Dsemantic segmentation 20 organs. We perform internal validation on a test setof 5137 CTs and external validation on 7000 clinical CTs and on two publicCT datasets VerSe TotalSegmentator. Beyond these clinically-relevantevaluations we assess the efficacy of various network architectures andtraining strategies to depict that Merlin has favorable performance to existingtask-specific baselines. We derive data scaling laws to empirically assesstraining data needs for requisite downstream task performance. Furthermoreunlike conventional VLMs that require hundreds of GPUs for training we performall training on a single GPU. |


| Item |Content|
| --- |---|
|idx| 2406.06508v1 |
|title| Monkey See, Monkey Do: Harnessing Self-attention in Motion Diffusion for Zero-shot Motion Transfer |
|authors| Sigal RaabInbar GatNathan SalaGuy TevetRotem Shalev-ArkushinOhad FriedAmit H. BermanoDaniel Cohen-Or
|links| http://arxiv.org/abs/2406.06508v1 |
|updated| 2024-06-10 17:47:14 UTC |
|summary| Given the remarkable results of motion synthesis with diffusion models anatural question arises: how can we effectively leverage these models formotion editing Existing diffusion-based motion editing methods overlook theprofound potential of the prior embedded within the weights of pre-trainedmodels which enables manipulating the latent feature space hence theyprimarily center on handling the motion space. In this work we explore theattention mechanism of pre-trained motion diffusion models. We uncover theroles and interactions of attention elements in capturing and representingintricate human motion patterns and carefully integrate these elements totransfer a leader motion to a follower one while maintaining the nuancedcharacteristics of the follower resulting in zero-shot motion transfer.Editing features associated with selected motions allows us to confront achallenge observed in prior motion diffusion approaches which use generaldirectives e.g. text music for editing ultimately failing to convey subtlenuances effectively. Our work is inspired by how a monkey closely imitates whatit sees while maintaining its unique motion patterns hence we call it MonkeySee Monkey Do and dub it MoMo. Employing our technique enables accomplishingtasks such as synthesizing out-of-distribution motions style transfer andspatial editing. Furthermore diffusion inversion is seldom employed formotions as a result editing efforts focus on generated motions limiting theeditability of real ones. MoMo harnesses motion inversion extending itsapplication to both real and generated motions. Experimental results show theadvantage of our approach over the current art. In particular unlike methodstailored for specific applications through training our approach is applied atinference time requiring no training. Our webpage is athttps://monkeyseedocg.github.io. |


| Item |Content|
| --- |---|
|idx| 2406.06500v1 |
|title| Adaptive Opponent Policy Detection in Multi-Agent MDPs: Real-Time Strategy Switch Identification Using Running Error Estimation |
|authors| Mohidul Haque MridulMohammad Foysal KhanRedwan Ahmed RizveeMd Mosaddek Khan
|links| http://arxiv.org/abs/2406.06500v1 |
|updated| 2024-06-10 17:34:44 UTC |
|summary| In Multi-agent Reinforcement Learning MARL accurately perceivingopponents strategies is essential for both cooperative and adversarialcontexts particularly within dynamic environments. While Proximal PolicyOptimization PPO and related algorithms such as Actor-Critic with ExperienceReplay ACER Trust Region Policy Optimization TRPO and Deep DeterministicPolicy Gradient DDPG perform well in single-agent stationary environmentsthey suffer from high variance in MARL due to non-stationary and hiddenpolicies of opponents leading to diminished reward performance. Additionallyexisting methods in MARL face significant challenges including the need forinter-agent communication reliance on explicit reward information highcomputational demands and sampling inefficiencies. These issues render themless effective in continuous environments where opponents may abruptly changetheir policies without prior notice. Against this background we presentOPS-DeMo Online Policy Switch-Detection Model an online algorithm thatemploys dynamic error decay to detect changes in opponents policies. OPS-DeMocontinuously updates its beliefs using an Assumed Opponent Policy AOP Bankand selects corresponding responses from a pre-trained Response Policy Bank.Each response policy is trained against consistently strategizing opponentsreducing training uncertainty and enabling the effective use of algorithms likePPO in multi-agent environments. Comparative assessments show that our approachoutperforms PPO-trained models in dynamic scenarios like the Predator-Preysetting providing greater robustness to sudden policy shifts and enabling moreinformed decision-making through precise opponent policy insights. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2406.06520v1 |
|title| Decentralized Personalized Federated Learning |
|authors| Salma KharratMarco CaniniSamuel Horvath
|links| http://arxiv.org/abs/2406.06520v1 |
|updated| 2024-06-10 17:58:48 UTC |
|summary| This work tackles the challenges of data heterogeneity and communicationlimitations in decentralized federated learning. We focus on creating acollaboration graph that guides each client in selecting suitable collaboratorsfor training personalized models that leverage their local data effectively.Our approach addresses these issues through a novel communication-efficientstrategy that enhances resource efficiency. Unlike traditional methods ourformulation identifies collaborators at a granular level by consideringcombinatorial relations of clients enhancing personalization while minimizingcommunication overhead. We achieve this through a bi-level optimizationframework that employs a constrained greedy algorithm resulting in aresource-efficient collaboration graph for personalized learning. Extensiveevaluation against various baselines across diverse datasets demonstrates thesuperiority of our method named DPFL. DPFL consistently outperforms otherapproaches showcasing its effectiveness in handling real-world dataheterogeneity minimizing communication overhead enhancing resourceefficiency and building personalized models in decentralized federatedlearning scenarios. |


| Item |Content|
| --- |---|
|idx| 2406.06518v1 |
|title| Data Augmentation for Multivariate Time Series Classification: An Experimental Study |
|authors| Romain IlbertThai V. HoangZonghua Zhang
|links| http://arxiv.org/abs/2406.06518v1 |
|updated| 2024-06-10 17:58:02 UTC |
|summary| Our study investigates the impact of data augmentation on the performance ofmultivariate time series models focusing on datasets from the UCR archive.Despite the limited size of these datasets we achieved classification accuracyimprovements in 10 out of 13 datasets using the Rocket and InceptionTimemodels. This highlights the essential role of sufficient data in trainingeffective models paralleling the advancements seen in computer vision. Ourwork delves into adapting and applying existing methods in innovative ways tothe domain of multivariate time series classification. Our comprehensiveexploration of these techniques sets a new standard for addressing datascarcity in time series analysis emphasizing that diverse augmentationstrategies are crucial for unlocking the potential of both traditional and deeplearning models. Moreover by meticulously analyzing and applying a variety ofaugmentation techniques we demonstrate that strategic data enrichment canenhance model accuracy. This not only establishes a benchmark for futureresearch in time series analysis but also underscores the importance ofadopting varied augmentation approaches to improve model performance in theface of limited data availability. |


| Item |Content|
| --- |---|
|idx| 2406.06516v1 |
|title| Distribution-Free Predictive Inference under Unknown Temporal Drift |
|authors| Elise HanChengpiao HuangKaizheng Wang
|links| http://arxiv.org/abs/2406.06516v1 |
|updated| 2024-06-10 17:55:43 UTC |
|summary| Distribution-free prediction sets play a pivotal role in uncertaintyquantification for complex statistical models. Their validity hinges onreliable calibration data which may not be readily available as real-worldenvironments often undergo unknown changes over time. In this paper we proposea strategy for choosing an adaptive window and use the data therein toconstruct prediction sets. The window is selected by optimizing an estimatedbias-variance tradeoff. We provide sharp coverage guarantees for our methodshowing its adaptivity to the underlying temporal drift. We also illustrate itsefficacy through numerical experiments on synthetic and real data. |


| Item |Content|
| --- |---|
|idx| 2406.06514v1 |
|title| Random Features Approximation for Control-Affine Systems |
|authors| Kimia KazemianYahya SattarSarah Dean
|links| http://arxiv.org/abs/2406.06514v1 |
|updated| 2024-06-10 17:54:57 UTC |
|summary| Modern data-driven control applications call for flexible nonlinear modelsthat are amenable to principled controller synthesis and realtime feedback.Many nonlinear dynamical systems of interest are control affine. We propose twonovel classes of nonlinear feature representations which capture control affinestructure while allowing for arbitrary complexity in the state dependence. Ourmethods make use of random features RF approximations inheriting theexpressiveness of kernel methods at a lower computational cost. We formalizethe representational capabilities of our methods by showing their relationshipto the Affine Dot Product ADP kernel proposed by Castaneda et al. 2021and a novel Affine Dense AD kernel that we introduce. We further illustratethe utility by presenting a case study of data-driven optimization-basedcontrol using control certificate functions CCF. Simulation experiments on adouble pendulum empirically demonstrate the advantages of our methods. |


| Item |Content|
| --- |---|
|idx| 2406.06509v1 |
|title| Robust Distribution Learning with Local and Global Adversarial Corruptions |
|authors| Sloan NietertZiv GoldfeldSoroosh Shafiee
|links| http://arxiv.org/abs/2406.06509v1 |
|updated| 2024-06-10 17:48:36 UTC |
|summary| We consider learning in an adversarial environment where anvarepsilon-fraction of samples from a distribution P are arbitrarilymodified global corruptions and the remaining perturbations have averagemagnitude bounded by rho local corruptions. Given access to n suchcorrupted samples we seek a computationally efficient estimator hatP_nthat minimizes the Wasserstein distance mathsfW_1hatP_nP. In factwe attack the fine-grained task of minimizing mathsfW_1Pi_ hatP_nPi_ P for all orthogonal projections Pi in mathbbRd times dwith performance scaling with mathrmrankPi  k. This allows us toaccount simultaneously for mean estimation k1 distribution estimationkd as well as the settings interpolating between these two extremes. Wecharacterize the optimal population-limit risk for this task and then developan efficient finite-sample algorithm with error bounded by sqrtvarepsilonk  rho  dO1tildeOn-1/k when P has bounded moments of order2delta for constant delta  0. For data distributions with boundedcovariance our finite-sample bounds match the minimax population-level optimumfor large sample sizes. Our efficient procedure relies on a novel trace normapproximation of an ideal yet intractable 2-Wasserstein projection estimator.We apply this algorithm to robust stochastic optimization and in the processuncover a new method for overcoming the curse of dimensionality in Wassersteindistributionally robust optimization. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2406.06527v1 |
|title| IllumiNeRF: 3D Relighting without Inverse Rendering |
|authors| Xiaoming ZhaoPratul P. SrinivasanDor VerbinKeunhong ParkRicardo Martin BruallaPhilipp Henzler
|links| http://arxiv.org/abs/2406.06527v1 |
|updated| 2024-06-10 17:59:59 UTC |
|summary| Existing methods for relightable view synthesis -- using a set of images ofan object under unknown lighting to recover a 3D representation that can berendered from novel viewpoints under a target illumination -- are based oninverse rendering and attempt to disentangle the object geometry materialsand lighting that explain the input images. Furthermore this typicallyinvolves optimization through differentiable Monte Carlo rendering which isbrittle and computationally-expensive. In this work we propose a simplerapproach: we first relight each input image using an image diffusion modelconditioned on lighting and then reconstruct a Neural Radiance Field NeRFwith these relit images from which we render novel views under the targetlighting. We demonstrate that this strategy is surprisingly competitive andachieves state-of-the-art results on multiple relighting benchmarks. Please seeour project page at https://illuminerf.github.io/. |


| Item |Content|
| --- |---|
|idx| 2406.06526v1 |
|title| GaussianCity: Generative Gaussian Splatting for Unbounded 3D City Generation |
|authors| Haozhe XieZhaoxi ChenFangzhou HongZiwei Liu
|links| http://arxiv.org/abs/2406.06526v1 |
|updated| 2024-06-10 17:59:55 UTC |
|summary| 3D city generation with NeRF-based methods shows promising generation resultsbut is computationally inefficient. Recently 3D Gaussian Splatting 3D-GS hasemerged as a highly efficient alternative for object-level 3D generation.However adapting 3D-GS from finite-scale 3D objects and humans toinfinite-scale 3D cities is non-trivial. Unbounded 3D city generation entailssignificant storage overhead out-of-memory issues arising from the need toexpand points to billions often demanding hundreds of Gigabytes of VRAM for acity scene spanning 10km2. In this paper we propose GaussianCity agenerative Gaussian Splatting framework dedicated to efficiently synthesizingunbounded 3D cities with a single feed-forward pass. Our key insights aretwo-fold: 1 Compact 3D Scene Representation: We introduce BEV-Point as ahighly compact intermediate representation ensuring that the growth in VRAMusage for unbounded scenes remains constant thus enabling unbounded citygeneration. 2 Spatial-aware Gaussian Attribute Decoder: We presentspatial-aware BEV-Point decoder to produce 3D Gaussian attributes whichleverages Point Serializer to integrate the structural and contextualcharacteristics of BEV points. Extensive experiments demonstrate thatGaussianCity achieves state-of-the-art results in both drone-view andstreet-view 3D city generation. Notably compared to CityDreamer GaussianCityexhibits superior performance with a speedup of 60 times 10.72 FPS v.s. 0.18FPS. |


| Item |Content|
| --- |---|
|idx| 2406.06525v1 |
|title| Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation |
|authors| Peize SunYi JiangShoufa ChenShilong ZhangBingyue PengPing LuoZehuan Yuan
|links| http://arxiv.org/abs/2406.06525v1 |
|updated| 2024-06-10 17:59:52 UTC |
|summary| We introduce LlamaGen a new family of image generation models that applyoriginal next-token prediction paradigm of large language models to visualgeneration domain. It is an affirmative answer to whether vanillaautoregressive models e.g. Llama without inductive biases on visual signalscan achieve state-of-the-art image generation performance if scaling properly.We reexamine design spaces of image tokenizers scalability properties of imagegeneration models and their training data quality. The outcome of thisexploration consists of: 1 An image tokenizer with downsample ratio of 16reconstruction quality of 0.94 rFID and codebook usage of 97 on ImageNetbenchmark. 2 A series of class-conditional image generation models rangingfrom 111M to 3.1B parameters achieving 2.18 FID on ImageNet 256x256benchmarks outperforming the popular diffusion models such as LDM DiT. 3 Atext-conditional image generation model with 775M parameters from two-stagetraining on LAION-COCO and high aesthetics quality images demonstratingcompetitive performance of visual quality and text alignment. 4 We verify theeffectiveness of LLM serving frameworks in optimizing the inference speed ofimage generation models and achieve 326 - 414 speedup. We release all modelsand codes to facilitate open-source community of visual generation andmultimodal foundation models. |


| Item |Content|
| --- |---|
|idx| 2406.06523v1 |
|title| NaRCan: Natural Refined Canonical Image with Integration of Diffusion Prior for Video Editing |
|authors| Ting-Hsuan ChenJiewen ChanHau-Shiang ShiuShih-Han YenChang-Han YehYu-Lun Liu
|links| http://arxiv.org/abs/2406.06523v1 |
|updated| 2024-06-10 17:59:46 UTC |
|summary| We propose a video editing framework NaRCan which integrates a hybriddeformation field and diffusion prior to generate high-quality naturalcanonical images to represent the input video. Our approach utilizes homographyto model global motion and employs multi-layer perceptrons MLPs to capturelocal residual deformations enhancing the models ability to handle complexvideo dynamics. By introducing a diffusion prior from the early stages oftraining our model ensures that the generated images retain a high-qualitynatural appearance making the produced canonical images suitable for variousdownstream tasks in video editing a capability not achieved by currentcanonical-based methods. Furthermore we incorporate low-rank adaptation LoRAfine-tuning and introduce a noise and diffusion prior update schedulingtechnique that accelerates the training process by 14 times. Extensiveexperimental results show that our method outperforms existing approaches invarious video editing tasks and produces coherent and high-quality edited videosequences. See our project page for video results athttps://koi953215.github.io/NaRCan_page/. |


| Item |Content|
| --- |---|
|idx| 2406.06521v1 |
|title| PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction |
|authors| Danpeng ChenHai LiWeicai YeYifan WangWeijian XieShangjin ZhaiNan WangHaomin LiuHujun BaoGuofeng Zhang
|links| http://arxiv.org/abs/2406.06521v1 |
|updated| 2024-06-10 17:59:01 UTC |
|summary| Recently 3D Gaussian Splatting 3DGS has attracted widespread attention dueto its high-quality rendering and ultra-fast training and rendering speed.However due to the unstructured and irregular nature of Gaussian point cloudsit is difficult to guarantee geometric reconstruction accuracy and multi-viewconsistency simply by relying on image reconstruction loss. Although manystudies on surface reconstruction based on 3DGS have emerged recently thequality of their meshes is generally unsatisfactory. To address this problemwe propose a fast planar-based Gaussian splatting reconstruction representationPGSR to achieve high-fidelity surface reconstruction while ensuringhigh-quality rendering. Specifically we first introduce an unbiased depthrendering method which directly renders the distance from the camera origin tothe Gaussian plane and the corresponding normal map based on the Gaussiandistribution of the point cloud and divides the two to obtain the unbiaseddepth. We then introduce single-view geometric multi-view photometric andgeometric regularization to preserve global geometric accuracy. We also proposea camera exposure compensation model to cope with scenes with largeillumination variations. Experiments on indoor and outdoor scenes show that ourmethod achieves fast training and rendering while maintaining high-fidelityrendering and geometric reconstruction outperforming 3DGS-based and NeRF-basedmethods. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2406.06516v1 |
|title| Distribution-Free Predictive Inference under Unknown Temporal Drift |
|authors| Elise HanChengpiao HuangKaizheng Wang
|links| http://arxiv.org/abs/2406.06516v1 |
|updated| 2024-06-10 17:55:43 UTC |
|summary| Distribution-free prediction sets play a pivotal role in uncertaintyquantification for complex statistical models. Their validity hinges onreliable calibration data which may not be readily available as real-worldenvironments often undergo unknown changes over time. In this paper we proposea strategy for choosing an adaptive window and use the data therein toconstruct prediction sets. The window is selected by optimizing an estimatedbias-variance tradeoff. We provide sharp coverage guarantees for our methodshowing its adaptivity to the underlying temporal drift. We also illustrate itsefficacy through numerical experiments on synthetic and real data. |


| Item |Content|
| --- |---|
|idx| 2406.06514v1 |
|title| Random Features Approximation for Control-Affine Systems |
|authors| Kimia KazemianYahya SattarSarah Dean
|links| http://arxiv.org/abs/2406.06514v1 |
|updated| 2024-06-10 17:54:57 UTC |
|summary| Modern data-driven control applications call for flexible nonlinear modelsthat are amenable to principled controller synthesis and realtime feedback.Many nonlinear dynamical systems of interest are control affine. We propose twonovel classes of nonlinear feature representations which capture control affinestructure while allowing for arbitrary complexity in the state dependence. Ourmethods make use of random features RF approximations inheriting theexpressiveness of kernel methods at a lower computational cost. We formalizethe representational capabilities of our methods by showing their relationshipto the Affine Dot Product ADP kernel proposed by Castaneda et al. 2021and a novel Affine Dense AD kernel that we introduce. We further illustratethe utility by presenting a case study of data-driven optimization-basedcontrol using control certificate functions CCF. Simulation experiments on adouble pendulum empirically demonstrate the advantages of our methods. |


| Item |Content|
| --- |---|
|idx| 2406.06509v1 |
|title| Robust Distribution Learning with Local and Global Adversarial Corruptions |
|authors| Sloan NietertZiv GoldfeldSoroosh Shafiee
|links| http://arxiv.org/abs/2406.06509v1 |
|updated| 2024-06-10 17:48:36 UTC |
|summary| We consider learning in an adversarial environment where anvarepsilon-fraction of samples from a distribution P are arbitrarilymodified global corruptions and the remaining perturbations have averagemagnitude bounded by rho local corruptions. Given access to n suchcorrupted samples we seek a computationally efficient estimator hatP_nthat minimizes the Wasserstein distance mathsfW_1hatP_nP. In factwe attack the fine-grained task of minimizing mathsfW_1Pi_ hatP_nPi_ P for all orthogonal projections Pi in mathbbRd times dwith performance scaling with mathrmrankPi  k. This allows us toaccount simultaneously for mean estimation k1 distribution estimationkd as well as the settings interpolating between these two extremes. Wecharacterize the optimal population-limit risk for this task and then developan efficient finite-sample algorithm with error bounded by sqrtvarepsilonk  rho  dO1tildeOn-1/k when P has bounded moments of order2delta for constant delta  0. For data distributions with boundedcovariance our finite-sample bounds match the minimax population-level optimumfor large sample sizes. Our efficient procedure relies on a novel trace normapproximation of an ideal yet intractable 2-Wasserstein projection estimator.We apply this algorithm to robust stochastic optimization and in the processuncover a new method for overcoming the curse of dimensionality in Wassersteindistributionally robust optimization. |


| Item |Content|
| --- |---|
|idx| 2406.06506v1 |
|title| Online Newton Method for Bandit Convex Optimisation |
|authors| Hidde FokkemaDirk van der HoevenTor LattimoreJack J. Mayo
|links| http://arxiv.org/abs/2406.06506v1 |
|updated| 2024-06-10 17:44:11 UTC |
|summary| We introduce a computationally efficient algorithm for zeroth-order banditconvex optimisation and prove that in the adversarial setting its regret is atmost d3.5 sqrtn mathrmpolylogn d with high probability where dis the dimension and n is the time horizon. In the stochastic setting thebound improves to M d2 sqrtn mathrmpolylogn d where M ind-1/2 d-1 / 4 is a constant that depends on the geometry of theconstraint set and the desired computational properties. |


| Item |Content|
| --- |---|
|idx| 2406.06470v1 |
|title| GKAN: Graph Kolmogorov-Arnold Networks |
|authors| Mehrdad KiamariMohammad KiamariBhaskar Krishnamachari
|links| http://arxiv.org/abs/2406.06470v1 |
|updated| 2024-06-10 17:09:38 UTC |
|summary| We introduce Graph Kolmogorov-Arnold Networks GKAN an innovative neuralnetwork architecture that extends the principles of the recently proposedKolmogorov-Arnold Networks KAN to graph-structured data. By adopting theunique characteristics of KANs notably the use of learnable univariatefunctions instead of fixed linear weights we develop a powerful model forgraph-based learning tasks. Unlike traditional Graph Convolutional NetworksGCNs that rely on a fixed convolutional architecture GKANs implementlearnable spline-based functions between layers transforming the wayinformation is processed across the graph structure. We present two differentways to incorporate KAN layers into GKAN: architecture 1 -- where the learnablefunctions are applied to input features after aggregation and architecture 2 --where the learnable functions are applied to input features before aggregation.We evaluate GKAN empirically using a semi-supervised graph learning task on areal-world dataset Cora. We find that architecture generally performs better.We find that GKANs achieve higher accuracy in semi-supervised learning tasks ongraphs compared to the traditional GCN model. For example when considering 100features GCN provides an accuracy of 53.5 while a GKAN with a comparablenumber of parameters gives an accuracy of 61.76 with 200 features GCNprovides an accuracy of 61.24 while a GKAN with a comparable number ofparameters gives an accuracy of 67.66. We also present results on the impact ofvarious parameters such as the number of hidden nodes grid-size and thepolynomial-degree of the spline on the performance of GKAN. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2406.06499v1 |
|title| NarrativeBridge: Enhancing Video Captioning with Causal-Temporal Narrative |
|authors| Asmar NadeemFaegheh SardariRobert DawesSyed Sameed HusainAdrian HiltonArmin Mustafa
|links| http://arxiv.org/abs/2406.06499v1 |
|updated| 2024-06-10 17:34:24 UTC |
|summary| Existing video captioning benchmarks and models lack coherent representationsof causal-temporal narrative which is sequences of events linked through causeand effect unfolding over time and driven by characters or agents. This lackof narrative restricts models ability to generate text descriptions thatcapture the causal and temporal dynamics inherent in video content. To addressthis gap we propose NarrativeBridge an approach comprising of: 1 a novelCausal-Temporal Narrative CTN captions benchmark generated using a largelanguage model and few-shot prompting explicitly encoding cause-effecttemporal relationships in video descriptions evaluated automatically to ensurecaption quality and relevance and 2 a dedicated Cause-Effect Network CENarchitecture with separate encoders for capturing cause and effect dynamicsindependently enabling effective learning and generation of captions withcausal-temporal narrative. Extensive experiments demonstrate that CEN is moreaccurate in articulating the causal and temporal aspects of video content thanthe second best model GIT: 17.88 and 17.44 CIDEr on the MSVD and MSR-VTTdatasets respectively. The proposed framework understands and generatesnuanced text descriptions with intricate causal-temporal narrative structurespresent in videos addressing a critical limitation in video captioning. Forproject details visit https://narrativebridge.github.io/. |


| Item |Content|
| --- |---|
|idx| 2406.06498v1 |
|title| Demonstrating HumanTHOR: A Simulation Platform and Benchmark for Human-Robot Collaboration in a Shared Workspace |
|authors| Chenxu WangBoyuan DuJiaxin XuPeiyan LiDi GuoHuaping Liu
|links| http://arxiv.org/abs/2406.06498v1 |
|updated| 2024-06-10 17:33:44 UTC |
|summary| Human-robot collaboration HRC in a shared workspace has become a commonpattern in real-world robot applications and has garnered significant researchinterest. However most existing studies for human-in-the-loop HITLcollaboration with robots in a shared workspace evaluate in either simplifiedgame environments or physical platforms falling short in limited realisticsignificance or limited scalability. To support future studies we build anembodied framework named HumanTHOR which enables humans to act in thesimulation environment through VR devices to support HITL collaborations in ashared workspace. To validate our system we build a benchmark of everydaytasks and conduct a preliminary user study with two baseline algorithms. Theresults show that the robot can effectively assist humans in collaborationdemonstrating the significance of HRC. The comparison among different levels ofbaselines affirms that our system can adequately evaluate robot capabilitiesand serve as a benchmark for different robot algorithms. The experimentalresults also indicate that there is still much room in the area and our systemcan provide a preliminary foundation for future HRC research in a sharedworkspace. More information about the simulation environment experimentvideos benchmark descriptions and additional supplementary materials can befound on the website: https://sites.google.com/view/humanthor/. |


| Item |Content|
| --- |---|
|idx| 2406.06451v1 |
|title| Insights from Social Shaping Theory: The Appropriation of Large Language Models in an Undergraduate Programming Course |
|authors| Aadarsh PadiyathXinying HouAmy PangDiego Viramontes VargasXingjian GuTamara Nelson-FrommZihan WuMark GuzdialBarbara Ericson
|links| http://dx.doi.org/10.1145/3632620.3671098 |
|updated| 2024-06-10 16:40:14 UTC |
|summary| The capability of large language models LLMs to generate debug andexplain code has sparked the interest of researchers and educators inundergraduate programming with many anticipating their transformativepotential in programming education. However decisions about why and how to useLLMs in programming education may involve more than just the assessment of anLLMs technical capabilities. Using the social shaping of technology theory asa guiding framework our study explores how students social perceptionsinfluence their own LLM usage. We then examine the correlation of self-reportedLLM usage with students self-efficacy and midterm performances in anundergraduate programming course. Triangulating data from an anonymousend-of-course student survey n  158 a mid-course self-efficacy surveyn158 student interviews n  10 self-reported LLM usage on homework andmidterm performances we discovered that students use of LLMs was associatedwith their expectations for their future careers and their perceptions of peerusage. Additionally early self-reported LLM usage in our context correlatedwith lower self-efficacy and lower midterm scores while students perceivedover-reliance on LLMs rather than their usage itself correlated withdecreased self-efficacy later in the course. |


| Item |Content|
| --- |---|
|idx| 2406.06448v1 |
|title| How is the Pilot Doing: VTOL Pilot Workload Estimation by Multimodal Machine Learning on Psycho-physiological Signals |
|authors| Jong Hoon ParkLawrence ChenIan HigginsZhaobo ZhengShashank MehrotraKevin SalubreMohammadreza MousaeiSteven WillitsBlain LevedahlTimothy BukerEliot XingTeruhisa MisuSebastian SchererJean Oh
|links| http://arxiv.org/abs/2406.06448v1 |
|updated| 2024-06-10 16:39:21 UTC |
|summary| Vertical take-off and landing VTOL aircraft do not require a prolongedrunway thus allowing them to land almost anywhere. In recent years theirflexibility has made them popular in development research and operation. Whencompared to traditional fixed-wing aircraft and rotorcraft VTOLs bring uniquechallenges as they combine many maneuvers from both types of aircraft. Pilotworkload is a critical factor for safe and efficient operation of VTOLs. Inthis work we conduct a user study to collect multimodal data from 28 pilotswhile they perform a variety of VTOL flight tasks. We analyze and interpolatebehavioral patterns related to their performance and perceived workload.Finally we build machine learning models to estimate their workload from thecollected data. Our results are promising suggesting that quantitative andaccurate VTOL pilot workload monitoring is viable. Such assistive tools wouldhelp the research field understand VTOL operations and serve as a steppingstone for the industry to ensure VTOL safe operations and further remoteoperations. |


| Item |Content|
| --- |---|
|idx| 2406.06438v1 |
|title| Multimodal Contextualized Semantic Parsing from Speech |
|authors| Jordan VoasRaymond MooneyDavid Harwath
|links| http://arxiv.org/abs/2406.06438v1 |
|updated| 2024-06-10 16:31:34 UTC |
|summary| We introduce Semantic Parsing in Contextual Environments SPICE a taskdesigned to enhance artificial agents contextual awareness by integratingmultimodal inputs with prior contexts. SPICE goes beyond traditional semanticparsing by offering a structured interpretable framework for dynamicallyupdating an agents knowledge with new information mirroring the complexity ofhuman communication. We develop the VG-SPICE dataset crafted to challengeagents with visual scene graph construction from spoken conversationalexchanges highlighting speech and visual data integration. We also present theAudio-Vision Dialogue Scene Parser AViD-SP developed for use on VG-SPICE.These innovations aim to improve multimodal information processing andintegration. Both the VG-SPICE dataset and the AViD-SP model are publiclyavailable. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2406.06520v1 |
|title| Decentralized Personalized Federated Learning |
|authors| Salma KharratMarco CaniniSamuel Horvath
|links| http://arxiv.org/abs/2406.06520v1 |
|updated| 2024-06-10 17:58:48 UTC |
|summary| This work tackles the challenges of data heterogeneity and communicationlimitations in decentralized federated learning. We focus on creating acollaboration graph that guides each client in selecting suitable collaboratorsfor training personalized models that leverage their local data effectively.Our approach addresses these issues through a novel communication-efficientstrategy that enhances resource efficiency. Unlike traditional methods ourformulation identifies collaborators at a granular level by consideringcombinatorial relations of clients enhancing personalization while minimizingcommunication overhead. We achieve this through a bi-level optimizationframework that employs a constrained greedy algorithm resulting in aresource-efficient collaboration graph for personalized learning. Extensiveevaluation against various baselines across diverse datasets demonstrates thesuperiority of our method named DPFL. DPFL consistently outperforms otherapproaches showcasing its effectiveness in handling real-world dataheterogeneity minimizing communication overhead enhancing resourceefficiency and building personalized models in decentralized federatedlearning scenarios. |


| Item |Content|
| --- |---|
|idx| 2406.06402v1 |
|title| Early Acceptance Matching Game for User-Centric Clustering in Scalable Cell-free MIMO Networks |
|authors| Ala Eddine NoualiMohamed SanaJean-Paul Jamont
|links| http://arxiv.org/abs/2406.06402v1 |
|updated| 2024-06-10 15:56:14 UTC |
|summary| The canonical setup is the primary approach adopted in cell-freemultiple-input multiple-output MIMO networks in which all access pointsAPs jointly serve every user equipment UE. This approach is not scalable interms of computational complexity and fronthaul signaling becoming impracticalin large networks. This work adopts a user-centric approach a scalablealternative in which only a set of preferred APs jointly serve a UE. Formingthe optimal cluster of APs for each UE is a challenging task especially whenit needs to be dynamically adjusted to meet the quality of service QoSrequirements of the UE. This complexity is even exacerbated when consideringthe constrained fronthaul capacity of the UE and the AP. We solve this problemwith a novel many-to-many matching game. More specifically we devise an earlyacceptance matching algorithm which immediately admits or rejects UEs based ontheir requests and available radio resources. The proposed solutionsignificantly reduces the fronthaul signaling while satisfying the maximum ofUEs in terms of requested QoS compared to state-of-the-art approaches. |


| Item |Content|
| --- |---|
|idx| 2406.06041v1 |
|title| Risk Sensitivity in Markov Games and Multi-Agent Reinforcement Learning: A Systematic Review |
|authors| Hafez GhaemiShirin JamshidiMohammad MashreghiMajid Nili AhmadabadiHamed Kebriaei
|links| http://arxiv.org/abs/2406.06041v1 |
|updated| 2024-06-10 06:19:33 UTC |
|summary| Markov games MGs and multi-agent reinforcement learning MARL are studiedto model decision making in multi-agent systems. Traditionally the objectivein MG and MARL has been risk-neutral i.e. agents are assumed to optimize aperformance metric such as expected return without taking into accountsubjective or cognitive preferences of themselves or of other agents. Howeverignoring such preferences leads to inaccurate models of decision making in manyreal-world scenarios in finance operations research and behavioral economics.Therefore when these preferences are present it is necessary to incorporate asuitable measure of risk into the optimization objective of agents which opensthe door to risk-sensitive MG and MARL. In this paper we systemically reviewthe literature on risk sensitivity in MG and MARL that has been growing inrecent years alongside other areas of reinforcement learning and game theory.We define and mathematically describe different risk measures used in MG andMARL and individually for each measure discuss articles that incorporate it.Finally we identify recent trends in theoretical and applied works in thefield and discuss possible directions of future research. |


| Item |Content|
| --- |---|
|idx| 2406.05724v1 |
|title| Deception Analysis with Artificial Intelligence: An Interdisciplinary Perspective |
|authors| Stefan Sarkadi
|links| http://arxiv.org/abs/2406.05724v1 |
|updated| 2024-06-09 10:31:26 UTC |
|summary| Humans and machines interact more frequently than ever and our societies arebecoming increasingly hybrid. A consequence of this hybridisation is thedegradation of societal trust due to the prevalence of AI-enabled deception.Yet despite our understanding of the role of trust in AI in the recent yearswe still do not have a computational theory to be able to fully understand andexplain the role deception plays in this context. This is a problem becausewhile our ability to explain deception in hybrid societies is delayed thedesign of AI agents may keep advancing towards fully autonomous deceptivemachines which would pose new challenges to dealing with deception. In thispaper we build a timely and meaningful interdisciplinary perspective ondeceptive AI and reinforce a 20 year old socio-cognitive perspective on trustand deception by proposing the development of DAMAS -- a holistic Multi-AgentSystems MAS framework for the socio-cognitive modelling and analysis ofdeception. In a nutshell this paper covers the topic of modelling andexplaining deception using AI approaches from the perspectives of ComputerScience Philosophy Psychology Ethics and Intelligence Analysis. |


| Item |Content|
| --- |---|
|idx| 2406.05720v1 |
|title| VillagerAgent: A Graph-Based Multi-Agent Framework for Coordinating Complex Task Dependencies in Minecraft |
|authors| Yubo DongXukun ZhuZhengzhe PanLinchao ZhuYi Yang
|links| http://arxiv.org/abs/2406.05720v1 |
|updated| 2024-06-09 10:21:47 UTC |
|summary| In this paper we aim to evaluate multi-agent systems against complexdependencies including spatial causal and temporal constraints. First weconstruct a new benchmark named VillagerBench within the Minecraftenvironment.VillagerBench comprises diverse tasks crafted to test variousaspects of multi-agent collaboration from workload distribution to dynamicadaptation and synchronized task execution. Second we introduce a DirectedAcyclic Graph Multi-Agent Framework VillagerAgent to resolve complexinter-agent dependencies and enhance collaborative efficiency. This solutionincorporates a task decomposer that creates a directed acyclic graph DAG forstructured task management an agent controller for task distribution and astate manager for tracking environmental and agent data. Our empiricalevaluation on VillagerBench demonstrates that VillagerAgent outperforms theexisting AgentVerse model reducing hallucinations and improving taskdecomposition efficacy. The results underscore VillagerAgents potential inadvancing multi-agent collaboration offering a scalable and generalizablesolution in dynamic environments. The source code is open-source on GitHubhttps://github.com/cnsdqd-dyb/VillagerAgent. |


