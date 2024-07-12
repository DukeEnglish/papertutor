# cs.CL 

| Item |Content|
| --- |---|
|idx| 2407.07895v1 |
|title| LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models |
|authors| Feng LiRenrui ZhangHao ZhangYuanhan ZhangBo LiWei LiZejun MaChunyuan Li
|links| http://arxiv.org/abs/2407.07895v1 |
|updated| 2024-07-10 17:59:43 UTC |
|summary| Visual instruction tuning has made considerable strides in enhancing thecapabilities of Large Multimodal Models LMMs. However existing open LMMslargely focus on single-image tasks their applications to multi-imagescenarios remains less explored. Additionally prior LMM research separatelytackles different scenarios leaving it impossible to generalize crossscenarios with new emerging capabilities. To this end we introduceLLaVA-NeXT-Interleave which simultaneously tackles Multi-image Multi-framevideo Multi-view 3D and Multi-patch single-image scenarios in LMMs. Toenable these capabilities we regard the interleaved data format as a generaltemplate and compile the M4-Instruct dataset with 1177.6k samples spanning 4primary domains with 14 tasks and 41 datasets. We also curate theLLaVA-Interleave Bench to comprehensively evaluate the multi-image performanceof LMMs. Through extensive experiments LLaVA-NeXT-Interleave achieves leadingresults in multi-image video and 3D benchmarks while maintaining theperformance of single-image tasks. Besides our model also exhibits severalemerging capabilities e.g. transferring tasks across different settings andmodalities. Code is available at https://github.com/LLaVA-VL/LLaVA-NeXT |


| Item |Content|
| --- |---|
|idx| 2407.07890v1 |
|title| Training on the Test Task Confounds Evaluation and Emergence |
|authors| Ricardo Dominguez-OlmedoFlorian E. DornerMoritz Hardt
|links| http://arxiv.org/abs/2407.07890v1 |
|updated| 2024-07-10 17:57:58 UTC |
|summary| We study a fundamental problem in the evaluation of large language modelsthat we call training on the test task. Unlike wrongful practices like trainingon the test data leakage or data contamination training on the test task isnot a malpractice. Rather the term describes a growing set of techniques toinclude task-relevant data in the pretraining stage of a language model. Wedemonstrate that training on the test task confounds both relative modelevaluations and claims about emergent capabilities. We argue that the seemingsuperiority of one model family over another may be explained by a differentdegree of training on the test task. To this end we propose an effectivemethod to adjust for training on the test task by fine-tuning each model undercomparison on the same task-relevant data before evaluation. We then show thatinstances of emergent behavior largely vanish once we adjust for training onthe test task. This also applies to reported instances of emergent behaviorthat cannot be explained by the choice of evaluation metric. Our work promotesa new perspective on the evaluation of large language models with broadimplications for benchmarking and the study of emergent capabilities. |


| Item |Content|
| --- |---|
|idx| 2407.07880v1 |
|title| Towards Robust Alignment of Language Models: Distributionally Robustifying Direct Preference Optimization |
|authors| Junkang WuYuexiang XieZhengyi YangJiancan WuJiawei ChenJinyang GaoBolin DingXiang WangXiangnan He
|links| http://arxiv.org/abs/2407.07880v1 |
|updated| 2024-07-10 17:48:25 UTC |
|summary| This study addresses the challenge of noise in training datasets for DirectPreference Optimization DPO a method for aligning Large Language ModelsLLMs with human preferences. We categorize noise into pointwise noise whichincludes low-quality data points and pairwise noise which encompasseserroneous data pair associations that affect preference rankings. UtilizingDistributionally Robust Optimization DRO we enhance DPOs resilience tothese types of noise. Our theoretical insights reveal that DPO inherentlyembeds DRO principles conferring robustness to pointwise noise with theregularization coefficient beta playing a critical role in its noiseresistance. Extending this framework we introduce DistributionallyRobustifying DPO Dr. DPO which integrates pairwise robustness by optimizingagainst worst-case pairwise scenarios. The novel hyperparameter beta in Dr.DPO allows for fine-tuned control over data pair reliability providing astrategic balance between exploration and exploitation in noisy trainingenvironments. Empirical evaluations demonstrate that Dr. DPO substantiallyimproves the quality of generated text and response accuracy in preferencedatasets showcasing enhanced performance in both noisy and noise-freesettings. The code is available at https://github.com/junkangwu/Dr_DPO. |


| Item |Content|
| --- |---|
|idx| 2407.07875v1 |
|title| Generative Image as Action Models |
|authors| Mohit ShridharYat Long LoStephen James
|links| http://arxiv.org/abs/2407.07875v1 |
|updated| 2024-07-10 17:41:10 UTC |
|summary| Image-generation diffusion models have been fine-tuned to unlock newcapabilities such as image-editing and novel view synthesis. Can we similarlyunlock image-generation models for visuomotor control We present GENIMA abehavior-cloning agent that fine-tunes Stable Diffusion to draw joint-actionsas targets on RGB images. These images are fed into a controller that maps thevisual targets into a sequence of joint-positions. We study GENIMA on 25RLBench and 9 real-world manipulation tasks. We find that by lifting actionsinto image-space internet pre-trained diffusion models can generate policiesthat outperform state-of-the-art visuomotor approaches especially inrobustness to scene perturbations and generalizing to novel objects. Our methodis also competitive with 3D agents despite lacking priors such as depthkeypoints or motion-planners. |


| Item |Content|
| --- |---|
|idx| 2407.07858v1 |
|title| FACTS About Building Retrieval Augmented Generation-based Chatbots |
|authors| Rama AkkirajuAnbang XuDeepak BoraTan YuLu AnVishal SethAaditya ShuklaPritam GundechaHridhay MehtaAshwin JhaPrithvi RajAbhinav BalasubramanianMurali MaramGuru MuthusamyShivakesh Reddy AnnepallySidney KnowlesMin DuNick BurnettSean JaviyaAshok MarannanMamta KumariSurbhi JhaEthan DereszenskiAnupam ChakrabortySubhash RanjanAmina TerfaiAnoop SuryaTracey MercerVinodh Kumar ThanigachalamTamar BarSanjana KrishnanSamy KilaruJasmine JaksicNave AlgariciJacob LibermanJoey ConwaySonu NayyarJustin Boitano
|links| http://arxiv.org/abs/2407.07858v1 |
|updated| 2024-07-10 17:20:59 UTC |
|summary| Enterprise chatbots powered by generative AI are emerging as keyapplications to enhance employee productivity. Retrieval Augmented GenerationRAG Large Language Models LLMs and orchestration frameworks likeLangchain and Llamaindex are crucial for building these chatbots. Howevercreating effective enterprise chatbots is challenging and requires meticulousRAG pipeline engineering. This includes fine-tuning embeddings and LLMsextracting documents from vector databases rephrasing queries rerankingresults designing prompts honoring document access controls providingconcise responses including references safeguarding personal information andbuilding orchestration agents. We present a framework for building RAG-basedchatbots based on our experience with three NVIDIA chatbots: for IT/HRbenefits financial earnings and general content. Our contributions arethree-fold: introducing the FACTS framework Freshness Architectures CostTesting Security presenting fifteen RAG pipeline control points andproviding empirical results on accuracy-latency tradeoffs between large andsmall LLMs. To the best of our knowledge this is the first paper of its kindthat provides a holistic view of the factors as well as solutions for buildingsecure enterprise-grade chatbots. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2407.07890v1 |
|title| Training on the Test Task Confounds Evaluation and Emergence |
|authors| Ricardo Dominguez-OlmedoFlorian E. DornerMoritz Hardt
|links| http://arxiv.org/abs/2407.07890v1 |
|updated| 2024-07-10 17:57:58 UTC |
|summary| We study a fundamental problem in the evaluation of large language modelsthat we call training on the test task. Unlike wrongful practices like trainingon the test data leakage or data contamination training on the test task isnot a malpractice. Rather the term describes a growing set of techniques toinclude task-relevant data in the pretraining stage of a language model. Wedemonstrate that training on the test task confounds both relative modelevaluations and claims about emergent capabilities. We argue that the seemingsuperiority of one model family over another may be explained by a differentdegree of training on the test task. To this end we propose an effectivemethod to adjust for training on the test task by fine-tuning each model undercomparison on the same task-relevant data before evaluation. We then show thatinstances of emergent behavior largely vanish once we adjust for training onthe test task. This also applies to reported instances of emergent behaviorthat cannot be explained by the choice of evaluation metric. Our work promotesa new perspective on the evaluation of large language models with broadimplications for benchmarking and the study of emergent capabilities. |


| Item |Content|
| --- |---|
|idx| 2407.07884v1 |
|title| Vegetable Peeling: A Case Study in Constrained Dexterous Manipulation |
|authors| Tao ChenEric CousineauNaveen KuppuswamyPulkit Agrawal
|links| http://arxiv.org/abs/2407.07884v1 |
|updated| 2024-07-10 17:51:33 UTC |
|summary| Recent studies have made significant progress in addressing dexterousmanipulation problems particularly in in-hand object reorientation. Howeverthere are few existing works that explore the potential utilization ofdeveloped dexterous manipulation controllers for downstream tasks. In thisstudy we focus on constrained dexterous manipulation for food peeling. Foodpeeling presents various constraints on the reorientation controller such asthe requirement for the hand to securely hold the object after reorientationfor peeling. We propose a simple system for learning a reorientation controllerthat facilitates the subsequent peeling task. Videos are available at:https://taochenshh.github.io/projects/veg-peeling. |


| Item |Content|
| --- |---|
|idx| 2407.07880v1 |
|title| Towards Robust Alignment of Language Models: Distributionally Robustifying Direct Preference Optimization |
|authors| Junkang WuYuexiang XieZhengyi YangJiancan WuJiawei ChenJinyang GaoBolin DingXiang WangXiangnan He
|links| http://arxiv.org/abs/2407.07880v1 |
|updated| 2024-07-10 17:48:25 UTC |
|summary| This study addresses the challenge of noise in training datasets for DirectPreference Optimization DPO a method for aligning Large Language ModelsLLMs with human preferences. We categorize noise into pointwise noise whichincludes low-quality data points and pairwise noise which encompasseserroneous data pair associations that affect preference rankings. UtilizingDistributionally Robust Optimization DRO we enhance DPOs resilience tothese types of noise. Our theoretical insights reveal that DPO inherentlyembeds DRO principles conferring robustness to pointwise noise with theregularization coefficient beta playing a critical role in its noiseresistance. Extending this framework we introduce DistributionallyRobustifying DPO Dr. DPO which integrates pairwise robustness by optimizingagainst worst-case pairwise scenarios. The novel hyperparameter beta in Dr.DPO allows for fine-tuned control over data pair reliability providing astrategic balance between exploration and exploitation in noisy trainingenvironments. Empirical evaluations demonstrate that Dr. DPO substantiallyimproves the quality of generated text and response accuracy in preferencedatasets showcasing enhanced performance in both noisy and noise-freesettings. The code is available at https://github.com/junkangwu/Dr_DPO. |


| Item |Content|
| --- |---|
|idx| 2407.07875v1 |
|title| Generative Image as Action Models |
|authors| Mohit ShridharYat Long LoStephen James
|links| http://arxiv.org/abs/2407.07875v1 |
|updated| 2024-07-10 17:41:10 UTC |
|summary| Image-generation diffusion models have been fine-tuned to unlock newcapabilities such as image-editing and novel view synthesis. Can we similarlyunlock image-generation models for visuomotor control We present GENIMA abehavior-cloning agent that fine-tunes Stable Diffusion to draw joint-actionsas targets on RGB images. These images are fed into a controller that maps thevisual targets into a sequence of joint-positions. We study GENIMA on 25RLBench and 9 real-world manipulation tasks. We find that by lifting actionsinto image-space internet pre-trained diffusion models can generate policiesthat outperform state-of-the-art visuomotor approaches especially inrobustness to scene perturbations and generalizing to novel objects. Our methodis also competitive with 3D agents despite lacking priors such as depthkeypoints or motion-planners. |


| Item |Content|
| --- |---|
|idx| 2407.07874v2 |
|title| Toto: Time Series Optimized Transformer for Observability |
|authors| Ben CohenEmaad KhwajaKan WangCharles MassonElise RaméYoussef DoubliOthmane Abou-Amal
|links| http://arxiv.org/abs/2407.07874v2 |
|updated| 2024-07-11 16:18:40 UTC |
|summary| This technical report describes the Time Series Optimized Transformer forObservability Toto a new state of the art foundation model for time seriesforecasting developed by Datadog. In addition to advancing the state of the arton generalized time series benchmarks in domains such as electricity andweather this model is the first general-purpose time series forecastingfoundation model to be specifically tuned for observability metrics.  Toto was trained on a dataset of one trillion time series data points thelargest among all currently published time series foundation models. Alongsidepublicly available time series datasets 75 of the data used to train Totoconsists of fully anonymous numerical metric data points from the Datadogplatform.  In our experiments Toto outperforms existing time series foundation modelson observability data. It does this while also excelling at general-purposeforecasting tasks achieving state-of-the-art zero-shot performance on multipleopen benchmark datasets. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2407.07896v1 |
|title| Pentagonal Photonic Crystal Mirrors: Scalable Lightsails with Enhanced Acceleration via Neural Topology Optimization |
|authors| L. NorderS. YinM. J. de JongF. StalloneH. AydogmusP. M. SbernaM. A. BessaR. A. Norte
|links| http://arxiv.org/abs/2407.07896v1 |
|updated| 2024-07-10 17:59:55 UTC |
|summary| The Starshot Breakthrough Initiative aims to send one-gram microchip probesto Alpha Centauri within 20 years using gram-scale lightsails propelled bylaser-based radiation pressure reaching velocities nearing a fifth of lightspeed. This mission requires lightsail materials that challenge thefundamentals of nanotechnology requiring innovations in optics materialscience and structural engineering. Unlike the microchip payload which must beminimized in every dimension such lightsails need meter-scale dimensions withnanoscale thickness and billions of nanoscale holes to enhance reflectivity andreduce mass. Our study employs neural topology optimization revealing a novelpentagonal lattice-based photonic crystal PhC reflector. The optimizeddesigns shorten acceleration times therefore lowering launch costssignificantly. Crucially these designs also enable lightsail materialfabrication with orders-of-magnitude reduction in costs. We have fabricated a60 x 60 mm2 200nm thick single-layer reflector perforated with over abillion nanoscale features the highest aspect-ratio nanophotonic element todate. We achieve this with nearly 9000 times cost reduction per m2.Starshot lightsails will have several stringent requirements but willultimately be driven by costs to build at scale. Here we highlight challengesand possible solutions in developing lightsail materials - showcasing thepotential of scaling nanophotonics for cost-effective next-generation spaceexploration. |


| Item |Content|
| --- |---|
|idx| 2407.07895v1 |
|title| LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models |
|authors| Feng LiRenrui ZhangHao ZhangYuanhan ZhangBo LiWei LiZejun MaChunyuan Li
|links| http://arxiv.org/abs/2407.07895v1 |
|updated| 2024-07-10 17:59:43 UTC |
|summary| Visual instruction tuning has made considerable strides in enhancing thecapabilities of Large Multimodal Models LMMs. However existing open LMMslargely focus on single-image tasks their applications to multi-imagescenarios remains less explored. Additionally prior LMM research separatelytackles different scenarios leaving it impossible to generalize crossscenarios with new emerging capabilities. To this end we introduceLLaVA-NeXT-Interleave which simultaneously tackles Multi-image Multi-framevideo Multi-view 3D and Multi-patch single-image scenarios in LMMs. Toenable these capabilities we regard the interleaved data format as a generaltemplate and compile the M4-Instruct dataset with 1177.6k samples spanning 4primary domains with 14 tasks and 41 datasets. We also curate theLLaVA-Interleave Bench to comprehensively evaluate the multi-image performanceof LMMs. Through extensive experiments LLaVA-NeXT-Interleave achieves leadingresults in multi-image video and 3D benchmarks while maintaining theperformance of single-image tasks. Besides our model also exhibits severalemerging capabilities e.g. transferring tasks across different settings andmodalities. Code is available at https://github.com/LLaVA-VL/LLaVA-NeXT |


| Item |Content|
| --- |---|
|idx| 2407.07890v1 |
|title| Training on the Test Task Confounds Evaluation and Emergence |
|authors| Ricardo Dominguez-OlmedoFlorian E. DornerMoritz Hardt
|links| http://arxiv.org/abs/2407.07890v1 |
|updated| 2024-07-10 17:57:58 UTC |
|summary| We study a fundamental problem in the evaluation of large language modelsthat we call training on the test task. Unlike wrongful practices like trainingon the test data leakage or data contamination training on the test task isnot a malpractice. Rather the term describes a growing set of techniques toinclude task-relevant data in the pretraining stage of a language model. Wedemonstrate that training on the test task confounds both relative modelevaluations and claims about emergent capabilities. We argue that the seemingsuperiority of one model family over another may be explained by a differentdegree of training on the test task. To this end we propose an effectivemethod to adjust for training on the test task by fine-tuning each model undercomparison on the same task-relevant data before evaluation. We then show thatinstances of emergent behavior largely vanish once we adjust for training onthe test task. This also applies to reported instances of emergent behaviorthat cannot be explained by the choice of evaluation metric. Our work promotesa new perspective on the evaluation of large language models with broadimplications for benchmarking and the study of emergent capabilities. |


| Item |Content|
| --- |---|
|idx| 2407.07889v1 |
|title| AdaptiGraph: Material-Adaptive Graph-Based Neural Dynamics for Robotic Manipulation |
|authors| Kaifeng ZhangBaoyu LiKris HauserYunzhu Li
|links| http://arxiv.org/abs/2407.07889v1 |
|updated| 2024-07-10 17:57:04 UTC |
|summary| Predictive models are a crucial component of many robotic systems. Yetconstructing accurate predictive models for a variety of deformable objectsespecially those with unknown physical properties remains a significantchallenge. This paper introduces AdaptiGraph a learning-based dynamicsmodeling approach that enables robots to predict adapt to and control a widearray of challenging deformable materials with unknown physical properties.AdaptiGraph leverages the highly flexible graph-based neural dynamics GBNDframework which represents material bits as particles and employs a graphneural network GNN to predict particle motion. Its key innovation is aunified physical property-conditioned GBND model capable of predicting themotions of diverse materials with varying physical properties withoutretraining. Upon encountering new materials during online deploymentAdaptiGraph utilizes a physical property optimization process for a few-shotadaptation of the model enhancing its fit to the observed interaction data.The adapted models can precisely simulate the dynamics and predict the motionof various deformable materials such as ropes granular media rigid boxesand cloth while adapting to different physical properties includingstiffness granular size and center of pressure. On prediction andmanipulation tasks involving a diverse set of real-world deformable objectsour method exhibits superior prediction accuracy and task proficiency overnon-material-conditioned and non-adaptive models. The project page is availableat https://robopil.github.io/adaptigraph/ . |


| Item |Content|
| --- |---|
|idx| 2407.07885v1 |
|title| Learning In-Hand Translation Using Tactile Skin With Shear and Normal Force Sensing |
|authors| Jessica YinHaozhi QiJitendra MalikJames PikulMark YimTess Hellebrekers
|links| http://arxiv.org/abs/2407.07885v1 |
|updated| 2024-07-10 17:52:30 UTC |
|summary| Recent progress in reinforcement learning RL and tactile sensing hassignificantly advanced dexterous manipulation. However these methods oftenutilize simplified tactile signals due to the gap between tactile simulationand the real world. We introduce a sensor model for tactile skin that enableszero-shot sim-to-real transfer of ternary shear and binary normal forces. Usingthis model we develop an RL policy that leverages sliding contact fordexterous in-hand translation. We conduct extensive real-world experiments toassess how tactile sensing facilitates policy adaptation to various unseenobject properties and robot hand orientations. We demonstrate that our 3-axistactile policies consistently outperform baselines that use only shear forcesonly normal forces or only proprioception. Website:https://jessicayin.github.io/tactile-skin-rl/ |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2407.07895v1 |
|title| LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models |
|authors| Feng LiRenrui ZhangHao ZhangYuanhan ZhangBo LiWei LiZejun MaChunyuan Li
|links| http://arxiv.org/abs/2407.07895v1 |
|updated| 2024-07-10 17:59:43 UTC |
|summary| Visual instruction tuning has made considerable strides in enhancing thecapabilities of Large Multimodal Models LMMs. However existing open LMMslargely focus on single-image tasks their applications to multi-imagescenarios remains less explored. Additionally prior LMM research separatelytackles different scenarios leaving it impossible to generalize crossscenarios with new emerging capabilities. To this end we introduceLLaVA-NeXT-Interleave which simultaneously tackles Multi-image Multi-framevideo Multi-view 3D and Multi-patch single-image scenarios in LMMs. Toenable these capabilities we regard the interleaved data format as a generaltemplate and compile the M4-Instruct dataset with 1177.6k samples spanning 4primary domains with 14 tasks and 41 datasets. We also curate theLLaVA-Interleave Bench to comprehensively evaluate the multi-image performanceof LMMs. Through extensive experiments LLaVA-NeXT-Interleave achieves leadingresults in multi-image video and 3D benchmarks while maintaining theperformance of single-image tasks. Besides our model also exhibits severalemerging capabilities e.g. transferring tasks across different settings andmodalities. Code is available at https://github.com/LLaVA-VL/LLaVA-NeXT |


| Item |Content|
| --- |---|
|idx| 2407.07889v1 |
|title| AdaptiGraph: Material-Adaptive Graph-Based Neural Dynamics for Robotic Manipulation |
|authors| Kaifeng ZhangBaoyu LiKris HauserYunzhu Li
|links| http://arxiv.org/abs/2407.07889v1 |
|updated| 2024-07-10 17:57:04 UTC |
|summary| Predictive models are a crucial component of many robotic systems. Yetconstructing accurate predictive models for a variety of deformable objectsespecially those with unknown physical properties remains a significantchallenge. This paper introduces AdaptiGraph a learning-based dynamicsmodeling approach that enables robots to predict adapt to and control a widearray of challenging deformable materials with unknown physical properties.AdaptiGraph leverages the highly flexible graph-based neural dynamics GBNDframework which represents material bits as particles and employs a graphneural network GNN to predict particle motion. Its key innovation is aunified physical property-conditioned GBND model capable of predicting themotions of diverse materials with varying physical properties withoutretraining. Upon encountering new materials during online deploymentAdaptiGraph utilizes a physical property optimization process for a few-shotadaptation of the model enhancing its fit to the observed interaction data.The adapted models can precisely simulate the dynamics and predict the motionof various deformable materials such as ropes granular media rigid boxesand cloth while adapting to different physical properties includingstiffness granular size and center of pressure. On prediction andmanipulation tasks involving a diverse set of real-world deformable objectsour method exhibits superior prediction accuracy and task proficiency overnon-material-conditioned and non-adaptive models. The project page is availableat https://robopil.github.io/adaptigraph/ . |


| Item |Content|
| --- |---|
|idx| 2407.07875v1 |
|title| Generative Image as Action Models |
|authors| Mohit ShridharYat Long LoStephen James
|links| http://arxiv.org/abs/2407.07875v1 |
|updated| 2024-07-10 17:41:10 UTC |
|summary| Image-generation diffusion models have been fine-tuned to unlock newcapabilities such as image-editing and novel view synthesis. Can we similarlyunlock image-generation models for visuomotor control We present GENIMA abehavior-cloning agent that fine-tunes Stable Diffusion to draw joint-actionsas targets on RGB images. These images are fed into a controller that maps thevisual targets into a sequence of joint-positions. We study GENIMA on 25RLBench and 9 real-world manipulation tasks. We find that by lifting actionsinto image-space internet pre-trained diffusion models can generate policiesthat outperform state-of-the-art visuomotor approaches especially inrobustness to scene perturbations and generalizing to novel objects. Our methodis also competitive with 3D agents despite lacking priors such as depthkeypoints or motion-planners. |


| Item |Content|
| --- |---|
|idx| 2407.07868v1 |
|title| Green Screen Augmentation Enables Scene Generalisation in Robotic Manipulation |
|authors| Eugene TeohSumit PatidarXiao MaStephen James
|links| http://arxiv.org/abs/2407.07868v1 |
|updated| 2024-07-10 17:32:05 UTC |
|summary| Generalising vision-based manipulation policies to novel environments remainsa challenging area with limited exploration. Current practices involvecollecting data in one location training imitation learning or reinforcementlearning policies with this data and deploying the policy in the samelocation. However this approach lacks scalability as it necessitates datacollection in multiple locations for each task. This paper proposes a novelapproach where data is collected in a location predominantly featuring greenscreens. We introduce Green-screen Augmentation GreenAug employing a chromakey algorithm to overlay background textures onto a green screen. Throughextensive real-world empirical studies with over 850 training demonstrationsand 8.2k evaluation episodes we demonstrate that GreenAug surpasses noaugmentation standard computer vision augmentation and prior generativeaugmentation methods in performance. While no algorithmic novelties areclaimed our paper advocates for a fundamental shift in data collectionpractices. We propose that real-world demonstrations in future research shouldutilise green screens followed by the application of GreenAug. We believeGreenAug unlocks policy generalisation to visually distinct novel locationsaddressing the current scene generalisation limitations in robot learning. |


| Item |Content|
| --- |---|
|idx| 2407.07860v1 |
|title| Controlling Space and Time with Diffusion Models |
|authors| Daniel WatsonSaurabh SaxenaLala LiAndrea TagliasacchiDavid J. Fleet
|links| http://arxiv.org/abs/2407.07860v1 |
|updated| 2024-07-10 17:23:33 UTC |
|summary| We present 4DiM a cascaded diffusion model for 4D novel view synthesisNVS conditioned on one or more images of a general scene and a set ofcamera poses and timestamps. To overcome challenges due to limited availabilityof 4D training data we advocate joint training on 3D with camera pose 4Dposetime and video time but no pose data and propose a new architecturethat enables the same. We further advocate the calibration of SfM posed datausing monocular metric depth estimators for metric scale camera control. Formodel evaluation we introduce new metrics to enrich and overcome shortcomingsof current evaluation schemes demonstrating state-of-the-art results in bothfidelity and pose control compared to existing diffusion models for 3D NVSwhile at the same time adding the ability to handle temporal dynamics. 4DiM isalso used for improved panorama stitching pose-conditioned video to videotranslation and several other tasks. For an overview seehttps://4d-diffusion.github.io |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2407.07873v1 |
|title| Dynamical Measure Transport and Neural PDE Solvers for Sampling |
|authors| Jingtong SunJulius BernerLorenz RichterMarius ZeinhoferJohannes MüllerKamyar AzizzadenesheliAnima Anandkumar
|links| http://arxiv.org/abs/2407.07873v1 |
|updated| 2024-07-10 17:39:50 UTC |
|summary| The task of sampling from a probability density can be approached astransporting a tractable density function to the target known as dynamicalmeasure transport. In this work we tackle it through a principled unifiedframework using deterministic or stochastic evolutions described by partialdifferential equations PDEs. This framework incorporates priortrajectory-based sampling methods such as diffusion models or Schrodingerbridges without relying on the concept of time-reversals. Moreover it allowsus to propose novel numerical methods for solving the transport task and thussampling from complicated targets without the need for the normalizationconstant or data samples. We employ physics-informed neural networks PINNs toapproximate the respective PDE solutions implying both conceptional andcomputational advantages. In particular PINNs allow for simulation- anddiscretization-free optimization and can be trained very efficiently leadingto significantly better mode coverage in the sampling task compared toalternative methods. Moreover they can readily be fine-tuned with Gauss-Newtonmethods to achieve high accuracy in sampling. |


| Item |Content|
| --- |---|
|idx| 2407.07829v1 |
|title| Disentangled Representation Learning through Geometry Preservation with the Gromov-Monge Gap |
|authors| Théo UsciddaLuca EyringKarsten RothFabian TheisZeynep AkataMarco Cuturi
|links| http://arxiv.org/abs/2407.07829v1 |
|updated| 2024-07-10 16:51:32 UTC |
|summary| Learning disentangled representations in an unsupervised manner is afundamental challenge in machine learning. Solving it may unlock otherproblems such as generalization interpretability or fairness. Whileremarkably difficult to solve in general recent works have shown thatdisentanglement is provably achievable under additional assumptions that canleverage geometrical constraints such as local isometry. To use theseinsights we propose a novel perspective on disentangled representationlearning built on quadratic optimal transport. Specifically we formulate theproblem in the Gromov-Monge setting which seeks isometric mappings betweendistributions supported on different spaces. We propose the Gromov-Monge-GapGMG a regularizer that quantifies the geometry-preservation of an arbitrarypush-forward map between two distributions supported on different spaces. Wedemonstrate the effectiveness of GMG regularization for disentanglement on fourstandard benchmarks. Moreover we show that geometry preservation can evenencourage unsupervised disentanglement without the standard reconstructionobjective - making the underlying model decoder-free and promising a morepractically viable and scalable perspective on unsupervised disentanglement. |


| Item |Content|
| --- |---|
|idx| 2407.07821v1 |
|title| When to Accept Automated Predictions and When to Defer to Human Judgment? |
|authors| Daniel SikarArtur GarcezTillman WeydeRobin BloomfieldKaleem Peeroo
|links| http://arxiv.org/abs/2407.07821v1 |
|updated| 2024-07-10 16:45:52 UTC |
|summary| Ensuring the reliability and safety of automated decision-making is crucial.It is well-known that data distribution shifts in machine learning can produceunreliable outcomes. This paper proposes a new approach for measuring thereliability of predictions under distribution shifts. We analyze how theoutputs of a trained neural network change using clustering to measuredistances between outputs and class centroids. We propose this distance as ametric to evaluate the confidence of predictions under distribution shifts. Weassign each prediction to a cluster with centroid representing the mean softmaxoutput for all correct predictions of a given class. We then define a safetythreshold for a class as the smallest distance from an incorrect prediction tothe given class centroid. We evaluate the approach on the MNIST and CIFAR-10datasets using a Convolutional Neural Network and a Vision Transformerrespectively. The results show that our approach is consistent across thesedata sets and network models and indicate that the proposed metric can offeran efficient way of determining when automated predictions are acceptable andwhen they should be deferred to human operators given a distribution shift. |


| Item |Content|
| --- |---|
|idx| 2407.07781v1 |
|title| Sequential Kalman Monte Carlo for gradient-free inference in Bayesian inverse problems |
|authors| Richard D. P. GrumittMinas KaramanisUroš Seljak
|links| http://arxiv.org/abs/2407.07781v1 |
|updated| 2024-07-10 15:56:30 UTC |
|summary| Ensemble Kalman Inversion EKI has been proposed as an efficient method forsolving inverse problems with expensive forward models. However the method isbased on the assumption that we proceed through a sequence of Gaussian measuresin moving from the prior to the posterior and that the forward model islinear. In this work we introduce Sequential Kalman Monte Carlo SKMCsamplers where we exploit EKI and Flow Annealed Kalman Inversion FAKI withina Sequential Monte Carlo SMC sampling scheme to perform efficientgradient-free inference in Bayesian inverse problems. FAKI employs normalizingflows NF to relax the Gaussian ansatz of the target measures in EKI. NFs areable to learn invertible maps between a Gaussian latent space and the originaldata space allowing us to perform EKI updates in the Gaussianized NF latentspace. However FAKI alone is not able to correct for the model linearityassumptions in EKI. Errors in the particle distribution as we move through thesequence of target measures can therefore compound to give incorrect posteriormoment estimates. In this work we consider the use of EKI and FAKI toinitialize the particle distribution for each target in an adaptive SMCannealing scheme before performing t-preconditioned Crank-Nicolson tpCNupdates to distribute particles according to the target. We demonstrate theperformance of these SKMC samplers on three challenging numerical benchmarksshowing significant improvements in the rate of convergence compared tostandard SMC with importance weighted resampling at each temperature level.Code implementing the SKMC samplers is available athttps://github.com/RichardGrumitt/KalmanMC. |


| Item |Content|
| --- |---|
|idx| 2407.07765v1 |
|title| Ramsey Theorems for Trees and a General 'Private Learning Implies Online Learning' Theorem |
|authors| Simone FioravantiSteve HannekeShay MoranHilla ScheflerIska Tsubari
|links| http://arxiv.org/abs/2407.07765v1 |
|updated| 2024-07-10 15:43:30 UTC |
|summary| This work continues to investigate the link between differentially privateDP and online learning. Alon Livni Malliaris and Moran 2019 showed thatfor binary concept classes DP learnability of a given class implies that ithas a finite Littlestone dimension equivalently that it is online learnable.Their proof relies on a model-theoretic result by Hodges 1997 whichdemonstrates that any binary concept class with a large Littlestone dimensioncontains a large subclass of thresholds. In a follow-up work Jung Kim andTewari 2020 extended this proof to multiclass PAC learning with a boundednumber of labels. Unfortunately Hodgess result does not apply in othernatural settings such as multiclass PAC learning with an unbounded label spaceand PAC learning of partial concept classes.  This naturally raises the question of whether DP learnability continues toimply online learnability in more general scenarios: indeed Alon HannekeHolzman and Moran 2021 explicitly leave it as an open question in thecontext of partial concept classes and the same question is open in thegeneral multiclass setting. In this work we give a positive answer to thesequestions showing that for general classification tasks DP learnabilityimplies online learnability. Our proof reasons directly about Littlestonetrees without relying on thresholds. We achieve this by establishing severalRamsey-type theorems for trees which might be of independent interest. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2407.07786v1 |
|title| The Human Factor in AI Red Teaming: Perspectives from Social and Collaborative Computing |
|authors| Alice Qian ZhangRyland ShawJacy Reese AnthisAshlee MiltonEmily TsengJina SuhLama AhmadRam Shankar Siva KumarJulian PosadaBenjamin ShestakofskySarah T. RobertsMary L. Gray
|links| http://arxiv.org/abs/2407.07786v1 |
|updated| 2024-07-10 16:02:13 UTC |
|summary| Rapid progress in general-purpose AI has sparked significant interest in redteaming a practice of adversarial testing originating in military andcybersecurity applications. AI red teaming raises many questions about thehuman factor such as how red teamers are selected biases and blindspots inhow tests are conducted and harmful contents psychological effects on redteamers. A growing body of HCI and CSCW literature examines relatedpractices-including data labeling content moderation and algorithmicauditing. However few if any have investigated red teaming itself. Thisworkshop seeks to consider the conceptual and empirical challenges associatedwith this practice often rendered opaque by non-disclosure agreements. Futurestudies may explore topics ranging from fairness to mental health and otherareas of potential harm. We aim to facilitate a community of researchers andpractitioners who can begin to meet these challenges with creativityinnovation and thoughtful reflection. |


| Item |Content|
| --- |---|
|idx| 2407.07698v1 |
|title| The V-Lab VR Educational Application Framework |
|authors| Vasilis ZafeiropoulosGeorge AnastassakisTheophanis OrphanoudakisDimitris KallesAnastasios FanariotisVassilis Fotopoulos
|links| http://dx.doi.org/10.1145/3565066.3608246 |
|updated| 2024-07-10 14:31:36 UTC |
|summary| This paper presents the V-Lab a VR application development framework foreducational scenarios mainly involving scientific processes executed inlaboratory environments such as chemistry and biology laboratories. This workis an extension of the Onlabs simulator which has been developed by theHellenic Open University as a distance teaching enabler for similar subjectshelping to alleviate the need for access to the physical laboratoryinfrastructure thus shortening training periods of students in the laboratoryand making their training during the periods of physical presence moreproductive and secure. The extensions of the Onlabs to deliver an enhanced andmodular framework that can be extended to multiple educational scenarios is thework performed within the context of the European project XR2Learn Leveragingthe European XR industry technologies to empower immersive learning andtraining. |


| Item |Content|
| --- |---|
|idx| 2407.07683v1 |
|title| The Language of Weather: Social Media Reactions to Weather Accounting for Climatic and Linguistic Baselines |
|authors| James C. YoungRudy ArthurHywel T. P. Williams
|links| http://arxiv.org/abs/2407.07683v1 |
|updated| 2024-07-10 14:08:24 UTC |
|summary| This study explores how different weather conditions influence publicsentiment on social media focusing on Twitter data from the UK. By consideringclimate and linguistic baselines we improve the accuracy of weather-relatedsentiment analysis. Our findings show that emotional responses to weather arecomplex influenced by combinations of weather variables and regional languagedifferences. The results highlight the importance of context-sensitive methodsfor better understanding public mood in response to weather which can enhanceimpact-based forecasting and risk communication in the context of climatechange. |


| Item |Content|
| --- |---|
|idx| 2407.07672v1 |
|title| StoryDiffusion: How to Support UX Storyboarding With Generative-AI |
|authors| Zhaohui LiangXiaoyu ZhangKevin MaZhao LiuXipei RenKosa Goucher-LambertCan Liu
|links| http://arxiv.org/abs/2407.07672v1 |
|updated| 2024-07-10 13:59:37 UTC |
|summary| Storyboarding is an established method for designing user experiences.Generative AI can support this process by helping designers quickly createvisual narratives. However existing tools only focus on accurate text-to-imagegeneration. Currently it is not clear how to effectively support the entirecreative process of storyboarding and how to develop AI-powered tools tosupport designers individual workflows. In this work we iteratively developedand implemented StoryDiffusion a system that integrates text-to-text andtext-to-image models to support the generation of narratives and images in asingle pipeline. With a user study we observed 12 UX designers using thesystem for both concept ideation and illustration tasks. Our findingsidentified AI-directed vs. user-directed creative strategies in both tasks andrevealed the importance of supporting the interchange between narrativeiteration and image generation. We also found effects of the design tasks ontheir strategies and preferences providing insights for future development. |


| Item |Content|
| --- |---|
|idx| 2407.07653v1 |
|title| AffectGPT: Dataset and Framework for Explainable Multimodal Emotion Recognition |
|authors| Zheng LianHaiyang SunLicai SunJiangyan YiBin LiuJianhua Tao
|links| http://arxiv.org/abs/2407.07653v1 |
|updated| 2024-07-10 13:34:14 UTC |
|summary| Explainable Multimodal Emotion Recognition EMER is an emerging task thataims to achieve reliable and accurate emotion recognition. However due to thehigh annotation cost the existing dataset denoted as EMER-Fine is smallmaking it difficult to perform supervised training. To reduce the annotationcost and expand the dataset size this paper reviews the previous datasetconstruction process. Then we simplify the annotation pipeline avoid manualchecks and replace the closed-source models with open-source models. Finallywe build textbfEMER-Coarse a coarsely-labeled dataset containinglarge-scale samples. Besides the dataset we propose a two-stage trainingframework textbfAffectGPT. The first stage exploits EMER-Coarse to learn acoarse mapping between multimodal inputs and emotion-related descriptions thesecond stage uses EMER-Fine to better align with manually-checked results.Experimental results demonstrate the effectiveness of our proposed method onthe challenging EMER task. To facilitate further research we will make thecode and dataset available at: https://github.com/zeroQiaoba/AffectGPT. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2407.06886v1 |
|title| Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI |
|authors| Yang LiuWeixing ChenYongjie BaiJingzhou LuoXinshuai SongKaixuan JiangZhida LiGanlong ZhaoJunyi LinGuanbin LiWen GaoLiang Lin
|links| http://arxiv.org/abs/2407.06886v1 |
|updated| 2024-07-09 14:14:47 UTC |
|summary| Embodied Artificial Intelligence Embodied AI is crucial for achievingArtificial General Intelligence AGI and serves as a foundation for variousapplications that bridge cyberspace and the physical world. Recently theemergence of Multi-modal Large Models MLMs and World Models WMs haveattracted significant attention due to their remarkable perceptioninteraction and reasoning capabilities making them a promising architecturefor the brain of embodied agents. However there is no comprehensive survey forEmbodied AI in the era of MLMs. In this survey we give a comprehensiveexploration of the latest advancements in Embodied AI. Our analysis firstlynavigates through the forefront of representative works of embodied robots andsimulators to fully understand the research focuses and their limitations.Then we analyze four main research targets: 1 embodied perception 2embodied interaction 3 embodied agent and 4 sim-to-real adaptationcovering the state-of-the-art methods essential paradigms and comprehensivedatasets. Additionally we explore the complexities of MLMs in virtual and realembodied agents highlighting their significance in facilitating interactionsin dynamic digital and physical environments. Finally we summarize thechallenges and limitations of embodied AI and discuss their potential futuredirections. We hope this survey will serve as a foundational reference for theresearch community and inspire continued innovation. The associated project canbe found at https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List. |


| Item |Content|
| --- |---|
|idx| 2407.06813v1 |
|title| Richelieu: Self-Evolving LLM-Based Agents for AI Diplomacy |
|authors| Zhenyu GuanXiangyu KongFangwei ZhongYizhou Wang
|links| http://arxiv.org/abs/2407.06813v1 |
|updated| 2024-07-09 12:37:54 UTC |
|summary| Diplomacy is one of the most sophisticated activities in human society. Thecomplex interactions among multiple parties/ agents involve various abilitieslike social reasoning negotiation arts and long-term strategy planning.Previous AI agents surely have proved their capability of handling multi-stepgames and larger action spaces on tasks involving multiple agents. Howeverdiplomacy involves a staggering magnitude of decision spaces especiallyconsidering the negotiation stage required. Recently LLM agents have showntheir potential for extending the boundary of previous agents on a couple ofapplications however it is still not enough to handle a very long planningperiod in a complex multi-agent environment. Empowered with cutting-edge LLMtechnology we make the first stab to explore AIs upper bound towards ahuman-like agent for such a highly comprehensive multi-agent mission bycombining three core and essential capabilities for stronger LLM-based societalagents: 1 strategic planner with memory and reflection 2 goal-orientednegotiate with social reasoning 3 augmenting memory by self-play games toself-evolving without any human in the loop. |


| Item |Content|
| --- |---|
|idx| 2407.06541v1 |
|title| Fast Distributed Optimization over Directed Graphs under Malicious Attacks using Trust |
|authors| Arif Kerem DayıOrhan Eren AkgünStephanie GilMichal YeminiAngelia Nedić
|links| http://arxiv.org/abs/2407.06541v1 |
|updated| 2024-07-09 04:22:35 UTC |
|summary| In this work we introduce the Resilient Projected Push-Pull RP3 algorithmdesigned for distributed optimization in multi-agent cyber-physical systemswith directed communication graphs and the presence of malicious agents. Ouralgorithm leverages stochastic inter-agent trust values and gradient trackingto achieve geometric convergence rates in expectation even in adversarialenvironments. We introduce growing constraint sets to limit the impact of themalicious agents without compromising the geometric convergence rate of thealgorithm. We prove that RP3 converges to the nominal optimal solution almostsurely and in the r-th mean for any rgeq 1 provided the step sizes aresufficiently small and the constraint sets are appropriately chosen. Wevalidate our approach with numerical studies on average consensus andmulti-robot target tracking problems demonstrating that RP3 effectivelymitigates the impact of malicious agents and achieves the desired geometricconvergence. |


| Item |Content|
| --- |---|
|idx| 2407.06454v2 |
|title| Analysis of Robotic System Models Through Property Inheritance from Petri Net Meta-models |
|authors| Maksym FigatCezary Zieliński
|links| http://arxiv.org/abs/2407.06454v2 |
|updated| 2024-07-11 12:04:34 UTC |
|summary| This article investigates the analysis of robotic system models using theRobotic System Hierarchic Petri Net RSHPN meta-model proposing streamlinedmethods by focusing on significant system fragments and inheriting propertiesfrom the meta-model. Our research demonstrates that it is feasible to: 1effectively analyze complex robotic systems expressed using RSHPN and 2enable models to inherit properties from the meta-model. This approachsignificantly simplifies the analysis process reduces design time and ensuresthe safety and reliability of the systems. These aspects are crucial for robotsoperating in human environments. Our results suggest that Petri nets could befurther explored as a useful tool for the formal description and in-depthanalysis of the properties of robotic systems. |


| Item |Content|
| --- |---|
|idx| 2407.06426v1 |
|title| DebUnc: Mitigating Hallucinations in Large Language Model Agent Communication with Uncertainty Estimations |
|authors| Luke YoffeAlfonso AmayuelasWilliam Yang Wang
|links| http://arxiv.org/abs/2407.06426v1 |
|updated| 2024-07-08 22:15:01 UTC |
|summary| To enhance Large Language Model LLM capabilities multi-agent debates havebeen introduced where multiple LLMs discuss solutions to a problem overseveral rounds of debate. However LLMs often produce incorrect responses thatappear deceptively confident which can mislead other agents. This is partlybecause agents do not express their confidence levels during standard debates.To address this we introduce DebUnc a multi-agent debate framework that usesuncertainty metrics to assess agent confidence levels. We adapted the LLMattention mechanism to adjust token weights based on confidence levels and alsoexplored using textual prompts to convey confidence. Our evaluations acrossvarious benchmarks show that attention-based methods are particularlyeffective and that as uncertainty metrics evolve performance will continue toincrease. The code is available at https://github.com/lukeyoffe/debunc |


