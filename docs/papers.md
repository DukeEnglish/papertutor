# cs.CL 

| Item |Content|
| --- |---|
|idx| 2404.10774v1 |
|title| MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents |
|authors| Liyan TangPhilippe LabanGreg Durrett
|links| http://arxiv.org/abs/2404.10774v1 |
|updated| 2024-04-16 17:59:10 UTC |
|summary| Recognizing if LLM output can be grounded in evidence is central to manytasks in NLP: retrieval-augmented generation summarization document-groundeddialogue and more. Current approaches to this kind of fact-checking arebased on verifying each piece of a model generation against potential evidenceusing an LLM. However this process can be very computationally expensiverequiring many calls to LLMs to check a single response. In this work we showhow to build small models that have GPT-4-level performance but for 400x lowercost. We do this by constructing synthetic training data with GPT-4 whichinvolves creating realistic yet challenging instances of factual errors via astructured generation procedure. Training on this data teaches models to checkeach fact in the claim and recognize synthesis of information across sentences.For evaluation we unify pre-existing datasets into a benchmark LLM-AggreFactcollected from recent work on fact-checking and grounding LLM generations. Ourbest system MiniCheck-FT5 770M parameters outperforms all systems ofcomparable size and reaches GPT-4 accuracy. We release LLM-AggreFact code fordata synthesis and models. |


| Item |Content|
| --- |---|
|idx| 2404.10763v1 |
|title| LaDiC: Are Diffusion Models Really Inferior to Autoregressive Counterparts for Image-to-Text Generation? |
|authors| Yuchi WangShuhuai RenRundong GaoLinli YaoQingyan GuoKaikai AnJianhong BaiXu Sun
|links| http://arxiv.org/abs/2404.10763v1 |
|updated| 2024-04-16 17:47:16 UTC |
|summary| Diffusion models have exhibited remarkable capabilities in text-to-imagegeneration. However their performance in image-to-text generationspecifically image captioning has lagged behind Auto-Regressive AR modelscasting doubt on their applicability for such tasks. In this work we revisitdiffusion models highlighting their capacity for holistic context modeling andparallel decoding. With these benefits diffusion models can alleviate theinherent limitations of AR methods including their slow inference speed errorpropagation and unidirectional constraints. Furthermore we identify the priorunderperformance of diffusion models stemming from the absence of an effectivelatent space for image-text alignment and the discrepancy between continuousdiffusion processes and discrete textual data. In response we introduce anovel architecture LaDiC which utilizes a split BERT to create a dedicatedlatent space for captions and integrates a regularization module to managevarying text lengths. Our framework also includes a diffuser for semanticimage-to-text conversion and a BackRefine technique to enhance tokeninteractivity during inference. LaDiC achieves state-of-the-art performance fordiffusion-based methods on the MS COCO dataset with 38.2 BLEU4 and 126.2CIDEr demonstrating exceptional performance without pre-training or ancillarymodules. This indicates strong competitiveness with AR models revealing thepreviously untapped potential of diffusion models in image-to-text generation. |


| Item |Content|
| --- |---|
|idx| 2404.10757v1 |
|title| Deep Learning and LLM-based Methods Applied to Stellar Lightcurve Classification |
|authors| Yu-Yang LiYu BaiCunshi WangMengwei QuZiteng LuRoberto SoriaJifeng Liu
|links| http://arxiv.org/abs/2404.10757v1 |
|updated| 2024-04-16 17:35:25 UTC |
|summary| Light curves serve as a valuable source of information on stellar formationand evolution. With the rapid advancement of machine learning techniques itcan be effectively processed to extract astronomical patterns and information.In this study we present a comprehensive evaluation of deep-learning and largelanguage model LLM based models for the automatic classification of variablestar light curves based on large datasets from the Kepler and K2 missions.Special emphasis is placed on Cepheids RR Lyrae and eclipsing binariesexamining the influence of observational cadence and phase distribution onclassification precision. Employing AutoDL optimization we achieve strikingperformance with the 1D-ConvolutionBiLSTM architecture and the SwinTransformer hitting accuracies of 94 and 99 correspondingly with thelatter demonstrating a notable 83 accuracy in discerning the elusive Type IICepheids-comprising merely 0.02 of the total dataset.We unveil StarWhisperLightCurve LC an innovative Series comprising three LLM-based models: LLMmultimodal large language model MLLM and Large Audio Language Model LALM.Each model is fine-tuned with strategic prompt engineering and customizedtraining methods to explore the emergent abilities of these models forastronomical data. Remarkably StarWhisper LC Series exhibit high accuraciesaround 90 significantly reducing the need for explicit feature engineeringthereby paving the way for streamlined parallel data processing and theprogression of multifaceted multimodal models in astronomical applications. Thestudy furnishes two detailed catalogs illustrating the impacts of phase andsampling intervals on deep learning classification accuracy showing that asubstantial decrease of up to 14 in observation duration and 21 in samplingpoints can be realized without compromising accuracy by more than 10. |


| Item |Content|
| --- |---|
|idx| 2404.10719v1 |
|title| Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study |
|authors| Shusheng XuWei FuJiaxuan GaoWenjie YeWeilin LiuZhiyu MeiGuangju WangChao YuYi Wu
|links| http://arxiv.org/abs/2404.10719v1 |
|updated| 2024-04-16 16:51:53 UTC |
|summary| Reinforcement Learning from Human Feedback RLHF is currently the mostwidely used method to align large language models LLMs with humanpreferences. Existing RLHF methods can be roughly categorized as eitherreward-based or reward-free. Novel applications such as ChatGPT and Claudeleverage reward-based methods that first learn a reward model and applyactor-critic algorithms such as Proximal Policy Optimization PPO. Howeverin academic benchmarks state-of-the-art results are often achieved viareward-free methods such as Direct Preference Optimization DPO. Is DPO trulysuperior to PPO Why does PPO perform poorly on these benchmarks In thispaper we first conduct both theoretical and empirical studies on thealgorithmic properties of DPO and show that DPO may have fundamentallimitations. Moreover we also comprehensively examine PPO and reveal the keyfactors for the best performances of PPO in fine-tuning LLMs. Finally webenchmark DPO and PPO across various a collection of RLHF testbeds rangingfrom dialogue to code generation. Experiment results demonstrate that PPO isable to surpass other alignment methods in all cases and achievestate-of-the-art results in challenging code competitions. |


| Item |Content|
| --- |---|
|idx| 2404.10710v1 |
|title| Dual Modalities of Text: Visual and Textual Generative Pre-training |
|authors| Yekun ChaiQingyi LiuJingwu XiaoShuohuan WangYu SunHua Wu
|links| http://arxiv.org/abs/2404.10710v1 |
|updated| 2024-04-16 16:36:50 UTC |
|summary| Harnessing visual texts represents a burgeoning frontier in the evolution oflanguage modeling. In this paper we introduce a novel pre-training frameworkfor a suite of pixel-based autoregressive language models pre-training on acorpus of over 400 million documents rendered as RGB images. Our approach ischaracterized by a dual-modality training regimen engaging both visual datathrough next patch prediction with a regression head and textual data via nexttoken prediction with a classification head. This study is particularly focusedon investigating the synergistic interplay between visual and textualmodalities of language. Our comprehensive evaluation across a diverse array ofbenchmarks reveals that the confluence of visual and textual data substantiallyaugments the efficacy of pixel-based language models. Notably our findingsshow that a unidirectional pixel-based model devoid of textual data duringtraining can match the performance levels of advanced bidirectionalpixel-based models on various language understanding benchmarks. This workhighlights the considerable untapped potential of integrating visual andtextual information for language modeling purposes. We will release our codedata and checkpoints to inspire further research advancement. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2404.10775v1 |
|title| COMBO: Compositional World Models for Embodied Multi-Agent Cooperation |
|authors| Hongxin ZhangZeyuan WangQiushi LyuZheyuan ZhangSunli ChenTianmin ShuYilun DuChuang Gan
|links| http://arxiv.org/abs/2404.10775v1 |
|updated| 2024-04-16 17:59:11 UTC |
|summary| In this paper we investigate the problem of embodied multi-agentcooperation where decentralized agents must cooperate given only partialegocentric views of the world. To effectively plan in this setting in contrastto learning world dynamics in a single-agent scenario we must simulate worlddynamics conditioned on an arbitrary number of agents actions given onlypartial egocentric visual observations of the world. To address this issue ofpartial observability we first train generative models to estimate the overallworld state given partial egocentric observations. To enable accuratesimulation of multiple sets of actions on this world state we then propose tolearn a compositional world model for multi-agent cooperation by factorizingthe naturally composable joint actions of multiple agents and compositionallygenerating the video. By leveraging this compositional world model incombination with Vision Language Models to infer the actions of other agentswe can use a tree search procedure to integrate these modules and facilitateonline cooperative planning. To evaluate the efficacy of our methods we createtwo challenging embodied multi-agent long-horizon cooperation tasks using theThreeDWorld simulator and conduct experiments with 2-4 agents. The results showour compositional world model is effective and the framework enables theembodied agents to cooperate efficiently with different agents across varioustasks and an arbitrary number of agents showing the promising future of ourproposed framework. More videos can be found athttps://vis-www.cs.umass.edu/combo/. |


| Item |Content|
| --- |---|
|idx| 2404.10774v1 |
|title| MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents |
|authors| Liyan TangPhilippe LabanGreg Durrett
|links| http://arxiv.org/abs/2404.10774v1 |
|updated| 2024-04-16 17:59:10 UTC |
|summary| Recognizing if LLM output can be grounded in evidence is central to manytasks in NLP: retrieval-augmented generation summarization document-groundeddialogue and more. Current approaches to this kind of fact-checking arebased on verifying each piece of a model generation against potential evidenceusing an LLM. However this process can be very computationally expensiverequiring many calls to LLMs to check a single response. In this work we showhow to build small models that have GPT-4-level performance but for 400x lowercost. We do this by constructing synthetic training data with GPT-4 whichinvolves creating realistic yet challenging instances of factual errors via astructured generation procedure. Training on this data teaches models to checkeach fact in the claim and recognize synthesis of information across sentences.For evaluation we unify pre-existing datasets into a benchmark LLM-AggreFactcollected from recent work on fact-checking and grounding LLM generations. Ourbest system MiniCheck-FT5 770M parameters outperforms all systems ofcomparable size and reaches GPT-4 accuracy. We release LLM-AggreFact code fordata synthesis and models. |


| Item |Content|
| --- |---|
|idx| 2404.10763v1 |
|title| LaDiC: Are Diffusion Models Really Inferior to Autoregressive Counterparts for Image-to-Text Generation? |
|authors| Yuchi WangShuhuai RenRundong GaoLinli YaoQingyan GuoKaikai AnJianhong BaiXu Sun
|links| http://arxiv.org/abs/2404.10763v1 |
|updated| 2024-04-16 17:47:16 UTC |
|summary| Diffusion models have exhibited remarkable capabilities in text-to-imagegeneration. However their performance in image-to-text generationspecifically image captioning has lagged behind Auto-Regressive AR modelscasting doubt on their applicability for such tasks. In this work we revisitdiffusion models highlighting their capacity for holistic context modeling andparallel decoding. With these benefits diffusion models can alleviate theinherent limitations of AR methods including their slow inference speed errorpropagation and unidirectional constraints. Furthermore we identify the priorunderperformance of diffusion models stemming from the absence of an effectivelatent space for image-text alignment and the discrepancy between continuousdiffusion processes and discrete textual data. In response we introduce anovel architecture LaDiC which utilizes a split BERT to create a dedicatedlatent space for captions and integrates a regularization module to managevarying text lengths. Our framework also includes a diffuser for semanticimage-to-text conversion and a BackRefine technique to enhance tokeninteractivity during inference. LaDiC achieves state-of-the-art performance fordiffusion-based methods on the MS COCO dataset with 38.2 BLEU4 and 126.2CIDEr demonstrating exceptional performance without pre-training or ancillarymodules. This indicates strong competitiveness with AR models revealing thepreviously untapped potential of diffusion models in image-to-text generation. |


| Item |Content|
| --- |---|
|idx| 2404.10740v1 |
|title| N-Agent Ad Hoc Teamwork |
|authors| Caroline WangArrasy RahmanIshan DurugkarElad LiebmanPeter Stone
|links| http://arxiv.org/abs/2404.10740v1 |
|updated| 2024-04-16 17:13:08 UTC |
|summary| Current approaches to learning cooperative behaviors in multi-agent settingsassume relatively restrictive settings. In standard fully cooperativemulti-agent reinforcement learning the learning algorithm controlstextitall agents in the scenario while in ad hoc teamwork the learningalgorithm usually assumes control over only a textitsingle agent in thescenario. However many cooperative settings in the real world are much lessrestrictive. For example in an autonomous driving scenario a company mighttrain its cars with the same learning algorithm yet once on the road thesecars must cooperate with cars from another company. Towards generalizing theclass of scenarios that cooperative learning methods can address we introduceN-agent ad hoc teamwork in which a set of autonomous agents must interactand cooperate with dynamically varying numbers and types of teammates atevaluation time. This paper formalizes the problem and proposes thetextitPolicy Optimization with Agent Modelling POAM algorithm. POAM is apolicy gradient multi-agent reinforcement learning approach to the NAHTproblem that enables adaptation to diverse teammate behaviors by learningrepresentations of teammate behaviors. Empirical evaluation on StarCraft IItasks shows that POAM improves cooperative task returns compared to baselineapproaches and enables out-of-distribution generalization to unseen teammates. |


| Item |Content|
| --- |---|
|idx| 2404.10733v1 |
|title| Bootstrapping Linear Models for Fast Online Adaptation in Human-Agent Collaboration |
|authors| Benjamin A NewmanChris PaxtonKris KitaniHenny Admoni
|links| http://arxiv.org/abs/2404.10733v1 |
|updated| 2024-04-16 17:05:43 UTC |
|summary| Agents that assist people need to have well-initialized policies that canadapt quickly to align with their partners reward functions. Initializingpolicies to maximize performance with unknown partners can be achieved bybootstrapping nonlinear models using imitation learning over large offlinedatasets. Such policies can require prohibitive computation to fine-tunein-situ and therefore may miss critical run-time information about a partnersreward function as expressed through their immediate behavior. In contrastonline logistic regression using low-capacity models performs rapid inferenceand fine-tuning updates and thus can make effective use of immediate in-taskbehavior for reward function alignment. However these low-capacity modelscannot be bootstrapped as effectively by offline datasets and thus have poorinitializations. We propose BLR-HAC Bootstrapped Logistic Regression for HumanAgent Collaboration which bootstraps large nonlinear models to learn theparameters of a low-capacity model which then uses online logistic regressionfor updates during collaboration. We test BLR-HAC in a simulated surfacerearrangement task and demonstrate that it achieves higher zero-shot accuracythan shallow methods and takes far less computation to adapt online while stillachieving similar performance to fine-tuned large nonlinear models. For codeplease see our project page https://sites.google.com/view/blr-hac. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2404.10776v1 |
|title| Nearly Optimal Algorithms for Contextual Dueling Bandits from Adversarial Feedback |
|authors| Qiwei DiJiafan HeQuanquan Gu
|links| http://arxiv.org/abs/2404.10776v1 |
|updated| 2024-04-16 17:59:55 UTC |
|summary| Learning from human feedback plays an important role in aligning generativemodels such as large language models LLM. However the effectiveness of thisapproach can be influenced by adversaries who may intentionally providemisleading preferences to manipulate the output in an undesirable or harmfuldirection. To tackle this challenge we study a specific model within thisproblem domain--contextual dueling bandits with adversarial feedback where thetrue preference label can be flipped by an adversary. We propose an algorithmnamely robust contextual dueling bandit algo which is based onuncertainty-weighted maximum likelihood estimation. Our algorithm achieves antilde OdsqrtTdC regret bound where T is the number of rounds dis the dimension of the context and  0 le C le T is the total number ofadversarial feedback. We also prove a lower bound to show that our regret boundis nearly optimal both in scenarios with and without C0 adversarialfeedback. Additionally we conduct experiments to evaluate our proposedalgorithm against various types of adversarial feedback. Experimental resultsdemonstrate its superiority over the state-of-the-art dueling bandit algorithmsin the presence of adversarial feedback. |


| Item |Content|
| --- |---|
|idx| 2404.10771v1 |
|title| TENG: Time-Evolving Natural Gradient for Solving PDEs with Deep Neural Net |
|authors| Zhuo ChenJacob McCarranEsteban VizcainoMarin SoljačićDi Luo
|links| http://arxiv.org/abs/2404.10771v1 |
|updated| 2024-04-16 17:55:31 UTC |
|summary| Partial differential equations PDEs are instrumental for modeling dynamicalsystems in science and engineering. The advent of neural networks has initiateda significant shift in tackling these complexities though challenges inaccuracy persist especially for initial value problems. In this paper weintroduce the textitTime-Evolving Natural Gradient TENG generalizingtime-dependent variational principles and optimization-based time integrationleveraging natural gradient optimization to obtain high accuracy inneural-network-based PDE solutions. Our comprehensive development includesalgorithms like TENG-Euler and its high-order variants such as TENG-Heuntailored for enhanced precision and efficiency. TENGs effectiveness is furthervalidated through its performance surpassing current leading methods andachieving machine precision in step-by-step optimizations across a spectrum ofPDEs including the heat equation Allen-Cahn equation and Burgers equation. |


| Item |Content|
| --- |---|
|idx| 2404.10769v1 |
|title| Finite-dimensional approximations of push-forwards on locally analytic functionals and truncation of least-squares polynomials |
|authors| Isao Ishikawa
|links| http://arxiv.org/abs/2404.10769v1 |
|updated| 2024-04-16 17:53:59 UTC |
|summary| This paper introduces a theoretical framework for investigating analytic mapsfrom finite discrete data elucidating mathematical machinery underlying thepolynomial approximation with least-squares in multivariate situations. Ourapproach is to consider the push-forward on the space of locally analyticfunctionals instead of directly handling the analytic map itself. We establisha methodology enabling appropriate finite-dimensional approximation of thepush-forward from finite discrete data through the theory of theFourier--Borel transform and the Fock space. Moreover we prove a rigorousconvergence result with a convergence rate. As an application we prove that itis not the least-squares polynomial but the polynomial obtained by truncatingits higher-degree terms that approximates analytic functions and furtherallows for approximation beyond the support of the data distribution. Oneadvantage of our theory is that it enables us to apply linear algebraicoperations to the finite-dimensional approximation of the push-forward.Utilizing this we prove the convergence of a method for approximating ananalytic vector field from finite data of the flow map of an ordinarydifferential equation. |


| Item |Content|
| --- |---|
|idx| 2404.10764v1 |
|title| Confidential Federated Computations |
|authors| Hubert EichnerDaniel RamageKallista BonawitzDzmitry HubaTiziano SantoroBrett McLarnonTimon Van OverveldtNova FallenPeter KairouzAlbert CheuKatharine DalyAdria GasconMarco GruteserBrendan McMahan
|links| http://arxiv.org/abs/2404.10764v1 |
|updated| 2024-04-16 17:47:27 UTC |
|summary| Federated Learning and Analytics FLA have seen widespread adoption bytechnology platforms for processing sensitive on-device data. However basicFLA systems have privacy limitations: they do not necessarily requireanonymization mechanisms like differential privacy DP and provide limitedprotections against a potentially malicious service provider. Adding DP to abasic FLA system currently requires either adding excessive noise to eachdevices updates or assuming an honest service provider that correctlyimplements the mechanism and only uses the privatized outputs. Securemultiparty computation SMPC -based oblivious aggregations can limit theservice providers access to individual user updates and improve DP tradeoffsbut the tradeoffs are still suboptimal and they suffer from scalabilitychallenges and susceptibility to Sybil attacks. This paper introduces a novelsystem architecture that leverages trusted execution environments TEEs andopen-sourcing to both ensure confidentiality of server-side computations andprovide externally verifiable privacy properties bolstering the robustness andtrustworthiness of private federated computations. |


| Item |Content|
| --- |---|
|idx| 2404.10761v1 |
|title| TorchSurv: A Lightweight Package for Deep Survival Analysis |
|authors| Melodie MonodPeter KruscheQian CaoBerkman SahinerNicholas PetrickDavid OhlssenThibaud Coroller
|links| http://arxiv.org/abs/2404.10761v1 |
|updated| 2024-04-16 17:41:17 UTC |
|summary| TorchSurv is a Python package that serves as a companion tool to perform deepsurvival modeling within the PyTorch environment. Unlike existing librariesthat impose specific parametric forms TorchSurv enables the use of customPyTorch-based deep survival mod- els. With its lightweight design minimalinput requirements full PyTorch backend and freedom from restrictive survivalmodel parameterizations TorchSurv facilitates efficient deep survival modelimplementation and is particularly beneficial for high-dimensional and complexinput data scenarios |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2404.10775v1 |
|title| COMBO: Compositional World Models for Embodied Multi-Agent Cooperation |
|authors| Hongxin ZhangZeyuan WangQiushi LyuZheyuan ZhangSunli ChenTianmin ShuYilun DuChuang Gan
|links| http://arxiv.org/abs/2404.10775v1 |
|updated| 2024-04-16 17:59:11 UTC |
|summary| In this paper we investigate the problem of embodied multi-agentcooperation where decentralized agents must cooperate given only partialegocentric views of the world. To effectively plan in this setting in contrastto learning world dynamics in a single-agent scenario we must simulate worlddynamics conditioned on an arbitrary number of agents actions given onlypartial egocentric visual observations of the world. To address this issue ofpartial observability we first train generative models to estimate the overallworld state given partial egocentric observations. To enable accuratesimulation of multiple sets of actions on this world state we then propose tolearn a compositional world model for multi-agent cooperation by factorizingthe naturally composable joint actions of multiple agents and compositionallygenerating the video. By leveraging this compositional world model incombination with Vision Language Models to infer the actions of other agentswe can use a tree search procedure to integrate these modules and facilitateonline cooperative planning. To evaluate the efficacy of our methods we createtwo challenging embodied multi-agent long-horizon cooperation tasks using theThreeDWorld simulator and conduct experiments with 2-4 agents. The results showour compositional world model is effective and the framework enables theembodied agents to cooperate efficiently with different agents across varioustasks and an arbitrary number of agents showing the promising future of ourproposed framework. More videos can be found athttps://vis-www.cs.umass.edu/combo/. |


| Item |Content|
| --- |---|
|idx| 2404.10772v1 |
|title| Gaussian Opacity Fields: Efficient and Compact Surface Reconstruction in Unbounded Scenes |
|authors| Zehao YuTorsten SattlerAndreas Geiger
|links| http://arxiv.org/abs/2404.10772v1 |
|updated| 2024-04-16 17:57:19 UTC |
|summary| Recently 3D Gaussian Splatting 3DGS has demonstrated impressive novel viewsynthesis results while allowing the rendering of high-resolution images inreal-time. However leveraging 3D Gaussians for surface reconstruction posessignificant challenges due to the explicit and disconnected nature of 3DGaussians. In this work we present Gaussian Opacity Fields GOF a novelapproach for efficient high-quality and compact surface reconstruction inunbounded scenes. Our GOF is derived from ray-tracing-based volume rendering of3D Gaussians enabling direct geometry extraction from 3D Gaussians byidentifying its levelset without resorting to Poisson reconstruction or TSDFfusion as in previous work. We approximate the surface normal of Gaussians asthe normal of the ray-Gaussian intersection plane enabling the application ofregularization that significantly enhances geometry. Furthermore we develop anefficient geometry extraction method utilizing marching tetrahedra where thetetrahedral grids are induced from 3D Gaussians and thus adapt to the scenescomplexity. Our evaluations reveal that GOF surpasses existing 3DGS-basedmethods in surface reconstruction and novel view synthesis. Further itcompares favorably to or even outperforms neural implicit methods in bothquality and speed. |


| Item |Content|
| --- |---|
|idx| 2404.10766v1 |
|title| RapidVol: Rapid Reconstruction of 3D Ultrasound Volumes from Sensorless 2D Scans |
|authors| Mark C. EidPak-Hei YeungMadeleine K. WyburdJoão F. HenriquesAna I. L. Namburete
|links| http://arxiv.org/abs/2404.10766v1 |
|updated| 2024-04-16 17:50:09 UTC |
|summary| Two-dimensional 2D freehand ultrasonography is one of the most commonlyused medical imaging modalities particularly in obstetrics and gynaecology.However it only captures 2D cross-sectional views of inherently 3D anatomieslosing valuable contextual information. As an alternative to requiring costlyand complex 3D ultrasound scanners 3D volumes can be constructed from 2D scansusing machine learning. However this usually requires long computational time.Here we propose RapidVol: a neural representation framework to speed upslice-to-volume ultrasound reconstruction. We use tensor-rank decomposition todecompose the typical 3D volume into sets of tri-planes and store thoseinstead as well as a small neural network. A set of 2D ultrasound scans withtheir ground truth or estimated 3D position and orientation pose is allthat is required to form a complete 3D reconstruction. Reconstructions areformed from real fetal brain scans and then evaluated by requesting novelcross-sectional views. When compared to prior approaches based on fullyimplicit representation e.g. neural radiance fields our method is over 3xquicker 46 more accurate and if given inaccurate poses is more robust.Further speed-up is also possible by reconstructing from a structural priorrather than from scratch. |


| Item |Content|
| --- |---|
|idx| 2404.10765v1 |
|title| RefFusion: Reference Adapted Diffusion Models for 3D Scene Inpainting |
|authors| Ashkan MirzaeiRiccardo De LutioSeung Wook KimDavid AcunaJonathan KellySanja FidlerIgor GilitschenskiZan Gojcic
|links| http://arxiv.org/abs/2404.10765v1 |
|updated| 2024-04-16 17:50:02 UTC |
|summary| Neural reconstruction approaches are rapidly emerging as the preferredrepresentation for 3D scenes but their limited editability is still posing achallenge. In this work we propose an approach for 3D scene inpainting -- thetask of coherently replacing parts of the reconstructed scene with desiredcontent. Scene inpainting is an inherently ill-posed task as there exist manysolutions that plausibly replace the missing content. A good inpainting methodshould therefore not only enable high-quality synthesis but also a high degreeof control. Based on this observation we focus on enabling explicit controlover the inpainted content and leverage a reference image as an efficient meansto achieve this goal. Specifically we introduce RefFusion a novel 3Dinpainting method based on a multi-scale personalization of an image inpaintingdiffusion model to the given reference view. The personalization effectivelyadapts the prior distribution to the target scene resulting in a lowervariance of score distillation objective and hence significantly sharperdetails. Our framework achieves state-of-the-art results for object removalwhile maintaining high controllability. We further demonstrate the generalityof our formulation on other downstream tasks such as object insertion sceneoutpainting and sparse view reconstruction. |


| Item |Content|
| --- |---|
|idx| 2404.10763v1 |
|title| LaDiC: Are Diffusion Models Really Inferior to Autoregressive Counterparts for Image-to-Text Generation? |
|authors| Yuchi WangShuhuai RenRundong GaoLinli YaoQingyan GuoKaikai AnJianhong BaiXu Sun
|links| http://arxiv.org/abs/2404.10763v1 |
|updated| 2024-04-16 17:47:16 UTC |
|summary| Diffusion models have exhibited remarkable capabilities in text-to-imagegeneration. However their performance in image-to-text generationspecifically image captioning has lagged behind Auto-Regressive AR modelscasting doubt on their applicability for such tasks. In this work we revisitdiffusion models highlighting their capacity for holistic context modeling andparallel decoding. With these benefits diffusion models can alleviate theinherent limitations of AR methods including their slow inference speed errorpropagation and unidirectional constraints. Furthermore we identify the priorunderperformance of diffusion models stemming from the absence of an effectivelatent space for image-text alignment and the discrepancy between continuousdiffusion processes and discrete textual data. In response we introduce anovel architecture LaDiC which utilizes a split BERT to create a dedicatedlatent space for captions and integrates a regularization module to managevarying text lengths. Our framework also includes a diffuser for semanticimage-to-text conversion and a BackRefine technique to enhance tokeninteractivity during inference. LaDiC achieves state-of-the-art performance fordiffusion-based methods on the MS COCO dataset with 38.2 BLEU4 and 126.2CIDEr demonstrating exceptional performance without pre-training or ancillarymodules. This indicates strong competitiveness with AR models revealing thepreviously untapped potential of diffusion models in image-to-text generation. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2404.10759v1 |
|title| Laplace-HDC: Understanding the geometry of binary hyperdimensional computing |
|authors| Saeid PourmandWyatt D. WhitingAlireza AghasiNicholas F. Marshall
|links| http://arxiv.org/abs/2404.10759v1 |
|updated| 2024-04-16 17:36:21 UTC |
|summary| This paper studies the geometry of binary hyperdimensional computing HDC acomputational scheme in which data are encoded using high-dimensional binaryvectors. We establish a result about the similarity structure induced by theHDC binding operator and show that the Laplace kernel naturally arises in thissetting motivating our new encoding method Laplace-HDC which improves uponprevious methods. We describe how our results indicate limitations of binaryHDC in encoding spatial information from images and discuss potentialsolutions including using Haar convolutional features and the definition of atranslation-equivariant HDC encoding. Several numerical experimentshighlighting the improved accuracy of Laplace-HDC in contrast to alternativemethods are presented. We also numerically study other aspects of the proposedframework such as robustness and the underlying translation-equivariantencoding. |


| Item |Content|
| --- |---|
|idx| 2404.10728v1 |
|title| Randomized Exploration in Cooperative Multi-Agent Reinforcement Learning |
|authors| Hao-Lun HsuWeixin WangMiroslav PajicPan Xu
|links| http://arxiv.org/abs/2404.10728v1 |
|updated| 2024-04-16 17:01:38 UTC |
|summary| We present the first study on provably efficient randomized exploration incooperative multi-agent reinforcement learning MARL. We propose a unifiedalgorithm framework for randomized exploration in parallel Markov DecisionProcesses MDPs and two Thompson Sampling TS-type algorithms CoopTS-PHEand CoopTS-LMC incorporating the perturbed-history exploration PHE strategyand the Langevin Monte Carlo exploration LMC strategy respectively which areflexible in design and easy to implement in practice. For a special class ofparallel MDPs where the transition is approximately linear we theoreticallyprove that both CoopTS-PHE and CoopTS-LMC achieve awidetildemathcalOd3/2H2sqrtMK regret bound with communicationcomplexity widetildemathcalOdHM2 where d is the featuredimension H is the horizon length M is the number of agents and K isthe number of episodes. This is the first theoretical result for randomizedexploration in cooperative MARL. We evaluate our proposed method on multipleparallel RL environments including a deep exploration problem textiti.e.N-chain a video game and a real-world problem in energy systems. Ourexperimental results support that our framework can achieve better performanceeven under conditions of misspecified transition models. Additionally weestablish a connection between our unified framework and the practicalapplication of federated learning. |


| Item |Content|
| --- |---|
|idx| 2404.10727v1 |
|title| How Deep Networks Learn Sparse and Hierarchical Data: the Sparse Random Hierarchy Model |
|authors| Umberto TomasiniMatthieu Wyart
|links| http://arxiv.org/abs/2404.10727v1 |
|updated| 2024-04-16 17:01:27 UTC |
|summary| Understanding what makes high-dimensional data learnable is a fundamentalquestion in machine learning. On the one hand it is believed that the successof deep learning lies in its ability to build a hierarchy of representationsthat become increasingly more abstract with depth going from simple featureslike edges to more complex concepts. On the other hand learning to beinsensitive to invariances of the task such as smooth transformations forimage datasets has been argued to be important for deep networks and itstrongly correlates with their performance. In this work we aim to explainthis correlation and unify these two viewpoints. We show that by introducingsparsity to generative hierarchical models of data the task acquiresinsensitivity to spatial transformations that are discrete versions of smoothtransformations. In particular we introduce the Sparse Random Hierarchy ModelSRHM where we observe and rationalize that a hierarchical representationmirroring the hierarchical model is learnt precisely when such insensitivity islearnt thereby explaining the strong correlation between the latter andperformance. Moreover we quantify how the sample complexity of CNNs learningthe SRHM depends on both the sparsity and hierarchical structure of the task. |


| Item |Content|
| --- |---|
|idx| 2404.10561v1 |
|title| HiGraphDTI: Hierarchical Graph Representation Learning for Drug-Target Interaction Prediction |
|authors| Bin LiuSiqi WuJin WangXin DengAo Zhou
|links| http://arxiv.org/abs/2404.10561v1 |
|updated| 2024-04-16 13:35:24 UTC |
|summary| The discovery of drug-target interactions DTIs plays a crucial role inpharmaceutical development. The deep learning model achieves more accurateresults in DTI prediction due to its ability to extract robust and expressivefeatures from drug and target chemical structures. However existing deeplearning methods typically generate drug features via aggregating molecularatom representations ignoring the chemical properties carried by motifs i.e.substructures of the molecular graph. The atom-drug double-level molecularrepresentation learning can not fully exploit structure information and failsto interpret the DTI mechanism from the motif perspective. In additionsequential model-based target feature extraction either fuses limitedcontextual information or requires expensive computational resources. To tacklethe above issues we propose a hierarchical graph representation learning-basedDTI prediction method HiGraphDTI. Specifically HiGraphDTI learnshierarchical drug representations from triple-level molecular graphs tothoroughly exploit chemical information embedded in atoms motifs andmolecules. Then an attentional feature fusion module incorporates informationfrom different receptive fields to extract expressive target features.Last thehierarchical attention mechanism identifies crucial molecular segments whichoffers complementary views for interpreting interaction mechanisms. Theexperiment results not only demonstrate the superiority of HiGraphDTI to thestate-of-the-art methods but also confirm the practical ability of our modelin interaction interpretation and new DTI discovery. |


| Item |Content|
| --- |---|
|idx| 2404.10550v1 |
|title| Analytical Approximation of the ELBO Gradient in the Context of the Clutter Problem |
|authors| Roumen Nikolaev Popov
|links| http://arxiv.org/abs/2404.10550v1 |
|updated| 2024-04-16 13:19:46 UTC |
|summary| We propose an analytical solution for approximating the gradient of theEvidence Lower Bound ELBO in variational inference problems where thestatistical model is a Bayesian network consisting of observations drawn from amixture of a Gaussian distribution embedded in unrelated clutter known as theclutter problem. The method employs the reparameterization trick to move thegradient operator inside the expectation and relies on the assumption thatbecause the likelihood factorizes over the observed data the variationaldistribution is generally more compactly supported than the Gaussiandistribution in the likelihood factors. This allows efficient localapproximation of the individual likelihood factors which leads to ananalytical solution for the integral defining the gradient expectation. Weintegrate the proposed gradient approximation as the expectation step in an EMExpectation Maximization algorithm for maximizing ELBO and test againstclassical deterministic approaches in Bayesian inference such as the Laplaceapproximation Expectation Propagation and Mean-Field Variational Inference.The proposed method demonstrates good accuracy and rate of convergence togetherwith linear computational complexity. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2404.10754v1 |
|title| A Systematic Survey of the Gemini Principles for Digital Twin Ontologies |
|authors| James Michael ToothNilufer TuptukJeremy Daniel McKendrick Watson
|links| http://arxiv.org/abs/2404.10754v1 |
|updated| 2024-04-16 17:34:24 UTC |
|summary| Ontologies are widely used for achieving interoperable Digital Twins DTwsyet competing DTw definitions compound interoperability issues. Semanticallylinking these differing twins is feasible through ontologies and CognitiveDigital Twins CDTws. However it is often unclear how ontology use bolstersbroader DTw advancements. This article presents a systematic survey followingthe PRISMA method to explore the potential of ontologies to support DTws tomeet the Centre for Digital Built Britains Gemini Principles and aims to linkprogress in ontologies to this framework. The Gemini Principles focus on commonDTw requirements considering: Purpose for 1 Public Good 2 Value Creationand 3 Insight Trustworthiness with sufficient 4 Security 5 Openness and6 Quality and appropriate Functionality of 7 Federation 8 Curation and 9Evolution. This systematic literature review examines the role of ontologies infacilitating each principle. Existing research uses ontologies to solve DTwchallenges within these principles particularly by connecting DTws optimisingdecisionmaking and reasoning governance policies. Furthermore analysing thesectoral distribution of literature found that research encompassing thecrossover of ontologies DTws and the Gemini Principles is emerging and thatmost innovation is predominantly within manufacturing and built environmentsectors. Critical gaps for researchers industry practitioners andpolicymakers are subsequently identified. |


| Item |Content|
| --- |---|
|idx| 2404.10733v1 |
|title| Bootstrapping Linear Models for Fast Online Adaptation in Human-Agent Collaboration |
|authors| Benjamin A NewmanChris PaxtonKris KitaniHenny Admoni
|links| http://arxiv.org/abs/2404.10733v1 |
|updated| 2024-04-16 17:05:43 UTC |
|summary| Agents that assist people need to have well-initialized policies that canadapt quickly to align with their partners reward functions. Initializingpolicies to maximize performance with unknown partners can be achieved bybootstrapping nonlinear models using imitation learning over large offlinedatasets. Such policies can require prohibitive computation to fine-tunein-situ and therefore may miss critical run-time information about a partnersreward function as expressed through their immediate behavior. In contrastonline logistic regression using low-capacity models performs rapid inferenceand fine-tuning updates and thus can make effective use of immediate in-taskbehavior for reward function alignment. However these low-capacity modelscannot be bootstrapped as effectively by offline datasets and thus have poorinitializations. We propose BLR-HAC Bootstrapped Logistic Regression for HumanAgent Collaboration which bootstraps large nonlinear models to learn theparameters of a low-capacity model which then uses online logistic regressionfor updates during collaboration. We test BLR-HAC in a simulated surfacerearrangement task and demonstrate that it achieves higher zero-shot accuracythan shallow methods and takes far less computation to adapt online while stillachieving similar performance to fine-tuned large nonlinear models. For codeplease see our project page https://sites.google.com/view/blr-hac. |


| Item |Content|
| --- |---|
|idx| 2404.10732v1 |
|title| Attention-Aware Visualization: Tracking and Responding to User Perception Over Time |
|authors| Arvind SrinivasanJohannes EllemosePeter W. S. ButcherPanagiotis D. RitsosNiklas Elmqvist
|links| http://arxiv.org/abs/2404.10732v1 |
|updated| 2024-04-16 17:04:32 UTC |
|summary| We propose the notion of Attention-Aware Visualizations AAVs that track theusers perception of a visual representation over time and feed thisinformation back to the visualization. Such context awareness is particularlyuseful for ubiquitous and immersive analytics where knowing which embeddedvisualizations the user is looking at can be used to make visualizations reactappropriately to the users attention: for example by highlighting data theuser has not yet seen. We can separate the approach into three components: 1measuring the users gaze on a visualization and its parts 2 tracking theusers attention over time and 3 reactively modifying the visualrepresentation based on the current attention metric. In this paper we presenttwo separate implementations of AAV: a 2D data-agnostic method for web-basedvisualizations that can use an embodied eyetracker to capture the users gazeand a 3D data-aware one that uses the stencil buffer to track the visibility ofeach individual mark in a visualization. Both methods provide similarmechanisms for accumulating attention over time and changing the appearance ofmarks in response. We also present results from a qualitative evaluationstudying visual feedback and triggering mechanisms for capturing andrevisualizing attention. |


| Item |Content|
| --- |---|
|idx| 2404.10706v1 |
|title| Cross-Language Evolution of Divergent Collective Memory Around the Arab Spring |
|authors| H. Laurie JonesBrian C. Keegan
|links| http://arxiv.org/abs/2404.10706v1 |
|updated| 2024-04-16 16:30:27 UTC |
|summary| The Arab Spring was a historic set of protests beginning in 2011 that toppledgovernments and led to major conflicts. Collective memories of events likethese can vary significantly across social contexts in response to politicalcultural and linguistic factors. While Wikipedia plays an important role indocumenting both historic and current events little attention has been givento how Wikipedia articles created in the aftermath of major events continueto evolve over years or decades. Using the archived content of ArabSpring-related topics across the Arabic and English Wikipedias between 2011 and2024 we define and evaluate multilingual measures of event saliencedeliberation contextualization and consolidation of collective memorysurrounding the Arab Spring. Our findings about the temporal evolution of theWikipedia articles content similarity across languages has implications fortheorizing about online collective memory processes and evaluating linguisticmodels trained on these data. |


| Item |Content|
| --- |---|
|idx| 2404.10690v1 |
|title| MathWriting: A Dataset For Handwritten Mathematical Expression Recognition |
|authors| Philippe GervaisAsya FadeevaAndrii Maksai
|links| http://arxiv.org/abs/2404.10690v1 |
|updated| 2024-04-16 16:10:23 UTC |
|summary| We introduce MathWriting the largest online handwritten mathematicalexpression dataset to date. It consists of 230k human-written samples and anadditional 400k synthetic ones. MathWriting can also be used for offline HMErecognition and is larger than all existing offline HME datasets likeIM2LATEX-100K. We introduce a benchmark based on MathWriting data in order toadvance research on both online and offline HME recognition. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2404.10775v1 |
|title| COMBO: Compositional World Models for Embodied Multi-Agent Cooperation |
|authors| Hongxin ZhangZeyuan WangQiushi LyuZheyuan ZhangSunli ChenTianmin ShuYilun DuChuang Gan
|links| http://arxiv.org/abs/2404.10775v1 |
|updated| 2024-04-16 17:59:11 UTC |
|summary| In this paper we investigate the problem of embodied multi-agentcooperation where decentralized agents must cooperate given only partialegocentric views of the world. To effectively plan in this setting in contrastto learning world dynamics in a single-agent scenario we must simulate worlddynamics conditioned on an arbitrary number of agents actions given onlypartial egocentric visual observations of the world. To address this issue ofpartial observability we first train generative models to estimate the overallworld state given partial egocentric observations. To enable accuratesimulation of multiple sets of actions on this world state we then propose tolearn a compositional world model for multi-agent cooperation by factorizingthe naturally composable joint actions of multiple agents and compositionallygenerating the video. By leveraging this compositional world model incombination with Vision Language Models to infer the actions of other agentswe can use a tree search procedure to integrate these modules and facilitateonline cooperative planning. To evaluate the efficacy of our methods we createtwo challenging embodied multi-agent long-horizon cooperation tasks using theThreeDWorld simulator and conduct experiments with 2-4 agents. The results showour compositional world model is effective and the framework enables theembodied agents to cooperate efficiently with different agents across varioustasks and an arbitrary number of agents showing the promising future of ourproposed framework. More videos can be found athttps://vis-www.cs.umass.edu/combo/. |


| Item |Content|
| --- |---|
|idx| 2404.10684v1 |
|title| Driver Fatigue Prediction using Randomly Activated Neural Networks for Smart Ridesharing Platforms |
|authors| Sree Pooja AkulaMukund TelukuntaVenkata Sriram Siddhardh Nadendla
|links| http://arxiv.org/abs/2404.10684v1 |
|updated| 2024-04-16 16:04:11 UTC |
|summary| Drivers in ridesharing platforms exhibit cognitive atrophy and fatigue asthey accept ride offers along the day which can have a significant impact onthe overall efficiency of the ridesharing platform. In contrast to the currentliterature which focuses primarily on modeling and learning driverspreferences across different ride offers this paper proposes a novel DynamicDiscounted Satisficing DDS heuristic to model and predict drivers sequentialride decisions during a given shift. Based on DDS heuristic a novel stochasticneural network with random activations is proposed to model DDS heuristic andpredict the final decision made by a given driver. The presence of randomactivations in the network necessitated the development of a novel trainingalgorithm called Sampling-Based Back Propagation Through Time SBPTT wheregradients are computed for independent instances of neural networks obtainedvia sampling the distribution of activation threshold and aggregated to updatethe network parameters. Using both simulation experiments as well as on realChicago taxi dataset this paper demonstrates the improved performance of theproposed approach when compared to state-of-the-art methods. |


| Item |Content|
| --- |---|
|idx| 2404.10641v1 |
|title| A Cloud Resources Portfolio Optimization Business Model - From Theory to Practice |
|authors| Valentin HaagMaximilian KiesslerBenedikt PittlErich Schikuta
|links| http://arxiv.org/abs/2404.10641v1 |
|updated| 2024-04-16 15:15:59 UTC |
|summary| Cloud resources have become increasingly important with many businessesusing cloud solutions to supplement or outright replace their existing ITinfrastructure. However as there is a plethora of providers with varyingproducts services and markets it has become increasingly more challenging tokeep track of the best solutions for each application. Cloud serviceintermediaries aim to alleviate this problem by offering services that helpusers meet their requirements.  This paper aims to lay the groundwork for developing a cloud portfoliomanagement platform and its business model defined via a business modelcanvas. Furthermore a prototype of a platform is developed offering a cloudportfolio optimization service using two algorithms developed in previousresearch to create suitable and well-utilized allocations for a customersapplications. |


| Item |Content|
| --- |---|
|idx| 2404.10421v1 |
|title| Concurrency Model of BDI Programming Frameworks: Why Should We Control It? |
|authors| Martina BaiardiSamuele BurattiniGiovanni CiattoDanilo PianiniAndrea OmiciniAlessandro Ricci
|links| http://arxiv.org/abs/2404.10421v1 |
|updated| 2024-04-16 09:38:40 UTC |
|summary| We provide a taxonomy of concurrency models for BDI frameworks elicited byanalysing state-of-the-art technologies and aimed at helping both BDIdesigners and developers in making informed decisions. Comparison among BDItechnologies w.r.t. concurrency models reveals heterogeneous support and lowcustomisability. |


| Item |Content|
| --- |---|
|idx| 2404.10397v1 |
|title| On the external concurrency of current BDI frameworks for MAS |
|authors| Martina BaiardiSamuele BurattiniGiovanni CiattoDanilo PianiniAlessandro RicciAndrea Omicini
|links| http://arxiv.org/abs/2404.10397v1 |
|updated| 2024-04-16 08:55:46 UTC |
|summary| The execution of Belief-Desire-Intention BDI agents in a Multi-Agent SystemMAS can be practically implemented on top of low-level concurrency mechanismsthat impact on efficiency determinism and reproducibility. We argue thatdevelopers should specify the MAS behaviour independently of the executionmodel and choose or configure the concurrency model later on according totheir target domains specific needs leaving the MAS specification unaffected.We identify patterns for mapping the agent execution over the underlyingconcurrency abstractions and investigate which concurrency models aresupported by some of the most commonly used BDI platforms. Although mostframeworks support multiple concurrency models we find that they tend to hidethem under the hood making them opaque to the developer and effectivelylimiting the possibility of fine-tuning the MAS. |


