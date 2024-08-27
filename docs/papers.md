# cs.CL 

| Item |Content|
| --- |---|
|idx| 2408.14471v1 |
|title| A Practitioner's Guide to Continual Multimodal Pretraining |
|authors| Karsten RothVishaal UdandaraoSebastian DziadzioAmeya PrabhuMehdi ChertiOriol VinyalsOlivier HénaffSamuel AlbanieMatthias BethgeZeynep Akata
|links| http://arxiv.org/abs/2408.14471v1 |
|updated| 2024-08-26 17:59:01 UTC |
|summary| Multimodal foundation models serve numerous applications at the intersectionof vision and language. Still despite being pretrained on extensive data theybecome outdated over time. To keep models updated research into continualpretraining mainly explores scenarios with either 1 infrequentindiscriminate updates on large-scale new data or 2 frequent sample-levelupdates. However practical model deployment often operates in the gap betweenthese two limit cases as real-world applications often demand adaptation tospecific subdomains tasks or concepts -- spread over the entire varying lifecycle of a model. In this work we complement current perspectives on continualpretraining through a research test bed as well as provide comprehensiveguidance for effective continual model updates in such scenarios. We firstintroduce FoMo-in-Flux a continual multimodal pretraining benchmark withrealistic compute constraints and practical deployment requirementsconstructed over 63 datasets with diverse visual and semantic coverage. UsingFoMo-in-Flux we explore the complex landscape of practical continualpretraining through multiple perspectives: 1 A data-centric investigation ofdata mixtures and stream orderings that emulate real-world deploymentsituations 2 a method-centric investigation ranging from simple fine-tuningand traditional continual learning strategies to parameter-efficient updatesand model merging 3 meta learning rate schedules and mechanistic designchoices and 4 the influence of model and compute scaling. Together ourinsights provide a practitioners guide to continual multimodal pretraining forreal-world deployment. Our benchmark and code is here:https://github.com/ExplainableML/fomo_in_flux. |


| Item |Content|
| --- |---|
|idx| 2408.14470v1 |
|title| Step-by-Step Unmasking for Parameter-Efficient Fine-tuning of Large Language Models |
|authors| Aradhye AgarwalSuhas K RameshAyan SenguptaTanmoy Chakraborty
|links| http://arxiv.org/abs/2408.14470v1 |
|updated| 2024-08-26 17:58:53 UTC |
|summary| Fine-tuning large language models LLMs on downstream tasks requiressubstantial computational resources. A class of parameter-efficient fine-tuningPEFT aims to mitigate these computational challenges by selectivelyfine-tuning only a small fraction of the model parameters. Althoughcomputationally efficient these techniques often fail to match the performanceof fully fine-tuned models primarily due to inherent biases introduced duringparameter selection. Traditional selective PEFT techniques use a fixed set ofparameters based on a predefined budget a process also known as unmaskingfailing to capture parameter importance dynamically and often ending upexceeding the budget. We introduce textID3 a novel selective PEFT methodthat calculates parameter importance continually and dynamically unmasksparameters by balancing exploration and exploitation in parameter selection.Our empirical study on 15 tasks spanning natural language understanding andgenerative tasks demonstrates the effectiveness of our method compared tofixed-masking-based PEFT techniques. We analytically show that textID3reduces the number of gradient updates by a factor of two enhancingcomputational efficiency. textID3 is robust to random initialization ofneurons and therefore can be seamlessly integrated into existing additive andreparametrization-based PEFT modules such as adapters and LoRA for dynamicsparsification. |


| Item |Content|
| --- |---|
|idx| 2408.14467v1 |
|title| Explicit Inductive Inference using Large Language Models |
|authors| Tianyang LiuTianyi LiLiang ChengMark Steedman
|links| http://arxiv.org/abs/2408.14467v1 |
|updated| 2024-08-26 17:58:17 UTC |
|summary| Large Language Models LLMs are reported to hold undesirable attestationbias on inference tasks: when asked to predict if a premise P entails ahypothesis H instead of considering Hs conditional truthfulness entailed byP LLMs tend to use the out-of-context truth label of H as a fragile proxy. Inthis paper we propose a pipeline that exploits this bias to do explicitinductive inference. Our pipeline uses an LLM to transform a premise into a setof attested alternatives and then aggregate answers of the derived newentailment inquiries to support the original inference prediction. On adirectional predicate entailment benchmark we demonstrate that by applyingthis simple pipeline we can improve the overall performance of LLMs oninference and substantially alleviate the impact of their attestation bias. |


| Item |Content|
| --- |---|
|idx| 2408.14438v1 |
|title| Evaluating Large Language Models on Spatial Tasks: A Multi-Task Benchmarking Study |
|authors| Liuchang Xu Shuo ZhaoQingming LinLuyao ChenQianqian LuoSensen WuXinyue YeHailin FengZhenhong Du
|links| http://arxiv.org/abs/2408.14438v1 |
|updated| 2024-08-26 17:25:16 UTC |
|summary| The advent of large language models such as ChatGPT Gemini and others hasunderscored the importance of evaluating their diverse capabilities rangingfrom natural language understanding to code generation. However theirperformance on spatial tasks has not been comprehensively assessed. This studyaddresses this gap by introducing a novel multi-task spatial evaluationdataset designed to systematically explore and compare the performance ofseveral advanced models on spatial tasks. The dataset encompasses twelvedistinct task types including spatial understanding and path planning eachwith verified accurate answers. We evaluated multiple models includingOpenAIs gpt-3.5-turbo gpt-4o and ZhipuAIs glm-4 through a two-phasetesting approach. Initially we conducted zero-shot testing followed bycategorizing the dataset by difficulty and performing prompt tuning tests.Results indicate that gpt-4o achieved the highest overall accuracy in the firstphase with an average of 71.3. Although moonshot-v1-8k slightlyunderperformed overall it surpassed gpt-4o in place name recognition tasks.The study also highlights the impact of prompt strategies on model performancein specific tasks. For example the Chain-of-Thought COT strategy increasedgpt-4os accuracy in path planning from 12.4 to 87.5 while a one-shotstrategy enhanced moonshot-v1-8ks accuracy in mapping tasks from 10.1 to76.3. |


| Item |Content|
| --- |---|
|idx| 2408.14419v1 |
|title| CHARTOM: A Visual Theory-of-Mind Benchmark for Multimodal Large Language Models |
|authors| Shubham BhartiShiyun ChengJihyun RhoMartina RaoXiaojin Zhu
|links| http://arxiv.org/abs/2408.14419v1 |
|updated| 2024-08-26 17:04:23 UTC |
|summary| We introduce CHARTOM a visual theory-of-mind benchmark for multimodal largelanguage models. CHARTOM consists of specially designed data visualizingcharts. Given a chart a language model needs to not only correctly comprehendthe chart the FACT question but also judge if the chart will be misleading toa human reader the MIND question. Both questions have significant societalbenefits. We detail the construction of the CHARTOM benchmark including itscalibration on human performance. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2408.14472v1 |
|title| Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning |
|authors| Xinyang GuYen-Jen WangXiang ZhuChengming ShiYanjiang GuoYichen LiuJianyu Chen
|links| http://arxiv.org/abs/2408.14472v1 |
|updated| 2024-08-26 17:59:03 UTC |
|summary| Humanoid robots with their human-like skeletal structure are especiallysuited for tasks in human-centric environments. However this structure isaccompanied by additional challenges in locomotion controller designespecially in complex real-world environments. As a result existing humanoidrobots are limited to relatively simple terrains either with model-basedcontrol or model-free reinforcement learning. In this work we introduceDenoising World Model Learning DWL an end-to-end reinforcement learningframework for humanoid locomotion control which demonstrates the worlds firsthumanoid robot to master real-world challenging terrains such as snowy andinclined land in the wild up and down stairs and extremely uneven terrains.All scenarios run the same learned neural network with zero-shot sim-to-realtransfer indicating the superior robustness and generalization capability ofthe proposed method. |


| Item |Content|
| --- |---|
|idx| 2408.14468v1 |
|title| K-Sort Arena: Efficient and Reliable Benchmarking for Generative Models via K-wise Human Preferences |
|authors| Zhikai LiXuewen LiuDongrong FuJianquan LiQingyi GuKurt KeutzerZhen Dong
|links| http://arxiv.org/abs/2408.14468v1 |
|updated| 2024-08-26 17:58:20 UTC |
|summary| The rapid advancement of visual generative models necessitates efficient andreliable evaluation methods. Arena platform which gathers user votes on modelcomparisons can rank models with human preferences. However traditional Arenamethods while established require an excessive number of comparisons forranking to converge and are vulnerable to preference noise in votingsuggesting the need for better approaches tailored to contemporary evaluationchallenges. In this paper we introduce K-Sort Arena an efficient and reliableplatform based on a key insight: images and videos possess higher perceptualintuitiveness than texts enabling rapid evaluation of multiple samplessimultaneously. Consequently K-Sort Arena employs K-wise comparisons allowingK models to engage in free-for-all competitions which yield much richerinformation than pairwise comparisons. To enhance the robustness of the systemwe leverage probabilistic modeling and Bayesian updating techniques. We proposean exploration-exploitation-based matchmaking strategy to facilitate moreinformative comparisons. In our experiments K-Sort Arena exhibits 16.3x fasterconvergence compared to the widely used ELO algorithm. To further validate thesuperiority and obtain a comprehensive leaderboard we collect human feedbackvia crowdsourced evaluations of numerous cutting-edge text-to-image andtext-to-video models. Thanks to its high efficiency K-Sort Arena cancontinuously incorporate emerging models and update the leaderboard withminimal votes. Our project has undergone several months of internal testing andis now available at https://huggingface.co/spaces/ksort/K-Sort-Arena |


| Item |Content|
| --- |---|
|idx| 2408.14443v1 |
|title| Temporal Ensemble Logic |
|authors| Guo-Qiang Zhang
|links| http://arxiv.org/abs/2408.14443v1 |
|updated| 2024-08-26 17:36:25 UTC |
|summary| We introduce Temporal Ensemble Logic TEL a monadic first-order modallogic for linear-time temporal reasoning. TEL includes primitive temporalconstructs such as always up to t time later Box_t sometimesbefore t time in the future Diamond_t and t-time latervarphi_t. TEL has been motivated from the requirement for rigor andreproducibility for cohort specification and discovery in clinical andpopulation health research to fill a gap in formalizing temporal reasoning inbiomedicine. In this paper we first introduce TEL in a general set up withdiscrete and dense time as special cases. We then focus on the theoreticaldevelopment of discrete TEL on the temporal domain of positive integersmathbbN denoted as rm TEL_mathbbN. rmTEL_mathbbN is strictly more expressive than the standard monadicsecond order logic characterized by Buchi automata. We present its formalsemantics a proof system and provide a proof for the undecidability of thesatisfiability of rm TEL_mathbbN. We also discuss expressivenessand decidability fragments for rm TEL_mathbbN followed byillustrative applications. |


| Item |Content|
| --- |---|
|idx| 2408.14441v1 |
|title| Attend-Fusion: Efficient Audio-Visual Fusion for Video Classification |
|authors| Mahrukh AwanAsmar NadeemMuhammad Junaid AwanArmin MustafaSyed Sameed Husain
|links| http://arxiv.org/abs/2408.14441v1 |
|updated| 2024-08-26 17:33:47 UTC |
|summary| Exploiting both audio and visual modalities for video classification is achallenging task as the existing methods require large model architecturesleading to high computational complexity and resource requirements. Smallerarchitectures on the other hand struggle to achieve optimal performance. Inthis paper we propose Attend-Fusion an audio-visual AV fusion approach thatintroduces a compact model architecture specifically designed to captureintricate audio-visual relationships in video data. Through extensiveexperiments on the challenging YouTube-8M dataset we demonstrate thatAttend-Fusion achieves an F1 score of 75.64 with only 72M parameters whichis comparable to the performance of larger baseline models such asFully-Connected Late Fusion 75.96 F1 score 341M parameters. Attend-Fusionachieves similar performance to the larger baseline model while reducing themodel size by nearly 80 highlighting its efficiency in terms of modelcomplexity. Our work demonstrates that the Attend-Fusion model effectivelycombines audio and visual information for video classification achievingcompetitive performance with significantly reduced model size. This approachopens new possibilities for deploying high-performance video understandingsystems in resource-constrained environments across various applications. |


| Item |Content|
| --- |---|
|idx| 2408.14437v1 |
|title| Sparsity-Aware Hardware-Software Co-Design of Spiking Neural Networks: An Overview |
|authors| Ilkin AliyevKama SvobodaTosiron AdegbijaJean-Marc Fellous
|links| http://arxiv.org/abs/2408.14437v1 |
|updated| 2024-08-26 17:22:11 UTC |
|summary| Spiking Neural Networks SNNs are inspired by the sparse and event-drivennature of biological neural processing and offer the potential forultra-low-power artificial intelligence. However realizing their efficiencybenefits requires specialized hardware and a co-design approach thateffectively leverages sparsity. We explore the hardware-software co-design ofsparse SNNs examining how sparsity representation hardware architectures andtraining techniques influence hardware efficiency. We analyze the impact ofstatic and dynamic sparsity discuss the implications of different neuronmodels and encoding schemes and investigate the need for adaptability inhardware designs. Our work aims to illuminate the path towards embeddedneuromorphic systems that fully exploit the computational advantages of sparseSNNs. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2408.14471v1 |
|title| A Practitioner's Guide to Continual Multimodal Pretraining |
|authors| Karsten RothVishaal UdandaraoSebastian DziadzioAmeya PrabhuMehdi ChertiOriol VinyalsOlivier HénaffSamuel AlbanieMatthias BethgeZeynep Akata
|links| http://arxiv.org/abs/2408.14471v1 |
|updated| 2024-08-26 17:59:01 UTC |
|summary| Multimodal foundation models serve numerous applications at the intersectionof vision and language. Still despite being pretrained on extensive data theybecome outdated over time. To keep models updated research into continualpretraining mainly explores scenarios with either 1 infrequentindiscriminate updates on large-scale new data or 2 frequent sample-levelupdates. However practical model deployment often operates in the gap betweenthese two limit cases as real-world applications often demand adaptation tospecific subdomains tasks or concepts -- spread over the entire varying lifecycle of a model. In this work we complement current perspectives on continualpretraining through a research test bed as well as provide comprehensiveguidance for effective continual model updates in such scenarios. We firstintroduce FoMo-in-Flux a continual multimodal pretraining benchmark withrealistic compute constraints and practical deployment requirementsconstructed over 63 datasets with diverse visual and semantic coverage. UsingFoMo-in-Flux we explore the complex landscape of practical continualpretraining through multiple perspectives: 1 A data-centric investigation ofdata mixtures and stream orderings that emulate real-world deploymentsituations 2 a method-centric investigation ranging from simple fine-tuningand traditional continual learning strategies to parameter-efficient updatesand model merging 3 meta learning rate schedules and mechanistic designchoices and 4 the influence of model and compute scaling. Together ourinsights provide a practitioners guide to continual multimodal pretraining forreal-world deployment. Our benchmark and code is here:https://github.com/ExplainableML/fomo_in_flux. |


| Item |Content|
| --- |---|
|idx| 2408.14461v1 |
|title| A domain decomposition-based autoregressive deep learning model for unsteady and nonlinear partial differential equations |
|authors| Sheel NidhanHaoliang JiangLalit GhuleClancy UmphreyRishikesh RanadeJay Pathak
|links| http://arxiv.org/abs/2408.14461v1 |
|updated| 2024-08-26 17:50:47 UTC |
|summary| In this paper we propose a domain-decomposition-based deep learning DLframework named transient-CoMLSim for accurately modeling unsteady andnonlinear partial differential equations PDEs. The framework consists of twokey components: a a convolutional neural network CNN-based autoencoderarchitecture and b an autoregressive model composed of fully connectedlayers. Unlike existing state-of-the-art methods that operate on the entirecomputational domain our CNN-based autoencoder computes a lower-dimensionalbasis for solution and condition fields represented on subdomains. Timesteppingis performed entirely in the latent space generating embeddings of thesolution variables from the time history of embeddings of solution andcondition variables. This approach not only reduces computational complexitybut also enhances scalability making it well-suited for large-scalesimulations. Furthermore to improve the stability of our rollouts we employ acurriculum learning CL approach during the training of the autoregressivemodel. The domain-decomposition strategy enables scaling to out-of-distributiondomain sizes while maintaining the accuracy of predictions -- a feature noteasily integrated into popular DL-based approaches for physics simulations. Webenchmark our model against two widely-used DL architectures Fourier NeuralOperator FNO and U-Net and demonstrate that our framework outperforms themin terms of accuracy extrapolation to unseen timesteps and stability for awide range of use cases. |


| Item |Content|
| --- |---|
|idx| 2408.14453v1 |
|title| Reconstructing physiological signals from fMRI across the adult lifespan |
|authors| Shiyu WangZiyuan XuYamin LiMara MatherRoza G. BayrakCatie Chang
|links| http://arxiv.org/abs/2408.14453v1 |
|updated| 2024-08-26 17:48:42 UTC |
|summary| Interactions between the brain and body are of fundamental importance forhuman behavior and health. Functional magnetic resonance imaging fMRIcaptures whole-brain activity noninvasively and modeling how fMRI signalsinteract with physiological dynamics of the body can provide new insight intobrain function and offer potential biomarkers of disease. Howeverphysiological recordings are not always possible to acquire since they requireextra equipment and setup and even when they are the recorded physiologicalsignals may contain substantial artifacts. To overcome this limitation machinelearning models have been proposed to directly extract features of respiratoryand cardiac activity from resting-state fMRI signals. To date such work hasbeen carried out only in healthy young adults and in a pediatric populationleaving open questions about the efficacy of these approaches on older adults.Here we propose a novel framework that leverages Transformer-basedarchitectures for reconstructing two key physiological signals - low-frequencyrespiratory volume RV and heart rate HR fluctuations - from fMRI data andtest these models on a dataset of individuals aged 36-89 years old. Ourframework outperforms previously proposed approaches attaining mediancorrelations between predicted and measured signals of r  .698 for RV and r .618 for HR indicating the potential of leveraging attention mechanisms tomodel fMRI-physiological signal relationships. We also evaluate several modeltraining and fine-tuning strategies and find that incorporating young-adultdata during training improves the performance when predicting physiologicalsignals in the aging cohort. Overall our approach successfully infers keyphysiological variables directly from fMRI data from individuals across a widerange of the adult lifespan. |


| Item |Content|
| --- |---|
|idx| 2408.14445v1 |
|title| Symmetry & Critical Points |
|authors| Yossi Arjevani
|links| http://arxiv.org/abs/2408.14445v1 |
|updated| 2024-08-26 17:36:51 UTC |
|summary| Critical points of an invariant function may or may not be symmetric. Weprove however that if a symmetric critical point exists those adjacent to itare generically symmetry breaking. This mathematical mechanism is shown tocarry important implications for our ability to efficiently minimize invariantnonconvex functions in particular those associated with neural networks. |


| Item |Content|
| --- |---|
|idx| 2408.14442v1 |
|title| Model Parallel Training and Transfer Learning for Convolutional Neural Networks by Domain Decomposition |
|authors| Axel KlawonnMartin LanserJanine Weber
|links| http://arxiv.org/abs/2408.14442v1 |
|updated| 2024-08-26 17:35:01 UTC |
|summary| Deep convolutional neural networks CNNs have been shown to be verysuccessful in a wide range of image processing applications. However due totheir increasing number of model parameters and an increasing availability oflarge amounts of training data parallelization strategies to efficiently traincomplex CNNs are necessary. In previous work by the authors a novel modelparallel CNN architecture was proposed which is loosely inspired by domaindecomposition. In particular the novel network architecture is based on adecomposition of the input data into smaller subimages. For each of thesesubimages local CNNs with a proportionally smaller number of parameters aretrained in parallel and the resulting local classifications are then aggregatedin a second step by a dense feedforward neural network DNN. In the presentwork we compare the resulting CNN-DNN architecture to less costly alternativesto combine the local classifications into a final global decision.Additionally we investigate the performance of the CNN-DNN trained as onecoherent model as well as using a transfer learning strategy where theparameters of the pre-trained local CNNs are used as initial values for asubsequently trained global coherent CNN-DNN model. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2408.14471v1 |
|title| A Practitioner's Guide to Continual Multimodal Pretraining |
|authors| Karsten RothVishaal UdandaraoSebastian DziadzioAmeya PrabhuMehdi ChertiOriol VinyalsOlivier HénaffSamuel AlbanieMatthias BethgeZeynep Akata
|links| http://arxiv.org/abs/2408.14471v1 |
|updated| 2024-08-26 17:59:01 UTC |
|summary| Multimodal foundation models serve numerous applications at the intersectionof vision and language. Still despite being pretrained on extensive data theybecome outdated over time. To keep models updated research into continualpretraining mainly explores scenarios with either 1 infrequentindiscriminate updates on large-scale new data or 2 frequent sample-levelupdates. However practical model deployment often operates in the gap betweenthese two limit cases as real-world applications often demand adaptation tospecific subdomains tasks or concepts -- spread over the entire varying lifecycle of a model. In this work we complement current perspectives on continualpretraining through a research test bed as well as provide comprehensiveguidance for effective continual model updates in such scenarios. We firstintroduce FoMo-in-Flux a continual multimodal pretraining benchmark withrealistic compute constraints and practical deployment requirementsconstructed over 63 datasets with diverse visual and semantic coverage. UsingFoMo-in-Flux we explore the complex landscape of practical continualpretraining through multiple perspectives: 1 A data-centric investigation ofdata mixtures and stream orderings that emulate real-world deploymentsituations 2 a method-centric investigation ranging from simple fine-tuningand traditional continual learning strategies to parameter-efficient updatesand model merging 3 meta learning rate schedules and mechanistic designchoices and 4 the influence of model and compute scaling. Together ourinsights provide a practitioners guide to continual multimodal pretraining forreal-world deployment. Our benchmark and code is here:https://github.com/ExplainableML/fomo_in_flux. |


| Item |Content|
| --- |---|
|idx| 2408.14469v1 |
|title| Grounded Multi-Hop VideoQA in Long-Form Egocentric Videos |
|authors| Qirui ChenShangzhe DiWeidi Xie
|links| http://arxiv.org/abs/2408.14469v1 |
|updated| 2024-08-26 17:58:47 UTC |
|summary| This paper considers the problem of Multi-Hop Video Question AnsweringMH-VidQA in long-form egocentric videos. This task not only requires toanswer visual questions but also to localize multiple relevant time intervalswithin the video as visual evidences. We develop an automated pipeline tocreate multi-hop question-answering pairs with associated temporal evidenceenabling to construct a large-scale dataset for instruction-tuning. To monitorthe progress of this new task we further curate a high-quality benchmarkMultiHop-EgoQA with careful manual verification and refinement. Experimentalresults reveal that existing multi-modal systems exhibit inadequate multi-hopgrounding and reasoning abilities resulting in unsatisfactory performance. Wethen propose a novel architecture termed as Grounding Scattered Evidence withLarge Language Model GeLM that enhances multi-modal large language modelsMLLMs by incorporating a grounding module to retrieve temporal evidence fromvideos using flexible grounding tokens. Trained on our visual instruction dataGeLM demonstrates improved multi-hop grounding and reasoning capabilitiessetting a new baseline for this challenging task. Furthermore when trained onthird-person view videos the same architecture also achieves state-of-the-artperformance on the single-hop VidQA benchmark ActivityNet-RTL demonstratingits effectiveness. |


| Item |Content|
| --- |---|
|idx| 2408.14457v1 |
|title| Dense Center-Direction Regression for Object Counting and Localization with Point Supervision |
|authors| Domen TabernikJon MuhovičDanijel Skočaj
|links| http://dx.doi.org/10.1016/j.patcog.2024.110540 |
|updated| 2024-08-26 17:49:27 UTC |
|summary| Object counting and localization problems are commonly addressed with pointsupervised learning which allows the use of less labor-intensive pointannotations. However learning based on point annotations poses challenges dueto the high imbalance between the sets of annotated and unannotated pixelswhich is often treated with Gaussian smoothing of point annotations and focalloss. However these approaches still focus on the pixels in the immediatevicinity of the point annotations and exploit the rest of the data onlyindirectly. In this work we propose a novel approach termed CeDiRNet forpoint-supervised learning that uses a dense regression of directions pointingtowards the nearest object centers i.e. center-directions. This providesgreater support for each center point arising from many surrounding pixelspointing towards the object center. We propose a formulation ofcenter-directions that allows the problem to be split into the domain-specificdense regression of center-directions and the final localization task based ona small lightweight and domain-agnostic localization network that can betrained with synthetic data completely independent of the target domain. Wedemonstrate the performance of the proposed method on six different datasetsfor object counting and localization and show that it outperforms the existingstate-of-the-art methods. The code is accessible on GitHub athttps://github.com/vicoslab/CeDiRNet.git. |


| Item |Content|
| --- |---|
|idx| 2408.14456v1 |
|title| Center Direction Network for Grasping Point Localization on Cloths |
|authors| Domen TabernikJon MuhovičMatej UrbasDanijel Skočaj
|links| http://arxiv.org/abs/2408.14456v1 |
|updated| 2024-08-26 17:49:05 UTC |
|summary| Object grasping is a fundamental challenge in robotics and computer visioncritical for advancing robotic manipulation capabilities. Deformable objectslike fabrics and cloths pose additional challenges due to their non-rigidnature. In this work we introduce CeDiRNet-3DoF a deep-learning model forgrasp point detection with a particular focus on cloth objects. CeDiRNet-3DoFemploys center direction regression alongside a localization network attainingfirst place in the perception task of ICRA 2023s Cloth Manipulation Challenge.Recognizing the lack of standardized benchmarks in the literature that hindereffective method comparison we present the ViCoS Towel Dataset. This extensivebenchmark dataset comprises 8000 real and 12000 synthetic images serving asa robust resource for training and evaluating contemporary data-drivendeep-learning approaches. Extensive evaluation revealed CeDiRNet-3DoFsrobustness in real-world performance outperforming state-of-the-art methodsincluding the latest transformer-based models. Our work bridges a crucial gapoffering a robust solution and benchmark for cloth grasping in computer visionand robotics. Code and dataset are available at:https://github.com/vicoslab/CeDiRNet-3DoF |


| Item |Content|
| --- |---|
|idx| 2408.14442v1 |
|title| Model Parallel Training and Transfer Learning for Convolutional Neural Networks by Domain Decomposition |
|authors| Axel KlawonnMartin LanserJanine Weber
|links| http://arxiv.org/abs/2408.14442v1 |
|updated| 2024-08-26 17:35:01 UTC |
|summary| Deep convolutional neural networks CNNs have been shown to be verysuccessful in a wide range of image processing applications. However due totheir increasing number of model parameters and an increasing availability oflarge amounts of training data parallelization strategies to efficiently traincomplex CNNs are necessary. In previous work by the authors a novel modelparallel CNN architecture was proposed which is loosely inspired by domaindecomposition. In particular the novel network architecture is based on adecomposition of the input data into smaller subimages. For each of thesesubimages local CNNs with a proportionally smaller number of parameters aretrained in parallel and the resulting local classifications are then aggregatedin a second step by a dense feedforward neural network DNN. In the presentwork we compare the resulting CNN-DNN architecture to less costly alternativesto combine the local classifications into a final global decision.Additionally we investigate the performance of the CNN-DNN trained as onecoherent model as well as using a transfer learning strategy where theparameters of the pre-trained local CNNs are used as initial values for asubsequently trained global coherent CNN-DNN model. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2408.14445v1 |
|title| Symmetry & Critical Points |
|authors| Yossi Arjevani
|links| http://arxiv.org/abs/2408.14445v1 |
|updated| 2024-08-26 17:36:51 UTC |
|summary| Critical points of an invariant function may or may not be symmetric. Weprove however that if a symmetric critical point exists those adjacent to itare generically symmetry breaking. This mathematical mechanism is shown tocarry important implications for our ability to efficiently minimize invariantnonconvex functions in particular those associated with neural networks. |


| Item |Content|
| --- |---|
|idx| 2408.14402v1 |
|title| A quasi-Bayesian sequential approach to deconvolution density estimation |
|authors| Stefano FavaroSandra Fortini
|links| http://arxiv.org/abs/2408.14402v1 |
|updated| 2024-08-26 16:40:04 UTC |
|summary| Density deconvolution addresses the estimation of the unknown probabilitydensity function f of a random signal from data that are observed with anindependent additive random noise. This is a classical problem in statisticsfor which frequentist and Bayesian nonparametric approaches are available todeal with static or batch data. In this paper we consider the problem ofdensity deconvolution in a streaming or online setting where noisy data arriveprogressively with no predetermined sample size and we develop a sequentialnonparametric approach to estimate f. By relying on a quasi-Bayesiansequential approach often referred to as Newtons algorithm we obtainestimates of f that are of easy evaluation computationally efficient andwith a computational cost that remains constant as the amount of dataincreases which is critical in the streaming setting. Large sample asymptoticproperties of the proposed estimates are studied yielding provable guaranteeswith respect to the estimation of f at a point local and on an intervaluniform. In particular we establish local and uniform central limittheorems providing corresponding asymptotic credible intervals and bands. Wevalidate empirically our methods on synthetic and real data by considering thecommon setting of Laplace and Gaussian noise distributions and make acomparison with respect to the kernel-based approach and a Bayesiannonparametric approach with a Dirichlet process mixture prior. |


| Item |Content|
| --- |---|
|idx| 2408.14332v1 |
|title| One-layer transformers fail to solve the induction heads task |
|authors| Clayton SanfordDaniel HsuMatus Telgarsky
|links| http://arxiv.org/abs/2408.14332v1 |
|updated| 2024-08-26 15:01:04 UTC |
|summary| A simple communication complexity argument proves that no one-layertransformer can solve the induction heads task unless its size is exponentiallylarger than the size sufficient for a two-layer transformer. |


| Item |Content|
| --- |---|
|idx| 2408.14325v1 |
|title| Function-Space MCMC for Bayesian Wide Neural Networks |
|authors| Lucia PezzettiStefano FavaroStefano Pelucchetti
|links| http://arxiv.org/abs/2408.14325v1 |
|updated| 2024-08-26 14:54:13 UTC |
|summary| Bayesian Neural Networks represent a fascinating confluence of deep learningand probabilistic reasoning offering a compelling framework for understandinguncertainty in complex predictive models. In this paper we investigate the useof the preconditioned Crank-Nicolson algorithm and its Langevin version tosample from the reparametrised posterior distribution of the weights as thewidths of Bayesian Neural Networks grow larger. In addition to being robust inthe infinite-dimensional setting we prove that the acceptance probabilities ofthe proposed methods approach 1 as the width of the network increasesindependently of any stepsize tuning. Moreover we examine and compare how themixing speeds of the underdamped Langevin Monte Carlo the preconditionedCrank-Nicolson and the preconditioned Crank-Nicolson Langevin samplers areinfluenced by changes in the network width in some real-world cases. Ourfindings suggest that in wide Bayesian Neural Networks configurations thepreconditioned Crank-Nicolson method allows for more efficient sampling of thereparametrised posterior distribution as evidenced by a higher effectivesample size and improved diagnostic results compared with the other analysedalgorithms. |


| Item |Content|
| --- |---|
|idx| 2408.14266v1 |
|title| HyperSBINN: A Hypernetwork-Enhanced Systems Biology-Informed Neural Network for Efficient Drug Cardiosafety Assessment |
|authors| Inass SoukariehGerhard HesslerHervé MinouxMarcel MohrFriedemann SchmidtJan WenzelPierre BarbillonHugo GangloffPierre Gloaguen
|links| http://arxiv.org/abs/2408.14266v1 |
|updated| 2024-08-26 13:40:33 UTC |
|summary| Mathematical modeling in systems toxicology enables a comprehensiveunderstanding of the effects of pharmaceutical substances on cardiac health.However the complexity of these models limits their widespread application inearly drug discovery. In this paper we introduce a novel approach to solvingparameterized models of cardiac action potentials by combining meta-learningtechniques with Systems Biology-Informed Neural Networks SBINNs. The proposedmethod HyperSBINN effectively addresses the challenge of predicting theeffects of various compounds at different concentrations on cardiac actionpotentials outperforming traditional differential equation solvers in speed.Our model efficiently handles scenarios with limited data and complexparameterized differential equations. The HyperSBINN model demonstrates robustperformance in predicting APD90 values indicating its potential as a reliabletool for modeling cardiac electrophysiology and aiding in preclinical drugdevelopment. This framework represents an advancement in computationalmodeling offering a scalable and efficient solution for simulating andunderstanding complex biological systems. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2408.14322v1 |
|title| Investigating Persuasive Socially Assistive Robot Behavior Strategies for Sustained Engagement in Long-Term Care |
|authors| Cristina GetsonGoldie Nejat
|links| http://arxiv.org/abs/2408.14322v1 |
|updated| 2024-08-26 14:51:49 UTC |
|summary| Socially assistive robots are increasingly being used to support the socialcognitive and physical well-being of those who provide care healthcareprofessionals and those in need of care older adults. However theeffectiveness of persuasive socially assistive robot behaviors and their impacton the sustained motivation of older adults is still not well understood. Thisextended abstract describes our prior human-robot interaction study oninvestigating the effectiveness of persuasive social robot behaviors with careproviders followed by our current research assessing the impact of thesepersuasive robot behaviors on the well-being of older adults in long-term care.The findings provide insights into engagement and sustained motivation of olderadults when providing assistance. |


| Item |Content|
| --- |---|
|idx| 2408.14305v1 |
|title| Collaborative XRTactics: A Formative Study on Tactical Communication in Outdoor Team Sports |
|authors| Ut GongQihan ZhangZiqing YinStefanie Zollmann
|links| http://arxiv.org/abs/2408.14305v1 |
|updated| 2024-08-26 14:37:16 UTC |
|summary| In team sports effective tactical communication is crucial for successparticularly in the fast-paced and complex environment of outdoor athletics.This paper investigates the challenges faced in transmitting strategic plans toplayers and explores potential solutions using eXtended Reality XRtechnologies. We conducted a formative study involving interviews with 4Division I professional soccer coaches 4 professional players 2 college clubcoaches and 2 college club players as well as a survey among 17 Division Iplayers. The study identified key requirements for tactical communicationtools including the need for rapid communication minimal disruption to gameflow reduced cognitive load clear visualization for all players and enhancedauditory clarity. Based on these insights we propose a potential solution - aMobile Augmented Reality AR system designed to address these challenges byproviding real-time intuitive tactical visualization and communication. Thesystem aims to improve strategic planning and execution thereby enhancing teamperformance and cohesion. This work represents a significant step towardsintegrating XR technologies into sports coaching offering a modern andpractical solution for real-time tactical communication. |


| Item |Content|
| --- |---|
|idx| 2408.14159v1 |
|title| "Hi. I'm Molly, Your Virtual Interviewer!" -- Exploring the Impact of Race and Gender in AI-powered Virtual Interview Experiences |
|authors| Shreyan BiswasJi-Youn JungAbhishek UnnamKuldeep YadavShreyansh GuptaUjwal Gadiraju
|links| http://arxiv.org/abs/2408.14159v1 |
|updated| 2024-08-26 10:14:07 UTC |
|summary| The persistent issue of human bias in recruitment processes poses aformidable challenge to achieving equitable hiring practices particularly wheninfluenced by demographic characteristics such as gender and race of bothinterviewers and candidates. Asynchronous Video Interviews AVIs powered byArtificial Intelligence AI have emerged as innovative tools aimed atstreamlining the application screening process while potentially mitigating theimpact of such biases. These AI-driven platforms present an opportunity tocustomize the demographic features of virtual interviewers to align withdiverse applicant preferences promising a more objective and fair evaluation.Despite their growing adoption the implications of virtual intervieweridentities on candidate experiences within AVIs remain underexplored. We aim toaddress this research and empirical gap in this paper. To this end we carriedout a comprehensive between-subjects study involving 218 participants acrosssix distinct experimental conditions manipulating the gender and skin color ofan AI virtual interviewer agent. Our empirical analysis revealed that while thedemographic attributes of the agents did not significantly influence theoverall experience of interviewees variations in the intervieweesdemographics significantly altered their perception of the AVI process.Further we uncovered that the mediating roles of Social Presence andPerception of the virtual interviewer critically affect intervieweesperceptions of fairness  privacy - and impression management . |


| Item |Content|
| --- |---|
|idx| 2408.14088v1 |
|title| Perceived Usability of Collaborative Modeling Tools |
|authors| Ranci RenJohn W. CastroSantiago R. AcuñaOscar DiesteSilvia T. Acuña
|links| http://dx.doi.org/10.1016/j.jss.2023.111807 |
|updated| 2024-08-26 08:19:23 UTC |
|summary| Context: Online collaborative creation of models is becoming commonplace.Collaborative modeling using chatbots and natural language may lower thebarriers to modeling for users from different domains. Objective: We comparethe perceived usability of two similarly online collaborative modeling toolsthe SOCIO chatbot and the Creately web-based tool. Method: We conducted acrossover experiment with 66 participants. The evaluation instrument was basedon the System Usability Scale SUS. We performed a quantitative andqualitative exploration employing inferential statistics and thematicanalysis. Results: The results indicate that chatbots enabling natural languagecommunication enhance communication and collaboration efficiency and improvethe user experience. Conclusion: Chatbots need to improve guidance and help fornovices but they appear beneficial for enhancing user experience. |


| Item |Content|
| --- |---|
|idx| 2408.13977v1 |
|title| Say Your Reason: Extract Contextual Rules In Situ for Context-aware Service Recommendation |
|authors| Yuxuan LiJiahui LiLihang PanChun YuYuanchun Shi
|links| http://arxiv.org/abs/2408.13977v1 |
|updated| 2024-08-26 01:50:29 UTC |
|summary| This paper introduces SayRea an interactive system that facilitates theextraction of contextual rules for personalized context-aware servicerecommendations in mobile scenarios. The system monitors a users execution ofregistered services on their smartphones via accessibility service andproactively requests a single-sentence reason from the user. By utilizing aLarge Language Model LLM SayRea parses the reason and predicts contextualrelationships between the observed service and potential contexts such assetting the alarm clock deep in the evening. In this way SayRea cansignificantly reduce the cognitive load on users in anticipating future needsand selecting contextual attributes. A 10-day field study involving 20participants showed that SayRea accumulated an average of 62.4 rules per userand successfully recommended 45 of service usage. The participants providedpositive feedback on the systems usability interpretability andcontrollability. The findings highlight SayReas effectiveness in personalizedservice recommendations and its potential to enhance user experience in mobilescenarios. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2408.14199v1 |
|title| A Survey on Small-Scale Testbeds for Connected and Automated Vehicles and Robot Swarms |
|authors| Armin MokhtarianJianye XuPatrick ScheffeMaximilian KloockSimon SchäferHeeseung BangViet-Anh LeSangeet UlhasJohannes BetzSean WilsonSpring BermanLiam PaullAmanda ProrokBassam Alrifaee
|links| http://dx.doi.org/10.13140/RG.2.2.16176.74248/1 |
|updated| 2024-08-26 11:54:27 UTC |
|summary| Connected and automated vehicles and robot swarms hold transformativepotential for enhancing safety efficiency and sustainability in thetransportation and manufacturing sectors. Extensive testing and validation ofthese technologies is crucial for their deployment in the real world. Whilesimulations are essential for initial testing they often have limitations incapturing the complex dynamics of real-world interactions. This limitationunderscores the importance of small-scale testbeds. These testbeds provide arealistic cost-effective and controlled environment for testing andvalidating algorithms acting as an essential intermediary between simulationand full-scale experiments. This work serves to facilitate researchers effortsin identifying existing small-scale testbeds suitable for their experiments andprovide insights for those who want to build their own. In addition itdelivers a comprehensive survey of the current landscape of these testbeds. Wederive 62 characteristics of testbeds based on the well-known sense-plan-actparadigm and offer an online table comparing 22 small-scale testbeds based onthese characteristics. The online table is hosted on our designated publicwebpage www.cpm-remote.de/testbeds and we invite testbed creators anddevelopers to contribute to it. We closely examine nine testbeds in this paperdemonstrating how the derived characteristics can be used to present testbeds.Furthermore we discuss three ongoing challenges concerning small-scaletestbeds that we identified i.e. small-scale to full-scale transitionsustainability and power and resource management. |


| Item |Content|
| --- |---|
|idx| 2408.13828v1 |
|title| Decentralized Stochastic Control in Standard Borel Spaces: Centralized MDP Reductions, Near Optimality of Finite Window Local Information, and Q-Learning |
|authors| Omar Mrani-ZentarSerdar Yüksel
|links| http://arxiv.org/abs/2408.13828v1 |
|updated| 2024-08-25 13:07:34 UTC |
|summary| Decentralized stochastic control problems are intrinsically difficult tostudy because of the inapplicability of standard tools from centralized controlsuch as dynamic programming and the resulting computational complexity. In thispaper we address some of these challenges for decentralized stochastic controlwith Borel spaces under three different but tightly related informationstructures under a unified theme: the one-step delayed information sharingpattern the K-step periodic information sharing pattern and the completelydecentralized information structure where no sharing of information occurs. Wewill show that the one-step delayed and K-step periodic problems can be reducedto a centralized MDP generalizing prior results which considered finitelinear or static models by addressing several measurability questions. Theseparated nature of policies under both information structures is thenestablished. We then provide sufficient conditions for the transition kernelsof both centralized reductions to be weak-Feller which facilitates rigorousapproximation and learning theoretic results. We will then show that for thecompletely decentralized control problem finite memory local policies are nearoptimal under a joint conditional mixing condition. This is achieved byobtaining a bound for finite memory policies which goes to zero as memory sizeincreases. We will also provide a performance bound for the K-periodic problemwhich results from replacing the full common information by a finite slidingwindow of information. The latter will depend on the condition of predictorstability in expected total variation which we will establish. We finally showthat under the periodic information sharing pattern a quantized Q-learningalgorithm converges asymptotically towards a near optimal solution. Each of theabove to our knowledge is a new contribution to the literature. |


| Item |Content|
| --- |---|
|idx| 2408.13750v1 |
|title| Multi-Agent Target Assignment and Path Finding for Intelligent Warehouse: A Cooperative Multi-Agent Deep Reinforcement Learning Perspective |
|authors| Qi LiuJianqi GaoDongjie ZhuXizheng PangPengbin ChenJingxiang GuoYanjie Li
|links| http://arxiv.org/abs/2408.13750v1 |
|updated| 2024-08-25 07:32:58 UTC |
|summary| Multi-agent target assignment and path planning TAPF are two key problemsin intelligent warehouse. However most literature only addresses one of thesetwo problems separately. In this study we propose a method to simultaneouslysolve target assignment and path planning from a perspective of cooperativemulti-agent deep reinforcement learning RL. To the best of our knowledgethis is the first work to model the TAPF problem for intelligent warehouse tocooperative multi-agent deep RL and the first to simultaneously address TAPFbased on multi-agent deep RL. Furthermore previous literature rarely considersthe physical dynamics of agents. In this study the physical dynamics of theagents is considered. Experimental results show that our method performs wellin various task settings which means that the target assignment is solvedreasonably well and the planned path is almost shortest. Moreover our methodis more time-efficient than baselines. |


| Item |Content|
| --- |---|
|idx| 2408.13630v1 |
|title| DeepVoting: Learning Voting Rules with Tailored Embeddings |
|authors| Leonardo MatoneBen AbramowitzNicholas MatteiAvinash Balakrishnan
|links| http://arxiv.org/abs/2408.13630v1 |
|updated| 2024-08-24 17:15:20 UTC |
|summary| Aggregating the preferences of multiple agents into a collective decision isa common step in many important problems across areas of computer scienceincluding information retrieval reinforcement learning and recommendersystems. As Social Choice Theory has shown the problem of designing algorithmsfor aggregation rules with specific properties axioms can be difficult orprovably impossible in some cases. Instead of designing algorithms by hand onecan learn aggregation rules particularly voting rules from data. However theprior work in this area has required extremely large models or been limited bythe choice of preference representation i.e. embedding. We recast the problemof designing a good voting rule into one of learning probabilistic versions ofvoting rules that output distributions over a set of candidates. Specificallywe use neural networks to learn probabilistic social choice functions from theliterature. We show that embeddings of preference profiles derived from thesocial choice literature allows us to learn existing voting rules moreefficiently and scale to larger populations of voters more easily than otherwork if the embedding is tailored to the learning objective. Moreover we showthat rules learned using embeddings can be tweaked to create novel voting ruleswith improved axiomatic properties. Namely we show that existing voting rulesrequire only minor modification to combat a probabilistic version of the NoShow Paradox. |


| Item |Content|
| --- |---|
|idx| 2408.13615v1 |
|title| Reaching New Heights in Multi-Agent Collective Construction |
|authors| Martin RamešPavel Surynek
|links| http://arxiv.org/abs/2408.13615v1 |
|updated| 2024-08-24 16:04:58 UTC |
|summary| We propose a new approach for multi-agent collective construction based onthe idea of reversible ramps. Our ReRamp algorithm utilizes reversibleside-ramps to generate construction plans for ramped block structures higherand larger than was previously possible using state-of-the-art planningalgorithms given the same building area. We compare the ReRamp algorithm tosimilar state-of-the-art algorithms on a set of benchmark instances where wedemonstrate its superior computational speed. We also establish in ourexperiments that the ReRamp algorithm is capable of generating plans for asingle-story house an important milestone on the road to real-worldmulti-agent construction applications. |


