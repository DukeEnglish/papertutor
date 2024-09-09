# cs.CL 

| Item |Content|
| --- |---|
|idx| 2409.04421v1 |
|title| RLPF: Reinforcement Learning from Prediction Feedback for User Summarization with LLMs |
|authors| Jiaxing WuLin NingLuyang LiuHarrison LeeNeo WuChao WangSushant PrakashShawn O'BanionBradley GreenJun Xie
|links| http://arxiv.org/abs/2409.04421v1 |
|updated| 2024-09-06 17:30:45 UTC |
|summary| LLM-powered personalization agent systems employ Large Language Models LLMsto predict users behavior from their past activities. However theireffectiveness often hinges on the ability to effectively leverage extensivelong user historical data due to its inherent noise and length of such data.Existing pretrained LLMs may generate summaries that are concise but lack thenecessary context for downstream tasks hindering their utility inpersonalization systems. To address these challenges we introduceReinforcement Learning from Prediction Feedback RLPF. RLPF fine-tunes LLMs togenerate concise human-readable user summaries that are optimized fordownstream task performance. By maximizing the usefulness of the generatedsummaries RLPF effectively distills extensive user history data whilepreserving essential information for downstream tasks. Our empirical evaluationdemonstrates significant improvements in both extrinsic downstream task utilityand intrinsic summary quality surpassing baseline methods by up to 22 ondownstream task performance and achieving an up to 84.59 win rate onFactuality Abstractiveness and Readability. RLPF also achieves a remarkable74 reduction in context length while improving performance on 16 out of 19unseen tasks and/or datasets showcasing its generalizability. This approachoffers a promising solution for enhancing LLM personalization by effectivelytransforming long noisy user histories into informative and human-readablerepresentations. |


| Item |Content|
| --- |---|
|idx| 2409.04384v1 |
|title| Empirical Bayesian image restoration by Langevin sampling with a denoising diffusion implicit prior |
|authors| Charlesquin Kemajou MbakamJean-Francois GiovannelliMarcelo Pereyra
|links| http://arxiv.org/abs/2409.04384v1 |
|updated| 2024-09-06 16:20:24 UTC |
|summary| Score-based diffusion methods provide a powerful strategy to solve imagerestoration tasks by flexibly combining a pre-trained foundational prior modelwith a likelihood function specified during test time. Such methods arepredominantly derived from two stochastic processes: reversingOrnstein-Uhlenbeck which underpins the celebrated denoising diffusionprobabilistic models DDPM and denoising diffusion implicit models DDIM andthe Langevin diffusion process. The solutions delivered by DDPM and DDIM areoften remarkably realistic but they are not always consistent withmeasurements because of likelihood intractability issues and the associatedrequired approximations. Alternatively using a Langevin process circumventsthe intractable likelihood issue but usually leads to restoration results ofinferior quality and longer computing times. This paper presents a novel andhighly computationally efficient image restoration method that carefully embedsa foundational DDPM denoiser within an empirical Bayesian Langevin algorithmwhich jointly calibrates key model hyper-parameters as it estimates the modelsposterior mean. Extensive experimental results on three canonical tasks imagedeblurring super-resolution and inpainting demonstrate that the proposedapproach improves on state-of-the-art strategies both in image estimationaccuracy and computing time. |


| Item |Content|
| --- |---|
|idx| 2409.04340v1 |
|title| AGR: Age Group fairness Reward for Bias Mitigation in LLMs |
|authors| Shuirong CaoRuoxi ChengZhiqiang Wang
|links| http://arxiv.org/abs/2409.04340v1 |
|updated| 2024-09-06 15:18:12 UTC |
|summary| LLMs can exhibit age biases resulting in unequal treatment of individualsacross age groups. While much research has addressed racial and gender biasesage bias remains little explored. The scarcity of instruction-tuning andpreference datasets for age bias hampers its detection and measurement andexisting fine-tuning methods seldom address age-related fairness. In thispaper we construct age bias preference datasets and instruction-tuningdatasets for RLHF. We introduce ARG an age fairness reward to reducedifferences in the response quality of LLMs across different age groups.Extensive experiments demonstrate that this reward significantly improvesresponse accuracy and reduces performance disparities across age groups. Oursource code and datasets are available at the anonymoushrefhttps://anonymous.4open.science/r/FairRLHF-D445/readme.mdlink. |


| Item |Content|
| --- |---|
|idx| 2409.04318v1 |
|title| Learning vs Retrieval: The Role of In-Context Examples in Regression with LLMs |
|authors| Aliakbar NafarKristen Brent VenableParisa Kordjamshidi
|links| http://arxiv.org/abs/2409.04318v1 |
|updated| 2024-09-06 14:46:37 UTC |
|summary| Generative Large Language Models LLMs are capable of being in-contextlearners. However the underlying mechanism of in-context learning ICL isstill a major research question and experimental research results about howmodels exploit ICL are not always consistent. In this work we propose aframework for evaluating in-context learning mechanisms which we claim are acombination of retrieving internal knowledge and learning from in-contextexamples by focusing on regression tasks. First we show that LLMs can performregression on real-world datasets and then design experiments to measure theextent to which the LLM retrieves its internal knowledge versus learning fromin-context examples. We argue that this process lies on a spectrum betweenthese two extremes. We provide an in-depth analysis of the degrees to whichthese mechanisms are triggered depending on various factors such as priorknowledge about the tasks and the type and richness of the information providedby the in-context examples. We employ three LLMs and utilize multiple datasetsto corroborate the robustness of our findings. Our results shed light on how toengineer prompts to leverage meta-learning from in-context examples and fosterknowledge retrieval depending on the problem being addressed. |


| Item |Content|
| --- |---|
|idx| 2409.04286v1 |
|title| Using Large Language Models to Generate Authentic Multi-agent Knowledge Work Datasets |
|authors| Desiree HeimChristian JilekAdrian UlgesAndreas Dengel
|links| http://arxiv.org/abs/2409.04286v1 |
|updated| 2024-09-06 13:53:28 UTC |
|summary| Current publicly available knowledge work data collections lack diversityextensive annotations and contextual information about the users and theirdocuments. These issues hinder objective and comparable data-driven evaluationsand optimizations of knowledge work assistance systems. Due to the considerableresources needed to collect such data in real-life settings and the necessityof data censorship collecting such a dataset appears nearly impossible. Forthis reason we propose a configurable multi-agent knowledge work datasetgenerator. This system simulates collaborative knowledge work among agentsproducing Large Language Model-generated documents and accompanying datatraces. Additionally the generator captures all background information givenin its configuration or created during the simulation process in a knowledgegraph. Finally the resulting dataset can be utilized and shared withoutprivacy or confidentiality concerns.  This paper introduces our approachs design and vision and focuses ongenerating authentic knowledge work documents using Large Language Models. Ourstudy involving human raters who assessed 53 of the generated and 74 of thereal documents as realistic demonstrates the potential of our approach.Furthermore we analyze the authenticity criteria mentioned in theparticipants comments and elaborate on potential improvements for identifiedcommon issues. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2409.04434v1 |
|title| Accelerating Training with Neuron Interaction and Nowcasting Networks |
|authors| Boris KnyazevAbhinav MoudgilGuillaume LajoieEugene BelilovskySimon Lacoste-Julien
|links| http://arxiv.org/abs/2409.04434v1 |
|updated| 2024-09-06 17:55:49 UTC |
|summary| Neural network training can be accelerated when a learnable update rule isused in lieu of classic adaptive optimizers e.g. Adam. However learnableupdate rules can be costly and unstable to train and use. A simpler recentlyproposed approach to accelerate training is to use Adam for most of theoptimization steps and periodically only every few steps nowcast predictfuture parameters. We improve this approach by Neuron interaction andNowcasting NiNo networks. NiNo leverages neuron connectivity and graph neuralnetworks to more accurately nowcast parameters by learning in a supervised wayfrom a set of training trajectories over multiple tasks. We show that in somenetworks such as Transformers neuron connectivity is non-trivial. Byaccurately modeling neuron connectivity we allow NiNo to accelerate Adamtraining by up to 50 in vision and language tasks. |


| Item |Content|
| --- |---|
|idx| 2409.04432v1 |
|title| A Survey on Knowledge Organization Systems of Research Fields: Resources and Challenges |
|authors| Angelo SalatinoTanay AggarwalAndrea MannocciFrancesco OsborneEnrico Motta
|links| http://arxiv.org/abs/2409.04432v1 |
|updated| 2024-09-06 17:54:43 UTC |
|summary| Knowledge Organization Systems KOSs such as term lists thesauritaxonomies and ontologies play a fundamental role in categorising managingand retrieving information. In the academic domain KOSs are often adopted forrepresenting research areas and their relationships primarily aiming toclassify research articles academic courses patents books scientificvenues domain experts grants software experiment materials and severalother relevant products and agents. These structured representations ofresearch areas widely embraced by many academic fields have proven effectivein empowering AI-based systems to i enhance retrievability of relevantdocuments ii enable advanced analytic solutions to quantify the impact ofacademic research and iii analyse and forecast research dynamics. This paperaims to present a comprehensive survey of the current KOS for academicdisciplines. We analysed and compared 45 KOSs according to five maindimensions: scope structure curation usage and links to other KOSs. Ourresults reveal a very heterogeneous scenario in terms of scope scale qualityand usage highlighting the need for more integrated solutions for representingresearch knowledge across academic fields. We conclude by discussing the mainchallenges and the most promising future directions. |


| Item |Content|
| --- |---|
|idx| 2409.04428v1 |
|title| Hybrid Spiking Neural Networks for Low-Power Intra-Cortical Brain-Machine Interfaces |
|authors| Alexandru VasilacheJann KrausseKlaus KnoblochJuergen Becker
|links| http://arxiv.org/abs/2409.04428v1 |
|updated| 2024-09-06 17:48:44 UTC |
|summary| Intra-cortical brain-machine interfaces iBMIs have the potential todramatically improve the lives of people with paraplegia by restoring theirability to perform daily activities. However current iBMIs suffer fromscalability and mobility limitations due to bulky hardware and wiring. WirelessiBMIs offer a solution but are constrained by a limited data rate. To overcomethis challenge we are investigating hybrid spiking neural networks forembedded neural decoding in wireless iBMIs. The networks consist of a temporalconvolution-based compression followed by recurrent processing and a finalinterpolation back to the original sequence length. As recurrent units weexplore gated recurrent units GRUs leaky integrate-and-fire LIF neuronsand a combination of both - spiking GRUs sGRUs and analyze their differencesin terms of accuracy footprint and activation sparsity. To that end we traindecoders on the Nonhuman Primate Reaching with Multichannel SensorimotorCortex Electrophysiology dataset and evaluate it using the NeuroBenchframework targeting both tracks of the IEEE BioCAS Grand Challenge on NeuralDecoding. Our approach achieves high accuracy in predicting velocities ofprimate reaching movements from multichannel primary motor cortex recordingswhile maintaining a low number of synaptic operations surpassing the currentbaseline models in the NeuroBench framework. This work highlights the potentialof hybrid neural networks to facilitate wireless iBMIs with high decodingprecision and a substantial increase in the number of monitored neurons pavingthe way toward more advanced neuroprosthetic technologies. |


| Item |Content|
| --- |---|
|idx| 2409.04421v1 |
|title| RLPF: Reinforcement Learning from Prediction Feedback for User Summarization with LLMs |
|authors| Jiaxing WuLin NingLuyang LiuHarrison LeeNeo WuChao WangSushant PrakashShawn O'BanionBradley GreenJun Xie
|links| http://arxiv.org/abs/2409.04421v1 |
|updated| 2024-09-06 17:30:45 UTC |
|summary| LLM-powered personalization agent systems employ Large Language Models LLMsto predict users behavior from their past activities. However theireffectiveness often hinges on the ability to effectively leverage extensivelong user historical data due to its inherent noise and length of such data.Existing pretrained LLMs may generate summaries that are concise but lack thenecessary context for downstream tasks hindering their utility inpersonalization systems. To address these challenges we introduceReinforcement Learning from Prediction Feedback RLPF. RLPF fine-tunes LLMs togenerate concise human-readable user summaries that are optimized fordownstream task performance. By maximizing the usefulness of the generatedsummaries RLPF effectively distills extensive user history data whilepreserving essential information for downstream tasks. Our empirical evaluationdemonstrates significant improvements in both extrinsic downstream task utilityand intrinsic summary quality surpassing baseline methods by up to 22 ondownstream task performance and achieving an up to 84.59 win rate onFactuality Abstractiveness and Readability. RLPF also achieves a remarkable74 reduction in context length while improving performance on 16 out of 19unseen tasks and/or datasets showcasing its generalizability. This approachoffers a promising solution for enhancing LLM personalization by effectivelytransforming long noisy user histories into informative and human-readablerepresentations. |


| Item |Content|
| --- |---|
|idx| 2409.04415v1 |
|title| Improved Parallel Algorithm for Non-Monotone Submodular Maximization under Knapsack Constraint |
|authors| Tan D. TranCanh V. PhamDung T. K. HaPhuong N. H. Pham
|links| http://arxiv.org/abs/2409.04415v1 |
|updated| 2024-09-06 17:17:52 UTC |
|summary| This work proposes an efficient parallel algorithm for non-monotonesubmodular maximization under a knapsack constraint problem over the ground setof size n. Our algorithm improves the best approximation factor of theexisting parallel one from 8epsilon to 7epsilon with Olog nadaptive complexity.  The key idea of our approach is to create a new alternate thresholdalgorithmic framework. This strategy alternately constructs two disjointcandidate solutions within a constant number of sequence rounds. Then thealgorithm boosts solution quality without sacrificing the adaptive complexity.Extensive experimental studies on three applications Revenue MaximizationImage Summarization and Maximum Weighted Cut show that our algorithm not onlysignificantly increases solution quality but also requires comparativeadaptivity to state-of-the-art algorithms. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2409.04434v1 |
|title| Accelerating Training with Neuron Interaction and Nowcasting Networks |
|authors| Boris KnyazevAbhinav MoudgilGuillaume LajoieEugene BelilovskySimon Lacoste-Julien
|links| http://arxiv.org/abs/2409.04434v1 |
|updated| 2024-09-06 17:55:49 UTC |
|summary| Neural network training can be accelerated when a learnable update rule isused in lieu of classic adaptive optimizers e.g. Adam. However learnableupdate rules can be costly and unstable to train and use. A simpler recentlyproposed approach to accelerate training is to use Adam for most of theoptimization steps and periodically only every few steps nowcast predictfuture parameters. We improve this approach by Neuron interaction andNowcasting NiNo networks. NiNo leverages neuron connectivity and graph neuralnetworks to more accurately nowcast parameters by learning in a supervised wayfrom a set of training trajectories over multiple tasks. We show that in somenetworks such as Transformers neuron connectivity is non-trivial. Byaccurately modeling neuron connectivity we allow NiNo to accelerate Adamtraining by up to 50 in vision and language tasks. |


| Item |Content|
| --- |---|
|idx| 2409.04431v1 |
|title| Theory, Analysis, and Best Practices for Sigmoid Self-Attention |
|authors| Jason RamapuramFederico DanieliEeshan DhekaneFloris WeersDan BusbridgePierre AblinTatiana LikhomanenkoJagrit DiganiZijin GuAmitis ShidaniRuss Webb
|links| http://arxiv.org/abs/2409.04431v1 |
|updated| 2024-09-06 17:53:26 UTC |
|summary| Attention is a key part of the transformer architecture. It is asequence-to-sequence mapping that transforms each sequence element into aweighted sum of values. The weights are typically obtained as the softmax ofdot products between keys and queries. Recent work has explored alternatives tosoftmax attention in transformers such as ReLU and sigmoid activations. Inthis work we revisit sigmoid attention and conduct an in-depth theoretical andempirical analysis. Theoretically we prove that transformers with sigmoidattention are universal function approximators and benefit from improvedregularity compared to softmax attention. Through detailed empirical analysiswe identify stabilization of large initial attention norms during the earlystages of training as a crucial factor for the successful training of modelswith sigmoid attention outperforming prior attempts. We also introduceFLASHSIGMOID a hardware-aware and memory-efficient implementation of sigmoidattention yielding a 17 inference kernel speed-up over FLASHATTENTION2 on H100GPUs. Experiments across language vision and speech show that properlynormalized sigmoid attention matches the strong performance of softmaxattention on a wide range of domains and scales which previous attempts atsigmoid attention were unable to fully achieve. Our work unifies prior art andestablishes best practices for sigmoid attention as a drop-in softmaxreplacement in transformers. |


| Item |Content|
| --- |---|
|idx| 2409.04429v1 |
|title| VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation |
|authors| Yecheng WuZhuoyang ZhangJunyu ChenHaotian TangDacheng LiYunhao FangLigeng ZhuEnze XieHongxu YinLi YiSong HanYao Lu
|links| http://arxiv.org/abs/2409.04429v1 |
|updated| 2024-09-06 17:49:56 UTC |
|summary| VILA-U is a Unified foundation model that integrates Video Image Languageunderstanding and generation. Traditional visual language models VLMs useseparate modules for understanding and generating visual content which canlead to misalignment and increased complexity. In contrast VILA-U employs asingle autoregressive next-token prediction framework for both taskseliminating the need for additional components like diffusion models. Thisapproach not only simplifies the model but also achieves near state-of-the-artperformance in visual language understanding and generation. The success ofVILA-U is attributed to two main factors: the unified vision tower that alignsdiscrete visual tokens with textual inputs during pretraining which enhancesvisual perception and autoregressive image generation can achieve similarquality as diffusion models with high-quality dataset. This allows VILA-U toperform comparably to more complex models using a fully token-basedautoregressive framework. |


| Item |Content|
| --- |---|
|idx| 2409.04428v1 |
|title| Hybrid Spiking Neural Networks for Low-Power Intra-Cortical Brain-Machine Interfaces |
|authors| Alexandru VasilacheJann KrausseKlaus KnoblochJuergen Becker
|links| http://arxiv.org/abs/2409.04428v1 |
|updated| 2024-09-06 17:48:44 UTC |
|summary| Intra-cortical brain-machine interfaces iBMIs have the potential todramatically improve the lives of people with paraplegia by restoring theirability to perform daily activities. However current iBMIs suffer fromscalability and mobility limitations due to bulky hardware and wiring. WirelessiBMIs offer a solution but are constrained by a limited data rate. To overcomethis challenge we are investigating hybrid spiking neural networks forembedded neural decoding in wireless iBMIs. The networks consist of a temporalconvolution-based compression followed by recurrent processing and a finalinterpolation back to the original sequence length. As recurrent units weexplore gated recurrent units GRUs leaky integrate-and-fire LIF neuronsand a combination of both - spiking GRUs sGRUs and analyze their differencesin terms of accuracy footprint and activation sparsity. To that end we traindecoders on the Nonhuman Primate Reaching with Multichannel SensorimotorCortex Electrophysiology dataset and evaluate it using the NeuroBenchframework targeting both tracks of the IEEE BioCAS Grand Challenge on NeuralDecoding. Our approach achieves high accuracy in predicting velocities ofprimate reaching movements from multichannel primary motor cortex recordingswhile maintaining a low number of synaptic operations surpassing the currentbaseline models in the NeuroBench framework. This work highlights the potentialof hybrid neural networks to facilitate wireless iBMIs with high decodingprecision and a substantial increase in the number of monitored neurons pavingthe way toward more advanced neuroprosthetic technologies. |


| Item |Content|
| --- |---|
|idx| 2409.04421v1 |
|title| RLPF: Reinforcement Learning from Prediction Feedback for User Summarization with LLMs |
|authors| Jiaxing WuLin NingLuyang LiuHarrison LeeNeo WuChao WangSushant PrakashShawn O'BanionBradley GreenJun Xie
|links| http://arxiv.org/abs/2409.04421v1 |
|updated| 2024-09-06 17:30:45 UTC |
|summary| LLM-powered personalization agent systems employ Large Language Models LLMsto predict users behavior from their past activities. However theireffectiveness often hinges on the ability to effectively leverage extensivelong user historical data due to its inherent noise and length of such data.Existing pretrained LLMs may generate summaries that are concise but lack thenecessary context for downstream tasks hindering their utility inpersonalization systems. To address these challenges we introduceReinforcement Learning from Prediction Feedback RLPF. RLPF fine-tunes LLMs togenerate concise human-readable user summaries that are optimized fordownstream task performance. By maximizing the usefulness of the generatedsummaries RLPF effectively distills extensive user history data whilepreserving essential information for downstream tasks. Our empirical evaluationdemonstrates significant improvements in both extrinsic downstream task utilityand intrinsic summary quality surpassing baseline methods by up to 22 ondownstream task performance and achieving an up to 84.59 win rate onFactuality Abstractiveness and Readability. RLPF also achieves a remarkable74 reduction in context length while improving performance on 16 out of 19unseen tasks and/or datasets showcasing its generalizability. This approachoffers a promising solution for enhancing LLM personalization by effectivelytransforming long noisy user histories into informative and human-readablerepresentations. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2409.04440v1 |
|title| Synergy and Synchrony in Couple Dances |
|authors| Vongani MalulekeLea MüllerJathushan RajasegaranGeorgios PavlakosShiry GinosarAngjoo KanazawaJitendra Malik
|links| http://arxiv.org/abs/2409.04440v1 |
|updated| 2024-09-06 17:59:01 UTC |
|summary| This paper asks to what extent social interaction influences ones behavior.We study this in the setting of two dancers dancing as a couple. We firstconsider a baseline in which we predict a dancers future moves conditionedonly on their past motion without regard to their partner. We then investigatethe advantage of taking social information into account by conditioning also onthe motion of their dancing partner. We focus our analysis on Swing a dancegenre with tight physical coupling for which we present an in-the-wild videodataset. We demonstrate that single-person future motion prediction in thiscontext is challenging. Instead we observe that prediction greatly benefitsfrom considering the interaction partners behavior resulting in surprisinglycompelling couple dance synthesis results see supp. video. Our contributionsare a demonstration of the advantages of socially conditioned future motionprediction and an in-the-wild couple dance video dataset to enable futureresearch in this direction. Video results are available on the project website:https://von31.github.io/synNsync |


| Item |Content|
| --- |---|
|idx| 2409.04429v1 |
|title| VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation |
|authors| Yecheng WuZhuoyang ZhangJunyu ChenHaotian TangDacheng LiYunhao FangLigeng ZhuEnze XieHongxu YinLi YiSong HanYao Lu
|links| http://arxiv.org/abs/2409.04429v1 |
|updated| 2024-09-06 17:49:56 UTC |
|summary| VILA-U is a Unified foundation model that integrates Video Image Languageunderstanding and generation. Traditional visual language models VLMs useseparate modules for understanding and generating visual content which canlead to misalignment and increased complexity. In contrast VILA-U employs asingle autoregressive next-token prediction framework for both taskseliminating the need for additional components like diffusion models. Thisapproach not only simplifies the model but also achieves near state-of-the-artperformance in visual language understanding and generation. The success ofVILA-U is attributed to two main factors: the unified vision tower that alignsdiscrete visual tokens with textual inputs during pretraining which enhancesvisual perception and autoregressive image generation can achieve similarquality as diffusion models with high-quality dataset. This allows VILA-U toperform comparably to more complex models using a fully token-basedautoregressive framework. |


| Item |Content|
| --- |---|
|idx| 2409.04424v1 |
|title| Exploring Foundation Models for Synthetic Medical Imaging: A Study on Chest X-Rays and Fine-Tuning Techniques |
|authors| Davide Clode da SilvaMarina Musse BernardesNathalia Giacomini CerettaGabriel Vaz de SouzaGabriel Fonseca SilvaRafael Heitor BordiniSoraia Raupp Musse
|links| http://arxiv.org/abs/2409.04424v1 |
|updated| 2024-09-06 17:36:08 UTC |
|summary| Machine learning has significantly advanced healthcare by aiding in diseaseprevention and treatment identification. However accessing patient data can bechallenging due to privacy concerns and strict regulations. Generatingsynthetic realistic data offers a potential solution for overcoming theselimitations and recent studies suggest that fine-tuning foundation models canproduce such data effectively. In this study we explore the potential offoundation models for generating realistic medical images particularly chestx-rays and assess how their performance improves with fine-tuning. We proposeusing a Latent Diffusion Model starting with a pre-trained foundation modeland refining it through various configurations. Additionally we performedexperiments with input from a medical professional to assess the realism of theimages produced by each trained model. |


| Item |Content|
| --- |---|
|idx| 2409.04410v1 |
|title| Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation |
|authors| Zhuoyan LuoFengyuan ShiYixiao GeYujiu YangLimin WangYing Shan
|links| http://arxiv.org/abs/2409.04410v1 |
|updated| 2024-09-06 17:14:53 UTC |
|summary| We present Open-MAGVIT2 a family of auto-regressive image generation modelsranging from 300M to 1.5B. The Open-MAGVIT2 project produces an open-sourcereplication of Googles MAGVIT-v2 tokenizer a tokenizer with a super-largecodebook i.e. 218 codes and achieves the state-of-the-artreconstruction performance 1.17 rFID on ImageNet 256 times 256.Furthermore we explore its application in plain auto-regressive models andvalidate scalability properties. To assist auto-regressive models in predictingwith a super-large vocabulary we factorize it into two sub-vocabulary ofdifferent sizes by asymmetric token factorization and further introduce nextsub-token prediction to enhance sub-token interaction for better generationquality. We release all models and codes to foster innovation and creativity inthe field of auto-regressive visual generation. |


| Item |Content|
| --- |---|
|idx| 2409.04409v1 |
|title| Train Till You Drop: Towards Stable and Robust Source-free Unsupervised 3D Domain Adaptation |
|authors| Björn MicheleAlexandre BoulchTuan-Hung VuGilles PuyRenaud MarletNicolas Courty
|links| http://arxiv.org/abs/2409.04409v1 |
|updated| 2024-09-06 17:13:14 UTC |
|summary| We tackle the challenging problem of source-free unsupervised domainadaptation SFUDA for 3D semantic segmentation. It amounts to performingdomain adaptation on an unlabeled target domain without any access to sourcedata the available information is a model trained to achieve good performanceon the source domain. A common issue with existing SFUDA approaches is thatperformance degrades after some training time which is a by product of anunder-constrained and ill-posed problem. We discuss two strategies to alleviatethis issue. First we propose a sensible way to regularize the learningproblem. Second we introduce a novel criterion based on agreement with areference model. It is used 1 to stop the training when appropriate and 2as validator to select hyperparameters without any knowledge on the targetdomain. Our contributions are easy to implement and readily amenable for allSFUDA methods ensuring stable improvements over all baselines. We validate ourfindings on various 3D lidar settings achieving state-of-the-art performance.The project repository with code is: github.com/valeoai/TTYD. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2409.04434v1 |
|title| Accelerating Training with Neuron Interaction and Nowcasting Networks |
|authors| Boris KnyazevAbhinav MoudgilGuillaume LajoieEugene BelilovskySimon Lacoste-Julien
|links| http://arxiv.org/abs/2409.04434v1 |
|updated| 2024-09-06 17:55:49 UTC |
|summary| Neural network training can be accelerated when a learnable update rule isused in lieu of classic adaptive optimizers e.g. Adam. However learnableupdate rules can be costly and unstable to train and use. A simpler recentlyproposed approach to accelerate training is to use Adam for most of theoptimization steps and periodically only every few steps nowcast predictfuture parameters. We improve this approach by Neuron interaction andNowcasting NiNo networks. NiNo leverages neuron connectivity and graph neuralnetworks to more accurately nowcast parameters by learning in a supervised wayfrom a set of training trajectories over multiple tasks. We show that in somenetworks such as Transformers neuron connectivity is non-trivial. Byaccurately modeling neuron connectivity we allow NiNo to accelerate Adamtraining by up to 50 in vision and language tasks. |


| Item |Content|
| --- |---|
|idx| 2409.04367v1 |
|title| Provable Hyperparameter Tuning for Structured Pfaffian Settings |
|authors| Maria-Florina BalcanAnh Tuan NguyenDravyansh Sharma
|links| http://arxiv.org/abs/2409.04367v1 |
|updated| 2024-09-06 15:58:20 UTC |
|summary| Data-driven algorithm design automatically adapts algorithms to specificapplication domains achieving better performance. In the context ofparameterized algorithms this approach involves tuning the algorithmparameters using problem instances drawn from the problem distribution of thetarget application domain. While empirical evidence supports the effectivenessof data-driven algorithm design providing theoretical guarantees for severalparameterized families remains challenging. This is due to the intricatebehaviors of their corresponding utility functions which typically admitpiece-wise and discontinuity structures. In this work we present refinedframeworks for providing learning guarantees for parameterized data-drivenalgorithm design problems in both distributional and online learning settings.For the distributional learning setting we introduce the Pfaffian GJframework an extension of the classical GJ framework capable of providinglearning guarantees for function classes for which the computation involvesPfaffian functions. Unlike the GJ framework which is limited to functionclasses with computation characterized by rational functions our proposedframework can deal with function classes involving Pfaffian functions whichare much more general and widely applicable. We then show that for manyparameterized algorithms of interest their utility function possesses arefined piece-wise structure which automatically translates to learningguarantees using our proposed framework. For the online learning setting weprovide a new tool for verifying dispersion property of a sequence of lossfunctions. This sufficient condition allows no-regret learning for sequences ofpiece-wise structured loss functions where the piece-wise structure involvesPfaffian transition boundaries. |


| Item |Content|
| --- |---|
|idx| 2409.04365v1 |
|title| Leveraging Machine Learning for Official Statistics: A Statistical Manifesto |
|authors| Marco PutsDavid SalgadoPiet Daas
|links| http://arxiv.org/abs/2409.04365v1 |
|updated| 2024-09-06 15:57:25 UTC |
|summary| It is important for official statistics production to apply ML withstatistical rigor as it presents both opportunities and challenges. Althoughmachine learning has enjoyed rapid technological advances in recent years itsapplication does not possess the methodological robustness necessary to producehigh quality statistical results. In order to account for all sources of errorin machine learning models the Total Machine Learning Error TMLE ispresented as a framework analogous to the Total Survey Error Model used insurvey methodology. As a means of ensuring that ML models are both internallyvalid as well as externally valid the TMLE model addresses issues such asrepresentativeness and measurement errors. There are several case studiespresented illustrating the importance of applying more rigor to theapplication of machine learning in official statistics. |


| Item |Content|
| --- |---|
|idx| 2409.04352v1 |
|title| A naive aggregation algorithm for improving generalization in a class of learning problems |
|authors| Getachew K Befekadu
|links| http://arxiv.org/abs/2409.04352v1 |
|updated| 2024-09-06 15:34:17 UTC |
|summary| In this brief paper we present a naive aggregation algorithm for a typicallearning problem with expert advice setting in which the task of improvinggeneralization i.e. model validation is embedded in the learning process asa sequential decision-making problem. In particular we consider a class oflearning problem of point estimations for modeling high-dimensional nonlinearfunctions where a group of experts update their parameter estimates using thediscrete-time version of gradient systems with small additive noise termguided by the corresponding subsample datasets obtained from the originaldataset. Here our main objective is to provide conditions under which such analgorithm will sequentially determine a set of mixing distribution strategiesused for aggregating the experts estimates that ultimately leading to anoptimal parameter estimate i.e. as a consensus solution for all expertswhich is better than any individual experts estimate in terms of improvedgeneralization or learning performances. Finally as part of this work wepresent some numerical results for a typical case of nonlinear regressionproblem. |


| Item |Content|
| --- |---|
|idx| 2409.04332v1 |
|title| Amortized Bayesian Workflow (Extended Abstract) |
|authors| Marvin SchmittChengkun LiAki VehtariLuigi AcerbiPaul-Christian BürknerStefan T. Radev
|links| http://arxiv.org/abs/2409.04332v1 |
|updated| 2024-09-06 15:09:04 UTC |
|summary| Bayesian inference often faces a trade-off between computational speed andsampling accuracy. We propose an adaptive workflow that integrates rapidamortized inference with gold-standard MCMC techniques to achieve both speedand accuracy when performing inference on many observed datasets. Our approachuses principled diagnostics to guide the choice of inference method for eachdataset moving along the Pareto front from fast amortized sampling to slowerbut guaranteed-accurate MCMC when necessary. By reusing computations acrosssteps our workflow creates synergies between amortized and MCMC-basedinference. We demonstrate the effectiveness of this integrated approach on ageneralized extreme value task with 1000 observed data sets showing 90x timeefficiency gains while maintaining high posterior quality. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2409.04414v1 |
|title| Virtual Reality-Based Preoperative Planning for Optimized Trocar Placement in Thoracic Surgery: A Preliminary Study |
|authors| Arash HarirpoushGeorge RakovichMarta Kersten-OertelYiming Xiao
|links| http://arxiv.org/abs/2409.04414v1 |
|updated| 2024-09-06 17:17:20 UTC |
|summary| Video-assisted thoracic surgery VATS is a minimally invasive approach fortreating early-stage non-small-cell lung cancer. Optimal trocar placementduring VATS ensures comprehensive access to the thoracic cavity provides apanoramic endoscopic view and prevents instrument crowding. While establishedprinciples such as the Baseball Diamond Principle BDP and Triangle TargetPrinciple TTP exist surgeons mainly rely on experience and patient-specificanatomy for trocar placement potentially leading to sub-optimal surgical plansthat increase operative time and fatigue. To address this we present the firstvirtual reality VR-based pre-operative planning tool with tailored datavisualization and interaction designs for efficient and optimal VATS trocarplacement following the established surgical principles and consultation withan experienced surgeon. In our preliminary study we demonstrate the systemsapplication in right upper lung lobectomy a common thoracic proceduretypically using three trocars. A preliminary user study of our system indicatesit is efficient robust and user-friendly for planning optimal trocarplacement with a great promise for clinical application while offeringpotentially valuable insights for the development of other surgical VR systems. |


| Item |Content|
| --- |---|
|idx| 2409.04109v1 |
|title| Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers |
|authors| Chenglei SiDiyi YangTatsunori Hashimoto
|links| http://arxiv.org/abs/2409.04109v1 |
|updated| 2024-09-06 08:25:03 UTC |
|summary| Recent advancements in large language models LLMs have sparked optimismabout their potential to accelerate scientific discovery with a growing numberof works proposing research agents that autonomously generate and validate newideas. Despite this no evaluations have shown that LLM systems can take thevery first step of producing novel expert-level ideas let alone perform theentire research process. We address this by establishing an experimental designthat evaluates research idea generation while controlling for confounders andperforms the first head-to-head comparison between expert NLP researchers andan LLM ideation agent. By recruiting over 100 NLP researchers to write novelideas and blind reviews of both LLM and human ideas we obtain the firststatistically significant conclusion on current LLM capabilities for researchideation: we find LLM-generated ideas are judged as more novel p  0.05 thanhuman expert ideas while being judged slightly weaker on feasibility. Studyingour agent baselines closely we identify open problems in building andevaluating research agents including failures of LLM self-evaluation and theirlack of diversity in generation. Finally we acknowledge that human judgementsof novelty can be difficult even by experts and propose an end-to-end studydesign which recruits researchers to execute these ideas into full projectsenabling us to study whether these novelty and feasibility judgements result inmeaningful differences in research outcome. |


| Item |Content|
| --- |---|
|idx| 2409.04104v1 |
|title| MixNet: Joining Force of Classical and Modern Approaches Toward the Comprehensive Pipeline in Motor Imagery EEG Classification |
|authors| Phairot AutthasanRattanaphon ChaisaenHuy PhanMaarten De VosTheerawit Wilaiprasitporn
|links| http://dx.doi.org/10.1109/JIOT.2024.3402254 |
|updated| 2024-09-06 08:14:58 UTC |
|summary| Recent advances in deep learning DL have significantly impacted motorimagery MI-based brain-computer interface BCI systems enhancing thedecoding of electroencephalography EEG signals. However most studiesstruggle to identify discriminative patterns across subjects during MI taskslimiting MI classification performance. In this article we propose MixNet anovel classification framework designed to overcome this limitation byutilizing spectral-spatial signals from MI data along with a multitasklearning architecture named MIN2Net for classification. Here thespectral-spatial signals are generated using the filter-bank common spatialpatterns FBCSPs method on MI data. Since the multitask learning architectureis used for the classification task the learning in each task may exhibitdifferent generalization rates and potential overfitting across tasks. Toaddress this issue we implement adaptive gradient blending simultaneouslyregulating multiple loss weights and adjusting the learning pace for each taskbased on its generalization/overfitting tendencies. Experimental results on sixbenchmark data sets of different data sizes demonstrate that MixNetconsistently outperforms all state-of-the-art algorithms in subject-dependentand -independent settings. Finally the low-density EEG MI classificationresults show that MixNet outperforms all state-of-the-art algorithms offeringpromising implications for Internet of Thing IoT applications such aslightweight and portable EEG wearable devices based on low-density montages. |


| Item |Content|
| --- |---|
|idx| 2409.04099v1 |
|title| What Guides Our Choices? Modeling Developers' Trust and Behavioral Intentions Towards GenAI |
|authors| Rudrajit ChoudhuriBianca TrinkenreichRahul PanditaEirini KalliamvakouIgor SteinmacherMarco GerosaChristopher SanchezAnita Sarma
|links| http://arxiv.org/abs/2409.04099v1 |
|updated| 2024-09-06 08:05:28 UTC |
|summary| Generative AI genAI tools such as ChatGPT or Copilot are advertised toimprove developer productivity and are being integrated into softwaredevelopment. However misaligned trust skepticism and usability concerns canimpede the adoption of such tools. Research also indicates that AI can beexclusionary failing to support diverse users adequately. One such aspect ofdiversity is cognitive diversity -- variations in users cognitive styles --that leads to divergence in perspectives and interaction styles. When anindividuals cognitive style is unsupported it creates barriers to technologyadoption. Therefore to understand how to effectively integrate genAI toolsinto software development it is first important to model what factors affectdevelopers trust and intentions to adopt genAI tools in practice  We developed a theoretical model to 1 identify factors that influencedevelopers trust in genAI tools and 2 examine the relationship betweendevelopers trust cognitive styles and their intentions to use these tools.We surveyed software developers N238 at two major global tech organizationsand employed Partial Least Squares-Structural Equation Modeling PLS-SEM toevaluate our model. Our findings reveal that genAIs system/output qualityfunctional value and goal maintenance significantly influence developerstrust in these tools. Furthermore developers trust and cognitive stylesinfluence their intentions to use these tools. We offer practical suggestionsfor designing genAI tools for effective use and inclusive user experience. |


| Item |Content|
| --- |---|
|idx| 2409.04081v1 |
|title| UI-JEPA: Towards Active Perception of User Intent through Onscreen User Activity |
|authors| Yicheng FuRaviteja AnanthaPrabal VashishtJianpeng ChengEtai Littwin
|links| http://arxiv.org/abs/2409.04081v1 |
|updated| 2024-09-06 07:44:44 UTC |
|summary| Generating user intent from a sequence of user interface UI actions is acore challenge in comprehensive UI understanding. Recent advancements inmultimodal large language models MLLMs have led to substantial progress inthis area but their demands for extensive model parameters computing powerand high latency makes them impractical for scenarios requiring lightweighton-device solutions with low latency or heightened privacy. Additionally thelack of high-quality datasets has hindered the development of such lightweightmodels. To address these challenges we propose UI-JEPA a novel framework thatemploys masking strategies to learn abstract UI embeddings from unlabeled datathrough self-supervised learning combined with an LLM decoder fine-tuned foruser intent prediction. We also introduce two new UI-grounded multimodaldatasets Intent in the Wild IIW and Intent in the Tame IIT designedfor few-shot and zero-shot UI understanding tasks. IIW consists of 1.7K videosacross 219 intent categories while IIT contains 914 videos across 10categories. We establish the first baselines for these datasets showing thatrepresentations learned using a JEPA-style objective combined with an LLMdecoder can achieve user intent predictions that match the performance ofstate-of-the-art large MLLMs but with significantly reduced annotation anddeployment resources. Measured by intent similarity scores UI-JEPA outperformsGPT-4 Turbo and Claude 3.5 Sonnet by 10.0 and 7.2 respectively averagedacross two datasets. Notably UI-JEPA accomplishes the performance with a 50.5xreduction in computational cost and a 6.6x improvement in latency in the IIWdataset. These results underscore the effectiveness of UI-JEPA highlightingits potential for lightweight high-performance UI understanding. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2409.04230v1 |
|title| SPACE: A Python-based Simulator for Evaluating Decentralized Multi-Robot Task Allocation Algorithms |
|authors| Inmo Jang
|links| http://arxiv.org/abs/2409.04230v1 |
|updated| 2024-09-06 12:38:24 UTC |
|summary| Swarm robotics explores the coordination of multiple robots to achievecollective goals with collective decision-making being a central focus. Thisprocess involves decentralized robots autonomously making local decisions andcommunicating them which influences the overall emergent behavior. Testingsuch decentralized algorithms in real-world scenarios with hundreds or morerobots is often impractical underscoring the need for effective simulationtools. We propose SPACE Swarm Planning and Control Evaluation a Python-basedsimulator designed to support the research evaluation and comparison ofdecentralized Multi-Robot Task Allocation MRTA algorithms. SPACE streamlinescore algorithmic development by allowing users to implement decision-makingalgorithms as Python plug-ins easily construct agent behavior trees via anintuitive GUI and leverage built-in support for inter-agent communication andlocal task awareness. To demonstrate its practical utility we implement andevaluate CBBA and GRAPE within the simulator comparing their performanceacross different metrics particularly in scenarios with dynamically introducedtasks. This evaluation shows the usefulness of SPACE in conducting rigorous andstandardized comparisons of MRTA algorithms helping to support future researchin the field. |


| Item |Content|
| --- |---|
|idx| 2409.03881v1 |
|title| Multi-agent Path Finding for Mixed Autonomy Traffic Coordination |
|authors| Han ZhengZhongxia YanCathy Wu
|links| http://arxiv.org/abs/2409.03881v1 |
|updated| 2024-09-05 19:37:01 UTC |
|summary| In the evolving landscape of urban mobility the prospective integration ofConnected and Automated Vehicles CAVs with Human-Driven Vehicles HDVspresents a complex array of challenges and opportunities for autonomous drivingsystems. While recent advancements in robotics have yielded Multi-Agent PathFinding MAPF algorithms tailored for agent coordination task characterized bysimplified kinematics and complete control over agent behaviors thesesolutions are inapplicable in mixed-traffic environments where uncontrollableHDVs must coexist and interact with CAVs. Addressing this gap we propose theBehavior Prediction Kinematic Priority Based Search BK-PBS which leveragesan offline-trained conditional prediction model to forecast HDV responses toCAV maneuvers integrating these insights into a Priority Based Search PBSwhere the A search proceeds over motion primitives to accommodate kinematicconstraints. We compare BK-PBS with CAV planning algorithms derived byrule-based car-following models and reinforcement learning. Throughcomprehensive simulation on a highway merging scenario across diverse scenariosof CAV penetration rate and traffic density BK-PBS outperforms these baselinesin reducing collision rates and enhancing system-level travel delay. Our workis directly applicable to many scenarios of multi-human multi-robotcoordination. |


| Item |Content|
| --- |---|
|idx| 2409.03811v1 |
|title| PARCO: Learning Parallel Autoregressive Policies for Efficient Multi-Agent Combinatorial Optimization |
|authors| Federico BertoChuanbo HuaLaurin LuttmannJiwoo SonJunyoung ParkKyuree AhnChanghyun KwonLin XieJinkyoo Park
|links| http://arxiv.org/abs/2409.03811v1 |
|updated| 2024-09-05 17:49:18 UTC |
|summary| Multi-agent combinatorial optimization problems such as routing andscheduling have great practical relevance but present challenges due to theirNP-hard combinatorial nature hard constraints on the number of possibleagents and hard-to-optimize objective functions. This paper introduces PARCOParallel AutoRegressive Combinatorial Optimization a novel approach thatlearns fast surrogate solvers for multi-agent combinatorial problems withreinforcement learning by employing parallel autoregressive decoding. Wepropose a model with a Multiple Pointer Mechanism to efficiently decodemultiple decisions simultaneously by different agents enhanced by aPriority-based Conflict Handling scheme. Moreover we design specializedCommunication Layers that enable effective agent collaboration thus enrichingdecision-making. We evaluate PARCO in representative multi-agent combinatorialproblems in routing and scheduling and demonstrate that our learned solversoffer competitive results against both classical and neural baselines in termsof both solution quality and speed. We make our code openly available athttps://github.com/ai4co/parco. |


| Item |Content|
| --- |---|
|idx| 2409.03149v1 |
|title| Non-stationary and Sparsely-correlated Multi-output Gaussian Process with Spike-and-Slab Prior |
|authors| Wang XinmingLi YongxiangYue XiaoweiWu Jianguo
|links| http://arxiv.org/abs/2409.03149v1 |
|updated| 2024-09-05 00:56:25 UTC |
|summary| Multi-output Gaussian process MGP is commonly used as a transfer learningmethod to leverage information among multiple outputs. A key advantage of MGPis providing uncertainty quantification for prediction which is highlyimportant for subsequent decision-making tasks. However traditional MGP maynot be sufficiently flexible to handle multivariate data with dynamiccharacteristics particularly when dealing with complex temporal correlations.Additionally since some outputs may lack correlation transferring informationamong them may lead to negative transfer. To address these issues this studyproposes a non-stationary MGP model that can capture both the dynamic andsparse correlation among outputs. Specifically the covariance functions of MGPare constructed using convolutions of time-varying kernel functions. Then adynamic spike-and-slab prior is placed on correlation parameters toautomatically decide which sources are informative to the target output in thetraining process. An expectation-maximization EM algorithm is proposed forefficient model fitting. Both numerical studies and a real case demonstrate itsefficacy in capturing dynamic and sparse correlation structure and mitigatingnegative transfer for high-dimensional time-series data. Finally amountain-car reinforcement learning case highlights its potential applicationin decision making problems. |


| Item |Content|
| --- |---|
|idx| 2409.03052v1 |
|title| An Introduction to Centralized Training for Decentralized Execution in Cooperative Multi-Agent Reinforcement Learning |
|authors| Christopher Amato
|links| http://arxiv.org/abs/2409.03052v1 |
|updated| 2024-09-04 19:54:40 UTC |
|summary| Multi-agent reinforcement learning MARL has exploded in popularity inrecent years. Many approaches have been developed but they can be divided intothree main types: centralized training and execution CTE centralizedtraining for decentralized execution CTDE and Decentralized training andexecution DTE.  CTDE methods are the most common as they can use centralized informationduring training but execute in a decentralized manner -- using only informationavailable to that agent during execution. CTDE is the only paradigm thatrequires a separate training phase where any available information e.g. otheragent policies underlying states can be used. As a result they can be morescalable than CTE methods do not require communication during execution andcan often perform well. CTDE fits most naturally with the cooperative case butcan be potentially applied in competitive or mixed settings depending on whatinformation is assumed to be observed.  This text is an introduction to CTDE in cooperative MARL. It is meant toexplain the setting basic concepts and common methods. It does not cover allwork in CTDE MARL as the subarea is quite extensive. I have included work thatI believe is important for understanding the main concepts in the subarea andapologize to those that I have omitted. |


