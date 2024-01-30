# cs.CL 

| Item |Content|
| --- |---|
|idx| 2401.15077v1 |
|title| EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty |
|authors| Yuhui LiFangyun WeiChao ZhangHongyang Zhang
|links| http://arxiv.org/abs/2401.15077v1 |
|updated| 2024-01-26 18:59:01 UTC |
|summary| Auto-regressive decoding makes the inference of Large Language Models LLMstime-consuming. We propose a simple framework EAGLE Extrapolation Algorithmfor Greater Language-model Efficiency for lossless acceleration. Unliketraditional speculative sampling methods EAGLE operates the drafting processauto-regressively at the more regular second-top-layer feature level andaddresses the sampling uncertainty issues in the next-feature predictionproblems by integrating tokens from one time step ahead. The accelerationprovided by EAGLE is lossless: it involves no fine-tuning of the target LLMand the generated text maintains the same distribution as that of vanillaauto-regressive decoding. As of the submission of this paper EAGLE is thefastest known framework within the speculative sampling family. On MT-benchEAGLE is 3x faster than vanilla decoding 2x faster than Lookahead and 1.6xfaster than Medusa. Using gpt-fast EAGLE attains on average 160 tokens/s withLLaMA2-Chat 13B on a single RTX 3090 GPU compared to 24 tokens/s ofHuggingfaces implementations. |


| Item |Content|
| --- |---|
|idx| 2401.15068v1 |
|title| Pairing Orthographically Variant Literary Words to Standard Equivalents Using Neural Edit Distance Models |
|authors| Craig MessnerTom Lippincott
|links| http://arxiv.org/abs/2401.15068v1 |
|updated| 2024-01-26 18:49:34 UTC |
|summary| We present a novel corpus consisting of orthographically variant words foundin works of 19th century U.S. literature annotated with their correspondingstandard word pair. We train a set of neural edit distance models to pairthese variants with their standard forms and compare the performance of thesemodels to the performance of a set of neural edit distance models trained on acorpus of orthographic errors made by L2 English learners. Finally we analyzethe relative performance of these models in the light of different negativetraining sample generation strategies and offer concluding remarks on theunique challenge literary orthographic variation poses to string pairingmethodologies. |


| Item |Content|
| --- |---|
|idx| 2401.15055v1 |
|title| Deep learning-based approach for tomato classification in complex scenes |
|authors| Mikael A. MousseBethel C. A. R. K. AtohounCina Motamed
|links| http://arxiv.org/abs/2401.15055v1 |
|updated| 2024-01-26 18:33:57 UTC |
|summary| Tracking ripening tomatoes is time consuming and labor intensive. Artificialintelligence technologies combined with those of computer vision can help usersoptimize the process of monitoring the ripening status of plants. To this endwe have proposed a tomato ripening monitoring approach based on deep learningin complex scenes. The objective is to detect mature tomatoes and harvest themin a timely manner. The proposed approach is declined in two parts. Firstlythe images of the scene are transmitted to the pre-processing layer. Thisprocess allows the detection of areas of interest area of the image containingtomatoes. Then these images are used as input to the maturity detectionlayer. This layer based on a deep neural network learning algorithmclassifies the tomato thumbnails provided to it in one of the following fivecategories: green brittle pink pale red mature red. The experiments arebased on images collected from the internet gathered through searches usingtomato state across diverse languages including English German French andSpanish. The experimental results of the maturity detection layer on a datasetcomposed of images of tomatoes taken under the extreme conditions gave a goodclassification rate. |


| Item |Content|
| --- |---|
|idx| 2401.15050v1 |
|title| LongFin: A Multimodal Document Understanding Model for Long Financial Domain Documents |
|authors| Ahmed MasryAmir Hajian
|links| http://arxiv.org/abs/2401.15050v1 |
|updated| 2024-01-26 18:23:45 UTC |
|summary| Document AI is a growing research field that focuses on the comprehension andextraction of information from scanned and digital documents to make everydaybusiness operations more efficient. Numerous downstream tasks and datasets havebeen introduced to facilitate the training of AI models capable of parsing andextracting information from various document types such as receipts and scannedforms. Despite these advancements both existing datasets and models fail toaddress critical challenges that arise in industrial contexts. Existingdatasets primarily comprise short documents consisting of a single page whileexisting models are constrained by a limited maximum length often set at 512tokens. Consequently the practical application of these methods in financialservices where documents can span multiple pages is severely impeded. Toovercome these challenges we introduce LongFin a multimodal document AI modelcapable of encoding up to 4K tokens. We also propose the LongForms dataset acomprehensive financial dataset that encapsulates several industrial challengesin financial documents. Through an extensive evaluation we demonstrate theeffectiveness of the LongFin model on the LongForms dataset surpassing theperformance of existing public models while maintaining comparable results onexisting single-page benchmarks. |


| Item |Content|
| --- |---|
|idx| 2401.15043v1 |
|title| Health Text Simplification: An Annotated Corpus for Digestive Cancer Education and Novel Strategies for Reinforcement Learning |
|authors| Md Mushfiqur RahmanMohammad Sabik IrbazKai NorthMichelle S. WilliamsMarcos ZampieriKevin Lybarger
|links| http://arxiv.org/abs/2401.15043v1 |
|updated| 2024-01-26 18:13:57 UTC |
|summary| Objective: The reading level of health educational materials significantlyinfluences information understandability and accessibility particularly forminoritized populations. Many patient educational resources surpass the readinglevel and complexity of widely accepted standards. There is a critical need forhigh-performing text simplification models in health information to enhancedissemination and literacy. This need is particularly acute in cancereducation where effective prevention and screening education can substantiallyreduce morbidity and mortality.  Methods: We introduce Simplified Digestive Cancer SimpleDC a parallelcorpus of cancer education materials tailored for health text simplificationresearch. Utilizing SimpleDC alongside the existing Med-EASi corpus we exploreLarge Language Model LLM-based simplification methods including fine-tuningreinforcement learning RL reinforcement learning with human feedback RLHFdomain adaptation and prompt-based approaches. Our experimentation encompassesLlama 2 and GPT-4. A novel RLHF reward function is introduced featuring alightweight model adept at distinguishing between original and simplifiedtexts thereby enhancing the models effectiveness with unlabeled data.  Results: Fine-tuned Llama 2 models demonstrated high performance acrossvarious metrics. Our innovative RLHF reward function surpassed existing RL textsimplification reward functions in effectiveness. The results underscore thatRL/RLHF can augment fine-tuning facilitating model training on unlabeled textand improving performance. Additionally these methods effectively adaptout-of-domain text simplification models to targeted domains. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2401.15075v1 |
|title| Annotated Hands for Generative Models |
|authors| Yue YangAtith N GandhiGreg Turk
|links| http://arxiv.org/abs/2401.15075v1 |
|updated| 2024-01-26 18:57:54 UTC |
|summary| Generative models such as GANs and diffusion models have demonstratedimpressive image generation capabilities. Despite these successes thesesystems are surprisingly poor at creating images with hands. We propose a noveltraining framework for generative models that substantially improves theability of such systems to create hand images. Our approach is to augment thetraining images with three additional channels that provide annotations tohands in the image. These annotations provide additional structure that coaxthe generative model to produce higher quality hand images. We demonstrate thisapproach on two different generative models: a generative adversarial networkand a diffusion model. We demonstrate our method both on a new syntheticdataset of hand images and also on real photographs that contain hands. Wemeasure the improved quality of the generated hands through higher confidencein finger joint identification using an off-the-shelf hand detector. |


| Item |Content|
| --- |---|
|idx| 2401.15043v1 |
|title| Health Text Simplification: An Annotated Corpus for Digestive Cancer Education and Novel Strategies for Reinforcement Learning |
|authors| Md Mushfiqur RahmanMohammad Sabik IrbazKai NorthMichelle S. WilliamsMarcos ZampieriKevin Lybarger
|links| http://arxiv.org/abs/2401.15043v1 |
|updated| 2024-01-26 18:13:57 UTC |
|summary| Objective: The reading level of health educational materials significantlyinfluences information understandability and accessibility particularly forminoritized populations. Many patient educational resources surpass the readinglevel and complexity of widely accepted standards. There is a critical need forhigh-performing text simplification models in health information to enhancedissemination and literacy. This need is particularly acute in cancereducation where effective prevention and screening education can substantiallyreduce morbidity and mortality.  Methods: We introduce Simplified Digestive Cancer SimpleDC a parallelcorpus of cancer education materials tailored for health text simplificationresearch. Utilizing SimpleDC alongside the existing Med-EASi corpus we exploreLarge Language Model LLM-based simplification methods including fine-tuningreinforcement learning RL reinforcement learning with human feedback RLHFdomain adaptation and prompt-based approaches. Our experimentation encompassesLlama 2 and GPT-4. A novel RLHF reward function is introduced featuring alightweight model adept at distinguishing between original and simplifiedtexts thereby enhancing the models effectiveness with unlabeled data.  Results: Fine-tuned Llama 2 models demonstrated high performance acrossvarious metrics. Our innovative RLHF reward function surpassed existing RL textsimplification reward functions in effectiveness. The results underscore thatRL/RLHF can augment fine-tuning facilitating model training on unlabeled textand improving performance. Additionally these methods effectively adaptout-of-domain text simplification models to targeted domains. |


| Item |Content|
| --- |---|
|idx| 2401.15042v1 |
|title| PROXYQA: An Alternative Framework for Evaluating Long-Form Text Generation with Large Language Models |
|authors| Haochen TanZhijiang GuoZhan ShiLu XuZhili LiuXiaoguang LiYasheng WangLifeng ShangQun LiuLinqi Song
|links| http://arxiv.org/abs/2401.15042v1 |
|updated| 2024-01-26 18:12:25 UTC |
|summary| Large Language Models LLMs have exhibited remarkable success in long-formcontext comprehension tasks. However their capacity to generate long contentssuch as reports and articles remains insufficiently explored. Currentbenchmarks do not adequately assess LLMs ability to produce informative andcomprehensive content necessitating a more rigorous evaluation approach. Inthis study we introduce textscProxyQA a framework for evaluating long-formtext generation comprising in-depth human-curated textitmeta-questionsspanning various domains. Each meta-question contains correspondingtextitproxy-questions with annotated answers. LLMs are prompted to generateextensive content in response to these meta-questions. Utilizing an evaluatorand incorporating generated content as background context textscProxyQAevaluates the quality of generated content based on the evaluators performancein answering the textitproxy-questions. We examine multiple LLMsemphasizing textscProxyQAs demanding nature as a high-quality assessmenttool. Human evaluation demonstrates that evaluating throughtextitproxy-questions is a highly self-consistent andhuman-criteria-correlated validation method. The dataset and leaderboard willbe available at urlhttps://github.com/Namco0816/ProxyQA. |


| Item |Content|
| --- |---|
|idx| 2401.15006v1 |
|title| Airavata: Introducing Hindi Instruction-tuned LLM |
|authors| Jay GalaThanmay JayakumarJaavid Aktar HusainAswanth Kumar MMohammed Safi Ur Rahman KhanDiptesh KanojiaRatish PuduppullyMitesh M. KhapraRaj DabreRudra MurthyAnoop Kunchukuttan
|links| http://arxiv.org/abs/2401.15006v1 |
|updated| 2024-01-26 17:07:08 UTC |
|summary| We announce the initial release of Airavata an instruction-tuned LLM forHindi. Airavata was created by fine-tuning OpenHathi with diverseinstruction-tuning Hindi datasets to make it better suited for assistive tasks.Along with the model we also share the IndicInstruct dataset which is acollection of diverse instruction-tuning datasets to enable further researchfor Indic LLMs. Additionally we present evaluation benchmarks and a frameworkfor assessing LLM performance across tasks in Hindi. Currently Airavatasupports Hindi but we plan to expand this to all 22 scheduled Indic languages.You can access all artifacts at https://ai4bharat.github.io/airavata. |


| Item |Content|
| --- |---|
|idx| 2401.14968v1 |
|title| Atmosphere: Context and situational-aware collaborative IoT architecture for edge-fog-cloud computing |
|authors| Guadalupe OrtizMeftah ZouaiOkba KazarAlfonso Garcia-de-PradoJuan Boubeta-Puig
|links| http://dx.doi.org/10.1016/j.csi.2021.103550 |
|updated| 2024-01-26 16:01:09 UTC |
|summary| The Internet of Things IoT has grown significantly in popularityaccompanied by increased capacity and lower cost of communications andoverwhelming development of technologies. At the same time big data andreal-time data analysis have taken on great importance and have beenaccompanied by unprecedented interest in sharing data among citizens publicadministrations and other organisms giving rise to what is known as theCollaborative Internet of Things. This growth in data and infrastructure mustbe accompanied by a software architecture that allows its exploitation.Although there are various proposals focused on the exploitation of the IoT atedge fog and/or cloud levels it is not easy to find a software solution thatexploits the three tiers together taking maximum advantage not only of theanalysis of contextual and situational data at each tier but also of two-waycommunications between adjacent ones. In this paper we propose an architecturethat solves these deficiencies by proposing novel technologies which areappropriate for managing the resources of each tier: edge fog and cloud. Inaddition the fact that two-way communications along the three tiers of thearchitecture is allowed considerably enriches the contextual and situationalinformation in each layer and substantially assists decision making in realtime. The paper illustrates the proposed software architecture through a casestudy of respiratory disease surveillance in hospitals. As a result theproposed architecture permits efficient communications between the differenttiers responding to the needs of these types of IoT scenarios. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2401.15077v1 |
|title| EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty |
|authors| Yuhui LiFangyun WeiChao ZhangHongyang Zhang
|links| http://arxiv.org/abs/2401.15077v1 |
|updated| 2024-01-26 18:59:01 UTC |
|summary| Auto-regressive decoding makes the inference of Large Language Models LLMstime-consuming. We propose a simple framework EAGLE Extrapolation Algorithmfor Greater Language-model Efficiency for lossless acceleration. Unliketraditional speculative sampling methods EAGLE operates the drafting processauto-regressively at the more regular second-top-layer feature level andaddresses the sampling uncertainty issues in the next-feature predictionproblems by integrating tokens from one time step ahead. The accelerationprovided by EAGLE is lossless: it involves no fine-tuning of the target LLMand the generated text maintains the same distribution as that of vanillaauto-regressive decoding. As of the submission of this paper EAGLE is thefastest known framework within the speculative sampling family. On MT-benchEAGLE is 3x faster than vanilla decoding 2x faster than Lookahead and 1.6xfaster than Medusa. Using gpt-fast EAGLE attains on average 160 tokens/s withLLaMA2-Chat 13B on a single RTX 3090 GPU compared to 24 tokens/s ofHuggingfaces implementations. |


| Item |Content|
| --- |---|
|idx| 2401.15062v1 |
|title| Expert with Clustering: Hierarchical Online Preference Learning Framework |
|authors| Tianyue ZhouJung-Hoon ChoBabak Rahimi ArdabiliHamed TabkhiCathy Wu
|links| http://arxiv.org/abs/2401.15062v1 |
|updated| 2024-01-26 18:44:49 UTC |
|summary| Emerging mobility systems are increasingly capable of recommending options tomobility users to guide them towards personalized yet sustainable systemoutcomes. Even more so than the typical recommendation system it is crucial tominimize regret because 1 the mobility options directly affect the lives ofthe users and 2 the system sustainability relies on sufficient userparticipation. In this study we consider accelerating user preference learningby exploiting a low-dimensional latent space that captures the mobilitypreferences of users. We introduce a hierarchical contextual bandit frameworknamed Expert with Clustering EWC which integrates clustering techniques andprediction with expert advice. EWC efficiently utilizes hierarchical userinformation and incorporates a novel Loss-guided Distance metric. This metricis instrumental in generating more representative cluster centroids. In arecommendation scenario with N users T rounds per user and K optionsour algorithm achieves a regret bound of ONsqrtTlog K  NT. This boundconsists of two parts: the first term is the regret from the Hedge algorithmand the second term depends on the average loss from clustering. The algorithmperforms with low regret especially when a latent hierarchical structureexists among users. This regret bound underscores the theoretical andexperimental efficacy of EWC particularly in scenarios that demand rapidlearning and adaptation. Experimental results highlight that EWC cansubstantially reduce regret by 27.57 compared to the LinUCB baseline. Our workoffers a data-efficient approach to capturing both individual and collectivebehaviors making it highly applicable to contexts with hierarchicalstructures. We expect the algorithm to be applicable to other settings withlayered nuances of user preferences and information. |


| Item |Content|
| --- |---|
|idx| 2401.15059v1 |
|title| Fully Independent Communication in Multi-Agent Reinforcement Learning |
|authors| Rafael PinaVaruna De SilvaCorentin ArtaudXiaolan Liu
|links| http://arxiv.org/abs/2401.15059v1 |
|updated| 2024-01-26 18:42:01 UTC |
|summary| Multi-Agent Reinforcement Learning MARL comprises a broad area of researchwithin the field of multi-agent systems. Several recent works have focusedspecifically on the study of communication approaches in MARL. While multiplecommunication methods have been proposed these might still be too complex andnot easily transferable to more practical contexts. One of the reasons for thatis due to the use of the famous parameter sharing trick. In this paper weinvestigate how independent learners in MARL that do not share parameters cancommunicate. We demonstrate that this setting might incur into some problemsto which we propose a new learning scheme as a solution. Our results show thatdespite the challenges independent agents can still learn communicationstrategies following our method. Additionally we use this method toinvestigate how communication in MARL is affected by different networkcapacities both for sharing and not sharing parameters. We observe thatcommunication may not always be needed and that the chosen agent network sizesneed to be considered when used together with communication in order to achieveefficient learning. |


| Item |Content|
| --- |---|
|idx| 2401.15043v1 |
|title| Health Text Simplification: An Annotated Corpus for Digestive Cancer Education and Novel Strategies for Reinforcement Learning |
|authors| Md Mushfiqur RahmanMohammad Sabik IrbazKai NorthMichelle S. WilliamsMarcos ZampieriKevin Lybarger
|links| http://arxiv.org/abs/2401.15043v1 |
|updated| 2024-01-26 18:13:57 UTC |
|summary| Objective: The reading level of health educational materials significantlyinfluences information understandability and accessibility particularly forminoritized populations. Many patient educational resources surpass the readinglevel and complexity of widely accepted standards. There is a critical need forhigh-performing text simplification models in health information to enhancedissemination and literacy. This need is particularly acute in cancereducation where effective prevention and screening education can substantiallyreduce morbidity and mortality.  Methods: We introduce Simplified Digestive Cancer SimpleDC a parallelcorpus of cancer education materials tailored for health text simplificationresearch. Utilizing SimpleDC alongside the existing Med-EASi corpus we exploreLarge Language Model LLM-based simplification methods including fine-tuningreinforcement learning RL reinforcement learning with human feedback RLHFdomain adaptation and prompt-based approaches. Our experimentation encompassesLlama 2 and GPT-4. A novel RLHF reward function is introduced featuring alightweight model adept at distinguishing between original and simplifiedtexts thereby enhancing the models effectiveness with unlabeled data.  Results: Fine-tuned Llama 2 models demonstrated high performance acrossvarious metrics. Our innovative RLHF reward function surpassed existing RL textsimplification reward functions in effectiveness. The results underscore thatRL/RLHF can augment fine-tuning facilitating model training on unlabeled textand improving performance. Additionally these methods effectively adaptout-of-domain text simplification models to targeted domains. |


| Item |Content|
| --- |---|
|idx| 2401.15030v1 |
|title| On the generalization capacity of neural networks during generic multimodal reasoning |
|authors| Takuya ItoSoham DanMattia RigottiJames KozloskiMurray Campbell
|links| http://arxiv.org/abs/2401.15030v1 |
|updated| 2024-01-26 17:42:59 UTC |
|summary| The advent of the Transformer has led to the development of large languagemodels LLM which appear to demonstrate human-like capabilities. To assessthe generality of this class of models and a variety of other base neuralnetwork architectures to multimodal domains we evaluated and compared theircapacity for multimodal generalization. We introduce a multimodalquestion-answer benchmark to evaluate three specific types ofout-of-distribution OOD generalization performance: distractor generalizationgeneralization in the presence of distractors systematic compositionalgeneralization generalization to new task permutations and productivecompositional generalization generalization to more complex tasks structures.We found that across model architectures e.g. RNNs Transformers Perceiversetc. models with multiple attention layers or models that leveragedcross-attention mechanisms between input domains fared better. Our positiveresults demonstrate that for multimodal distractor and systematicgeneralization either cross-modal attention or models with deeper attentionlayers are key architectural features required to integrate multimodal inputs.On the other hand neither of these architectural features led to productivegeneralization suggesting fundamental limitations of existing architecturesfor specific types of multimodal generalization. These results demonstrate thestrengths and limitations of specific architectural components underlyingmodern neural models for multimodal reasoning. Finally we provide Generic COGgCOG a configurable benchmark with several multimodal generalization splitsfor future studies to explore. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2401.15075v1 |
|title| Annotated Hands for Generative Models |
|authors| Yue YangAtith N GandhiGreg Turk
|links| http://arxiv.org/abs/2401.15075v1 |
|updated| 2024-01-26 18:57:54 UTC |
|summary| Generative models such as GANs and diffusion models have demonstratedimpressive image generation capabilities. Despite these successes thesesystems are surprisingly poor at creating images with hands. We propose a noveltraining framework for generative models that substantially improves theability of such systems to create hand images. Our approach is to augment thetraining images with three additional channels that provide annotations tohands in the image. These annotations provide additional structure that coaxthe generative model to produce higher quality hand images. We demonstrate thisapproach on two different generative models: a generative adversarial networkand a diffusion model. We demonstrate our method both on a new syntheticdataset of hand images and also on real photographs that contain hands. Wemeasure the improved quality of the generated hands through higher confidencein finger joint identification using an off-the-shelf hand detector. |


| Item |Content|
| --- |---|
|idx| 2401.15071v2 |
|title| From GPT-4 to Gemini and Beyond: Assessing the Landscape of MLLMs on Generalizability, Trustworthiness and Causality through Four Modalities |
|authors| Chaochao LuChen QianGuodong ZhengHongxing FanHongzhi GaoJie ZhangJing ShaoJingyi DengJinlan FuKexin HuangKunchang LiLijun LiLimin WangLu ShengMeiqi ChenMing ZhangQibing RenSirui ChenTao GuiWanli OuyangYali WangYan TengYaru WangYi WangYinan HeYingchun WangYixu WangYongting ZhangYu QiaoYujiong ShenYurong MouYuxi ChenZaibin ZhangZhelun ShiZhenfei YinZhipin Wang
|links| http://arxiv.org/abs/2401.15071v2 |
|updated| 2024-01-29 15:18:45 UTC |
|summary| Multi-modal Large Language Models MLLMs have shown impressive abilities ingenerating reasonable responses with respect to multi-modal contents. Howeverthere is still a wide gap between the performance of recent MLLM-basedapplications and the expectation of the broad public even though the mostpowerful OpenAIs GPT-4 and Googles Gemini have been deployed. This paperstrives to enhance understanding of the gap through the lens of a qualitativestudy on the generalizability trustworthiness and causal reasoningcapabilities of recent proprietary and open-source MLLMs across fourmodalities: ie text code image and video ultimately aiming to improve thetransparency of MLLMs. We believe these properties are several representativefactors that define the reliability of MLLMs in supporting various downstreamapplications. To be specific we evaluate the closed-source GPT-4 and Geminiand 6 open-source LLMs and MLLMs. Overall we evaluate 230 manually designedcases where the qualitative results are then summarized into 12 scores ie 4modalities times 3 properties. In total we uncover 14 empirical findings thatare useful to understand the capabilities and limitations of both proprietaryand open-source MLLMs towards more reliable downstream multi-modalapplications. |


| Item |Content|
| --- |---|
|idx| 2401.15055v1 |
|title| Deep learning-based approach for tomato classification in complex scenes |
|authors| Mikael A. MousseBethel C. A. R. K. AtohounCina Motamed
|links| http://arxiv.org/abs/2401.15055v1 |
|updated| 2024-01-26 18:33:57 UTC |
|summary| Tracking ripening tomatoes is time consuming and labor intensive. Artificialintelligence technologies combined with those of computer vision can help usersoptimize the process of monitoring the ripening status of plants. To this endwe have proposed a tomato ripening monitoring approach based on deep learningin complex scenes. The objective is to detect mature tomatoes and harvest themin a timely manner. The proposed approach is declined in two parts. Firstlythe images of the scene are transmitted to the pre-processing layer. Thisprocess allows the detection of areas of interest area of the image containingtomatoes. Then these images are used as input to the maturity detectionlayer. This layer based on a deep neural network learning algorithmclassifies the tomato thumbnails provided to it in one of the following fivecategories: green brittle pink pale red mature red. The experiments arebased on images collected from the internet gathered through searches usingtomato state across diverse languages including English German French andSpanish. The experimental results of the maturity detection layer on a datasetcomposed of images of tomatoes taken under the extreme conditions gave a goodclassification rate. |


| Item |Content|
| --- |---|
|idx| 2401.15048v1 |
|title| Unrecognizable Yet Identifiable: Image Distortion with Preserved Embeddings |
|authors| Dmytro ZakharovOleksandr KuznetsovEmanuele Frontoni
|links| http://arxiv.org/abs/2401.15048v1 |
|updated| 2024-01-26 18:20:53 UTC |
|summary| In the realm of security applications biometric authentication systems playa crucial role yet one often encounters challenges concerning privacy andsecurity while developing one. One of the most fundamental challenges lies inavoiding storing biometrics directly in the storage but still achievingdecently high accuracy. Addressing this issue we contribute to both artificialintelligence and engineering fields. We introduce an innovative imagedistortion technique that effectively renders facial images unrecognizable tothe eye while maintaining their identifiability by neural network models. Fromthe theoretical perspective we explore how reliable state-of-the-artbiometrics recognition neural networks are by checking the maximal degree ofimage distortion which leaves the predicted identity unchanged. On the otherhand applying this technique demonstrates a practical solution to theengineering challenge of balancing security precision and performance inbiometric authentication systems. Through experimenting on the widely useddatasets we assess the effectiveness of our method in preserving AI featurerepresentation and distorting relative to conventional metrics. We also compareour method with previously used approaches. |


| Item |Content|
| --- |---|
|idx| 2401.15029v1 |
|title| Learning Neural Radiance Fields of Forest Structure for Scalable and Fine Monitoring |
|authors| Juan Castorena
|links| http://arxiv.org/abs/2401.15029v1 |
|updated| 2024-01-26 17:42:52 UTC |
|summary| This work leverages neural radiance fields and remote sensing for forestryapplications. Here we show neural radiance fields offer a wide range ofpossibilities to improve upon existing remote sensing methods in forestmonitoring. We present experiments that demonstrate their potential to: 1express fine features of forest 3D structure 2 fuse available remote sensingmodalities and 3 improve upon 3D structure derived forest metrics.Altogether these properties make neural fields an attractive computationaltool with great potential to further advance the scalability and accuracy offorest monitoring programs. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2401.14989v1 |
|title| Mapping-to-Parameter Nonlinear Functional Regression with Novel B-spline Free Knot Placement Algorithm |
|authors| Chengdong ShiChing-Hsun TsengWei ZhaoXiao-Jun Zeng
|links| http://arxiv.org/abs/2401.14989v1 |
|updated| 2024-01-26 16:35:48 UTC |
|summary| We propose a novel approach to nonlinear functional regression called theMapping-to-Parameter function model which addresses complex and nonlinearfunctional regression problems in parameter space by employing any supervisedlearning technique. Central to this model is the mapping of function data froman infinite-dimensional function space to a finite-dimensional parameter space.This is accomplished by concurrently approximating multiple functions with acommon set of B-spline basis functions by any chosen order with their knotdistribution determined by the Iterative Local Placement Algorithm a newlyproposed free knot placement algorithm. In contrast to the conventionalequidistant knot placement strategy that uniformly distributes knot locationsbased on a predefined number of knots our proposed algorithms determine knotlocation according to the local complexity of the input or output functions.The performance of our knot placement algorithms is shown to be robust in bothsingle-function approximation and multiple-function approximation contexts.Furthermore the effectiveness and advantage of the proposed prediction modelin handling both function-on-scalar regression and function-on-functionregression problems are demonstrated through several real data applications incomparison with four groups of state-of-the-art methods. |


| Item |Content|
| --- |---|
|idx| 2401.14973v1 |
|title| Discovering group dynamics in synchronous time series via hierarchical recurrent switching-state models |
|authors| Michael WojnowiczPreetish RathEric MillerJeffrey MillerClifford HancockMeghan O'DonovanSeth Elkin-FrankstonThaddeus BrunyeMichael C. Hughes
|links| http://arxiv.org/abs/2401.14973v1 |
|updated| 2024-01-26 16:06:01 UTC |
|summary| We seek to model a collection of time series arising from multiple entitiesinteracting over the same time period. Recent work focused on modelingindividual time series is inadequate for our intended applications wherecollective system-level behavior influences the trajectories of individualentities. To address such problems we present a new hierarchicalswitching-state model that can be trained in an unsupervised fashion tosimultaneously explain both system-level and individual-level dynamics. Weemploy a latent system-level discrete state Markov chain that drives latententity-level chains which in turn govern the dynamics of each observed timeseries. Feedback from the observations to the chains at both the entity andsystem levels improves flexibility via context-dependent state transitions. Ourhierarchical switching recurrent dynamical models can be learned viaclosed-form variational coordinate ascent updates to all latent chains thatscale linearly in the number of individual time series. This is asymptoticallyno more costly than fitting separate models for each entity. Experiments onsynthetic and real datasets show that our model can produce better forecasts offuture entity behavior than existing methods. Moreover the availability oflatent state chains at both the entity and system level enables interpretationof group dynamics. |


| Item |Content|
| --- |---|
|idx| 2401.14893v1 |
|title| A structured regression approach for evaluating model performance across intersectional subgroups |
|authors| Christine HerlihyKimberly TruongAlexandra ChouldechovaMiroslav Dudik
|links| http://arxiv.org/abs/2401.14893v1 |
|updated| 2024-01-26 14:21:45 UTC |
|summary| Disaggregated evaluation is a central task in AI fairness assessment withthe goal to measure an AI systems performance across different subgroupsdefined by combinations of demographic or other sensitive attributes. Thestandard approach is to stratify the evaluation data across subgroups andcompute performance metrics separately for each group. However even formoderately-sized evaluation datasets sample sizes quickly get small onceconsidering intersectional subgroups which greatly limits the extent to whichintersectional groups are considered in many disaggregated evaluations. In thiswork we introduce a structured regression approach to disaggregated evaluationthat we demonstrate can yield reliable system performance estimates even forvery small subgroups. We also provide corresponding inference strategies forconstructing confidence intervals and explore how goodness-of-fit testing canyield insight into the structure of fairness-related harms experienced byintersectional groups. We evaluate our approach on two publicly availabledatasets and several variants of semi-synthetic data. The results show thatour method is considerably more accurate than the standard approach especiallyfor small subgroups and goodness-of-fit testing helps identify the key factorsthat drive differences in performance. |


| Item |Content|
| --- |---|
|idx| 2401.14884v1 |
|title| P3LS: Partial Least Squares under Privacy Preservation |
|authors| Du Nguyen DuyRamin Nikzad-Langerodi
|links| http://arxiv.org/abs/2401.14884v1 |
|updated| 2024-01-26 14:08:43 UTC |
|summary| Modern manufacturing value chains require intelligent orchestration ofprocesses across company borders in order to maximize profits while fosteringsocial and environmental sustainability. However the implementation ofintegrated systems-level approaches for data-informed decision-making alongvalue chains is currently hampered by privacy concerns associated withcross-organizational data exchange and integration. We here proposePrivacy-Preserving Partial Least Squares P3LS regression a novel federatedlearning technique that enables cross-organizational data integration andprocess modeling with privacy guarantees. P3LS involves a singular valuedecomposition SVD based PLS algorithm and employs removable random masksgenerated by a trusted authority in order to protect the privacy of the datacontributed by each data holder. We demonstrate the capability of P3LS tovertically integrate process data along a hypothetical value chain consistingof three parties and to improve the prediction performance on severalprocess-related key performance indicators. Furthermore we show the numericalequivalence of P3LS and PLS model components on simulated data and provide athorough privacy analysis of the former. Moreover we propose a mechanism fordetermining the relevance of the contributed data to the problem beingaddressed thus creating a basis for quantifying the contribution ofparticipants. |


| Item |Content|
| --- |---|
|idx| 2401.14868v1 |
|title| Particle-MALA and Particle-mGRAD: Gradient-based MCMC methods for high-dimensional state-space models |
|authors| Adrien CorenflosAxel Finke
|links| http://arxiv.org/abs/2401.14868v1 |
|updated| 2024-01-26 13:52:40 UTC |
|summary| State-of-the-art methods for Bayesian inference in state-space models are aconditional sequential Monte Carlo CSMC algorithms b sophisticatedclassical MCMC algorithms like MALA or mGRAD from Titsias andPapaspiliopoulos 2018 arXiv:1610.09641v3 stat.ML. The former propose Nparticles at each time step to exploit the models decorrelation-over-timeproperty and thus scale favourably with the time horizon T  but break downif the dimension of the latent states D is large. The latter leveragegradient-/prior-informed local proposals to scale favourably with D butexhibit sub-optimal scalability with T due to a lack of model-structureexploitation. We introduce methods which combine the strengths of bothapproaches. The first Particle-MALA spreads N particles locally around thecurrent state using gradient information thus extending MALA to T  1 timesteps and N  1 proposals. The second Particle-mGRAD additionallyincorporates conditionally Gaussian prior dynamics into the proposal thusextending the mGRAD algorithm to T  1 time steps and N  1 proposals. Weprove that Particle-mGRAD interpolates between CSMC and Particle-MALAresolving the tuning problem of choosing between CSMC superior for highlyinformative prior dynamics and Particle-MALA superior for weakly informativeprior dynamics. We similarly extend other classical MCMC approaches likeauxiliary MALA aGRAD and preconditioned Crank-Nicolson-Langevin PCNL to T 1 time steps and N  1 proposals. In experiments for both highly andweakly informative prior dynamics our methods substantially improve upon bothCSMC and sophisticated classical MCMC approaches. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2401.15032v1 |
|title| Color Maker: a Mixed-Initiative Approach to Creating Accessible Color Maps |
|authors| Amey SalviKecheng LuMichael E. PapkaYunhai WangKhairi Reda
|links| http://dx.doi.org/10.1145/3613904.3642265 |
|updated| 2024-01-26 17:48:28 UTC |
|summary| Quantitative data is frequently represented using color yet designingeffective color mappings is a challenging task requiring one to balanceperceptual standards with personal color preference. Current design toolseither overwhelm novices with complexity or offer limited customizationoptions. We present ColorMaker a mixed-initiative approach for creatingcolormaps. ColorMaker combines fluid user interaction with real-timeoptimization to generate smooth continuous color ramps. Users specify theirloose color preferences while leaving the algorithm to generate precise colorsequences meeting both designer needs and established guidelines. ColorMakercan create new colormaps including designs accessible for people withcolor-vision deficiencies starting from scratch or with only partial inputthus supporting ideation and iterative refinement. We show that our approachcan generate designs with similar or superior perceptual characteristics tostandard colormaps. A user study demonstrates how designers of varying skilllevels can use this tool to create custom high-quality colormaps. ColorMakeris available at https://colormaker.org |


| Item |Content|
| --- |---|
|idx| 2401.14978v1 |
|title| Robust Dual-Modal Speech Keyword Spotting for XR Headsets |
|authors| Zhuojiang CaiYuhan MaFeng Lu
|links| http://arxiv.org/abs/2401.14978v1 |
|updated| 2024-01-26 16:09:18 UTC |
|summary| While speech interaction finds widespread utility within the Extended RealityXR domain conventional vocal speech keyword spotting systems continue tograpple with formidable challenges including suboptimal performance in noisyenvironments impracticality in situations requiring silence andsusceptibility to inadvertent activations when others speak nearby. Thesechallenges however can potentially be surmounted through the cost-effectivefusion of voice and lip movement information. Consequently we propose a novelvocal-echoic dual-modal keyword spotting system designed for XR headsets. Wedevise two different modal fusion approches and conduct experiments to test thesystems performance across diverse scenarios. The results show that ourdual-modal system not only consistently outperforms its single-modalcounterparts demonstrating higher precision in both typical and noisyenvironments but also excels in accurately identifying silent utterances.Furthermore we have successfully applied the system in real-timedemonstrations achieving promising results. The code is available athttps://github.com/caizhuojiang/VE-KWS. |


| Item |Content|
| --- |---|
|idx| 2401.14972v1 |
|title| "It's Sink or Swim'': Exploring Patients' Challenges and Tool Needs for Self-Management of Postoperative Acute Pain |
|authors| Souleima ZghabGabrielle PagéMélanie LussierSylvain BédardJinghui Cheng
|links| http://dx.doi.org/10.1145/3613904.3642916 |
|updated| 2024-01-26 16:05:47 UTC |
|summary| Poorly managed postoperative acute pain can have long-lasting negativeimpacts and pose a major healthcare issue. There is limited investigation tounderstand and address the unique needs of patients experiencing acute pain. Inthis paper we tackle this gap through an interview study with 14 patients whorecently underwent postoperative acute pain to understand their challenges inpain self-management and their need for supportive tools. Our analysisidentified various factors associated with the major aspects of acute painself-management. Together our findings indicated that tools for supportingthese patients need to carefully consider information and support delivery toadapt to rapid changes in pain experiences offer personalized and dynamicassistance that adapts to individual situations in context and monitor emotionwhen promoting motivation. Overall our work provided valuable knowledge toaddress the less-investigated but highly-needed problem of designing technologyfor the self-management of acute pain and similar health conditions. |


| Item |Content|
| --- |---|
|idx| 2401.14936v1 |
|title| Reassessing Java Code Readability Models with a Human-Centered Approach |
|authors| Agnia SergeyukOlga LvovaSergey TitovAnastasiia SerovaFarid BagirovEvgeniia KirillovaTimofey Bryksin
|links| http://arxiv.org/abs/2401.14936v1 |
|updated| 2024-01-26 15:18:22 UTC |
|summary| To ensure that Large Language Models LLMs effectively support userproductivity they need to be adjusted. Existing Code Readability CR modelscan guide this alignment. However there are concerns about their relevance inmodern software engineering since they often miss the developers notion ofreadability and rely on outdated code. This research assesses existing Java CRmodels for LLM adjustments measuring the correlation between their anddevelopers evaluations of AI-generated Java code. Using the Repertory GridTechnique with 15 developers we identified 12 key code aspects influencing CRthat were consequently assessed by 390 programmers when labeling 120AI-generated snippets. Our findings indicate that when AI generates concise andexecutable code it is often considered readable by CR models and developers.However a limited correlation between these evaluations underscores theimportance of future research on learning objectives for adjusting LLMs and onthe aspects influencing CR evaluations included in predictive models. |


| Item |Content|
| --- |---|
|idx| 2401.14935v1 |
|title| Appropriateness of LLM-equipped Robotic Well-being Coach Language in the Workplace: A Qualitative Evaluation |
|authors| Micol SpitaleMinja AxelssonHatice Gunes
|links| http://arxiv.org/abs/2401.14935v1 |
|updated| 2024-01-26 15:17:28 UTC |
|summary| Robotic coaches have been recently investigated to promote mental well-beingin various contexts such as workplaces and homes. With the widespread use ofLarge Language Models LLMs HRI researchers are called to consider languageappropriateness when using such generated language for robotic mentalwell-being coaches in the real world. Therefore this paper presents the firstwork that investigated the language appropriateness of robot mental well-beingcoach in the workplace. To this end we conducted an empirical study thatinvolved 17 employees who interacted over 4 weeks with a robotic mentalwell-being coach equipped with LLM-based capabilities. After the study weindividually interviewed them and we conducted a focus group of 1.5 hours with11 of them. The focus group consisted of: i an ice-breaking activity iievaluation of robotic coach language appropriateness in various scenarios andiii listing shoulds and shouldnts for designing appropriate robotic coachlanguage for mental well-being. From our qualitative evaluation we found thata language-appropriate robotic coach should 1 ask deep questions whichexplore feelings of the coachees rather than superficial questions 2express and show emotional and empathic understanding of the context and 3not make any assumptions without clarifying with follow-up questions to avoidbias and stereotyping. These results can inform the design oflanguage-appropriate robotic coach to promote mental well-being in real-worldcontexts. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2401.15059v1 |
|title| Fully Independent Communication in Multi-Agent Reinforcement Learning |
|authors| Rafael PinaVaruna De SilvaCorentin ArtaudXiaolan Liu
|links| http://arxiv.org/abs/2401.15059v1 |
|updated| 2024-01-26 18:42:01 UTC |
|summary| Multi-Agent Reinforcement Learning MARL comprises a broad area of researchwithin the field of multi-agent systems. Several recent works have focusedspecifically on the study of communication approaches in MARL. While multiplecommunication methods have been proposed these might still be too complex andnot easily transferable to more practical contexts. One of the reasons for thatis due to the use of the famous parameter sharing trick. In this paper weinvestigate how independent learners in MARL that do not share parameters cancommunicate. We demonstrate that this setting might incur into some problemsto which we propose a new learning scheme as a solution. Our results show thatdespite the challenges independent agents can still learn communicationstrategies following our method. Additionally we use this method toinvestigate how communication in MARL is affected by different networkcapacities both for sharing and not sharing parameters. We observe thatcommunication may not always be needed and that the chosen agent network sizesneed to be considered when used together with communication in order to achieveefficient learning. |


| Item |Content|
| --- |---|
|idx| 2401.15036v1 |
|title| Distributed Simultaneous Localisation and Auto-Calibration using Gaussian Belief Propagation |
|authors| Riku MuraiIgnacio AlzugarayPaul H. J. KellyAndrew J. Davison
|links| http://dx.doi.org/10.1109/LRA.2024.3352361 |
|updated| 2024-01-26 17:54:55 UTC |
|summary| We present a novel scalable fully distributed and online method forsimultaneous localisation and extrinsic calibration for multi-robot setups.Individual a priori unknown robot poses are probabilistically inferred asrobots sense each other while simultaneously calibrating their sensors andmarkers extrinsic using Gaussian Belief Propagation. In the presentedexperiments we show how our method not only yields accurate robot localisationand auto-calibration but also is able to perform under challengingcircumstances such as highly noisy measurements significant communicationfailures or limited communication range. |


| Item |Content|
| --- |---|
|idx| 2401.15026v1 |
|title| Multi-Agent Coordination for a Partially Observable and Dynamic Robot Soccer Environment with Limited Communication |
|authors| Daniele AffinitaFlavio VolpiValerio SpagnoliVincenzo SurianiDaniele NardiDomenico D. Bloisi
|links| http://arxiv.org/abs/2401.15026v1 |
|updated| 2024-01-26 17:37:25 UTC |
|summary| RoboCup represents an International testbed for advancing research in AI androbotics focusing on a definite goal: developing a robot team that can winagainst the human world soccer champion team by the year 2050. To achieve thisgoal autonomous humanoid robots coordination is crucial. This paper exploresnovel solutions within the RoboCup Standard Platform League SPL where areduction in WiFi communication is imperative leading to the development ofnew coordination paradigms. The SPL has experienced a substantial decrease innetwork packet rate compelling the need for advanced coordinationarchitectures to maintain optimal team functionality in dynamic environments.Inspired by market-based task assignment we introduce a novel distributedcoordination system to orchestrate autonomous robots actions efficiently inlow communication scenarios. This approach has been tested with NAO robotsduring official RoboCup competitions and in the SimRobot simulatordemonstrating a notable reduction in task overlaps in limited communicationsettings. |


| Item |Content|
| --- |---|
|idx| 2401.14903v1 |
|title| Energy Flexibility Potential in the Brewery Sector: A Multi-agent Based Simulation of 239 Danish Breweries |
|authors| Daniel Anthony HowardZheng Grace MaJacob Alstrup EngvangMorten HagenauKathrine Lau JorgensenJonas Fausing OlesenBo Nørregaard Jørgensen
|links| http://dx.doi.org/10.1109/APPEEC53445.2022.10072200 |
|updated| 2024-01-26 14:32:35 UTC |
|summary| The beverage industry is a typical food processing industry accounts forsignificant energy consumption and has flexible demands. However thedeployment of energy flexibility in the beverage industry is complex andchallenging. Furthermore activation of energy flexibility from the wholebrewery industry is necessary to ensure grid stability. Therefore this paperassesses the energy flexibility potential of Denmarks brewery sector based ona multi-agent-based simulation. 239 individual brewery facilities aresimulated and each facility as an agent can interact with the energy systemmarket and make decisions based on its underlying parameters and operationalrestrictions. The results show that the Danish breweries could save 1.56  ofelectricity costs annually while maintaining operational security and reducingapproximately 1745 tonnes of CO2 emissions. Furthermore medium-size breweriescould obtain higher relative benefits by providing energy flexibilityespecially those producing lager and ale. The result also shows that thebreweries relative saving potential is electricity market-dependent. |


| Item |Content|
| --- |---|
|idx| 2401.14825v1 |
|title| Keeping the Harmony Between Neighbors: Local Fairness in Graph Fair Division |
|authors| Halvard HummelAyumi Igarashi
|links| http://arxiv.org/abs/2401.14825v1 |
|updated| 2024-01-26 12:52:49 UTC |
|summary| We study the problem of allocating indivisible resources under theconnectivity constraints of a graph G. This model initially introduced byBouveret et al. published in IJCAI 2017 effectively encompasses a diversearray of scenarios characterized by spatial or temporal limitations includingthe division of land plots and the allocation of time plots. In this paper weintroduce a novel fairness concept that integrates local comparisons within thesocial network formed by a connected allocation of the item graph. Ourparticular focus is to achieve pairwise-maximin fair share PMMS among theneighbors within this network. For any underlying graph structure we showthat a connected allocation that maximizes Nash welfare guarantees a1/2-PMMS fairness. Moreover for two agents we establish that a3/4-PMMS allocation can be efficiently computed. Additionally wedemonstrate that for three agents and the items aligned on a path a PMMSallocation is always attainable and can be computed in polynomial time. Lastlywhen agents have identical additive utilities we present apseudo-polynomial-time algorithm for a 3/4-PMMS allocation irrespective ofthe underlying graph G. Furthermore we provide a polynomial-time algorithmfor obtaining a PMMS allocation when G is a tree. |


