# cs.CL 

| Item |Content|
| --- |---|
|idx| 2401.18070v1 |
|title| Do Language Models Exhibit the Same Cognitive Biases in Problem Solving as Human Learners? |
|authors| Andreas OpedalAlessandro StolfoHaruki ShirakamiYing JiaoRyan CotterellBernhard SchölkopfAbulhair SaparovMrinmaya Sachan
|links| http://arxiv.org/abs/2401.18070v1 |
|updated| 2024-01-31 18:48:20 UTC |
|summary| There is increasing interest in employing large language models LLMs ascognitive models. For such purposes it is central to understand whichcognitive properties are well-modeled by LLMs and which are not. In this workwe study the biases of LLMs in relation to those known in children when solvingarithmetic word problems. Surveying the learning science literature we positthat the problem-solving process can be split into three distinct steps: textcomprehension solution planning and solution execution. We construct tests foreach one in order to understand which parts of this process can be faithfullymodeled by current state-of-the-art LLMs. We generate a novel set of wordproblems for each of these tests using a neuro-symbolic method that enablesfine-grained control over the problem features. We find evidence that LLMswith and without instruction-tuning exhibit human-like biases in both thetext-comprehension and the solution-planning steps of the solving process butnot during the final step which relies on the problems arithmetic expressionssolution execution. |


| Item |Content|
| --- |---|
|idx| 2401.18059v1 |
|title| RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval |
|authors| Parth SarthiSalman AbdullahAditi TuliShubh KhannaAnna GoldieChristopher D. Manning
|links| http://arxiv.org/abs/2401.18059v1 |
|updated| 2024-01-31 18:30:21 UTC |
|summary| Retrieval-augmented language models can better adapt to changes in worldstate and incorporate long-tail knowledge. However most existing methodsretrieve only short contiguous chunks from a retrieval corpus limitingholistic understanding of the overall document context. We introduce the novelapproach of recursively embedding clustering and summarizing chunks of textconstructing a tree with differing levels of summarization from the bottom up.At inference time our RAPTOR model retrieves from this tree integratinginformation across lengthy documents at different levels of abstraction.Controlled experiments show that retrieval with recursive summaries offerssignificant improvements over traditional retrieval-augmented LMs on severaltasks. On question-answering tasks that involve complex multi-step reasoningwe show state-of-the-art results for example by coupling RAPTOR retrievalwith the use of GPT-4 we can improve the best performance on the QuALITYbenchmark by 20 in absolute accuracy. |


| Item |Content|
| --- |---|
|idx| 2401.18058v1 |
|title| LongAlign: A Recipe for Long Context Alignment of Large Language Models |
|authors| Yushi BaiXin LvJiajie ZhangYuze HeJi QiLei HouJie TangYuxiao DongJuanzi Li
|links| http://arxiv.org/abs/2401.18058v1 |
|updated| 2024-01-31 18:29:39 UTC |
|summary| Extending large language models to effectively handle long contexts requiresinstruction fine-tuning on input sequences of similar length. To address thiswe present LongAlign -- a recipe of the instruction data training andevaluation for long context alignment. First we construct a longinstruction-following dataset using Self-Instruct. To ensure the datadiversity it covers a broad range of tasks from various long context sources.Second we adopt the packing and sorted batching strategies to speed upsupervised fine-tuning on data with varied length distributions. Additionallywe develop a loss weighting method to balance the contribution to the lossacross different sequences during packing training. Third we introduce theLongBench-Chat benchmark for evaluating instruction-following capabilities onqueries of 10k-100k in length. Experiments show that LongAlign outperformsexisting recipes for LLMs in long context tasks by up to 30 while alsomaintaining their proficiency in handling short generic tasks. The code dataand long-aligned models are open-sourced at https://github.com/THUDM/LongAlign. |


| Item |Content|
| --- |---|
|idx| 2401.18046v1 |
|title| Multipath parsing in the brain |
|authors| Berta FranzluebbersDonald DunaganMiloš StanojevićJan BuysJohn T. Hale
|links| http://arxiv.org/abs/2401.18046v1 |
|updated| 2024-01-31 18:07:12 UTC |
|summary| Humans understand sentences word-by-word in the order that they hear them.This incrementality entails resolving temporary ambiguities about syntacticrelationships. We investigate how humans process these syntactic ambiguities bycorrelating predictions from incremental generative dependency parsers withtimecourse data from people undergoing functional neuroimaging while listeningto an audiobook. In particular we compare competing hypotheses regarding thenumber of developing syntactic analyses in play during word-by-wordcomprehension: one vs more than one. This comparison involves evaluatingsyntactic surprisal from a state-of-the-art dependency parser with LLM-adaptedencodings against an existing fMRI dataset. In both English and Chinese datawe find evidence for multipath parsing. Brain regions associated with thismultipath effect include bilateral superior temporal gyrus. |


| Item |Content|
| --- |---|
|idx| 2401.18045v1 |
|title| SpeechComposer: Unifying Multiple Speech Tasks with Prompt Composition |
|authors| Yihan WuSoumi MaitiYifan PengWangyou ZhangChenda LiYuyue WangXihua WangShinji WatanabeRuihua Song
|links| http://arxiv.org/abs/2401.18045v1 |
|updated| 2024-01-31 18:06:29 UTC |
|summary| Recent advancements in language models have significantly enhancedperformance in multiple speech-related tasks. Existing speech language modelstypically utilize task-dependent prompt tokens to unify various speech tasks ina single model. However this design omits the intrinsic connections betweendifferent speech tasks which can potentially boost the performance of eachtask. In this work we propose a novel decoder-only speech language modelSpeechComposer that can unify common speech tasks by composing a fixed set ofprompt tokens. Built upon four primary tasks -- speech synthesis speechrecognition speech language modeling and text language modeling --SpeechComposer can easily extend to more speech tasks via compositions ofwell-designed prompt tokens like voice conversion and speech enhancement. Theunification of prompt tokens also makes it possible for knowledge sharing amongdifferent speech tasks in a more structured manner. Experimental resultsdemonstrate that our proposed SpeechComposer can improve the performance ofboth primary tasks and composite tasks showing the effectiveness of the sharedprompt tokens. Remarkably the unified decoder-only model achieves a comparableand even better performance than the baselines which are expert models designedfor single tasks. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2401.18070v1 |
|title| Do Language Models Exhibit the Same Cognitive Biases in Problem Solving as Human Learners? |
|authors| Andreas OpedalAlessandro StolfoHaruki ShirakamiYing JiaoRyan CotterellBernhard SchölkopfAbulhair SaparovMrinmaya Sachan
|links| http://arxiv.org/abs/2401.18070v1 |
|updated| 2024-01-31 18:48:20 UTC |
|summary| There is increasing interest in employing large language models LLMs ascognitive models. For such purposes it is central to understand whichcognitive properties are well-modeled by LLMs and which are not. In this workwe study the biases of LLMs in relation to those known in children when solvingarithmetic word problems. Surveying the learning science literature we positthat the problem-solving process can be split into three distinct steps: textcomprehension solution planning and solution execution. We construct tests foreach one in order to understand which parts of this process can be faithfullymodeled by current state-of-the-art LLMs. We generate a novel set of wordproblems for each of these tests using a neuro-symbolic method that enablesfine-grained control over the problem features. We find evidence that LLMswith and without instruction-tuning exhibit human-like biases in both thetext-comprehension and the solution-planning steps of the solving process butnot during the final step which relies on the problems arithmetic expressionssolution execution. |


| Item |Content|
| --- |---|
|idx| 2401.18045v1 |
|title| SpeechComposer: Unifying Multiple Speech Tasks with Prompt Composition |
|authors| Yihan WuSoumi MaitiYifan PengWangyou ZhangChenda LiYuyue WangXihua WangShinji WatanabeRuihua Song
|links| http://arxiv.org/abs/2401.18045v1 |
|updated| 2024-01-31 18:06:29 UTC |
|summary| Recent advancements in language models have significantly enhancedperformance in multiple speech-related tasks. Existing speech language modelstypically utilize task-dependent prompt tokens to unify various speech tasks ina single model. However this design omits the intrinsic connections betweendifferent speech tasks which can potentially boost the performance of eachtask. In this work we propose a novel decoder-only speech language modelSpeechComposer that can unify common speech tasks by composing a fixed set ofprompt tokens. Built upon four primary tasks -- speech synthesis speechrecognition speech language modeling and text language modeling --SpeechComposer can easily extend to more speech tasks via compositions ofwell-designed prompt tokens like voice conversion and speech enhancement. Theunification of prompt tokens also makes it possible for knowledge sharing amongdifferent speech tasks in a more structured manner. Experimental resultsdemonstrate that our proposed SpeechComposer can improve the performance ofboth primary tasks and composite tasks showing the effectiveness of the sharedprompt tokens. Remarkably the unified decoder-only model achieves a comparableand even better performance than the baselines which are expert models designedfor single tasks. |


| Item |Content|
| --- |---|
|idx| 2401.18040v1 |
|title| Enhancing End-to-End Multi-Task Dialogue Systems: A Study on Intrinsic Motivation Reinforcement Learning Algorithms for Improved Training and Adaptability |
|authors| Navin KamuniHardik ShahSathishkumar ChintalaNaveen KunchakuriSujatha Alla Old Dominion
|links| http://arxiv.org/abs/2401.18040v1 |
|updated| 2024-01-31 18:03:39 UTC |
|summary| End-to-end multi-task dialogue systems are usually designed with separatemodules for the dialogue pipeline. Among these the policy module is essentialfor deciding what to do in response to user input. This policy is trained byreinforcement learning algorithms by taking advantage of an environment inwhich an agent receives feedback in the form of a reward signal. The currentdialogue systems however only provide meagre and simplistic rewards.Investigating intrinsic motivation reinforcement learning algorithms is thegoal of this study. Through this the agent can quickly accelerate training andimprove its capacity to judge the quality of its actions by teaching it aninternal incentive system. In particular we adapt techniques for randomnetwork distillation and curiosity-driven reinforcement learning to measure thefrequency of state visits and encourage exploration by using semanticsimilarity between utterances. Experimental results on MultiWOZ aheterogeneous dataset show that intrinsic motivation-based debate systemsoutperform policies that depend on extrinsic incentives. By adopting randomnetwork distillation for example which is trained using semantic similaritybetween user-system dialogues an astounding average success rate of 73 isachieved. This is a significant improvement over the baseline Proximal PolicyOptimization PPO which has an average success rate of 60. In additionperformance indicators such as booking rates and completion rates show a 10rise over the baseline. Furthermore these intrinsic incentive models helpimprove the systems policys resilience in an increasing amount of domains.This implies that they could be useful in scaling up to settings that cover awider range of domains. |


| Item |Content|
| --- |---|
|idx| 2401.18034v1 |
|title| Paramanu: A Family of Novel Efficient Indic Generative Foundation Language Models |
|authors| Mitodru NiyogiArnab Bhattacharya
|links| http://arxiv.org/abs/2401.18034v1 |
|updated| 2024-01-31 17:58:10 UTC |
|summary| We present Gyan AI Paramanu atom a family of novel language models forIndian languages. It is a collection of auto-regressive monolingual bilingualand multilingual Indic language models pretrained from scratch on a single GPUfor 10 Indian languages Assamese Bangla Hindi Konkani Maithili MarathiOdia Sanskrit Tamil Telugu across 5 scripts Bangla Devanagari OdiaTamil Telugu of varying sizes ranging from 13.29M to 367.5M.The models arepretrained with a context size of 1024 on a single GPU. The models are veryefficient small fast and powerful. We have also developed an efficient mostadvanced Indic tokenizer that can even tokenize unseen languages. In order toavoid the curse of multi-linguality in our multilingual mParamanu model wepretrained on comparable corpora by typological grouping using the same script.We performed human evaluation of our pretrained models for open end textgeneration on grammar coherence creativity and factuality metrics forBangla Hindi and Sanskrit. Our Bangla Hindi and Sanskrit modelsoutperformed GPT-3.5-Turbo ChatGPT Bloom 7B LLaMa-2 7B OPT 6.7B GPT-J 6BGPTNeo 1.3B GPT2-XL large language models LLMs by a large margin despitebeing smaller in size by 66 to 20 times compared to standard 7B LLMs. To runinference on our pretrained models CPU is enough and GPU is not needed. Wealso instruction-tuned our pretrained Bangla Hindi Marathi Tamil and Telugumodels on 23k instructions in respective languages. Our pretrained andinstruction-tuned models which are first of its kind most powerful efficientsmall generative language models ever developed for Indic languages and thevarious results lead to the conclusion that high quality generative languagemodels are possible without high amount of compute power and humongous numberof parameters. We plan to release our models at https://www.bharatgpts.com. |


| Item |Content|
| --- |---|
|idx| 2401.18028v1 |
|title| Supporting Anticipatory Governance using LLMs: Evaluating and Aligning Large Language Models with the News Media to Anticipate the Negative Impacts of AI |
|authors| Mowafak AllahamNicholas Diakopoulos
|links| http://arxiv.org/abs/2401.18028v1 |
|updated| 2024-01-31 17:43:04 UTC |
|summary| Anticipating the negative impacts of emerging AI technologies is a challengeespecially in the early stages of development. An understudied approach to suchanticipation is the use of LLMs to enhance and guide this process. Despiteadvancements in LLMs and evaluation metrics to account for biases in generatedtext it is unclear how well these models perform in anticipatory tasks.Specifically the use of LLMs to anticipate AI impacts raises questions aboutthe quality and range of categories of negative impacts these models arecapable of generating. In this paper we leverage news media a diverse datasource that is rich with normative assessments of emerging technologies toformulate a taxonomy of impacts to act as a baseline for comparing against. Bycomputationally analyzing thousands of news articles published by hundreds ofonline news domains around the world we develop a taxonomy consisting of tencategories of AI impacts. We then evaluate both instruction-based GPT-4 andMistral-7B-Instruct and fine-tuned completion models Mistral-7B and GPT-3using a sample from this baseline. We find that the generated impacts usingMistral-7B fine-tuned on impacts from the news media tend to be qualitativelyon par with impacts generated using a larger scale model such as GPT-4.Moreover we find that these LLMs generate impacts that largely reflect thetaxonomy of negative impacts identified in the news media however the impactsproduced by instruction-based models had gaps in the production of certaincategories of impacts in comparison to fine-tuned models. This researchhighlights a potential bias in state-of-the-art LLMs when used for anticipatingimpacts and demonstrates the advantages of aligning smaller LLMs with a diverserange of impacts such as those reflected in the news media to better reflectsuch impacts during anticipatory exercises. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2401.18079v1 |
|title| KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization |
|authors| Coleman HooperSehoon KimHiva MohammadzadehMichael W. MahoneyYakun Sophia ShaoKurt KeutzerAmir Gholami
|links| http://arxiv.org/abs/2401.18079v1 |
|updated| 2024-01-31 18:58:14 UTC |
|summary| LLMs are seeing growing use for applications such as document analysis andsummarization which require large context windows and with these large contextwindows KV cache activations surface as the dominant contributor to memoryconsumption during inference. Quantization is a promising approach forcompressing KV cache activations however existing solutions fail to representactivations accurately in ultra-low precisions such as sub-4-bit. In thiswork we present KVQuant which addresses this problem by incorporating novelmethods for quantizing cached KV activations including: i Per-Channel KeyQuantization where we adjust the dimension along which we quantize the Keyactivations to better match the distribution ii Pre-RoPE Key Quantizationwhere we quantize Key activations before the rotary positional embedding tomitigate its impact on quantization iii Non-Uniform KV Cache Quantizationwhere we derive per-layer sensitivity-weighted non-uniform datatypes thatbetter represent the distributions iv Per-Vector Dense-and-SparseQuantization where we isolate outliers separately for each vector to minimizeskews in quantization ranges and v Q-Norm where we normalize quantizationcentroids in order to mitigate distribution shift providing additionalbenefits for 2-bit quantization. By applying our method to the LLaMA LLaMA-2and Mistral models we achieve 0.1 perplexity degradation with 3-bitquantization on both Wikitext-2 and C4 outperforming existing approaches. Ourmethod enables serving the LLaMA-7B model with a context length of up to 1million on a single A100-80GB GPU and up to 10 million on an 8-GPU system. |


| Item |Content|
| --- |---|
|idx| 2401.18070v1 |
|title| Do Language Models Exhibit the Same Cognitive Biases in Problem Solving as Human Learners? |
|authors| Andreas OpedalAlessandro StolfoHaruki ShirakamiYing JiaoRyan CotterellBernhard SchölkopfAbulhair SaparovMrinmaya Sachan
|links| http://arxiv.org/abs/2401.18070v1 |
|updated| 2024-01-31 18:48:20 UTC |
|summary| There is increasing interest in employing large language models LLMs ascognitive models. For such purposes it is central to understand whichcognitive properties are well-modeled by LLMs and which are not. In this workwe study the biases of LLMs in relation to those known in children when solvingarithmetic word problems. Surveying the learning science literature we positthat the problem-solving process can be split into three distinct steps: textcomprehension solution planning and solution execution. We construct tests foreach one in order to understand which parts of this process can be faithfullymodeled by current state-of-the-art LLMs. We generate a novel set of wordproblems for each of these tests using a neuro-symbolic method that enablesfine-grained control over the problem features. We find evidence that LLMswith and without instruction-tuning exhibit human-like biases in both thetext-comprehension and the solution-planning steps of the solving process butnot during the final step which relies on the problems arithmetic expressionssolution execution. |


| Item |Content|
| --- |---|
|idx| 2401.18059v1 |
|title| RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval |
|authors| Parth SarthiSalman AbdullahAditi TuliShubh KhannaAnna GoldieChristopher D. Manning
|links| http://arxiv.org/abs/2401.18059v1 |
|updated| 2024-01-31 18:30:21 UTC |
|summary| Retrieval-augmented language models can better adapt to changes in worldstate and incorporate long-tail knowledge. However most existing methodsretrieve only short contiguous chunks from a retrieval corpus limitingholistic understanding of the overall document context. We introduce the novelapproach of recursively embedding clustering and summarizing chunks of textconstructing a tree with differing levels of summarization from the bottom up.At inference time our RAPTOR model retrieves from this tree integratinginformation across lengthy documents at different levels of abstraction.Controlled experiments show that retrieval with recursive summaries offerssignificant improvements over traditional retrieval-augmented LMs on severaltasks. On question-answering tasks that involve complex multi-step reasoningwe show state-of-the-art results for example by coupling RAPTOR retrievalwith the use of GPT-4 we can improve the best performance on the QuALITYbenchmark by 20 in absolute accuracy. |


| Item |Content|
| --- |---|
|idx| 2401.18058v1 |
|title| LongAlign: A Recipe for Long Context Alignment of Large Language Models |
|authors| Yushi BaiXin LvJiajie ZhangYuze HeJi QiLei HouJie TangYuxiao DongJuanzi Li
|links| http://arxiv.org/abs/2401.18058v1 |
|updated| 2024-01-31 18:29:39 UTC |
|summary| Extending large language models to effectively handle long contexts requiresinstruction fine-tuning on input sequences of similar length. To address thiswe present LongAlign -- a recipe of the instruction data training andevaluation for long context alignment. First we construct a longinstruction-following dataset using Self-Instruct. To ensure the datadiversity it covers a broad range of tasks from various long context sources.Second we adopt the packing and sorted batching strategies to speed upsupervised fine-tuning on data with varied length distributions. Additionallywe develop a loss weighting method to balance the contribution to the lossacross different sequences during packing training. Third we introduce theLongBench-Chat benchmark for evaluating instruction-following capabilities onqueries of 10k-100k in length. Experiments show that LongAlign outperformsexisting recipes for LLMs in long context tasks by up to 30 while alsomaintaining their proficiency in handling short generic tasks. The code dataand long-aligned models are open-sourced at https://github.com/THUDM/LongAlign. |


| Item |Content|
| --- |---|
|idx| 2401.18057v1 |
|title| Rank Supervised Contrastive Learning for Time Series Classification |
|authors| Qianying RenDongsheng LuoDongjin Song
|links| http://arxiv.org/abs/2401.18057v1 |
|updated| 2024-01-31 18:29:10 UTC |
|summary| Recently various contrastive learning techniques have been developed tocategorize time series data and exhibit promising performance. A generalparadigm is to utilize appropriate augmentations and construct feasiblepositive samples such that the encoder can yield robust and discriminativerepresentations by mapping similar data points closer together in the featurespace while pushing dissimilar data points farther apart. Despite its efficacythe fine-grained relative similarity e.g. rank information of positivesamples is largely ignored especially when labeled samples are limited. Tothis end we present Rank Supervised Contrastive Learning RankSCL to performtime series classification. Different from conventional contrastive learningframeworks RankSCL augments raw data in a targeted way in the embedding spaceand adopts certain filtering rules to select more informative positive andnegative pairs of samples. Moreover a novel rank loss is developed to assigndifferent weights for different levels of positive samples enable the encoderto extract the fine-grained information of the same class and produce a clearboundary among different classes. Thoroughly empirical studies on 128 UCRdatasets and 30 UEA datasets demonstrate that the proposed RankSCL can achievestate-of-the-art performance compared to existing baseline methods. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2401.18085v1 |
|title| Motion Guidance: Diffusion-Based Image Editing with Differentiable Motion Estimators |
|authors| Daniel GengAndrew Owens
|links| http://arxiv.org/abs/2401.18085v1 |
|updated| 2024-01-31 18:59:59 UTC |
|summary| Diffusion models are capable of generating impressive images conditioned ontext descriptions and extensions of these models allow users to edit images ata relatively coarse scale. However the ability to precisely edit the layoutposition pose and shape of objects in images with diffusion models is stilldifficult. To this end we propose motion guidance a zero-shot technique thatallows a user to specify dense complex motion fields that indicate where eachpixel in an image should move. Motion guidance works by steering the diffusionsampling process with the gradients through an off-the-shelf optical flownetwork. Specifically we design a guidance loss that encourages the sample tohave the desired motion as estimated by a flow network while also beingvisually similar to the source image. By simultaneously sampling from adiffusion model and guiding the sample to have low guidance loss we can obtaina motion-edited image. We demonstrate that our technique works on complexmotions and produces high quality edits of real and generated images. |


| Item |Content|
| --- |---|
|idx| 2401.18084v1 |
|title| Binding Touch to Everything: Learning Unified Multimodal Tactile Representations |
|authors| Fengyu YangChao FengZiyang ChenHyoungseob ParkDaniel WangYiming DouZiyao ZengXien ChenRit GangopadhyayAndrew OwensAlex Wong
|links| http://arxiv.org/abs/2401.18084v1 |
|updated| 2024-01-31 18:59:57 UTC |
|summary| The ability to associate touch with other modalities has huge implicationsfor humans and computational systems. However multimodal learning with touchremains challenging due to the expensive data collection process andnon-standardized sensor outputs. We introduce UniTouch a unified tactile modelfor vision-based touch sensors connected to multiple modalities includingvision language and sound. We achieve this by aligning our UniTouchembeddings to pretrained image embeddings already associated with a variety ofother modalities. We further propose learnable sensor-specific tokens allowingthe model to learn from a set of heterogeneous tactile sensors all at the sametime. UniTouch is capable of conducting various touch sensing tasks in thezero-shot setting from robot grasping prediction to touch image questionanswering. To the best of our knowledge UniTouch is the first to demonstratesuch capabilities. Project page: https://cfeng16.github.io/UniTouch/ |


| Item |Content|
| --- |---|
|idx| 2401.18083v1 |
|title| Improved Scene Landmark Detection for Camera Localization |
|authors| Tien DoSudipta N. Sinha
|links| http://arxiv.org/abs/2401.18083v1 |
|updated| 2024-01-31 18:59:12 UTC |
|summary| Camera localization methods based on retrieval local feature matching and3D structure-based pose estimation are accurate but require high storage areslow and are not privacy-preserving. A method based on scene landmarkdetection SLD was recently proposed to address these limitations. It involvestraining a convolutional neural network CNN to detect a few predeterminedsalient scene-specific 3D points or landmarks and computing camera pose fromthe associated 2D-3D correspondences. Although SLD outperformed existinglearning-based approaches it was notably less accurate than 3D structure-basedmethods. In this paper we show that the accuracy gap was due to insufficientmodel capacity and noisy labels during training. To mitigate the capacityissue we propose to split the landmarks into subgroups and train a separatenetwork for each subgroup. To generate better training labels we propose usingdense reconstructions to estimate visibility of scene landmarks. Finally wepresent a compact architecture to improve memory efficiency. Accuracy wise ourapproach is on par with state of the art structure based methods on theINDOOR-6 dataset but runs significantly faster and uses less storage. Code andmodels can be found at https://github.com/microsoft/SceneLandmarkLocalization. |


| Item |Content|
| --- |---|
|idx| 2401.18075v1 |
|title| CARFF: Conditional Auto-encoded Radiance Field for 3D Scene Forecasting |
|authors| Jiezhi YangKhushi DesaiCharles PackerHarshil BhatiaNicholas RhinehartRowan McAllisterJoseph Gonzalez
|links| http://arxiv.org/abs/2401.18075v1 |
|updated| 2024-01-31 18:56:09 UTC |
|summary| We propose CARFF: Conditional Auto-encoded Radiance Field for 3D SceneForecasting a method for predicting future 3D scenes given past observationssuch as 2D ego-centric images. Our method maps an image to a distribution overplausible 3D latent scene configurations using a probabilistic encoder andpredicts the evolution of the hypothesized scenes through time. Our latentscene representation conditions a global Neural Radiance Field NeRF torepresent a 3D scene model which enables explainable predictions andstraightforward downstream applications. This approach extends beyond previousneural rendering work by considering complex scenarios of uncertainty inenvironmental states and dynamics. We employ a two-stage training ofPose-Conditional-VAE and NeRF to learn 3D representations. Additionally weauto-regressively predict latent scene representations as a partiallyobservable Markov decision process utilizing a mixture density network. Wedemonstrate the utility of our method in realistic scenarios using the CARLAdriving simulator where CARFF can be used to enable efficient trajectory andcontingency planning in complex multi-agent autonomous driving scenariosinvolving visual occlusions. |


| Item |Content|
| --- |---|
|idx| 2401.18054v1 |
|title| Benchmarking Sensitivity of Continual Graph Learning for Skeleton-Based Action Recognition |
|authors| Wei WeiTom De SchepperKevin Mets
|links| http://arxiv.org/abs/2401.18054v1 |
|updated| 2024-01-31 18:20:42 UTC |
|summary| Continual learning CL is the research field that aims to build machinelearning models that can accumulate knowledge continuously over different taskswithout retraining from scratch. Previous studies have shown that pre-traininggraph neural networks GNN may lead to negative transfer Hu et al. 2020after fine-tuning a setting which is closely related to CL. Thus we focus onstudying GNN in the continual graph learning CGL setting. We propose thefirst continual graph learning benchmark for spatio-temporal graphs and use itto benchmark well-known CGL methods in this novel setting. The benchmark isbased on the N-UCLA and NTU-RGBD datasets for skeleton-based actionrecognition. Beyond benchmarking for standard performance metrics we study theclass and task-order sensitivity of CGL methods i.e. the impact of learningorder on each class/tasks performance and the architectural sensitivity ofCGL methods with backbone GNN at various widths and depths. We reveal thattask-order robust methods can still be class-order sensitive and observeresults that contradict previous empirical observations on architecturalsensitivity in CL. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2401.18039v1 |
|title| Variable selection for Naïve Bayes classification |
|authors| Rafael BlanqueroEmilio CarrizosaPepa Ramírez-CoboM. Remedios Sillero-Denamiel
|links| http://dx.doi.org/10.1016/j.cor.2021.105456 |
|updated| 2024-01-31 18:01:36 UTC |
|summary| The Naive Bayes has proven to be a tractable and efficient method forclassification in multivariate analysis. However features are usuallycorrelated a fact that violates the Naive Bayes assumption of conditionalindependence and may deteriorate the methods performance. Moreover datasetsare often characterized by a large number of features which may complicate theinterpretation of the results as well as slow down the methods execution.  In this paper we propose a sparse version of the Naive Bayes classifierthat is characterized by three properties. First the sparsity is achievedtaking into account the correlation structure of the covariates. Seconddifferent performance measures can be used to guide the selection of features.Third performance constraints on groups of higher interest can be included.Our proposal leads to a smart search which yields competitive running timeswhereas the flexibility in terms of performance measure for classification isintegrated. Our findings show that when compared against well-referencedfeature selection approaches the proposed sparse Naive Bayes obtainscompetitive results regarding accuracy sparsity and running times for balanceddatasets. In the case of datasets with unbalanced or with differentimportance classes a better compromise between classification rates for thedifferent classes is achieved. |


| Item |Content|
| --- |---|
|idx| 2401.18023v1 |
|title| A cost-sensitive constrained Lasso |
|authors| Rafael BlanqueroEmilio CarrizosaPepa Ramírez-CoboM. Remedios Sillero-Denamiel
|links| http://dx.doi.org/10.1007/s11634-020-00389-5 |
|updated| 2024-01-31 17:36:21 UTC |
|summary| The Lasso has become a benchmark data analysis procedure and numerousvariants have been proposed in the literature. Although the Lasso formulationsare stated so that overall prediction error is optimized no full control overthe accuracy prediction on certain individuals of interest is allowed. In thiswork we propose a novel version of the Lasso in which quadratic performanceconstraints are added to Lasso-based objective functions in such a way thatthreshold values are set to bound the prediction errors in the different groupsof interest not necessarily disjoint. As a result a constrained sparseregression model is defined by a nonlinear optimization problem. Thiscost-sensitive constrained Lasso has a direct application in heterogeneoussamples where data are collected from distinct sources as it is standard inmany biomedical contexts. Both theoretical properties and empirical studiesconcerning the new method are explored in this paper. In addition twoillustrations of the method on biomedical and sociological contexts areconsidered. |


| Item |Content|
| --- |---|
|idx| 2401.18017v1 |
|title| Causal Discovery by Kernel Deviance Measures with Heterogeneous Transforms |
|authors| Tim TseZhitang ChenShengyu ZhuYue Liu
|links| http://arxiv.org/abs/2401.18017v1 |
|updated| 2024-01-31 17:28:05 UTC |
|summary| The discovery of causal relationships in a set of random variables is afundamental objective of science and has also recently been argued as being anessential component towards real machine intelligence. One class of causaldiscovery techniques are founded based on the argument that there are inherentstructural asymmetries between the causal and anti-causal direction which couldbe leveraged in determining the direction of causation. To go about capturingthese discrepancies between cause and effect remains to be a challenge and manycurrent state-of-the-art algorithms propose to compare the norms of the kernelmean embeddings of the conditional distributions. In this work we argue thatsuch approaches based on RKHS embeddings are insufficient in capturingprincipal markers of cause-effect asymmetry involving higher-order structuralvariabilities of the conditional distributions. We propose Kernel IntrinsicInvariance Measure with Heterogeneous Transform KIIM-HT which introduces anovel score measure based on heterogeneous transformation of RKHS embeddings toextract relevant higher-order moments of the conditional densities for causaldiscovery. Inference is made via comparing the score of each hypotheticalcause-effect direction. Tests and comparisons on a synthetic dataset atwo-dimensional synthetic dataset and the real-world benchmark datasetTubingen Cause-Effect Pairs verify our approach. In addition we conduct asensitivity analysis to the regularization parameter to faithfully compareprevious work to our method and an experiment with trials on variedhyperparameter values to showcase the robustness of our algorithm. |


| Item |Content|
| --- |---|
|idx| 2401.18012v1 |
|title| Causal Coordinated Concurrent Reinforcement Learning |
|authors| Tim TseIsaac ChanZhitang Chen
|links| http://arxiv.org/abs/2401.18012v1 |
|updated| 2024-01-31 17:20:28 UTC |
|summary| In this work we propose a novel algorithmic framework for data sharing andcoordinated exploration for the purpose of learning more data-efficient andbetter performing policies under a concurrent reinforcement learning CRLsetting. In contrast to other work which make the assumption that all agentsact under identical environments we relax this restriction and insteadconsider the formulation where each agent acts within an environment whichshares a global structure but also exhibits individual variations. Ouralgorithm leverages a causal inference algorithm in the form of Additive NoiseModel - Mixture Model ANM-MM in extracting model parameters governingindividual differentials via independence enforcement. We propose a new datasharing scheme based on a similarity measure of the extracted model parametersand demonstrate superior learning speeds on a set of autoregressive pendulumand cart-pole swing-up tasks and finally we show the effectiveness of diverseaction selection between common agents under a sparse reward setting. To thebest of our knowledge this is the first work in considering non-identicalenvironments in CRL and one of the few works which seek to integrate causalinference with reinforcement learning RL. |


| Item |Content|
| --- |---|
|idx| 2401.17958v1 |
|title| Convergence Analysis for General Probability Flow ODEs of Diffusion Models in Wasserstein Distances |
|authors| Xuefeng GaoLingjiong Zhu
|links| http://arxiv.org/abs/2401.17958v1 |
|updated| 2024-01-31 16:07:44 UTC |
|summary| Score-based generative modeling with probability flow ordinary differentialequations ODEs has achieved remarkable success in a variety of applications.While various fast ODE-based samplers have been proposed in the literature andemployed in practice the theoretical understandings about convergenceproperties of the probability flow ODE are still quite limited. In this paperwe provide the first non-asymptotic convergence analysis for a general class ofprobability flow ODE samplers in 2-Wasserstein distance assuming accuratescore estimates. We then consider various examples and establish results on theiteration complexity of the corresponding ODE-based samplers. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2401.18013v1 |
|title| On The Power of Subtle Expressive Cues in the Perception of Human Affects |
|authors| Ezgi DedeKamile Asli AgilonuErgun AklemanMetin Sezgin
|links| http://arxiv.org/abs/2401.18013v1 |
|updated| 2024-01-31 17:20:36 UTC |
|summary| In this study we introduce a sketch-based method for testing how subtleexpressive cues influence the perception of affect in illustrations of humanfigures. We specifically study the impact of human posture and gaze directionimplicitly specified through nose orientation on perceived emotions and mood.Through a series of user studies using sketchy illustrations of a runningfigure where a professional illustrator manipulated gaze direction throughadjustments on the nose orientation we found that this simple change resultedin a diverse range of perceived affects spanning from fear to concern andwonder. These findings shed light on the importance of fine details in definingcontext for context-aware system designs and underscore the importance ofrecognizing and expressing affect. Understanding minor expressive cues iscrucial to developing emotionally intelligent systems capable of expressingaffect. |


| Item |Content|
| --- |---|
|idx| 2401.17929v1 |
|title| Technological Shocks and Algorithmic Decision Aids in Credence Goods Markets |
|authors| Alexander ErleiLukas Meub
|links| http://arxiv.org/abs/2401.17929v1 |
|updated| 2024-01-31 15:41:37 UTC |
|summary| In credence goods markets such as health care or repair services consumersrely on experts with superior information to adequately diagnose and treatthem. Experts however are constrained in their diagnostic abilities whichhurts market efficiency and consumer welfare. Technological breakthroughs thatsubstitute or complement expert judgments have the potential to alleviateconsumer mistreatment. This article studies how competitive experts adopt noveldiagnostic technologies when skills are heterogeneously distributed andobfuscated to consumers. We differentiate between novel technologies thatincrease expert abilities and algorithmic decision aids that complement expertjudgments but do not affect an experts personal diagnostic precision. We showthat high-ability experts may be incentivized to forego the decision aid inorder to escape a pooling equilibrium by differentiating themselves fromlow-ability experts. Results from an online experiment support our hypothesisshowing that high-ability experts are significantly less likely thanlow-ability experts to invest into an algorithmic decision aid. Furthermore wedocument pervasive under-investments and no effect on expert honesty. |


| Item |Content|
| --- |---|
|idx| 2401.17866v1 |
|title| Making Sense of Knowledge Intensive Processes: an Oil & Gas Industry Scenario |
|authors| Juliana Jansen FerreiraVinícius SeguraAna FucsRogério de Paula
|links| http://arxiv.org/abs/2401.17866v1 |
|updated| 2024-01-31 14:25:05 UTC |
|summary| Sensemaking is a constant and ongoing process by which people associatemeaning to experiences. It can be an individual process known as abduction ora group process by which people give meaning to collective experiences. Thesensemaking of a group is influenced by the abduction process of each personabout the experience. Every collaborative process needs some level ofsensemaking to show results. For a knowledge intensive process sensemaking iscentral and related to most of its tasks. We present findings from a fieldworkexecuted in knowledge intensive process from the Oil and Gas industry. Ourfindings indicated that different types of knowledge can be combined to composethe result of a sensemaking process e.g. decision the need for morediscussion etc.. This paper presents an initial set of knowledge types thatcan be combined to compose the result of the sensemaking of a collaborativedecision making process. We also discuss ideas for using systems powered byArtificial Intelligence to support sensemaking processes. |


| Item |Content|
| --- |---|
|idx| 2401.17856v1 |
|title| Beyond Numbers: Creating Analogies to Enhance Data Comprehension and Communication with Generative AI |
|authors| Qing ChenWei ShuaiJiyao ZhangZhida SunNan Cao
|links| http://arxiv.org/abs/2401.17856v1 |
|updated| 2024-01-31 14:17:52 UTC |
|summary| Unfamiliar measurements usually hinder readers from grasping the scale of thenumerical data understanding the content and feeling engaged with thecontext. To enhance data comprehension and communication we leverage analogiesto bridge the gap between abstract data and familiar measurements. In thiswork we first conduct semi-structured interviews with design experts toidentify design problems and summarize design considerations. Then we collectan analogy dataset of 138 cases from various online sources. Based on thecollected dataset we characterize a design space for creating data analogies.Next we build a prototype system AnalogyMate that automatically suggestsdata analogies their corresponding design solutions and generated visualrepresentations powered by generative AI. The study results show the usefulnessof AnalogyMate in aiding the creation process of data analogies and theeffectiveness of data analogy in enhancing data comprehension andcommunication. |


| Item |Content|
| --- |---|
|idx| 2401.17855v1 |
|title| Network-based Topic Structure Visualization |
|authors| Yeseul JeonJina ParkIck Hoon JinDongjun Chungc
|links| http://arxiv.org/abs/2401.17855v1 |
|updated| 2024-01-31 14:17:00 UTC |
|summary| In the real world many topics are inter-correlated making it challenging toinvestigate their structure and relationships. Understanding the interplaybetween topics and their relevance can provide valuable insights forresearchers guiding their studies and informing the direction of research. Inthis paper we utilize the topic-words distribution obtained from topicmodels as item-response data to model the structure of topics using a latentspace item response model. By estimating the latent positions of topics basedon their distances toward words we can capture the underlying topic structureand reveal their relationships. Visualizing the latent positions of topics inEuclidean space allows for an intuitive understanding of their proximity andassociations. We interpret relationships among topics by characterizing eachtopic based on representative words selected using a newly proposed scoringscheme. Additionally we assess the maturity of topics by tracking their latentpositions using different word sets providing insights into the robustness oftopics. To demonstrate the effectiveness of our approach we analyze the topiccomposition of COVID-19 studies during the early stage of its emergence usingbiomedical literature in the PubMed database. The software and data used inthis paper are publicly available at https://github.com/jeon9677/gViz . |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2401.18065v1 |
|title| Game susceptibility, Correlation and Payoff capacity as a measure of Cooperative behavior in the thermodynamic limit of some Social dilemmas |
|authors| Rajdeep TahColin Benjamin
|links| http://arxiv.org/abs/2401.18065v1 |
|updated| 2024-01-31 18:41:22 UTC |
|summary| Analytically finding the origins of cooperative behavior in infinite-playergames is an exciting topic of current interest. Previously cooperativebehavior has been studied by considering game magnetization and individualplayers average payoff as indicators. This paper shows that gamesusceptibility correlation and payoff capacity can aid in understandingcooperative behavior in social dilemmas in the thermodynamic limit. In thispaper we compare three analytical methods i.e. Nash equilibrium mappingNEM Darwinian selection DS and Aggregate selection AS with anumerical-based method ABM via the game susceptibility correlation andpayoff capacity as indicators of cooperative behavior. AS and DS fail comparedto NEM and ABM by giving incorrect results for the indicators in question. Theresults obtained via NEM and ABM are in good agreement for all three indicatorsin question for both Hawk-Dove and the Public goods games. After comparing theresults obtained for all five indicators we see that individual playersaverage payoff and payoff capacity are the best indicators to study cooperativebehavior among players in the thermodynamic limit. This paper finds that NEMand ABM along with the selected indicators offer valuable insights intocooperative behavior in infinite-player games contributing to understandingsocial dilemmas in the thermodynamic limit. |


| Item |Content|
| --- |---|
|idx| 2401.18030v1 |
|title| Distributed fixed-point algorithms for dynamic convex optimization over decentralized and unbalanced wireless networks |
|authors| Navneet AgrawalRenato L. G. CavalcanteSlawomir Stanczak
|links| http://arxiv.org/abs/2401.18030v1 |
|updated| 2024-01-31 17:49:09 UTC |
|summary| We consider problems where agents in a network seek a common quantitymeasured independently and periodically by each agent through a localtime-varying process. Numerous solvers addressing such problems have beendeveloped in the past featuring various adaptations of the local processingand the consensus step. However existing solvers still lack support foradvanced techniques such as superiorization and over-the-air functioncomputation OTA-C. To address this limitation we introduce a comprehensiveframework for the analysis of distributed algorithms by characterizing themusing the quasi-Fejer type algorithms and an extensive communication model.Under weak assumptions we prove almost sure convergence of the algorithm to acommon estimate for all agents. Moreover we develop a specific class ofalgorithms within this framework to tackle distributed optimization problemswith time-varying objectives and assuming that a time-invariant solutionexists prove its convergence to a solution. We also present a novel OTA-Cprotocol for consensus step in large decentralized networks reducingcommunication overhead and enhancing network autonomy as compared to theexisting protocols. The effectiveness of the algorithm featuringsuperiorization and OTA-C is demonstrated in a real-world application ofdistributed supervised learning over time-varying wireless networkshighlighting its low-latency and energy-efficiency compared to standardapproaches. |


| Item |Content|
| --- |---|
|idx| 2401.17880v1 |
|title| Graph Attention-based Reinforcement Learning for Trajectory Design and Resource Assignment in Multi-UAV Assisted Communication |
|authors| Zikai FengDi WuMengxing HuangChau Yuen
|links| http://arxiv.org/abs/2401.17880v1 |
|updated| 2024-01-31 14:37:06 UTC |
|summary| In the multiple unmanned aerial vehicle UAV- assisted downlinkcommunication it is challenging for UAV base stations UAV BSs to realizetrajectory design and resource assignment in unknown environments. Thecooperation and competition between UAV BSs in the communication network leadsto a Markov game problem. Multi-agent reinforcement learning is a significantsolution for the above decision-making. However there are still many commonissues such as the instability of the system and low utilization of historicaldata that limit its application. In this paper a novel graph-attentionmulti-agent trust region GA-MATR reinforcement learning framework is proposedto solve the multi-UAV assisted communication problem. Graph recurrent networkis introduced to process and analyze complex topology of the communicationnetwork so as to extract useful information and patterns from observationalinformation. The attention mechanism provides additional weighting for conveyedinformation so that the critic network can accurately evaluate the value ofbehavior for UAV BSs. This provides more reliable feedback signals and helpsthe actor network update the strategy more effectively. Ablation simulationsindicate that the proposed approach attains improved convergence over thebaselines. UAV BSs learn the optimal communication strategies to achieve theirmaximum cumulative rewards. Additionally multi-agent trust region method withmonotonic convergence provides an estimated Nash equilibrium for the multi-UAVassisted communication Markov game. |


| Item |Content|
| --- |---|
|idx| 2401.17460v1 |
|title| Rendering Wireless Environments Useful for Gradient Estimators: A Zero-Order Stochastic Federated Learning Method |
|authors| Elissa MhannaMohamad Assaad
|links| http://arxiv.org/abs/2401.17460v1 |
|updated| 2024-01-30 21:46:09 UTC |
|summary| Federated learning FL is a novel approach to machine learning that allowsmultiple edge devices to collaboratively train a model without disclosing theirraw data. However several challenges hinder the practical implementation ofthis approach especially when devices and the server communicate over wirelesschannels as it suffers from communication and computation bottlenecks in thiscase. By utilizing a communication-efficient framework we propose a novelzero-order ZO method with a one-point gradient estimator that harnesses thenature of the wireless communication channel without requiring the knowledge ofthe channel state coefficient. It is the first method that includes thewireless channel in the learning algorithm itself instead of wasting resourcesto analyze it and remove its impact. The two main difficulties of this work arethat in FL the objective function is usually not convex which makes theextension of FL to ZO methods challenging and that including the impact ofwireless channels requires extra attention. However we overcome thesedifficulties and comprehensively analyze the proposed zero-order federatedlearning ZOFL framework. We establish its convergence theoretically and weprove a convergence rate of Ofrac1sqrt3K in the nonconvexsetting. We further demonstrate the potential of our algorithm withexperimental results taking into account independent and identicallydistributed IID and non-IID device data distributions. |


| Item |Content|
| --- |---|
|idx| 2401.17443v1 |
|title| Liquid Democracy for Low-Cost Ensemble Pruning |
|authors| Ben ArmstrongKate Larson
|links| http://arxiv.org/abs/2401.17443v1 |
|updated| 2024-01-30 21:11:35 UTC |
|summary| We argue that there is a strong connection between ensemble learning and adelegative voting paradigm -- liquid democracy -- that can be leveraged toreduce ensemble training costs. We present an incremental training procedurethat identifies and removes redundant classifiers from an ensemble viadelegation mechanisms inspired by liquid democracy. Through both analysis andextensive experiments we show that this process greatly reduces thecomputational cost of training compared to training a full ensemble. Bycarefully selecting the underlying delegation mechanism weight centralizationin the classifier population is avoided leading to higher accuracy than someboosting methods. Furthermore this work serves as an exemplar of howframeworks from computational social choice literature can be applied toproblems in nontraditional domains. |


