# cs.CL 

| Item |Content|
| --- |---|
|idx| 2405.12209v1 |
|title| MathBench: Evaluating the Theory and Application Proficiency of LLMs with a Hierarchical Mathematics Benchmark |
|authors| Hongwei LiuZilong ZhengYuxuan QiaoHaodong DuanZhiwei FeiFengzhe ZhouWenwei ZhangSongyang ZhangDahua LinKai Chen
|links| http://arxiv.org/abs/2405.12209v1 |
|updated| 2024-05-20 17:52:29 UTC |
|summary| Recent advancements in large language models LLMs have showcasedsignificant improvements in mathematics. However traditional math benchmarkslike GSM8k offer a unidimensional perspective falling short in providing aholistic assessment of the LLMs math capabilities. To address this gap weintroduce MathBench a new benchmark that rigorously assesses the mathematicalcapabilities of large language models. MathBench spans a wide range ofmathematical disciplines offering a detailed evaluation of both theoreticalunderstanding and practical problem-solving skills. The benchmark progressesthrough five distinct stages from basic arithmetic to college mathematics andis structured to evaluate models at various depths of knowledge. Each stageincludes theoretical questions and application problems allowing us to measurea models mathematical proficiency and its ability to apply concepts inpractical scenarios. MathBench aims to enhance the evaluation of LLMsmathematical abilities providing a nuanced view of their knowledgeunderstanding levels and problem solving skills in a bilingual context. Theproject is released at https://github.com/open-compass/MathBench . |


| Item |Content|
| --- |---|
|idx| 2405.12206v1 |
|title| Modeling citation worthiness by using attention-based bidirectional long short-term memory networks and interpretable models |
|authors| Tong ZengDaniel E. Acuna
|links| http://dx.doi.org/10.1007/s11192-020-03421-9 |
|updated| 2024-05-20 17:45:36 UTC |
|summary| Scientist learn early on how to cite scientific sources to support theirclaims. Sometimes however scientists have challenges determining where acitation should be situated -- or even worse fail to cite a sourcealtogether. Automatically detecting sentences that need a citation i.e.citation worthiness could solve both of these issues leading to more robustand well-constructed scientific arguments. Previous researchers have appliedmachine learning to this task but have used small datasets and models that donot take advantage of recent algorithmic developments such as attentionmechanisms in deep learning. We hypothesize that we can develop significantlyaccurate deep learning architectures that learn from large supervised datasetsconstructed from open access publications. In this work we propose aBidirectional Long Short-Term Memory BiLSTM network with attention mechanismand contextual information to detect sentences that need citations. We alsoproduce a new large dataset PMOA-CITE based on PubMed Open Access Subsetwhich is orders of magnitude larger than previous datasets. Our experimentsshow that our architecture achieves state of the art performance on thestandard ACL-ARC dataset F_10.507 and exhibits high performanceF_10.856 on the new PMOA-CITE. Moreover we show that it can transferlearning across these datasets. We further use interpretable models toilluminate how specific language is used to promote and inhibit citations. Wediscover that sections and surrounding sentences are crucial for our improvedpredictions. We further examined purported mispredictions of the model anduncovered systematic human mistakes in citation behavior and source data. Thisopens the door for our model to check documents during pre-submission andpre-archival procedures. We make this new dataset the code and a web-basedtool available to the community. |


| Item |Content|
| --- |---|
|idx| 2405.12174v1 |
|title| CT-Eval: Benchmarking Chinese Text-to-Table Performance in Large Language Models |
|authors| Haoxiang ShiJiaan WangJiarong XuCen WangTetsuya Sakai
|links| http://arxiv.org/abs/2405.12174v1 |
|updated| 2024-05-20 16:58:02 UTC |
|summary| Text-to-Table aims to generate structured tables to convey the keyinformation from unstructured documents. Existing text-to-table datasets aretypically oriented English limiting the research in non-English languages.Meanwhile the emergence of large language models LLMs has shown greatsuccess as general task solvers in multi-lingual settings e.g. ChatGPTtheoretically enabling text-to-table in other languages. In this paper wepropose a Chinese text-to-table dataset CT-Eval to benchmark LLMs on thistask. Our preliminary analysis of English text-to-table datasets highlights twokey factors for dataset construction: data diversity and data hallucination.Inspired by this the CT-Eval dataset selects a popular Chinesemultidisciplinary online encyclopedia as the source and covers 28 domains toensure data diversity. To minimize data hallucination we first train an LLM tojudge and filter out the task samples with hallucination then employ humanannotators to clean the hallucinations in the validation and testing sets.After this process CT-Eval contains 88.6K task samples. Using CT-Eval weevaluate the performance of open-source and closed-source LLMs. Our resultsreveal that zero-shot LLMs including GPT-4 still have a significantperformance gap compared with human judgment. Furthermore after fine-tuningopen-source LLMs can significantly improve their text-to-table abilityoutperforming GPT-4 by a large margin. In short CT-Eval not only helpsresearchers evaluate and quickly understand the Chinese text-to-table abilityof existing LLMs but also serves as a valuable resource to significantlyimprove the text-to-table performance of LLMs. |


| Item |Content|
| --- |---|
|idx| 2405.12163v1 |
|title| Fennec: Fine-grained Language Model Evaluation and Correction Extended through Branching and Bridging |
|authors| Xiaobo LiangHaoke ZhangHelan huJuntao LiJun XuMin Zhang
|links| http://arxiv.org/abs/2405.12163v1 |
|updated| 2024-05-20 16:47:22 UTC |
|summary| The rapid advancement of large language models has given rise to a plethoraof applications across a myriad of real-world tasks mainly centered onaligning with human intent. However the complexities inherent in human intentnecessitate a dependence on labor-intensive and time-consuming humanevaluation. To alleviate this constraint we delve into the paradigm ofemploying open-source large language models as evaluators aligning with theprevailing trend of utilizing GPT-4. Particularly we present a step-by-stepevaluation framework: textbfFennec capable of textbfFine-grainedtextbfEvaluatiotextbfN and correctiotextbfN textbfExtended throughbrantextbfChing and bridging. Specifically the branching operation dissectsthe evaluation task into various dimensions and granularities therebyalleviating the challenges associated with evaluation. Concurrently thebridging operation amalgamates diverse training datasets augmenting thevariety of evaluation tasks. In experimental trials our 7B model consistentlyoutperforms open-source larger-scale evaluation models across various widelyadopted benchmarks in terms of both textitAgreement andtextitConsistency closely approaching the capabilities of GPT-4. We employthe fine-grained correction capabilities induced by the evaluation model torefine multiple model responses and the results show that the refinementelevates the quality of responses leading to an improvement of 1-2 points onthe MT-Bench. Our code is available atGithubfootnoteurlhttps://github.com/dropreg/Fennec. |


| Item |Content|
| --- |---|
|idx| 2405.12147v1 |
|title| Eliciting Problem Specifications via Large Language Models |
|authors| Robert E. WrayJames R. KirkJohn E. Laird
|links| http://arxiv.org/abs/2405.12147v1 |
|updated| 2024-05-20 16:19:02 UTC |
|summary| Cognitive systems generally require a human to translate a problem definitioninto some specification that the cognitive system can use to attempt to solvethe problem or perform the task. In this paper we illustrate that largelanguage models LLMs can be utilized to map a problem class defined innatural language into a semi-formal specification that can then be utilized byan existing reasoning and learning system to solve instances from the problemclass. We present the design of LLM-enabled cognitive task analyst agents.Implemented with LLM agents this system produces a definition of problemspaces for tasks specified in natural language. LLM prompts are derived fromthe definition of problem spaces in the AI literature and generalproblem-solving strategies Polyas How to Solve It. A cognitive system canthen use the problem-space specification applying domain-general problemsolving strategies weak methods such as search to solve multiple instancesof problems from the problem class. This result while preliminary suggeststhe potential for speeding cognitive systems research via disintermediation ofproblem formulation while also retaining core capabilities of cognitivesystems such as robust inference and online learning. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2405.12217v1 |
|title| Adapting Large Multimodal Models to Distribution Shifts: The Role of In-Context Learning |
|authors| Guanglin ZhouZhongyi HanShiming ChenBiwei HuangLiming ZhuSalman KhanXin GaoLina Yao
|links| http://arxiv.org/abs/2405.12217v1 |
|updated| 2024-05-20 17:59:21 UTC |
|summary| Recent studies indicate that large multimodal models LMMs are highly robustagainst natural distribution shifts often surpassing previous baselines.Despite this domain-specific adaptation is still necessary particularly inspecialized areas like healthcare. Due to the impracticality of fine-tuningLMMs given their vast parameter space this work investigates in-contextlearning ICL as an effective alternative for enhancing LMMs adaptability. Wefind that the success of ICL heavily relies on the choice of demonstrationmirroring challenges seen in large language models but introducing uniquecomplexities for LMMs facing distribution shifts. Our study addresses this byevaluating an unsupervised ICL method TopKNearestPR which selects in-contextexamples through a nearest example search based on feature similarity. Weuncover that its effectiveness is limited by the deficiencies of pre-trainedvision encoders under distribution shift scenarios. To address thesechallenges we propose InvariantSelectPR a novel method leveragingClass-conditioned Contrastive Invariance CCI for more robust demonstrationselection. Specifically CCI enhances pre-trained vision encoders by improvingtheir discriminative capabilities across different classes and ensuringinvariance to domain-specific variations. This enhancement allows the encodersto effectively identify and retrieve the most informative examples which arethen used to guide LMMs in adapting to new query samples under varyingdistributions. Our experiments show that InvariantSelectPR substantiallyimproves the adaptability of LMMs achieving significant performance gains onbenchmark datasets with a 34.2uparrow accuracy increase in 7-shot onCamelyon17 and 16.9uparrow increase in 7-shot on HAM10000 compared to thebaseline zero-shot performance. |


| Item |Content|
| --- |---|
|idx| 2405.12205v1 |
|title| Metacognitive Capabilities of LLMs: An Exploration in Mathematical Problem Solving |
|authors| Aniket DidolkarAnirudh GoyalNan Rosemary KeSiyuan GuoMichal ValkoTimothy LillicrapDanilo RezendeYoshua BengioMichael MozerSanjeev Arora
|links| http://arxiv.org/abs/2405.12205v1 |
|updated| 2024-05-20 17:45:26 UTC |
|summary| Metacognitive knowledge refers to humans intuitive knowledge of their ownthinking and reasoning processes. Todays best LLMs clearly possess somereasoning processes. The paper gives evidence that they also have metacognitiveknowledge including ability to name skills and procedures to apply given atask. We explore this primarily in context of math reasoning developing aprompt-guided interaction procedure to get a powerful LLM to assign sensibleskill labels to math questions followed by having it perform semanticclustering to obtain coarser families of skill labels. These coarse skilllabels look interpretable to humans.  To validate that these skill labels are meaningful and relevant to the LLMsreasoning processes we perform the following experiments. a We ask GPT-4 toassign skill labels to training questions in math datasets GSM8K and MATH. bWhen using an LLM to solve the test questions we present it with the full listof skill labels and ask it to identify the skill needed. Then it is presentedwith randomly selected exemplar solved questions associated with that skilllabel. This improves accuracy on GSM8k and MATH for several strong LLMsincluding code-assisted models. The methodology presented is domain-agnosticeven though this article applies it to math problems. |


| Item |Content|
| --- |---|
|idx| 2405.12202v1 |
|title| Hierarchical Neural Operator Transformer with Learnable Frequency-aware Loss Prior for Arbitrary-scale Super-resolution |
|authors| Xihaier LuoXiaoning QianByung-Jun Yoon
|links| http://arxiv.org/abs/2405.12202v1 |
|updated| 2024-05-20 17:39:29 UTC |
|summary| In this work we present an arbitrary-scale super-resolution SR method toenhance the resolution of scientific data which often involves complexchallenges such as continuity multi-scale physics and the intricacies ofhigh-frequency signals. Grounded in operator learning the proposed method isresolution-invariant. The core of our model is a hierarchical neural operatorthat leverages a Galerkin-type self-attention mechanism enabling efficientlearning of mappings between function spaces. Sinc filters are used tofacilitate the information transfer across different levels in the hierarchythereby ensuring representation equivalence in the proposed neural operator.Additionally we introduce a learnable prior structure that is derived from thespectral resizing of the input data. This loss prior is model-agnostic and isdesigned to dynamically adjust the weighting of pixel contributions therebybalancing gradients effectively across the model. We conduct extensiveexperiments on diverse datasets from different domains and demonstrateconsistent improvements compared to strong baselines which consist of variousstate-of-the-art SR methods. |


| Item |Content|
| --- |---|
|idx| 2405.12183v1 |
|title| Multi-order Graph Clustering with Adaptive Node-level Weight Learning |
|authors| Ye LiuXuelei LinYejia ChenReynold Cheng
|links| http://arxiv.org/abs/2405.12183v1 |
|updated| 2024-05-20 17:09:58 UTC |
|summary| Current graph clustering methods emphasize individual node and edge connections while ignoring higher-order organization at the level of motif. Recently higher-order graph clustering approaches have been designed by motifbased hypergraphs. However these approaches often suffer from hypergraphfragmentation issue seriously which degrades the clustering performancegreatly. Moreover real-world graphs usually contain diverse motifs with nodesparticipating in multiple motifs. A key challenge is how to achieve preciseclustering results by integrating information from multiple motifs at the nodelevel. In this paper we propose a multi-order graph clustering model MOGC tointegrate multiple higher-order structures and edge connections at node level.MOGC employs an adaptive weight learning mechanism to au tomatically adjust thecontributions of different motifs for each node. This not only tackleshypergraph fragmentation issue but enhances clustering accuracy. MOGC isefficiently solved by an alternating minimization algo rithm. Experiments onseven real-world datasets illustrate the effectiveness of MOGC. |


| Item |Content|
| --- |---|
|idx| 2405.12179v1 |
|title| Building Temporal Kernels with Orthogonal Polynomials |
|authors| Yan Ru PeiOlivier Coenen
|links| http://arxiv.org/abs/2405.12179v1 |
|updated| 2024-05-20 17:06:24 UTC |
|summary| We introduce a class of models named PLEIADES PoLynomial Expansion InAdaptive Distributed Event-based Systems which contains temporal convolutionkernels generated from orthogonal polynomial basis functions. We focus oninterfacing these networks with event-based data to perform onlinespatiotemporal classification and detection with low latency. By virtue ofusing structured temporal kernels and event-based data we have the freedom tovary the sample rate of the data along with the discretization step-size of thenetwork without additional finetuning. We experimented with three event-basedbenchmarks and obtained state-of-the-art results on all three by large marginswith significantly smaller memory and compute costs. We achieved: 1 99.59accuracy with 192K parameters on the DVS128 hand gesture recognition datasetand 100 with a small additional output filter 2 99.58 test accuracy with277K parameters on the AIS 2024 eye tracking challenge and 3 0.556 mAP with576k parameters on the PROPHESEE 1 Megapixel Automotive Detection Dataset. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2405.12221v1 |
|title| Images that Sound: Composing Images and Sounds on a Single Canvas |
|authors| Ziyang ChenDaniel GengAndrew Owens
|links| http://arxiv.org/abs/2405.12221v1 |
|updated| 2024-05-20 17:59:59 UTC |
|summary| Spectrograms are 2D representations of sound that look very different fromthe images found in our visual world. And natural images when played asspectrograms make unnatural sounds. In this paper we show that it is possibleto synthesize spectrograms that simultaneously look like natural images andsound like natural audio. We call these spectrograms images that sound. Ourapproach is simple and zero-shot and it leverages pre-trained text-to-imageand text-to-spectrogram diffusion models that operate in a shared latent space.During the reverse process we denoise noisy latents with both the audio andimage diffusion models in parallel resulting in a sample that is likely underboth models. Through quantitative evaluations and perceptual studies we findthat our method successfully generates spectrograms that align with a desiredaudio prompt while also taking the visual appearance of a desired image prompt.Please see our project page for video results:https://ificl.github.io/images-that-sound/ |


| Item |Content|
| --- |---|
|idx| 2405.12217v1 |
|title| Adapting Large Multimodal Models to Distribution Shifts: The Role of In-Context Learning |
|authors| Guanglin ZhouZhongyi HanShiming ChenBiwei HuangLiming ZhuSalman KhanXin GaoLina Yao
|links| http://arxiv.org/abs/2405.12217v1 |
|updated| 2024-05-20 17:59:21 UTC |
|summary| Recent studies indicate that large multimodal models LMMs are highly robustagainst natural distribution shifts often surpassing previous baselines.Despite this domain-specific adaptation is still necessary particularly inspecialized areas like healthcare. Due to the impracticality of fine-tuningLMMs given their vast parameter space this work investigates in-contextlearning ICL as an effective alternative for enhancing LMMs adaptability. Wefind that the success of ICL heavily relies on the choice of demonstrationmirroring challenges seen in large language models but introducing uniquecomplexities for LMMs facing distribution shifts. Our study addresses this byevaluating an unsupervised ICL method TopKNearestPR which selects in-contextexamples through a nearest example search based on feature similarity. Weuncover that its effectiveness is limited by the deficiencies of pre-trainedvision encoders under distribution shift scenarios. To address thesechallenges we propose InvariantSelectPR a novel method leveragingClass-conditioned Contrastive Invariance CCI for more robust demonstrationselection. Specifically CCI enhances pre-trained vision encoders by improvingtheir discriminative capabilities across different classes and ensuringinvariance to domain-specific variations. This enhancement allows the encodersto effectively identify and retrieve the most informative examples which arethen used to guide LMMs in adapting to new query samples under varyingdistributions. Our experiments show that InvariantSelectPR substantiallyimproves the adaptability of LMMs achieving significant performance gains onbenchmark datasets with a 34.2uparrow accuracy increase in 7-shot onCamelyon17 and 16.9uparrow increase in 7-shot on HAM10000 compared to thebaseline zero-shot performance. |


| Item |Content|
| --- |---|
|idx| 2405.12213v1 |
|title| Octo: An Open-Source Generalist Robot Policy |
|authors| Octo Model TeamDibya GhoshHomer WalkeKarl PertschKevin BlackOier MeesSudeep DasariJoey HejnaTobias KreimanCharles XuJianlan LuoYou Liang TanPannag SanketiQuan VuongTed XiaoDorsa SadighChelsea FinnSergey Levine
|links| http://arxiv.org/abs/2405.12213v1 |
|updated| 2024-05-20 17:57:01 UTC |
|summary| Large policies pretrained on diverse robot datasets have the potential totransform robotic learning: instead of training new policies from scratch suchgeneralist robot policies may be finetuned with only a little in-domain datayet generalize broadly. However to be widely applicable across a range ofrobotic learning scenarios environments and tasks such policies need tohandle diverse sensors and action spaces accommodate a variety of commonlyused robotic platforms and finetune readily and efficiently to new domains. Inthis work we aim to lay the groundwork for developing open-source widelyapplicable generalist policies for robotic manipulation. As a first step weintroduce Octo a large transformer-based policy trained on 800k trajectoriesfrom the Open X-Embodiment dataset the largest robot manipulation dataset todate. It can be instructed via language commands or goal images and can beeffectively finetuned to robot setups with new sensory inputs and action spaceswithin a few hours on standard consumer GPUs. In experiments across 9 roboticplatforms we demonstrate that Octo serves as a versatile policy initializationthat can be effectively finetuned to new observation and action spaces. We alsoperform detailed ablations of design decisions for the Octo model fromarchitecture to training data to guide future research on building generalistrobot models. |


| Item |Content|
| --- |---|
|idx| 2405.12207v1 |
|title| Optimistic Query Routing in Clustering-based Approximate Maximum Inner Product Search |
|authors| Sebastian BruchAditya KrishnanFranco Maria Nardini
|links| http://arxiv.org/abs/2405.12207v1 |
|updated| 2024-05-20 17:47:18 UTC |
|summary| Clustering-based nearest neighbor search is a simple yet effective method inwhich data points are partitioned into geometric shards to form an index andonly a few shards are searched during query processing to find an approximateset of top-k vectors. Even though the search efficacy is heavily influencedby the algorithm that identifies the set of shards to probe it has receivedlittle attention in the literature. This work attempts to bridge that gap bystudying the problem of routing in clustering-based maximum inner productsearch MIPS. We begin by unpacking existing routing protocols and notice thesurprising contribution of optimism. We then take a page from the sequentialdecision making literature and formalize that insight following the principleof optimism in the face of uncertainty. In particular we present a newframework that incorporates the moments of the distribution of inner productswithin each shard to optimistically estimate the maximum inner product. We thenpresent a simple instance of our algorithm that uses only the first two momentsto reach the same accuracy as state-of-the-art routers such as scann byprobing up to 50 fewer points on a suite of benchmark MIPS datasets. Ouralgorithm is also space-efficient: we design a sketch of the second momentwhose size is independent of the number of points and in practice requiresstoring only O1 additional vectors per shard. |


| Item |Content|
| --- |---|
|idx| 2405.12206v1 |
|title| Modeling citation worthiness by using attention-based bidirectional long short-term memory networks and interpretable models |
|authors| Tong ZengDaniel E. Acuna
|links| http://dx.doi.org/10.1007/s11192-020-03421-9 |
|updated| 2024-05-20 17:45:36 UTC |
|summary| Scientist learn early on how to cite scientific sources to support theirclaims. Sometimes however scientists have challenges determining where acitation should be situated -- or even worse fail to cite a sourcealtogether. Automatically detecting sentences that need a citation i.e.citation worthiness could solve both of these issues leading to more robustand well-constructed scientific arguments. Previous researchers have appliedmachine learning to this task but have used small datasets and models that donot take advantage of recent algorithmic developments such as attentionmechanisms in deep learning. We hypothesize that we can develop significantlyaccurate deep learning architectures that learn from large supervised datasetsconstructed from open access publications. In this work we propose aBidirectional Long Short-Term Memory BiLSTM network with attention mechanismand contextual information to detect sentences that need citations. We alsoproduce a new large dataset PMOA-CITE based on PubMed Open Access Subsetwhich is orders of magnitude larger than previous datasets. Our experimentsshow that our architecture achieves state of the art performance on thestandard ACL-ARC dataset F_10.507 and exhibits high performanceF_10.856 on the new PMOA-CITE. Moreover we show that it can transferlearning across these datasets. We further use interpretable models toilluminate how specific language is used to promote and inhibit citations. Wediscover that sections and surrounding sentences are crucial for our improvedpredictions. We further examined purported mispredictions of the model anduncovered systematic human mistakes in citation behavior and source data. Thisopens the door for our model to check documents during pre-submission andpre-archival procedures. We make this new dataset the code and a web-basedtool available to the community. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2405.12221v1 |
|title| Images that Sound: Composing Images and Sounds on a Single Canvas |
|authors| Ziyang ChenDaniel GengAndrew Owens
|links| http://arxiv.org/abs/2405.12221v1 |
|updated| 2024-05-20 17:59:59 UTC |
|summary| Spectrograms are 2D representations of sound that look very different fromthe images found in our visual world. And natural images when played asspectrograms make unnatural sounds. In this paper we show that it is possibleto synthesize spectrograms that simultaneously look like natural images andsound like natural audio. We call these spectrograms images that sound. Ourapproach is simple and zero-shot and it leverages pre-trained text-to-imageand text-to-spectrogram diffusion models that operate in a shared latent space.During the reverse process we denoise noisy latents with both the audio andimage diffusion models in parallel resulting in a sample that is likely underboth models. Through quantitative evaluations and perceptual studies we findthat our method successfully generates spectrograms that align with a desiredaudio prompt while also taking the visual appearance of a desired image prompt.Please see our project page for video results:https://ificl.github.io/images-that-sound/ |


| Item |Content|
| --- |---|
|idx| 2405.12218v1 |
|title| Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo |
|authors| Tianqi LiuGuangcong WangShoukang HuLiao ShenXinyi YeYuhang ZangZhiguo CaoWei LiZiwei Liu
|links| http://arxiv.org/abs/2405.12218v1 |
|updated| 2024-05-20 17:59:30 UTC |
|summary| We present MVSGaussian a new generalizable 3D Gaussian representationapproach derived from Multi-View Stereo MVS that can efficiently reconstructunseen scenes. Specifically 1 we leverage MVS to encode geometry-awareGaussian representations and decode them into Gaussian parameters. 2 Tofurther enhance performance we propose a hybrid Gaussian rendering thatintegrates an efficient volume rendering design for novel view synthesis. 3 Tosupport fast fine-tuning for specific scenes we introduce a multi-viewgeometric consistent aggregation strategy to effectively aggregate the pointclouds generated by the generalizable model serving as the initialization forper-scene optimization. Compared with previous generalizable NeRF-basedmethods which typically require minutes of fine-tuning and seconds ofrendering per image MVSGaussian achieves real-time rendering with bettersynthesis quality for each scene. Compared with the vanilla 3D-GS MVSGaussianachieves better view synthesis with less training computational cost. Extensiveexperiments on DTU Real Forward-facing NeRF Synthetic and Tanks and Templesdatasets validate that MVSGaussian attains state-of-the-art performance withconvincing generalizability real-time rendering speed and fast per-sceneoptimization. |


| Item |Content|
| --- |---|
|idx| 2405.12217v1 |
|title| Adapting Large Multimodal Models to Distribution Shifts: The Role of In-Context Learning |
|authors| Guanglin ZhouZhongyi HanShiming ChenBiwei HuangLiming ZhuSalman KhanXin GaoLina Yao
|links| http://arxiv.org/abs/2405.12217v1 |
|updated| 2024-05-20 17:59:21 UTC |
|summary| Recent studies indicate that large multimodal models LMMs are highly robustagainst natural distribution shifts often surpassing previous baselines.Despite this domain-specific adaptation is still necessary particularly inspecialized areas like healthcare. Due to the impracticality of fine-tuningLMMs given their vast parameter space this work investigates in-contextlearning ICL as an effective alternative for enhancing LMMs adaptability. Wefind that the success of ICL heavily relies on the choice of demonstrationmirroring challenges seen in large language models but introducing uniquecomplexities for LMMs facing distribution shifts. Our study addresses this byevaluating an unsupervised ICL method TopKNearestPR which selects in-contextexamples through a nearest example search based on feature similarity. Weuncover that its effectiveness is limited by the deficiencies of pre-trainedvision encoders under distribution shift scenarios. To address thesechallenges we propose InvariantSelectPR a novel method leveragingClass-conditioned Contrastive Invariance CCI for more robust demonstrationselection. Specifically CCI enhances pre-trained vision encoders by improvingtheir discriminative capabilities across different classes and ensuringinvariance to domain-specific variations. This enhancement allows the encodersto effectively identify and retrieve the most informative examples which arethen used to guide LMMs in adapting to new query samples under varyingdistributions. Our experiments show that InvariantSelectPR substantiallyimproves the adaptability of LMMs achieving significant performance gains onbenchmark datasets with a 34.2uparrow accuracy increase in 7-shot onCamelyon17 and 16.9uparrow increase in 7-shot on HAM10000 compared to thebaseline zero-shot performance. |


| Item |Content|
| --- |---|
|idx| 2405.12211v1 |
|title| Slicedit: Zero-Shot Video Editing With Text-to-Image Diffusion Models Using Spatio-Temporal Slices |
|authors| Nathaniel CohenVladimir KulikovMatan KleinerInbar Huberman-SpiegelglasTomer Michaeli
|links| http://arxiv.org/abs/2405.12211v1 |
|updated| 2024-05-20 17:55:56 UTC |
|summary| Text-to-image T2I diffusion models achieve state-of-the-art results inimage synthesis and editing. However leveraging such pretrained models forvideo editing is considered a major challenge. Many existing works attempt toenforce temporal consistency in the edited video through explicitcorrespondence mechanisms either in pixel space or between deep features.These methods however struggle with strong nonrigid motion. In this paper weintroduce a fundamentally different approach which is based on the observationthat spatiotemporal slices of natural videos exhibit similar characteristics tonatural images. Thus the same T2I diffusion model that is normally used onlyas a prior on video frames can also serve as a strong prior for enhancingtemporal consistency by applying it on spatiotemporal slices. Based on thisobservation we present Slicedit a method for text-based video editing thatutilizes a pretrained T2I diffusion model to process both spatial andspatiotemporal slices. Our method generates videos that retain the structureand motion of the original video while adhering to the target text. Throughextensive experiments we demonstrate Slicedits ability to edit a wide rangeof real-world videos confirming its clear advantages compared to existingcompeting methods. Webpage: https://matankleiner.github.io/slicedit/ |


| Item |Content|
| --- |---|
|idx| 2405.12202v1 |
|title| Hierarchical Neural Operator Transformer with Learnable Frequency-aware Loss Prior for Arbitrary-scale Super-resolution |
|authors| Xihaier LuoXiaoning QianByung-Jun Yoon
|links| http://arxiv.org/abs/2405.12202v1 |
|updated| 2024-05-20 17:39:29 UTC |
|summary| In this work we present an arbitrary-scale super-resolution SR method toenhance the resolution of scientific data which often involves complexchallenges such as continuity multi-scale physics and the intricacies ofhigh-frequency signals. Grounded in operator learning the proposed method isresolution-invariant. The core of our model is a hierarchical neural operatorthat leverages a Galerkin-type self-attention mechanism enabling efficientlearning of mappings between function spaces. Sinc filters are used tofacilitate the information transfer across different levels in the hierarchythereby ensuring representation equivalence in the proposed neural operator.Additionally we introduce a learnable prior structure that is derived from thespectral resizing of the input data. This loss prior is model-agnostic and isdesigned to dynamically adjust the weighting of pixel contributions therebybalancing gradients effectively across the model. We conduct extensiveexperiments on diverse datasets from different domains and demonstrateconsistent improvements compared to strong baselines which consist of variousstate-of-the-art SR methods. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2405.11848v1 |
|title| Alternators For Sequence Modeling |
|authors| Mohammad Reza RezaeiAdji Bousso Dieng
|links| http://arxiv.org/abs/2405.11848v1 |
|updated| 2024-05-20 07:47:06 UTC |
|summary| This paper introduces alternators a novel family of non-Markovian dynamicalmodels for sequences. An alternator features two neural networks: theobservation trajectory network OTN and the feature trajectory network FTN.The OTN and the FTN work in conjunction alternating between outputting samplesin the observation space and some feature space respectively over a cycle.The parameters of the OTN and the FTN are not time-dependent and are learnedvia a minimum cross-entropy criterion over the trajectories. Alternators areversatile. They can be used as dynamical latent-variable generative models oras sequence-to-sequence predictors. When alternators are used as generativemodels the FTN produces interpretable low-dimensional latent variables thatcapture the dynamics governing the observations. When alternators are used assequence-to-sequence predictors the FTN learns to predict the observedfeatures. In both cases the OTN learns to produce sequences that match thedata. Alternators can uncover the latent dynamics underlying complex sequentialdata accurately forecast and impute missing data and sample new trajectories.We showcase the capabilities of alternators in three applications. We firstused alternators to model the Lorenz equations often used to describe chaoticbehavior. We then applied alternators to Neuroscience to map brain activity tophysical activity. Finally we applied alternators to Climate Science focusingon sea-surface temperature forecasting. In all our experiments we foundalternators are stable to train fast to sample from yield high-qualitygenerated samples and latent variables and outperform strong baselines such asneural ODEs and diffusion models in the domains we studied. |


| Item |Content|
| --- |---|
|idx| 2405.11795v1 |
|title| Application of time-series quantum generative model to financial data |
|authors| Shun OkumuraMasayuki OhzekiMasaya Abe
|links| http://arxiv.org/abs/2405.11795v1 |
|updated| 2024-05-20 05:29:45 UTC |
|summary| Despite proposing a quantum generative model for time series thatsuccessfully learns correlated series with multiple Brownian motions the modelhas not been adapted and evaluated for financial problems. In this study atime-series generative model was applied as a quantum generative model toactual financial data. Future data for two correlated time series weregenerated and compared with classical methods such as long short-term memoryand vector autoregression. Furthermore numerical experiments were performed tocomplete missing values. Based on the results we evaluated the practicalapplications of the time-series quantum generation model. It was observed thatfewer parameter values were required compared with the classical method. Inaddition the quantum time-series generation model was feasible for bothstationary and nonstationary data. These results suggest that severalparameters can be applied to various types of time-series data. |


| Item |Content|
| --- |---|
|idx| 2405.11780v1 |
|title| General bounds on the quality of Bayesian coresets |
|authors| Trevor Campbell
|links| http://arxiv.org/abs/2405.11780v1 |
|updated| 2024-05-20 04:46:14 UTC |
|summary| Bayesian coresets speed up posterior inference in the large-scale data regimeby approximating the full-data log-likelihood function with a surrogatelog-likelihood based on a small weighted subset of the data. But whileBayesian coresets and methods for construction are applicable in a wide rangeof models existing theoretical analysis of the posterior inferential errorincurred by coreset approximations only apply in restrictive settings -- i.e.exponential family models or models with strong log-concavity and smoothnessassumptions. This work presents general upper and lower bounds on theKullback-Leibler KL divergence of coreset approximations that reflect thefull range of applicability of Bayesian coresets. The lower bounds require onlymild model assumptions typical of Bayesian asymptotic analyses while the upperbounds require the log-likelihood functions to satisfy a generalizedsubexponentiality criterion that is weaker than conditions used in earlierwork. The lower bounds are applied to obtain fundamental limitations on thequality of coreset approximations and to provide a theoretical explanation forthe previously-observed poor empirical performance of importance sampling-basedconstruction methods. The upper bounds are used to analyze the performance ofrecent subsample-optimize methods. The flexibility of the theory isdemonstrated in validation experiments involving multimodal unidentifiableheavy-tailed Bayesian posterior distributions. |


| Item |Content|
| --- |---|
|idx| 2405.11751v1 |
|title| Asymptotic theory of in-context learning by linear attention |
|authors| Yue M. LuMary I. LeteyJacob A. Zavatone-VethAnindita MaitiCengiz Pehlevan
|links| http://arxiv.org/abs/2405.11751v1 |
|updated| 2024-05-20 03:24:24 UTC |
|summary| Transformers have a remarkable ability to learn and execute tasks based onexamples provided within the input itself without explicit prior training. Ithas been argued that this capability known as in-context learning ICL is acornerstone of Transformers success yet questions about the necessary samplecomplexity pretraining task diversity and context length for successful ICLremain unresolved. Here we provide a precise answer to these questions in anexactly solvable model of ICL of a linear regression task by linear attention.We derive sharp asymptotics for the learning curve in a phenomenologically-richscaling regime where the token dimension is taken to infinity the contextlength and pretraining task diversity scale proportionally with the tokendimension and the number of pretraining examples scales quadratically. Wedemonstrate a double-descent learning curve with increasing pretrainingexamples and uncover a phase transition in the models behavior between lowand high task diversity regimes: In the low diversity regime the model tendstoward memorization of training tasks whereas in the high diversity regime itachieves genuine in-context learning and generalization beyond the scope ofpretrained tasks. These theoretical insights are empirically validated throughexperiments with both linear attention and full nonlinear Transformerarchitectures. |


| Item |Content|
| --- |---|
|idx| 2405.11723v1 |
|title| Inference with non-differentiable surrogate loss in a general high-dimensional classification framework |
|authors| Muxuan LiangYang NingMaureen A SmithYing-Qi Zhao
|links| http://arxiv.org/abs/2405.11723v1 |
|updated| 2024-05-20 01:50:35 UTC |
|summary| Penalized empirical risk minimization with a surrogate loss function is oftenused to derive a high-dimensional linear decision rule in classificationproblems. Although much of the literature focuses on the generalization errorthere is a lack of valid inference procedures to identify the driving factorsof the estimated decision rule especially when the surrogate loss isnon-differentiable. In this work we propose a kernel-smoothed decorrelatedscore to construct hypothesis testing and interval estimations for the lineardecision rule estimated using a piece-wise linear surrogate loss which has adiscontinuous gradient and non-regular Hessian. Specifically we adopt kernelapproximations to smooth the discontinuous gradient near discontinuity pointsand approximate the non-regular Hessian of the surrogate loss. In applicationswhere additional nuisance parameters are involved we propose a novelcross-fitted version to accommodate flexible nuisance estimates and kernelapproximations. We establish the limiting distribution of the kernel-smootheddecorrelated score and its cross-fitted version in a high-dimensional setup.Simulation and real data analysis are conducted to demonstrate the validity andsuperiority of the proposed method. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2405.12040v1 |
|title| Reputation Transfer in the Twitter Diaspora |
|authors| Kristina RadivojevicDJ AdamsGriffin LaszloFelixander KeryTim Weninger
|links| http://arxiv.org/abs/2405.12040v1 |
|updated| 2024-05-20 14:07:59 UTC |
|summary| Social media platforms have witnessed a dynamic landscape of user migrationin recent years fueled by changes in ownership policy and user preferences.This paper explores the phenomenon of user migration from established platformslike X/Twitter to emerging alternatives such as Threads Mastodon and TruthSocial. Leveraging a large dataset from X/Twitter we investigate the extent ofuser departure from X/Twitter and the destinations they migrate to.Additionally we examine whether a users reputation on one platform correlateswith their reputation on another shedding light on the transferability ofdigital reputation across social media ecosystems. Overall we find that userswith a large following on X/Twitter are more likely to migrate to anotherplatform and that their reputation on X/Twitter is highly correlated withreputations on Threads but not Mastodon or Truth Social. |


| Item |Content|
| --- |---|
|idx| 2405.11958v1 |
|title| Exploring Commonalities in Explanation Frameworks: A Multi-Domain Survey Analysis |
|authors| Eduard BarbuMarharytha DomnichRaul VicenteNikos SakkasAndré Morim
|links| http://arxiv.org/abs/2405.11958v1 |
|updated| 2024-05-20 11:28:32 UTC |
|summary| This study presents insights gathered from surveys and discussions withspecialists in three domains aiming to find essential elements for a universalexplanation framework that could be applied to these and other similar usecases. The insights are incorporated into a software tool that utilizes GPalgorithms known for their interpretability. The applications analyzed includea medical scenario involving predictive ML a retail use case involvingprescriptive ML and an energy use case also involving predictive ML. Weinterviewed professionals from each sector transcribing their conversationsfor further analysis. Additionally experts and non-experts in these fieldsfilled out questionnaires designed to probe various dimensions of explanatorymethods. The findings indicate a universal preference for sacrificing a degreeof accuracy in favor of greater explainability. Additionally we highlight thesignificance of feature importance and counterfactual explanations as criticalcomponents of such a framework. Our questionnaires are publicly available tofacilitate the dissemination of knowledge in the field of XAI. |


| Item |Content|
| --- |---|
|idx| 2405.11912v1 |
|title| ARAIDA: Analogical Reasoning-Augmented Interactive Data Annotation |
|authors| Chen HuangYiping JinIlija IlievskiWenqiang LeiJiancheng Lv
|links| http://arxiv.org/abs/2405.11912v1 |
|updated| 2024-05-20 09:48:15 UTC |
|summary| Human annotation is a time-consuming task that requires a significant amountof effort. To address this issue interactive data annotation utilizes anannotation model to provide suggestions for humans to approve or correct.However annotation models trained with limited labeled data are prone togenerating incorrect suggestions leading to extra human correction effort. Totackle this challenge we propose Araida an analogical reasoning-basedapproach that enhances automatic annotation accuracy in the interactive dataannotation setting and reduces the need for human corrections. Araida involvesan error-aware integration strategy that dynamically coordinates an annotationmodel and a k-nearest neighbors KNN model giving more importance to KNNspredictions when predictions from the annotation model are deemed inaccurate.Empirical studies demonstrate that Araida is adaptable to different annotationtasks and models. On average it reduces human correction labor by 11.02compared to vanilla interactive data annotation methods. |


| Item |Content|
| --- |---|
|idx| 2405.11835v1 |
|title| Demo Paper: A Game Agents Battle Driven by Free-Form Text Commands Using Code-Generation LLM |
|authors| Ray ItoJunichiro Takahashi
|links| http://arxiv.org/abs/2405.11835v1 |
|updated| 2024-05-20 07:14:22 UTC |
|summary| This paper presents a demonstration of our monster battle game in which thegame agents fight in accordance with their players language commands. Thecommands were translated into the knowledge expression called behavior branchesby a code-generation large language model. This work facilitated the design ofthe commanding system more easily enabling the game agent to comprehend morevarious and continuous commands than rule-based methods. The results of thecommanding and translation process were stored in a database on an Amazon WebServices server for more comprehensive validation. This implementation wouldprovide a sufficient evaluation of this ongoing work and give insights to theindustry that they could use this to develop their interactive game agents. |


| Item |Content|
| --- |---|
|idx| 2405.11807v1 |
|title| Dual-sided Peltier Elements for Rapid Thermal Feedback in Wearables |
|authors| Seongjun KangGwangbin KimSeokhyun HwangJeongju ParkAhmed ElsharkawySeungJun Kim
|links| http://arxiv.org/abs/2405.11807v1 |
|updated| 2024-05-20 06:00:12 UTC |
|summary| This paper introduces a motor-driven Peltier device designed to deliverimmediate thermal sensations within extended reality XR environments. Thesystem incorporates eight motor-driven Peltier elements facilitating swifttransitions between warm and cool sensations by rotating preheated or cooledelements to opposite sides. A multi-layer structure comprising aluminum andsilicone layers ensures user comfort and safety while maintaining optimaltemperatures for thermal stimuli. Time-temperature characteristic analysisdemonstrates the systems ability to provide warm and cool sensationsefficiently with a dual-sided lifetime of up to 206 seconds at a 2V input. Oursystem design is adaptable to various body parts and can be synchronized withcorresponding visual stimuli to enhance the immersive sensation of virtualobject interaction and information delivery. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2405.11998v1 |
|title| Learning to connect in action: Measuring and understanding the emergence of boundary spanners in volatile times |
|authors| Vittorio NespecaTina ComesFrances Brazier
|links| http://arxiv.org/abs/2405.11998v1 |
|updated| 2024-05-20 13:09:52 UTC |
|summary| Collective intelligence of diverse groups is key for tackling many of todaysgrand challenges such as fostering resilience and climate adaptation.Information exchange across such diverse groups is crucial for collectiveintelligence especially in volatile environments. To facilitate inter-groupinformation exchange Informational Boundary Spanners IBSs as pivotalinformation exchange hubs are promising. However the mechanisms that drivethe emergence of IBSs remain poorly understood. To address this gap there isfirst a need for a method to identify and measure the emergence of IBSs.Second an Agent-Based Modelling ABM framework is not available tosystematically study mechanisms for the emergence of IBSs in volatileenvironments. Third even though the ability to learn who provides high-qualityinformation is thought to be essential to explain the emergence of IBSs arigorous test of this mechanism is missing. The learning mechanism isformalized using an ABM framework with the models outputs analyzed using theproposed IBS emergence measurement method. To illustrate both the method andthe learning mechanism we present a case study focused on information sharingin the volatile environment of a disaster. The study shows that learningconstitutes a mechanism for the emergence of effective IBSs in alow-volatility environments characterised by low uncertainty and b inhigh-volatility environments characterised by rapid change if the number ofinter-group connections is sufficient. With the method and model this paperaims to lay the foundations for exploring mechanisms for the emergence of IBSsthat facilitate inter-group information exchange. This article advancescollective intelligence by providing the essential elements for measuring andunderstanding the emergence of IBSs and exploring the effect of learning ontheir emergence in volatile environments. |


| Item |Content|
| --- |---|
|idx| 2405.11995v1 |
|title| Safe by Design Autonomous Driving Systems |
|authors| Marius BozgaJoseph Sifakis
|links| http://arxiv.org/abs/2405.11995v1 |
|updated| 2024-05-20 12:58:25 UTC |
|summary| Developing safe autonomous driving systems is a major scientific andtechnical challenge. Existing AI-based end-to-end solutions do not offer thenecessary safety guarantees while traditional systems engineering approachesare defeated by the complexity of the problem. Currently there is anincreasing interest in hybrid design solutions integrating machine learningcomponents when necessary while using model-based components for goalmanagement and planning.  We study a method for building safe by design autonomous driving systemsbased on the assumption that the capability to drive boils down to thecoordinated execution of a given set of driving operations. The assumption issubstantiated by a compositionality result considering that autopilots aredynamic systems receiving a small number of types of vistas as input eachvista defining a free space in its neighborhood. It is shown that safe drivingfor each type of vista in the corresponding free space implies safe drivingfor any possible scenario under some easy-to-check conditions concerning thetransition between vistas. The designed autopilot comprises distinct controlpolicies one per type of vista articulated in two consecutive phases. Thefirst phase consists of carefully managing a potentially risky situation byvirtually reducing speed while the second phase consists of exiting thesituation by accelerating.  The autopilots designed use for their predictions simple functionscharacterizing the acceleration and deceleration capabilities of the vehicles.They cover the main driving operations including entering a main roadovertaking crossing intersections protected by traffic lights or signals anddriving on freeways. The results presented reinforce the case for hybridsolutions that incorporate mathematically elegant and robust decision methodsthat are safe by design. |


| Item |Content|
| --- |---|
|idx| 2405.11873v1 |
|title| Equilibria in multiagent online problems with predictions |
|authors| Gabriel IstrateCosmin BonchişVictor Bogdan
|links| http://arxiv.org/abs/2405.11873v1 |
|updated| 2024-05-20 08:30:11 UTC |
|summary| We study the power of competitive algorithms with predictions in amultiagent setting. For this we introduce a multiagent version of theski-rental problem. In this problem agents can collaborate by pooling resourcesto get a group license for some asset. If the license price is not met agentshave to rent the asset individually for the day at a unit price. Otherwise thelicense becomes available forever to everyone at no extra cost. Our maincontribution is a best-response analysis of a single-agent competitivealgorithm that assumes perfect knowledge of other agents actions but noknowledge of its own renting time. We then analyze the setting when agentshave a predictor for their own active time yielding a tradeoff betweenrobustness and consistency. We investigate the effect of using such a predictorin an equilibrium as well as the new equilibria formed in this way. |


| Item |Content|
| --- |---|
|idx| 2405.11778v1 |
|title| Efficient Multi-agent Reinforcement Learning by Planning |
|authors| Qihan LiuJianing YeXiaoteng MaJun YangBin LiangChongjie Zhang
|links| http://arxiv.org/abs/2405.11778v1 |
|updated| 2024-05-20 04:36:02 UTC |
|summary| Multi-agent reinforcement learning MARL algorithms have accomplishedremarkable breakthroughs in solving large-scale decision-making tasks.Nonetheless most existing MARL algorithms are model-free limiting sampleefficiency and hindering their applicability in more challenging scenarios. Incontrast model-based reinforcement learning MBRL particularly algorithmsintegrating planning such as MuZero has demonstrated superhuman performancewith limited data in many tasks. Hence we aim to boost the sample efficiencyof MARL by adopting model-based approaches. However incorporating planning andsearch methods into multi-agent systems poses significant challenges. Theexpansive action space of multi-agent systems often necessitates leveraging thenearly-independent property of agents to accelerate learning. To tackle thisissue we propose the MAZero algorithm which combines a centralized model withMonte Carlo Tree Search MCTS for policy search. We design a novel networkstructure to facilitate distributed execution and parameter sharing. To enhancesearch efficiency in deterministic environments with sizable action spaces weintroduce two novel techniques: Optimistic Search Lambda OSlambda andAdvantage-Weighted Policy Optimization AWPO. Extensive experiments on theSMAC benchmark demonstrate that MAZero outperforms model-free approaches interms of sample efficiency and provides comparable or better performance thanexisting model-based methods in terms of both sample and computationalefficiency. Our code is available at https://github.com/liuqh16/MAZero. |


| Item |Content|
| --- |---|
|idx| 2405.11746v1 |
|title| Configurable Mirror Descent: Towards a Unification of Decision Making |
|authors| Pengdeng LiShuxin LiChang YangXinrun WangShuyue HuXiao HuangHau ChanBo An
|links| http://arxiv.org/abs/2405.11746v1 |
|updated| 2024-05-20 03:10:22 UTC |
|summary| Decision-making problems categorized as single-agent e.g. Ataricooperative multi-agent e.g. Hanabi competitive multi-agent e.g. Holdempoker and mixed cooperative and competitive e.g. football are ubiquitous inthe real world. Various methods are proposed to address the specificdecision-making problems. Despite the successes in specific categories thesemethods typically evolve independently and cannot generalize to othercategories. Therefore a fundamental question for decision-making is: emphCanwe develop textbfa single algorithm to tackle textbfALL categories ofdecision-making problems There are several main challenges to address thisquestion: i different decision-making categories involve different numbers ofagents and different relationships between agents ii different categorieshave different solution concepts and evaluation measures and iii there lacksa comprehensive benchmark covering all the categories. This work presents apreliminary attempt to address the question with three main contributions. iWe propose the generalized mirror descent GMD a generalization of MDvariants which considers multiple historical policies and works with a broaderclass of Bregman divergences. ii We propose the configurable mirror descentCMD where a meta-controller is introduced to dynamically adjust thehyper-parameters in GMD conditional on the evaluation measures. iii Weconstruct the textscGameBench with 15 academic-friendly games acrossdifferent decision-making categories. Extensive experiments demonstrate thatCMD achieves empirically competitive or better outcomes compared to baselineswhile providing the capability of exploring diverse dimensions of decisionmaking. |


