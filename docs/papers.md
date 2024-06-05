# cs.CL 

| Item |Content|
| --- |---|
|idx| 2406.02543v1 |
|title| To Believe or Not to Believe Your LLM |
|authors| Yasin Abbasi YadkoriIlja KuzborskijAndrás GyörgyCsaba Szepesvári
|links| http://arxiv.org/abs/2406.02543v1 |
|updated| 2024-06-04 17:58:18 UTC |
|summary| We explore uncertainty quantification in large language models LLMs withthe goal to identify when uncertainty in responses given a query is large. Wesimultaneously consider both epistemic and aleatoric uncertainties where theformer comes from the lack of knowledge about the ground truth such as aboutfacts or the language and the latter comes from irreducible randomness suchas multiple possible answers. In particular we derive aninformation-theoretic metric that allows to reliably detect when only epistemicuncertainty is large in which case the output of the model is unreliable. Thiscondition can be computed based solely on the output of the model obtainedsimply by some special iterative prompting based on the previous responses.Such quantification for instance allows to detect hallucinations cases whenepistemic uncertainty is high in both single- and multi-answer responses. Thisis in contrast to many standard uncertainty quantification strategies such asthresholding the log-likelihood of a response where hallucinations in themulti-answer case cannot be detected. We conduct a series of experiments whichdemonstrate the advantage of our formulation. Further our investigations shedsome light on how the probabilities assigned to a given output by an LLM can beamplified by iterative prompting which might be of independent interest. |


| Item |Content|
| --- |---|
|idx| 2406.02539v1 |
|title| Parrot: Multilingual Visual Instruction Tuning |
|authors| Hai-Long SunDa-Wei ZhouYang LiShiyin LuChao YiQing-Guo ChenZhao XuWeihua LuoKaifu ZhangDe-Chuan ZhanHan-Jia Ye
|links| http://arxiv.org/abs/2406.02539v1 |
|updated| 2024-06-04 17:56:28 UTC |
|summary| The rapid development of Multimodal Large Language Models MLLMs like GPT-4Vhas marked a significant step towards artificial general intelligence. Existingmethods mainly focus on aligning vision encoders with LLMs through supervisedfine-tuning SFT to endow LLMs with multimodal abilities making MLLMsinherent ability to react to multiple languages progressively deteriorate asthe training process evolves. We empirically find that the imbalanced SFTdatasets primarily composed of English-centric image-text pairs lead tosignificantly reduced performance in non-English languages. This is due to thefailure of aligning the vision encoder and LLM with multilingual tokens duringthe SFT process. In this paper we introduce Parrot a novel method thatutilizes textual guidance to drive visual token alignment at the languagelevel. Parrot makes the visual tokens condition on diverse language inputs anduses Mixture-of-Experts MoE to promote the alignment of multilingual tokens.Specifically to enhance non-English visual tokens alignment we compute thecross-attention using the initial visual features and textual embeddings theresult of which is then fed into the MoE router to select the most relevantexperts. The selected experts subsequently convert the initial visual tokensinto language-specific visual tokens. Moreover considering the current lack ofbenchmarks for evaluating multilingual capabilities within the field wecollect and make available a Massive Multilingual Multimodal Benchmark whichincludes 6 languages 15 categories and 12000 questions named as MMMB. Ourmethod not only demonstrates state-of-the-art performance on multilingualMMBench and MMMB but also excels across a broad range of multimodal tasks.Both the source code and the training dataset of Parrot will be made publiclyavailable. |


| Item |Content|
| --- |---|
|idx| 2406.02537v1 |
|title| TopViewRS: Vision-Language Models as Top-View Spatial Reasoners |
|authors| Chengzu LiCaiqi ZhangHan ZhouNigel CollierAnna KorhonenIvan Vulić
|links| http://arxiv.org/abs/2406.02537v1 |
|updated| 2024-06-04 17:55:43 UTC |
|summary| Top-view perspective denotes a typical way in which humans read and reasonover different types of maps and it is vital for localization and navigationof humans as well as of non-human agents such as the ones backed by largeVision-Language Models VLMs. Nonetheless spatial reasoning capabilities ofmodern VLMs remain unattested and underexplored. In this work we thus studytheir capability to understand and reason over spatial relations from the topview. The focus on top view also enables controlled evaluations at differentgranularity of spatial reasoning we clearly disentangle different abilitiese.g. recognizing particular objects versus understanding their relativepositions. We introduce the TopViewRS Top-View Reasoning in Space datasetconsisting of 11384 multiple-choice questions with either realistic orsemantic top-view map as visual input. We then use it to study and evaluateVLMs across 4 perception and reasoning tasks with different levels ofcomplexity. Evaluation of 10 representative open- and closed-source VLMsreveals the gap of more than 50 compared to average human performance and itis even lower than the random baseline in some cases. Although additionalexperiments show that Chain-of-Thought reasoning can boost model capabilitiesby 5.82 on average the overall performance of VLMs remains limited. Ourfindings underscore the critical need for enhanced model capability in top-viewspatial reasoning and set a foundation for further research towards human-levelproficiency of VLMs in real-world multimodal tasks. |


| Item |Content|
| --- |---|
|idx| 2406.02536v1 |
|title| Mitigate Position Bias in Large Language Models via Scaling a Single Dimension |
|authors| Yijiong YuHuiqiang JiangXufang LuoQianhui WuChin-Yew LinDongsheng LiYuqing YangYongfeng HuangLili Qiu
|links| http://arxiv.org/abs/2406.02536v1 |
|updated| 2024-06-04 17:55:38 UTC |
|summary| Large Language Models LLMs are increasingly applied in various real-worldscenarios due to their excellent generalization capabilities and robustgenerative abilities. However they exhibit position bias also known as lostin the middle a phenomenon that is especially pronounced in long-contextscenarios which indicates the placement of the key information in differentpositions of a prompt can significantly affect accuracy. This paper firstexplores the micro-level manifestations of position bias concluding thatattention weights are a micro-level expression of position bias. It furtheridentifies that in addition to position embeddings causal attention mask alsocontributes to position bias by creating position-specific hidden states. Basedon these insights we propose a method to mitigate position bias by scalingthis positional hidden states. Experiments on the NaturalQuestionsMulti-document QA KV retrieval LongBench and timeline reorder tasks usingvarious models including RoPE models context windowextended models and Alibimodels demonstrate the effectiveness and generalizability of our approach. Ourmethod can improve performance by up to 15.2 by modifying just one dimensionof hidden states. Our code is available at https://aka.ms/PositionalHidden. |


| Item |Content|
| --- |---|
|idx| 2406.02532v1 |
|title| SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices |
|authors| Ruslan SvirschevskiAvner MayZhuoming ChenBeidi ChenZhihao JiaMax Ryabinin
|links| http://arxiv.org/abs/2406.02532v1 |
|updated| 2024-06-04 17:53:36 UTC |
|summary| As large language models gain widespread adoption running them efficientlybecomes crucial. Recent works on LLM inference use speculative decoding toachieve extreme speedups. However most of these works implicitly design theiralgorithms for high-end datacenter hardware. In this work we ask the oppositequestion: how fast can we run LLMs on consumer machines Consumer GPUs can nolonger fit the largest available models 50B parameters and must offload themto RAM or SSD. When running with offloaded parameters the inference engine canprocess batches of hundreds or thousands of tokens at the same time as just onetoken making it a natural fit for speculative decoding. We propose SpecExecSpeculative Execution a simple parallel decoding method that can generate upto 20 tokens per target model iteration for popular LLM families. It utilizesthe high spikiness of the token probabilities distribution in modern LLMs and ahigh degree of alignment between model output probabilities. SpecExec takes themost probable tokens continuation from the draft model to build a cache treefor the target model which then gets validated in a single pass. UsingSpecExec we demonstrate inference of 50B parameter LLMs on consumer GPUs withRAM offloading at 4-6 tokens per second with 4-bit quantization or 2-3 tokensper second with 16-bit weights. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2406.02543v1 |
|title| To Believe or Not to Believe Your LLM |
|authors| Yasin Abbasi YadkoriIlja KuzborskijAndrás GyörgyCsaba Szepesvári
|links| http://arxiv.org/abs/2406.02543v1 |
|updated| 2024-06-04 17:58:18 UTC |
|summary| We explore uncertainty quantification in large language models LLMs withthe goal to identify when uncertainty in responses given a query is large. Wesimultaneously consider both epistemic and aleatoric uncertainties where theformer comes from the lack of knowledge about the ground truth such as aboutfacts or the language and the latter comes from irreducible randomness suchas multiple possible answers. In particular we derive aninformation-theoretic metric that allows to reliably detect when only epistemicuncertainty is large in which case the output of the model is unreliable. Thiscondition can be computed based solely on the output of the model obtainedsimply by some special iterative prompting based on the previous responses.Such quantification for instance allows to detect hallucinations cases whenepistemic uncertainty is high in both single- and multi-answer responses. Thisis in contrast to many standard uncertainty quantification strategies such asthresholding the log-likelihood of a response where hallucinations in themulti-answer case cannot be detected. We conduct a series of experiments whichdemonstrate the advantage of our formulation. Further our investigations shedsome light on how the probabilities assigned to a given output by an LLM can beamplified by iterative prompting which might be of independent interest. |


| Item |Content|
| --- |---|
|idx| 2406.02539v1 |
|title| Parrot: Multilingual Visual Instruction Tuning |
|authors| Hai-Long SunDa-Wei ZhouYang LiShiyin LuChao YiQing-Guo ChenZhao XuWeihua LuoKaifu ZhangDe-Chuan ZhanHan-Jia Ye
|links| http://arxiv.org/abs/2406.02539v1 |
|updated| 2024-06-04 17:56:28 UTC |
|summary| The rapid development of Multimodal Large Language Models MLLMs like GPT-4Vhas marked a significant step towards artificial general intelligence. Existingmethods mainly focus on aligning vision encoders with LLMs through supervisedfine-tuning SFT to endow LLMs with multimodal abilities making MLLMsinherent ability to react to multiple languages progressively deteriorate asthe training process evolves. We empirically find that the imbalanced SFTdatasets primarily composed of English-centric image-text pairs lead tosignificantly reduced performance in non-English languages. This is due to thefailure of aligning the vision encoder and LLM with multilingual tokens duringthe SFT process. In this paper we introduce Parrot a novel method thatutilizes textual guidance to drive visual token alignment at the languagelevel. Parrot makes the visual tokens condition on diverse language inputs anduses Mixture-of-Experts MoE to promote the alignment of multilingual tokens.Specifically to enhance non-English visual tokens alignment we compute thecross-attention using the initial visual features and textual embeddings theresult of which is then fed into the MoE router to select the most relevantexperts. The selected experts subsequently convert the initial visual tokensinto language-specific visual tokens. Moreover considering the current lack ofbenchmarks for evaluating multilingual capabilities within the field wecollect and make available a Massive Multilingual Multimodal Benchmark whichincludes 6 languages 15 categories and 12000 questions named as MMMB. Ourmethod not only demonstrates state-of-the-art performance on multilingualMMBench and MMMB but also excels across a broad range of multimodal tasks.Both the source code and the training dataset of Parrot will be made publiclyavailable. |


| Item |Content|
| --- |---|
|idx| 2406.02534v1 |
|title| Enhancing predictive imaging biomarker discovery through treatment effect analysis |
|authors| Shuhan XiaoLukas KleinJens PetersenPhilipp VollmuthPaul F. JaegerKlaus H. Maier-Hein
|links| http://arxiv.org/abs/2406.02534v1 |
|updated| 2024-06-04 17:54:44 UTC |
|summary| Identifying predictive biomarkers which forecast individual treatmenteffectiveness is crucial for personalized medicine and informs decision-makingacross diverse disciplines. These biomarkers are extracted from pre-treatmentdata often within randomized controlled trials and have to be distinguishedfrom prognostic biomarkers which are independent of treatment assignment. Ourstudy focuses on the discovery of predictive imaging biomarkers aiming toleverage pre-treatment images to unveil new causal relationships. Previousapproaches relied on labor-intensive handcrafted or manually derived featureswhich may introduce biases. In response we present a new task of discoveringpredictive imaging biomarkers directly from the pre-treatment images to learnrelevant image features. We propose an evaluation protocol for this task toassess a models ability to identify predictive imaging biomarkers anddifferentiate them from prognostic ones. It employs statistical testing and acomprehensive analysis of image feature attribution. We explore the suitabilityof deep learning models originally designed for estimating the conditionalaverage treatment effect CATE for this task which previously have beenprimarily assessed for the precision of CATE estimation overlooking theevaluation of imaging biomarker discovery. Our proof-of-concept analysisdemonstrates promising results in discovering and validating predictive imagingbiomarkers from synthetic outcomes and real-world image datasets. |


| Item |Content|
| --- |---|
|idx| 2406.02529v1 |
|title| ReLUs Are Sufficient for Learning Implicit Neural Representations |
|authors| Joseph ShenoudaYamin ZhouRobert D. Nowak
|links| http://arxiv.org/abs/2406.02529v1 |
|updated| 2024-06-04 17:51:08 UTC |
|summary| Motivated by the growing theoretical understanding of neural networks thatemploy the Rectified Linear Unit ReLU as their activation function werevisit the use of ReLU activation functions for learning implicit neuralrepresentations INRs. Inspired by second order B-spline wavelets weincorporate a set of simple constraints to the ReLU neurons in each layer of adeep neural network DNN to remedy the spectral bias. This in turn enables itsuse for various INR tasks. Empirically we demonstrate that contrary topopular belief one can learn state-of-the-art INRs based on a DNN composed ofonly ReLU neurons. Next by leveraging recent theoretical works whichcharacterize the kinds of functions ReLU neural networks learn we provide away to quantify the regularity of the learned function. This offers aprincipled approach to selecting the hyperparameters in INR architectures. Wesubstantiate our claims through experiments in signal representation superresolution and computed tomography demonstrating the versatility andeffectiveness of our method. The code for all experiments can be found athttps://github.com/joeshenouda/relu-inrs. |


| Item |Content|
| --- |---|
|idx| 2406.02523v1 |
|title| RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots |
|authors| Soroush NasirianyAbhiram MaddukuriLance ZhangAdeet ParikhAaron LoAbhishek JoshiAjay MandlekarYuke Zhu
|links| http://arxiv.org/abs/2406.02523v1 |
|updated| 2024-06-04 17:41:31 UTC |
|summary| Recent advancements in Artificial Intelligence AI have largely beenpropelled by scaling. In Robotics scaling is hindered by the lack of access tomassive robot datasets. We advocate using realistic physical simulation as ameans to scale environments tasks and datasets for robot learning methods. Wepresent RoboCasa a large-scale simulation framework for training generalistrobots in everyday environments. RoboCasa features realistic and diverse scenesfocusing on kitchen environments. We provide thousands of 3D assets across over150 object categories and dozens of interactable furniture and appliances. Weenrich the realism and diversity of our simulation with generative AI toolssuch as object assets from text-to-3D models and environment textures fromtext-to-image models. We design a set of 100 tasks for systematic evaluationincluding composite tasks generated by the guidance of large language models.To facilitate learning we provide high-quality human demonstrations andintegrate automated trajectory generation methods to substantially enlarge ourdatasets with minimal human burden. Our experiments show a clear scaling trendin using synthetically generated robot data for large-scale imitation learningand show great promise in harnessing simulation data in real-world tasks.Videos and open-source code are available at https://robocasa.ai/ |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2406.02550v1 |
|title| Learning to grok: Emergence of in-context learning and skill composition in modular arithmetic tasks |
|authors| Tianyu HeDarshil DoshiAritra DasAndrey Gromov
|links| http://arxiv.org/abs/2406.02550v1 |
|updated| 2024-06-04 17:59:36 UTC |
|summary| Large language models can solve tasks that were not present in the trainingset. This capability is believed to be due to in-context learning and skillcomposition. In this work we study the emergence of in-context learning andskill composition in a collection of modular arithmetic tasks. Specifically weconsider a finite collection of linear modular functions z  a  x  b  ymathrmmod p labeled by the vector a b in mathbbZ_p2. We usesome of these tasks for pre-training and the rest for out-of-distributiontesting. We empirically show that a GPT-style transformer exhibits a transitionfrom in-distribution to out-of-distribution generalization as the number ofpre-training tasks increases. We find that the smallest model capable ofout-of-distribution generalization requires two transformer blocks while fordeeper models the out-of-distribution generalization phase isemphtransient necessitating early stopping. Finally we perform aninterpretability study of the pre-trained models revealing the highlystructured representations in both phases and discuss the learnt algorithm. |


| Item |Content|
| --- |---|
|idx| 2406.02545v1 |
|title| Robust and highly scalable estimation of directional couplings from time-shifted signals |
|authors| Luca AmbrogioniLouis RouillardDemian Wassermann
|links| http://arxiv.org/abs/2406.02545v1 |
|updated| 2024-06-04 17:58:33 UTC |
|summary| The estimation of directed couplings between the nodes of a network fromindirect measurements is a central methodological challenge in scientificfields such as neuroscience systems biology and economics. Unfortunately theproblem is generally ill-posed due to the possible presence of unknown delaysin the measurements. In this paper we offer a solution of this problem byusing a variational Bayes framework where the uncertainty over the delays ismarginalized in order to obtain conservative coupling estimates. To overcomethe well-known overconfidence of classical variational methods we use ahybrid-VI scheme where the possibly flat or multimodal posterior over themeasurement parameters is estimated using a forward KL loss while the nearlyconvex conditional posterior over the couplings is estimated using the highlyscalable gradient-based VI. In our ground-truth experiments we show that thenetwork provides reliable and conservative estimates of the couplings greatlyoutperforming similar methods such as regression DCM. |


| Item |Content|
| --- |---|
|idx| 2406.02543v1 |
|title| To Believe or Not to Believe Your LLM |
|authors| Yasin Abbasi YadkoriIlja KuzborskijAndrás GyörgyCsaba Szepesvári
|links| http://arxiv.org/abs/2406.02543v1 |
|updated| 2024-06-04 17:58:18 UTC |
|summary| We explore uncertainty quantification in large language models LLMs withthe goal to identify when uncertainty in responses given a query is large. Wesimultaneously consider both epistemic and aleatoric uncertainties where theformer comes from the lack of knowledge about the ground truth such as aboutfacts or the language and the latter comes from irreducible randomness suchas multiple possible answers. In particular we derive aninformation-theoretic metric that allows to reliably detect when only epistemicuncertainty is large in which case the output of the model is unreliable. Thiscondition can be computed based solely on the output of the model obtainedsimply by some special iterative prompting based on the previous responses.Such quantification for instance allows to detect hallucinations cases whenepistemic uncertainty is high in both single- and multi-answer responses. Thisis in contrast to many standard uncertainty quantification strategies such asthresholding the log-likelihood of a response where hallucinations in themulti-answer case cannot be detected. We conduct a series of experiments whichdemonstrate the advantage of our formulation. Further our investigations shedsome light on how the probabilities assigned to a given output by an LLM can beamplified by iterative prompting which might be of independent interest. |


| Item |Content|
| --- |---|
|idx| 2406.02542v1 |
|title| Loki: Low-Rank Keys for Efficient Sparse Attention |
|authors| Prajwal SinghaniaSiddharth SinghShwai HeSoheil FeiziAbhinav Bhatele
|links| http://arxiv.org/abs/2406.02542v1 |
|updated| 2024-06-04 17:58:03 UTC |
|summary| Inference on large language models can be expensive in terms of the computeand memory costs involved especially when long sequence lengths are used. Inparticular the self-attention mechanism used in such models contributessignificantly to these costs which has resulted in several recent works thatpropose sparse attention approximations for inference. In this work we proposeto approximate the self-attention computation by focusing on the dimensionalityof key vectors computed in the attention block. Our analysis reveals that thekey vectors lie in a significantly lower-dimensional space consistently acrossseveral datasets and models. Exploiting this observation we propose Loki anovel sparse attention method that ranks and selects tokens in the KV-cachebased on attention scores computed in low-dimensional space. Our evaluationsshow that Loki is able to maintain the efficacy of the models better than otherpopular approximation methods while speeding up the attention computation dueto reduced data movement load/store and compute costs. |


| Item |Content|
| --- |---|
|idx| 2406.02539v1 |
|title| Parrot: Multilingual Visual Instruction Tuning |
|authors| Hai-Long SunDa-Wei ZhouYang LiShiyin LuChao YiQing-Guo ChenZhao XuWeihua LuoKaifu ZhangDe-Chuan ZhanHan-Jia Ye
|links| http://arxiv.org/abs/2406.02539v1 |
|updated| 2024-06-04 17:56:28 UTC |
|summary| The rapid development of Multimodal Large Language Models MLLMs like GPT-4Vhas marked a significant step towards artificial general intelligence. Existingmethods mainly focus on aligning vision encoders with LLMs through supervisedfine-tuning SFT to endow LLMs with multimodal abilities making MLLMsinherent ability to react to multiple languages progressively deteriorate asthe training process evolves. We empirically find that the imbalanced SFTdatasets primarily composed of English-centric image-text pairs lead tosignificantly reduced performance in non-English languages. This is due to thefailure of aligning the vision encoder and LLM with multilingual tokens duringthe SFT process. In this paper we introduce Parrot a novel method thatutilizes textual guidance to drive visual token alignment at the languagelevel. Parrot makes the visual tokens condition on diverse language inputs anduses Mixture-of-Experts MoE to promote the alignment of multilingual tokens.Specifically to enhance non-English visual tokens alignment we compute thecross-attention using the initial visual features and textual embeddings theresult of which is then fed into the MoE router to select the most relevantexperts. The selected experts subsequently convert the initial visual tokensinto language-specific visual tokens. Moreover considering the current lack ofbenchmarks for evaluating multilingual capabilities within the field wecollect and make available a Massive Multilingual Multimodal Benchmark whichincludes 6 languages 15 categories and 12000 questions named as MMMB. Ourmethod not only demonstrates state-of-the-art performance on multilingualMMBench and MMMB but also excels across a broad range of multimodal tasks.Both the source code and the training dataset of Parrot will be made publiclyavailable. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2406.02552v1 |
|title| VHS: High-Resolution Iterative Stereo Matching with Visual Hull Priors |
|authors| Markus PlackHannah DrögeLeif Van HollandMatthias B. Hullin
|links| http://arxiv.org/abs/2406.02552v1 |
|updated| 2024-06-04 17:59:57 UTC |
|summary| We present a stereo-matching method for depth estimation from high-resolutionimages using visual hulls as priors and a memory-efficient technique for thecorrelation computation. Our method uses object masks extracted fromsupplementary views of the scene to guide the disparity estimation effectivelyreducing the search space for matches. This approach is specifically tailoredto stereo rigs in volumetric capture systems where an accurate depth plays akey role in the downstream reconstruction task. To enable training andregression at high resolutions targeted by recent systems our approach extendsa sparse correlation computation into a hybrid sparse-dense scheme suitable forapplication in leading recurrent network architectures. We evaluate theperformance-efficiency trade-off of our method compared to state-of-the-artmethods and demonstrate the efficacy of the visual hull guidance. In additionwe propose a training scheme for a further reduction of memory requirementsduring optimization facilitating training on high-resolution data. |


| Item |Content|
| --- |---|
|idx| 2406.02549v1 |
|title| Dreamguider: Improved Training free Diffusion-based Conditional Generation |
|authors| Nithin Gopalakrishnan NairVishal M Patel
|links| http://arxiv.org/abs/2406.02549v1 |
|updated| 2024-06-04 17:59:32 UTC |
|summary| Diffusion models have emerged as a formidable tool for training-freeconditional generation.However a key hurdle in inference-time guidancetechniques is the need for compute-heavy backpropagation through the diffusionnetwork for estimating the guidance direction. Moreover these techniques oftenrequire handcrafted parameter tuning on a case-by-case basis. Although somerecent works have introduced minimal compute methods for linear inverseproblems a generic lightweight guidance solution to both linear and non-linearguidance problems is still missing. To this end we propose Dreamguider amethod that enables inference-time guidance without compute-heavybackpropagation through the diffusion network. The key idea is to regulate thegradient flow through a time-varying factor. Moreover we propose an empiricalguidance scale that works for a wide variety of tasks hence removing the needfor handcrafted parameter tuning. We further introduce an effective lightweightaugmentation strategy that significantly boosts the performance duringinference-time guidance. We present experiments using Dreamguider on multipletasks across multiple datasets and models to show the effectiveness of theproposed modules. To facilitate further research we will make the code publicafter the review process. |


| Item |Content|
| --- |---|
|idx| 2406.02548v1 |
|title| Open-YOLO 3D: Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation |
|authors| Mohamed El Amine BoudjoghraAngela DaiJean LahoudHisham CholakkalRao Muhammad AnwerSalman KhanFahad Shahbaz Khan
|links| http://arxiv.org/abs/2406.02548v1 |
|updated| 2024-06-04 17:59:31 UTC |
|summary| Recent works on open-vocabulary 3D instance segmentation show strong promisebut at the cost of slow inference speed and high computation requirements. Thishigh computation cost is typically due to their heavy reliance on 3D clipfeatures which require computationally expensive 2D foundation models likeSegment Anything SAM and CLIP for multi-view aggregation into 3D. As aconsequence this hampers their applicability in many real-world applicationsthat require both fast and accurate predictions. To this end we propose a fastyet accurate open-vocabulary 3D instance segmentation approach named Open-YOLO3D that effectively leverages only 2D object detection from multi-view RGBimages for open-vocabulary 3D instance segmentation. We address this task bygenerating class-agnostic 3D masks for objects in the scene and associatingthem with text prompts. We observe that the projection of class-agnostic 3Dpoint cloud instances already holds instance information thus using SAM mightonly result in redundancy that unnecessarily increases the inference time. Weempirically find that a better performance of matching text prompts to 3D maskscan be achieved in a faster fashion with a 2D object detector. We validate ourOpen-YOLO 3D on two benchmarks ScanNet200 and Replica under two scenarios:i with ground truth masks where labels are required for given objectproposals and ii with class-agnostic 3D proposals generated from a 3Dproposal network. Our Open-YOLO 3D achieves state-of-the-art performance onboth datasets while obtaining up to sim16times speedup compared to thebest existing method in literature. On ScanNet200 val. set our Open-YOLO 3Dachieves mean average precision mAP of 24.7 while operating at 22 secondsper scene. Code and model are available at github.com/aminebdj/OpenYOLO3D. |


| Item |Content|
| --- |---|
|idx| 2406.02547v1 |
|title| Leveraging Visual Tokens for Extended Text Contexts in Multi-Modal Learning |
|authors| Alex Jinpeng WangLinjie LiYiqi LinMin LiLijuan WangMike Zheng Shou
|links| http://arxiv.org/abs/2406.02547v1 |
|updated| 2024-06-04 17:59:25 UTC |
|summary| Training models with longer in-context lengths is a significant challenge formultimodal model due to substantial GPU memory and computational costs. Thisexploratory study does not present state-of-the-art models rather itintroduces an innovative method designed to increase in-context text length inmulti-modality large language models MLLMs efficiently. We present VisualizedIn-Context Text Processing VisInContext which processes long in-context textusing visual tokens. This technique significantly reduces GPU memory usage andfloating point operations FLOPs for both training and inferenceing stage. Forinstance our method expands the pre-training in-context text length from 256to 2048 tokens with nearly same FLOPs for a 56 billion parameter MOE model.Experimental results demonstrate that model trained with VisInContext deliverssuperior performance on common downstream benchmarks for in-context few-shotevaluation. Additionally VisInContext is complementary to existing methods forincreasing in-context text length and enhances document understandingcapabilities showing great potential in document QA tasks and sequentialdocument retrieval. |


| Item |Content|
| --- |---|
|idx| 2406.02541v1 |
|title| Enhancing Temporal Consistency in Video Editing by Reconstructing Videos with 3D Gaussian Splatting |
|authors| Inkyu ShinQihang YuXiaohui ShenIn So KweonKuk-Jin YoonLiang-Chieh Chen
|links| http://arxiv.org/abs/2406.02541v1 |
|updated| 2024-06-04 17:57:37 UTC |
|summary| Recent advancements in zero-shot video diffusion models have shown promisefor text-driven video editing but challenges remain in achieving high temporalconsistency. To address this we introduce Video-3DGS a 3D Gaussian Splatting3DGS-based video refiner designed to enhance temporal consistency inzero-shot video editors. Our approach utilizes a two-stage 3D Gaussianoptimizing process tailored for editing dynamic monocular videos. In the firststage Video-3DGS employs an improved version of COLMAP referred to asMC-COLMAP which processes original videos using a Masked and Clipped approach.For each video clip MC-COLMAP generates the point clouds for dynamicforeground objects and complex backgrounds. These point clouds are utilized toinitialize two sets of 3D Gaussians Frg-3DGS and Bkg-3DGS aiming to representforeground and background views. Both foreground and background views are thenmerged with a 2D learnable parameter map to reconstruct full views. In thesecond stage we leverage the reconstruction ability developed in the firststage to impose the temporal constraints on the video diffusion model. Todemonstrate the efficacy of Video-3DGS on both stages we conduct extensiveexperiments across two related tasks: Video Reconstruction and Video Editing.Video-3DGS trained with 3k iterations significantly improves videoreconstruction quality 3 PSNR 7 PSNR increase and training efficiencyx1.9 x4.5 times faster over NeRF-based and 3DGS-based state-of-art methodson DAVIS dataset respectively. Moreover it enhances video editing by ensuringtemporal consistency across 58 dynamic monocular videos. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2406.02550v1 |
|title| Learning to grok: Emergence of in-context learning and skill composition in modular arithmetic tasks |
|authors| Tianyu HeDarshil DoshiAritra DasAndrey Gromov
|links| http://arxiv.org/abs/2406.02550v1 |
|updated| 2024-06-04 17:59:36 UTC |
|summary| Large language models can solve tasks that were not present in the trainingset. This capability is believed to be due to in-context learning and skillcomposition. In this work we study the emergence of in-context learning andskill composition in a collection of modular arithmetic tasks. Specifically weconsider a finite collection of linear modular functions z  a  x  b  ymathrmmod p labeled by the vector a b in mathbbZ_p2. We usesome of these tasks for pre-training and the rest for out-of-distributiontesting. We empirically show that a GPT-style transformer exhibits a transitionfrom in-distribution to out-of-distribution generalization as the number ofpre-training tasks increases. We find that the smallest model capable ofout-of-distribution generalization requires two transformer blocks while fordeeper models the out-of-distribution generalization phase isemphtransient necessitating early stopping. Finally we perform aninterpretability study of the pre-trained models revealing the highlystructured representations in both phases and discuss the learnt algorithm. |


| Item |Content|
| --- |---|
|idx| 2406.02507v1 |
|title| Guiding a Diffusion Model with a Bad Version of Itself |
|authors| Tero KarrasMiika AittalaTuomas KynkäänniemiJaakko LehtinenTimo AilaSamuli Laine
|links| http://arxiv.org/abs/2406.02507v1 |
|updated| 2024-06-04 17:25:59 UTC |
|summary| The primary axes of interest in image-generating diffusion models are imagequality the amount of variation in the results and how well the results alignwith a given condition e.g. a class label or a text prompt. The popularclassifier-free guidance approach uses an unconditional model to guide aconditional model leading to simultaneously better prompt alignment andhigher-quality images at the cost of reduced variation. These effects seeminherently entangled and thus hard to control. We make the surprisingobservation that it is possible to obtain disentangled control over imagequality without compromising the amount of variation by guiding generationusing a smaller less-trained version of the model itself rather than anunconditional model. This leads to significant improvements in ImageNetgeneration setting record FIDs of 1.01 for 64x64 and 1.25 for 512x512 usingpublicly available networks. Furthermore the method is also applicable tounconditional diffusion models drastically improving their quality. |


| Item |Content|
| --- |---|
|idx| 2406.02490v1 |
|title| Ai-Sampler: Adversarial Learning of Markov kernels with involutive maps |
|authors| Evgenii EgorovRicardo ValpergaEfstratios Gavves
|links| http://arxiv.org/abs/2406.02490v1 |
|updated| 2024-06-04 17:00:14 UTC |
|summary| Markov chain Monte Carlo methods have become popular in statistics asversatile techniques to sample from complicated probability distributions. Inthis work we propose a method to parameterize and train transition kernels ofMarkov chains to achieve efficient sampling and good mixing. This trainingprocedure minimizes the total variation distance between the stationarydistribution of the chain and the empirical distribution of the data. Ourapproach leverages involutive Metropolis-Hastings kernels constructed fromreversible neural networks that ensure detailed balance by construction. Wefind that reversibility also implies C_2-equivariance of the discriminatorfunction which can be used to restrict its function space. |


| Item |Content|
| --- |---|
|idx| 2406.02464v1 |
|title| Meta-Learners for Partially-Identified Treatment Effects Across Multiple Environments |
|authors| Jonas SchweisthalDennis FrauenMihaela van der SchaarStefan Feuerriegel
|links| http://arxiv.org/abs/2406.02464v1 |
|updated| 2024-06-04 16:31:43 UTC |
|summary| Estimating the conditional average treatment effect CATE from observationaldata is relevant for many applications such as personalized medicine. Here wefocus on the widespread setting where the observational data come from multipleenvironments such as different hospitals physicians or countries.Furthermore we allow for violations of standard causal assumptions namelyoverlap within the environments and unconfoundedness. To this end we move awayfrom point identification and focus on partial identification. Specifically weshow that current assumptions from the literature on multiple environmentsallow us to interpret the environment as an instrumental variable IV. Thisallows us to adapt bounds from the IV literature for partial identification ofCATE by leveraging treatment assignment mechanisms across environments. Thenwe propose different model-agnostic learners so-called meta-learners toestimate the bounds that can be used in combination with arbitrary machinelearning models. We further demonstrate the effectiveness of our meta-learnersacross various experiments using both simulated and real-world data. Finallywe discuss the applicability of our meta-learners to partial identification ininstrumental variable settings such as randomized controlled trials withnon-compliance. |


| Item |Content|
| --- |---|
|idx| 2406.02432v1 |
|title| Coresets for Multiple $\ell_p$ Regression |
|authors| David P. WoodruffTaisuke Yasuda
|links| http://arxiv.org/abs/2406.02432v1 |
|updated| 2024-06-04 15:50:42 UTC |
|summary| A coreset of a dataset with n examples and d features is a weightedsubset of examples that is sufficient for solving downstream data analytictasks. Nearly optimal constructions of coresets for least squares and ell_plinear regression with a single response are known in prior work. However formultiple ell_p regression where there can be m responses there are noknown constructions with size sublinear in m. In this work we constructcoresets of size tilde Ovarepsilon-2d for p2 and tildeOvarepsilon-pdp/2 for p2 independently of m i.e.dimension-free that approximate the multiple ell_p regression objective atevery point in the domain up to 1pmvarepsilon relative error. If we onlyneed to preserve the minimizer subject to a subspace constraint we improvethese bounds by an varepsilon factor for all p1. All of our bounds arenearly tight.  We give two application of our results. First we settle the number ofuniform samples needed to approximate ell_p Euclidean power means up to a1varepsilon factor showing that tildeThetavarepsilon-2 samplesfor p  1 tildeThetavarepsilon-1 samples for 1  p  2 andtildeThetavarepsilon1-p samples for p2 is tight answering aquestion of Cohen-Addad Saulpic and Schwiegelshohn. Second we show that for1p2 every matrix has a subset of tilde Ovarepsilon-1k rows whichspans a 1varepsilon-approximately optimal k-dimensional subspace forell_p subspace approximation which is also nearly optimal. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2406.02520v1 |
|title| Digital Privacy for Migrants: Exploring Current Research Trends and Future Prospects |
|authors| Sarah TabassumCori Faklaris
|links| http://arxiv.org/abs/2406.02520v1 |
|updated| 2024-06-04 17:41:20 UTC |
|summary| This paper explores digital privacy challenges for migrants analyzing trendsfrom 2013 to 2023. Migrants face heightened risks such as governmentsurveillance and identity theft. Understanding these threats is vital forraising awareness and guiding research towards effective solutions and policiesto protect migrant digital privacy. |


| Item |Content|
| --- |---|
|idx| 2406.02090v1 |
|title| How Western, Educated, Industrialized, Rich, and Democratic is Social Computing Research? |
|authors| Ali Akbar SeptiandriMarios ConstantinidesDaniele Quercia
|links| http://arxiv.org/abs/2406.02090v1 |
|updated| 2024-06-04 08:17:47 UTC |
|summary| Much of the research in social computing analyzes data from social mediaplatforms which may inherently carry biases. An overlooked source of such biasis the over-representation of WEIRD Western Educated Industrialized Richand Democratic populations which might not accurately mirror the globaldemographic diversity. We evaluated the dependence on WEIRD populations inresearch presented at the AAAI ICWSM conference the only venue whoseproceedings are fully dedicated to social computing research. We did so byanalyzing 494 papers published from 2018 to 2022 which included full researchpapers dataset papers and posters. After filtering out papers that analyzesynthetic datasets or those lacking clear country of origin we were left with420 papers from which 188 participants in a crowdsourcing study with fullmanual validation extracted data for the WEIRD scores computation. This datawas then used to adapt existing WEIRD metrics to be applicable for social mediadata. We found that 37 of these papers focused solely on data from Westerncountries. This percentage is significantly less than the percentages observedin research from CHI 76 and FAccT 84 conferences suggesting a greaterdiversity of dataset origins within ICWSM. However the studies at ICWSM stillpredominantly examine populations from countries that are more EducatedIndustrialized and Rich in comparison to those in FAccT with a special noteon the Democratic variable reflecting political freedoms and rights. Thispoints out the utility of social media data in shedding light on findings fromcountries with restricted political freedoms. Based on these insights werecommend extensions of current paper checklists to include considerationsabout the WEIRD bias and call for the community to broaden research inclusivityby encouraging the use of diverse datasets from underrepresented regions. |


| Item |Content|
| --- |---|
|idx| 2406.02018v1 |
|title| Why Would You Suggest That? Human Trust in Language Model Responses |
|authors| Manasi SharmaHo Chit SiuRohan PalejaJaime D. Peña
|links| http://arxiv.org/abs/2406.02018v1 |
|updated| 2024-06-04 06:57:47 UTC |
|summary| The emergence of Large Language Models LLMs has revealed a growing need forhuman-AI collaboration especially in creative decision-making scenarios wheretrust and reliance are paramount. Through human studies and model evaluationson the open-ended News Headline Generation task from the LaMP benchmark weanalyze how the framing and presence of explanations affect user trust andmodel performance. Overall we provide evidence that adding an explanation inthe model response to justify its reasoning significantly increasesself-reported user trust in the model when the user has the opportunity tocompare various responses. Position and faithfulness of these explanations arealso important factors. However these gains disappear when users are shownresponses independently suggesting that humans trust all model responsesincluding deceptive ones equitably when they are shown in isolation. Ourfindings urge future research to delve deeper into the nuanced evaluation oftrust in human-machine teaming systems. |


| Item |Content|
| --- |---|
|idx| 2406.01964v1 |
|title| Measure-Observe-Remeasure: An Interactive Paradigm for Differentially-Private Exploratory Analysis |
|authors| Priyanka NanayakkaraHyeok KimYifan WuAli SarvghadNarges MahyarGerome MiklauJessica Hullman
|links| http://dx.doi.org/10.1109/SP54263.2024.00182 |
|updated| 2024-06-04 04:48:40 UTC |
|summary| Differential privacy DP has the potential to enable privacy-preservinganalysis on sensitive data but requires analysts to judiciously spend alimited privacy loss budget epsilon across queries. Analysts conductingexploratory analyses do not however know all queries in advance and seldomhave DP expertise. Thus they are limited in their ability to specifyepsilon allotments across queries prior to an analysis. To support analystsin spending epsilon efficiently we propose a new interactive analysisparadigm Measure-Observe-Remeasure where analysts measure the databasewith a limited amount of epsilon observe estimates and their errors andremeasure with more epsilon as needed.  We instantiate the paradigm in an interactive visualization interface whichallows analysts to spend increasing amounts of epsilon under a total budget.To observe how analysts interact with the Measure-Observe-Remeasure paradigmvia the interface we conduct a user study that compares the utility ofepsilon allocations and findings from sensitive data participants make tothe allocations and findings expected of a rational agent who faces the samedecision task. We find that participants are able to use the workflowrelatively successfully including using budget allocation strategies thatmaximize over half of the available utility stemming from epsilonallocation. Their loss in performance relative to a rational agent appears tobe driven more by their inability to access information and report it than toallocate epsilon. |


| Item |Content|
| --- |---|
|idx| 2406.01915v1 |
|title| Enhancing Human-Robot Collaborative Assembly in Manufacturing Systems Using Large Language Models |
|authors| Jonghan LimSujani PatelAlex EvansJohn PimleyYifei LiIlya Kovalenko
|links| http://arxiv.org/abs/2406.01915v1 |
|updated| 2024-06-04 02:52:26 UTC |
|summary| The development of human-robot collaboration has the ability to improvemanufacturing system performance by leveraging the unique strengths of bothhumans and robots. On the shop floor human operators contribute with theiradaptability and flexibility in dynamic situations while robots provideprecision and the ability to perform repetitive tasks. However thecommunication gap between human operators and robots limits the collaborationand coordination of human-robot teams in manufacturing systems. Our researchpresents a human-robot collaborative assembly framework that utilizes a largelanguage model for enhancing communication in manufacturing environments. Theframework facilitates human-robot communication by integrating voice commandsthrough natural language for task management. A case study for an assembly taskdemonstrates the frameworks ability to process natural language inputs andaddress real-time assembly challenges emphasizing adaptability to languagevariation and efficiency in error resolution. The results suggest that largelanguage models have the potential to improve human-robot interaction forcollaborative manufacturing assembly applications. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2406.02126v1 |
|title| CityLight: A Universal Model Towards Real-world City-scale Traffic Signal Control Coordination |
|authors| Jinwei ZengChao YuXinyi YangWenxuan AoJian YuanYong LiYu WangHuazhong Yang
|links| http://arxiv.org/abs/2406.02126v1 |
|updated| 2024-06-04 09:10:14 UTC |
|summary| Traffic signal control TSC is a promising low-cost measure to enhancetransportation efficiency without affecting existing road infrastructure. Whilevarious reinforcement learning-based TSC methods have been proposed andexperimentally outperform conventional rule-based methods none of them hasbeen deployed in the real world. An essential gap lies in theoversimplification of the scenarios in terms of intersection heterogeneity androad network intricacy. To make TSC applicable in urban traffic management wetarget TSC coordination in city-scale high-authenticity road networks aimingto solve the three unique and important challenges: city-level scalabilityheterogeneity of real-world intersections and effective coordination amongintricate neighbor connections. Since optimizing multiple agents in aparameter-sharing paradigm can boost the training efficiency and help achievescalability we propose our method CityLight based on the well-acknowledgedoptimization framework parameter-sharing MAPPO. To ensure the unified policynetwork can learn to fit large-scale heterogeneous intersections and tackle theintricate between-neighbor coordination CityLight proposes a universalrepresentation module that consists of two key designs: heterogeneousintersection alignment and neighborhood impact alignment for coordination. Tofurther boost coordination CityLight adopts neighborhood-integrated rewards totransition from achieving local optimal to global optimal. Extensiveexperiments on datasets with hundreds to tens of thousands of real-worldintersections and authentic traffic demands validate the surprisingeffectiveness and generalizability of CityLight with an overall performancegain of 11.66 and a 22.59 improvement in transfer scenarios in terms ofthroughput. |


| Item |Content|
| --- |---|
|idx| 2406.02081v1 |
|title| FightLadder: A Benchmark for Competitive Multi-Agent Reinforcement Learning |
|authors| Wenzhe LiZihan DingSeth KartenChi Jin
|links| http://arxiv.org/abs/2406.02081v1 |
|updated| 2024-06-04 08:04:23 UTC |
|summary| Recent advances in reinforcement learning RL heavily rely on a variety ofwell-designed benchmarks which provide environmental platforms and consistentcriteria to evaluate existing and novel algorithms. Specifically inmulti-agent RL MARL a plethora of benchmarks based on cooperative games havespurred the development of algorithms that improve the scalability ofcooperative multi-agent systems. However for the competitive setting alightweight and open-sourced benchmark with challenging gaming dynamics andvisual inputs has not yet been established. In this work we presentFightLadder a real-time fighting game platform to empower competitive MARLresearch. Along with the platform we provide implementations ofstate-of-the-art MARL algorithms for competitive games as well as a set ofevaluation metrics to characterize the performance and exploitability ofagents. We demonstrate the feasibility of this platform by training a generalagent that consistently defeats 12 built-in characters in single-player modeand expose the difficulty of training a non-exploitable agent without humanknowledge and demonstrations in two-player mode. FightLadder providesmeticulously designed environments to address critical challenges incompetitive MARL research aiming to catalyze a new era of discovery andadvancement in the field. Videos and code athttps://sites.google.com/view/fightladder/home. |


| Item |Content|
| --- |---|
|idx| 2406.02063v1 |
|title| An agent-based model of modal choice with perception biases and habits |
|authors| Carole AdamBenoit Gaudou
|links| http://arxiv.org/abs/2406.02063v1 |
|updated| 2024-06-04 07:44:57 UTC |
|summary| This paper presents an agent-based model of mobility choice influenced byhuman factors such as habits and perception biases. It is implemented in aNetlogo simulator calibrated from results of an online survey aboutperceptions of mobility. The simulator can be played online. It allows tomodify urban infrastructure and observe modal report. |


| Item |Content|
| --- |---|
|idx| 2406.01893v1 |
|title| Large Language Model-Enabled Multi-Agent Manufacturing Systems |
|authors| Jonghan LimBirgit Vogel-HeuserIlya Kovalenko
|links| http://arxiv.org/abs/2406.01893v1 |
|updated| 2024-06-04 01:57:37 UTC |
|summary| Traditional manufacturing faces challenges adapting to dynamic environmentsand quickly responding to manufacturing changes. The use of multi-agent systemshas improved adaptability and coordination but requires further advancements inrapid human instruction comprehension operational adaptability andcoordination through natural language integration. Large language models likeGPT-3.5 and GPT-4 enhance multi-agent manufacturing systems by enabling agentsto communicate in natural language and interpret human instructions fordecision-making. This research introduces a novel framework where largelanguage models enhance the capabilities of agents in manufacturing makingthem more adaptable and capable of processing context-specific instructions. Acase study demonstrates the practical application of this framework showinghow agents can effectively communicate understand tasks and executemanufacturing processes including precise G-code allocation among agents. Thefindings highlight the importance of continuous large language modelintegration into multi-agent manufacturing systems and the development ofsophisticated agent communication protocols for a more flexible manufacturingsystem. |


| Item |Content|
| --- |---|
|idx| 2406.01853v1 |
|title| Multi-Agent Reinforcement Learning Meets Leaf Sequencing in Radiotherapy |
|authors| Riqiang GaoFlorin C. GhesuSimon ArberetShahab BasiriEsa KuuselaMartin KrausDorin ComaniciuAli Kamen
|links| http://arxiv.org/abs/2406.01853v1 |
|updated| 2024-06-03 23:55:20 UTC |
|summary| In contemporary radiotherapy planning RTP a key module leaf sequencing ispredominantly addressed by optimization-based approaches. In this paper wepropose a novel deep reinforcement learning DRL model termed as ReinforcedLeaf Sequencer RLS in a multi-agent framework for leaf sequencing. The RLSmodel offers improvements to time-consuming iterative optimization steps vialarge-scale training and can control movement patterns through the design ofreward mechanisms. We have conducted experiments on four datasets with fourmetrics and compared our model with a leading optimization sequencer. Ourfindings reveal that the proposed RLS model can achieve reduced fluencereconstruction errors and potential faster convergence when integrated in anoptimization planner. Additionally RLS has shown promising results in a fullartificial intelligence RTP pipeline. We hope this pioneer multi-agent RL leafsequencer can foster future research on machine learning for RTP. |


