# cs.CL 

| Item |Content|
| --- |---|
|idx| 2406.11839v1 |
|title| mDPO: Conditional Preference Optimization for Multimodal Large Language Models |
|authors| Fei WangWenxuan ZhouJames Y. HuangNan XuSheng ZhangHoifung PoonMuhao Chen
|links| http://arxiv.org/abs/2406.11839v1 |
|updated| 2024-06-17 17:59:58 UTC |
|summary| Direct preference optimization DPO has shown to be an effective method forlarge language model LLM alignment. Recent works have attempted to apply DPOto multimodal scenarios but have found it challenging to achieve consistentimprovement. Through a comparative experiment we identify the unconditionalpreference problem in multimodal preference optimization where the modeloverlooks the image condition. To address this problem we propose mDPO amultimodal DPO objective that prevents the over-prioritization of language-onlypreferences by also optimizing image preference. Moreover we introduce areward anchor that forces the reward to be positive for chosen responsesthereby avoiding the decrease in their likelihood -- an intrinsic problem ofrelative preference optimization. Experiments on two multimodal LLMs ofdifferent sizes and three widely used benchmarks demonstrate that mDPOeffectively addresses the unconditional preference problem in multimodalpreference optimization and significantly improves model performanceparticularly in reducing hallucination. |


| Item |Content|
| --- |---|
|idx| 2406.11830v1 |
|title| Language Modeling with Editable External Knowledge |
|authors| Belinda Z. LiEmmy LiuAlexis RossAbbas ZeitounGraham NeubigJacob Andreas
|links| http://arxiv.org/abs/2406.11830v1 |
|updated| 2024-06-17 17:59:35 UTC |
|summary| When the world changes so does the text that humans write about it. How dowe build language models that can be easily updated to reflect these changesOne popular approach is retrieval-augmented generation in which new documentsare inserted into a knowledge base and retrieved during prediction fordownstream tasks. Most prior work on these systems have focused on improvingbehavior during prediction through better retrieval or reasoning. This paperintroduces ERASE which instead improves model behavior when new documents areacquired by incrementally deleting or rewriting other entries in the knowledgebase each time a document is added. In two new benchmark datasets evaluatingmodels ability to answer questions about a stream of news articles orconversations ERASE improves accuracy relative to conventionalretrieval-augmented generation by 7-13 Mixtral-8x7B and 6-10 Llama-3-8Babsolute. Code and data are available at https://github.com/belindal/ERASE |


| Item |Content|
| --- |---|
|idx| 2406.11827v1 |
|title| WPO: Enhancing RLHF with Weighted Preference Optimization |
|authors| Wenxuan ZhouRavi AgrawalShujian ZhangSathish Reddy IndurthiSanqiang ZhaoKaiqiang SongSilei XuChenguang Zhu
|links| http://arxiv.org/abs/2406.11827v1 |
|updated| 2024-06-17 17:59:13 UTC |
|summary| Reinforcement learning from human feedback RLHF is a promising solution toalign large language models LLMs more closely with human values. Off-policypreference optimization where the preference data is obtained from othermodels is widely adopted due to its cost efficiency and scalability. Howeveroff-policy preference optimization often suffers from a distributional gapbetween the policy used for data collection and the target policy leading tosuboptimal optimization. In this paper we propose a novel strategy to mitigatethis problem by simulating on-policy learning with off-policy preference data.Our Weighted Preference Optimization WPO method adapts off-policy data toresemble on-policy data more closely by reweighting preference pairs accordingto their probability under the current policy. This method not only addressesthe distributional gap problem but also enhances the optimization processwithout incurring additional costs. We validate our method on instructionfollowing benchmarks including Alpaca Eval 2 and MT-bench. WPO not onlyoutperforms Direct Preference Optimization DPO by up to 5.6 on Alpaca Eval 2but also establishes a remarkable length-controlled winning rate againstGPT-4-turbo of 48.6 based on Llama-3-8B-Instruct making it the strongest 8Bmodel on the leaderboard. We will release the code and models athttps://github.com/wzhouad/WPO. |


| Item |Content|
| --- |---|
|idx| 2406.11823v1 |
|title| On Efficient Language and Vision Assistants for Visually-Situated Natural Language Understanding: What Matters in Reading and Reasoning |
|authors| Geewook KimMinjoon Seo
|links| http://arxiv.org/abs/2406.11823v1 |
|updated| 2024-06-17 17:57:30 UTC |
|summary| Recent advancements in language and vision assistants have showcasedimpressive capabilities but suffer from a lack of transparency limitingbroader research and reproducibility. While open-source models handle generalimage tasks effectively they face challenges with the high computationaldemands of complex visually-situated text understanding. Such tasks oftenrequire increased token inputs and large vision modules to harnesshigh-resolution information. Striking a balance between model size and dataimportance remains an open question. This study aims to redefine the design ofvision-language models by identifying key components and creating efficientmodels with constrained inference costs. By strategically formulating datasetsoptimizing vision modules and enhancing supervision techniques we achievesignificant improvements in inference throughput while maintaining highperformance. Extensive experiments across models ranging from 160M to 13Bparameters offer insights into model optimization. We will fully open-sourceour codebase models and datasets at https://github.com/naver-ai/elva . |


| Item |Content|
| --- |---|
|idx| 2406.11817v1 |
|title| Iterative Length-Regularized Direct Preference Optimization: A Case Study on Improving 7B Language Models to GPT-4 Level |
|authors| Jie LiuZhanhui ZhouJiaheng LiuXingyuan BuChao YangHan-Sen ZhongWanli Ouyang
|links| http://arxiv.org/abs/2406.11817v1 |
|updated| 2024-06-17 17:55:38 UTC |
|summary| Direct Preference Optimization DPO a standard method for aligning languagemodels with human preferences is traditionally applied to offline preferences.Recent studies show that DPO benefits from iterative training with onlinepreferences labeled by a trained reward model. In this work we identify apitfall of vanilla iterative DPO - improved response quality can lead toincreased verbosity. To address this we introduce iterative length-regularizedDPO iLR-DPO to penalize response length. Our empirical results show thatiLR-DPO can enhance a 7B model to perform on par with GPT-4 without increasingverbosity. Specifically our 7B model achieves a 50.5 length-controlled winrate against textttGPT-4 Preview on AlpacaEval 2.0 and excels acrossstandard benchmarks including MT-Bench Arena-Hard and OpenLLM Leaderboard.These results demonstrate the effectiveness of iterative DPO in aligninglanguage models with human feedback. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2406.11839v1 |
|title| mDPO: Conditional Preference Optimization for Multimodal Large Language Models |
|authors| Fei WangWenxuan ZhouJames Y. HuangNan XuSheng ZhangHoifung PoonMuhao Chen
|links| http://arxiv.org/abs/2406.11839v1 |
|updated| 2024-06-17 17:59:58 UTC |
|summary| Direct preference optimization DPO has shown to be an effective method forlarge language model LLM alignment. Recent works have attempted to apply DPOto multimodal scenarios but have found it challenging to achieve consistentimprovement. Through a comparative experiment we identify the unconditionalpreference problem in multimodal preference optimization where the modeloverlooks the image condition. To address this problem we propose mDPO amultimodal DPO objective that prevents the over-prioritization of language-onlypreferences by also optimizing image preference. Moreover we introduce areward anchor that forces the reward to be positive for chosen responsesthereby avoiding the decrease in their likelihood -- an intrinsic problem ofrelative preference optimization. Experiments on two multimodal LLMs ofdifferent sizes and three widely used benchmarks demonstrate that mDPOeffectively addresses the unconditional preference problem in multimodalpreference optimization and significantly improves model performanceparticularly in reducing hallucination. |


| Item |Content|
| --- |---|
|idx| 2406.11833v1 |
|title| MMDU: A Multi-Turn Multi-Image Dialog Understanding Benchmark and Instruction-Tuning Dataset for LVLMs |
|authors| Ziyu LiuTao ChuYuhang ZangXilin WeiXiaoyi DongPan ZhangZijian LiangYuanjun XiongYu QiaoDahua LinJiaqi Wang
|links| http://arxiv.org/abs/2406.11833v1 |
|updated| 2024-06-17 17:59:47 UTC |
|summary| Generating natural and meaningful responses to communicate with multi-modalhuman inputs is a fundamental capability of Large Vision-LanguageModelsLVLMs. While current open-source LVLMs demonstrate promisingperformance in simplified scenarios such as single-turn single-image inputthey fall short in real-world conversation scenarios such as followinginstructions in a long context history with multi-turn and multi-images.Existing LVLM benchmarks primarily focus on single-choice questions orshort-form responses which do not adequately assess the capabilities of LVLMsin real-world human-AI interaction applications. Therefore we introduce MMDUa comprehensive benchmark and MMDU-45k a large-scale instruction tuningdataset designed to evaluate and improve LVLMs abilities in multi-turn andmulti-image conversations. We employ the clustering algorithm to ffnd therelevant images and textual descriptions from the open-source Wikipedia andconstruct the question-answer pairs by human annotators with the assistance ofthe GPT-4o model. MMDU has a maximum of 18k imagetext tokens 20 images and27 turns which is at least 5x longer than previous benchmarks and poseschallenges to current LVLMs. Our in-depth analysis of 15 representative LVLMsusing MMDU reveals that open-source LVLMs lag behind closed-source counterpartsdue to limited conversational instruction tuning data. We demonstrate thatffne-tuning open-source LVLMs on MMDU-45k signiffcantly address this gapgenerating longer and more accurate conversations and improving scores on MMDUand existing benchmarks MMStar: 1.1 MathVista: 1.5 ChartQA:1.2. Ourcontributions pave the way for bridging the gap between current LVLM models andreal-world application demands. This project is available athttps://github.com/Liuziyu77/MMDU. |


| Item |Content|
| --- |---|
|idx| 2406.11830v1 |
|title| Language Modeling with Editable External Knowledge |
|authors| Belinda Z. LiEmmy LiuAlexis RossAbbas ZeitounGraham NeubigJacob Andreas
|links| http://arxiv.org/abs/2406.11830v1 |
|updated| 2024-06-17 17:59:35 UTC |
|summary| When the world changes so does the text that humans write about it. How dowe build language models that can be easily updated to reflect these changesOne popular approach is retrieval-augmented generation in which new documentsare inserted into a knowledge base and retrieved during prediction fordownstream tasks. Most prior work on these systems have focused on improvingbehavior during prediction through better retrieval or reasoning. This paperintroduces ERASE which instead improves model behavior when new documents areacquired by incrementally deleting or rewriting other entries in the knowledgebase each time a document is added. In two new benchmark datasets evaluatingmodels ability to answer questions about a stream of news articles orconversations ERASE improves accuracy relative to conventionalretrieval-augmented generation by 7-13 Mixtral-8x7B and 6-10 Llama-3-8Babsolute. Code and data are available at https://github.com/belindal/ERASE |


| Item |Content|
| --- |---|
|idx| 2406.11827v1 |
|title| WPO: Enhancing RLHF with Weighted Preference Optimization |
|authors| Wenxuan ZhouRavi AgrawalShujian ZhangSathish Reddy IndurthiSanqiang ZhaoKaiqiang SongSilei XuChenguang Zhu
|links| http://arxiv.org/abs/2406.11827v1 |
|updated| 2024-06-17 17:59:13 UTC |
|summary| Reinforcement learning from human feedback RLHF is a promising solution toalign large language models LLMs more closely with human values. Off-policypreference optimization where the preference data is obtained from othermodels is widely adopted due to its cost efficiency and scalability. Howeveroff-policy preference optimization often suffers from a distributional gapbetween the policy used for data collection and the target policy leading tosuboptimal optimization. In this paper we propose a novel strategy to mitigatethis problem by simulating on-policy learning with off-policy preference data.Our Weighted Preference Optimization WPO method adapts off-policy data toresemble on-policy data more closely by reweighting preference pairs accordingto their probability under the current policy. This method not only addressesthe distributional gap problem but also enhances the optimization processwithout incurring additional costs. We validate our method on instructionfollowing benchmarks including Alpaca Eval 2 and MT-bench. WPO not onlyoutperforms Direct Preference Optimization DPO by up to 5.6 on Alpaca Eval 2but also establishes a remarkable length-controlled winning rate againstGPT-4-turbo of 48.6 based on Llama-3-8B-Instruct making it the strongest 8Bmodel on the leaderboard. We will release the code and models athttps://github.com/wzhouad/WPO. |


| Item |Content|
| --- |---|
|idx| 2406.11818v1 |
|title| Embodied Instruction Following in Unknown Environments |
|authors| Zhenyu WuZiwei WangXiuwei XuJiwen LuHaibin Yan
|links| http://arxiv.org/abs/2406.11818v1 |
|updated| 2024-06-17 17:55:40 UTC |
|summary| Enabling embodied agents to complete complex human instructions from naturallanguage is crucial to autonomous systems in household services. Conventionalmethods can only accomplish human instructions in the known environment whereall interactive objects are provided to the embodied agent and directlydeploying the existing approaches for the unknown environment usually generatesinfeasible plans that manipulate non-existing objects. On the contrary wepropose an embodied instruction following EIF method for complex tasks in theunknown environment where the agent efficiently explores the unknownenvironment to generate feasible plans with existing objects to accomplishabstract instructions. Specifically we build a hierarchical embodiedinstruction following framework including the high-level task planner and thelow-level exploration controller with multimodal large language models. We thenconstruct a semantic representation map of the scene with dynamic regionattention to demonstrate the known visual clues where the goal of taskplanning and scene exploration is aligned for human instruction. For the taskplanner we generate the feasible step-by-step plans for human goalaccomplishment according to the task completion process and the known visualclues. For the exploration controller the optimal navigation or objectinteraction policy is predicted based on the generated step-wise plans and theknown visual clues. The experimental results demonstrate that our method canachieve 45.09 success rate in 204 complex human instructions such as makingbreakfast and tidying rooms in large house-level scenes. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2406.11839v1 |
|title| mDPO: Conditional Preference Optimization for Multimodal Large Language Models |
|authors| Fei WangWenxuan ZhouJames Y. HuangNan XuSheng ZhangHoifung PoonMuhao Chen
|links| http://arxiv.org/abs/2406.11839v1 |
|updated| 2024-06-17 17:59:58 UTC |
|summary| Direct preference optimization DPO has shown to be an effective method forlarge language model LLM alignment. Recent works have attempted to apply DPOto multimodal scenarios but have found it challenging to achieve consistentimprovement. Through a comparative experiment we identify the unconditionalpreference problem in multimodal preference optimization where the modeloverlooks the image condition. To address this problem we propose mDPO amultimodal DPO objective that prevents the over-prioritization of language-onlypreferences by also optimizing image preference. Moreover we introduce areward anchor that forces the reward to be positive for chosen responsesthereby avoiding the decrease in their likelihood -- an intrinsic problem ofrelative preference optimization. Experiments on two multimodal LLMs ofdifferent sizes and three widely used benchmarks demonstrate that mDPOeffectively addresses the unconditional preference problem in multimodalpreference optimization and significantly improves model performanceparticularly in reducing hallucination. |


| Item |Content|
| --- |---|
|idx| 2406.11833v1 |
|title| MMDU: A Multi-Turn Multi-Image Dialog Understanding Benchmark and Instruction-Tuning Dataset for LVLMs |
|authors| Ziyu LiuTao ChuYuhang ZangXilin WeiXiaoyi DongPan ZhangZijian LiangYuanjun XiongYu QiaoDahua LinJiaqi Wang
|links| http://arxiv.org/abs/2406.11833v1 |
|updated| 2024-06-17 17:59:47 UTC |
|summary| Generating natural and meaningful responses to communicate with multi-modalhuman inputs is a fundamental capability of Large Vision-LanguageModelsLVLMs. While current open-source LVLMs demonstrate promisingperformance in simplified scenarios such as single-turn single-image inputthey fall short in real-world conversation scenarios such as followinginstructions in a long context history with multi-turn and multi-images.Existing LVLM benchmarks primarily focus on single-choice questions orshort-form responses which do not adequately assess the capabilities of LVLMsin real-world human-AI interaction applications. Therefore we introduce MMDUa comprehensive benchmark and MMDU-45k a large-scale instruction tuningdataset designed to evaluate and improve LVLMs abilities in multi-turn andmulti-image conversations. We employ the clustering algorithm to ffnd therelevant images and textual descriptions from the open-source Wikipedia andconstruct the question-answer pairs by human annotators with the assistance ofthe GPT-4o model. MMDU has a maximum of 18k imagetext tokens 20 images and27 turns which is at least 5x longer than previous benchmarks and poseschallenges to current LVLMs. Our in-depth analysis of 15 representative LVLMsusing MMDU reveals that open-source LVLMs lag behind closed-source counterpartsdue to limited conversational instruction tuning data. We demonstrate thatffne-tuning open-source LVLMs on MMDU-45k signiffcantly address this gapgenerating longer and more accurate conversations and improving scores on MMDUand existing benchmarks MMStar: 1.1 MathVista: 1.5 ChartQA:1.2. Ourcontributions pave the way for bridging the gap between current LVLM models andreal-world application demands. This project is available athttps://github.com/Liuziyu77/MMDU. |


| Item |Content|
| --- |---|
|idx| 2406.11828v1 |
|title| Learning sum of diverse features: computational hardness and efficient gradient-based training for ridge combinations |
|authors| Kazusato OkoYujin SongTaiji SuzukiDenny Wu
|links| http://arxiv.org/abs/2406.11828v1 |
|updated| 2024-06-17 17:59:17 UTC |
|summary| We study the computational and sample complexity of learning a targetfunction f_:mathbbRdtomathbbR with additive structure that isf_x  frac1sqrtMsum_m1M f_mlangle x v_mrangle wheref_1f_2...f_M:mathbbRtomathbbR are nonlinear link functions ofsingle-index models ridge functions with diverse and near-orthogonal indexfeatures v_m_m1M and the number of additive tasks M grows with thedimensionality Masymp dgamma for gammage 0. This problem setting ismotivated by the classical additive model literature the recent representationlearning theory of two-layer neural network and large-scale pretraining wherethe model simultaneously acquires a large number of skills that are oftenlocalized in distinct parts of the trained network. We prove that a largesubset of polynomial f_ can be efficiently learned by gradient descenttraining of a two-layer neural network with a polynomial statistical andcomputational complexity that depends on the number of tasks M and theinformation exponent of f_m despite the unknown link function and Mgrowing with the dimensionality. We complement this learnability guarantee withcomputational hardness result by establishing statistical query SQ lowerbounds for both the correlational SQ and full SQ algorithms. |


| Item |Content|
| --- |---|
|idx| 2406.11827v1 |
|title| WPO: Enhancing RLHF with Weighted Preference Optimization |
|authors| Wenxuan ZhouRavi AgrawalShujian ZhangSathish Reddy IndurthiSanqiang ZhaoKaiqiang SongSilei XuChenguang Zhu
|links| http://arxiv.org/abs/2406.11827v1 |
|updated| 2024-06-17 17:59:13 UTC |
|summary| Reinforcement learning from human feedback RLHF is a promising solution toalign large language models LLMs more closely with human values. Off-policypreference optimization where the preference data is obtained from othermodels is widely adopted due to its cost efficiency and scalability. Howeveroff-policy preference optimization often suffers from a distributional gapbetween the policy used for data collection and the target policy leading tosuboptimal optimization. In this paper we propose a novel strategy to mitigatethis problem by simulating on-policy learning with off-policy preference data.Our Weighted Preference Optimization WPO method adapts off-policy data toresemble on-policy data more closely by reweighting preference pairs accordingto their probability under the current policy. This method not only addressesthe distributional gap problem but also enhances the optimization processwithout incurring additional costs. We validate our method on instructionfollowing benchmarks including Alpaca Eval 2 and MT-bench. WPO not onlyoutperforms Direct Preference Optimization DPO by up to 5.6 on Alpaca Eval 2but also establishes a remarkable length-controlled winning rate againstGPT-4-turbo of 48.6 based on Llama-3-8B-Instruct making it the strongest 8Bmodel on the leaderboard. We will release the code and models athttps://github.com/wzhouad/WPO. |


| Item |Content|
| --- |---|
|idx| 2406.11825v1 |
|title| Spectral Introspection Identifies Group Training Dynamics in Deep Neural Networks for Neuroimaging |
|authors| Bradley T. BakerVince D. CalhounSergey M. Plis
|links| http://arxiv.org/abs/2406.11825v1 |
|updated| 2024-06-17 17:58:15 UTC |
|summary| Neural networks whice have had a profound effect on how researchers studycomplex phenomena do so through a complex nonlinear mathematical structurewhich can be difficult for human researchers to interpret. This obstacle can beespecially salient when researchers want to better understand the emergence ofparticular model behaviors such as bias overfitting overparametrization andmore. In Neuroimaging the understanding of how such phenomena emerge isfundamental to preventing and informing users of the potential risks involvedin practice. In this work we present a novel introspection framework for DeepLearning on Neuroimaging data which exploits the natural structure of gradientcomputations via the singular value decomposition of gradient components duringreverse-mode auto-differentiation. Unlike post-hoc introspection techniqueswhich require fully-trained models for evaluation our method allows for thestudy of training dynamics on the fly and even more interestingly allow forthe decomposition of gradients based on which samples belong to particulargroups of interest. We demonstrate how the gradient spectra for several commondeep learning models differ between schizophrenia and control participants fromthe COBRE study and illustrate how these trajectories may reveal specifictraining dynamics helpful for further analysis. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2406.11840v1 |
|title| LLaNA: Large Language and NeRF Assistant |
|authors| Andrea AmaduzziPierluigi Zama RamirezGiuseppe LisantiSamuele SaltiLuigi Di Stefano
|links| http://arxiv.org/abs/2406.11840v1 |
|updated| 2024-06-17 17:59:59 UTC |
|summary| Multimodal Large Language Models MLLMs have demonstrated an excellentunderstanding of images and 3D data. However both modalities have shortcomingsin holistically capturing the appearance and geometry of objects. MeanwhileNeural Radiance Fields NeRFs which encode information within the weights ofa simple Multi-Layer Perceptron MLP have emerged as an increasinglywidespread modality that simultaneously encodes the geometry and photorealisticappearance of objects. This paper investigates the feasibility andeffectiveness of ingesting NeRF into MLLM. We create LLaNA the firstgeneral-purpose NeRF-language assistant capable of performing new tasks such asNeRF captioning and QA. Notably our method directly processes the weights ofthe NeRFs MLP to extract information about the represented objects without theneed to render images or materialize 3D data structures. Moreover we build adataset of NeRFs with text annotations for various NeRF-language tasks with nohuman intervention. Based on this dataset we develop a benchmark to evaluatethe NeRF understanding capability of our method. Results show that processingNeRF weights performs favourably against extracting 2D or 3D representationsfrom NeRFs. |


| Item |Content|
| --- |---|
|idx| 2406.11838v1 |
|title| Autoregressive Image Generation without Vector Quantization |
|authors| Tianhong LiYonglong TianHe LiMingyang DengKaiming He
|links| http://arxiv.org/abs/2406.11838v1 |
|updated| 2024-06-17 17:59:58 UTC |
|summary| Conventional wisdom holds that autoregressive models for image generation aretypically accompanied by vector-quantized tokens. We observe that while adiscrete-valued space can facilitate representing a categorical distributionit is not a necessity for autoregressive modeling. In this work we propose tomodel the per-token probability distribution using a diffusion procedure whichallows us to apply autoregressive models in a continuous-valued space. Ratherthan using categorical cross-entropy loss we define a Diffusion Loss functionto model the per-token probability. This approach eliminates the need fordiscrete-valued tokenizers. We evaluate its effectiveness across a wide rangeof cases including standard autoregressive models and generalized maskedautoregressive MAR variants. By removing vector quantization our imagegenerator achieves strong results while enjoying the speed advantage ofsequence modeling. We hope this work will motivate the use of autoregressivegeneration in other continuous-valued domains and applications. |


| Item |Content|
| --- |---|
|idx| 2406.11839v1 |
|title| mDPO: Conditional Preference Optimization for Multimodal Large Language Models |
|authors| Fei WangWenxuan ZhouJames Y. HuangNan XuSheng ZhangHoifung PoonMuhao Chen
|links| http://arxiv.org/abs/2406.11839v1 |
|updated| 2024-06-17 17:59:58 UTC |
|summary| Direct preference optimization DPO has shown to be an effective method forlarge language model LLM alignment. Recent works have attempted to apply DPOto multimodal scenarios but have found it challenging to achieve consistentimprovement. Through a comparative experiment we identify the unconditionalpreference problem in multimodal preference optimization where the modeloverlooks the image condition. To address this problem we propose mDPO amultimodal DPO objective that prevents the over-prioritization of language-onlypreferences by also optimizing image preference. Moreover we introduce areward anchor that forces the reward to be positive for chosen responsesthereby avoiding the decrease in their likelihood -- an intrinsic problem ofrelative preference optimization. Experiments on two multimodal LLMs ofdifferent sizes and three widely used benchmarks demonstrate that mDPOeffectively addresses the unconditional preference problem in multimodalpreference optimization and significantly improves model performanceparticularly in reducing hallucination. |


| Item |Content|
| --- |---|
|idx| 2406.11837v1 |
|title| Scaling the Codebook Size of VQGAN to 100,000 with a Utilization Rate of 99% |
|authors| Lei ZhuFangyun WeiYanye LuDong Chen
|links| http://arxiv.org/abs/2406.11837v1 |
|updated| 2024-06-17 17:59:57 UTC |
|summary| In the realm of image quantization exemplified by VQGAN the process encodesimages into discrete tokens drawn from a codebook with a predefined size.Recent advancements particularly with LLAMA 3 reveal that enlarging thecodebook significantly enhances model performance. However VQGAN and itsderivatives such as VQGAN-FC Factorized Codes and VQGAN-EMA continue tograpple with challenges related to expanding the codebook size and enhancingcodebook utilization. For instance VQGAN-FC is restricted to learning acodebook with a maximum size of 16384 maintaining a typically low utilizationrate of less than 12 on ImageNet. In this work we propose a novel imagequantization model named VQGAN-LC Large Codebook which extends the codebooksize to 100000 achieving an utilization rate exceeding 99. Unlike previousmethods that optimize each codebook entry our approach begins with a codebookinitialized with 100000 features extracted by a pre-trained vision encoder.Optimization then focuses on training a projector that aligns the entirecodebook with the feature distributions of the encoder in VQGAN-LC. Wedemonstrate the superior performance of our model over its counterparts acrossa variety of tasks including image reconstruction image classificationauto-regressive image generation using GPT and image creation with diffusion-and flow-based generative models. Code and models are available athttps://github.com/zh460045050/VQGAN-LC. |


| Item |Content|
| --- |---|
|idx| 2406.11835v1 |
|title| OoDIS: Anomaly Instance Segmentation Benchmark |
|authors| Alexey NekrasovRui ZhouMiriam AckermannAlexander HermansBastian LeibeMatthias Rottmann
|links| http://arxiv.org/abs/2406.11835v1 |
|updated| 2024-06-17 17:59:56 UTC |
|summary| Autonomous vehicles require a precise understanding of their environment tonavigate safely. Reliable identification of unknown objects especially thosethat are absent during training such as wild animals is critical due to theirpotential to cause serious accidents. Significant progress in semanticsegmentation of anomalies has been driven by the availability ofout-of-distribution OOD benchmarks. However a comprehensive understanding ofscene dynamics requires the segmentation of individual objects and thus thesegmentation of instances is essential. Development in this area has beenlagging largely due to the lack of dedicated benchmarks. To address this gapwe have extended the most commonly used anomaly segmentation benchmarks toinclude the instance segmentation task. Our evaluation of anomaly instancesegmentation methods shows that this challenge remains an unsolved problem. Thebenchmark website and the competition page can be found at:https://vision.rwth-aachen.de/oodis . |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2406.11828v1 |
|title| Learning sum of diverse features: computational hardness and efficient gradient-based training for ridge combinations |
|authors| Kazusato OkoYujin SongTaiji SuzukiDenny Wu
|links| http://arxiv.org/abs/2406.11828v1 |
|updated| 2024-06-17 17:59:17 UTC |
|summary| We study the computational and sample complexity of learning a targetfunction f_:mathbbRdtomathbbR with additive structure that isf_x  frac1sqrtMsum_m1M f_mlangle x v_mrangle wheref_1f_2...f_M:mathbbRtomathbbR are nonlinear link functions ofsingle-index models ridge functions with diverse and near-orthogonal indexfeatures v_m_m1M and the number of additive tasks M grows with thedimensionality Masymp dgamma for gammage 0. This problem setting ismotivated by the classical additive model literature the recent representationlearning theory of two-layer neural network and large-scale pretraining wherethe model simultaneously acquires a large number of skills that are oftenlocalized in distinct parts of the trained network. We prove that a largesubset of polynomial f_ can be efficiently learned by gradient descenttraining of a two-layer neural network with a polynomial statistical andcomputational complexity that depends on the number of tasks M and theinformation exponent of f_m despite the unknown link function and Mgrowing with the dimensionality. We complement this learnability guarantee withcomputational hardness result by establishing statistical query SQ lowerbounds for both the correlational SQ and full SQ algorithms. |


| Item |Content|
| --- |---|
|idx| 2406.11814v1 |
|title| Stochastic Neural Network Symmetrisation in Markov Categories |
|authors| Rob Cornish
|links| http://arxiv.org/abs/2406.11814v1 |
|updated| 2024-06-17 17:54:42 UTC |
|summary| We consider the problem of symmetrising a neural network along a grouphomomorphism: given a homomorphism varphi : H to G we would like aprocedure that converts H-equivariant neural networks into G-equivariantones. We formulate this in terms of Markov categories which allows us toconsider neural networks whose outputs may be stochastic but withmeasure-theoretic details abstracted away. We obtain a flexible compositionaland generic framework for symmetrisation that relies on minimal assumptionsabout the structure of the group and the underlying neural networkarchitecture. Our approach recovers existing methods for deterministicsymmetrisation as special cases and extends directly to provide a novelmethodology for stochastic symmetrisation also. Beyond this we believe ourfindings also demonstrate the utility of Markov categories for addressingproblems in machine learning in a conceptual yet mathematically rigorous way. |


| Item |Content|
| --- |---|
|idx| 2406.11803v1 |
|title| Efficient Discovery of Significant Patterns with Few-Shot Resampling |
|authors| Leonardo PellegrinaFabio Vandin
|links| http://arxiv.org/abs/2406.11803v1 |
|updated| 2024-06-17 17:49:27 UTC |
|summary| Significant pattern mining is a fundamental task in mining transactionaldata requiring to identify patterns significantly associated with the value ofa given feature the target. In several applications such as biomedicinebasket market analysis and social networks the goal is to discover patternswhose association with the target is defined with respect to an underlyingpopulation or process of which the dataset represents only a collection ofobservations or samples. A natural way to capture the association of a patternwith the target is to consider its statistical significance assessing itsdeviation from the null hypothesis of independence between the pattern andthe target. While several algorithms have been proposed to find statisticallysignificant patterns it remains a computationally demanding task and forcomplex patterns such as subgroups no efficient solution exists.  We present FSR an efficient algorithm to identify statistically significantpatterns with rigorous guarantees on the probability of false discoveries. FSRbuilds on a novel general framework for mining significant patterns thatcaptures some of the most commonly considered patterns including itemsetssequential patterns and subgroups. FSR uses a small number of resampleddatasets obtained by assigning i.i.d. labels to each transaction torigorously bound the supremum deviation of a quality statistic measuring thesignificance of patterns. FSR builds on novel tight bounds on the supremumdeviation that require to mine a small number of resampled datasets whileproviding a high effectiveness in discovering significant patterns. As a testcase we consider significant subgroup mining and our evaluation on severalreal datasets shows that FSR is effective in discovering significant subgroupswhile requiring a small number of resampled datasets. |


| Item |Content|
| --- |---|
|idx| 2406.11761v1 |
|title| Joint Linked Component Analysis for Multiview Data |
|authors| Lin XiaoLuo Xiao
|links| http://arxiv.org/abs/2406.11761v1 |
|updated| 2024-06-17 17:25:23 UTC |
|summary| In this work we propose the joint linked component analysis joint_LCA formultiview data. Unlike classic methods which extract the shared components in asequential manner the objective of joint_LCA is to identify the view-specificloading matrices and the rank of the common latent subspace simultaneously. Weformulate a matrix decomposition model where a joint structure and anindividual structure are present in each data view which enables us to arriveat a clean svd representation for the cross covariance between any pair of dataviews. An objective function with a novel penalty term is then proposed toachieve simultaneous estimation and rank selection. In addition a refittingprocedure is employed as a remedy to reduce the shrinkage bias caused by thepenalization. |


| Item |Content|
| --- |---|
|idx| 2406.11733v1 |
|title| A Clipped Trip: the Dynamics of SGD with Gradient Clipping in High-Dimensions |
|authors| Noah MarshallKe Liang XiaoAtish AgarwalaElliot Paquette
|links| http://arxiv.org/abs/2406.11733v1 |
|updated| 2024-06-17 16:50:22 UTC |
|summary| The success of modern machine learning is due in part to the adaptiveoptimization methods that have been developed to deal with the difficulties oftraining large models over complex datasets. One such method is gradientclipping: a practical procedure with limited theoretical underpinnings. In thiswork we study clipping in a least squares problem under streaming SGD. Wedevelop a theoretical analysis of the learning dynamics in the limit of largeintrinsic dimension-a model and dataset dependent notion of dimensionality. Inthis limit we find a deterministic equation that describes the evolution of theloss. We show that with Gaussian noise clipping cannot improve SGD performance.Yet in other noisy settings clipping can provide benefits with tuning of theclipping threshold. In these cases clipping biases updates in a way beneficialto training which cannot be recovered by SGD under any schedule. We concludewith a discussion about the links between high-dimensional clipping and neuralnetwork training. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2406.11759v1 |
|title| Folk-ontological stances toward robots and psychological human likeness |
|authors| Edoardo Datteri
|links| http://arxiv.org/abs/2406.11759v1 |
|updated| 2024-06-17 17:23:17 UTC |
|summary| It has often been argued that people can attribute mental states to robotswithout making any ontological commitments to the reality of those states. Butwhat does it mean to attribute a mental state to a robot and what is anontological commitment It will be argued that on a plausible interpretationof these two notions it is not clear how mental state attribution can occurwithout any ontological commitment. Taking inspiration from the philosophicaldebate on scientific realism a provisional taxonomy of folk-ontologicalstances towards robots will also be identified corresponding to different waysof understanding robotic minds. They include realism non-realismeliminativism reductionism fictionalism and agnosticism. Instrumentalism willalso be discussed and presented as a folk-epistemological stance. In the lastpart of the article it will be argued that peoples folk-ontological stancestowards robots and humans can influence their perception of the human-likenessof robots. The analysis carried out here can be seen as encouraging afolk-ontological turn in human-robot interaction research aimed atexplicitly determining what beliefs people have about the reality of robotminds. |


| Item |Content|
| --- |---|
|idx| 2406.11757v1 |
|title| STAR: SocioTechnical Approach to Red Teaming Language Models |
|authors| Laura WeidingerJohn MellorBernat Guillen PeguerolesNahema MarchalRavin KumarKristian LumCanfer AkbulutMark DiazStevie BergmanMikel RodriguezVerena RieserWilliam Isaac
|links| http://arxiv.org/abs/2406.11757v1 |
|updated| 2024-06-17 17:16:45 UTC |
|summary| This research introduces STAR a sociotechnical framework that improves oncurrent best practices for red teaming safety of large language models. STARmakes two key contributions: it enhances steerability by generatingparameterised instructions for human red teamers leading to improved coverageof the risk surface. Parameterised instructions also provide more detailedinsights into model failures at no increased cost. Second STAR improves signalquality by matching demographics to assess harms for specific groups resultingin more sensitive annotations. STAR further employs a novel step of arbitrationto leverage diverse viewpoints and improve label reliability treatingdisagreement not as noise but as a valuable contribution to signal quality. |


| Item |Content|
| --- |---|
|idx| 2406.11645v1 |
|title| SeamPose: Repurposing Seams as Capacitive Sensors in a Shirt for Upper-Body Pose Tracking |
|authors| Tianhong Catherine YuManruZhangPeter HeChi-Jung LeeCassidy CheesmanSaif MahmudRuidong ZhangFrançois GuimbretièreCheng Zhang
|links| http://arxiv.org/abs/2406.11645v1 |
|updated| 2024-06-17 15:28:35 UTC |
|summary| Seams are areas of overlapping fabric formed by stitching two or more piecesof fabric together in the cut-and-sew apparel manufacturing process. InSeamPose we repurposed seams as capacitive sensors in a shirt for continuousupper-body pose estimation. Compared to previous all-textile motion-capturinggarments that place the electrodes on the surface of clothing our solutionleverages existing seams inside of a shirt by machine-sewing insulatedconductive threads over the seams. The unique invisibilities and placements ofthe seams afford the sensing shirt to look and wear the same as a conventionalshirt while providing exciting pose-tracking capabilities. To validate thisapproach we implemented a proof-of-concept untethered shirt. With eightcapacitive sensing seams our customized deep-learning pipeline accuratelyestimates the upper-body 3D joint positions relative to the pelvis. With a12-participant user study we demonstrated promising cross-user andcross-session tracking performance. SeamPose represents a step towardsunobtrusive integration of smart clothing for everyday pose estimation. |


| Item |Content|
| --- |---|
|idx| 2406.11637v1 |
|title| PyGWalker: On-the-fly Assistant for Exploratory Visual Data Analysis |
|authors| Yue YuLeixian ShenFei LongHuamin QuHao Chen
|links| http://arxiv.org/abs/2406.11637v1 |
|updated| 2024-06-17 15:16:32 UTC |
|summary| Exploratory visual data analysis tools empower data analysts to efficientlyand intuitively explore data insights throughout the entire analysis cycle.However the gap between common programmatic analysis e.g. withincomputational notebooks and exploratory visual analysis leads to a disjointedand inefficient data analysis experience. To bridge this gap we developedPyGWalker a Python library that offers on-the-fly assistance for exploratoryvisual data analysis. It features a lightweight and intuitive GUI with a shelfbuilder modality. Its loosely coupled architecture supports multiplecomputational environments to accommodate varying data sizes. Since its releasein February 2023 PyGWalker has gained much attention with 612k downloads onPyPI and over 10.5k stars on GitHub as of June 2024. This demonstrates itsvalue to the data science and visualization community with researchers anddevelopers integrating it into their own applications and studies. |


| Item |Content|
| --- |---|
|idx| 2406.11500v1 |
|title| ESI-GAL: EEG Source Imaging-based Kinemaics Parameter Estimation for Grasp and Lift Task |
|authors| Anant JainLalan Kumar
|links| http://arxiv.org/abs/2406.11500v1 |
|updated| 2024-06-17 13:02:40 UTC |
|summary| Objective: Electroencephalogram EEG signals-based motor kinematicsprediction MKP has been an active area of research to develop brain-computerinterface BCI systems such as exosuits prostheses and rehabilitationdevices. However EEG source imaging ESI based kinematics prediction issparsely explored in the literature. Approach: In this study pre-movement EEGfeatures are utilized to predict three-dimensional 3D hand kinematics for thegrasp-and-lift motor task. A public dataset WAY-EEG-GAL is utilized for MKPanalysis. In particular sensor-domain EEG data and source-domain ESI databased features from the frontoparietal region are explored for MKP. Deeplearning-based models are explored to achieve efficient kinematics decoding.Various time-lagged and window sizes are analyzed for hand kinematicsprediction. Subsequently intra-subject and inter-subject MKP analysis isperformed to investigate the subject-specific and subject-independentmotor-learning capabilities of the neural decoders. The Pearson correlationcoefficient PCC is used as the performance metric for kinematics trajectorydecoding. Main results: The rEEGNet neural decoder achieved the bestperformance with sensor-domain and source-domain features with the time lag andwindow size of 100 ms and 450 ms respectively. The highest mean PCC values of0.790 0.795 and 0.637 are achieved using sensor-domain features while 0.7690.777 and 0.647 are achieved using source-domain features in x y andz-directions respectively. Significance: This study explores the feasibilityof trajectory prediction using EEG sensor-domain and source-domain EEG featuresfor the grasp-and-lift task. Furthermore inter-subject trajectory estimationis performed using the proposed deep learning decoder with EEG source domainfeatures. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2406.11709v1 |
|title| Instruct, Not Assist: LLM-based Multi-Turn Planning and Hierarchical Questioning for Socratic Code Debugging |
|authors| Priyanka KarguptaIshika AgarwalDilek Hakkani-TurJiawei Han
|links| http://arxiv.org/abs/2406.11709v1 |
|updated| 2024-06-17 16:28:21 UTC |
|summary| Socratic questioning is an effective teaching strategy encouraging criticalthinking and problem-solving. The conversational capabilities of large languagemodels LLMs show great potential for providing scalable real-time studentguidance. However current LLMs often give away solutions directly making themineffective instructors. We tackle this issue in the code debugging domain withTreeInstruct an Instructor agent guided by a novel state space-based planningalgorithm. TreeInstruct asks probing questions to help students independentlyidentify and resolve errors. It estimates a students conceptual andsyntactical knowledge to dynamically construct a question tree based on theirresponses and current knowledge state effectively addressing both independentand dependent mistakes concurrently in a multi-turn interaction setting. Inaddition to using an existing single-bug debugging benchmark we construct amore challenging multi-bug dataset of 150 coding problems incorrect solutionsand bug fixes -- all carefully constructed and annotated by experts. Extensiveevaluation shows TreeInstructs state-of-the-art performance on both datasetsproving it to be a more effective instructor than baselines. Furthermore areal-world case study with five students of varying skill levels furtherdemonstrates TreeInstructs ability to guide students to debug their codeefficiently with minimal turns and highly Socratic questioning. |


| Item |Content|
| --- |---|
|idx| 2406.11496v1 |
|title| Decentralized Collaborative Pricing and Shunting for Multiple EV Charging Stations Based on Multi-Agent Reinforcement Learning |
|authors| Tianhao BuHang LiGuojie Li
|links| http://arxiv.org/abs/2406.11496v1 |
|updated| 2024-06-17 12:59:30 UTC |
|summary| The extraordinary electric vehicle EV popularization in the recent yearshas facilitated research studies in alleviating EV energy charging demand.Previous studies primarily focused on the optimizations over charging stationsCS profit and EV users cost savings through charge/discharge schedulingevents. In this work the random behaviors of EVs are considered with EV userspreferences over multi-CS characteristics modelled to imitate the potential CSselection disequilibrium. A price scheduling strategy under decentralizedcollaborative framework is proposed to achieve EV shunting in a multi-CSenvironment while minimizing the charging cost through multi agentreinforcement learning. The proposed problem is formulated as a Markov DecisionProcess MDP with uncertain transition probability. |


| Item |Content|
| --- |---|
|idx| 2406.11342v1 |
|title| KAOS: Large Model Multi-Agent Operating System |
|authors| Zhao ZhuoRongzhen LiKai LiuHuhai ZouKaiMao LiJie YuTianhao SunQingbo Wu
|links| http://arxiv.org/abs/2406.11342v1 |
|updated| 2024-06-17 08:59:32 UTC |
|summary| The intelligent interaction model based on large models reduces thedifferences in user experience across various system platforms but faceschallenges in multi-agent collaboration and resource sharing. To demonstrate auniform user experience across different foundational software platforms andaddress resource coordination management challenges this paper proposes amulti-agent operating system based on the open-source Kylin. The researchmethod involves empowering agents with large models to serve applications.First by introducing management role agents and vertical multi-agentcollaboration to construct or replace typical application software. Second bystudying system-level shared resource scheduling strategies to enhance userexperience and optimize resource utilization. And finally by validating theefficiency and superiority of the large model multi-agent operating systemthrough real applications and scoring intelligence. The feasibility of thissystem is demonstrated providing a new perspective for the development ofmulti-agent operating systems. Experimental results show significant advantagesof multi-agent collaboration in various application scenarios. |


| Item |Content|
| --- |---|
|idx| 2406.11318v1 |
|title| Reconfigurable Intelligent Surface Assisted VEC Based on Multi-Agent Reinforcement Learning |
|authors| Kangwei QiQiong WuPingyi FanNan ChengQiang FanJiangzhou Wang
|links| http://arxiv.org/abs/2406.11318v1 |
|updated| 2024-06-17 08:35:32 UTC |
|summary| Vehicular edge computing VEC is an emerging technology that enablesvehicles to perform high-intensity tasks by executing tasks locally oroffloading them to nearby edge devices. However obstacles such as buildingsmay degrade the communications and incur communication interruptions and thusthe vehicle may not meet the requirement for task offloading. Reconfigurableintelligent surfaces RIS is introduced to support vehicle communication andprovide an alternative communication path. The system performance can beimproved by flexibly adjusting the phase-shift of the RIS. For RIS-assisted VECsystem where tasks arrive randomly we design a control scheme that considersoffloading power local power allocation and phase-shift optimization. To solvethis non-convex problem we propose a new deep reinforcement learning DRLframework that employs modified multi-agent deep deterministic policy gradientMADDPG approach to optimize the power allocation for vehicle users VUs andblock coordinate descent BCD algorithm to optimize the phase-shift of theRIS. Simulation results show that our proposed scheme outperforms thecentralized deep deterministic policy gradient DDPG scheme and random scheme. |


| Item |Content|
| --- |---|
|idx| 2406.11240v1 |
|title| The Benefits of Power Regularization in Cooperative Reinforcement Learning |
|authors| Michelle LiMichael Dennis
|links| http://arxiv.org/abs/2406.11240v1 |
|updated| 2024-06-17 06:10:37 UTC |
|summary| Cooperative Multi-Agent Reinforcement Learning MARL algorithms trainedonly to optimize task reward can lead to a concentration of power where thefailure or adversarial intent of a single agent could decimate the reward ofevery agent in the system. In the context of teams of people it is oftenuseful to explicitly consider how power is distributed to ensure no personbecomes a single point of failure. Here we argue that explicitly regularizingthe concentration of power in cooperative RL systems can result in systemswhich are more robust to single agent failure adversarial attacks andincentive changes of co-players. To this end we define a practical pairwisemeasure of power that captures the ability of any co-player to influence theego agents reward and then propose a power-regularized objective whichbalances task reward and power concentration. Given this new objective we showthat there always exists an equilibrium where every agent is playing apower-regularized best-response balancing power and task reward. Moreover wepresent two algorithms for training agents towards this power-regularizedobjective: Sample Based Power Regularization SBPR which injects adversarialdata during training and Power Regularization via Intrinsic Motivation PRIMwhich adds an intrinsic motivation to regulate power to the training objective.Our experiments demonstrate that both algorithms successfully balance taskreward and power leading to lower power behavior than the baseline oftask-only reward and avoid catastrophic events in case an agent in the systemgoes off-policy. |


