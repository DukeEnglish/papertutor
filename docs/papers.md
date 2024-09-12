# cs.CL 

| Item |Content|
| --- |---|
|idx| 2409.07440v1 |
|title| SUPER: Evaluating Agents on Setting Up and Executing Tasks from Research Repositories |
|authors| Ben BoginKejuan YangShashank GuptaKyle RichardsonErin BransomPeter ClarkAshish SabharwalTushar Khot
|links| http://arxiv.org/abs/2409.07440v1 |
|updated| 2024-09-11 17:37:48 UTC |
|summary| Given that Large Language Models LLMs have made significant progress inwriting code can they now be used to autonomously reproduce results fromresearch repositories Such a capability would be a boon to the researchcommunity helping researchers validate understand and extend prior work. Toadvance towards this goal we introduce SUPER the first benchmark designed toevaluate the capability of LLMs in setting up and executing tasks from researchrepositories. SUPERaims to capture the realistic challenges faced byresearchers working with Machine Learning ML and Natural Language ProcessingNLP research repositories. Our benchmark comprises three distinct problemsets: 45 end-to-end problems with annotated expert solutions 152 sub problemsderived from the expert set that focus on specific challenges e.g.configuring a trainer and 602 automatically generated problems forlarger-scale development. We introduce various evaluation measures to assessboth task success and progress utilizing gold solutions when available orapproximations otherwise. We show that state-of-the-art approaches struggle tosolve these problems with the best model GPT-4o solving only 16.3 of theend-to-end set and 46.1 of the scenarios. This illustrates the challenge ofthis task and suggests that SUPER can serve as a valuable resource for thecommunity to make and measure progress. |


| Item |Content|
| --- |---|
|idx| 2409.07437v1 |
|title| A Suite for Acoustic Language Model Evaluation |
|authors| Gallil MaimonAmit RothYossi Adi
|links| http://arxiv.org/abs/2409.07437v1 |
|updated| 2024-09-11 17:34:52 UTC |
|summary| Speech language models have recently demonstrated great potential asuniversal speech processing systems. Such models have the ability to model therich acoustic information existing in audio signals beyond spoken contentsuch as emotion background noise etc. Despite this evaluation benchmarkswhich evaluate awareness to a wide range of acoustic aspects are lacking. Tohelp bridge this gap we introduce SALMon a novel evaluation suiteencompassing background noise emotion speaker identity and room impulseresponse. The proposed benchmarks both evaluate the consistency of theinspected element and how much it matches the spoken text. We follow amodelling based approach measuring whether a model gives correct sampleshigher scores than incorrect ones. This approach makes the benchmark fast tocompute even for large models. We evaluated several speech language models onSALMon thus highlighting the strengths and weaknesses of each evaluatedmethod. Code and data are publicly available athttps://pages.cs.huji.ac.il/adiyoss-lab/salmon/ . |


| Item |Content|
| --- |---|
|idx| 2409.07431v1 |
|title| Synthetic continued pretraining |
|authors| Zitong YangNeil BandShuangping LiEmmanuel CandèsTatsunori Hashimoto
|links| http://arxiv.org/abs/2409.07431v1 |
|updated| 2024-09-11 17:21:59 UTC |
|summary| Pretraining on large-scale unstructured internet text has enabled languagemodels to acquire a significant amount of world knowledge. However thisknowledge acquisition is data-inefficient -- to learn a given fact models mustbe trained on hundreds to thousands of diverse representations of it. Thisposes a challenge when adapting a pretrained model to a small corpus ofdomain-specific documents where each fact may appear rarely or only once. Wepropose to bridge this gap with synthetic continued pretraining: using thesmall domain-specific corpus to synthesize a large corpus more amenable tolearning and then performing continued pretraining on the synthesized corpus.We instantiate this proposal with EntiGraph a synthetic data augmentationalgorithm that extracts salient entities from the source documents and thengenerates diverse text by drawing connections between the sampled entities.Synthetic continued pretraining using EntiGraph enables a language model toanswer questions and follow generic instructions related to the sourcedocuments without access to them. If instead the source documents areavailable at inference time we show that the knowledge acquired through ourapproach compounds with retrieval-augmented generation. To better understandthese results we build a simple mathematical model of EntiGraph and show howsynthetic data augmentation can rearrange knowledge to enable moredata-efficient learning. |


| Item |Content|
| --- |---|
|idx| 2409.07429v1 |
|title| Agent Workflow Memory |
|authors| Zora Zhiruo WangJiayuan MaoDaniel FriedGraham Neubig
|links| http://arxiv.org/abs/2409.07429v1 |
|updated| 2024-09-11 17:21:00 UTC |
|summary| Despite the potential of language model-based agents to solve real-worldtasks such as web navigation current methods still struggle with long-horizontasks with complex action trajectories. In contrast humans can flexibly solvecomplex tasks by learning reusable task workflows from past experiences andusing them to guide future actions. To build agents that can similarly benefitfrom this process we introduce Agent Workflow Memory AWM a method forinducing commonly reused routines i.e. workflows and selectively providingworkflows to the agent to guide subsequent generations. AWM flexibly applies toboth offline and online scenarios where agents induce workflows from trainingexamples beforehand or from test queries on the fly. We experiment on two majorweb navigation benchmarks -- Mind2Web and WebArena -- that collectively cover1000 tasks from 200 domains across travel shopping and social media amongothers. AWM substantially improves the baseline results by 24.6 and 51.1relative success rate on Mind2Web and WebArena while reducing the number ofsteps taken to solve WebArena tasks successfully. Furthermore online AWMrobustly generalizes in cross-task website and domain evaluations surpassingbaselines from 8.9 to 14.0 absolute points as train-test task distribution gapswiden. |


| Item |Content|
| --- |---|
|idx| 2409.07424v1 |
|title| Towards Fairer Health Recommendations: finding informative unbiased samples via Word Sense Disambiguation |
|authors| Gavin ButtsPegah EmdadJethro LeeShannon SongChiman SalavatiWillmar Sosa DiazShiri Dori-HacohenFabricio Murai
|links| http://arxiv.org/abs/2409.07424v1 |
|updated| 2024-09-11 17:10:20 UTC |
|summary| There have been growing concerns around high-stake applications that rely onmodels trained with biased data which consequently produce biased predictionsoften harming the most vulnerable. In particular biased medical data couldcause health-related applications and recommender systems to create outputsthat jeopardize patient care and widen disparities in health outcomes. A recentframework titled Fairness via AI posits that instead of attempting to correctmodel biases researchers must focus on their root causes by using AI to debiasdata. Inspired by this framework we tackle bias detection in medical curriculausing NLP models including LLMs and evaluate them on a gold standard datasetcontaining 4105 excerpts annotated by medical experts for bias from a largecorpus. We build on previous work by coauthors which augments the set ofnegative samples with non-annotated text containing social identifier terms.However some of these terms especially those related to race and ethnicitycan carry different meanings e.g. white matter of spinal cord. To addressthis issue we propose the use of Word Sense Disambiguation models to refinedataset quality by removing irrelevant sentences. We then evaluate fine-tunedvariations of BERT models as well as GPT models with zero- and few-shotprompting. We found LLMs considered SOTA on many NLP tasks unsuitable forbias detection while fine-tuned BERT models generally perform well across allevaluated metrics. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2409.07453v1 |
|title| "My Grade is Wrong!": A Contestable AI Framework for Interactive Feedback in Evaluating Student Essays |
|authors| Shengxin HongChang CaiSixuan DuHaiyue FengSiyuan LiuXiuyi Fan
|links| http://arxiv.org/abs/2409.07453v1 |
|updated| 2024-09-11 17:59:01 UTC |
|summary| Interactive feedback where feedback flows in both directions between teacherand student is more effective than traditional one-way feedback. However itis often too time-consuming for widespread use in educational practice. WhileLarge Language Models LLMs have potential for automating feedback theystruggle with reasoning and interaction in an interactive setting. This paperintroduces CAELF a Contestable AI Empowered LLM Framework for automatinginteractive feedback. CAELF allows students to query challenge and clarifytheir feedback by integrating a multi-agent system with computationalargumentation. Essays are first assessed by multiple Teaching-Assistant AgentsTA Agents and then a Teacher Agent aggregates the evaluations through formalreasoning to generate feedback and grades. Students can further engage with thefeedback to refine their understanding. A case study on 500 critical thinkingessays with user studies demonstrates that CAELF significantly improvesinteractive feedback enhancing the reasoning and interaction capabilities ofLLMs. This approach offers a promising solution to overcoming the time andresource barriers that have limited the adoption of interactive feedback ineducational settings. |


| Item |Content|
| --- |---|
|idx| 2409.07448v1 |
|title| Introducing Perturb-ability Score (PS) to Enhance Robustness Against Evasion Adversarial Attacks on ML-NIDS |
|authors| Mohamed elShehabyAshraf Matrawy
|links| http://arxiv.org/abs/2409.07448v1 |
|updated| 2024-09-11 17:52:37 UTC |
|summary| This paper proposes a novel Perturb-ability Score PS that can be used toidentify Network Intrusion Detection Systems NIDS features that can be easilymanipulated by attackers in the problem-space. We demonstrate that using PS toselect only non-perturb-able features for ML-based NIDS maintains detectionperformance while enhancing robustness against adversarial attacks. |


| Item |Content|
| --- |---|
|idx| 2409.07440v1 |
|title| SUPER: Evaluating Agents on Setting Up and Executing Tasks from Research Repositories |
|authors| Ben BoginKejuan YangShashank GuptaKyle RichardsonErin BransomPeter ClarkAshish SabharwalTushar Khot
|links| http://arxiv.org/abs/2409.07440v1 |
|updated| 2024-09-11 17:37:48 UTC |
|summary| Given that Large Language Models LLMs have made significant progress inwriting code can they now be used to autonomously reproduce results fromresearch repositories Such a capability would be a boon to the researchcommunity helping researchers validate understand and extend prior work. Toadvance towards this goal we introduce SUPER the first benchmark designed toevaluate the capability of LLMs in setting up and executing tasks from researchrepositories. SUPERaims to capture the realistic challenges faced byresearchers working with Machine Learning ML and Natural Language ProcessingNLP research repositories. Our benchmark comprises three distinct problemsets: 45 end-to-end problems with annotated expert solutions 152 sub problemsderived from the expert set that focus on specific challenges e.g.configuring a trainer and 602 automatically generated problems forlarger-scale development. We introduce various evaluation measures to assessboth task success and progress utilizing gold solutions when available orapproximations otherwise. We show that state-of-the-art approaches struggle tosolve these problems with the best model GPT-4o solving only 16.3 of theend-to-end set and 46.1 of the scenarios. This illustrates the challenge ofthis task and suggests that SUPER can serve as a valuable resource for thecommunity to make and measure progress. |


| Item |Content|
| --- |---|
|idx| 2409.07431v1 |
|title| Synthetic continued pretraining |
|authors| Zitong YangNeil BandShuangping LiEmmanuel CandèsTatsunori Hashimoto
|links| http://arxiv.org/abs/2409.07431v1 |
|updated| 2024-09-11 17:21:59 UTC |
|summary| Pretraining on large-scale unstructured internet text has enabled languagemodels to acquire a significant amount of world knowledge. However thisknowledge acquisition is data-inefficient -- to learn a given fact models mustbe trained on hundreds to thousands of diverse representations of it. Thisposes a challenge when adapting a pretrained model to a small corpus ofdomain-specific documents where each fact may appear rarely or only once. Wepropose to bridge this gap with synthetic continued pretraining: using thesmall domain-specific corpus to synthesize a large corpus more amenable tolearning and then performing continued pretraining on the synthesized corpus.We instantiate this proposal with EntiGraph a synthetic data augmentationalgorithm that extracts salient entities from the source documents and thengenerates diverse text by drawing connections between the sampled entities.Synthetic continued pretraining using EntiGraph enables a language model toanswer questions and follow generic instructions related to the sourcedocuments without access to them. If instead the source documents areavailable at inference time we show that the knowledge acquired through ourapproach compounds with retrieval-augmented generation. To better understandthese results we build a simple mathematical model of EntiGraph and show howsynthetic data augmentation can rearrange knowledge to enable moredata-efficient learning. |


| Item |Content|
| --- |---|
|idx| 2409.07416v1 |
|title| Hierarchical Reinforcement Learning for Temporal Abstraction of Listwise Recommendation |
|authors| Luo JiGao LiuMingyang YinHongxia YangJingren Zhou
|links| http://arxiv.org/abs/2409.07416v1 |
|updated| 2024-09-11 17:01:06 UTC |
|summary| Modern listwise recommendation systems need to consider both long-term userperceptions and short-term interest shifts. Reinforcement learning can beapplied on recommendation to study such a problem but is also subject to largesearch space sparse user feedback and long interactive latency. Motivated byrecent progress in hierarchical reinforcement learning we propose a novelframework called mccHRL to provide different levels of temporal abstraction onlistwise recommendation. Within the hierarchical framework the high-levelagent studies the evolution of user perception while the low-level agentproduces the item selection policy by modeling the process as a sequentialdecision-making problem. We argue that such framework has a well-defineddecomposition of the outra-session context and the intra-session context whichare encoded by the high-level and low-level agents respectively. To verifythis argument we implement both a simulator-based environment and anindustrial dataset-based experiment. Results observe significant performanceimprovement by our method compared with several well-known baselines. Data andcodes have been made public. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2409.07448v1 |
|title| Introducing Perturb-ability Score (PS) to Enhance Robustness Against Evasion Adversarial Attacks on ML-NIDS |
|authors| Mohamed elShehabyAshraf Matrawy
|links| http://arxiv.org/abs/2409.07448v1 |
|updated| 2024-09-11 17:52:37 UTC |
|summary| This paper proposes a novel Perturb-ability Score PS that can be used toidentify Network Intrusion Detection Systems NIDS features that can be easilymanipulated by attackers in the problem-space. We demonstrate that using PS toselect only non-perturb-able features for ML-based NIDS maintains detectionperformance while enhancing robustness against adversarial attacks. |


| Item |Content|
| --- |---|
|idx| 2409.07446v1 |
|title| Adaptive Adapter Routing for Long-Tailed Class-Incremental Learning |
|authors| Zhi-Hong QiDa-Wei ZhouYiran YaoHan-Jia YeDe-Chuan Zhan
|links| http://arxiv.org/abs/2409.07446v1 |
|updated| 2024-09-11 17:52:00 UTC |
|summary| In our ever-evolving world new data exhibits a long-tailed distributionsuch as e-commerce platform reviews. This necessitates continuous modellearning imbalanced data without forgetting addressing the challenge oflong-tailed class-incremental learning LTCIL. Existing methods often rely onretraining linear classifiers with former data which is impractical inreal-world settings. In this paper we harness the potent representationcapabilities of pre-trained models and introduce AdaPtive Adapter RouTingAPART as an exemplar-free solution for LTCIL. To counteract forgetting wetrain inserted adapters with frozen pre-trained weights for deeper adaptationand maintain a pool of adapters for selection during sequential model updates.Additionally we present an auxiliary adapter pool designed for effectivegeneralization especially on minority classes. Adaptive instance routingacross these pools captures crucial correlations facilitating a comprehensiverepresentation of all classes. Consequently APART tackles the imbalanceproblem as well as catastrophic forgetting in a unified framework. Extensivebenchmark experiments validate the effectiveness of APART. Code is availableat: https://github.com/vita-qzh/APART |


| Item |Content|
| --- |---|
|idx| 2409.07434v1 |
|title| Asymptotics of Stochastic Gradient Descent with Dropout Regularization in Linear Models |
|authors| Jiaqi LiJohannes Schmidt-HieberWei Biao Wu
|links| http://arxiv.org/abs/2409.07434v1 |
|updated| 2024-09-11 17:28:38 UTC |
|summary| This paper proposes an asymptotic theory for online inference of thestochastic gradient descent SGD iterates with dropout regularization inlinear regression. Specifically we establish the geometric-moment contractionGMC for constant step-size SGD dropout iterates to show the existence of aunique stationary distribution of the dropout recursive function. By the GMCproperty we provide quenched central limit theorems CLT for the differencebetween dropout and ell2-regularized iterates regardless ofinitialization. The CLT for the difference between the Ruppert-Polyak averagedSGD ASGD with dropout and ell2-regularized iterates is also presented.Based on these asymptotic normality results we further introduce an onlineestimator for the long-run covariance matrix of ASGD dropout to facilitateinference in a recursive manner with efficiency in computational time andmemory. The numerical experiments demonstrate that for sufficiently largesamples the proposed confidence intervals for ASGD with dropout nearly achievethe nominal coverage probability. |


| Item |Content|
| --- |---|
|idx| 2409.07431v1 |
|title| Synthetic continued pretraining |
|authors| Zitong YangNeil BandShuangping LiEmmanuel CandèsTatsunori Hashimoto
|links| http://arxiv.org/abs/2409.07431v1 |
|updated| 2024-09-11 17:21:59 UTC |
|summary| Pretraining on large-scale unstructured internet text has enabled languagemodels to acquire a significant amount of world knowledge. However thisknowledge acquisition is data-inefficient -- to learn a given fact models mustbe trained on hundreds to thousands of diverse representations of it. Thisposes a challenge when adapting a pretrained model to a small corpus ofdomain-specific documents where each fact may appear rarely or only once. Wepropose to bridge this gap with synthetic continued pretraining: using thesmall domain-specific corpus to synthesize a large corpus more amenable tolearning and then performing continued pretraining on the synthesized corpus.We instantiate this proposal with EntiGraph a synthetic data augmentationalgorithm that extracts salient entities from the source documents and thengenerates diverse text by drawing connections between the sampled entities.Synthetic continued pretraining using EntiGraph enables a language model toanswer questions and follow generic instructions related to the sourcedocuments without access to them. If instead the source documents areavailable at inference time we show that the knowledge acquired through ourapproach compounds with retrieval-augmented generation. To better understandthese results we build a simple mathematical model of EntiGraph and show howsynthetic data augmentation can rearrange knowledge to enable moredata-efficient learning. |


| Item |Content|
| --- |---|
|idx| 2409.07424v1 |
|title| Towards Fairer Health Recommendations: finding informative unbiased samples via Word Sense Disambiguation |
|authors| Gavin ButtsPegah EmdadJethro LeeShannon SongChiman SalavatiWillmar Sosa DiazShiri Dori-HacohenFabricio Murai
|links| http://arxiv.org/abs/2409.07424v1 |
|updated| 2024-09-11 17:10:20 UTC |
|summary| There have been growing concerns around high-stake applications that rely onmodels trained with biased data which consequently produce biased predictionsoften harming the most vulnerable. In particular biased medical data couldcause health-related applications and recommender systems to create outputsthat jeopardize patient care and widen disparities in health outcomes. A recentframework titled Fairness via AI posits that instead of attempting to correctmodel biases researchers must focus on their root causes by using AI to debiasdata. Inspired by this framework we tackle bias detection in medical curriculausing NLP models including LLMs and evaluate them on a gold standard datasetcontaining 4105 excerpts annotated by medical experts for bias from a largecorpus. We build on previous work by coauthors which augments the set ofnegative samples with non-annotated text containing social identifier terms.However some of these terms especially those related to race and ethnicitycan carry different meanings e.g. white matter of spinal cord. To addressthis issue we propose the use of Word Sense Disambiguation models to refinedataset quality by removing irrelevant sentences. We then evaluate fine-tunedvariations of BERT models as well as GPT models with zero- and few-shotprompting. We found LLMs considered SOTA on many NLP tasks unsuitable forbias detection while fine-tuned BERT models generally perform well across allevaluated metrics. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2409.07456v1 |
|title| Self-Evolving Depth-Supervised 3D Gaussian Splatting from Rendered Stereo Pairs |
|authors| Sadra SafadoustFabio TosiFatma GüneyMatteo Poggi
|links| http://arxiv.org/abs/2409.07456v1 |
|updated| 2024-09-11 17:59:58 UTC |
|summary| 3D Gaussian Splatting GS significantly struggles to accurately representthe underlying 3D scene geometry resulting in inaccuracies and floatingartifacts when rendering depth maps. In this paper we address this limitationundertaking a comprehensive analysis of the integration of depth priorsthroughout the optimization process of Gaussian primitives and present a novelstrategy for this purpose. This latter dynamically exploits depth cues from areadily available stereo network processing virtual stereo pairs rendered bythe GS model itself during training and achieving consistent self-improvementof the scene representation. Experimental results on three popular datasetsbreaking ground as the first to assess depth accuracy for these modelsvalidate our findings. |


| Item |Content|
| --- |---|
|idx| 2409.07454v1 |
|title| DreamMesh: Jointly Manipulating and Texturing Triangle Meshes for Text-to-3D Generation |
|authors| Haibo YangYang ChenYingwei PanTing YaoZhineng ChenZuxuan WuYu-Gang JiangTao Mei
|links| http://arxiv.org/abs/2409.07454v1 |
|updated| 2024-09-11 17:59:02 UTC |
|summary| Learning radiance fields NeRF with powerful 2D diffusion models hasgarnered popularity for text-to-3D generation. Nevertheless the implicit 3Drepresentations of NeRF lack explicit modeling of meshes and textures oversurfaces and such surface-undefined way may suffer from the issues e.g.noisy surfaces with ambiguous texture details or cross-view inconsistency. Toalleviate this we present DreamMesh a novel text-to-3D architecture thatpivots on well-defined surfaces triangle meshes to generate high-fidelityexplicit 3D model. Technically DreamMesh capitalizes on a distinctivecoarse-to-fine scheme. In the coarse stage the mesh is first deformed bytext-guided Jacobians and then DreamMesh textures the mesh with an interlaceduse of 2D diffusion models in a tuning free manner from multiple viewpoints. Inthe fine stage DreamMesh jointly manipulates the mesh and refines the texturemap leading to high-quality triangle meshes with high-fidelity texturedmaterials. Extensive experiments demonstrate that DreamMesh significantlyoutperforms state-of-the-art text-to-3D methods in faithfully generating 3Dcontent with richer textual details and enhanced geometry. Our project page isavailable at https://dreammesh.github.io. |


| Item |Content|
| --- |---|
|idx| 2409.07452v1 |
|title| Hi3D: Pursuing High-Resolution Image-to-3D Generation with Video Diffusion Models |
|authors| Haibo YangYang ChenYingwei PanTing YaoZhineng ChenChong-Wah NgoTao Mei
|links| http://arxiv.org/abs/2409.07452v1 |
|updated| 2024-09-11 17:58:57 UTC |
|summary| Despite having tremendous progress in image-to-3D generation existingmethods still struggle to produce multi-view consistent images withhigh-resolution textures in detail especially in the paradigm of 2D diffusionthat lacks 3D awareness. In this work we present High-resolution Image-to-3Dmodel Hi3D a new video diffusion based paradigm that redefines a singleimage to multi-view images as 3D-aware sequential image generation i.e.orbital video generation. This methodology delves into the underlying temporalconsistency knowledge in video diffusion model that generalizes well togeometry consistency across multiple views in 3D generation. Technically Hi3Dfirst empowers the pre-trained video diffusion model with 3D-aware priorcamera pose condition yielding multi-view images with low-resolution texturedetails. A 3D-aware video-to-video refiner is learnt to further scale up themulti-view images with high-resolution texture details. Such high-resolutionmulti-view images are further augmented with novel views through 3D GaussianSplatting which are finally leveraged to obtain high-fidelity meshes via 3Dreconstruction. Extensive experiments on both novel view synthesis and singleview reconstruction demonstrate that our Hi3D manages to produce superiormulti-view consistency images with highly-detailed textures. Source code anddata are available at urlhttps://github.com/yanghb22-fdu/Hi3D-Official. |


| Item |Content|
| --- |---|
|idx| 2409.07451v1 |
|title| FreeEnhance: Tuning-Free Image Enhancement via Content-Consistent Noising-and-Denoising Process |
|authors| Yang LuoYiheng ZhangZhaofan QiuTing YaoZhineng ChenYu-Gang JiangTao Mei
|links| http://arxiv.org/abs/2409.07451v1 |
|updated| 2024-09-11 17:58:50 UTC |
|summary| The emergence of text-to-image generation models has led to the recognitionthat image enhancement performed as post-processing would significantlyimprove the visual quality of the generated images. Exploring diffusion modelsto enhance the generated images nevertheless is not trivial and necessitates todelicately enrich plentiful details while preserving the visual appearance ofkey content in the original image. In this paper we propose a novel frameworknamely FreeEnhance for content-consistent image enhancement using theoff-the-shelf image diffusion models. Technically FreeEnhance is a two-stageprocess that firstly adds random noise to the input image and then capitalizeson a pre-trained image diffusion model i.e. Latent Diffusion Models todenoise and enhance the image details. In the noising stage FreeEnhance isdevised to add lighter noise to the region with higher frequency to preservethe high-frequent patterns e.g. edge corner in the original image. In thedenoising stage we present three target properties as constraints toregularize the predicted noise enhancing images with high acutance and highvisual quality. Extensive experiments conducted on the HPDv2 datasetdemonstrate that our FreeEnhance outperforms the state-of-the-art imageenhancement models in terms of quantitative metrics and human preference. Moreremarkably FreeEnhance also shows higher human preference compared to thecommercial image enhancement solution of Magnific AI. |


| Item |Content|
| --- |---|
|idx| 2409.07450v1 |
|title| VMAS: Video-to-Music Generation via Semantic Alignment in Web Music Videos |
|authors| Yan-Bo LinYu TianLinjie YangGedas BertasiusHeng Wang
|links| http://arxiv.org/abs/2409.07450v1 |
|updated| 2024-09-11 17:56:48 UTC |
|summary| We present a framework for learning to generate background music from videoinputs. Unlike existing works that rely on symbolic musical annotations whichare limited in quantity and diversity our method leverages large-scale webvideos accompanied by background music. This enables our model to learn togenerate realistic and diverse music. To accomplish this goal we develop agenerative video-music Transformer with a novel semantic video-music alignmentscheme. Our model uses a joint autoregressive and contrastive learningobjective which encourages the generation of music aligned with high-levelvideo content. We also introduce a novel video-beat alignment scheme to matchthe generated music beats with the low-level motions in the video. Lastly tocapture fine-grained visual cues in a video needed for realistic backgroundmusic generation we introduce a new temporal video encoder architectureallowing us to efficiently process videos consisting of many densely sampledframes. We train our framework on our newly curated DISCO-MV datasetconsisting of 2.2M video-music samples which is orders of magnitude largerthan any prior datasets used for video music generation. Our method outperformsexisting approaches on the DISCO-MV and MusicCaps datasets according to variousmusic generation evaluation metrics including human evaluation. Results areavailable at https://genjib.github.io/project_page/VMAs/index.html |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2409.07434v1 |
|title| Asymptotics of Stochastic Gradient Descent with Dropout Regularization in Linear Models |
|authors| Jiaqi LiJohannes Schmidt-HieberWei Biao Wu
|links| http://arxiv.org/abs/2409.07434v1 |
|updated| 2024-09-11 17:28:38 UTC |
|summary| This paper proposes an asymptotic theory for online inference of thestochastic gradient descent SGD iterates with dropout regularization inlinear regression. Specifically we establish the geometric-moment contractionGMC for constant step-size SGD dropout iterates to show the existence of aunique stationary distribution of the dropout recursive function. By the GMCproperty we provide quenched central limit theorems CLT for the differencebetween dropout and ell2-regularized iterates regardless ofinitialization. The CLT for the difference between the Ruppert-Polyak averagedSGD ASGD with dropout and ell2-regularized iterates is also presented.Based on these asymptotic normality results we further introduce an onlineestimator for the long-run covariance matrix of ASGD dropout to facilitateinference in a recursive manner with efficiency in computational time andmemory. The numerical experiments demonstrate that for sufficiently largesamples the proposed confidence intervals for ASGD with dropout nearly achievethe nominal coverage probability. |


| Item |Content|
| --- |---|
|idx| 2409.07431v1 |
|title| Synthetic continued pretraining |
|authors| Zitong YangNeil BandShuangping LiEmmanuel CandèsTatsunori Hashimoto
|links| http://arxiv.org/abs/2409.07431v1 |
|updated| 2024-09-11 17:21:59 UTC |
|summary| Pretraining on large-scale unstructured internet text has enabled languagemodels to acquire a significant amount of world knowledge. However thisknowledge acquisition is data-inefficient -- to learn a given fact models mustbe trained on hundreds to thousands of diverse representations of it. Thisposes a challenge when adapting a pretrained model to a small corpus ofdomain-specific documents where each fact may appear rarely or only once. Wepropose to bridge this gap with synthetic continued pretraining: using thesmall domain-specific corpus to synthesize a large corpus more amenable tolearning and then performing continued pretraining on the synthesized corpus.We instantiate this proposal with EntiGraph a synthetic data augmentationalgorithm that extracts salient entities from the source documents and thengenerates diverse text by drawing connections between the sampled entities.Synthetic continued pretraining using EntiGraph enables a language model toanswer questions and follow generic instructions related to the sourcedocuments without access to them. If instead the source documents areavailable at inference time we show that the knowledge acquired through ourapproach compounds with retrieval-augmented generation. To better understandthese results we build a simple mathematical model of EntiGraph and show howsynthetic data augmentation can rearrange knowledge to enable moredata-efficient learning. |


| Item |Content|
| --- |---|
|idx| 2409.07412v1 |
|title| Manifold Learning via Foliations and Knowledge Transfer |
|authors| E. TronE. Fioresi
|links| http://arxiv.org/abs/2409.07412v1 |
|updated| 2024-09-11 16:53:53 UTC |
|summary| Understanding how real data is distributed in high dimensional spaces is thekey to many tasks in machine learning. We want to provide a natural geometricstructure on the space of data employing a deep ReLU neural network trained asa classifier. Through the data information matrix DIM a variation of theFisher information matrix the model will discern a singular foliationstructure on the space of data. We show that the singular points of suchfoliation are contained in a measure zero set and that a local regularfoliation exists almost everywhere. Experiments show that the data iscorrelated with leaves of such foliation. Moreover we show the potential of ourapproach for knowledge transfer by analyzing the spectrum of the DIM to measuredistances between datasets. |


| Item |Content|
| --- |---|
|idx| 2409.07401v1 |
|title| Convergence of continuous-time stochastic gradient descent with applications to linear deep neural networks |
|authors| Gabor LugosiEulalia Nualart
|links| http://arxiv.org/abs/2409.07401v1 |
|updated| 2024-09-11 16:40:24 UTC |
|summary| We study a continuous-time approximation of the stochastic gradient descentprocess for minimizing the expected loss in learning problems. The main resultsestablish general sufficient conditions for the convergence extending theresults of Chatterjee 2022 established for nonstochastic gradient descent.We show how the main result can be applied to the case of overparametrizedlinear neural network training. |


| Item |Content|
| --- |---|
|idx| 2409.07392v1 |
|title| A Scalable Algorithm for Active Learning |
|authors| Youguang ChenZheyu WenGeorge Biros
|links| http://arxiv.org/abs/2409.07392v1 |
|updated| 2024-09-11 16:34:01 UTC |
|summary| FIRAL is a recently proposed deterministic active learning algorithm formulticlass classification using logistic regression. It was shown to outperformthe state-of-the-art in terms of accuracy and robustness and comes withtheoretical performance guarantees. However its scalability suffers whendealing with datasets featuring a large number of points n dimensions dand classes c due to its mathcalOc2d2nc2d storage andmathcalOc3nd2  bd3  bn computational complexity where b is thenumber of points to select in active learning. To address these challenges wepropose an approximate algorithm with storage requirements reduced tomathcalOndc  cd2 and a computational complexity ofmathcalObncd2. Additionally we present a parallel implementation onGPUs. We demonstrate the accuracy and scalability of our approach using MNISTCIFAR-10 Caltech101 and ImageNet. The accuracy tests reveal no deteriorationin accuracy compared to FIRAL. We report strong and weak scaling tests on up to12 GPUs for three million point synthetic dataset. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2409.07453v1 |
|title| "My Grade is Wrong!": A Contestable AI Framework for Interactive Feedback in Evaluating Student Essays |
|authors| Shengxin HongChang CaiSixuan DuHaiyue FengSiyuan LiuXiuyi Fan
|links| http://arxiv.org/abs/2409.07453v1 |
|updated| 2024-09-11 17:59:01 UTC |
|summary| Interactive feedback where feedback flows in both directions between teacherand student is more effective than traditional one-way feedback. However itis often too time-consuming for widespread use in educational practice. WhileLarge Language Models LLMs have potential for automating feedback theystruggle with reasoning and interaction in an interactive setting. This paperintroduces CAELF a Contestable AI Empowered LLM Framework for automatinginteractive feedback. CAELF allows students to query challenge and clarifytheir feedback by integrating a multi-agent system with computationalargumentation. Essays are first assessed by multiple Teaching-Assistant AgentsTA Agents and then a Teacher Agent aggregates the evaluations through formalreasoning to generate feedback and grades. Students can further engage with thefeedback to refine their understanding. A case study on 500 critical thinkingessays with user studies demonstrates that CAELF significantly improvesinteractive feedback enhancing the reasoning and interaction capabilities ofLLMs. This approach offers a promising solution to overcoming the time andresource barriers that have limited the adoption of interactive feedback ineducational settings. |


| Item |Content|
| --- |---|
|idx| 2409.07444v1 |
|title| Echoes of Privacy: Uncovering the Profiling Practices of Voice Assistants |
|authors| Tina KhezresmaeilzadehElaine ZhuKiersten GriecoDaniel J. DuboisKonstantinos PsounisDavid Choffnes
|links| http://arxiv.org/abs/2409.07444v1 |
|updated| 2024-09-11 17:44:41 UTC |
|summary| Many companies including Google Amazon and Apple offer voice assistantsas a convenient solution for answering general voice queries and accessingtheir services. These voice assistants have gained popularity and can be easilyaccessed through various smart devices such as smartphones smart speakerssmartwatches and an increasing array of other devices. However thisconvenience comes with potential privacy risks. For instance while companiesvaguely mention in their privacy policies that they may use voice interactionsfor user profiling it remains unclear to what extent this profiling occurs andwhether voice interactions pose greater privacy risks compared to otherinteraction modalities.  In this paper we conduct 1171 experiments involving a total of 24530 querieswith different personas and interaction modalities over the course of 20 monthsto characterize how the three most popular voice assistants profile theirusers. We analyze factors such as the labels assigned to users their accuracythe time taken to assign these labels differences between voice and webinteractions and the effectiveness of profiling remediation tools offered byeach voice assistant. Our findings reveal that profiling can happen withoutinteraction can be incorrect and inconsistent at times may take several daysto weeks for changes to occur and can be influenced by the interactionmodality. |


| Item |Content|
| --- |---|
|idx| 2409.07406v1 |
|title| Trust Dynamics in Human-Autonomy Interaction: Uncover Associations between Trust Dynamics and Personal Characteristics |
|authors| Hyesun ChungX. Jessie Yang
|links| http://arxiv.org/abs/2409.07406v1 |
|updated| 2024-09-11 16:49:31 UTC |
|summary| While personal characteristics influence peoples snapshot trust towardsautonomous systems their relationships with trust dynamics remain poorlyunderstood. We conducted a human-subject experiment with 130 participantsperforming a simulated surveillance task aided by an automated threat detector.A comprehensive pre-experimental survey collected data on participantspersonal characteristics across 12 constructs and 28 dimensions. Based on datacollected in the experiment we clustered participants trust dynamics intothree types and assessed differences among the three clusters in terms ofpersonal characteristics behaviors performance and post-experiment ratings.Participants were clustered into three groups namely Bayesian decision makersdisbelievers and oscillators. Results showed that the clusters differsignificantly in seven personal characteristics: masculinity positive affectextraversion neuroticism intellect performance expectancy and highexpectations. The disbelievers tend to have high neuroticism and lowperformance expectancy. The oscillators tend to have higher scores inmasculinity positive affect extraversion and intellect. We also foundsignificant differences in the behaviors and post-experiment ratings among thethree groups. The disbelievers are the least likely to blindly follow therecommendations made by the automated threat detector. Based on the significantpersonal characteristics we developed a decision tree model to predict clustertypes with an accuracy of 70. |


| Item |Content|
| --- |---|
|idx| 2409.07372v1 |
|title| Awaking the Slides: A Tuning-free and Knowledge-regulated AI Tutoring System via Language Model Coordination |
|authors| Daniel Zhang-LiZheyuan ZhangJifan YuJoy Lim Jia YinShangqing TuLinlu GongHaohua WangZhiyuan LiuHuiqin LiuLei HouJuanzi Li
|links| http://arxiv.org/abs/2409.07372v1 |
|updated| 2024-09-11 16:03:09 UTC |
|summary| The vast pre-existing slides serve as rich and important materials to carrylecture knowledge. However effectively leveraging lecture slides to servestudents is difficult due to the multi-modal nature of slide content and theheterogeneous teaching actions. We study the problem of discovering effectivedesigns that convert a slide into an interactive lecture. We developSlide2Lecture a tuning-free and knowledge-regulated intelligent tutoringsystem that can 1 effectively convert an input lecture slide into astructured teaching agenda consisting of a set of heterogeneous teachingactions 2 create and manage an interactive lecture that generates responsiveinteractions catering to student learning demands while regulating theinteractions to follow teaching actions. Slide2Lecture contains a completepipeline for learners to obtain an interactive classroom experience to learnthe slide. For teachers and developers Slide2Lecture enables customization tocater to personalized demands. The evaluation rated by annotators and studentsshows that Slide2Lecture is effective in outperforming the remainingimplementation. Slide2Lectures online deployment has made more than 200Kinteraction with students in the 3K lecture sessions. We open sourceSlide2Lectures implementation inhttps://anonymous.4open.science/r/slide2lecture-4210/. |


| Item |Content|
| --- |---|
|idx| 2409.07306v1 |
|title| Visual Compositional Data Analytics for Spatial Transcriptomics |
|authors| David HägeleYuxuan TangDaniel Weiskopf
|links| http://arxiv.org/abs/2409.07306v1 |
|updated| 2024-09-11 14:36:03 UTC |
|summary| For the BioMed-Vis Challenge 2024 we propose a visual analytics system as aredesign for the scatter pie chart visualization of cell type proportions ofspatial transcriptomics data. Our design uses three linked views: a view of thehistological image of the tissue a stacked bar chart showing cell typeproportions of the spots and a scatter plot showing a dimensionality reductionof the multivariate proportions. Furthermore we apply a compositional dataanalysis framework the Aitchison geometry to the proportions fordimensionality reduction and k-means clustering. Leveraging brushing andlinking the system allows one to explore and uncover patterns in the cell typemixtures and relate them to their spatial locations on the cellular tissue.This redesign shifts the pattern recognition workload from the human visualsystem to computational methods commonly used in visual analytics. We providethe code and setup instructions of our visual analytics system on GitHubhttps://github.com/UniStuttgart-VISUS/va-for-spatial-transcriptomics. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2409.07136v1 |
|title| Leveraging Unstructured Text Data for Federated Instruction Tuning of Large Language Models |
|authors| Rui YeRui GeYuchi FengtingJingyi ChaiYanfeng WangSiheng Chen
|links| http://arxiv.org/abs/2409.07136v1 |
|updated| 2024-09-11 09:31:44 UTC |
|summary| Federated instruction tuning enables multiple clients to collaborativelyfine-tune a shared large language model LLM that can follow humansinstructions without directly sharing raw data. However existing literatureimpractically requires that all the clients readily hold instruction-tuningdata i.e. structured instruction-response pairs which necessitates massivehuman annotations since clients data is usually unstructured text instead.Addressing this we propose a novel and flexible framework FedIT-U2S which canautomatically transform unstructured corpus into structured data for federatedinstruction tuning. FedIT-U2S consists two key steps: 1 few-shotinstruction-tuning data generation where each unstructured data piece togetherwith several examples is combined to prompt an LLM in generating aninstruction-response pair. To further enhance the flexibility aretrieval-based example selection technique is proposed where the examples areautomatically selected based on the relatedness between the clients data pieceand example pool bypassing the need of determining examples in advance. 2 Atypical federated instruction tuning process based on the generated data.Overall FedIT-U2S can be applied to diverse scenarios as long as the clientholds valuable text corpus broadening the application scope of federatedinstruction tuning. We conduct a series of experiments on three domainsmedicine knowledge and math showing that our proposed FedIT-U2S canconsistently and significantly brings improvement over the base LLM. |


| Item |Content|
| --- |---|
|idx| 2409.07127v1 |
|title| DCMAC: Demand-aware Customized Multi-Agent Communication via Upper Bound Training |
|authors| Dongkun HuoHuateng ZhangYixue HaoYuanlin YeLong HuRui WangMin Chen
|links| http://arxiv.org/abs/2409.07127v1 |
|updated| 2024-09-11 09:23:27 UTC |
|summary| Efficient communication can enhance the overall performance of collaborativemulti-agent reinforcement learning. A common approach is to share observationsthrough full communication leading to significant communication overhead.Existing work attempts to perceive the global state by conducting teammatemodel based on local information. However they ignore that the uncertaintygenerated by prediction may lead to difficult training. To address thisproblem we propose a Demand-aware Customized Multi-Agent Communication DCMACprotocol which use an upper bound training to obtain the ideal policy. Byutilizing the demand parsing module agent can interpret the gain of sendinglocal message on teammate and generate customized messages via compute thecorrelation between demands and local observation using cross-attentionmechanism. Moreover our method can adapt to the communication resources ofagents and accelerate the training progress by appropriating the ideal policywhich is trained with joint observation. Experimental results reveal that DCMACsignificantly outperforms the baseline algorithms in both unconstrained andcommunication constrained scenarios. |


| Item |Content|
| --- |---|
|idx| 2409.06888v1 |
|title| A Quality Diversity Approach to Automatically Generate Multi-Agent Path Finding Benchmark Maps |
|authors| Cheng QianYulun ZhangVarun BhattMatthew Christopher FontaineStefanos NikolaidisJiaoyang Li
|links| http://arxiv.org/abs/2409.06888v1 |
|updated| 2024-09-10 22:08:33 UTC |
|summary| We use the Quality Diversity QD algorithm with Neural Cellular AutomataNCA to generate benchmark maps for Multi-Agent Path Finding MAPFalgorithms. Previously MAPF algorithms are tested using fixed human-designedbenchmark maps. However such fixed benchmark maps have several problems.First these maps may not cover all the potential failure scenarios for thealgorithms. Second when comparing different algorithms fixed benchmark mapsmay introduce bias leading to unfair comparisons between algorithms. In thiswork we take advantage of the QD algorithm and NCA with different objectivesand diversity measures to generate maps with patterns to comprehensivelyunderstand the performance of MAPF algorithms and be able to make faircomparisons between two MAPF algorithms to provide further information on theselection between two algorithms. Empirically we employ this technique togenerate diverse benchmark maps to evaluate and compare the behavior ofdifferent types of MAPF algorithms such as bounded-suboptimal algorithmssuboptimal algorithms and reinforcement-learning-based algorithms. Throughboth single-planner experiments and comparisons between algorithms we identifypatterns where each algorithm excels and detect disparities in runtime orsuccess rates between different algorithms. |


| Item |Content|
| --- |---|
|idx| 2409.06750v1 |
|title| Can Agents Spontaneously Form a Society? Introducing a Novel Architecture for Generative Multi-Agents to Elicit Social Emergence |
|authors| H. ZhangJ. YinM. JiangC. Su
|links| http://arxiv.org/abs/2409.06750v1 |
|updated| 2024-09-10 13:39:29 UTC |
|summary| Generative agents have demonstrated impressive capabilities in specifictasks but most of these frameworks focus on independent tasks and lackattention to social interactions. We introduce a generative agent architecturecalled ITCMA-S which includes a basic framework for individual agents and aframework called LTRHA that supports social interactions among multi-agents.This architecture enables agents to identify and filter out behaviors that aredetrimental to social interactions guiding them to choose more favorableactions. We designed a sandbox environment to simulate the natural evolution ofsocial relationships among multiple identity-less agents for experimentalevaluation. The results showed that ITCMA-S performed well on multipleevaluation indicators demonstrating its ability to actively explore theenvironment recognize new agents and acquire new information throughcontinuous actions and dialogue. Observations show that as agents establishconnections with each other they spontaneously form cliques with internalhierarchies around a selected leader and organize collective activities. |


| Item |Content|
| --- |---|
|idx| 2409.06345v1 |
|title| Foragax: An Agent Based Modelling framework based on JAX |
|authors| Siddharth ChaturvediAhmed El-GazzarMarcel van Gerven
|links| http://arxiv.org/abs/2409.06345v1 |
|updated| 2024-09-10 08:57:42 UTC |
|summary| Foraging for resources is a ubiquitous activity conducted by living organismsin a shared environment to maintain their homeostasis. Modelling multi-agentforaging in-silico allows us to study both individual and collective emergentbehaviour in a tractable manner. Agent-based modelling has proven to beeffective in simulating such tasks though scaling the simulations toaccommodate large numbers of agents with complex dynamics remains challenging.In this work we present Foragax a general-purpose scalablehardware-accelerated multi-agent foraging toolkit. Leveraging the JAX libraryour toolkit can simulate thousands of agents foraging in a common environmentin an end-to-end vectorized and differentiable manner. The toolkit providesagent-based modelling tools to model various foraging tasks including optionsto design custom spatial and temporal agent dynamics control policies sensormodels and boundary conditions. Further the number of agents during suchsimulations can be increased or decreased based on custom rules. The toolkitcan also be used to potentially model more general multi-agent scenarios. |


