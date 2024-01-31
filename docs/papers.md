# cs.CL 

| Item |Content|
| --- |---|
|idx| 2401.16421v1 |
|title| Two Stones Hit One Bird: Bilevel Positional Encoding for Better Length Extrapolation |
|authors| Zhenyu HeGuhao FengShengjie LuoKai YangDi HeJingjing XuZhi ZhangHongxia YangLiwei Wang
|links| http://arxiv.org/abs/2401.16421v1 |
|updated| 2024-01-29 18:59:07 UTC |
|summary| In this work we leverage the intrinsic segmentation of language sequencesand design a new positional encoding method called Bilevel Positional EncodingBiPE. For each position our BiPE blends an intra-segment encoding and aninter-segment encoding. The intra-segment encoding identifies the locationswithin a segment and helps the model capture the semantic information thereinvia absolute positional encoding. The inter-segment encoding specifies thesegment index models the relationships between segments and aims to improveextrapolation capabilities via relative positional encoding. Theoreticalanalysis shows this disentanglement of positional information makes learningmore effective. The empirical results also show that our BiPE has superiorlength extrapolation capabilities across a wide range of tasks in diverse textmodalities. |


| Item |Content|
| --- |---|
|idx| 2401.16420v1 |
|title| InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model |
|authors| Xiaoyi DongPan ZhangYuhang ZangYuhang CaoBin WangLinke OuyangXilin WeiSongyang ZhangHaodong DuanMaosong CaoWenwei ZhangYining LiHang YanYang GaoXinyue ZhangWei LiJingwen LiKai ChenConghui HeXingcheng ZhangYu QiaoDahua LinJiaqi Wang
|links| http://arxiv.org/abs/2401.16420v1 |
|updated| 2024-01-29 18:59:02 UTC |
|summary| We introduce InternLM-XComposer2 a cutting-edge vision-language modelexcelling in free-form text-image composition and comprehension. This modelgoes beyond conventional vision-language understanding adeptly craftinginterleaved text-image content from diverse inputs like outlines detailedtextual specifications and reference images enabling highly customizablecontent creation. InternLM-XComposer2 proposes a Partial LoRA PLoRA approachthat applies additional LoRA parameters exclusively to image tokens to preservethe integrity of pre-trained language knowledge striking a balance betweenprecise vision understanding and text composition with literary talent.Experimental results demonstrate the superiority of InternLM-XComposer2 basedon InternLM2-7B in producing high-quality long-text multi-modal content and itsexceptional vision-language understanding performance across variousbenchmarks where it not only significantly outperforms existing multimodalmodels but also matches or even surpasses GPT-4V and Gemini Pro in certainassessments. This highlights its remarkable proficiency in the realm ofmultimodal understanding. The InternLM-XComposer2 model series with 7Bparameters are publicly available athttps://github.com/InternLM/InternLM-XComposer. |


| Item |Content|
| --- |---|
|idx| 2401.16405v1 |
|title| Scaling Sparse Fine-Tuning to Large Language Models |
|authors| Alan AnsellIvan VulićHannah SterzAnna KorhonenEdoardo M. Ponti
|links| http://arxiv.org/abs/2401.16405v1 |
|updated| 2024-01-29 18:43:49 UTC |
|summary| Large Language Models LLMs are difficult to fully fine-tune e.g. withinstructions or human feedback due to their sheer number of parameters. Afamily of parameter-efficient sparse fine-tuning SFT methods have provenpromising in terms of performance but their memory requirements increaseproportionally to the size of the LLMs. In this work we scale sparsefine-tuning to state-of-the-art LLMs like LLaMA 2 7B and 13B. At any giventime for a desired density level we maintain an array of parameter indicesand the deltas of these parameters relative to their pretrained values. Weiterate among: a updating the active deltas b pruning indices based onthe change of magnitude of their deltas and c regrowth of indices. Forregrowth we explore two criteria based on either the accumulated gradients ofa few candidate parameters or their approximate momenta estimated using theefficient SM3 optimizer. We experiment with instruction-tuning of LLMs onstandard dataset mixtures finding that SFT is often superior to popularparameter-efficient fine-tuning methods like LoRA low-rank adaptation interms of performance and comparable in terms of run time. We additionally showthat SFT is compatible with both quantization and efficient optimizers tofacilitate scaling to ever-larger model sizes. We release the code for SFT athttps://github.com/AlanAnsell/peft and for the instruction-tuning experimentsat https://github.com/ducdauge/sft-llm. |


| Item |Content|
| --- |---|
|idx| 2401.16403v1 |
|title| ViLexNorm: A Lexical Normalization Corpus for Vietnamese Social Media Text |
|authors| Thanh-Nhi NguyenThanh-Phong LeKiet Van Nguyen
|links| http://arxiv.org/abs/2401.16403v1 |
|updated| 2024-01-29 18:41:39 UTC |
|summary| Lexical normalization a fundamental task in Natural Language ProcessingNLP involves the transformation of words into their canonical forms. Thisprocess has been proven to benefit various downstream NLP tasks greatly. Inthis work we introduce Vietnamese Lexical Normalization ViLexNorm thefirst-ever corpus developed for the Vietnamese lexical normalization task. Thecorpus comprises over 10000 pairs of sentences meticulously annotated by humanannotators sourced from public comments on Vietnams most popular social mediaplatforms. Various methods were used to evaluate our corpus and thebest-performing system achieved a result of 57.74 using the Error ReductionRate ERR metric van der Goot 2019a with the Leave-As-Is LAI baseline.For extrinsic evaluation employing the model trained on ViLexNorm demonstratesthe positive impact of the Vietnamese lexical normalization task on other NLPtasks. Our corpus is publicly available exclusively for research purposes. |


| Item |Content|
| --- |---|
|idx| 2401.16380v1 |
|title| Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling |
|authors| Pratyush MainiSkyler SetoHe BaiDavid GrangierYizhe ZhangNavdeep Jaitly
|links| http://arxiv.org/abs/2401.16380v1 |
|updated| 2024-01-29 18:19:08 UTC |
|summary| Large language models are trained on massive scrapes of the web which areoften unstructured noisy and poorly phrased. Current scaling laws show thatlearning from such data requires an abundance of both compute and data whichgrows with the size of the model being trained. This is infeasible both becauseof the large compute costs and duration associated with pre-training and theimpending scarcity of high-quality data on the web. In this work we proposeWeb Rephrase Augmented Pre-training textbfWRAP that uses anoff-the-shelf instruction-tuned model prompted to paraphrase documents on theweb in specific styles such as like Wikipedia or in question-answer formatto jointly pre-train LLMs on real and synthetic rephrases. First we show thatusing WRAP on the C4 dataset which is naturally noisy speeds up pre-trainingby sim3x. At the same pre-training compute budget it improves perplexity bymore than 10 on average across different subsets of the Pile and improveszero-shot question answer accuracy across 13 tasks by more than 2. Second weinvestigate the impact of the re-phrasing style on the performance of themodel offering insights into how the composition of the training data canimpact the performance of LLMs in OOD settings. Our gains are attributed to thefact that re-phrased synthetic data has higher utility than just real databecause it i incorporates style diversity that closely reflects downstreamevaluation style and ii has higher quality than web-scraped data. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2401.16421v1 |
|title| Two Stones Hit One Bird: Bilevel Positional Encoding for Better Length Extrapolation |
|authors| Zhenyu HeGuhao FengShengjie LuoKai YangDi HeJingjing XuZhi ZhangHongxia YangLiwei Wang
|links| http://arxiv.org/abs/2401.16421v1 |
|updated| 2024-01-29 18:59:07 UTC |
|summary| In this work we leverage the intrinsic segmentation of language sequencesand design a new positional encoding method called Bilevel Positional EncodingBiPE. For each position our BiPE blends an intra-segment encoding and aninter-segment encoding. The intra-segment encoding identifies the locationswithin a segment and helps the model capture the semantic information thereinvia absolute positional encoding. The inter-segment encoding specifies thesegment index models the relationships between segments and aims to improveextrapolation capabilities via relative positional encoding. Theoreticalanalysis shows this disentanglement of positional information makes learningmore effective. The empirical results also show that our BiPE has superiorlength extrapolation capabilities across a wide range of tasks in diverse textmodalities. |


| Item |Content|
| --- |---|
|idx| 2401.16412v1 |
|title| Learning to Manipulate under Limited Information |
|authors| Wesley H. HollidayAlexander KristoffersenEric Pacuit
|links| http://arxiv.org/abs/2401.16412v1 |
|updated| 2024-01-29 18:49:50 UTC |
|summary| By classic results in social choice theory any reasonable preferentialvoting method sometimes gives individuals an incentive to report an insincerepreference. The extent to which different voting methods are more or lessresistant to such strategic manipulation has become a key consideration forcomparing voting methods. Here we measure resistance to manipulation by whetherneural networks of varying sizes can learn to profitably manipulate a givenvoting method in expectation given different types of limited informationabout how other voters will vote. We trained nearly 40000 neural networks of26 sizes to manipulate against 8 different voting methods under 6 types oflimited information in committee-sized elections with 5-21 voters and 3-6candidates. We find that some voting methods such as Borda are highlymanipulable by networks with limited information while others such as InstantRunoff are not despite being quite profitably manipulated by an idealmanipulator with full information. |


| Item |Content|
| --- |---|
|idx| 2401.16405v1 |
|title| Scaling Sparse Fine-Tuning to Large Language Models |
|authors| Alan AnsellIvan VulićHannah SterzAnna KorhonenEdoardo M. Ponti
|links| http://arxiv.org/abs/2401.16405v1 |
|updated| 2024-01-29 18:43:49 UTC |
|summary| Large Language Models LLMs are difficult to fully fine-tune e.g. withinstructions or human feedback due to their sheer number of parameters. Afamily of parameter-efficient sparse fine-tuning SFT methods have provenpromising in terms of performance but their memory requirements increaseproportionally to the size of the LLMs. In this work we scale sparsefine-tuning to state-of-the-art LLMs like LLaMA 2 7B and 13B. At any giventime for a desired density level we maintain an array of parameter indicesand the deltas of these parameters relative to their pretrained values. Weiterate among: a updating the active deltas b pruning indices based onthe change of magnitude of their deltas and c regrowth of indices. Forregrowth we explore two criteria based on either the accumulated gradients ofa few candidate parameters or their approximate momenta estimated using theefficient SM3 optimizer. We experiment with instruction-tuning of LLMs onstandard dataset mixtures finding that SFT is often superior to popularparameter-efficient fine-tuning methods like LoRA low-rank adaptation interms of performance and comparable in terms of run time. We additionally showthat SFT is compatible with both quantization and efficient optimizers tofacilitate scaling to ever-larger model sizes. We release the code for SFT athttps://github.com/AlanAnsell/peft and for the instruction-tuning experimentsat https://github.com/ducdauge/sft-llm. |


| Item |Content|
| --- |---|
|idx| 2401.16402v1 |
|title| A Survey on Visual Anomaly Detection: Challenge, Approach, and Prospect |
|authors| Yunkang CaoXiaohao XuJiangning ZhangYuqi ChengXiaonan HuangGuansong PangWeiming Shen
|links| http://arxiv.org/abs/2401.16402v1 |
|updated| 2024-01-29 18:41:21 UTC |
|summary| Visual Anomaly Detection VAD endeavors to pinpoint deviations from theconcept of normality in visual data widely applied across diverse domainse.g. industrial defect inspection and medical lesion detection. This surveycomprehensively examines recent advancements in VAD by identifying threeprimary challenges: 1 scarcity of training data 2 diversity of visualmodalities and 3 complexity of hierarchical anomalies. Starting with a briefoverview of the VAD background and its generic concept definitions weprogressively categorize emphasize and discuss the latest VAD progress fromthe perspective of sample number data modality and anomaly hierarchy. Throughan in-depth analysis of the VAD field we finally summarize future developmentsfor VAD and conclude the key findings and contributions of this survey. |


| Item |Content|
| --- |---|
|idx| 2401.16398v1 |
|title| Zero-shot Imitation Policy via Search in Demonstration Dataset |
|authors| Federco MalatoFlorian LeopoldAndrew MelnikVille Hautamaki
|links| http://arxiv.org/abs/2401.16398v1 |
|updated| 2024-01-29 18:38:29 UTC |
|summary| Behavioral cloning uses a dataset of demonstrations to learn a policy. Toovercome computationally expensive training procedures and address the policyadaptation problem we propose to use latent spaces of pre-trained foundationmodels to index a demonstration dataset instantly access similar relevantexperiences and copy behavior from these situations. Actions from a selectedsimilar situation can be performed by the agent until representations of theagents current situation and the selected experience diverge in the latentspace. Thus we formulate our control problem as a dynamic search problem overa dataset of experts demonstrations. We test our approach on BASALTMineRL-dataset in the latent representation of a Video Pre-Training model. Wecompare our model to state-of-the-art Imitation Learning-based Minecraftagents. Our approach can effectively recover meaningful demonstrations and showhuman-like behavior of an agent in the Minecraft environment in a wide varietyof scenarios. Experimental results reveal that performance of our search-basedapproach clearly wins in terms of accuracy and perceptual evaluation overlearning-based models. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2401.16423v1 |
|title| Synchformer: Efficient Synchronization from Sparse Cues |
|authors| Vladimir IashinWeidi XieEsa RahtuAndrew Zisserman
|links| http://arxiv.org/abs/2401.16423v1 |
|updated| 2024-01-29 18:59:55 UTC |
|summary| Our objective is audio-visual synchronization with a focus on in-the-wildvideos such as those on YouTube where synchronization cues can be sparse. Ourcontributions include a novel audio-visual synchronization model and trainingthat decouples feature extraction from synchronization modelling throughmulti-modal segment-level contrastive pre-training. This approach achievesstate-of-the-art performance in both dense and sparse settings. We also extendsynchronization model training to AudioSet a million-scale in-the-wilddataset investigate evidence attribution techniques for interpretability andexplore a new capability for synchronization models: audio-visualsynchronizability. |


| Item |Content|
| --- |---|
|idx| 2401.16422v1 |
|title| Strategic Usage in a Multi-Learner Setting |
|authors| Eliot ShekhtmanSarah Dean
|links| http://arxiv.org/abs/2401.16422v1 |
|updated| 2024-01-29 18:59:22 UTC |
|summary| Real-world systems often involve some pool of users choosing between a set ofservices. With the increase in popularity of online learning algorithms theseservices can now self-optimize leveraging data collected on users to maximizesome reward such as service quality. On the flipside users may strategicallychoose which services to use in order to pursue their own reward functions inthe process wielding power over which services can see and use their data.Extensive prior research has been conducted on the effects of strategic usersin single-service settings with strategic behavior manifesting in themanipulation of observable features to achieve a desired classificationhowever this can often be costly or unattainable for users and fails tocapture the full behavior of multi-service dynamic systems. As such we analyzea setting in which strategic users choose among several available services inorder to pursue positive classifications while services seek to minimize lossfunctions on their observations. We focus our analysis on realizable settingsand show that naive retraining can still lead to oscillation even if all usersare observed at different times however if this retraining uses memory ofpast observations convergent behavior can be guaranteed for certain lossfunction classes. We provide results obtained from synthetic and real-worlddata to empirically validate our theoretical findings. |


| Item |Content|
| --- |---|
|idx| 2401.16421v1 |
|title| Two Stones Hit One Bird: Bilevel Positional Encoding for Better Length Extrapolation |
|authors| Zhenyu HeGuhao FengShengjie LuoKai YangDi HeJingjing XuZhi ZhangHongxia YangLiwei Wang
|links| http://arxiv.org/abs/2401.16421v1 |
|updated| 2024-01-29 18:59:07 UTC |
|summary| In this work we leverage the intrinsic segmentation of language sequencesand design a new positional encoding method called Bilevel Positional EncodingBiPE. For each position our BiPE blends an intra-segment encoding and aninter-segment encoding. The intra-segment encoding identifies the locationswithin a segment and helps the model capture the semantic information thereinvia absolute positional encoding. The inter-segment encoding specifies thesegment index models the relationships between segments and aims to improveextrapolation capabilities via relative positional encoding. Theoreticalanalysis shows this disentanglement of positional information makes learningmore effective. The empirical results also show that our BiPE has superiorlength extrapolation capabilities across a wide range of tasks in diverse textmodalities. |


| Item |Content|
| --- |---|
|idx| 2401.16419v1 |
|title| Semi-parametric Expert Bayesian Network Learning with Gaussian Processes and Horseshoe Priors |
|authors| Yidou WengFinale Doshi-Velez
|links| http://arxiv.org/abs/2401.16419v1 |
|updated| 2024-01-29 18:57:45 UTC |
|summary| This paper proposes a model learning Semi-parametric rela- tionships in anExpert Bayesian Network SEBN with linear parameter and structure constraints.We use Gaussian Pro- cesses and a Horseshoe prior to introduce minimal nonlin-ear components. To prioritize modifying the expert graph over adding new edgeswe optimize differential Horseshoe scales. In real-world datasets with unknowntruth we gen- erate diverse graphs to accommodate user input addressingidentifiability issues and enhancing interpretability. Evalua- tion onsynthetic and UCI Liver Disorders datasets using metrics like structuralHamming Distance and test likelihood demonstrates our models outperformstate-of-the-art semi- parametric Bayesian Network model. |


| Item |Content|
| --- |---|
|idx| 2401.16418v1 |
|title| Boolean Logic as an Error feedback mechanism |
|authors| Louis Leconte
|links| http://arxiv.org/abs/2401.16418v1 |
|updated| 2024-01-29 18:56:21 UTC |
|summary| The notion of Boolean logic backpropagation was introduced to build neuralnetworks with weights and activations being Boolean numbers. Most ofcomputations can be done with Boolean logic instead of real arithmetic bothduring training and inference phases. But the underlying discrete optimizationproblem is NP-hard and the Boolean logic has no guarantee. In this work wepropose the first convergence analysis under standard non-convex assumptions. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2401.16424v1 |
|title| Computer Vision for Primate Behavior Analysis in the Wild |
|authors| Richard VoggTimo LüddeckeJonathan HenrichSharmita DeyMatthias NuskeValentin HasslerDerek MurphyJulia FischerJulia OstnerOliver SchülkePeter M. KappelerClaudia FichtelAlexander GailStefan TreueHansjörg ScherbergerFlorentin WörgötterAlexander S. Ecker
|links| http://arxiv.org/abs/2401.16424v1 |
|updated| 2024-01-29 18:59:56 UTC |
|summary| Advances in computer vision as well as increasingly widespread video-basedbehavioral monitoring have great potential for transforming how we study animalcognition and behavior. However there is still a fairly large gap between theexciting prospects and what can actually be achieved in practice todayespecially in videos from the wild. With this perspective paper we want tocontribute towards closing this gap by guiding behavioral scientists in whatcan be expected from current methods and steering computer vision researcherstowards problems that are relevant to advance research in animal behavior. Westart with a survey of the state-of-the-art methods for computer visionproblems that are directly relevant to the video-based study of animalbehavior including object detection multi-individual tracking interactionrecognition and individual identification. We then review methods foreffort-efficient learning which is one of the biggest challenges from apractical perspective. Finally we close with an outlook into the future of theemerging field of computer vision for animal behavior where we argue that thefield should move fast beyond the common frame-by-frame processing and treatvideo as a first-class citizen. |


| Item |Content|
| --- |---|
|idx| 2401.16423v1 |
|title| Synchformer: Efficient Synchronization from Sparse Cues |
|authors| Vladimir IashinWeidi XieEsa RahtuAndrew Zisserman
|links| http://arxiv.org/abs/2401.16423v1 |
|updated| 2024-01-29 18:59:55 UTC |
|summary| Our objective is audio-visual synchronization with a focus on in-the-wildvideos such as those on YouTube where synchronization cues can be sparse. Ourcontributions include a novel audio-visual synchronization model and trainingthat decouples feature extraction from synchronization modelling throughmulti-modal segment-level contrastive pre-training. This approach achievesstate-of-the-art performance in both dense and sparse settings. We also extendsynchronization model training to AudioSet a million-scale in-the-wilddataset investigate evidence attribution techniques for interpretability andexplore a new capability for synchronization models: audio-visualsynchronizability. |


| Item |Content|
| --- |---|
|idx| 2401.16420v1 |
|title| InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model |
|authors| Xiaoyi DongPan ZhangYuhang ZangYuhang CaoBin WangLinke OuyangXilin WeiSongyang ZhangHaodong DuanMaosong CaoWenwei ZhangYining LiHang YanYang GaoXinyue ZhangWei LiJingwen LiKai ChenConghui HeXingcheng ZhangYu QiaoDahua LinJiaqi Wang
|links| http://arxiv.org/abs/2401.16420v1 |
|updated| 2024-01-29 18:59:02 UTC |
|summary| We introduce InternLM-XComposer2 a cutting-edge vision-language modelexcelling in free-form text-image composition and comprehension. This modelgoes beyond conventional vision-language understanding adeptly craftinginterleaved text-image content from diverse inputs like outlines detailedtextual specifications and reference images enabling highly customizablecontent creation. InternLM-XComposer2 proposes a Partial LoRA PLoRA approachthat applies additional LoRA parameters exclusively to image tokens to preservethe integrity of pre-trained language knowledge striking a balance betweenprecise vision understanding and text composition with literary talent.Experimental results demonstrate the superiority of InternLM-XComposer2 basedon InternLM2-7B in producing high-quality long-text multi-modal content and itsexceptional vision-language understanding performance across variousbenchmarks where it not only significantly outperforms existing multimodalmodels but also matches or even surpasses GPT-4V and Gemini Pro in certainassessments. This highlights its remarkable proficiency in the realm ofmultimodal understanding. The InternLM-XComposer2 model series with 7Bparameters are publicly available athttps://github.com/InternLM/InternLM-XComposer. |


| Item |Content|
| --- |---|
|idx| 2401.16416v1 |
|title| Endo-4DGS: Distilling Depth Ranking for Endoscopic Monocular Scene Reconstruction with 4D Gaussian Splatting |
|authors| Yiming HuangBeilei CuiLong BaiZiqi GuoMengya XuHongliang Ren
|links| http://arxiv.org/abs/2401.16416v1 |
|updated| 2024-01-29 18:55:29 UTC |
|summary| In the realm of robot-assisted minimally invasive surgery dynamic scenereconstruction can significantly enhance downstream tasks and improve surgicaloutcomes. Neural Radiance Fields NeRF-based methods have recently risen toprominence for their exceptional ability to reconstruct scenes. Nonethelessthese methods are hampered by slow inference prolonged training andsubstantial computational demands. Additionally some rely on stereo depthestimation which is often infeasible due to the high costs and logisticalchallenges associated with stereo cameras. Moreover the monocularreconstruction quality for deformable scenes is currently inadequate. Toovercome these obstacles we present Endo-4DGS an innovative real-timeendoscopic dynamic reconstruction approach that utilizes 4D Gaussian SplattingGS and requires no ground truth depth data. This method extends 3D GS byincorporating a temporal component and leverages a lightweight MLP to capturetemporal Gaussian deformations. This effectively facilitates the reconstructionof dynamic surgical scenes with variable conditions. We also integrateDepth-Anything to generate pseudo-depth maps from monocular views enhancingthe depth-guided reconstruction process. Our approach has been validated on twosurgical datasets where it has proven to render in real-time computeefficiently and reconstruct with remarkable accuracy. These results underlinethe vast potential of Endo-4DGS to improve surgical assistance. |


| Item |Content|
| --- |---|
|idx| 2401.16402v1 |
|title| A Survey on Visual Anomaly Detection: Challenge, Approach, and Prospect |
|authors| Yunkang CaoXiaohao XuJiangning ZhangYuqi ChengXiaonan HuangGuansong PangWeiming Shen
|links| http://arxiv.org/abs/2401.16402v1 |
|updated| 2024-01-29 18:41:21 UTC |
|summary| Visual Anomaly Detection VAD endeavors to pinpoint deviations from theconcept of normality in visual data widely applied across diverse domainse.g. industrial defect inspection and medical lesion detection. This surveycomprehensively examines recent advancements in VAD by identifying threeprimary challenges: 1 scarcity of training data 2 diversity of visualmodalities and 3 complexity of hierarchical anomalies. Starting with a briefoverview of the VAD background and its generic concept definitions weprogressively categorize emphasize and discuss the latest VAD progress fromthe perspective of sample number data modality and anomaly hierarchy. Throughan in-depth analysis of the VAD field we finally summarize future developmentsfor VAD and conclude the key findings and contributions of this survey. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2401.16421v1 |
|title| Two Stones Hit One Bird: Bilevel Positional Encoding for Better Length Extrapolation |
|authors| Zhenyu HeGuhao FengShengjie LuoKai YangDi HeJingjing XuZhi ZhangHongxia YangLiwei Wang
|links| http://arxiv.org/abs/2401.16421v1 |
|updated| 2024-01-29 18:59:07 UTC |
|summary| In this work we leverage the intrinsic segmentation of language sequencesand design a new positional encoding method called Bilevel Positional EncodingBiPE. For each position our BiPE blends an intra-segment encoding and aninter-segment encoding. The intra-segment encoding identifies the locationswithin a segment and helps the model capture the semantic information thereinvia absolute positional encoding. The inter-segment encoding specifies thesegment index models the relationships between segments and aims to improveextrapolation capabilities via relative positional encoding. Theoreticalanalysis shows this disentanglement of positional information makes learningmore effective. The empirical results also show that our BiPE has superiorlength extrapolation capabilities across a wide range of tasks in diverse textmodalities. |


| Item |Content|
| --- |---|
|idx| 2401.16419v1 |
|title| Semi-parametric Expert Bayesian Network Learning with Gaussian Processes and Horseshoe Priors |
|authors| Yidou WengFinale Doshi-Velez
|links| http://arxiv.org/abs/2401.16419v1 |
|updated| 2024-01-29 18:57:45 UTC |
|summary| This paper proposes a model learning Semi-parametric rela- tionships in anExpert Bayesian Network SEBN with linear parameter and structure constraints.We use Gaussian Pro- cesses and a Horseshoe prior to introduce minimal nonlin-ear components. To prioritize modifying the expert graph over adding new edgeswe optimize differential Horseshoe scales. In real-world datasets with unknowntruth we gen- erate diverse graphs to accommodate user input addressingidentifiability issues and enhancing interpretability. Evalua- tion onsynthetic and UCI Liver Disorders datasets using metrics like structuralHamming Distance and test likelihood demonstrates our models outperformstate-of-the-art semi- parametric Bayesian Network model. |


| Item |Content|
| --- |---|
|idx| 2401.16418v1 |
|title| Boolean Logic as an Error feedback mechanism |
|authors| Louis Leconte
|links| http://arxiv.org/abs/2401.16418v1 |
|updated| 2024-01-29 18:56:21 UTC |
|summary| The notion of Boolean logic backpropagation was introduced to build neuralnetworks with weights and activations being Boolean numbers. Most ofcomputations can be done with Boolean logic instead of real arithmetic bothduring training and inference phases. But the underlying discrete optimizationproblem is NP-hard and the Boolean logic has no guarantee. In this work wepropose the first convergence analysis under standard non-convex assumptions. |


| Item |Content|
| --- |---|
|idx| 2401.16410v1 |
|title| ReTaSA: A Nonparametric Functional Estimation Approach for Addressing Continuous Target Shift |
|authors| Hwanwoo KimXin ZhangJiwei ZhaoQinglong Tian
|links| http://arxiv.org/abs/2401.16410v1 |
|updated| 2024-01-29 18:47:36 UTC |
|summary| The presence of distribution shifts poses a significant challenge fordeploying modern machine learning models in real-world applications. This workfocuses on the target shift problem in a regression setting Zhang et al.2013 Nguyen et al. 2016. More specifically the target variable y alsoknown as the response variable which is continuous has different marginaldistributions in the training source and testing domain while the conditionaldistribution of features x given y remains the same. While most literaturefocuses on classification tasks with finite target space the regressionproblem has an infinite dimensional target space which makes many of theexisting methods inapplicable. In this work we show that the continuous targetshift problem can be addressed by estimating the importance weight functionfrom an ill-posed integral equation. We propose a nonparametric regularizedapproach named ReTaSA to solve the ill-posed integral equation and providetheoretical justification for the estimated importance weight function. Theeffectiveness of the proposed method has been demonstrated with extensivenumerical studies on synthetic and real-world datasets. |


| Item |Content|
| --- |---|
|idx| 2401.16407v1 |
|title| Is K-fold cross validation the best model selection method for Machine Learning? |
|authors| Juan M GorrizF SegoviaJ RamirezA OrtizJ. Suckling
|links| http://arxiv.org/abs/2401.16407v1 |
|updated| 2024-01-29 18:46:53 UTC |
|summary| As a technique that can compactly represent complex patterns machinelearning has significant potential for predictive inference. K-foldcross-validation CV is the most common approach to ascertaining thelikelihood that a machine learning outcome is generated by chance andfrequently outperforms conventional hypothesis testing. This improvement usesmeasures directly obtained from machine learning classifications such asaccuracy that do not have a parametric description. To approach a frequentistanalysis within machine learning pipelines a permutation test or simplestatistics from data partitions i.e. folds can be added to estimateconfidence intervals. Unfortunately neither parametric nor non-parametrictests solve the inherent problems around partitioning small sample-sizedatasets and learning from heterogeneous data sources. The fact that machinelearning strongly depends on the learning parameters and the distribution ofdata across folds recapitulates familiar difficulties around excess falsepositives and replication. The origins of this problem are demonstrated bysimulating common experimental circumstances including small sample sizes lownumbers of predictors and heterogeneous data sources. A novel statistical testbased on K-fold CV and the Upper Bound of the actual error K-fold CUBV iscomposed where uncertain predictions of machine learning with CV are boundedby the emphworst case through the evaluation of concentration inequalities.Probably Approximately Correct-Bayesian upper bounds for linear classifiers incombination with K-fold CV is used to estimate the empirical error. Theperformance with neuroimaging datasets suggests this is a robust criterion fordetecting effects validating accuracy values obtained from machine learningwhilst avoiding excess false positives. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2401.16348v1 |
|title| Beyond Automated Evaluation Metrics: Evaluating Topic Models On Practical Social Science Content Analysis Tasks |
|authors| Zongxia LiAndrew MaoDaniel StephensPranav GoelEmily WalpoleAlden DimaJuan FungJordan Boyd-Graber
|links| http://arxiv.org/abs/2401.16348v1 |
|updated| 2024-01-29 17:54:04 UTC |
|summary| Topic models are a popular tool for understanding text collections but theirevaluation has been a point of contention. Automated evaluation metrics such ascoherence are often used however their validity has been questioned forneural topic models NTMs and can overlook the benefits of a model in realworld applications. To this end we conduct the first evaluation of neuralsupervised and classical topic models in an interactive task based setting. Wecombine topic models with a classifier and test their ability to help humansconduct content analysis and document annotation. From simulated real user andexpert pilot studies the Contextual Neural Topic Model does the best oncluster evaluation metrics and human evaluations however LDA is competitivewith two other NTMs under our simulated experiment and user study resultscontrary to what coherence scores suggest. We show that current automatedmetrics do not provide a complete picture of topic modeling capabilities butthe right choice of NTMs can be better than classical models on practicaltasks. |


| Item |Content|
| --- |---|
|idx| 2401.16307v1 |
|title| Momentary Stressor Logging and Reflective Visualizations: Implications for Stress Management with Wearables |
|authors| Sameer NeupaneMithun SahaNasir AliTimothy HnatShahin Alan SamieiAnandatirtha NandugudiDavid M. AlmeidaSantosh Kumar
|links| http://dx.doi.org/10.1145/3613904.3642662 |
|updated| 2024-01-29 17:08:57 UTC |
|summary| Commercial wearables from Fitbit Garmin and Whoop have recently introducedreal-time notifications based on detecting changes in physiological responsesindicating potential stress. In this paper we investigate how these newcapabilities can be leveraged to improve stress management. We developed asmartwatch app a smartphone app and a cloud service and conducted a 100-dayfield study with 122 participants who received prompts triggered byphysiological responses several times a day. They were asked whether they werestressed and if so to log the most likely stressor. Each week participantsreceived new visualizations of their data to self-reflect on patterns andtrends. Participants reported better awareness of their stressors andself-initiating fourteen kinds of behavioral changes to reduce stress in theirdaily lives. Repeated self-reports over 14 weeks showed reductions in bothstress intensity in 26521 momentary ratings and stress frequency in 1057weekly surveys. |


| Item |Content|
| --- |---|
|idx| 2401.16167v1 |
|title| "You tell me": A Dataset of GPT-4-Based Behaviour Change Support Conversations |
|authors| Selina MeyerDavid Elsweiler
|links| http://arxiv.org/abs/2401.16167v1 |
|updated| 2024-01-29 13:54:48 UTC |
|summary| Conversational agents are increasingly used to address emotional needs on topof information needs. One use case of increasing interest are counselling-stylemental health and behaviour change interventions with large language modelLLM-based approaches becoming more popular. Research in this context so farhas been largely system-focused foregoing the aspect of user behaviour and theimpact this can have on LLM-generated texts. To address this issue we share adataset containing text-based user interactions related to behaviour changewith two GPT-4-based conversational agents collected in a preregistered userstudy. This dataset includes conversation data user language analysisperception measures and user feedback for LLM-generated turns and can offervaluable insights to inform the design of such systems based on realinteractions. |


| Item |Content|
| --- |---|
|idx| 2401.16123v1 |
|title| Looking for a better fit? An Incremental Learning Multimodal Object Referencing Framework adapting to Individual Drivers |
|authors| Amr GomaaGuillermo ReyesMichael FeldAntonio Krüger
|links| http://arxiv.org/abs/2401.16123v1 |
|updated| 2024-01-29 12:48:56 UTC |
|summary| The rapid advancement of the automotive industry towards automated andsemi-automated vehicles has rendered traditional methods of vehicleinteraction such as touch-based and voice command systems inadequate for awidening range of non-driving related tasks such as referencing objectsoutside of the vehicle. Consequently research has shifted toward gesturalinput e.g. hand gaze and head pose gestures as a more suitable mode ofinteraction during driving. However due to the dynamic nature of driving andindividual variation there are significant differences in drivers gesturalinput performance. While in theory this inherent variability could bemoderated by substantial data-driven machine learning models prevalentmethodologies lean towards constrained single-instance trained models forobject referencing. These models show a limited capacity to continuously adaptto the divergent behaviors of individual drivers and the variety of drivingscenarios. To address this we propose textitIcRegress a novelregression-based incremental learning approach that adapts to changing behaviorand the unique characteristics of drivers engaged in the dual task of drivingand referencing objects. We suggest a more personalized and adaptable solutionfor multimodal gestural interfaces employing continuous lifelong learning toenhance driver experience safety and convenience. Our approach was evaluatedusing an outside-the-vehicle object referencing use case highlighting thesuperiority of the incremental learning models adapted over a single trainedmodel across various driver traits such as handedness driving experience andnumerous driving conditions. Finally to facilitate reproducibility easedeployment and promote further research we offer our approach as anopen-source framework at urlhttps://github.com/amrgomaaelhady/IcRegress. |


| Item |Content|
| --- |---|
|idx| 2401.15996v1 |
|title| AccessLens: Auto-detecting Inaccessibility of Everyday Objects |
|authors| Nahyun KwonQian LuMuhammad Hasham QaziJoanne LiuChanghoon OhShu KongJeeeun Kim
|links| http://dx.doi.org/10.1145/3613904.3642767 |
|updated| 2024-01-29 09:27:55 UTC |
|summary| In our increasingly diverse society everyday physical interfaces oftenpresent barriers impacting individuals across various contexts. Thisoversight from small cabinet knobs to identical wall switches that can posedifferent contextual challenges highlights an imperative need for solutions.Leveraging low-cost 3D-printed augmentations such as knob magnifiers andtactile labels seems promising yet the process of discovering unrecognizedbarriers remains challenging because disability is context-dependent. Weintroduce AccessLens an end-to-end system designed to identify inaccessibleinterfaces in daily objects and recommend 3D-printable augmentations foraccessibility enhancement. Our approach involves training a detector using thenovel AccessDB dataset designed to automatically recognize 21 distinctInaccessibility Classes e.g. bar-small and round-rotate within 6 commonobject categories e.g. handle and knob. AccessMeta serves as a robust way tobuild a comprehensive dictionary linking these accessibility classes toopen-source 3D augmentation designs. Experiments demonstrate our detectorsperformance in detecting inaccessible objects. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2401.16412v1 |
|title| Learning to Manipulate under Limited Information |
|authors| Wesley H. HollidayAlexander KristoffersenEric Pacuit
|links| http://arxiv.org/abs/2401.16412v1 |
|updated| 2024-01-29 18:49:50 UTC |
|summary| By classic results in social choice theory any reasonable preferentialvoting method sometimes gives individuals an incentive to report an insincerepreference. The extent to which different voting methods are more or lessresistant to such strategic manipulation has become a key consideration forcomparing voting methods. Here we measure resistance to manipulation by whetherneural networks of varying sizes can learn to profitably manipulate a givenvoting method in expectation given different types of limited informationabout how other voters will vote. We trained nearly 40000 neural networks of26 sizes to manipulate against 8 different voting methods under 6 types oflimited information in committee-sized elections with 5-21 voters and 3-6candidates. We find that some voting methods such as Borda are highlymanipulable by networks with limited information while others such as InstantRunoff are not despite being quite profitably manipulated by an idealmanipulator with full information. |


| Item |Content|
| --- |---|
|idx| 2401.16236v1 |
|title| Effective Communication with Dynamic Feature Compression |
|authors| Pietro TalliFrancesco PaseFederico ChiariottiAndrea ZanellaMichele Zorzi
|links| http://arxiv.org/abs/2401.16236v1 |
|updated| 2024-01-29 15:35:05 UTC |
|summary| The remote wireless control of industrial systems is one of the major usecases for 5G and beyond systems: in these cases the massive amounts of sensoryinformation that need to be shared over the wireless medium may overload evenhigh-capacity connections. Consequently solving the effective communicationproblem by optimizing the transmission strategy to discard irrelevantinformation can provide a significant advantage but is often a very complextask. In this work we consider a prototypal system in which an observer mustcommunicate its sensory data to a robot controlling a task e.g. a mobilerobot in a factory. We then model it as a remote Partially Observable MarkovDecision Process POMDP considering the effect of adopting semantic andeffective communication-oriented solutions on the overall system performance.We split the communication problem by considering an ensemble Vector QuantizedVariational Autoencoder VQ-VAE encoding and train a Deep ReinforcementLearning DRL agent to dynamically adapt the quantization level consideringboth the current state of the environment and the memory of past messages. Wetested the proposed approach on the well-known CartPole reference controlproblem obtaining a significant performance increase over traditionalapproaches. |


| Item |Content|
| --- |---|
|idx| 2401.16216v1 |
|title| A mechanism for discovering semantic relationships among agent communication protocols |
|authors| Idoia BergesJesús BermúdezAlfredo GoñiArantza Illarramendi
|links| http://dx.doi.org/10.1007/s10458-010-9154-1 |
|updated| 2024-01-29 15:10:09 UTC |
|summary| One relevant aspect in the development of the Semantic Web framework is theachievement of a real inter-agents communication capability at the semanticlevel. Agents should be able to communicate with each other freely usingdifferent communication protocols constituted by communication acts. For thatscenario we introduce in this paper an efficient mechanism presenting thefollowing main features: - It promotes the description of the communicationacts of protocols as classes that belong to a communication acts ontology andassociates to those acts a social commitment semantics formalized throughpredicates in the Event Calculus. - It is sustained on the idea that differentprotocols can be compared semantically by looking to the set of fluentsassociated to each branch of the protocols. Those sets are generated usingSemantic Web technology rules. - It discovers the following types of protocolrelationships: equivalence specialization restriction prefix suffix infixand complement_to_infix. |


| Item |Content|
| --- |---|
|idx| 2401.15838v1 |
|title| Distributed Markov Chain Monte Carlo Sampling based on the Alternating Direction Method of Multipliers |
|authors| Alexandros E. TzikasLicio RomaoMert PilanciAlessandro AbateMykel J. Kochenderfer
|links| http://arxiv.org/abs/2401.15838v1 |
|updated| 2024-01-29 02:08:40 UTC |
|summary| Many machine learning applications require operating on a spatiallydistributed dataset. Despite technological advances privacy considerations andcommunication constraints may prevent gathering the entire dataset in a centralunit. In this paper we propose a distributed sampling scheme based on thealternating direction method of multipliers which is commonly used in theoptimization literature due to its fast convergence. In contrast to distributedoptimization distributed sampling allows for uncertainty quantification inBayesian inference tasks. We provide both theoretical guarantees of ouralgorithms convergence and experimental evidence of its superiority to thestate-of-the-art. For our theoretical results we use convex optimization toolsto establish a fundamental inequality on the generated local sample iterates.This inequality enables us to show convergence of the distribution associatedwith these iterates to the underlying target distribution in Wassersteindistance. In simulation we deploy our algorithm on linear and logisticregression tasks and illustrate its fast convergence compared to existinggradient-based methods. |


| Item |Content|
| --- |---|
|idx| 2401.15598v1 |
|title| Accelerated Distributed Allocation |
|authors| Mohammadreza DoostmohammadianAlireza Aghasi
|links| http://arxiv.org/abs/2401.15598v1 |
|updated| 2024-01-28 07:54:12 UTC |
|summary| Distributed allocation finds applications in many scenarios including CPUscheduling distributed energy resource management and networked coveragecontrol. In this paper we propose a fast convergent optimization algorithmwith a tunable rate using the signum function. The convergence rate of theproposed algorithm can be managed by changing two parameters. We proveconvergence over uniformly-connected multi-agent networks. Therefore thesolution converges even if the network loses connectivity at some finite timeintervals. The proposed algorithm is all-time feasible implying that at anytermination time of the algorithm the resource-demand feasibility holds. Thisis in contrast to asymptotic feasibility in many dual formulation solutionse.g. ADMM that meet resource-demand feasibility over time andasymptotically. |


