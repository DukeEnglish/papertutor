# cs.CL 

| Item |Content|
| --- |---|
|idx| 2403.18814v1 |
|title| Mini-Gemini: Mining the Potential of Multi-modality Vision Language Models |
|authors| Yanwei LiYuechen ZhangChengyao WangZhisheng ZhongYixin ChenRuihang ChuShaoteng LiuJiaya Jia
|links| http://arxiv.org/abs/2403.18814v1 |
|updated| 2024-03-27 17:59:04 UTC |
|summary| In this work we introduce Mini-Gemini a simple and effective frameworkenhancing multi-modality Vision Language Models VLMs. Despite theadvancements in VLMs facilitating basic visual dialog and reasoning aperformance gap persists compared to advanced models like GPT-4 and Gemini. Wetry to narrow the gap by mining the potential of VLMs for better performanceand any-to-any workflow from three aspects i.e. high-resolution visualtokens high-quality data and VLM-guided generation. To enhance visual tokenswe propose to utilize an additional visual encoder for high-resolutionrefinement without increasing the visual token count. We further construct ahigh-quality dataset that promotes precise image comprehension andreasoning-based generation expanding the operational scope of current VLMs. Ingeneral Mini-Gemini further mines the potential of VLMs and empowers currentframeworks with image understanding reasoning and generation simultaneously.Mini-Gemini supports a series of dense and MoE Large Language Models LLMsfrom 2B to 34B. It is demonstrated to achieve leading performance in severalzero-shot benchmarks and even surpasses the developed private models. Code andmodels are available at https://github.com/dvlab-research/MiniGemini. |


| Item |Content|
| --- |---|
|idx| 2403.18804v1 |
|title| Is Modularity Transferable? A Case Study through the Lens of Knowledge Distillation |
|authors| Mateusz KlimaszewskiPiotr AndruszkiewiczAlexandra Birch
|links| http://arxiv.org/abs/2403.18804v1 |
|updated| 2024-03-27 17:50:00 UTC |
|summary| The rise of Modular Deep Learning showcases its potential in various NaturalLanguage Processing applications. Parameter-efficient fine-tuning PEFTmodularity has been shown to work for various use cases from domain adaptationto multilingual setups. However all this work covers the case where themodular components are trained and deployed within one single Pre-trainedLanguage Model PLM. This model-specific setup is a substantial limitation onthe very modularity that modular architectures are trying to achieve. We askwhether current modular approaches are transferable between models and whetherwe can transfer the modules from more robust and larger PLMs to smaller ones.In this work we aim to fill this gap via a lens of Knowledge Distillationcommonly used for model compression and present an extremely straightforwardapproach to transferring pre-trained task-specific PEFT modules betweensame-family PLMs. Moreover we propose a method that allows the transfer ofmodules between incompatible PLMs without any change in the inferencecomplexity. The experiments on Named Entity Recognition Natural LanguageInference and Paraphrase Identification tasks over multiple languages and PEFTmethods showcase the initial potential of transferable modularity. |


| Item |Content|
| --- |---|
|idx| 2403.18803v1 |
|title| Projective Methods for Mitigating Gender Bias in Pre-trained Language Models |
|authors| Hillary DawkinsIsar NejadgholiDaniel GillisJudi McCuaig
|links| http://arxiv.org/abs/2403.18803v1 |
|updated| 2024-03-27 17:49:31 UTC |
|summary| Mitigation of gender bias in NLP has a long history tied to debiasing staticword embeddings. More recently attention has shifted to debiasing pre-trainedlanguage models. We study to what extent the simplest projective debiasingmethods developed for word embeddings can help when applied to BERTsinternal representations. Projective methods are fast to implement use a smallnumber of saved parameters and make no updates to the existing modelparameters. We evaluate the efficacy of the methods in reducing both intrinsicbias as measured by BERTs next sentence prediction task and in mitigatingobserved bias in a downstream setting when fine-tuned. To this end we alsoprovide a critical analysis of a popular gender-bias assessment test forquantifying intrinsic bias resulting in an enhanced test set and new biasmeasures. We find that projective methods can be effective at both intrinsicbias and downstream bias mitigation but that the two outcomes are notnecessarily correlated. This finding serves as a warning that intrinsic biastest sets based either on language modeling tasks or next sentence predictionshould not be the only benchmark in developing a debiased language model. |


| Item |Content|
| --- |---|
|idx| 2403.18802v1 |
|title| Long-form factuality in large language models |
|authors| Jerry WeiChengrun YangXinying SongYifeng LuNathan HuDustin TranDaiyi PengRuibo LiuDa HuangCosmo DuQuoc V. Le
|links| http://arxiv.org/abs/2403.18802v1 |
|updated| 2024-03-27 17:48:55 UTC |
|summary| Large language models LLMs often generate content that contains factualerrors when responding to fact-seeking prompts on open-ended topics. Tobenchmark a models long-form factuality in open domains we first use GPT-4 togenerate LongFact a prompt set comprising thousands of questions spanning 38topics. We then propose that LLM agents can be used as automated evaluators forlong-form factuality through a method which we call Search-Augmented FactualityEvaluator SAFE. SAFE utilizes an LLM to break down a long-form response intoa set of individual facts and to evaluate the accuracy of each fact using amulti-step reasoning process comprising sending search queries to Google Searchand determining whether a fact is supported by the search results. Furthermorewe propose extending F1 score as an aggregated metric for long-form factuality.To do so we balance the percentage of supported facts in a responseprecision with the percentage of provided facts relative to a hyperparameterrepresenting a users preferred response length recall.  Empirically we demonstrate that LLM agents can achieve superhuman ratingperformance - on a set of 16k individual facts SAFE agrees with crowdsourcedhuman annotators 72 of the time and on a random subset of 100 disagreementcases SAFE wins 76 of the time. At the same time SAFE is more than 20 timescheaper than human annotators. We also benchmark thirteen language models onLongFact across four model families Gemini GPT Claude and PaLM-2 findingthat larger language models generally achieve better long-form factuality.LongFact SAFE and all experimental code are available athttps://github.com/google-deepmind/long-form-factuality. |


| Item |Content|
| --- |---|
|idx| 2403.18783v1 |
|title| Towards a World-English Language Model for On-Device Virtual Assistants |
|authors| Rricha JalotaLyan VerwimpMarkus Nussbaum-ThomAmr MousaArturo ArguetaYoussef Oualil
|links| http://dx.doi.org/10.1109/ICASSP48485.2024.10448018 |
|updated| 2024-03-27 17:31:39 UTC |
|summary| Neural Network Language Models NNLMs for Virtual Assistants VAs aregenerally language- region- and in some cases device-dependent whichincreases the effort to scale and maintain them. Combining NNLMs for one ormore of the categories is one way to improve scalability. In this work wecombine regional variants of English to build a World English NNLM foron-device VAs. In particular we investigate the application of adapterbottlenecks to model dialect-specific characteristics in our existingproduction NNLMs and enhance the multi-dialect baselines. We find thatadapter modules are more effective in modeling dialects than specializingentire sub-networks. Based on this insight and leveraging the design of ourproduction models we introduce a new architecture for World English NNLM thatmeets the accuracy latency and memory constraints of our single-dialectmodels. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2403.18814v1 |
|title| Mini-Gemini: Mining the Potential of Multi-modality Vision Language Models |
|authors| Yanwei LiYuechen ZhangChengyao WangZhisheng ZhongYixin ChenRuihang ChuShaoteng LiuJiaya Jia
|links| http://arxiv.org/abs/2403.18814v1 |
|updated| 2024-03-27 17:59:04 UTC |
|summary| In this work we introduce Mini-Gemini a simple and effective frameworkenhancing multi-modality Vision Language Models VLMs. Despite theadvancements in VLMs facilitating basic visual dialog and reasoning aperformance gap persists compared to advanced models like GPT-4 and Gemini. Wetry to narrow the gap by mining the potential of VLMs for better performanceand any-to-any workflow from three aspects i.e. high-resolution visualtokens high-quality data and VLM-guided generation. To enhance visual tokenswe propose to utilize an additional visual encoder for high-resolutionrefinement without increasing the visual token count. We further construct ahigh-quality dataset that promotes precise image comprehension andreasoning-based generation expanding the operational scope of current VLMs. Ingeneral Mini-Gemini further mines the potential of VLMs and empowers currentframeworks with image understanding reasoning and generation simultaneously.Mini-Gemini supports a series of dense and MoE Large Language Models LLMsfrom 2B to 34B. It is demonstrated to achieve leading performance in severalzero-shot benchmarks and even surpasses the developed private models. Code andmodels are available at https://github.com/dvlab-research/MiniGemini. |


| Item |Content|
| --- |---|
|idx| 2403.18807v1 |
|title| ECoDepth: Effective Conditioning of Diffusion Models for Monocular Depth Estimation |
|authors| Suraj PatniAradhye AgarwalChetan Arora
|links| http://arxiv.org/abs/2403.18807v1 |
|updated| 2024-03-27 17:53:30 UTC |
|summary| In the absence of parallax cues a learning-based single image depthestimation SIDE model relies heavily on shading and contextual cues in theimage. While this simplicity is attractive it is necessary to train suchmodels on large and varied datasets which are difficult to capture. It hasbeen shown that using embeddings from pre-trained foundational models such asCLIP improves zero shot transfer in several applications. Taking inspirationfrom this in our paper we explore the use of global image priors generatedfrom a pre-trained ViT model to provide more detailed contextual information.We argue that the embedding vector from a ViT model pre-trained on a largedataset captures greater relevant information for SIDE than the usual route ofgenerating pseudo image captions followed by CLIP based text embeddings. Basedon this idea we propose a new SIDE model using a diffusion backbone which isconditioned on ViT embeddings. Our proposed design establishes a newstate-of-the-art SOTA for SIDE on NYUv2 dataset achieving Abs Rel error of0.05914 improvement compared to 0.069 by the current SOTA VPD. And onKITTI dataset achieving Sq Rel error of 0.139 2 improvement compared to0.142 by the current SOTA GEDepth. For zero-shot transfer with a modeltrained on NYUv2 we report mean relative improvement of 20 23 81 25over NeWCRFs on Sun-RGBD iBims1 DIODE HyperSim datasets compared to 1618 45 9 by ZoeDepth. The code is available athttps://github.com/Aradhye2002/EcoDepth. |


| Item |Content|
| --- |---|
|idx| 2403.18802v1 |
|title| Long-form factuality in large language models |
|authors| Jerry WeiChengrun YangXinying SongYifeng LuNathan HuDustin TranDaiyi PengRuibo LiuDa HuangCosmo DuQuoc V. Le
|links| http://arxiv.org/abs/2403.18802v1 |
|updated| 2024-03-27 17:48:55 UTC |
|summary| Large language models LLMs often generate content that contains factualerrors when responding to fact-seeking prompts on open-ended topics. Tobenchmark a models long-form factuality in open domains we first use GPT-4 togenerate LongFact a prompt set comprising thousands of questions spanning 38topics. We then propose that LLM agents can be used as automated evaluators forlong-form factuality through a method which we call Search-Augmented FactualityEvaluator SAFE. SAFE utilizes an LLM to break down a long-form response intoa set of individual facts and to evaluate the accuracy of each fact using amulti-step reasoning process comprising sending search queries to Google Searchand determining whether a fact is supported by the search results. Furthermorewe propose extending F1 score as an aggregated metric for long-form factuality.To do so we balance the percentage of supported facts in a responseprecision with the percentage of provided facts relative to a hyperparameterrepresenting a users preferred response length recall.  Empirically we demonstrate that LLM agents can achieve superhuman ratingperformance - on a set of 16k individual facts SAFE agrees with crowdsourcedhuman annotators 72 of the time and on a random subset of 100 disagreementcases SAFE wins 76 of the time. At the same time SAFE is more than 20 timescheaper than human annotators. We also benchmark thirteen language models onLongFact across four model families Gemini GPT Claude and PaLM-2 findingthat larger language models generally achieve better long-form factuality.LongFact SAFE and all experimental code are available athttps://github.com/google-deepmind/long-form-factuality. |


| Item |Content|
| --- |---|
|idx| 2403.18795v1 |
|title| Gamba: Marry Gaussian Splatting with Mamba for single view 3D reconstruction |
|authors| Qiuhong ShenXuanyu YiZike WuPan ZhouHanwang ZhangShuicheng YanXinchao Wang
|links| http://arxiv.org/abs/2403.18795v1 |
|updated| 2024-03-27 17:40:14 UTC |
|summary| We tackle the challenge of efficiently reconstructing a 3D asset from asingle image with growing demands for automated 3D content creation pipelines.Previous methods primarily rely on Score Distillation Sampling SDS and NeuralRadiance Fields NeRF. Despite their significant success these approachesencounter practical limitations due to lengthy optimization and considerablememory usage. In this report we introduce Gamba an end-to-end amortized 3Dreconstruction model from single-view images emphasizing two main insights:1 3D representation: leveraging a large number of 3D Gaussians for anefficient 3D Gaussian splatting process 2 Backbone design: introducing aMamba-based sequential network that facilitates context-dependent reasoning andlinear scalability with the sequence token length accommodating asubstantial number of Gaussians. Gamba incorporates significant advancements indata preprocessing regularization design and training methodologies. Weassessed Gamba against existing optimization-based and feed-forward 3Dgeneration approaches using the real-world scanned OmniObject3D dataset. HereGamba demonstrates competitive generation capabilities both qualitatively andquantitatively while achieving remarkable speed approximately 0.6 second on asingle NVIDIA A100 GPU. |


| Item |Content|
| --- |---|
|idx| 2403.18775v1 |
|title| ImageNet-D: Benchmarking Neural Network Robustness on Diffusion Synthetic Object |
|authors| Chenshuang ZhangFei PanJunmo KimIn So KweonChengzhi Mao
|links| http://arxiv.org/abs/2403.18775v1 |
|updated| 2024-03-27 17:23:39 UTC |
|summary| We establish rigorous benchmarks for visual perception robustness. Syntheticimages such as ImageNet-C ImageNet-9 and Stylized ImageNet provide specifictype of evaluation over synthetic corruptions backgrounds and textures yetthose robustness benchmarks are restricted in specified variations and have lowsynthetic quality. In this work we introduce generative model as a data sourcefor synthesizing hard images that benchmark deep models robustness. Leveragingdiffusion models we are able to generate images with more diversifiedbackgrounds textures and materials than any prior work where we term thisbenchmark as ImageNet-D. Experimental results show that ImageNet-D results in asignificant accuracy drop to a range of vision models from the standard ResNetvisual classifier to the latest foundation models like CLIP and MiniGPT-4significantly reducing their accuracy by up to 60. Our work suggests thatdiffusion models can be an effective source to test vision models. The code anddataset are available at https://github.com/chenshuang-zhang/imagenet_d. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2403.18807v1 |
|title| ECoDepth: Effective Conditioning of Diffusion Models for Monocular Depth Estimation |
|authors| Suraj PatniAradhye AgarwalChetan Arora
|links| http://arxiv.org/abs/2403.18807v1 |
|updated| 2024-03-27 17:53:30 UTC |
|summary| In the absence of parallax cues a learning-based single image depthestimation SIDE model relies heavily on shading and contextual cues in theimage. While this simplicity is attractive it is necessary to train suchmodels on large and varied datasets which are difficult to capture. It hasbeen shown that using embeddings from pre-trained foundational models such asCLIP improves zero shot transfer in several applications. Taking inspirationfrom this in our paper we explore the use of global image priors generatedfrom a pre-trained ViT model to provide more detailed contextual information.We argue that the embedding vector from a ViT model pre-trained on a largedataset captures greater relevant information for SIDE than the usual route ofgenerating pseudo image captions followed by CLIP based text embeddings. Basedon this idea we propose a new SIDE model using a diffusion backbone which isconditioned on ViT embeddings. Our proposed design establishes a newstate-of-the-art SOTA for SIDE on NYUv2 dataset achieving Abs Rel error of0.05914 improvement compared to 0.069 by the current SOTA VPD. And onKITTI dataset achieving Sq Rel error of 0.139 2 improvement compared to0.142 by the current SOTA GEDepth. For zero-shot transfer with a modeltrained on NYUv2 we report mean relative improvement of 20 23 81 25over NeWCRFs on Sun-RGBD iBims1 DIODE HyperSim datasets compared to 1618 45 9 by ZoeDepth. The code is available athttps://github.com/Aradhye2002/EcoDepth. |


| Item |Content|
| --- |---|
|idx| 2403.18802v1 |
|title| Long-form factuality in large language models |
|authors| Jerry WeiChengrun YangXinying SongYifeng LuNathan HuDustin TranDaiyi PengRuibo LiuDa HuangCosmo DuQuoc V. Le
|links| http://arxiv.org/abs/2403.18802v1 |
|updated| 2024-03-27 17:48:55 UTC |
|summary| Large language models LLMs often generate content that contains factualerrors when responding to fact-seeking prompts on open-ended topics. Tobenchmark a models long-form factuality in open domains we first use GPT-4 togenerate LongFact a prompt set comprising thousands of questions spanning 38topics. We then propose that LLM agents can be used as automated evaluators forlong-form factuality through a method which we call Search-Augmented FactualityEvaluator SAFE. SAFE utilizes an LLM to break down a long-form response intoa set of individual facts and to evaluate the accuracy of each fact using amulti-step reasoning process comprising sending search queries to Google Searchand determining whether a fact is supported by the search results. Furthermorewe propose extending F1 score as an aggregated metric for long-form factuality.To do so we balance the percentage of supported facts in a responseprecision with the percentage of provided facts relative to a hyperparameterrepresenting a users preferred response length recall.  Empirically we demonstrate that LLM agents can achieve superhuman ratingperformance - on a set of 16k individual facts SAFE agrees with crowdsourcedhuman annotators 72 of the time and on a random subset of 100 disagreementcases SAFE wins 76 of the time. At the same time SAFE is more than 20 timescheaper than human annotators. We also benchmark thirteen language models onLongFact across four model families Gemini GPT Claude and PaLM-2 findingthat larger language models generally achieve better long-form factuality.LongFact SAFE and all experimental code are available athttps://github.com/google-deepmind/long-form-factuality. |


| Item |Content|
| --- |---|
|idx| 2403.18775v1 |
|title| ImageNet-D: Benchmarking Neural Network Robustness on Diffusion Synthetic Object |
|authors| Chenshuang ZhangFei PanJunmo KimIn So KweonChengzhi Mao
|links| http://arxiv.org/abs/2403.18775v1 |
|updated| 2024-03-27 17:23:39 UTC |
|summary| We establish rigorous benchmarks for visual perception robustness. Syntheticimages such as ImageNet-C ImageNet-9 and Stylized ImageNet provide specifictype of evaluation over synthetic corruptions backgrounds and textures yetthose robustness benchmarks are restricted in specified variations and have lowsynthetic quality. In this work we introduce generative model as a data sourcefor synthesizing hard images that benchmark deep models robustness. Leveragingdiffusion models we are able to generate images with more diversifiedbackgrounds textures and materials than any prior work where we term thisbenchmark as ImageNet-D. Experimental results show that ImageNet-D results in asignificant accuracy drop to a range of vision models from the standard ResNetvisual classifier to the latest foundation models like CLIP and MiniGPT-4significantly reducing their accuracy by up to 60. Our work suggests thatdiffusion models can be an effective source to test vision models. The code anddataset are available at https://github.com/chenshuang-zhang/imagenet_d. |


| Item |Content|
| --- |---|
|idx| 2403.18766v1 |
|title| Superior Parallel Big Data Clustering through Competitive Stochastic Sample Size Optimization in Big-means |
|authors| Rustam MussabayevRavil Mussabayev
|links| http://arxiv.org/abs/2403.18766v1 |
|updated| 2024-03-27 17:05:03 UTC |
|summary| This paper introduces a novel K-means clustering algorithm an advancement onthe conventional Big-means methodology. The proposed method efficientlyintegrates parallel processing stochastic sampling and competitiveoptimization to create a scalable variant designed for big data applications.It addresses scalability and computation time challenges typically faced withtraditional techniques. The algorithm adjusts sample sizes dynamically for eachworker during execution optimizing performance. Data from these sample sizesare continually analyzed facilitating the identification of the most efficientconfiguration. By incorporating a competitive element among workers usingdifferent sample sizes efficiency within the Big-means algorithm is furtherstimulated. In essence the algorithm balances computational time andclustering quality by employing a stochastic competitive sampling strategy ina parallel computing setting. |


| Item |Content|
| --- |---|
|idx| 2403.18765v1 |
|title| CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning |
|authors| Elliot Chane-SanePierre-Alexandre LeziartThomas FlayolsOlivier StassePhilippe SouèresNicolas Mansard
|links| http://arxiv.org/abs/2403.18765v1 |
|updated| 2024-03-27 17:03:31 UTC |
|summary| Deep Reinforcement Learning RL has demonstrated impressive results insolving complex robotic tasks such as quadruped locomotion. Yet currentsolvers fail to produce efficient policies respecting hard constraints. In thiswork we advocate for integrating constraints into robot learning and presentConstraints as Terminations CaT a novel constrained RL algorithm. Departingfrom classical constrained RL formulations we reformulate constraints throughstochastic terminations during policy learning: any violation of a constrainttriggers a probability of terminating potential future rewards the RL agentcould attain. We propose an algorithmic approach to this formulation byminimally modifying widely used off-the-shelf RL algorithms in robot learningsuch as Proximal Policy Optimization. Our approach leads to excellentconstraint adherence without introducing undue complexity and computationaloverhead thus mitigating barriers to broader adoption. Through empiricalevaluation on the real quadruped robot Solo crossing challenging obstacles wedemonstrate that CaT provides a compelling solution for incorporatingconstraints into RL frameworks. Videos and code are available athttps://constraints-as-terminations.github.io. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2403.18821v1 |
|title| Real Acoustic Fields: An Audio-Visual Room Acoustics Dataset and Benchmark |
|authors| Ziyang ChenIsrael D. GebruChristian RichardtAnurag KumarWilliam LaneyAndrew OwensAlexander Richard
|links| http://arxiv.org/abs/2403.18821v1 |
|updated| 2024-03-27 17:59:56 UTC |
|summary| We present a new dataset called Real Acoustic Fields RAF that captures realacoustic room data from multiple modalities. The dataset includes high-qualityand densely captured room impulse response data paired with multi-view imagesand precise 6DoF pose tracking data for sound emitters and listeners in therooms. We used this dataset to evaluate existing methods for novel-viewacoustic synthesis and impulse response generation which previously relied onsynthetic data. In our evaluation we thoroughly assessed existing audio andaudio-visual models against multiple criteria and proposed settings to enhancetheir performance on real-world data. We also conducted experiments toinvestigate the impact of incorporating visual data i.e. images and depthinto neural acoustic field models. Additionally we demonstrated theeffectiveness of a simple sim2real approach where a model is pre-trained withsimulated data and fine-tuned with sparse real-world data resulting insignificant improvements in the few-shot learning approach. RAF is the firstdataset to provide densely captured room acoustic data making it an idealresource for researchers working on audio and audio-visual neural acousticfield modeling techniques. Demos and datasets are available on our projectpage: https://facebookresearch.github.io/real-acoustic-fields/ |


| Item |Content|
| --- |---|
|idx| 2403.18820v1 |
|title| MetaCap: Meta-learning Priors from Multi-View Imagery for Sparse-view Human Performance Capture and Rendering |
|authors| Guoxing SunRishabh DabralPascal FuaChristian TheobaltMarc Habermann
|links| http://arxiv.org/abs/2403.18820v1 |
|updated| 2024-03-27 17:59:54 UTC |
|summary| Faithful human performance capture and free-view rendering from sparse RGBobservations is a long-standing problem in Vision and Graphics. The mainchallenges are the lack of observations and the inherent ambiguities of thesetting e.g. occlusions and depth ambiguity. As a result radiance fieldswhich have shown great promise in capturing high-frequency appearance andgeometry details in dense setups perform poorly when naively supervisingthem on sparse camera views as the field simply overfits to the sparse-viewinputs. To address this we propose MetaCap a method for efficient andhigh-quality geometry recovery and novel view synthesis given very sparse oreven a single view of the human. Our key idea is to meta-learn the radiancefield weights solely from potentially sparse multi-view videos which can serveas a prior when fine-tuning them on sparse imagery depicting the human. Thisprior provides a good network weight initialization thereby effectivelyaddressing ambiguities in sparse-view capture. Due to the articulated structureof the human body and motion-induced surface deformations learning such aprior is non-trivial. Therefore we propose to meta-learn the field weights ina pose-canonicalized space which reduces the spatial feature range and makesfeature learning more effective. Consequently one can fine-tune our fieldparameters to quickly generalize to unseen poses novel illumination conditionsas well as novel and sparse even monocular camera views. For evaluating ourmethod under different scenarios we collect a new dataset WildDynaCap whichcontains subjects captured in both a dense camera dome and in-the-wild sparsecamera rigs and demonstrate superior results compared to recentstate-of-the-art methods on both public and WildDynaCap dataset. |


| Item |Content|
| --- |---|
|idx| 2403.18819v1 |
|title| Benchmarking Object Detectors with COCO: A New Path Forward |
|authors| Shweta SinghAayan YadavJitesh JainHumphrey ShiJustin JohnsonKaran Desai
|links| http://arxiv.org/abs/2403.18819v1 |
|updated| 2024-03-27 17:59:53 UTC |
|summary| The Common Objects in Context COCO dataset has been instrumental inbenchmarking object detectors over the past decade. Like every dataset COCOcontains subtle errors and imperfections stemming from its annotationprocedure. With the advent of high-performing models we ask whether theseerrors of COCO are hindering its utility in reliably benchmarking furtherprogress. In search for an answer we inspect thousands of masks from COCO2017 version and uncover different types of errors such as imprecise maskboundaries non-exhaustively annotated instances and mislabeled masks. Due tothe prevalence of COCO we choose to correct these errors to maintaincontinuity with prior research. We develop COCO-ReM Refined Masks a cleanerset of annotations with visibly better mask quality than COCO-2017. We evaluatefifty object detectors and find that models that predict visually sharper masksscore higher on COCO-ReM affirming that they were being incorrectly penalizeddue to errors in COCO-2017. Moreover our models trained using COCO-ReMconverge faster and score higher than their larger variants trained usingCOCO-2017 highlighting the importance of data quality in improving objectdetectors. With these findings we advocate using COCO-ReM for future objectdetection research. Our dataset is available at https://cocorem.xyz |


| Item |Content|
| --- |---|
|idx| 2403.18818v1 |
|title| ObjectDrop: Bootstrapping Counterfactuals for Photorealistic Object Removal and Insertion |
|authors| Daniel WinterMatan CohenShlomi FruchterYael PritchAlex Rav-AchaYedid Hoshen
|links| http://arxiv.org/abs/2403.18818v1 |
|updated| 2024-03-27 17:59:52 UTC |
|summary| Diffusion models have revolutionized image editing but often generate imagesthat violate physical laws particularly the effects of objects on the scenee.g. occlusions shadows and reflections. By analyzing the limitations ofself-supervised approaches we propose a practical solution centered on aqcounterfactual dataset. Our method involves capturing a scene before andafter removing a single object while minimizing other changes. By fine-tuninga diffusion model on this dataset we are able to not only remove objects butalso their effects on the scene. However we find that applying this approachfor photorealistic object insertion requires an impractically large dataset. Totackle this challenge we propose bootstrap supervision leveraging our objectremoval model trained on a small counterfactual dataset we syntheticallyexpand this dataset considerably. Our approach significantly outperforms priormethods in photorealistic object removal and insertion particularly atmodeling the effects of objects on the scene. |


| Item |Content|
| --- |---|
|idx| 2403.18816v1 |
|title| Garment3DGen: 3D Garment Stylization and Texture Generation |
|authors| Nikolaos SarafianosTuur StuyckXiaoyu XiangYilei LiJovan PopovicRakesh Ranjan
|links| http://arxiv.org/abs/2403.18816v1 |
|updated| 2024-03-27 17:59:33 UTC |
|summary| We introduce Garment3DGen a new method to synthesize 3D garment assets from abase mesh given a single input image as guidance. Our proposed approach allowsusers to generate 3D textured clothes based on both real and synthetic imagessuch as those generated by text prompts. The generated assets can be directlydraped and simulated on human bodies. First we leverage the recent progress ofimage to 3D diffusion methods to generate 3D garment geometries. However sincethese geometries cannot be utilized directly for downstream tasks we proposeto use them as pseudo ground-truth and set up a mesh deformation optimizationprocedure that deforms a base template mesh to match the generated 3D target.Second we introduce carefully designed losses that allow the input base meshto freely deform towards the desired target yet preserve mesh quality andtopology such that they can be simulated. Finally a texture estimation modulegenerates high-fidelity texture maps that are globally and locally consistentand faithfully capture the input guidance allowing us to render the generated3D assets. With Garment3DGen users can generate the textured 3D garment oftheir choice without the need of artist intervention. One can provide a textualprompt describing the garment they desire to generate a simulation-ready 3Dasset. We present a plethora of quantitative and qualitative comparisons onvarious assets both real and generated and provide use-cases of how one cangenerate simulation-ready 3D garments. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2403.18739v1 |
|title| Usage-Specific Survival Modeling Based on Operational Data and Neural Networks |
|authors| Olov HolmerMattias KrysanderErik Frisk
|links| http://arxiv.org/abs/2403.18739v1 |
|updated| 2024-03-27 16:32:32 UTC |
|summary| Accurate predictions of when a component will fail are crucial when planningmaintenance and by modeling the distribution of these failure times survivalmodels have shown to be particularly useful in this context. The presentedmethodology is based on conventional neural network-based survival models thatare trained using data that is continuously gathered and stored at specifictimes called snapshots. An important property of this type of training data isthat it can contain more than one snapshot from a specific individual whichresults in that standard maximum likelihood training can not be directlyapplied since the data is not independent. However the papers show that if thedata is in a specific format where all snapshot times are the same for allindividuals called homogeneously sampled maximum likelihood training can beapplied and produce desirable results. In many cases the data is nothomogeneously sampled and in this case it is proposed to resample the data tomake it homogeneously sampled. How densely the dataset is sampled turns out tobe an important parameter it should be chosen large enough to produce goodresults but this also increases the size of the dataset which makes trainingslow. To reduce the number of samples needed during training the paper alsoproposes a technique to instead of resampling the dataset once before thetraining starts randomly resample the dataset at the start of each epochduring the training. The proposed methodology is evaluated on both a simulateddataset and an experimental dataset of starter battery failures. The resultsshow that if the data is homogeneously sampled the methodology works asintended and produces accurate survival models. The results also show thatrandomly resampling the dataset on each epoch is an effective way to reduce thesize of the training data. |


| Item |Content|
| --- |---|
|idx| 2403.18717v1 |
|title| Semi-Supervised Learning for Deep Causal Generative Models |
|authors| Yasin IbrahimHermione WarrKonstantinos Kamnitsas
|links| http://arxiv.org/abs/2403.18717v1 |
|updated| 2024-03-27 16:06:37 UTC |
|summary| Developing models that can answer questions of the form How would x changeif y had been z is fundamental for advancing medical image analysis.Training causal generative models that address such counterfactual questionsthough currently requires that all relevant variables have been observed andthat corresponding labels are available in training data. However clinicaldata may not have complete records for all patients and state of the art causalgenerative models are unable to take full advantage of this. We thus developfor the first time a semi-supervised deep causal generative model thatexploits the causal relationships between variables to maximise the use of allavailable data. We explore this in the setting where each sample is eitherfully labelled or fully unlabelled as well as the more clinically realisticcase of having different labels missing for each sample. We leverage techniquesfrom causal inference to infer missing values and subsequently generaterealistic counterfactuals even for samples with incomplete labels. |


| Item |Content|
| --- |---|
|idx| 2403.18668v1 |
|title| Aiming for Relevance |
|authors| Bar Eini PoratDanny EytanUri Shalit
|links| http://arxiv.org/abs/2403.18668v1 |
|updated| 2024-03-27 15:11:07 UTC |
|summary| Vital signs are crucial in intensive care units ICUs. They are used totrack the patients state and to identify clinically significant changes.Predicting vital sign trajectories is valuable for early detection of adverseevents. However conventional machine learning metrics like RMSE often fail tocapture the true clinical relevance of such predictions. We introduce novelvital sign prediction performance metrics that align with clinical contextsfocusing on deviations from clinical norms overall trends and trenddeviations. These metrics are derived from empirical utility curves obtained ina previous study through interviews with ICU clinicians. We validate themetrics usefulness using simulated and real clinical datasets MIMIC andeICU. Furthermore we employ these metrics as loss functions for neuralnetworks resulting in models that excel in predicting clinically significantevents. This research paves the way for clinically relevant machine learningmodel evaluation and optimization promising to improve ICU patient care. 10pages 9 figures. |


| Item |Content|
| --- |---|
|idx| 2403.18664v1 |
|title| Neural Network-Based Piecewise Survival Models |
|authors| Olov HolmerErik FriskMattias Krysander
|links| http://arxiv.org/abs/2403.18664v1 |
|updated| 2024-03-27 15:08:00 UTC |
|summary| In this paper a family of neural network-based survival models is presented.The models are specified based on piecewise definitions of the hazard functionand the density function on a partitioning of the time both constant andlinear piecewise definitions are presented resulting in a family of fourmodels. The models can be seen as an extension of the commonly useddiscrete-time and piecewise exponential models and thereby add flexibility tothis set of standard models. Using a simulated dataset the models are shown toperform well compared to the highly expressive state-of-the-art energy-basedmodel while only requiring a fraction of the computation time. |


| Item |Content|
| --- |---|
|idx| 2403.18658v1 |
|title| Theoretical Guarantees for the Subspace-Constrained Tyler's Estimator |
|authors| Gilad LermanFeng YuTeng Zhang
|links| http://arxiv.org/abs/2403.18658v1 |
|updated| 2024-03-27 15:03:29 UTC |
|summary| This work analyzes the subspace-constrained Tylers estimator STE designedfor recovering a low-dimensional subspace within a dataset that may be highlycorrupted with outliers. It assumes a weak inlier-outlier model and allows thefraction of inliers to be smaller than a fraction that leads to computationalhardness of the robust subspace recovery problem. It shows that in thissetting if the initialization of STE which is an iterative algorithmsatisfies a certain condition then STE can effectively recover the underlyingsubspace. It further shows that under the generalized haystack model STEinitialized by the Tylers M-estimator TME can recover the subspace when thefraction of iniliers is too small for TME to handle. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2403.18797v1 |
|title| SolderlessPCB: Reusing Electronic Components in PCB Prototyping through Detachable 3D Printed Housings |
|authors| Zeyu YanJiasheng LiZining ZhangHuaishu Peng
|links| http://dx.doi.org/10.1145/3613904.3642765 |
|updated| 2024-03-27 17:44:29 UTC |
|summary| The iterative prototyping process for printed circuit boards PCBsfrequently employs surface-mounted device SMD components which are oftendiscarded rather than reused due to the challenges associated with desolderingleading to unnecessary electronic waste. This paper introduces SolderlessPCB acollection of techniques for solder-free PCB prototyping specifically designedto promote the recycling and reuse of electronic components. Central to thisapproach are custom 3D-printable housings that allow SMD components to bemounted onto PCBs without soldering. We detail the design of SolderlessPCB andthe experiments conducted to evaluate its design parameters electricalperformance and durability. To illustrate the potential for reusing SMDcomponents with SolderlessPCB we discuss two scenarios: the reuse ofcomponents from earlier design iterations and from obsolete prototypes. We alsoprovide examples demonstrating that SolderlessPCB can handle high-currentapplications and is suitable for high-speed data transmission. The paperconcludes by discussing the limitations of our approach and suggesting futuredirections to overcome these challenges. |


| Item |Content|
| --- |---|
|idx| 2403.18692v1 |
|title| Teaching Introductory HRI: UChicago Course "Human-Robot Interaction: Research and Practice" |
|authors| Sarah Sebo
|links| http://arxiv.org/abs/2403.18692v1 |
|updated| 2024-03-27 15:42:01 UTC |
|summary| In 2020 I designed the course CMSC 20630/30630 Human-Robot Interaction:Research and Practice as a hands-on introduction to human-robot interactionHRI research for both undergraduate and graduate students at the Universityof Chicago. Since 2020 I have taught and refined this course each academicyear. Human-Robot Interaction: Research and Practice focuses on the coreconcepts and cutting-edge research in the field of human-robot interactionHRI covering topics that include: nonverbal robot behavior verbal robotbehavior social dynamics norms  ethics collaboration  learning groupinteractions applications and future challenges of HRI. Course meetingsinvolve students in the class leading discussions about cutting-edgepeer-reviewed research HRI publications. Students also participate in aquarter-long collaborative research project where they pursue an HRI researchquestion that often involves conducing their own human-subjects research studywhere they recruit human subjects to interact with a robot. In this paper Idetail the structure of the course and its learning goals as well as myreflections and student feedback on the course. |


| Item |Content|
| --- |---|
|idx| 2403.18679v1 |
|title| An Exploratory Study on Upper-Level Computing Students' Use of Large Language Models as Tools in a Semester-Long Project |
|authors| Ben Arie TanayLexy ArinzeSiddhant S. JoshiKirsten A. DavisJames C. Davis
|links| http://arxiv.org/abs/2403.18679v1 |
|updated| 2024-03-27 15:21:58 UTC |
|summary| Background: Large Language Models LLMs such as ChatGPT and CoPilot areinfluencing software engineering practice. Software engineering educators mustteach future software engineers how to use such tools well. As of yet therehave been few studies that report on the use of LLMs in the classroom. It istherefore important to evaluate students perception of LLMs and possible waysof adapting the computing curriculum to these shifting paradigms.  Purpose: The purpose of this study is to explore computing studentsexperiences and approaches to using LLMs during a semester-long softwareengineering project.  Design/Method: We collected data from a senior-level software engineeringcourse at Purdue University. This course uses a project-based learning PBLdesign. The students used LLMs such as ChatGPT and Copilot in their projects. Asample of these student teams were interviewed to understand 1 how they usedLLMs in their projects and 2 whether and how their perspectives on LLMschanged over the course of the semester. We analyzed the data to identifythemes related to students usage patterns and learning outcomes.  Results/Discussion: When computing students utilize LLMs within a projecttheir use cases cover both technical and professional applications. Inaddition these students perceive LLMs to be efficient tools in obtaininginformation and completion of tasks. However there were concerns about theresponsible use of LLMs without being detrimental to their own learningoutcomes. Based on our findings we recommend future research to investigatethe usage of LLMs in lower-level computer engineering courses to understandwhether and how LLMs can be integrated as a learning aid without hurting thelearning outcomes. |


| Item |Content|
| --- |---|
|idx| 2403.18668v1 |
|title| Aiming for Relevance |
|authors| Bar Eini PoratDanny EytanUri Shalit
|links| http://arxiv.org/abs/2403.18668v1 |
|updated| 2024-03-27 15:11:07 UTC |
|summary| Vital signs are crucial in intensive care units ICUs. They are used totrack the patients state and to identify clinically significant changes.Predicting vital sign trajectories is valuable for early detection of adverseevents. However conventional machine learning metrics like RMSE often fail tocapture the true clinical relevance of such predictions. We introduce novelvital sign prediction performance metrics that align with clinical contextsfocusing on deviations from clinical norms overall trends and trenddeviations. These metrics are derived from empirical utility curves obtained ina previous study through interviews with ICU clinicians. We validate themetrics usefulness using simulated and real clinical datasets MIMIC andeICU. Furthermore we employ these metrics as loss functions for neuralnetworks resulting in models that excel in predicting clinically significantevents. This research paves the way for clinically relevant machine learningmodel evaluation and optimization promising to improve ICU patient care. 10pages 9 figures. |


| Item |Content|
| --- |---|
|idx| 2403.18623v1 |
|title| Antitrust, Amazon, and Algorithmic Auditing |
|authors| Abhisek DashAbhijnan ChakrabortySaptarshi GhoshAnimesh MukherjeeJens FrankenreiterStefan BechtoldKrishna P. Gummadi
|links| http://arxiv.org/abs/2403.18623v1 |
|updated| 2024-03-27 14:34:22 UTC |
|summary| In digital markets antitrust law and special regulations aim to ensure thatmarkets remain competitive despite the dominating role that digital platformsplay today in everyones life. Unlike traditional markets market participantbehavior is easily observable in these markets. We present a series ofempirical investigations into the extent to which Amazon engages in practicesthat are typically described as self-preferencing. We discuss how the computerscience tools used in this paper can be used in a regulatory environment thatis based on algorithmic auditing and requires regulating digital markets atscale. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2403.18591v1 |
|title| Safety Verification of Wait-Only Non-Blocking Broadcast Protocols |
|authors| Lucie GuillouArnaud SangnierNathalie Sznajder
|links| http://arxiv.org/abs/2403.18591v1 |
|updated| 2024-03-27 14:17:33 UTC |
|summary| We study networks of processes that all execute the same finite protocol andcommunicate synchronously in two different ways: a process can broadcast onemessage to all other processes or send it to at most one other process. In bothcases if no process can receive the message it will still be sent. Weestablish a precise complexity class for two coverability problems with aparameterised number of processes: the state coverability problem and theconfiguration coverability problem. It is already known that these problems areAckermann-hard but decidable in the general case. We show that when theprotocol is Wait-Only i.e. it has no state from which a process can send andreceive messages the complexity drops to P and PSPACE respectively. |


| Item |Content|
| --- |---|
|idx| 2403.18166v1 |
|title| Incentive-Compatible Vertiport Reservation in Advanced Air Mobility: An Auction-Based Approach |
|authors| Pan-Yang SuChinmay MaheshwariVictoria TuckShankar Sastry
|links| http://arxiv.org/abs/2403.18166v1 |
|updated| 2024-03-27 00:21:49 UTC |
|summary| The rise of advanced air mobility AAM is expected to become amultibillion-dollar industry in the near future. Market-based mechanisms aretouted to be an integral part of AAM operations which comprise heterogeneousoperators with private valuations. In this work we study the problem ofdesigning a mechanism to coordinate the movement of electric vertical take-offand landing eVTOL aircraft operated by multiple operators each havingheterogeneous valuations associated with their fleet between vertiports whileenforcing the arrival departure and parking constraints at vertiports.Particularly we propose an incentive-compatible and individually rationalvertiport reservation mechanism that maximizes a social welfare metric whichencapsulates the objective of maximizing the overall valuations of alloperators while minimizing the congestion at vertiports. Additionally weimprove the computational tractability of designing the reservation mechanismby proposing a mixed binary linear programming approach that is based onconstructing network flow graph corresponding to the underlying problem. |


| Item |Content|
| --- |---|
|idx| 2403.18145v1 |
|title| A Real-Time Rescheduling Algorithm for Multi-robot Plan Execution |
|authors| Ying FengAdittyo PaulZhe ChenJiaoyang Li
|links| http://arxiv.org/abs/2403.18145v1 |
|updated| 2024-03-26 23:10:41 UTC |
|summary| One area of research in multi-agent path finding is to determine howreplanning can be efficiently achieved in the case of agents being delayedduring execution. One option is to reschedule the passing order of agentsi.e. the sequence in which agents visit the same location. In response wepropose Switchable-Edge Search SES an A-style algorithm designed to findoptimal passing orders. We prove the optimality of SES and evaluate itsefficiency via simulations. The best variant of SES takes less than 1 secondfor small- and medium-sized problems and runs up to 4 times faster thanbaselines for large-sized problems. |


| Item |Content|
| --- |---|
|idx| 2403.17916v1 |
|title| CMP: Cooperative Motion Prediction with Multi-Agent Communication |
|authors| Zhuoyuan WuYuping WangHengbo MaZhaowei LiHang QiuJiachen Li
|links| http://arxiv.org/abs/2403.17916v1 |
|updated| 2024-03-26 17:53:27 UTC |
|summary| The confluence of the advancement of Autonomous Vehicles AVs and thematurity of Vehicle-to-Everything V2X communication has enabled thecapability of cooperative connected and automated vehicles CAVs. Building ontop of cooperative perception this paper explores the feasibility andeffectiveness of cooperative motion prediction. Our method CMP takes LiDARsignals as input to enhance tracking and prediction capabilities. Unlikeprevious work that focuses separately on either cooperative perception ormotion prediction our framework to the best of our knowledge is the first toaddress the unified problem where CAVs share information in both perception andprediction modules. Incorporated into our design is the unique capability totolerate realistic V2X bandwidth limitations and transmission delays whiledealing with bulky perception representations. We also propose a predictionaggregation module which unifies the predictions obtained by different CAVsand generates the final prediction. Through extensive experiments and ablationstudies we demonstrate the effectiveness of our method in cooperativeperception tracking and motion prediction tasks. In particular CMP reducesthe average prediction error by 17.2 with fewer missing detections comparedwith the no cooperation setting. Our work marks a significant step forward inthe cooperative capabilities of CAVs showcasing enhanced performance incomplex scenarios. |


| Item |Content|
| --- |---|
|idx| 2403.17805v1 |
|title| Scenario-Based Curriculum Generation for Multi-Agent Autonomous Driving |
|authors| Axel BrunnbauerLuigi BerducciPeter PrillerDejan NickovicRadu Grosu
|links| http://arxiv.org/abs/2403.17805v1 |
|updated| 2024-03-26 15:42:04 UTC |
|summary| The automated generation of diverse and complex training scenarios has beenan important ingredient in many complex learning tasks. Especially inreal-world application domains such as autonomous driving auto-curriculumgeneration is considered vital for obtaining robust and general policies.However crafting traffic scenarios with multiple heterogeneous agents istypically considered as a tedious and time-consuming task especially in morecomplex simulation environments. In our work we introduce MATS-Gym aMulti-Agent Traffic Scenario framework to train agents in CARLA ahigh-fidelity driving simulator. MATS-Gym is a multi-agent training frameworkfor autonomous driving that uses partial scenario specifications to generatetraffic scenarios with variable numbers of agents. This paper unifies variousexisting approaches to traffic scenario description into a single trainingframework and demonstrates how it can be integrated with techniques fromunsupervised environment design to automate the generation of adaptiveauto-curricula. The code is available athttps://github.com/AutonomousDrivingExaminer/mats-gym. |


