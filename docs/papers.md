# cs.CL 

| Item |Content|
| --- |---|
|idx| 2406.07546v1 |
|title| Commonsense-T2I Challenge: Can Text-to-Image Generation Models Understand Commonsense? |
|authors| Xingyu FuMuyu HeYujie LuWilliam Yang WangDan Roth
|links| http://arxiv.org/abs/2406.07546v1 |
|updated| 2024-06-11 17:59:48 UTC |
|summary| We present a novel task and benchmark for evaluating the ability oftext-to-imageT2I generation models to produce images that fit commonsense inreal life which we call Commonsense-T2I. Given two adversarial text promptscontaining an identical set of action words with minor differences such as alightbulb without electricity v.s. a lightbulb with electricity we evaluatewhether T2I models can conduct visual-commonsense reasoning e.g. produceimages that fit the lightbulb is unlit vs. the lightbulb is litcorrespondingly. Commonsense-T2I presents an adversarial challenge providingpairwise text prompts along with expected outputs. The dataset is carefullyhand-curated by experts and annotated with fine-grained labels such ascommonsense type and likelihood of the expected outputs to assist analyzingmodel behavior. We benchmark a variety of state-of-the-art sota T2I modelsand surprisingly find that there is still a large gap between image synthesisand real life photos--even the DALL-E 3 model could only achieve 48.92 onCommonsense-T2I and the stable diffusion XL model only achieves 24.92accuracy. Our experiments show that GPT-enriched prompts cannot solve thischallenge and we include a detailed analysis about possible reasons for suchdeficiency. We aim for Commonsense-T2I to serve as a high-quality evaluationbenchmark for T2I commonsense checking fostering advancements in real lifeimage generation. |


| Item |Content|
| --- |---|
|idx| 2406.07545v1 |
|title| Open-LLM-Leaderboard: From Multi-choice to Open-style Questions for LLMs Evaluation, Benchmark, and Arena |
|authors| Aidar MyrzakhanSondos Mahmoud BsharatZhiqiang Shen
|links| http://arxiv.org/abs/2406.07545v1 |
|updated| 2024-06-11 17:59:47 UTC |
|summary| Multiple-choice questions MCQ are frequently used to assess large languagemodels LLMs. Typically an LLM is given a question and selects the answerdeemed most probable after adjustments for factors like length. UnfortunatelyLLMs may inherently favor certain answer choice IDs such as A/B/C/D due toinherent biases of priori unbalanced probabilities influencing the predictionof answers based on these IDs. Previous research has introduced methods toreduce this selection bias by simply permutating options on a few testsamples and applying to new ones. Another problem of MCQ is the lottery ticketchoice by random guessing. The LLM does not learn particular knowledge butthe option is guessed correctly. This situation is especially serious for thosesmall-scale LLMs. To address them a more thorough approach involves shiftingfrom MCQ to open-style questions which can fundamentally eliminate selectionbias and random guessing issues. However transitioning causes its own set ofchallenges in 1 identifying suitable open-style questions and 2 validatingthe correctness of LLM open-style responses against human-annotatedground-truths. This work aims to tackle these significant difficulties andestablish a new LLM evaluation benchmark through entirely open-style questions.Consequently we introduce the Open-LLM-Leaderboard to track various LLMsperformance and reflect true capability of them such as GPT-4o/4/3.5 Claude3 Gemini etc. Our code and dataset are available athttps://github.com/VILA-Lab/Open-LLM-Leaderboard. |


| Item |Content|
| --- |---|
|idx| 2406.07544v1 |
|title| Situational Awareness Matters in 3D Vision Language Reasoning |
|authors| Yunze ManLiang-Yan GuiYu-Xiong Wang
|links| http://arxiv.org/abs/2406.07544v1 |
|updated| 2024-06-11 17:59:45 UTC |
|summary| Being able to carry out complicated vision language reasoning tasks in 3Dspace represents a significant milestone in developing household robots andhuman-centered embodied AI. In this work we demonstrate that a critical anddistinct challenge in 3D vision language reasoning is situational awarenesswhich incorporates two key components: 1 The autonomous agent grounds itsself-location based on a language prompt. 2 The agent answers open-endedquestions from the perspective of its calculated position. To address thischallenge we introduce SIG3D an end-to-end Situation-Grounded model for 3Dvision language reasoning. We tokenize the 3D scene into sparse voxelrepresentation and propose a language-grounded situation estimator followed bya situated question answering module. Experiments on the SQA3D and ScanQAdatasets show that SIG3D outperforms state-of-the-art models in situationestimation and question answering by a large margin e.g. an enhancement ofover 30 on situation estimation accuracy. Subsequent analysis corroboratesour architectural design choices explores the distinct functions of visual andtextual tokens and highlights the importance of situational awareness in thedomain of 3D question answering. |


| Item |Content|
| --- |---|
|idx| 2406.07524v1 |
|title| Simple and Effective Masked Diffusion Language Models |
|authors| Subham Sekhar SahooMarianne ArriolaYair SchiffAaron GokaslanEdgar MarroquinJustin T ChiuAlexander RushVolodymyr Kuleshov
|links| http://arxiv.org/abs/2406.07524v1 |
|updated| 2024-06-11 17:51:40 UTC |
|summary| While diffusion models excel at generating high-quality images prior workreports a significant performance gap between diffusion and autoregressive ARmethods in language modeling. In this work we show that simple masked discretediffusion is more performant than previously thought. We apply an effectivetraining recipe that improves the performance of masked diffusion models andderive a simplified Rao-Blackwellized objective that results in additionalimprovements. Our objective has a simple form -- it is a mixture of classicalmasked language modeling losses -- and can be used to train encoder-onlylanguage models that admit efficient samplers including ones that can generatearbitrary lengths of text semi-autoregressively like a traditional languagemodel. On language modeling benchmarks a range of masked diffusion modelstrained with modern engineering practices achieves a new state-of-the-art amongdiffusion models and approaches AR perplexity. We release our code at:https://github.com/kuleshov-group/mdlm |


| Item |Content|
| --- |---|
|idx| 2406.07522v1 |
|title| Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling |
|authors| Liliang RenYang LiuYadong LuYelong ShenChen LiangWeizhu Chen
|links| http://arxiv.org/abs/2406.07522v1 |
|updated| 2024-06-11 17:50:51 UTC |
|summary| Efficiently modeling sequences with infinite context length has been along-standing problem. Past works suffer from either the quadratic computationcomplexity or the limited extrapolation ability on length generalization. Inthis work we present Samba a simple hybrid architecture that layer-wisecombines Mamba a selective State Space Model SSM with Sliding WindowAttention SWA. Samba selectively compresses a given sequence into recurrenthidden states while still maintaining the ability to precisely recall memorieswith the attention mechanism. We scale Samba up to 3.8B parameters with 3.2Ttraining tokens and show that Samba substantially outperforms thestate-of-the-art models based on pure attention or SSMs on a wide range ofbenchmarks. When trained on 4K length sequences Samba can be efficientlyextrapolated to 256K context length with perfect memory recall and showimproved token predictions up to 1M context length. As a linear-time sequencemodel Samba enjoys a 3.73x higher throughput compared to Transformers withgrouped-query attention when processing user prompts of 128K length and 3.64xspeedup when generating 64K tokens with unlimited streaming. A sampleimplementation of Samba is publicly available inhttps://github.com/microsoft/Samba. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2406.07546v1 |
|title| Commonsense-T2I Challenge: Can Text-to-Image Generation Models Understand Commonsense? |
|authors| Xingyu FuMuyu HeYujie LuWilliam Yang WangDan Roth
|links| http://arxiv.org/abs/2406.07546v1 |
|updated| 2024-06-11 17:59:48 UTC |
|summary| We present a novel task and benchmark for evaluating the ability oftext-to-imageT2I generation models to produce images that fit commonsense inreal life which we call Commonsense-T2I. Given two adversarial text promptscontaining an identical set of action words with minor differences such as alightbulb without electricity v.s. a lightbulb with electricity we evaluatewhether T2I models can conduct visual-commonsense reasoning e.g. produceimages that fit the lightbulb is unlit vs. the lightbulb is litcorrespondingly. Commonsense-T2I presents an adversarial challenge providingpairwise text prompts along with expected outputs. The dataset is carefullyhand-curated by experts and annotated with fine-grained labels such ascommonsense type and likelihood of the expected outputs to assist analyzingmodel behavior. We benchmark a variety of state-of-the-art sota T2I modelsand surprisingly find that there is still a large gap between image synthesisand real life photos--even the DALL-E 3 model could only achieve 48.92 onCommonsense-T2I and the stable diffusion XL model only achieves 24.92accuracy. Our experiments show that GPT-enriched prompts cannot solve thischallenge and we include a detailed analysis about possible reasons for suchdeficiency. We aim for Commonsense-T2I to serve as a high-quality evaluationbenchmark for T2I commonsense checking fostering advancements in real lifeimage generation. |


| Item |Content|
| --- |---|
|idx| 2406.07545v1 |
|title| Open-LLM-Leaderboard: From Multi-choice to Open-style Questions for LLMs Evaluation, Benchmark, and Arena |
|authors| Aidar MyrzakhanSondos Mahmoud BsharatZhiqiang Shen
|links| http://arxiv.org/abs/2406.07545v1 |
|updated| 2024-06-11 17:59:47 UTC |
|summary| Multiple-choice questions MCQ are frequently used to assess large languagemodels LLMs. Typically an LLM is given a question and selects the answerdeemed most probable after adjustments for factors like length. UnfortunatelyLLMs may inherently favor certain answer choice IDs such as A/B/C/D due toinherent biases of priori unbalanced probabilities influencing the predictionof answers based on these IDs. Previous research has introduced methods toreduce this selection bias by simply permutating options on a few testsamples and applying to new ones. Another problem of MCQ is the lottery ticketchoice by random guessing. The LLM does not learn particular knowledge butthe option is guessed correctly. This situation is especially serious for thosesmall-scale LLMs. To address them a more thorough approach involves shiftingfrom MCQ to open-style questions which can fundamentally eliminate selectionbias and random guessing issues. However transitioning causes its own set ofchallenges in 1 identifying suitable open-style questions and 2 validatingthe correctness of LLM open-style responses against human-annotatedground-truths. This work aims to tackle these significant difficulties andestablish a new LLM evaluation benchmark through entirely open-style questions.Consequently we introduce the Open-LLM-Leaderboard to track various LLMsperformance and reflect true capability of them such as GPT-4o/4/3.5 Claude3 Gemini etc. Our code and dataset are available athttps://github.com/VILA-Lab/Open-LLM-Leaderboard. |


| Item |Content|
| --- |---|
|idx| 2406.07544v1 |
|title| Situational Awareness Matters in 3D Vision Language Reasoning |
|authors| Yunze ManLiang-Yan GuiYu-Xiong Wang
|links| http://arxiv.org/abs/2406.07544v1 |
|updated| 2024-06-11 17:59:45 UTC |
|summary| Being able to carry out complicated vision language reasoning tasks in 3Dspace represents a significant milestone in developing household robots andhuman-centered embodied AI. In this work we demonstrate that a critical anddistinct challenge in 3D vision language reasoning is situational awarenesswhich incorporates two key components: 1 The autonomous agent grounds itsself-location based on a language prompt. 2 The agent answers open-endedquestions from the perspective of its calculated position. To address thischallenge we introduce SIG3D an end-to-end Situation-Grounded model for 3Dvision language reasoning. We tokenize the 3D scene into sparse voxelrepresentation and propose a language-grounded situation estimator followed bya situated question answering module. Experiments on the SQA3D and ScanQAdatasets show that SIG3D outperforms state-of-the-art models in situationestimation and question answering by a large margin e.g. an enhancement ofover 30 on situation estimation accuracy. Subsequent analysis corroboratesour architectural design choices explores the distinct functions of visual andtextual tokens and highlights the importance of situational awareness in thedomain of 3D question answering. |


| Item |Content|
| --- |---|
|idx| 2406.07542v1 |
|title| Cognitive Insights Across Languages: Enhancing Multimodal Interview Analysis |
|authors| David Ortiz-PerezJose Garcia-RodriguezDavid Tomás
|links| http://arxiv.org/abs/2406.07542v1 |
|updated| 2024-06-11 17:59:31 UTC |
|summary| Cognitive decline is a natural process that occurs as individuals age. Earlydiagnosis of anomalous decline is crucial for initiating professional treatmentthat can enhance the quality of life of those affected. To address this issuewe propose a multimodal model capable of predicting Mild Cognitive Impairmentand cognitive scores. The TAUKADIAL dataset is used to conduct the evaluationwhich comprises audio recordings of clinical interviews. The proposed modeldemonstrates the ability to transcribe and differentiate between languages usedin the interviews. Subsequently the model extracts audio and text featurescombining them into a multimodal architecture to achieve robust and generalizedresults. Our approach involves in-depth research to implement various featuresobtained from the proposed modalities. |


| Item |Content|
| --- |---|
|idx| 2406.07524v1 |
|title| Simple and Effective Masked Diffusion Language Models |
|authors| Subham Sekhar SahooMarianne ArriolaYair SchiffAaron GokaslanEdgar MarroquinJustin T ChiuAlexander RushVolodymyr Kuleshov
|links| http://arxiv.org/abs/2406.07524v1 |
|updated| 2024-06-11 17:51:40 UTC |
|summary| While diffusion models excel at generating high-quality images prior workreports a significant performance gap between diffusion and autoregressive ARmethods in language modeling. In this work we show that simple masked discretediffusion is more performant than previously thought. We apply an effectivetraining recipe that improves the performance of masked diffusion models andderive a simplified Rao-Blackwellized objective that results in additionalimprovements. Our objective has a simple form -- it is a mixture of classicalmasked language modeling losses -- and can be used to train encoder-onlylanguage models that admit efficient samplers including ones that can generatearbitrary lengths of text semi-autoregressively like a traditional languagemodel. On language modeling benchmarks a range of masked diffusion modelstrained with modern engineering practices achieves a new state-of-the-art amongdiffusion models and approaches AR perplexity. We release our code at:https://github.com/kuleshov-group/mdlm |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2406.07548v1 |
|title| Image and Video Tokenization with Binary Spherical Quantization |
|authors| Yue ZhaoYuanjun XiongPhilipp Krähenbühl
|links| http://arxiv.org/abs/2406.07548v1 |
|updated| 2024-06-11 17:59:53 UTC |
|summary| We propose a new transformer-based image and video tokenizer with BinarySpherical Quantization BSQ. BSQ projects the high-dimensional visualembedding to a lower-dimensional hypersphere and then applies binaryquantization. BSQ is 1 parameter-efficient without an explicit codebook 2scalable to arbitrary token dimensions and 3 compact: compressing visualdata by up to 100times with minimal distortion. Our tokenizer uses atransformer encoder and decoder with simple block-wise causal masking tosupport variable-length videos as input. The resulting BSQ-ViT achievesstate-of-the-art visual reconstruction quality on image and videoreconstruction benchmarks with 2.4times throughput compared to the bestprior methods. Furthermore by learning an autoregressive prior for adaptivearithmetic coding BSQ-ViT achieves comparable results on video compressionwith state-of-the-art video compression standards. BSQ-ViT also enables maskedlanguage models to achieve competitive image synthesis quality to GAN- anddiffusion-based methods. |


| Item |Content|
| --- |---|
|idx| 2406.07544v1 |
|title| Situational Awareness Matters in 3D Vision Language Reasoning |
|authors| Yunze ManLiang-Yan GuiYu-Xiong Wang
|links| http://arxiv.org/abs/2406.07544v1 |
|updated| 2024-06-11 17:59:45 UTC |
|summary| Being able to carry out complicated vision language reasoning tasks in 3Dspace represents a significant milestone in developing household robots andhuman-centered embodied AI. In this work we demonstrate that a critical anddistinct challenge in 3D vision language reasoning is situational awarenesswhich incorporates two key components: 1 The autonomous agent grounds itsself-location based on a language prompt. 2 The agent answers open-endedquestions from the perspective of its calculated position. To address thischallenge we introduce SIG3D an end-to-end Situation-Grounded model for 3Dvision language reasoning. We tokenize the 3D scene into sparse voxelrepresentation and propose a language-grounded situation estimator followed bya situated question answering module. Experiments on the SQA3D and ScanQAdatasets show that SIG3D outperforms state-of-the-art models in situationestimation and question answering by a large margin e.g. an enhancement ofover 30 on situation estimation accuracy. Subsequent analysis corroboratesour architectural design choices explores the distinct functions of visual andtextual tokens and highlights the importance of situational awareness in thedomain of 3D question answering. |


| Item |Content|
| --- |---|
|idx| 2406.07542v1 |
|title| Cognitive Insights Across Languages: Enhancing Multimodal Interview Analysis |
|authors| David Ortiz-PerezJose Garcia-RodriguezDavid Tomás
|links| http://arxiv.org/abs/2406.07542v1 |
|updated| 2024-06-11 17:59:31 UTC |
|summary| Cognitive decline is a natural process that occurs as individuals age. Earlydiagnosis of anomalous decline is crucial for initiating professional treatmentthat can enhance the quality of life of those affected. To address this issuewe propose a multimodal model capable of predicting Mild Cognitive Impairmentand cognitive scores. The TAUKADIAL dataset is used to conduct the evaluationwhich comprises audio recordings of clinical interviews. The proposed modeldemonstrates the ability to transcribe and differentiate between languages usedin the interviews. Subsequently the model extracts audio and text featurescombining them into a multimodal architecture to achieve robust and generalizedresults. Our approach involves in-depth research to implement various featuresobtained from the proposed modalities. |


| Item |Content|
| --- |---|
|idx| 2406.07541v1 |
|title| CDSA: Conservative Denoising Score-based Algorithm for Offline Reinforcement Learning |
|authors| Zeyuan LiuKai YangXiu Li
|links| http://arxiv.org/abs/2406.07541v1 |
|updated| 2024-06-11 17:59:29 UTC |
|summary| Distribution shift is a major obstacle in offline reinforcement learningwhich necessitates minimizing the discrepancy between the learned policy andthe behavior policy to avoid overestimating rare or unseen actions. Previousconservative offline RL algorithms struggle to generalize to unseen actionsdespite their success in learning good in-distribution policy. In contrast wepropose to use the gradient fields of the dataset density generated from apre-trained offline RL algorithm to adjust the original actions. We decouplethe conservatism constraints from the policy thus can benefit wide offline RLalgorithms. As a consequence we propose the Conservative Denoising Score-basedAlgorithm CDSA which utilizes the denoising score-based model to model thegradient of the dataset density rather than the dataset density itself andfacilitates a more accurate and efficient method to adjust the action generatedby the pre-trained policy in a deterministic and continuous MDP environment. Inexperiments we show that our approach significantly improves the performanceof baseline algorithms in D4RL datasets and demonstrate the generalizabilityand plug-and-play capability of our model across different pre-trained offlineRL policy in different tasks. We also validate that the agent exhibits greaterrisk aversion after employing our method while showcasing its ability togeneralize effectively across diverse tasks. |


| Item |Content|
| --- |---|
|idx| 2406.07540v1 |
|title| Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance |
|authors| Kuan Heng LinSicheng MoBen KlingherFangzhou MuBolei Zhou
|links| http://arxiv.org/abs/2406.07540v1 |
|updated| 2024-06-11 17:59:01 UTC |
|summary| Recent controllable generation approaches such as FreeControl and DiffusionSelf-guidance bring fine-grained spatial and appearance control totext-to-image T2I diffusion models without training auxiliary modules.However these methods optimize the latent embedding for each type of scorefunction with longer diffusion steps making the generation processtime-consuming and limiting their flexibility and use. This work presentsCtrl-X a simple framework for T2I diffusion controlling structure andappearance without additional training or guidance. Ctrl-X designs feed-forwardstructure control to enable the structure alignment with a structure image andsemantic-aware appearance transfer to facilitate the appearance transfer from auser-input image. Extensive qualitative and quantitative experiments illustratethe superior performance of Ctrl-X on various condition inputs and modelcheckpoints. In particular Ctrl-X supports novel structure and appearancecontrol with arbitrary condition images of any modality exhibits superiorimage quality and appearance transfer compared to existing works and providesinstant plug-and-play functionality to any T2I and text-to-video T2Vdiffusion model. See our project page for an overview of the results:https://genforce.github.io/ctrl-x |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2406.07550v1 |
|title| An Image is Worth 32 Tokens for Reconstruction and Generation |
|authors| Qihang YuMark WeberXueqing DengXiaohui ShenDaniel CremersLiang-Chieh Chen
|links| http://arxiv.org/abs/2406.07550v1 |
|updated| 2024-06-11 17:59:56 UTC |
|summary| Recent advancements in generative models have highlighted the crucial role ofimage tokenization in the efficient synthesis of high-resolution images.Tokenization which transforms images into latent representations reducescomputational demands compared to directly processing pixels and enhances theeffectiveness and efficiency of the generation process. Prior methods such asVQGAN typically utilize 2D latent grids with fixed downsampling factors.However these 2D tokenizations face challenges in managing the inherentredundancies present in images where adjacent regions frequently displaysimilarities. To overcome this issue we introduce Transformer-based1-Dimensional Tokenizer TiTok an innovative approach that tokenizes imagesinto 1D latent sequences. TiTok provides a more compact latent representationyielding substantially more efficient and effective representations thanconventional techniques. For example a 256 x 256 x 3 image can be reduced tojust 32 discrete tokens a significant reduction from the 256 or 1024 tokensobtained by prior methods. Despite its compact nature TiTok achievescompetitive performance to state-of-the-art approaches. Specifically using thesame generator framework TiTok attains 1.97 gFID outperforming MaskGITbaseline significantly by 4.21 at ImageNet 256 x 256 benchmark. The advantagesof TiTok become even more significant when it comes to higher resolution. AtImageNet 512 x 512 benchmark TiTok not only outperforms state-of-the-artdiffusion model DiT-XL/2 gFID 2.74 vs. 3.04 but also reduces the imagetokens by 64x leading to 410x faster generation process. Our best-performingvariant can significantly surpasses DiT-XL/2 gFID 2.13 vs. 3.04 while stillgenerating high-quality samples 74x faster. |


| Item |Content|
| --- |---|
|idx| 2406.07551v1 |
|title| Blur-aware Spatio-temporal Sparse Transformer for Video Deblurring |
|authors| Huicong ZhangHaozhe XieHongxun Yao
|links| http://arxiv.org/abs/2406.07551v1 |
|updated| 2024-06-11 17:59:56 UTC |
|summary| Video deblurring relies on leveraging information from other frames in thevideo sequence to restore the blurred regions in the current frame. Mainstreamapproaches employ bidirectional feature propagation spatio-temporaltransformers or a combination of both to extract information from the videosequence. However limitations in memory and computational resourcesconstraints the temporal window length of the spatio-temporal transformerpreventing the extraction of longer temporal contextual information from thevideo sequence. Additionally bidirectional feature propagation is highlysensitive to inaccurate optical flow in blurry frames leading to erroraccumulation during the propagation process. To address these issues wepropose textbfBSSTNet textbfBlur-aware textbfSpatio-temporaltextbfSparse textbfTransformer Network. It introduces the blur map whichconverts the originally dense attention into a sparse form enabling a moreextensive utilization of information throughout the entire video sequence.Specifically BSSTNet 1 uses a longer temporal window in the transformerleveraging information from more distant frames to restore the blurry pixels inthe current frame. 2 introduces bidirectional feature propagation guided byblur maps which reduces error accumulation caused by the blur frame. Theexperimental results demonstrate the proposed BSSTNet outperforms thestate-of-the-art methods on the GoPro and DVD datasets. |


| Item |Content|
| --- |---|
|idx| 2406.07548v1 |
|title| Image and Video Tokenization with Binary Spherical Quantization |
|authors| Yue ZhaoYuanjun XiongPhilipp Krähenbühl
|links| http://arxiv.org/abs/2406.07548v1 |
|updated| 2024-06-11 17:59:53 UTC |
|summary| We propose a new transformer-based image and video tokenizer with BinarySpherical Quantization BSQ. BSQ projects the high-dimensional visualembedding to a lower-dimensional hypersphere and then applies binaryquantization. BSQ is 1 parameter-efficient without an explicit codebook 2scalable to arbitrary token dimensions and 3 compact: compressing visualdata by up to 100times with minimal distortion. Our tokenizer uses atransformer encoder and decoder with simple block-wise causal masking tosupport variable-length videos as input. The resulting BSQ-ViT achievesstate-of-the-art visual reconstruction quality on image and videoreconstruction benchmarks with 2.4times throughput compared to the bestprior methods. Furthermore by learning an autoregressive prior for adaptivearithmetic coding BSQ-ViT achieves comparable results on video compressionwith state-of-the-art video compression standards. BSQ-ViT also enables maskedlanguage models to achieve competitive image synthesis quality to GAN- anddiffusion-based methods. |


| Item |Content|
| --- |---|
|idx| 2406.07547v1 |
|title| Zero-shot Image Editing with Reference Imitation |
|authors| Xi ChenYutong FengMengting ChenYiyang WangShilong ZhangYu LiuYujun ShenHengshuang Zhao
|links| http://arxiv.org/abs/2406.07547v1 |
|updated| 2024-06-11 17:59:51 UTC |
|summary| Image editing serves as a practical yet challenging task considering thediverse demands from users where one of the hardest parts is to preciselydescribe how the edited image should look like. In this work we present a newform of editing termed imitative editing to help users exercise theircreativity more conveniently. Concretely to edit an image region of interestusers are free to directly draw inspiration from some in-the-wild referencese.g. some relative pictures come across online without having to cope withthe fit between the reference and the source. Such a design requires the systemto automatically figure out what to expect from the reference to perform theediting. For this purpose we propose a generative training framework dubbedMimicBrush which randomly selects two frames from a video clip masks someregions of one frame and learns to recover the masked regions using theinformation from the other frame. That way our model developed from adiffusion prior is able to capture the semantic correspondence betweenseparate images in a self-supervised manner. We experimentally show theeffectiveness of our method under various test cases as well as its superiorityover existing alternatives. We also construct a benchmark to facilitate furtherresearch. |


| Item |Content|
| --- |---|
|idx| 2406.07546v1 |
|title| Commonsense-T2I Challenge: Can Text-to-Image Generation Models Understand Commonsense? |
|authors| Xingyu FuMuyu HeYujie LuWilliam Yang WangDan Roth
|links| http://arxiv.org/abs/2406.07546v1 |
|updated| 2024-06-11 17:59:48 UTC |
|summary| We present a novel task and benchmark for evaluating the ability oftext-to-imageT2I generation models to produce images that fit commonsense inreal life which we call Commonsense-T2I. Given two adversarial text promptscontaining an identical set of action words with minor differences such as alightbulb without electricity v.s. a lightbulb with electricity we evaluatewhether T2I models can conduct visual-commonsense reasoning e.g. produceimages that fit the lightbulb is unlit vs. the lightbulb is litcorrespondingly. Commonsense-T2I presents an adversarial challenge providingpairwise text prompts along with expected outputs. The dataset is carefullyhand-curated by experts and annotated with fine-grained labels such ascommonsense type and likelihood of the expected outputs to assist analyzingmodel behavior. We benchmark a variety of state-of-the-art sota T2I modelsand surprisingly find that there is still a large gap between image synthesisand real life photos--even the DALL-E 3 model could only achieve 48.92 onCommonsense-T2I and the stable diffusion XL model only achieves 24.92accuracy. Our experiments show that GPT-enriched prompts cannot solve thischallenge and we include a detailed analysis about possible reasons for suchdeficiency. We aim for Commonsense-T2I to serve as a high-quality evaluationbenchmark for T2I commonsense checking fostering advancements in real lifeimage generation. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2406.07536v1 |
|title| Towards Fundamentally Scalable Model Selection: Asymptotically Fast Update and Selection |
|authors| Wenxiao WangWeiming ZhuangLingjuan Lyu
|links| http://arxiv.org/abs/2406.07536v1 |
|updated| 2024-06-11 17:57:49 UTC |
|summary| The advancement of deep learning technologies is bringing new models everyday motivating the study of scalable model selection. An ideal model selectionscheme should minimally support two operations efficiently over a large pool ofcandidate models: update which involves either adding a new candidate model orremoving an existing candidate model and selection which involves locatinghighly performing models for a given task. However previous solutions to modelselection require high computational complexity for at least one of these twooperations. In this work we target fundamentally more scalable modelselection that supports asymptotically fast update and asymptotically fastselection at the same time. Firstly we define isolated model embedding afamily of model selection schemes supporting asymptotically fast update andselection: With respect to the number of candidate models m the updatecomplexity is O1 and the selection consists of a single sweep over mvectors in addition to O1 model operations. Isolated model embedding alsoimplies several desirable properties for applications. Secondly we presentStandardized Embedder an empirical realization of isolated model embedding. Weassess its effectiveness by using it to select representations from a pool of100 pre-trained vision models for classification tasks and measuring theperformance gaps between the selected models and the best candidates with alinear probing protocol. Experiments suggest our realization is effective inselecting models with competitive performances and highlight isolated modelembedding as a promising direction towards model selection that isfundamentally more scalable. |


| Item |Content|
| --- |---|
|idx| 2406.07515v1 |
|title| Beyond Model Collapse: Scaling Up with Synthesized Data Requires Reinforcement |
|authors| Yunzhen FengElvis DohmatobPu YangFrancois ChartonJulia Kempe
|links| http://arxiv.org/abs/2406.07515v1 |
|updated| 2024-06-11 17:46:16 UTC |
|summary| Synthesized data from generative models is increasingly considered as analternative to human-annotated data for fine-tuning Large Language Models. Thisraises concerns about model collapse: a drop in performance of modelsfine-tuned on generated data. Considering that it is easier for both humans andmachines to tell between good and bad examples than to generate high-qualitysamples we investigate the use of feedback on synthesized data to preventmodel collapse. We derive theoretical conditions under which a Gaussian mixtureclassification model can achieve asymptotically optimal performance whentrained on feedback-augmented synthesized data and provide supportingsimulations for finite regimes. We illustrate our theoretical predictions ontwo practical problems: computing matrix eigenvalues with transformers and newssummarization with large language models which both undergo model collapsewhen trained on model-generated data. We show that training fromfeedback-augmented synthesized data either by pruning incorrect predictions orby selecting the best of several guesses can prevent model collapsevalidating popular approaches like RLHF. |


| Item |Content|
| --- |---|
|idx| 2406.07475v1 |
|title| Partially Observed Trajectory Inference using Optimal Transport and a Dynamics Prior |
|authors| Anming GuEdward ChienKristjan Greenewald
|links| http://arxiv.org/abs/2406.07475v1 |
|updated| 2024-06-11 17:21:15 UTC |
|summary| Trajectory inference seeks to recover the temporal dynamics of a populationfrom snapshots of its uncoupled temporal marginals i.e. where observedparticles are not tracked over time. Lavenant et al. arXiv:2102.09204 addressedthis challenging problem under a stochastic differential equation SDE modelwith a gradient-driven drift in the observed space introducing a minimumentropy estimator relative to the Wiener measure. Chizat et al.arXiv:2205.07146 then provided a practical grid-free mean-field Langevin MFLalgorithm using Schrodinger bridges. Motivated by the overwhelming success ofobservable state space models in the traditional paired trajectory inferenceproblem e.g. target tracking we extend the above framework to a class oflatent SDEs in the form of observable state space models. In this setting weuse partial observations to infer trajectories in the latent space under aspecified dynamics model e.g. the constant velocity/acceleration models fromtarget tracking. We introduce PO-MFL to solve this latent trajectory inferenceproblem and provide theoretical guarantees by extending the results ofarXiv:2102.09204 to the partially observed setting. We leverage the MFLframework of arXiv:2205.07146 yielding an algorithm based on entropic OTbetween dynamics-adjusted adjacent time marginals. Experiments validate therobustness of our method and the exponential convergence of the MFL dynamicsand demonstrate significant outperformance over the latent-free method ofarXiv:2205.07146 in key scenarios. |


| Item |Content|
| --- |---|
|idx| 2406.07474v1 |
|title| Quantifying Local Model Validity using Active Learning |
|authors| Sven LämmleCan BogocluRobert VoßhallAnselm HaselhoffDirk Roos
|links| http://arxiv.org/abs/2406.07474v1 |
|updated| 2024-06-11 17:20:28 UTC |
|summary| Real-world applications of machine learning models are often subject to legalor policy-based regulations. Some of these regulations require ensuring thevalidity of the model i.e. the approximation error being smaller than athreshold. A global metric is generally too insensitive to determine thevalidity of a specific prediction whereas evaluating local validity is costlysince it requires gathering additional data.We propose learning the model errorto acquire a local validity estimate while reducing the amount of required datathrough active learning. Using model validation benchmarks we provideempirical evidence that the proposed method can lead to an error model withsufficient discriminative properties using a relatively small amount of data.Furthermore an increased sensitivity to local changes of the validity boundscompared to alternative approaches is demonstrated. |


| Item |Content|
| --- |---|
|idx| 2406.07457v1 |
|title| Estimating the Hallucination Rate of Generative AI |
|authors| Andrew JessonNicolas Beltran-VelezQuentin ChuSweta KarlekarJannik KossenYarin GalJohn P. CunninghamDavid Blei
|links| http://arxiv.org/abs/2406.07457v1 |
|updated| 2024-06-11 17:01:52 UTC |
|summary| This work is about estimating the hallucination rate for in-context learningICL with Generative AI. In ICL a conditional generative model CGM isprompted with a dataset and asked to make a prediction based on that dataset.The Bayesian interpretation of ICL assumes that the CGM is calculating aposterior predictive distribution over an unknown Bayesian model of a latentparameter and data. With this perspective we define a textithallucinationas a generated prediction that has low-probability under the true latentparameter. We develop a new method that takes an ICL problem -- that is a CGMa dataset and a prediction question -- and estimates the probability that aCGM will generate a hallucination. Our method only requires generating queriesand responses from the model and evaluating its response log probability. Weempirically evaluate our method on synthetic regression and natural languageICL tasks using large language models. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2406.07485v1 |
|title| PITCH: Productivity and Mental Well-being Coaching through Daily Conversational Interaction |
|authors| Adnan AbbasSang Won Lee
|links| http://arxiv.org/abs/2406.07485v1 |
|updated| 2024-06-11 17:26:58 UTC |
|summary| Efficient task planning is essential for productivity and mental well-beingyet individuals often struggle to create realistic plans and reflect upon theirproductivity. Leveraging the advancement in artificial intelligence AIconversational agents have emerged as a promising tool for enhancingproductivity. Our work focuses on externalizing plans through conversationaiming to solidify intentions and foster focused action thereby positivelyimpacting their productivity and mental well-being. We share our plan ofdesigning a conversational agent to offer insightful questions and reflectiveprompts for increasing plan adherence by leveraging the social interactivity ofnatural conversations. Previous studies have shown the effectiveness of suchagents but many interventions remain static leading to decreased userengagement over time. To address this limitation we propose a novel rotationand context-aware prompting strategy providing users with varied interventionsdaily. Our system PITCH utilizes large language models LLMs to facilitateexternalization and reflection on daily plans. Through this study weinvestigate the impact of externalizing tasks with conversational agents onproductivity and mental well-being and the effectiveness of a rotationstrategy in maintaining user engagement. |


| Item |Content|
| --- |---|
|idx| 2406.07379v1 |
|title| Politics in Games -- An Overview and Classification |
|authors| Lisa GutwengerStephan KellerMartin DolezalBernhard SchnöglSebastian RousKlaus PoierJohanna Pirker
|links| http://arxiv.org/abs/2406.07379v1 |
|updated| 2024-06-11 15:46:24 UTC |
|summary| The representation of politics in media influences societal perceptions andattitudes. Video games as a pervasive form of media contribute significantlyto this phenomenon. In this work we explore political themes within videogames by analyzing politically-themed games on game distribution platformsincluding Steam. We conducted a statistical examination of games with politicalcontext to identify patterns and use this as a basis to introduce a firsttaxonomy to categorize and better understand the interplay between politics andvideo games. This taxonomy offers a first framework for analyzing politicalcontent in games and also sets a foundation for future research in this field. |


| Item |Content|
| --- |---|
|idx| 2406.07369v1 |
|title| A qualitative field study on explainable AI for lay users subjected to AI cyberattacks |
|authors| Kevin McAreaveyWeiru LiuKim BautersDennis IvoryGeorge LoukasManos PanaousisHsueh-Ju ChenRea GillRachael PaylerAsimina Vasalou
|links| http://arxiv.org/abs/2406.07369v1 |
|updated| 2024-06-11 15:37:31 UTC |
|summary| In this paper we present results from a qualitative field study onexplainable AI XAI for lay users n  18 who were subjected to AIcyberattacks. The study was based on a custom-built smart heating applicationcalled Squid and was conducted over seven weeks in early 2023. Squid combined asmart radiator valve installed in participant homes with a web application thatimplemented an AI feature known as setpoint learning which is commonlyavailable in consumer smart thermostats. Development of Squid followed the XAIprinciple of interpretability-by-design where the AI feature was implementedusing a simple glass-box machine learning model with the model subsequentlyexposed to users via the web interface e.g. as interactive visualisations. AIattacks on users were simulated by injecting malicious training data and bymanipulating data used for model predictions. Research data consisted ofsemi-structured interviews researcher field notes participant diaries andapplication logs. In our analysis we reflect on the impact of XAI on usersatisfaction and user comprehension as well as its use as a tool for diagnosingAI attacks. Our results show only limited engagement with XAI features andsuggest that for Squid users common assumptions found in the XAI literaturewere not aligned to reality. On the positive side users appear to havedeveloped better mental models of the AI feature compared to previous work andthere is evidence that users did make some use of XAI as a diagnostic tool. |


| Item |Content|
| --- |---|
|idx| 2406.07362v1 |
|title| AI.vs.Clinician: Unveiling Intricate Interactions Between AI and Clinicians through an Open-Access Database |
|authors| Wanling GaoYuan LiuZhuoming YuDandan CuiWenjing LiuXiaoshuang LiangJiahui ZhaoJiyue XieHao LiLi MaNing YeYumiao KangDingfeng LuoPeng PanWei HuangZhongmou LiuJizhong HuFan HuangGangyuan ZhaoChongrong JiangTianyi WeiZhifei ZhangYunyou HuangJianfeng Zhan
|links| http://arxiv.org/abs/2406.07362v1 |
|updated| 2024-06-11 15:28:58 UTC |
|summary| Artificial Intelligence AI plays a crucial role in medical field and hasthe potential to revolutionize healthcare practices. However the success of AImodels and their impacts hinge on the synergy between AI and medicalspecialists with clinicians assuming a dominant role. Unfortunately theintricate dynamics and interactions between AI and clinicians remainundiscovered and thus hinder AI from being translated into medical practice. Toaddress this gap we have curated a groundbreaking database calledAI.vs.Clinician. This database is the first of its kind for studying theinteractions between AI and clinicians. It derives from 7500 collaborativediagnosis records on a life-threatening medical emergency -- Sepsis -- from 14medical centers across China. For the patient cohorts well-chosen from MIMICdatabases the AI-related information comprises the model property featureinput diagnosis decision and inferred probabilities of sepsis onset presentlyand within next three hours. The clinician-related information includes theviewed examination data and sequence viewed time preliminary and finaldiagnosis decisions with or without AI assistance and recommended treatment. |


| Item |Content|
| --- |---|
|idx| 2406.07323v1 |
|title| Should XAI Nudge Human Decisions with Explanation Biasing? |
|authors| Yosuke FukuchiSeiji Yamada
|links| http://arxiv.org/abs/2406.07323v1 |
|updated| 2024-06-11 14:53:07 UTC |
|summary| This paper reviews our previous trials of Nudge-XAI an approach thatintroduces automatic biases into explanations from explainable AIs XAIs withthe aim of leading users to better decisions and it discusses the benefits andchallenges. Nudge-XAI uses a user model that predicts the influence ofproviding an explanation or emphasizing it and attempts to guide users towardAI-suggested decisions without coercion. The nudge design is expected toenhance the autonomy of users reduce the risk associated with an AI makingdecisions without users full agreement and enable users to avoid AI failures.To discuss the potential of Nudge-XAI this paper reports a post-hocinvestigation of previous experimental results using cluster analysis. Theresults demonstrate the diversity of user behavior in response to Nudge-XAIwhich supports our aim of enhancing user autonomy. However it also highlightsthe challenge of users who distrust AI and falsely make decisions contrary toAI suggestions suggesting the need for personalized adjustment of the strengthof nudges to make this approach work more generally. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2406.07473v1 |
|title| Choreographing the Rhythms of Observation: Dynamics for Ranged Observer Bipartite-Unipartite SpatioTemporal (ROBUST) Networks |
|authors| Ted Edward Holmberg
|links| http://arxiv.org/abs/2406.07473v1 |
|updated| 2024-06-11 17:20:01 UTC |
|summary| Existing network analysis methods struggle to optimize observer placements indynamic environments with limited visibility. This dissertation introduces thenovel ROBUST Ranged Observer Bipartite-Unipartite SpatioTemporal frameworkoffering a significant advancement in modeling analyzing and optimizingobserver networks within complex spatiotemporal domains. ROBUST leverages aunique bipartite-unipartite approach distinguishing between observer andobservable entities while incorporating spatial constraints and temporaldynamics.  This research extends spatiotemporal network theory by introducing novelgraph-based measures including myopic degree spatial closeness centralityand edge length proportion. These measures coupled with advanced clusteringtechniques like Proximal Recurrence provide insights into network structureresilience and the effectiveness of observer placements. The ROBUST frameworkdemonstrates superior resource allocation and strategic responsiveness comparedto conventional models. Case studies in oceanographic monitoring urban safetynetworks and multi-agent path planning showcases its practical applicabilityand adaptability. Results demonstrate significant improvements in coverageresponse times and overall network efficiency.  This work paves the way for future research in incorporating imperfectknowledge refining temporal pathing methodologies and expanding the scope ofapplications. By bridging theoretical advancements with practical solutionsROBUST stands as a significant contribution to the field promising to informand inspire ongoing and future endeavors in network optimization andmulti-agent system planning. |


| Item |Content|
| --- |---|
|idx| 2406.07431v1 |
|title| Active Scout: Multi-Target Tracking Using Neural Radiance Fields in Dense Urban Environments |
|authors| Christopher D. HsuPratik Chaudhari
|links| http://arxiv.org/abs/2406.07431v1 |
|updated| 2024-06-11 16:34:16 UTC |
|summary| We study pursuit-evasion games in highly occluded urban environments e.g.tall buildings in a city where a scout quadrotor tracks multiple dynamictargets on the ground. We show that we can build a neural radiance field NeRFrepresentation of the city -- online -- using RGB and depth images fromdifferent vantage points. This representation is used to calculate theinformation gain to both explore unknown parts of the city and track thetargets -- thereby giving a completely first-principles approach to activelytracking dynamic targets. We demonstrate using a custom-built simulator usingOpen Street Maps data of Philadelphia and New York City that we can exploreand locate 20 stationary targets within 300 steps. This is slower than a greedybaseline which which does not use active perception. But for dynamic targetsthat actively hide behind occlusions we show that our approach maintains atworst a tracking error of 200m the greedy baseline can have a tracking erroras large as 600m. We observe a number of interesting properties in the scoutspolicies e.g. it switches its attention to track a different targetperiodically as the quality of the NeRF representation improves over time thescout also becomes better in terms of target tracking. |


| Item |Content|
| --- |---|
|idx| 2406.07277v1 |
|title| Speaking Your Language: Spatial Relationships in Interpretable Emergent Communication |
|authors| Olaf LipinskiAdam J. SobeyFederico CeruttiTimothy J. Norman
|links| http://arxiv.org/abs/2406.07277v1 |
|updated| 2024-06-11 14:04:25 UTC |
|summary| Effective communication requires the ability to refer to specific parts of anobservation in relation to others. While emergent communication literatureshows success in developing various language properties no research has shownthe emergence of such positional references. This paper demonstrates how agentscan communicate about spatial relationships within their observations. Theresults indicate that agents can develop a language capable of expressing therelationships between parts of their observation achieving over 90 accuracywhen trained in a referential game which requires such communication. Using acollocation measure we demonstrate how the agents create such references. Thisanalysis suggests that agents use a mixture of non-compositional andcompositional messages to convey spatial relationships. We also show that theemergent language is interpretable by humans. The translation accuracy istested by communicating with the receiver agent where the receiver achievesover 78 accuracy using parts of this lexicon confirming that theinterpretation of the emergent language was successful. |


| Item |Content|
| --- |---|
|idx| 2406.07155v1 |
|title| Scaling Large-Language-Model-based Multi-Agent Collaboration |
|authors| Chen QianZihao XieYifei WangWei LiuYufan DangZhuoyun DuWeize ChenCheng YangZhiyuan LiuMaosong Sun
|links| http://arxiv.org/abs/2406.07155v1 |
|updated| 2024-06-11 11:02:04 UTC |
|summary| Pioneering advancements in large language model-powered agents haveunderscored the design pattern of multi-agent collaboration demonstrating thatcollective intelligence can surpass the capabilities of each individual.Inspired by the neural scaling law which posits that increasing neurons leadsto emergent abilities this study investigates whether a similar principleapplies to increasing agents in multi-agent collaboration. Technically wepropose multi-agent collaboration networks MacNet which utilize directedacyclic graphs to organize agents and streamline their interactive reasoningvia topological ordering with solutions derived from their dialogues.Extensive experiments show that MacNet consistently outperforms baselinemodels enabling effective agent collaboration across various networktopologies and supporting cooperation among more than a thousand agents.Notably we observed a small-world collaboration phenomenon where topologiesresembling small-world properties achieved superior performance. Additionallywe identified a collaborative scaling law indicating that normalized solutionquality follows a logistic growth pattern as scaling agents with collaborativeemergence occurring much earlier than previously observed instances of neuralemergence. The code and data will be available athttps://github.com/OpenBMB/ChatDev. |


| Item |Content|
| --- |---|
|idx| 2406.07031v1 |
|title| Arbitrary-Order Distributed Finite-Time Differentiator for Multi-Agent Systems |
|authors| Weile ChenHaibo DuShihua LiXinghuo Yu
|links| http://arxiv.org/abs/2406.07031v1 |
|updated| 2024-06-11 07:40:46 UTC |
|summary| This paper proposes arbitrary-order distributed finite-time differentiatorAODFD for leader-follower multi-agent systems MAS under directed graph byonly using relative or absolute output information. By using arbitrary-orderdistributed finite-time differentiator via relative output informationAODFD-R each follower agent can obtain the relative output informationbetween itself and leader and the relative outputs arbitrary-orderderivatives where the information to be measured is only the local relativeoutput information between each follower agent and its neighboring agents. As asimple extension of AODFD-R the arbitrary-order distributed finite-timedifferentiator via absolute output information AODFD-A is also given. Thefinite-time stability of the closed-loop system under AODFD is proved byconstructing a Lyapunov function skillfully. Finally several simulationexamples are given to verify the effectiveness of the AODFD. |


