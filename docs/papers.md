# cs.CL 

| Item |Content|
| --- |---|
|idx| 2410.02763v1 |
|title| Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos |
|authors| Jianrui ZhangMu CaiYong Jae Lee
|links| http://arxiv.org/abs/2410.02763v1 |
|updated| 2024-10-03 17:59:58 UTC |
|summary| There has been growing sentiment recently that modern large multimodal modelsLMMs have addressed most of the key challenges related to short videocomprehension. As a result both academia and industry are gradually shiftingtheir attention towards the more complex challenges posed by understandinglong-form videos. However is this really the case Our studies indicate thatLMMs still lack many fundamental reasoning capabilities even when dealing withshort videos. We introduce Vinoground a temporal counterfactual LMM evaluationbenchmark encompassing 1000 short and natural video-caption pairs. Wedemonstrate that existing LMMs severely struggle to distinguish temporaldifferences between different actions and object transformations. For examplethe best model GPT-4o only obtains 50 on our text and video scores showing alarge gap compared to the human baseline of 90. All open-source multimodalmodels and CLIP-based models perform much worse producing mostly random chanceperformance. Through this work we shed light onto the fact that temporalreasoning in short videos is a problem yet to be fully solved. The dataset andevaluation code are available at https://vinoground.github.io. |


| Item |Content|
| --- |---|
|idx| 2410.02760v1 |
|title| Erasing Conceptual Knowledge from Language Models |
|authors| Rohit GandikotaSheridan FeuchtSamuel MarksDavid Bau
|links| http://arxiv.org/abs/2410.02760v1 |
|updated| 2024-10-03 17:59:30 UTC |
|summary| Concept erasure in language models has traditionally lacked a comprehensiveevaluation framework leading to incomplete assessments of effectiveness oferasure methods. We propose an evaluation paradigm centered on three criticalcriteria: innocence complete knowledge removal seamlessness maintainingconditional fluent generation and specificity preserving unrelated taskperformance. Our evaluation metrics naturally motivate the development ofErasure of Language Memory ELM a new method designed to address all threedimensions. ELM employs targeted low-rank updates to alter output distributionsfor erased concepts while preserving overall model capabilities includingfluency when prompted for an erased concept. We demonstrate ELMs efficacy onbiosecurity cybersecurity and literary domain erasure tasks. Comparativeanalysis shows that ELM achieves superior performance across our proposedmetrics including near-random scores on erased topic assessments generationfluency maintained accuracy on unrelated benchmarks and robustness underadversarial attacks. Our code data and trained models are available athttps://elm.baulab.info |


| Item |Content|
| --- |---|
|idx| 2410.02756v1 |
|title| CorPipe at CRAC 2024: Predicting Zero Mentions from Raw Text |
|authors| Milan Straka
|links| http://arxiv.org/abs/2410.02756v1 |
|updated| 2024-10-03 17:58:55 UTC |
|summary| We present CorPipe 24 the winning entry to the CRAC 2024 Shared Task onMultilingual Coreference Resolution. In this third iteration of the sharedtask a novel objective is to also predict empty nodes needed for zerocoreference mentions while the empty nodes were given on input in previousyears. This way coreference resolution can be performed on raw text. Weevaluate two model variants: atwo-stage approach where the empty nodes arepredicted first using a pretrained encoder model and then processed togetherwith sentence words by another pretrained model and a single-stage approachwhere a single pretrained encoder model generates empty nodes coreferencementions and coreference links jointly. In both settings CorPipe surpassesother participants by a large margin of 3.9 and 2.8 percent pointsrespectively. The source code and the trained model are available athttps://github.com/ufal/crac2024-corpipe . |


| Item |Content|
| --- |---|
|idx| 2410.02755v1 |
|title| SIEVE: General Purpose Data Filtering System Matching GPT-4o Accuracy at 1% the Cost |
|authors| Jifan ZhangRobert Nowak
|links| http://arxiv.org/abs/2410.02755v1 |
|updated| 2024-10-03 17:58:29 UTC |
|summary| Creating specialized large language models requires vast amounts of cleanspecial purpose data for training and fine-tuning. With only a handful ofexisting large-scale domain-specific datasets creation of new datasets isrequired in most applications. This requires the development of newapplication-specific filtering of web-scale data. Filtering with ahigh-performance general-purpose LLM such as GPT-4o can be highly effectivebut this is extremely expensive at web-scale. This paper proposes SIEVE alightweight alternative that matches GPT-4o accuracy at a fraction of the cost.SIEVE can perform up to 500 filtering operations for the cost of one GPT-4ofiltering call. The key to SIEVE is a seamless integration of GPT-4o andlightweight T5 models using active learning to fine-tune T5 in the backgroundwith a small number of calls to GPT-4o. Once trained it performs as well asGPT-4o at a tiny fraction of the cost. We experimentally validate SIEVE on theOpenWebText dataset using five highly customized filter tasks targeting highquality and domain-specific content. Our results demonstrate the effectivenessand efficiency of our method in curating large high-quality datasets forlanguage model training at a substantially lower cost 1 than existingtechniques. To further validate SIEVE experiments show that SIEVE and GPT-4oachieve similar accuracy with human evaluators preferring SIEVEs filteringresults to those of GPT-4o. |


| Item |Content|
| --- |---|
|idx| 2410.02749v1 |
|title| Training Language Models on Synthetic Edit Sequences Improves Code Synthesis |
|authors| Ulyana PiterbargLerrel PintoRob Fergus
|links| http://arxiv.org/abs/2410.02749v1 |
|updated| 2024-10-03 17:57:22 UTC |
|summary| Software engineers mainly write code by editing existing programs. Incontrast large language models LLMs autoregressively synthesize programs ina single pass. One explanation for this is the scarcity of open-sourced editdata. While high-quality instruction data for code synthesis is already scarcehigh-quality edit data is even scarcer. To fill this gap we develop asynthetic data generation algorithm called LintSeq. This algorithm refactorsexisting code into a sequence of code edits by using a linter to procedurallysample across the error-free insertions that can be used to sequentially writeprograms. It outputs edit sequences as text strings consisting of consecutiveprogram diffs. To test LintSeq we use it to refactor a dataset of instruction program pairs into instruction  program-diff-sequence tuples. Then weinstruction finetune a series of smaller LLMs ranging from 2.6B to 14Bparameters on both the re-factored and original versions of this datasetcomparing zero-shot performance on code synthesis benchmarks. We show thatduring repeated sampling edit sequence finetuned models produce more diverseprograms than baselines. This results in better inference-time scaling forbenchmark coverage as a function of samples i.e. the fraction of problemspassk solved by any attempt given k tries. For example on HumanEvalpass50 small LLMs finetuned on synthetic edit sequences are competitive withGPT-4 and outperform models finetuned on the baseline dataset by 20 /-3in absolute score. Finally we also pretrain our own tiny LMs for codeunderstanding. We show that finetuning tiny models on synthetic code editsresults in state-of-the-art code synthesis for the on-device model class. Our150M parameter edit sequence LM matches or outperforms code models with twiceas many parameters both with and without repeated sampling including Codexand AlphaCode. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2410.02763v1 |
|title| Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos |
|authors| Jianrui ZhangMu CaiYong Jae Lee
|links| http://arxiv.org/abs/2410.02763v1 |
|updated| 2024-10-03 17:59:58 UTC |
|summary| There has been growing sentiment recently that modern large multimodal modelsLMMs have addressed most of the key challenges related to short videocomprehension. As a result both academia and industry are gradually shiftingtheir attention towards the more complex challenges posed by understandinglong-form videos. However is this really the case Our studies indicate thatLMMs still lack many fundamental reasoning capabilities even when dealing withshort videos. We introduce Vinoground a temporal counterfactual LMM evaluationbenchmark encompassing 1000 short and natural video-caption pairs. Wedemonstrate that existing LMMs severely struggle to distinguish temporaldifferences between different actions and object transformations. For examplethe best model GPT-4o only obtains 50 on our text and video scores showing alarge gap compared to the human baseline of 90. All open-source multimodalmodels and CLIP-based models perform much worse producing mostly random chanceperformance. Through this work we shed light onto the fact that temporalreasoning in short videos is a problem yet to be fully solved. The dataset andevaluation code are available at https://vinoground.github.io. |


| Item |Content|
| --- |---|
|idx| 2410.02761v1 |
|title| FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models |
|authors| Zhipei XuXuanyu ZhangRunyi LiZecheng TangQing HuangJian Zhang
|links| http://arxiv.org/abs/2410.02761v1 |
|updated| 2024-10-03 17:59:34 UTC |
|summary| The rapid development of generative AI is a double-edged sword which notonly facilitates content creation but also makes image manipulation easier andmore difficult to detect. Although current image forgery detection andlocalization IFDL methods are generally effective they tend to face twochallenges: textbf1 black-box nature with unknown detection principletextbf2 limited generalization across diverse tampering methods e.g.Photoshop DeepFake AIGC-Editing. To address these issues we propose theexplainable IFDL task and design FakeShield a multi-modal framework capable ofevaluating image authenticity generating tampered region masks and providinga judgment basis based on pixel-level and image-level tampering clues.Additionally we leverage GPT-4o to enhance existing IFDL datasets creatingthe Multi-Modal Tamper Description dataSet MMTD-Set for training FakeShieldstampering analysis capabilities. Meanwhile we incorporate a Domain Tag-guidedExplainable Forgery Detection Module DTE-FDM and a Multi-modal ForgeryLocalization Module MFLM to address various types of tamper detectioninterpretation and achieve forgery localization guided by detailed textualdescriptions. Extensive experiments demonstrate that FakeShield effectivelydetects and localizes various tampering techniques offering an explainable andsuperior solution compared to previous IFDL methods. |


| Item |Content|
| --- |---|
|idx| 2410.02748v1 |
|title| CriSPO: Multi-Aspect Critique-Suggestion-guided Automatic Prompt Optimization for Text Generation |
|authors| Han HeQianchu LiuLei XuChaitanya ShivadeYi ZhangSundararajan SrinivasanKatrin Kirchhoff
|links| http://arxiv.org/abs/2410.02748v1 |
|updated| 2024-10-03 17:57:01 UTC |
|summary| Large language models LLMs can generate fluent summaries across domainsusing prompting techniques reducing the need to train models for summarizationapplications. However crafting effective prompts that guide LLMs to generatesummaries with the appropriate level of detail and writing style remains achallenge. In this paper we explore the use of salient information extractedfrom the source document to enhance summarization prompts. We show that addingkeyphrases in prompts can improve ROUGE F1 and recall making the generatedsummaries more similar to the reference and more complete. The number ofkeyphrases can control the precision-recall trade-off. Furthermore ouranalysis reveals that incorporating phrase-level salient information issuperior to word- or sentence-level. However the impact on hallucination isnot universally positive across LLMs. To conduct this analysis we introduceKeyphrase Signal Extractor CriSPO a lightweight model that can be finetunedto extract salient keyphrases. By using CriSPO we achieve consistent ROUGEimprovements across datasets and open-weight and proprietary LLMs without anyLLM customization. Our findings provide insights into leveraging salientinformation in building prompt-based summarization systems. |


| Item |Content|
| --- |---|
|idx| 2410.02744v1 |
|title| Neutral residues: revisiting adapters for model extension |
|authors| Franck Signe TallaHerve JegouEdouard Grave
|links| http://arxiv.org/abs/2410.02744v1 |
|updated| 2024-10-03 17:55:17 UTC |
|summary| We address the problem of extending a pretrained large language model to anew domain that was not seen at training time like adding a language for whichthe original model has seen no or little training data. Popular solutions likefine-tuning or low-rank adaptation are successful at domain adaptation butformally they do not add any extra capacity and degrade the performance in theoriginal domain.  Our paper analyzes this extension problem under three angles: dataarchitecture and training procedure which are advantageously consideredjointly. In particular we improve adapters and make it possible to learn anentire new language while ensuring that the output of the neural network isalmost unchanged in the original domain. For this purpose we modify the newresidual blocks in a way that leads each new residual block to outputnear-zeros in the original domain.  This solution of neutral residues which borrows architectural componentsfrom mixture of experts is effective: with only 20 extra learnable weightscompared to an original model trained on English we get results that aresignificantly better than concurrent approaches fine-tuning low-rank orvanilla adapters in terms of the trade-off between learning a new language andnot forgetting English. |


| Item |Content|
| --- |---|
|idx| 2410.02741v1 |
|title| Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization |
|authors| Lei XuMohammed Asad KarimSaket DingliwalAparna Elangovan
|links| http://arxiv.org/abs/2410.02741v1 |
|updated| 2024-10-03 17:54:56 UTC |
|summary| Large language models LLMs can generate fluent summaries across domainsusing prompting techniques reducing the need to train models for summarizationapplications. However crafting effective prompts that guide LLMs to generatesummaries with the appropriate level of detail and writing style remains achallenge. In this paper we explore the use of salient information extractedfrom the source document to enhance summarization prompts. We show that addingkeyphrases in prompts can improve ROUGE F1 and recall making the generatedsummaries more similar to the reference and more complete. The number ofkeyphrases can control the precision-recall trade-off. Furthermore ouranalysis reveals that incorporating phrase-level salient information issuperior to word- or sentence-level. However the impact on hallucination isnot universally positive across LLMs. To conduct this analysis we introduceKeyphrase Signal Extractor SigExt a lightweight model that can be finetunedto extract salient keyphrases. By using SigExt we achieve consistent ROUGEimprovements across datasets and open-weight and proprietary LLMs without anyLLM customization. Our findings provide insights into leveraging salientinformation in building prompt-based summarization systems. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2410.02764v1 |
|title| Flash-Splat: 3D Reflection Removal with Flash Cues and Gaussian Splats |
|authors| Mingyang XieHaoming CaiSachin ShahYiran XuBrandon Y. FengJia-Bin HuangChristopher A. Metzler
|links| http://arxiv.org/abs/2410.02764v1 |
|updated| 2024-10-03 17:59:59 UTC |
|summary| We introduce a simple yet effective approach for separating transmitted andreflected light. Our key insight is that the powerful novel view synthesiscapabilities provided by modern inverse rendering methods e.g.3D Gaussiansplatting allow one to perform flash/no-flash reflection separation usingunpaired measurements -- this relaxation dramatically simplifies imageacquisition over conventional paired flash/no-flash reflection separationmethods. Through extensive real-world experiments we demonstrate our methodFlash-Splat accurately reconstructs both transmitted and reflected scenes in3D. Our method outperforms existing 3D reflection separation methods which donot leverage illumination control by a large margin. Our project webpage is athttps://flash-splat.github.io/. |


| Item |Content|
| --- |---|
|idx| 2410.02763v1 |
|title| Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos |
|authors| Jianrui ZhangMu CaiYong Jae Lee
|links| http://arxiv.org/abs/2410.02763v1 |
|updated| 2024-10-03 17:59:58 UTC |
|summary| There has been growing sentiment recently that modern large multimodal modelsLMMs have addressed most of the key challenges related to short videocomprehension. As a result both academia and industry are gradually shiftingtheir attention towards the more complex challenges posed by understandinglong-form videos. However is this really the case Our studies indicate thatLMMs still lack many fundamental reasoning capabilities even when dealing withshort videos. We introduce Vinoground a temporal counterfactual LMM evaluationbenchmark encompassing 1000 short and natural video-caption pairs. Wedemonstrate that existing LMMs severely struggle to distinguish temporaldifferences between different actions and object transformations. For examplethe best model GPT-4o only obtains 50 on our text and video scores showing alarge gap compared to the human baseline of 90. All open-source multimodalmodels and CLIP-based models perform much worse producing mostly random chanceperformance. Through this work we shed light onto the fact that temporalreasoning in short videos is a problem yet to be fully solved. The dataset andevaluation code are available at https://vinoground.github.io. |


| Item |Content|
| --- |---|
|idx| 2410.02762v1 |
|title| Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations |
|authors| Nick JiangAnish KachinthayaSuzie PetrykYossi Gandelsman
|links| http://arxiv.org/abs/2410.02762v1 |
|updated| 2024-10-03 17:59:57 UTC |
|summary| We investigate the internal representations of vision-language models VLMsto address hallucinations a persistent challenge despite advances in modelsize and training. We project VLMs internal image representations to theirlanguage vocabulary and observe more confident output probabilities on realobjects than hallucinated objects. We additionally use these outputprobabilities to spatially localize real objects. Building on this approach weintroduce a knowledge erasure algorithm that removes hallucinations by linearlyorthogonalizing image features with respect to hallucinated object features. Weshow that targeted edits to a models latent representations can reducehallucinations by up to 25.7 on the COCO2014 dataset while preservingperformance. Our findings demonstrate how a deeper understanding of VLMslatent representations can enhance reliability and enable novel capabilitiessuch as zero-shot segmentation. |


| Item |Content|
| --- |---|
|idx| 2410.02760v1 |
|title| Erasing Conceptual Knowledge from Language Models |
|authors| Rohit GandikotaSheridan FeuchtSamuel MarksDavid Bau
|links| http://arxiv.org/abs/2410.02760v1 |
|updated| 2024-10-03 17:59:30 UTC |
|summary| Concept erasure in language models has traditionally lacked a comprehensiveevaluation framework leading to incomplete assessments of effectiveness oferasure methods. We propose an evaluation paradigm centered on three criticalcriteria: innocence complete knowledge removal seamlessness maintainingconditional fluent generation and specificity preserving unrelated taskperformance. Our evaluation metrics naturally motivate the development ofErasure of Language Memory ELM a new method designed to address all threedimensions. ELM employs targeted low-rank updates to alter output distributionsfor erased concepts while preserving overall model capabilities includingfluency when prompted for an erased concept. We demonstrate ELMs efficacy onbiosecurity cybersecurity and literary domain erasure tasks. Comparativeanalysis shows that ELM achieves superior performance across our proposedmetrics including near-random scores on erased topic assessments generationfluency maintained accuracy on unrelated benchmarks and robustness underadversarial attacks. Our code data and trained models are available athttps://elm.baulab.info |


| Item |Content|
| --- |---|
|idx| 2410.02759v1 |
|title| Forecasting Smog Clouds With Deep Learning |
|authors| Valentijn OldenburgJuan Cardenas-CartagenaMatias Valdenegro-Toro
|links| http://arxiv.org/abs/2410.02759v1 |
|updated| 2024-10-03 17:59:13 UTC |
|summary| In this proof-of-concept study we conduct multivariate timeseriesforecasting for the concentrations of nitrogen dioxide NO2 ozone O3 andfine particulate matter PM10  PM2.5 with meteorological covariates betweentwo locations using various deep learning models with a focus on longshort-term memory LSTM and gated recurrent unit GRU architectures. Inparticular we propose an integrated hierarchical model architecture inspiredby air pollution dynamics and atmospheric science that employs multi-tasklearning and is benchmarked by unidirectional and fully-connected models.Results demonstrate that above all the hierarchical GRU proves itself as acompetitive and efficient method for forecasting the concentration ofsmog-related pollutants. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2410.02764v1 |
|title| Flash-Splat: 3D Reflection Removal with Flash Cues and Gaussian Splats |
|authors| Mingyang XieHaoming CaiSachin ShahYiran XuBrandon Y. FengJia-Bin HuangChristopher A. Metzler
|links| http://arxiv.org/abs/2410.02764v1 |
|updated| 2024-10-03 17:59:59 UTC |
|summary| We introduce a simple yet effective approach for separating transmitted andreflected light. Our key insight is that the powerful novel view synthesiscapabilities provided by modern inverse rendering methods e.g.3D Gaussiansplatting allow one to perform flash/no-flash reflection separation usingunpaired measurements -- this relaxation dramatically simplifies imageacquisition over conventional paired flash/no-flash reflection separationmethods. Through extensive real-world experiments we demonstrate our methodFlash-Splat accurately reconstructs both transmitted and reflected scenes in3D. Our method outperforms existing 3D reflection separation methods which donot leverage illumination control by a large margin. Our project webpage is athttps://flash-splat.github.io/. |


| Item |Content|
| --- |---|
|idx| 2410.02763v1 |
|title| Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos |
|authors| Jianrui ZhangMu CaiYong Jae Lee
|links| http://arxiv.org/abs/2410.02763v1 |
|updated| 2024-10-03 17:59:58 UTC |
|summary| There has been growing sentiment recently that modern large multimodal modelsLMMs have addressed most of the key challenges related to short videocomprehension. As a result both academia and industry are gradually shiftingtheir attention towards the more complex challenges posed by understandinglong-form videos. However is this really the case Our studies indicate thatLMMs still lack many fundamental reasoning capabilities even when dealing withshort videos. We introduce Vinoground a temporal counterfactual LMM evaluationbenchmark encompassing 1000 short and natural video-caption pairs. Wedemonstrate that existing LMMs severely struggle to distinguish temporaldifferences between different actions and object transformations. For examplethe best model GPT-4o only obtains 50 on our text and video scores showing alarge gap compared to the human baseline of 90. All open-source multimodalmodels and CLIP-based models perform much worse producing mostly random chanceperformance. Through this work we shed light onto the fact that temporalreasoning in short videos is a problem yet to be fully solved. The dataset andevaluation code are available at https://vinoground.github.io. |


| Item |Content|
| --- |---|
|idx| 2410.02762v1 |
|title| Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations |
|authors| Nick JiangAnish KachinthayaSuzie PetrykYossi Gandelsman
|links| http://arxiv.org/abs/2410.02762v1 |
|updated| 2024-10-03 17:59:57 UTC |
|summary| We investigate the internal representations of vision-language models VLMsto address hallucinations a persistent challenge despite advances in modelsize and training. We project VLMs internal image representations to theirlanguage vocabulary and observe more confident output probabilities on realobjects than hallucinated objects. We additionally use these outputprobabilities to spatially localize real objects. Building on this approach weintroduce a knowledge erasure algorithm that removes hallucinations by linearlyorthogonalizing image features with respect to hallucinated object features. Weshow that targeted edits to a models latent representations can reducehallucinations by up to 25.7 on the COCO2014 dataset while preservingperformance. Our findings demonstrate how a deeper understanding of VLMslatent representations can enhance reliability and enable novel capabilitiessuch as zero-shot segmentation. |


| Item |Content|
| --- |---|
|idx| 2410.02761v1 |
|title| FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models |
|authors| Zhipei XuXuanyu ZhangRunyi LiZecheng TangQing HuangJian Zhang
|links| http://arxiv.org/abs/2410.02761v1 |
|updated| 2024-10-03 17:59:34 UTC |
|summary| The rapid development of generative AI is a double-edged sword which notonly facilitates content creation but also makes image manipulation easier andmore difficult to detect. Although current image forgery detection andlocalization IFDL methods are generally effective they tend to face twochallenges: textbf1 black-box nature with unknown detection principletextbf2 limited generalization across diverse tampering methods e.g.Photoshop DeepFake AIGC-Editing. To address these issues we propose theexplainable IFDL task and design FakeShield a multi-modal framework capable ofevaluating image authenticity generating tampered region masks and providinga judgment basis based on pixel-level and image-level tampering clues.Additionally we leverage GPT-4o to enhance existing IFDL datasets creatingthe Multi-Modal Tamper Description dataSet MMTD-Set for training FakeShieldstampering analysis capabilities. Meanwhile we incorporate a Domain Tag-guidedExplainable Forgery Detection Module DTE-FDM and a Multi-modal ForgeryLocalization Module MFLM to address various types of tamper detectioninterpretation and achieve forgery localization guided by detailed textualdescriptions. Extensive experiments demonstrate that FakeShield effectivelydetects and localizes various tampering techniques offering an explainable andsuperior solution compared to previous IFDL methods. |


| Item |Content|
| --- |---|
|idx| 2410.02757v1 |
|title| Loong: Generating Minute-level Long Videos with Autoregressive Language Models |
|authors| Yuqing WangTianwei XiongDaquan ZhouZhijie LinYang ZhaoBingyi KangJiashi FengXihui Liu
|links| http://arxiv.org/abs/2410.02757v1 |
|updated| 2024-10-03 17:59:02 UTC |
|summary| It is desirable but challenging to generate content-rich long videos in thescale of minutes. Autoregressive large language models LLMs have achievedgreat success in generating coherent and long sequences of tokens in the domainof natural language processing while the exploration of autoregressive LLMsfor video generation is limited to generating short videos of several seconds.In this work we conduct a deep analysis of the challenges that preventautoregressive LLM-based video generators from generating long videos. Based onthe observations and analysis we propose Loong a new autoregressive LLM-basedvideo generator that can generate minute-long videos. Specifically we modelthe text tokens and video tokens as a unified sequence for autoregressive LLMsand train the model from scratch. We propose progressive short-to-long trainingwith a loss re-weighting scheme to mitigate the loss imbalance problem for longvideo training. We further investigate inference strategies including videotoken re-encoding and sampling strategies to diminish error accumulationduring inference. Our proposed Loong can be trained on 10-second videos and beextended to generate minute-level long videos conditioned on text prompts asdemonstrated by the results. More samples are available at:https://epiphqny.github.io/Loong-video. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2410.02724v1 |
|title| Large Language Models as Markov Chains |
|authors| Oussama ZekriAmbroise OdonnatAbdelhakim BenechehabLinus BleisteinNicolas BoulléIevgen Redko
|links| http://arxiv.org/abs/2410.02724v1 |
|updated| 2024-10-03 17:45:31 UTC |
|summary| Large language models LLMs have proven to be remarkably efficient bothacross a wide range of natural language processing tasks and well beyond them.However a comprehensive theoretical analysis of the origins of theirimpressive performance remains elusive. In this paper we approach thischallenging task by drawing an equivalence between generic autoregressivelanguage models with vocabulary of size T and context window of size K andMarkov chains defined on a finite state space of size mathcalOTK. Wederive several surprising findings related to the existence of a stationarydistribution of Markov chains that capture the inference power of LLMs theirspeed of convergence to it and the influence of the temperature on the latter.We then prove pre-training and in-context generalization bounds and show howthe drawn equivalence allows us to enrich their interpretation. Finally weillustrate our theoretical guarantees with experiments on several recent LLMsto highlight how they capture the behavior observed in practice. |


| Item |Content|
| --- |---|
|idx| 2410.02680v1 |
|title| Highly Adaptive Ridge |
|authors| Alejandro SchulerAlexander HagemeisterMark van der Laan
|links| http://arxiv.org/abs/2410.02680v1 |
|updated| 2024-10-03 17:06:06 UTC |
|summary| In this paper we propose the Highly Adaptive Ridge HAR: a regression methodthat achieves a n-1/3 dimension-free L2 convergence rate in the class ofright-continuous functions with square-integrable sectional derivatives. Thisis a large nonparametric function class that is particularly appropriate fortabular data. HAR is exactly kernel ridge regression with a specificdata-adaptive kernel based on a saturated zero-order tensor-product splinebasis expansion. We use simulation and real data to confirm our theory. Wedemonstrate empirical performance better than state-of-the-art algorithms forsmall datasets in particular. |


| Item |Content|
| --- |---|
|idx| 2410.02667v1 |
|title| GUD: Generation with Unified Diffusion |
|authors| Mathis GerdesMax WellingMiranda C. N. Cheng
|links| http://arxiv.org/abs/2410.02667v1 |
|updated| 2024-10-03 16:51:14 UTC |
|summary| Diffusion generative models transform noise into data by inverting a processthat progressively adds noise to data samples. Inspired by concepts from therenormalization group in physics which analyzes systems across differentscales we revisit diffusion models by exploring three key design aspects: 1the choice of representation in which the diffusion process operates e.g.pixel- PCA- Fourier- or wavelet-basis 2 the prior distribution that datais transformed into during diffusion e.g. Gaussian with covariance Sigmaand 3 the scheduling of noise levels applied separately to different parts ofthe data captured by a component-wise noise schedule. Incorporating theflexibility in these choices we develop a unified framework for diffusiongenerative models with greatly enhanced design freedom. In particular weintroduce soft-conditioning models that smoothly interpolate between standarddiffusion models and autoregressive models in any basis conceptuallybridging these two approaches. Our framework opens up a wide design space whichmay lead to more efficient training and data generation and paves the way tonovel architectures integrating different generative approaches and generationtasks. |


| Item |Content|
| --- |---|
|idx| 2410.02626v1 |
|title| Online Learning Guided Quasi-Newton Methods with Global Non-Asymptotic Convergence |
|authors| Ruichen JiangAryan Mokhtari
|links| http://arxiv.org/abs/2410.02626v1 |
|updated| 2024-10-03 16:08:16 UTC |
|summary| In this paper we propose a quasi-Newton method for solving smooth andmonotone nonlinear equations including unconstrained minimization and minimaxoptimization as special cases. For the strongly monotone setting we establishtwo global convergence bounds: i a linear convergence rate that matches therate of the celebrated extragradient method and ii an explicit globalsuperlinear convergence rate that provably surpasses the linear convergencerate after at most Od iterations where d is the problems dimension.In addition for the case where the operator is only monotone we prove aglobal convergence rate of Omin1/ksqrtd/k1.25 interms of the duality gap. This matches the rate of the extragradient methodwhen k  Od2 and is faster when k  Omegad2. These results are thefirst global convergence results to demonstrate a provable advantage of aquasi-Newton method over the extragradient method without querying theJacobian of the operator. Unlike classical quasi-Newton methods we achievethis by using the hybrid proximal extragradient framework and a novel onlinelearning approach for updating the Jacobian approximation matrices.Specifically guided by the convergence analysis we formulate the Jacobianapproximation update as an online convex optimization problem overnon-symmetric matrices relating the regret of the online problem to theconvergence rate of our method. To facilitate efficient implementation wefurther develop a tailored online learning algorithm based on an approximateseparation oracle which preserves structures such as symmetry and sparsity inthe Jacobian matrices. |


| Item |Content|
| --- |---|
|idx| 2410.02623v1 |
|title| Ranking Perspective for Tree-based Methods with Applications to Symbolic Feature Selection |
|authors| Hengrui LuoMeng Li
|links| http://arxiv.org/abs/2410.02623v1 |
|updated| 2024-10-03 16:03:39 UTC |
|summary| Tree-based methods are powerful nonparametric techniques in statistics andmachine learning. However their effectiveness particularly in finite-samplesettings is not fully understood. Recent applications have revealed theirsurprising ability to distinguish transformations which we call symbolicfeature selection that remain obscure under current theoretical understanding.This work provides a finite-sample analysis of tree-based methods from aranking perspective. We link oracle partitions in tree methods to responserankings at local splits offering new insights into their finite-samplebehavior in regression and feature selection tasks. Building on this localranking perspective we extend our analysis in two ways: i We examine theglobal ranking performance of individual trees and ensembles includingClassification and Regression Trees CART and Bayesian Additive RegressionTrees BART providing finite-sample oracle bounds ranking consistency andposterior contraction results. ii Inspired by the ranking perspective wepropose concordant divergence statistics mathcalT_0 to evaluate symbolicfeature mappings and establish their properties. Numerical experimentsdemonstrate the competitive performance of these statistics in symbolic featureselection tasks compared to existing methods. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2410.02454v1 |
|title| Aggregation of Constrained Crowd Opinions for Urban Planning |
|authors| Akanksha DasJyoti PatelMalay Bhattacharyya
|links| http://arxiv.org/abs/2410.02454v1 |
|updated| 2024-10-03 13:02:34 UTC |
|summary| Collective decision making is often a customary action taken in governmentcrowdsourcing. Through ensemble of opinions popularly known as judgmentanalysis governments can satisfy majority of the people who providedopinions. This has various real-world applications like urban planning orparticipatory budgeting that require setting up em facilities based on theopinions of citizens. Recently there is an emerging interest in performingjudgment analysis on opinions that are constrained. We consider a new dimensionof this problem that accommodate background constraints in the problem ofjudgment analysis which ensures the collection of more responsible opinions.The background constraints refer to the restrictions with respect to theexisting infrastructure to be taken care of while performing the consensus ofopinions. In this paper we address the said kind of problems with efficientunsupervised approaches of learning suitably modified to cater to theconstraints of urban planning. We demonstrate the effectiveness of thisapproach in various scenarios where the opinions are taken for setting up ATMcounters and sewage lines. Our main contributions encompass a novel approach ofcollecting data for smart city planning in the presence of constraintsdevelopment of methods for opinion aggregation in various formats. As a wholewe present a new dimension of judgment analysis by adding backgroundconstraints to the problem. |


| Item |Content|
| --- |---|
|idx| 2410.02406v1 |
|title| ELLMA-T: an Embodied LLM-agent for Supporting English Language Learning in Social VR |
|authors| Mengxu PanAlexandra KitsonHongyu WanMirjana Prpa
|links| http://arxiv.org/abs/2410.02406v1 |
|updated| 2024-10-03 11:32:53 UTC |
|summary| Many people struggle with learning a new language with traditional toolsfalling short in providing contextualized learning tailored to each learnersneeds. The recent development of large language models LLMs and embodiedconversational agents ECAs in social virtual reality VR provide newopportunities to practice language learning in a contextualized andnaturalistic way that takes into account the learners language level andneeds. To explore this opportunity we developed ELLMA-T an ECA that leveragesan LLM GPT-4 and situated learning framework for supporting learning Englishlanguage in social VR VRChat. Drawing on qualitative interviews N12 wereveal the potential of ELLMA-T to generate realistic believable andcontext-specific role plays for agent-learner interaction in VR and LLMscapability to provide initial language assessment and continuous feedback tolearners. We provide five design implications for the future development ofLLM-based language agents in social VR. |


| Item |Content|
| --- |---|
|idx| 2410.02360v1 |
|title| Source Data Selection for Brain-Computer Interfaces based on Simple Features |
|authors| Frida HeskebeckCarolina BergelingBo Bernhardsson
|links| http://arxiv.org/abs/2410.02360v1 |
|updated| 2024-10-03 10:17:37 UTC |
|summary| This paper demonstrates that simple features available during the calibrationof a brain-computer interface can be utilized for source data selection toimprove the performance of the brain-computer interface for a new target userthrough transfer learning. To support this a public motor imagery dataset isused for analysis and a method called the Transfer Performance Predictormethod is presented. The simple features are based on the covariance matricesof the data and the Riemannian distance between them. The Transfer PerformancePredictor method outperforms other source data selection methods as it selectssource data that gives a better transfer learning performance for the targetusers. |


| Item |Content|
| --- |---|
|idx| 2410.02264v1 |
|title| Can Capacitive Touch Images Enhance Mobile Keyboard Decoding? |
|authors| Piyawat LertvittayakumjornShanqing CaiBilly DouCedric HoShumin Zhai
|links| http://dx.doi.org/10.1145/3654777.3676420 |
|updated| 2024-10-03 07:29:04 UTC |
|summary| Capacitive touch sensors capture the two-dimensional spatial profilereferred to as a touch heatmap of a fingers contact with a mobiletouchscreen. However the research and design of touchscreen mobile keyboards-- one of the most speed and accuracy demanding touch interfaces -- has focusedon the location of the touch centroid derived from the touch image heatmap asthe input discarding the rest of the raw spatial signals. In this paper weinvestigate whether touch heatmaps can be leveraged to further improve the tapdecoding accuracy for mobile touchscreen keyboards. Specifically we developedand evaluated machine-learning models that interpret user taps by using thecentroids and/or the heatmaps as their input and studied the contribution ofthe heatmaps to model performance. The results show that adding the heatmapinto the input feature set led to 21.4 relative reduction of character errorrates on average compared to using the centroid alone. Furthermore weconducted a live user study with the centroid-based and heatmap-based decodersbuilt into Pixel 6 Pro devices and observed lower error rate faster typingspeed and higher self-reported satisfaction score based on the heatmap-baseddecoder than the centroid-based decoder. These findings underline the promiseof utilizing touch heatmaps for improving typing experience in mobilekeyboards. |


| Item |Content|
| --- |---|
|idx| 2410.02221v1 |
|title| Capturing complex hand movements and object interactions using machine learning-powered stretchable smart textile gloves |
|authors| Arvin TashakoriZenan JiangAmir ServatiSaeid SoltanianHarishkumar NarayanaKatherine LeCaroline NakayamaChieh-ling YangZ. Jane WangJanice J. EngPeyman Servati
|links| http://dx.doi.org/10.1038/s42256-023-00780-9 |
|updated| 2024-10-03 05:32:16 UTC |
|summary| Accurate real-time tracking of dexterous hand movements and interactions hasnumerous applications in human-computer interaction metaverse robotics andtele-health. Capturing realistic hand movements is challenging because of thelarge number of articulations and degrees of freedom. Here we report accurateand dynamic tracking of articulated hand and finger movements usingstretchable washable smart gloves with embedded helical sensor yarns andinertial measurement units. The sensor yarns have a high dynamic rangeresponding to low 0.005  to high 155  strains and show stability duringextensive use and washing cycles. We use multi-stage machine learning to reportaverage joint angle estimation root mean square errors of 1.21 and 1.45 degreesfor intra- and inter-subjects cross-validation respectively matching accuracyof costly motion capture cameras without occlusion or field of viewlimitations. We report a data augmentation technique that enhances robustnessto noise and variations of sensors. We demonstrate accurate tracking ofdexterous hand movements during object interactions opening new avenues ofapplications including accurate typing on a mock paper keyboard recognition ofcomplex dynamic and static gestures adapted from American Sign Language andobject identification. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2410.02664v1 |
|title| Grounded Answers for Multi-agent Decision-making Problem through Generative World Model |
|authors| Zeyang LiuXinrui YangShiguang SunLong QianLipeng WanXingyu ChenXuguang Lan
|links| http://arxiv.org/abs/2410.02664v1 |
|updated| 2024-10-03 16:49:59 UTC |
|summary| Recent progress in generative models has stimulated significant innovationsin many fields such as image generation and chatbots. Despite their successthese models often produce sketchy and misleading solutions for complexmulti-agent decision-making problems because they miss the trial-and-errorexperience and reasoning as humans. To address this limitation we explore aparadigm that integrates a language-guided simulator into the multi-agentreinforcement learning pipeline to enhance the generated answer. The simulatoris a world model that separately learns dynamics and reward where the dynamicsmodel comprises an image tokenizer as well as a causal transformer to generateinteraction transitions autoregressively and the reward model is abidirectional transformer learned by maximizing the likelihood of trajectoriesin the expert demonstrations under language guidance. Given an image of thecurrent state and the task description we use the world model to train thejoint policy and produce the image sequence as the answer by running theconverged policy on the dynamics model. The empirical results demonstrate thatthis framework can improve the answers for multi-agent decision-making problemsby showing superior performance on the training and unseen tasks of theStarCraft Multi-Agent Challenge benchmark. In particular it can generateconsistent interaction sequences and explainable reward functions atinteraction states opening the path for training generative models of thefuture. |


| Item |Content|
| --- |---|
|idx| 2410.02603v1 |
|title| Agents' Room: Narrative Generation through Multi-step Collaboration |
|authors| Fantine HuotReinald Kim AmplayoJennimaria PalomakiAlice Shoshana JakobovitsElizabeth ClarkMirella Lapata
|links| http://arxiv.org/abs/2410.02603v1 |
|updated| 2024-10-03 15:44:42 UTC |
|summary| Writing compelling fiction is a multifaceted process combining elements suchas crafting a plot developing interesting characters and using evocativelanguage. While large language models LLMs show promise for story writingthey currently rely heavily on intricate prompting which limits their use. Wepropose Agents Room a generation framework inspired by narrative theory thatdecomposes narrative writing into subtasks tackled by specialized agents. Toillustrate our method we introduce Tell Me A Story a high-quality dataset ofcomplex writing prompts and human-written stories and a novel evaluationframework designed specifically for assessing long narratives. We show thatAgents Room generates stories that are preferred by expert evaluators overthose produced by baseline systems by leveraging collaboration andspecialization to decompose the complex story writing task into tractablecomponents. We provide extensive analysis with automated and human-basedmetrics of the generated output. |


| Item |Content|
| --- |---|
|idx| 2410.02516v1 |
|title| Learning Emergence of Interaction Patterns across Independent RL Agents in Multi-Agent Environments |
|authors| Vasanth Reddy BaddamSuat GumussoyAlmuatazbellah BokerHoda Eldardiry
|links| http://arxiv.org/abs/2410.02516v1 |
|updated| 2024-10-03 14:25:02 UTC |
|summary| Many real-world problems such as controlling swarms of drones and urbantraffic naturally lend themselves to modeling as multi-agent reinforcementlearning RL problems. However existing multi-agent RL methods often sufferfrom scalability challenges primarily due to the introduction of communicationamong agents. Consequently a key challenge lies in adapting the success ofdeep learning in single-agent RL to the multi-agent setting. In response tothis challenge we propose an approach that fundamentally reimaginesmulti-agent environments. Unlike conventional methods that model each agentindividually with separate networks our approach the Bottom Up Network BUNadopts a unique perspective. BUN treats the collective of multi-agents as aunified entity while employing a specialized weight initialization strategythat promotes independent learning. Furthermore we dynamically establishconnections among agents using gradient information enabling coordination whennecessary while maintaining these connections as limited and sparse toeffectively manage the computational budget. Our extensive empiricalevaluations across a variety of cooperative multi-agent scenarios includingtasks such as cooperative navigation and traffic control consistentlydemonstrate BUNs superiority over baseline methods with substantially reducedcomputational costs. |


| Item |Content|
| --- |---|
|idx| 2410.02511v1 |
|title| Choices are More Important than Efforts: LLM Enables Efficient Multi-Agent Exploration |
|authors| Yun QuBoyuan WangYuhang JiangJianzhun ShaoYixiu MaoCheems WangChang LiuXiangyang Ji
|links| http://arxiv.org/abs/2410.02511v1 |
|updated| 2024-10-03 14:21:23 UTC |
|summary| With expansive state-action spaces efficient multi-agent exploration remainsa longstanding challenge in reinforcement learning. Although pursuing noveltydiversity or uncertainty attracts increasing attention redundant effortsbrought by exploration without proper guidance choices poses a practical issuefor the community. This paper introduces a systematic approach termed LEMAEchoosing to channel informative task-relevant guidance from a knowledgeableLarge Language Model LLM for Efficient Multi-Agent Exploration. Specificallywe ground linguistic knowledge from LLM into symbolic key states that arecritical for task fulfillment in a discriminative manner at low LLM inferencecosts. To unleash the power of key states we design Subspace-based HindsightIntrinsic Reward SHIR to guide agents toward key states by increasing rewarddensity. Additionally we build the Key State Memory Tree KSMT to tracktransitions between key states in a specific task for organized exploration.Benefiting from diminishing redundant explorations LEMAE outperforms existingSOTA approaches on the challenging benchmarks e.g. SMAC and MPE by a largemargin achieving a 10x acceleration in certain scenarios. |


| Item |Content|
| --- |---|
|idx| 2410.02510v1 |
|title| SwarmCVT: Centroidal Voronoi Tessellation-Based Path Planning for Very-Large-Scale Robotics |
|authors| James GaoJacob LeeYuting ZhouYunze HuChang LiuPingping Zhu
|links| http://arxiv.org/abs/2410.02510v1 |
|updated| 2024-10-03 14:17:20 UTC |
|summary| Swarm robotics or very large-scale robotics VLSR has many meaningfulapplications for complicated tasks. However the complexity of motion controland energy costs stack up quickly as the number of robots increases. Inaddressing this problem our previous studies have formulated various methodsemploying macroscopic and microscopic approaches. These methods enablemicroscopic robots to adhere to a reference Gaussian mixture model GMMdistribution observed at the macroscopic scale. As a result optimizing themacroscopic level will result in an optimal overall result. However all thesemethods require systematic and global generation of Gaussian components GCswithin obstacle-free areas to construct the GMM trajectories. This workutilizes centroidal Voronoi tessellation to generate GCs methodically.Consequently it demonstrates performance improvement while also ensuringconsistency and reliability. |


