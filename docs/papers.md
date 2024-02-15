# cs.CL 

| Item |Content|
| --- |---|
|idx| 2402.08680v1 |
|title| Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance |
|authors| Linxi ZhaoYihe DengWeitong ZhangQuanquan Gu
|links| http://arxiv.org/abs/2402.08680v1 |
|updated| 2024-02-13 18:59:05 UTC |
|summary| The advancement of Large Vision-Language Models LVLMs has increasinglyhighlighted the critical issue of their tendency to hallucinate non-existingobjects in the images. To address this issue previous works focused on usingspecially curated datasets or powerful LLMs e.g. GPT-3.5 to rectify theoutputs of LVLMs. However these approaches require either expensivetraining/fine-tuning or API access to advanced LLMs to correct the modelsoutput post-generation. In this paper we tackle this challenge by introducinga framework called Mitigating hallucinAtion via classifieR-Free guIdaNcEMARINE which is both training-free and API-free and can effectively andefficiently reduce object hallucinations during the generation process.Specifically MARINE enriches the visual context of LVLMs by integratingexisting open-source vision models and employs classifier-free guidance toincorporate the additional object grounding features to improve the precisionof LVLMs generations. Through comprehensive evaluations across 6 popularLVLMs with diverse evaluation metrics we demonstrate the effectiveness ofMARINE which even outperforms existing fine-tuning-based methods. Remarkablyit not only reduces hallucinations but also improves the detailedness of LVLMsgenerations as assessed by GPT-4V. |


| Item |Content|
| --- |---|
|idx| 2402.08679v1 |
|title| COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability |
|authors| Xingang GuoFangxu YuHuan ZhangLianhui QinBin Hu
|links| http://arxiv.org/abs/2402.08679v1 |
|updated| 2024-02-13 18:58:48 UTC |
|summary| Jailbreaks on Large language models LLMs have recently received increasingattention. For a comprehensive assessment of LLM safety it is essential toconsider jailbreaks with diverse attributes such as contextual coherence andsentiment/stylistic variations and hence it is beneficial to studycontrollable jailbreaking i.e. how to enforce control on LLM attacks. In thispaper we formally formulate the controllable attack generation problem andbuild a novel connection between this problem and controllable text generationa well-explored topic of natural language processing. Based on this connectionwe adapt the Energy-based Constrained Decoding with Langevin Dynamics COLD astate-of-the-art highly efficient algorithm in controllable text generationand introduce the COLD-Attack framework which unifies and automates the searchof adversarial LLM attacks under a variety of control requirements such asfluency stealthiness sentiment and left-right-coherence. The controllabilityenabled by COLD-Attack leads to diverse new jailbreak scenarios which not onlycover the standard setting of generating fluent suffix attacks but also allowus to address new controllable attack settings such as revising a user queryadversarially with minimal paraphrasing and inserting stealthy attacks incontext with left-right-coherence. Our extensive experiments on various LLMsLlama-2 Mistral Vicuna Guanaco GPT-3.5 show COLD-Attacks broadapplicability strong controllability high success rate and attacktransferability. Our code is available athttps://github.com/Yu-Fangxu/COLD-Attack. |


| Item |Content|
| --- |---|
|idx| 2402.08666v1 |
|title| Improving Generalization in Semantic Parsing by Increasing Natural Language Variation |
|authors| Irina SaparinaMirella Lapata
|links| http://arxiv.org/abs/2402.08666v1 |
|updated| 2024-02-13 18:48:23 UTC |
|summary| Text-to-SQL semantic parsing has made significant progress in recent yearswith various models demonstrating impressive performance on the challengingSpider benchmark. However it has also been shown that these models oftenstruggle to generalize even when faced with small perturbations of previouslyaccurately parsed expressions. This is mainly due to the linguistic form ofquestions in Spider which are overly specific unnatural and display limitedvariation. In this work we use data augmentation to enhance the robustness oftext-to-SQL parsers against natural language variations. Existing approachesgenerate question reformulations either via models trained on Spider or onlyintroduce local changes. In contrast we leverage the capabilities of largelanguage models to generate more realistic and diverse questions. Using only afew prompts we achieve a two-fold increase in the number of questions inSpider. Training on this augmented dataset yields substantial improvements on arange of evaluation sets including robustness benchmarks and out-of-domaindata. |


| Item |Content|
| --- |---|
|idx| 2402.08644v1 |
|title| Tandem Transformers for Inference Efficient LLMs |
|authors| Aishwarya P SPranav Ajit NairYashas SamagaToby BoydSanjiv KumarPrateek JainPraneeth Netrapalli
|links| http://arxiv.org/abs/2402.08644v1 |
|updated| 2024-02-13 18:24:08 UTC |
|summary| The autoregressive nature of conventional large language models LLMsinherently limits inference speed as tokens are generated sequentially. Whilespeculative and parallel decoding techniques attempt to mitigate this theyface limitations: either relying on less accurate smaller models for generationor failing to fully leverage the base LLMs representations.  We introduce a novel architecture Tandem transformers to address theseissues. This architecture uniquely combines 1 a small autoregressive modeland 2 a large model operating in block mode processing multiple tokenssimultaneously. The small models predictive accuracy is substantiallyenhanced by granting it attention to the large models richer representations.On the PaLM2 pretraining dataset a tandem of PaLM2-Bison and PaLM2-Geckodemonstrates a 3.3 improvement in next-token prediction accuracy over astandalone PaLM2-Gecko offering a 1.16x speedup compared to a PaLM2-Ottermodel with comparable downstream performance. We further incorporate the tandemmodel within the speculative decoding SPEED framework where the large modelvalidates tokens from the small model. This ensures that the Tandem ofPaLM2-Bison and PaLM2-Gecko achieves substantial speedup around 1.14x fasterthan using vanilla PaLM2-Gecko in SPEED while maintaining identical downstreamtask accuracy. |


| Item |Content|
| --- |---|
|idx| 2402.08638v2 |
|title| SemRel2024: A Collection of Semantic Textual Relatedness Datasets for 14 Languages |
|authors| Nedjma OusidhoumShamsuddeen Hassan MuhammadMohamed AbdallaIdris AbdulmuminIbrahim Said AhmadSanchit AhujaAlham Fikri AjiVladimir AraujoAbinew Ali AyelePavan BaswaniMeriem BeloucifChris BiemannSofia BourhimChristine De KockGenet Shanko DekeboOumaima HourraneGopichand KanumoluLokesh MadasuSamuel RutundaManish ShrivastavaThamar SolorioNirmal SurangeHailegnaw Getaneh TilayeKrishnapriya VishnubhotlaGenta WinataSeid Muhie YimamSaif M. Mohammad
|links| http://arxiv.org/abs/2402.08638v2 |
|updated| 2024-02-14 09:49:52 UTC |
|summary| Exploring and quantifying semantic relatedness is central to representinglanguage. It holds significant implications across various NLP tasks includingoffering insights into the capabilities and performance of Large LanguageModels LLMs. While earlier NLP research primarily focused on semanticsimilarity often within the English language context we instead investigatethe broader phenomenon of semantic relatedness. In this paper we presentSemRel a new semantic relatedness dataset collection annotated by nativespeakers across 14 languages:Afrikaans Algerian Arabic Amharic EnglishHausa Hindi Indonesian Kinyarwanda Marathi Moroccan Arabic ModernStandard Arabic Punjabi Spanish and Telugu. These languages originate fromfive distinct language families and are predominantly spoken in Africa and Asia-- regions characterised by a relatively limited availability of NLP resources.Each instance in the SemRel datasets is a sentence pair associated with a scorethat represents the degree of semantic textual relatedness between the twosentences. The scores are obtained using a comparative annotation framework. Wedescribe the data collection and annotation processes related challenges whenbuilding the datasets and their impact and utility in NLP. We further reportexperiments for each language and across the different languages. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2402.08682v1 |
|title| IM-3D: Iterative Multiview Diffusion and Reconstruction for High-Quality 3D Generation |
|authors| Luke Melas-KyriaziIro LainaChristian RupprechtNatalia NeverovaAndrea VedaldiOran GafniFilippos Kokkinos
|links| http://arxiv.org/abs/2402.08682v1 |
|updated| 2024-02-13 18:59:51 UTC |
|summary| Most text-to-3D generators build upon off-the-shelf text-to-image modelstrained on billions of images. They use variants of Score Distillation SamplingSDS which is slow somewhat unstable and prone to artifacts. A mitigationis to fine-tune the 2D generator to be multi-view aware which can helpdistillation or can be combined with reconstruction networks to output 3Dobjects directly. In this paper we further explore the design space oftext-to-3D models. We significantly improve multi-view generation byconsidering video instead of image generators. Combined with a 3Dreconstruction algorithm which by using Gaussian splatting can optimize arobust image-based loss we directly produce high-quality 3D outputs from thegenerated views. Our new method IM-3D reduces the number of evaluations ofthe 2D generator network 10-100x resulting in a much more efficient pipelinebetter quality fewer geometric inconsistencies and higher yield of usable 3Dassets. |


| Item |Content|
| --- |---|
|idx| 2402.08680v1 |
|title| Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance |
|authors| Linxi ZhaoYihe DengWeitong ZhangQuanquan Gu
|links| http://arxiv.org/abs/2402.08680v1 |
|updated| 2024-02-13 18:59:05 UTC |
|summary| The advancement of Large Vision-Language Models LVLMs has increasinglyhighlighted the critical issue of their tendency to hallucinate non-existingobjects in the images. To address this issue previous works focused on usingspecially curated datasets or powerful LLMs e.g. GPT-3.5 to rectify theoutputs of LVLMs. However these approaches require either expensivetraining/fine-tuning or API access to advanced LLMs to correct the modelsoutput post-generation. In this paper we tackle this challenge by introducinga framework called Mitigating hallucinAtion via classifieR-Free guIdaNcEMARINE which is both training-free and API-free and can effectively andefficiently reduce object hallucinations during the generation process.Specifically MARINE enriches the visual context of LVLMs by integratingexisting open-source vision models and employs classifier-free guidance toincorporate the additional object grounding features to improve the precisionof LVLMs generations. Through comprehensive evaluations across 6 popularLVLMs with diverse evaluation metrics we demonstrate the effectiveness ofMARINE which even outperforms existing fine-tuning-based methods. Remarkablyit not only reduces hallucinations but also improves the detailedness of LVLMsgenerations as assessed by GPT-4V. |


| Item |Content|
| --- |---|
|idx| 2402.08679v1 |
|title| COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability |
|authors| Xingang GuoFangxu YuHuan ZhangLianhui QinBin Hu
|links| http://arxiv.org/abs/2402.08679v1 |
|updated| 2024-02-13 18:58:48 UTC |
|summary| Jailbreaks on Large language models LLMs have recently received increasingattention. For a comprehensive assessment of LLM safety it is essential toconsider jailbreaks with diverse attributes such as contextual coherence andsentiment/stylistic variations and hence it is beneficial to studycontrollable jailbreaking i.e. how to enforce control on LLM attacks. In thispaper we formally formulate the controllable attack generation problem andbuild a novel connection between this problem and controllable text generationa well-explored topic of natural language processing. Based on this connectionwe adapt the Energy-based Constrained Decoding with Langevin Dynamics COLD astate-of-the-art highly efficient algorithm in controllable text generationand introduce the COLD-Attack framework which unifies and automates the searchof adversarial LLM attacks under a variety of control requirements such asfluency stealthiness sentiment and left-right-coherence. The controllabilityenabled by COLD-Attack leads to diverse new jailbreak scenarios which not onlycover the standard setting of generating fluent suffix attacks but also allowus to address new controllable attack settings such as revising a user queryadversarially with minimal paraphrasing and inserting stealthy attacks incontext with left-right-coherence. Our extensive experiments on various LLMsLlama-2 Mistral Vicuna Guanaco GPT-3.5 show COLD-Attacks broadapplicability strong controllability high success rate and attacktransferability. Our code is available athttps://github.com/Yu-Fangxu/COLD-Attack. |


| Item |Content|
| --- |---|
|idx| 2402.08672v1 |
|title| Model Assessment and Selection under Temporal Distribution Shift |
|authors| Elise HanChengpiao HuangKaizheng Wang
|links| http://arxiv.org/abs/2402.08672v1 |
|updated| 2024-02-13 18:54:08 UTC |
|summary| We investigate model assessment and selection in a changing environment bysynthesizing datasets from both the current time period and historical epochs.To tackle unknown and potentially arbitrary temporal distribution shift wedevelop an adaptive rolling window approach to estimate the generalizationerror of a given model. This strategy also facilitates the comparison betweenany two candidate models by estimating the difference of their generalizationerrors. We further integrate pairwise comparisons into a single-eliminationtournament achieving near-optimal model selection from a collection ofcandidates. Theoretical analyses and numerical experiments demonstrate theadaptivity of our proposed methods to the non-stationarity in data. |


| Item |Content|
| --- |---|
|idx| 2402.08671v1 |
|title| Are Semi-Dense Detector-Free Methods Good at Matching Local Features? |
|authors| Matthieu VilainRémi GiraudHugo GermainGuillaume Bourmaud
|links| http://arxiv.org/abs/2402.08671v1 |
|updated| 2024-02-13 18:53:13 UTC |
|summary| Semi-dense detector-free approaches SDF such as LoFTR are currently amongthe most popular image matching methods. While SDF methods are trained toestablish correspondences between two images their performances are almostexclusively evaluated using relative pose estimation metrics. Thus the linkbetween their ability to establish correspondences and the quality of theresulting estimated pose has thus far received little attention. This paper isa first attempt to study this link. We start with proposing a novel structuredattention-based image matching architecture SAM. It allows us to show acounter-intuitive result on two datasets MegaDepth and HPatches: on the onehand SAM either outperforms or is on par with SDF methods in terms ofpose/homography estimation metrics but on the other hand SDF approaches aresignificantly better than SAM in terms of matching accuracy. We then propose tolimit the computation of the matching accuracy to textured regions and showthat in this case SAM often surpasses SDF methods. Our findings highlight astrong correlation between the ability to establish accurate correspondences intextured regions and the accuracy of the resulting estimated pose/homography.Our code will be made available. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2402.08682v1 |
|title| IM-3D: Iterative Multiview Diffusion and Reconstruction for High-Quality 3D Generation |
|authors| Luke Melas-KyriaziIro LainaChristian RupprechtNatalia NeverovaAndrea VedaldiOran GafniFilippos Kokkinos
|links| http://arxiv.org/abs/2402.08682v1 |
|updated| 2024-02-13 18:59:51 UTC |
|summary| Most text-to-3D generators build upon off-the-shelf text-to-image modelstrained on billions of images. They use variants of Score Distillation SamplingSDS which is slow somewhat unstable and prone to artifacts. A mitigationis to fine-tune the 2D generator to be multi-view aware which can helpdistillation or can be combined with reconstruction networks to output 3Dobjects directly. In this paper we further explore the design space oftext-to-3D models. We significantly improve multi-view generation byconsidering video instead of image generators. Combined with a 3Dreconstruction algorithm which by using Gaussian splatting can optimize arobust image-based loss we directly produce high-quality 3D outputs from thegenerated views. Our new method IM-3D reduces the number of evaluations ofthe 2D generator network 10-100x resulting in a much more efficient pipelinebetter quality fewer geometric inconsistencies and higher yield of usable 3Dassets. |


| Item |Content|
| --- |---|
|idx| 2402.08680v1 |
|title| Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance |
|authors| Linxi ZhaoYihe DengWeitong ZhangQuanquan Gu
|links| http://arxiv.org/abs/2402.08680v1 |
|updated| 2024-02-13 18:59:05 UTC |
|summary| The advancement of Large Vision-Language Models LVLMs has increasinglyhighlighted the critical issue of their tendency to hallucinate non-existingobjects in the images. To address this issue previous works focused on usingspecially curated datasets or powerful LLMs e.g. GPT-3.5 to rectify theoutputs of LVLMs. However these approaches require either expensivetraining/fine-tuning or API access to advanced LLMs to correct the modelsoutput post-generation. In this paper we tackle this challenge by introducinga framework called Mitigating hallucinAtion via classifieR-Free guIdaNcEMARINE which is both training-free and API-free and can effectively andefficiently reduce object hallucinations during the generation process.Specifically MARINE enriches the visual context of LVLMs by integratingexisting open-source vision models and employs classifier-free guidance toincorporate the additional object grounding features to improve the precisionof LVLMs generations. Through comprehensive evaluations across 6 popularLVLMs with diverse evaluation metrics we demonstrate the effectiveness ofMARINE which even outperforms existing fine-tuning-based methods. Remarkablyit not only reduces hallucinations but also improves the detailedness of LVLMsgenerations as assessed by GPT-4V. |


| Item |Content|
| --- |---|
|idx| 2402.08679v1 |
|title| COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability |
|authors| Xingang GuoFangxu YuHuan ZhangLianhui QinBin Hu
|links| http://arxiv.org/abs/2402.08679v1 |
|updated| 2024-02-13 18:58:48 UTC |
|summary| Jailbreaks on Large language models LLMs have recently received increasingattention. For a comprehensive assessment of LLM safety it is essential toconsider jailbreaks with diverse attributes such as contextual coherence andsentiment/stylistic variations and hence it is beneficial to studycontrollable jailbreaking i.e. how to enforce control on LLM attacks. In thispaper we formally formulate the controllable attack generation problem andbuild a novel connection between this problem and controllable text generationa well-explored topic of natural language processing. Based on this connectionwe adapt the Energy-based Constrained Decoding with Langevin Dynamics COLD astate-of-the-art highly efficient algorithm in controllable text generationand introduce the COLD-Attack framework which unifies and automates the searchof adversarial LLM attacks under a variety of control requirements such asfluency stealthiness sentiment and left-right-coherence. The controllabilityenabled by COLD-Attack leads to diverse new jailbreak scenarios which not onlycover the standard setting of generating fluent suffix attacks but also allowus to address new controllable attack settings such as revising a user queryadversarially with minimal paraphrasing and inserting stealthy attacks incontext with left-right-coherence. Our extensive experiments on various LLMsLlama-2 Mistral Vicuna Guanaco GPT-3.5 show COLD-Attacks broadapplicability strong controllability high success rate and attacktransferability. Our code is available athttps://github.com/Yu-Fangxu/COLD-Attack. |


| Item |Content|
| --- |---|
|idx| 2402.08678v1 |
|title| Graph Mamba: Towards Learning on Graphs with State Space Models |
|authors| Ali BehrouzFarnoosh Hashemi
|links| http://arxiv.org/abs/2402.08678v1 |
|updated| 2024-02-13 18:58:17 UTC |
|summary| Graph Neural Networks GNNs have shown promising potential in graphrepresentation learning. The majority of GNNs define a local message-passingmechanism propagating information over the graph by stacking multiple layers.These methods however are known to suffer from two major limitations:over-squashing and poor capturing of long-range dependencies. Recently GraphTransformers GTs emerged as a powerful alternative to Message-Passing NeuralNetworks MPNNs. GTs however have quadratic computational cost lackinductive biases on graph structures and rely on complex Positional/StructuralEncodings SE/PE. In this paper we show that while Transformers complexmessage-passing and SE/PE are sufficient for good performance in practiceneither is necessary. Motivated by the recent success of State Space ModelsSSMs such as Mamba we present Graph Mamba Networks GMNs a generalframework for a new class of GNNs based on selective SSMs. We discuss andcategorize the new challenges when adopting SSMs to graph-structured data andpresent four required and one optional steps to design GMNs where we choose1 Neighborhood Tokenization 2 Token Ordering 3 Architecture ofBidirectional Selective SSM Encoder 4 Local Encoding and dispensable 5 PEand SE. We further provide theoretical justification for the power of GMNs.Experiments demonstrate that despite much less computational cost GMNs attainan outstanding performance in long-range small-scale large-scale andheterophilic benchmark datasets. |


| Item |Content|
| --- |---|
|idx| 2402.08676v1 |
|title| A Convergence Analysis of Approximate Message Passing with Non-Separable Functions and Applications to Multi-Class Classification |
|authors| Burak ÇakmakYue M. LuManfred Opper
|links| http://arxiv.org/abs/2402.08676v1 |
|updated| 2024-02-13 18:56:55 UTC |
|summary| Motivated by the recent application of approximate message passing AMP tothe analysis of convex optimizations in multi-class classifications Loureiroet. al. 2021 we present a convergence analysis of AMP dynamics withnon-separable multivariate nonlinearities. As an application we present acomplete and independent analysis of the motivated convex optimizationproblem. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2402.08682v1 |
|title| IM-3D: Iterative Multiview Diffusion and Reconstruction for High-Quality 3D Generation |
|authors| Luke Melas-KyriaziIro LainaChristian RupprechtNatalia NeverovaAndrea VedaldiOran GafniFilippos Kokkinos
|links| http://arxiv.org/abs/2402.08682v1 |
|updated| 2024-02-13 18:59:51 UTC |
|summary| Most text-to-3D generators build upon off-the-shelf text-to-image modelstrained on billions of images. They use variants of Score Distillation SamplingSDS which is slow somewhat unstable and prone to artifacts. A mitigationis to fine-tune the 2D generator to be multi-view aware which can helpdistillation or can be combined with reconstruction networks to output 3Dobjects directly. In this paper we further explore the design space oftext-to-3D models. We significantly improve multi-view generation byconsidering video instead of image generators. Combined with a 3Dreconstruction algorithm which by using Gaussian splatting can optimize arobust image-based loss we directly produce high-quality 3D outputs from thegenerated views. Our new method IM-3D reduces the number of evaluations ofthe 2D generator network 10-100x resulting in a much more efficient pipelinebetter quality fewer geometric inconsistencies and higher yield of usable 3Dassets. |


| Item |Content|
| --- |---|
|idx| 2402.08680v1 |
|title| Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance |
|authors| Linxi ZhaoYihe DengWeitong ZhangQuanquan Gu
|links| http://arxiv.org/abs/2402.08680v1 |
|updated| 2024-02-13 18:59:05 UTC |
|summary| The advancement of Large Vision-Language Models LVLMs has increasinglyhighlighted the critical issue of their tendency to hallucinate non-existingobjects in the images. To address this issue previous works focused on usingspecially curated datasets or powerful LLMs e.g. GPT-3.5 to rectify theoutputs of LVLMs. However these approaches require either expensivetraining/fine-tuning or API access to advanced LLMs to correct the modelsoutput post-generation. In this paper we tackle this challenge by introducinga framework called Mitigating hallucinAtion via classifieR-Free guIdaNcEMARINE which is both training-free and API-free and can effectively andefficiently reduce object hallucinations during the generation process.Specifically MARINE enriches the visual context of LVLMs by integratingexisting open-source vision models and employs classifier-free guidance toincorporate the additional object grounding features to improve the precisionof LVLMs generations. Through comprehensive evaluations across 6 popularLVLMs with diverse evaluation metrics we demonstrate the effectiveness ofMARINE which even outperforms existing fine-tuning-based methods. Remarkablyit not only reduces hallucinations but also improves the detailedness of LVLMsgenerations as assessed by GPT-4V. |


| Item |Content|
| --- |---|
|idx| 2402.08671v1 |
|title| Are Semi-Dense Detector-Free Methods Good at Matching Local Features? |
|authors| Matthieu VilainRémi GiraudHugo GermainGuillaume Bourmaud
|links| http://arxiv.org/abs/2402.08671v1 |
|updated| 2024-02-13 18:53:13 UTC |
|summary| Semi-dense detector-free approaches SDF such as LoFTR are currently amongthe most popular image matching methods. While SDF methods are trained toestablish correspondences between two images their performances are almostexclusively evaluated using relative pose estimation metrics. Thus the linkbetween their ability to establish correspondences and the quality of theresulting estimated pose has thus far received little attention. This paper isa first attempt to study this link. We start with proposing a novel structuredattention-based image matching architecture SAM. It allows us to show acounter-intuitive result on two datasets MegaDepth and HPatches: on the onehand SAM either outperforms or is on par with SDF methods in terms ofpose/homography estimation metrics but on the other hand SDF approaches aresignificantly better than SAM in terms of matching accuracy. We then propose tolimit the computation of the matching accuracy to textured regions and showthat in this case SAM often surpasses SDF methods. Our findings highlight astrong correlation between the ability to establish accurate correspondences intextured regions and the accuracy of the resulting estimated pose/homography.Our code will be made available. |


| Item |Content|
| --- |---|
|idx| 2402.08657v1 |
|title| PIN: Positional Insert Unlocks Object Localisation Abilities in VLMs |
|authors| Michael DorkenwaldNimrod BarazaniCees G. M. SnoekYuki M. Asano
|links| http://arxiv.org/abs/2402.08657v1 |
|updated| 2024-02-13 18:39:18 UTC |
|summary| Vision-Language Models VLMs such as Flamingo and GPT-4V have shownimmense potential by integrating large language models with vision systems.Nevertheless these models face challenges in the fundamental computer visiontask of object localisation due to their training on multimodal datacontaining mostly captions without explicit spatial grounding. While it ispossible to construct custom supervised training pipelines with bounding boxannotations that integrate with VLMs these result in specialized andhard-to-scale models. In this paper we aim to explore the limits ofcaption-based VLMs and instead propose to tackle the challenge in a simplermanner by i keeping the weights of a caption-based VLM frozen and ii notusing any supervised detection data. To this end we introduce aninput-agnostic Positional Insert PIN a learnable spatial prompt containinga minimal set of parameters that are slid inside the frozen VLM unlockingobject localisation capabilities. Our PIN module is trained with a simplenext-token prediction task on synthetic data without requiring the introductionof new output heads. Our experiments demonstrate strong zero-shot localisationperformances on a variety of images including Pascal VOC COCO LVIS anddiverse images like paintings or cartoons. |


| Item |Content|
| --- |---|
|idx| 2402.08654v1 |
|title| Learning Continuous 3D Words for Text-to-Image Generation |
|authors| Ta-Ying ChengMatheus GadelhaThibault GroueixMatthew FisherRadomir MechAndrew MarkhamNiki Trigoni
|links| http://arxiv.org/abs/2402.08654v1 |
|updated| 2024-02-13 18:34:10 UTC |
|summary| Current controls over diffusion models e.g. through text or ControlNet forimage generation fall short in recognizing abstract continuous attributes likeillumination direction or non-rigid shape change. In this paper we present anapproach for allowing users of text-to-image models to have fine-grainedcontrol of several attributes in an image. We do this by engineering specialsets of input tokens that can be transformed in a continuous manner -- we callthem Continuous 3D Words. These attributes can for example be represented assliders and applied jointly with text prompts for fine-grained control overimage generation. Given only a single mesh and a rendering engine we show thatour approach can be adopted to provide continuous user control over several3D-aware attributes including time-of-day illumination bird wing orientationdollyzoom effect and object poses. Our method is capable of conditioning imagecreation with multiple Continuous 3D Words and text descriptions simultaneouslywhile adding no overhead to the generative process. Project Page:https://ttchengab.github.io/continuous_3d_words |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2402.08667v1 |
|title| Target Score Matching |
|authors| Valentin De BortoliMichael HutchinsonPeter WirnsbergerArnaud Doucet
|links| http://arxiv.org/abs/2402.08667v1 |
|updated| 2024-02-13 18:48:28 UTC |
|summary| Denoising Score Matching estimates the score of a noised version of a targetdistribution by minimizing a regression loss and is widely used to train thepopular class of Denoising Diffusion Models. A well known limitation ofDenoising Score Matching however is that it yields poor estimates of thescore at low noise levels. This issue is particularly unfavourable for problemsin the physical sciences and for Monte Carlo sampling tasks for which the scoreof the clean original target is known. Intuitively estimating the score of aslightly noised version of the target should be a simple task in such cases. Inthis paper we address this shortcoming and show that it is indeed possible toleverage knowledge of the target score. We present a Target Score Identity andcorresponding Target Score Matching regression loss which allows us to obtainscore estimates admitting favourable properties at low noise levels. |


| Item |Content|
| --- |---|
|idx| 2402.08621v1 |
|title| A Generalized Approach to Online Convex Optimization |
|authors| Mohammad PedramfarVaneet Aggarwal
|links| http://arxiv.org/abs/2402.08621v1 |
|updated| 2024-02-13 17:42:27 UTC |
|summary| In this paper we analyze the problem of online convex optimization indifferent settings. We show that any algorithm for online linear optimizationwith fully adaptive adversaries is an algorithm for online convex optimization.We also show that any such algorithm that requires full-information feedbackmay be transformed to an algorithm with semi-bandit feedback with comparableregret bound. We further show that algorithms that are designed for fullyadaptive adversaries using deterministic semi-bandit feedback can obtainsimilar bounds using only stochastic semi-bandit feedback when facing obliviousadversaries. We use this to describe general meta-algorithms to convert firstorder algorithms to zeroth order algorithms with comparable regret bounds. Ourframework allows us to analyze online optimization in various settings suchfull-information feedback bandit feedback stochastic regret adversarialregret and various forms of non-stationary regret. Using our analysis weprovide the first efficient projection-free online convex optimizationalgorithm using linear optimization oracles. |


| Item |Content|
| --- |---|
|idx| 2402.08616v1 |
|title| Adjustment Identification Distance: A gadjid for Causal Structure Learning |
|authors| Leonard HenckelTheo WürtzenSebastian Weichwald
|links| http://arxiv.org/abs/2402.08616v1 |
|updated| 2024-02-13 17:32:59 UTC |
|summary| Evaluating graphs learned by causal discovery algorithms is difficult: Thenumber of edges that differ between two graphs does not reflect how the graphsdiffer with respect to the identifying formulas they suggest for causaleffects. We introduce a framework for developing causal distances betweengraphs which includes the structural intervention distance for directed acyclicgraphs as a special case. We use this framework to develop improvedadjustment-based distances as well as extensions to completed partiallydirected acyclic graphs and causal orders. We develop polynomial-timereachability algorithms to compute the distances efficiently. In our packagegadjid open source at https://github.com/CausalDisco/gadjid we provideimplementations of our distances they are orders of magnitude faster than thestructural intervention distance and thereby provide a success metric forcausal discovery that scales to graph sizes that were previously prohibitive. |


| Item |Content|
| --- |---|
|idx| 2402.08602v1 |
|title| Globally-Optimal Greedy Experiment Selection for Active Sequential Estimation |
|authors| Xiaoou LiHongru Zhao
|links| http://arxiv.org/abs/2402.08602v1 |
|updated| 2024-02-13 17:09:29 UTC |
|summary| Motivated by modern applications such as computerized adaptive testingsequential rank aggregation and heterogeneous data source selection we studythe problem of active sequential estimation which involves adaptivelyselecting experiments for sequentially collected data. The goal is to designexperiment selection rules for more accurate model estimation. Greedyinformation-based experiment selection methods optimizing the information gainfor one-step ahead have been employed in practice thanks to theircomputational convenience flexibility to context or task changes and broadapplicability. However statistical analysis is restricted to one-dimensionalcases due to the problems combinatorial nature and the seemingly limitedcapacity of greedy algorithms leaving the multidimensional problem open.  In this study we close the gap for multidimensional problems. In particularwe propose adopting a class of greedy experiment selection methods and providestatistical analysis for the maximum likelihood estimator following theseselection rules. This class encompasses both existing methods and introducesnew methods with improved numerical efficiency. We prove that these methodsproduce consistent and asymptotically normal estimators. Additionally within adecision theory framework we establish that the proposed methods achieveasymptotic optimality when the risk measure aligns with the selection rule. Wealso conduct extensive numerical studies on both simulated and real data toillustrate the efficacy of the proposed methods.  From a technical perspective we devise new analytical tools to addresstheoretical challenges. These analytical tools are of independent theoreticalinterest and may be reused in related problems involving stochasticapproximation and sequential designs. |


| Item |Content|
| --- |---|
|idx| 2402.08543v2 |
|title| Theoretical Analysis of Leave-one-out Cross Validation for Non-differentiable Penalties under High-dimensional Settings |
|authors| Haolin ZouArnab AuddyKamiar Rahnama RadArian Maleki
|links| http://arxiv.org/abs/2402.08543v2 |
|updated| 2024-02-14 16:28:59 UTC |
|summary| Despite a large and significant body of recent work focused on estimating theout-of-sample risk of regularized models in the high dimensional regime atheoretical understanding of this problem for non-differentiable penalties suchas generalized LASSO and nuclear norm is missing. In this paper we resolve thischallenge. We study this problem in the proportional high dimensional regimewhere both the sample size n and number of features p are large and n/p andthe signal-to-noise ratio per observation remain finite. We provide finitesample upper bounds on the expected squared error of leave-one-outcross-validation LO in estimating the out-of-sample risk. The theoreticalframework presented here provides a solid foundation for elucidating empiricalfindings that show the accuracy of LO. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2402.08658v1 |
|title| The Last JITAI? The Unreasonable Effectiveness of Large Language Models in Issuing Just-in-Time Adaptive Interventions: Fostering Physical Activity in a Prospective Cardiac Rehabilitation Setting |
|authors| David HaagDevender KumarSebastian GruberMahdi SarebanGunnar TreffJosef NiebauerChristopher BullJan David Smeddinck
|links| http://arxiv.org/abs/2402.08658v1 |
|updated| 2024-02-13 18:39:36 UTC |
|summary| We explored the viability of Large Language Models LLMs for triggering andpersonalizing content for Just-in-Time Adaptive Interventions JITAIs indigital health. JITAIs are being explored as a key mechanism for sustainablebehavior change adapting interventions to an individuals current context andneeds. However traditional rule-based and machine learning models for JITAIimplementation face scalability and reliability limitations such as lack ofpersonalization difficulty in managing multi-parametric systems and issueswith data sparsity. To investigate JITAI implementation via LLMs we tested thecontemporary overall performance-leading model GPT-4 with examples groundedin the use case of fostering heart-healthy physical activity in outpatientcardiac rehabilitation. Three personas and five sets of context information perpersona were used as a basis of triggering and personalizing JITAIs.Subsequently we generated a total of 450 proposed JITAI decisions and messagecontent divided equally into JITAIs generated by 10 iterations with GPT-4 abaseline provided by 10 laypersons LayPs and a gold standard set by 10healthcare professionals HCPs. Ratings from 27 LayPs indicated that JITAIsgenerated by GPT-4 were superior to those by HCPs and LayPs over all assessedscales: i.e. appropriateness engagement effectiveness and professionality.This study indicates that LLMs have significant potential for implementingJITAIs as a building block of personalized or precision health offeringscalability effective personalization based on opportunistically sampledinformation and good acceptability. |


| Item |Content|
| --- |---|
|idx| 2402.08655v1 |
|title| Assessing the Privacy Risk of Cross-Platform Identity Linkage using Eye Movement Biometrics |
|authors| Samantha AzizOleg Komogortsev
|links| http://arxiv.org/abs/2402.08655v1 |
|updated| 2024-02-13 18:37:23 UTC |
|summary| The recent emergence of ubiquitous multi-platform eye tracking has raiseduser privacy concerns over re-identification across platforms where a personis re-identified across multiple eye tracking-enabled platforms usingpersonally identifying information that is implicitly expressed through theireye movement. We present an empirical investigation quantifying a modern eyemovement biometric models ability to link subject identities across threedifferent eye tracking devices using eye movement signals from each device. Weshow that a state-of-the art eye movement biometrics model demonstratesabove-chance levels of biometric performance 34.99 equal error rate 15rank-1 identification rate when linking user identities across one pair ofdevices but not for the other. Considering these findings we also discuss theimpact that eye tracking signal quality has on the models ability tomeaningfully associate a subjects identity between two substantially differenteye tracking devices. Our investigation advances a fundamental understanding ofthe privacy risks for identity linkage across platforms by employing bothquantitative and qualitative measures of biometric performance including avisualization of the models ability to distinguish genuine and imposterauthentication attempts across platforms. |


| Item |Content|
| --- |---|
|idx| 2402.08565v1 |
|title| Artificial Intelligence for Literature Reviews: Opportunities and Challenges |
|authors| Francisco BolanosAngelo SalatinoFrancesco OsborneEnrico Motta
|links| http://arxiv.org/abs/2402.08565v1 |
|updated| 2024-02-13 16:05:51 UTC |
|summary| This manuscript presents a comprehensive review of the use of ArtificialIntelligence AI in Systematic Literature Reviews SLRs. A SLR is a rigorousand organised methodology that assesses and integrates previous research on agiven topic. Numerous tools have been developed to assist and partiallyautomate the SLR process. The increasing role of AI in this field shows greatpotential in providing more effective support for researchers moving towardsthe semi-automatic creation of literature reviews. Our study focuses on how AItechniques are applied in the semi-automation of SLRs specifically in thescreening and extraction phases. We examine 21 leading SLR tools using aframework that combines 23 traditional features with 11 AI features. We alsoanalyse 11 recent tools that leverage large language models for searching theliterature and assisting academic writing. Finally the paper discusses currenttrends in the field outlines key research challenges and suggests directionsfor future research. |


| Item |Content|
| --- |---|
|idx| 2402.08558v1 |
|title| Exploring diversity perceptions in a community through a Q&A chatbot |
|authors| Peter KunAmalia De GötzenMiriam BidogliaNiels Jørgen GommesenGeorge Gaskell
|links| http://dx.doi.org/10.21606/drs.2022.807 |
|updated| 2024-02-13 15:59:19 UTC |
|summary| While diversity has become a debated issue in design very little researchexists on positive use-cases for diversity beyond scholarly criticism. Thecurrent work addresses this gap through the case of a diversity-aware chatbotexploring what benefits a diversity-aware chatbot could bring to people and howdo people interpret diversity when being presented with it. In this paper wemotivate a QA chatbot as a technology probe and deploy it in two studentcommunities within a study. During the study we collected contextual data onpeoples expectations and perceptions when presented with diversity during thestudy. Our key findings show that people seek out others with shared nicheinterests or their search is driven by exploration and inspiration whenpresented with diversity. Although interacting with chatbots is limitedparticipants found the engagement novel and interesting to motivate futureresearch. |


| Item |Content|
| --- |---|
|idx| 2402.08451v1 |
|title| Moonwalk: Advancing Gait-Based User Recognition on Wearable Devices with Metric Learning |
|authors| Asaf LibermanOron LevySoroush ShahiCori Tymoszek ParkMike RalphRichard KangAbdelkareem BedriGierad Laput
|links| http://arxiv.org/abs/2402.08451v1 |
|updated| 2024-02-13 13:38:06 UTC |
|summary| Personal devices have adopted diverse authentication methods includingbiometric recognition and passcodes. In contrast headphones have limited inputmechanisms depending solely on the authentication of connected devices. Wepresent Moonwalk a novel method for passive user recognition utilizing thebuilt-in headphone accelerometer. Our approach centers on gait recognitionenabling users to establish their identity simply by walking for a briefinterval despite the sensors placement away from the feet. We employself-supervised metric learning to train a model that yields a highlydiscriminative representation of a users 3D acceleration with no retrainingrequired. We tested our method in a study involving 50 participants achievingan average F1 score of 92.9 and equal error rate of 2.3. We extend ourevaluation by assessing performance under various conditions e.g. shoe typesand surfaces. We discuss the opportunities and challenges these variationsintroduce and propose new directions for advancing passive authentication forwearable devices. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2402.08567v1 |
|title| Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast |
|authors| Xiangming GuXiaosen ZhengTianyu PangChao DuQian LiuYe WangJing JiangMin Lin
|links| http://arxiv.org/abs/2402.08567v1 |
|updated| 2024-02-13 16:06:17 UTC |
|summary| A multimodal large language model MLLM agent can receive instructionscapture images retrieve histories from memory and decide which tools to use.Nonetheless red-teaming efforts have revealed that adversarial images/promptscan jailbreak an MLLM and cause unaligned behaviors. In this work we report aneven more severe safety issue in multi-agent environments referred to asinfectious jailbreak. It entails the adversary simply jailbreaking a singleagent and without any further intervention from the adversary almost allagents will become infected exponentially fast and exhibit harmful behaviors.To validate the feasibility of infectious jailbreak we simulate multi-agentenvironments containing up to one million LLaVA-1.5 agents and employrandomized pair-wise chat as a proof-of-concept instantiation for multi-agentinteraction. Our results show that feeding an infectious adversarial imageinto the memory of any randomly chosen agent is sufficient to achieveinfectious jailbreak. Finally we derive a simple principle for determiningwhether a defense mechanism can provably restrain the spread of infectiousjailbreak but how to design a practical defense that meets this principleremains an open question to investigate. Our project page is available athttps://sail-sg.github.io/Agent-Smith/. |


| Item |Content|
| --- |---|
|idx| 2402.08421v1 |
|title| Conservative and Risk-Aware Offline Multi-Agent Reinforcement Learning for Digital Twins |
|authors| Eslam EldeebHoussem SifaouOsvaldo SimeoneMohammad ShehabHirley Alves
|links| http://arxiv.org/abs/2402.08421v1 |
|updated| 2024-02-13 12:49:22 UTC |
|summary| Digital twin DT platforms are increasingly regarded as a promisingtechnology for controlling optimizing and monitoring complex engineeringsystems such as next-generation wireless networks. An important challenge inadopting DT solutions is their reliance on data collected offline lackingdirect access to the physical environment. This limitation is particularlysevere in multi-agent systems for which conventional multi-agent reinforcementMARL requires online interactions with the environment. A direct applicationof online MARL schemes to an offline setting would generally fail due to theepistemic uncertainty entailed by the limited availability of data. In thiswork we propose an offline MARL scheme for DT-based wireless networks thatintegrates distributional RL and conservative Q-learning to address theenvironments inherent aleatoric uncertainty and the epistemic uncertaintyarising from limited data. To further exploit the offline data we adapt theproposed scheme to the centralized training decentralized execution frameworkallowing joint training of the agents policies. The proposed MARL schemereferred to as multi-agent conservative quantile regression MA-CQR addressesgeneral risk-sensitive design criteria and is applied to the trajectoryplanning problem in drone networks showcasing its advantages. |


| Item |Content|
| --- |---|
|idx| 2402.08282v1 |
|title| Logic of Awareness for Nested Knowledge |
|authors| Yudai Kubono
|links| http://arxiv.org/abs/2402.08282v1 |
|updated| 2024-02-13 08:18:14 UTC |
|summary| Reasoning abilities of human beings are limited. Logics that treat logicalinference for human knowledge should reflect these limited abilities. Logic ofawareness is one of those logics. In the logic what an agent with a limitedreasoning ability actually knows at a given moment explicit knowledge isdistinguished from the ideal knowledge that an agent obtains by performing allpossible inferences with what she already knows implicit knowledge. Thispaper proposes a logic for explicit knowledge. In particular we focus more onnested explicit knowledge which means another agents knowledge that an agentactually knows at a given moment. We develope a new formalization of two ideasand propose Kripke-style semantics. The first idea is the effect on an agentsreasoning ability by a state of an agents awareness. We incorporate a relationon possible worlds called an indistinguishable relation to represent ignorancedue to lack of awareness. The second idea is a state of each agents awarenessin the other agents mind. We incorporate a non-empty finite sequence of agentscalled textita chain of belief for awareness. Our logic is called AwarenessLogic with Partitions and Chains ALPC. Employing an example we show hownested explicit knowledge is formalized with our logic. Thereafter we proposethe proof system and prove the completeness. Finally we discuss directions forextending and applying our logic and conclude. Our logic offers a foundationfor a formal representation of human knowledge. We expect that the logic can beapplied to computer science and game theory by describing and analyzingstrategic behavior in a game and practical agent communication. |


| Item |Content|
| --- |---|
|idx| 2402.08156v1 |
|title| Group Decision-Making among Privacy-Aware Agents |
|authors| Marios PapachristouM. Amin Rahimian
|links| http://arxiv.org/abs/2402.08156v1 |
|updated| 2024-02-13 01:38:01 UTC |
|summary| How can individuals exchange information to learn from each other despitetheir privacy needs and security concerns For example consider individualsdeliberating a contentious topic and being concerned about divulging theirprivate experiences. Preserving individual privacy and enabling efficientsocial learning are both important desiderata but seem fundamentally at oddswith each other and very hard to reconcile. We do so by controlling informationleakage using rigorous statistical guarantees that are based on differentialprivacy DP. Our agents use log-linear rules to update their beliefs aftercommunicating with their neighbors. Adding DP randomization noise to beliefsprovides communicating agents with plausible deniability with regard to theirprivate information and their network neighborhoods. We consider two learningenvironments one for distributed maximum-likelihood estimation given a finitenumber of private signals and another for online learning from an infiniteintermittent signal stream. Noisy information aggregation in the finite caseleads to interesting tradeoffs between rejecting low-quality states and makingsure all high-quality states are accepted in the algorithm output. Our resultsflesh out the nature of the trade-offs in both cases between the quality of thegroup decision outcomes learning accuracy communication cost and the levelof privacy protections that the agents are afforded. |


| Item |Content|
| --- |---|
|idx| 2402.07752v1 |
|title| Mixed Q-Functionals: Advancing Value-Based Methods in Cooperative MARL with Continuous Action Domains |
|authors| Yasin FindikS. Reza Ahmadzadeh
|links| http://arxiv.org/abs/2402.07752v1 |
|updated| 2024-02-12 16:21:50 UTC |
|summary| Tackling multi-agent learning problems efficiently is a challenging task incontinuous action domains. While value-based algorithms excel in sampleefficiency when applied to discrete action domains they are usuallyinefficient when dealing with continuous actions. Policy-based algorithms onthe other hand attempt to address this challenge by leveraging critic networksfor guiding the learning process and stabilizing the gradient estimation. Thelimitations in the estimation of true return and falling into local optima inthese methods result in inefficient and often sub-optimal policies. In thispaper we diverge from the trend of further enhancing critic networks andfocus on improving the effectiveness of value-based methods in multi-agentcontinuous domains by concurrently evaluating numerous actions. We propose anovel multi-agent value-based algorithm Mixed Q-Functionals MQF inspiredfrom the idea of Q-Functionals that enables agents to transform their statesinto basis functions. Our algorithm fosters collaboration among agents bymixing their action-values. We evaluate the efficacy of our algorithm in sixcooperative multi-agent scenarios. Our empirical findings reveal that MQFoutperforms four variants of Deep Deterministic Policy Gradient through rapidaction evaluation and increased sample efficiency. |


