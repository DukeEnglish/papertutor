# cs.CL 

| Item |Content|
| --- |---|
|idx| 2404.02138v1 |
|title| Topic-based Watermarks for LLM-Generated Text |
|authors| Alexander NemecekYuzhou JiangErman Ayday
|links| http://arxiv.org/abs/2404.02138v1 |
|updated| 2024-04-02 17:49:40 UTC |
|summary| Recent advancements of large language models LLMs have resulted inindistinguishable text outputs comparable to human-generated text. Watermarkingalgorithms are potential tools that offer a way to differentiate between LLM-and human-generated text by embedding detectable signatures withinLLM-generated output. However current watermarking schemes lack robustnessagainst known attacks against watermarking algorithms. In addition they areimpractical considering an LLM generates tens of thousands of text outputs perday and the watermarking algorithm needs to memorize each output it generatesfor the detection to work. In this work focusing on the limitations of currentwatermarking schemes we propose the concept of a topic-based watermarkingalgorithm for LLMs. The proposed algorithm determines how to generate tokensfor the watermarked LLM output based on extracted topics of an input prompt orthe output of a non-watermarked LLM. Inspired from previous work we proposeusing a pair of lists that are generated based on the specified extractedtopics that specify certain tokens to be included or excluded whilegenerating the watermarked output of the LLM. Using the proposed watermarkingalgorithm we show the practicality of a watermark detection algorithm.Furthermore we discuss a wide range of attacks that can emerge againstwatermarking algorithms for LLMs and the benefit of the proposed watermarkingscheme for the feasibility of modeling a potential attacker considering itsbenefit vs. loss. |


| Item |Content|
| --- |---|
|idx| 2404.02127v1 |
|title| FLawN-T5: An Empirical Examination of Effective Instruction-Tuning Data Mixtures for Legal Reasoning |
|authors| Joel NiklausLucia ZhengArya D. McCarthyChristopher HahnBrian M. RosenPeter HendersonDaniel E. HoGarrett HonkePercy LiangChristopher Manning
|links| http://arxiv.org/abs/2404.02127v1 |
|updated| 2024-04-02 17:33:34 UTC |
|summary| Instruction tuning is an important step in making language models useful fordirect user interaction. However many legal tasks remain out of reach for mostopen LLMs and there do not yet exist any large scale instruction datasets forthe domain. This critically limits research in this application area. In thiswork we curate LawInstruct a large legal instruction dataset covering 17jurisdictions 24 languages and a total of 12M examples. We present evidencethat domain-specific pretraining and instruction tuning improve performance onLegalBench including improving Flan-T5 XL by 8 points or 16 over thebaseline. However the effect does not generalize across all tasks trainingregimes model sizes and other factors. LawInstruct is a resource foraccelerating the development of models with stronger information processing anddecision making capabilities in the legal domain. |


| Item |Content|
| --- |---|
|idx| 2404.02126v1 |
|title| Rematch: Robust and Efficient Matching of Local Knowledge Graphs to Improve Structural and Semantic Similarity |
|authors| Zoher KachwalaJisun AnHaewoon KwakFilippo Menczer
|links| http://arxiv.org/abs/2404.02126v1 |
|updated| 2024-04-02 17:33:00 UTC |
|summary| Knowledge graphs play a pivotal role in various applications such asquestion-answering and fact-checking. Abstract Meaning Representation AMRrepresents text as knowledge graphs. Evaluating the quality of these graphsinvolves matching them structurally to each other and semantically to thesource text. Existing AMR metrics are inefficient and struggle to capturesemantic similarity. We also lack a systematic evaluation benchmark forassessing structural similarity between AMR graphs. To overcome theselimitations we introduce a novel AMR similarity metric rematch alongside anew evaluation for structural similarity called RARE. Among state-of-the-artmetrics rematch ranks second in structural similarity and first in semanticsimilarity by 1--5 percentage points on the STS-B and SICK-R benchmarks.Rematch is also five times faster than the next most efficient metric. |


| Item |Content|
| --- |---|
|idx| 2404.02124v1 |
|title| Exploring Automated Distractor Generation for Math Multiple-choice Questions via Large Language Models |
|authors| Wanyong FengJaewook LeeHunter McNicholsAlexander ScarlatosDigory SmithSimon WoodheadNancy Otero OrnelasAndrew Lan
|links| http://arxiv.org/abs/2404.02124v1 |
|updated| 2024-04-02 17:31:58 UTC |
|summary| Multiple-choice questions MCQs are ubiquitous in almost all levels ofeducation since they are easy to administer grade and are a reliable formatin assessments and practices. One of the most important aspects of MCQs is thedistractors i.e. incorrect options that are designed to target common errorsor misconceptions among real students. To date the task of craftinghigh-quality distractors largely remains a labor and time-intensive process forteachers and learning content designers which has limited scalability. In thiswork we study the task of automated distractor generation in the domain ofmath MCQs and explore a wide variety of large language model LLM-basedapproaches from in-context learning to fine-tuning. We conduct extensiveexperiments using a real-world math MCQ dataset and find that although LLMs cangenerate some mathematically valid distractors they are less adept atanticipating common errors or misconceptions among real students. |


| Item |Content|
| --- |---|
|idx| 2404.02115v1 |
|title| GINopic: Topic Modeling with Graph Isomorphism Network |
|authors| Suman AdhyaDebarshi Kumar Sanyal
|links| http://arxiv.org/abs/2404.02115v1 |
|updated| 2024-04-02 17:18:48 UTC |
|summary| Topic modeling is a widely used approach for analyzing and exploring largedocument collections. Recent research efforts have incorporated pre-trainedcontextualized language models such as BERT embeddings into topic modeling.However they often neglect the intrinsic informational value conveyed bymutual dependencies between words. In this study we introduce GINopic a topicmodeling framework based on graph isomorphism networks to capture thecorrelation between words. By conducting intrinsic quantitative as well asqualitative and extrinsic evaluations on diverse benchmark datasets wedemonstrate the effectiveness of GINopic compared to existing topic models andhighlight its potential for advancing topic modeling. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2404.02157v1 |
|title| Segment Any 3D Object with Language |
|authors| Seungjun LeeYuyang ZhaoGim Hee Lee
|links| http://arxiv.org/abs/2404.02157v1 |
|updated| 2024-04-02 17:59:10 UTC |
|summary| In this paper we investigate Open-Vocabulary 3D Instance SegmentationOV-3DIS with free-form language instructions. Earlier works that rely on onlyannotated base categories for training suffer from limited generalization tounseen novel categories. Recent works mitigate poor generalizability to novelcategories by generating class-agnostic masks or projecting generalized masksfrom 2D to 3D but disregard semantic or geometry information leading tosub-optimal performance. Instead generating generalizable but semantic-relatedmasks directly from 3D point clouds would result in superior outcomes. In thispaper we introduce Segment any 3D Object with LanguagE SOLE which is asemantic and geometric-aware visual-language learning framework with stronggeneralizability by generating semantic-related masks directly from 3D pointclouds. Specifically we propose a multimodal fusion network to incorporatemultimodal semantics in both backbone and decoder. In addition to align the 3Dsegmentation model with various language instructions and enhance the maskquality we introduce three types of multimodal associations as supervision.Our SOLE outperforms previous methods by a large margin on ScanNetv2ScanNet200 and Replica benchmarks and the results are even close to thefully-supervised counterpart despite the absence of class annotations in thetraining. Furthermore extensive qualitative results demonstrate theversatility of our SOLE to language instructions. |


| Item |Content|
| --- |---|
|idx| 2404.02151v1 |
|title| Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks |
|authors| Maksym AndriushchenkoFrancesco CroceNicolas Flammarion
|links| http://arxiv.org/abs/2404.02151v1 |
|updated| 2024-04-02 17:58:27 UTC |
|summary| We show that even the most recent safety-aligned LLMs are not robust tosimple adaptive jailbreaking attacks. First we demonstrate how to successfullyleverage access to logprobs for jailbreaking: we initially design anadversarial prompt template sometimes adapted to the target LLM and then weapply random search on a suffix to maximize the target logprob e.g. of thetoken Sure potentially with multiple restarts. In this way we achievenearly 100 attack success rate -- according to GPT-4 as a judge -- onGPT-3.5/4 Llama-2-Chat-7B/13B/70B Gemma-7B and R2D2 from HarmBench that wasadversarially trained against the GCG attack. We also show how to jailbreak allClaude models -- that do not expose logprobs -- via either a transfer orprefilling attack with 100 success rate. In addition we show how to userandom search on a restricted set of tokens for finding trojan strings inpoisoned models -- a task that shares many similarities with jailbreaking --which is the algorithm that brought us the first place in the SaTML24 TrojanDetection Competition. The common theme behind these attacks is that adaptivityis crucial: different models are vulnerable to different prompting templatese.g. R2D2 is very sensitive to in-context learning prompts some models haveunique vulnerabilities based on their APIs e.g. prefilling for Claude andin some settings it is crucial to restrict the token search space based onprior knowledge e.g. for trojan detection. We provide the code prompts andlogs of the attacks at https://github.com/tml-epfl/llm-adaptive-attacks. |


| Item |Content|
| --- |---|
|idx| 2404.02127v1 |
|title| FLawN-T5: An Empirical Examination of Effective Instruction-Tuning Data Mixtures for Legal Reasoning |
|authors| Joel NiklausLucia ZhengArya D. McCarthyChristopher HahnBrian M. RosenPeter HendersonDaniel E. HoGarrett HonkePercy LiangChristopher Manning
|links| http://arxiv.org/abs/2404.02127v1 |
|updated| 2024-04-02 17:33:34 UTC |
|summary| Instruction tuning is an important step in making language models useful fordirect user interaction. However many legal tasks remain out of reach for mostopen LLMs and there do not yet exist any large scale instruction datasets forthe domain. This critically limits research in this application area. In thiswork we curate LawInstruct a large legal instruction dataset covering 17jurisdictions 24 languages and a total of 12M examples. We present evidencethat domain-specific pretraining and instruction tuning improve performance onLegalBench including improving Flan-T5 XL by 8 points or 16 over thebaseline. However the effect does not generalize across all tasks trainingregimes model sizes and other factors. LawInstruct is a resource foraccelerating the development of models with stronger information processing anddecision making capabilities in the legal domain. |


| Item |Content|
| --- |---|
|idx| 2404.02090v1 |
|title| Already Moderate Population Sizes Provably Yield Strong Robustness to Noise |
|authors| Denis AntipovBenjamin DoerrAlexandra Ivanova
|links| http://arxiv.org/abs/2404.02090v1 |
|updated| 2024-04-02 16:35:52 UTC |
|summary| Experience shows that typical evolutionary algorithms can cope well withstochastic disturbances such as noisy function evaluations.  In this first mathematical runtime analysis of the 1lambda and1lambda evolutionary algorithms in the presence of prior bit-wise noisewe show that both algorithms can tolerate constant noise probabilities withoutincreasing the asymptotic runtime on the OneMax benchmark. For this apopulation size lambda suffices that is at least logarithmic in the problemsize n. The only previous result in this direction regarded the lessrealistic one-bit noise model required a population size super-linear in theproblem size and proved a runtime guarantee roughly cubic in the noiselessruntime for the OneMax benchmark. Our significantly stronger results are basedon the novel proof argument that the noiseless offspring can be seen as abiased uniform crossover between the parent and the noisy offspring. We areoptimistic that the technical lemmas resulting from this insight will findapplications also in future mathematical runtime analyses of evolutionaryalgorithms. |


| Item |Content|
| --- |---|
|idx| 2404.02078v1 |
|title| Advancing LLM Reasoning Generalists with Preference Trees |
|authors| Lifan YuanGanqu CuiHanbin WangNing DingXingyao WangJia DengBoji ShanHuimin ChenRuobing XieYankai LinZhenghao LiuBowen ZhouHao PengZhiyuan LiuMaosong Sun
|links| http://arxiv.org/abs/2404.02078v1 |
|updated| 2024-04-02 16:25:30 UTC |
|summary| We introduce Eurus a suite of large language models LLMs optimized forreasoning. Finetuned from Mistral-7B and CodeLlama-70B Eurus models achievestate-of-the-art results among open-source models on a diverse set ofbenchmarks covering mathematics code generation and logical reasoningproblems. Notably Eurus-70B beats GPT-3.5 Turbo in reasoning through acomprehensive benchmarking across 12 tests covering five tasks and achieves a33.3 pass1 accuracy on LeetCode and 32.6 on TheoremQA two challengingbenchmarks substantially outperforming existing open-source models by marginsmore than 13.3. The strong performance of Eurus can be primarily attributed toUltraInteract our newly-curated large-scale high-quality alignment datasetspecifically designed for complex reasoning tasks. UltraInteract can be used inboth supervised fine-tuning and preference learning. For each instruction itincludes a preference tree consisting of 1 reasoning chains with diverseplanning strategies in a unified format 2 multi-turn interactiontrajectories with the environment and the critique and 3 pairwise data tofacilitate preference learning. UltraInteract allows us to conduct an in-depthexploration of preference learning for reasoning tasks. Our investigationreveals that some well-established preference learning algorithms may be lesssuitable for reasoning tasks compared to their effectiveness in generalconversations. Inspired by this we derive a novel reward modeling objectivewhich together with UltraInteract leads to a strong reward model. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2404.02151v1 |
|title| Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks |
|authors| Maksym AndriushchenkoFrancesco CroceNicolas Flammarion
|links| http://arxiv.org/abs/2404.02151v1 |
|updated| 2024-04-02 17:58:27 UTC |
|summary| We show that even the most recent safety-aligned LLMs are not robust tosimple adaptive jailbreaking attacks. First we demonstrate how to successfullyleverage access to logprobs for jailbreaking: we initially design anadversarial prompt template sometimes adapted to the target LLM and then weapply random search on a suffix to maximize the target logprob e.g. of thetoken Sure potentially with multiple restarts. In this way we achievenearly 100 attack success rate -- according to GPT-4 as a judge -- onGPT-3.5/4 Llama-2-Chat-7B/13B/70B Gemma-7B and R2D2 from HarmBench that wasadversarially trained against the GCG attack. We also show how to jailbreak allClaude models -- that do not expose logprobs -- via either a transfer orprefilling attack with 100 success rate. In addition we show how to userandom search on a restricted set of tokens for finding trojan strings inpoisoned models -- a task that shares many similarities with jailbreaking --which is the algorithm that brought us the first place in the SaTML24 TrojanDetection Competition. The common theme behind these attacks is that adaptivityis crucial: different models are vulnerable to different prompting templatese.g. R2D2 is very sensitive to in-context learning prompts some models haveunique vulnerabilities based on their APIs e.g. prefilling for Claude andin some settings it is crucial to restrict the token search space based onprior knowledge e.g. for trojan detection. We provide the code prompts andlogs of the attacks at https://github.com/tml-epfl/llm-adaptive-attacks. |


| Item |Content|
| --- |---|
|idx| 2404.02141v1 |
|title| Robustly estimating heterogeneity in factorial data using Rashomon Partitions |
|authors| Aparajithan VenkateswaranAnirudh SankarArun G. ChandrasekharTyler H. McCormick
|links| http://arxiv.org/abs/2404.02141v1 |
|updated| 2024-04-02 17:53:28 UTC |
|summary| Many statistical analyses in both observational data and randomized controltrials ask: how does the outcome of interest vary with combinations ofobservable covariates How do various drug combinations affect health outcomesor how does technology adoption depend on incentives and demographics Our goalis to partition this factorial space into pools of covariate combinationswhere the outcome differs across the pools but not within a pool. Existingapproaches i search for a single optimal partition under assumptionsabout the association between covariates or ii sample from the entire set ofpossible partitions. Both these approaches ignore the reality that especiallywith correlation structure in covariates many ways to partition the covariatespace may be statistically indistinguishable despite very differentimplications for policy or science. We develop an alternative perspectivecalled Rashomon Partition Sets RPSs. Each item in the RPS partitions thespace of covariates using a tree-like geometry. RPSs incorporate all partitionsthat have posterior values near the maximum a posteriori partition even ifthey offer substantively different explanations and do so using a prior thatmakes no assumptions about associations between covariates. This prior is theell_0 prior which we show is minimax optimal. Given the RPS we calculatethe posterior of any measurable function of the feature effects vector onoutcomes conditional on being in the RPS. We also characterize approximationerror relative to the entire posterior and provide bounds on the size of theRPS. Simulations demonstrate this framework allows for robust conclusionsrelative to conventional regularization techniques. We apply our method tothree empirical settings: price effects on charitable giving chromosomalstructure telomere length and the introduction of microfinance. |


| Item |Content|
| --- |---|
|idx| 2404.02138v1 |
|title| Topic-based Watermarks for LLM-Generated Text |
|authors| Alexander NemecekYuzhou JiangErman Ayday
|links| http://arxiv.org/abs/2404.02138v1 |
|updated| 2024-04-02 17:49:40 UTC |
|summary| Recent advancements of large language models LLMs have resulted inindistinguishable text outputs comparable to human-generated text. Watermarkingalgorithms are potential tools that offer a way to differentiate between LLM-and human-generated text by embedding detectable signatures withinLLM-generated output. However current watermarking schemes lack robustnessagainst known attacks against watermarking algorithms. In addition they areimpractical considering an LLM generates tens of thousands of text outputs perday and the watermarking algorithm needs to memorize each output it generatesfor the detection to work. In this work focusing on the limitations of currentwatermarking schemes we propose the concept of a topic-based watermarkingalgorithm for LLMs. The proposed algorithm determines how to generate tokensfor the watermarked LLM output based on extracted topics of an input prompt orthe output of a non-watermarked LLM. Inspired from previous work we proposeusing a pair of lists that are generated based on the specified extractedtopics that specify certain tokens to be included or excluded whilegenerating the watermarked output of the LLM. Using the proposed watermarkingalgorithm we show the practicality of a watermark detection algorithm.Furthermore we discuss a wide range of attacks that can emerge againstwatermarking algorithms for LLMs and the benefit of the proposed watermarkingscheme for the feasibility of modeling a potential attacker considering itsbenefit vs. loss. |


| Item |Content|
| --- |---|
|idx| 2404.02127v1 |
|title| FLawN-T5: An Empirical Examination of Effective Instruction-Tuning Data Mixtures for Legal Reasoning |
|authors| Joel NiklausLucia ZhengArya D. McCarthyChristopher HahnBrian M. RosenPeter HendersonDaniel E. HoGarrett HonkePercy LiangChristopher Manning
|links| http://arxiv.org/abs/2404.02127v1 |
|updated| 2024-04-02 17:33:34 UTC |
|summary| Instruction tuning is an important step in making language models useful fordirect user interaction. However many legal tasks remain out of reach for mostopen LLMs and there do not yet exist any large scale instruction datasets forthe domain. This critically limits research in this application area. In thiswork we curate LawInstruct a large legal instruction dataset covering 17jurisdictions 24 languages and a total of 12M examples. We present evidencethat domain-specific pretraining and instruction tuning improve performance onLegalBench including improving Flan-T5 XL by 8 points or 16 over thebaseline. However the effect does not generalize across all tasks trainingregimes model sizes and other factors. LawInstruct is a resource foraccelerating the development of models with stronger information processing anddecision making capabilities in the legal domain. |


| Item |Content|
| --- |---|
|idx| 2404.02115v1 |
|title| GINopic: Topic Modeling with Graph Isomorphism Network |
|authors| Suman AdhyaDebarshi Kumar Sanyal
|links| http://arxiv.org/abs/2404.02115v1 |
|updated| 2024-04-02 17:18:48 UTC |
|summary| Topic modeling is a widely used approach for analyzing and exploring largedocument collections. Recent research efforts have incorporated pre-trainedcontextualized language models such as BERT embeddings into topic modeling.However they often neglect the intrinsic informational value conveyed bymutual dependencies between words. In this study we introduce GINopic a topicmodeling framework based on graph isomorphism networks to capture thecorrelation between words. By conducting intrinsic quantitative as well asqualitative and extrinsic evaluations on diverse benchmark datasets wedemonstrate the effectiveness of GINopic compared to existing topic models andhighlight its potential for advancing topic modeling. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2404.02157v1 |
|title| Segment Any 3D Object with Language |
|authors| Seungjun LeeYuyang ZhaoGim Hee Lee
|links| http://arxiv.org/abs/2404.02157v1 |
|updated| 2024-04-02 17:59:10 UTC |
|summary| In this paper we investigate Open-Vocabulary 3D Instance SegmentationOV-3DIS with free-form language instructions. Earlier works that rely on onlyannotated base categories for training suffer from limited generalization tounseen novel categories. Recent works mitigate poor generalizability to novelcategories by generating class-agnostic masks or projecting generalized masksfrom 2D to 3D but disregard semantic or geometry information leading tosub-optimal performance. Instead generating generalizable but semantic-relatedmasks directly from 3D point clouds would result in superior outcomes. In thispaper we introduce Segment any 3D Object with LanguagE SOLE which is asemantic and geometric-aware visual-language learning framework with stronggeneralizability by generating semantic-related masks directly from 3D pointclouds. Specifically we propose a multimodal fusion network to incorporatemultimodal semantics in both backbone and decoder. In addition to align the 3Dsegmentation model with various language instructions and enhance the maskquality we introduce three types of multimodal associations as supervision.Our SOLE outperforms previous methods by a large margin on ScanNetv2ScanNet200 and Replica benchmarks and the results are even close to thefully-supervised counterpart despite the absence of class annotations in thetraining. Furthermore extensive qualitative results demonstrate theversatility of our SOLE to language instructions. |


| Item |Content|
| --- |---|
|idx| 2404.02155v1 |
|title| Alpha Invariance: On Inverse Scaling Between Distance and Volume Density in Neural Radiance Fields |
|authors| Joshua AhnHaochen WangRaymond A. YehGreg Shakhnarovich
|links| http://arxiv.org/abs/2404.02155v1 |
|updated| 2024-04-02 17:58:57 UTC |
|summary| Scale-ambiguity in 3D scene dimensions leads to magnitude-ambiguity ofvolumetric densities in neural radiance fields i.e. the densities double whenscene size is halved and vice versa. We call this property alpha invariance.For NeRFs to better maintain alpha invariance we recommend 1 parameterizingboth distance and volume densities in log space and 2 adiscretization-agnostic initialization strategy to guarantee high raytransmittance. We revisit a few popular radiance field models and find thatthese systems use various heuristics to deal with issues arising from scenescaling. We test their behaviors and show our recipe to be more robust. |


| Item |Content|
| --- |---|
|idx| 2404.02154v1 |
|title| Dynamic Pre-training: Towards Efficient and Scalable All-in-One Image Restoration |
|authors| Akshay DudhaneOmkar ThawakarSyed Waqas ZamirSalman KhanFahad Shahbaz KhanMing-Hsuan Yang
|links| http://arxiv.org/abs/2404.02154v1 |
|updated| 2024-04-02 17:58:49 UTC |
|summary| All-in-one image restoration tackles different types of degradations with aunified model instead of having task-specific non-generic models for eachdegradation. The requirement to tackle multiple degradations using the samemodel can lead to high-complexity designs with fixed configuration that lackthe adaptability to more efficient alternatives. We propose DyNet a dynamicfamily of networks designed in an encoder-decoder style for all-in-one imagerestoration tasks. Our DyNet can seamlessly switch between its bulkier andlightweight variants thereby offering flexibility for efficient modeldeployment with a single round of training. This seamless switching is enabledby our weights-sharing mechanism forming the core of our architecture andfacilitating the reuse of initialized module weights. Further to establishrobust weights initialization we introduce a dynamic pre-training strategythat trains variants of the proposed DyNet concurrently thereby achieving a50 reduction in GPU hours. To tackle the unavailability of large-scale datasetrequired in pre-training we curate a high-quality high-resolution imagedataset named Million-IRD having 2M image samples. We validate our DyNet forimage denoising deraining and dehazing in all-in-one setting achievingstate-of-the-art results with 31.34 reduction in GFlops and a 56.75 reductionin parameters compared to baseline models. The source codes and trained modelsare available at https://github.com/akshaydudhane16/DyNet. |


| Item |Content|
| --- |---|
|idx| 2404.02152v1 |
|title| GeneAvatar: Generic Expression-Aware Volumetric Head Avatar Editing from a Single Image |
|authors| Chong BaoYinda ZhangYuan LiXiyu ZhangBangbang YangHujun BaoMarc PollefeysGuofeng ZhangZhaopeng Cui
|links| http://arxiv.org/abs/2404.02152v1 |
|updated| 2024-04-02 17:58:35 UTC |
|summary| Recently we have witnessed the explosive growth of various volumetricrepresentations in modeling animatable head avatars. However due to thediversity of frameworks there is no practical method to support high-levelapplications like 3D head avatar editing across different representations. Inthis paper we propose a generic avatar editing approach that can beuniversally applied to various 3DMM driving volumetric head avatars. To achievethis goal we design a novel expression-aware modification generative modelwhich enables lift 2D editing from a single image to a consistent 3Dmodification field. To ensure the effectiveness of the generative modificationprocess we develop several techniques including an expression-dependentmodification distillation scheme to draw knowledge from the large-scale headavatar model and 2D facial texture editing tools implicit latent spaceguidance to enhance model convergence and a segmentation-based loss reweightstrategy for fine-grained texture inversion. Extensive experiments demonstratethat our method delivers high-quality and consistent results across multipleexpression and viewpoints. Project page: https://zju3dv.github.io/geneavatar/ |


| Item |Content|
| --- |---|
|idx| 2404.02148v1 |
|title| Diffusion$^2$: Dynamic 3D Content Generation via Score Composition of Orthogonal Diffusion Models |
|authors| Zeyu YangZijie PanChun GuLi Zhang
|links| http://arxiv.org/abs/2404.02148v1 |
|updated| 2024-04-02 17:58:03 UTC |
|summary| Recent advancements in 3D generation are predominantly propelled byimprovements in 3D-aware image diffusion models which are pretrained onInternet-scale image data and fine-tuned on massive 3D data offering thecapability of producing highly consistent multi-view images. However due tothe scarcity of synchronized multi-view video data it is impractical to adaptthis paradigm to 4D generation directly. Despite that the available video and3D data are adequate for training video and multi-view diffusion models thatcan provide satisfactory dynamic and geometric priors respectively. In thispaper we present Diffusion2 a novel framework for dynamic 3D contentcreation that leverages the knowledge about geometric consistency and temporalsmoothness from these models to directly sample dense multi-view andmulti-frame images which can be employed to optimize continuous 4Drepresentation. Specifically we design a simple yet effective denoisingstrategy via score composition of video and multi-view diffusion models basedon the probability structure of the images to be generated. Owing to the highparallelism of the image generation and the efficiency of the modern 4Dreconstruction pipeline our framework can generate 4D content within fewminutes. Furthermore our method circumvents the reliance on 4D data therebyhaving the potential to benefit from the scalability of the foundation videoand multi-view diffusion models. Extensive experiments demonstrate the efficacyof our proposed framework and its capability to flexibly adapt to various typesof prompts. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2404.02151v1 |
|title| Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks |
|authors| Maksym AndriushchenkoFrancesco CroceNicolas Flammarion
|links| http://arxiv.org/abs/2404.02151v1 |
|updated| 2024-04-02 17:58:27 UTC |
|summary| We show that even the most recent safety-aligned LLMs are not robust tosimple adaptive jailbreaking attacks. First we demonstrate how to successfullyleverage access to logprobs for jailbreaking: we initially design anadversarial prompt template sometimes adapted to the target LLM and then weapply random search on a suffix to maximize the target logprob e.g. of thetoken Sure potentially with multiple restarts. In this way we achievenearly 100 attack success rate -- according to GPT-4 as a judge -- onGPT-3.5/4 Llama-2-Chat-7B/13B/70B Gemma-7B and R2D2 from HarmBench that wasadversarially trained against the GCG attack. We also show how to jailbreak allClaude models -- that do not expose logprobs -- via either a transfer orprefilling attack with 100 success rate. In addition we show how to userandom search on a restricted set of tokens for finding trojan strings inpoisoned models -- a task that shares many similarities with jailbreaking --which is the algorithm that brought us the first place in the SaTML24 TrojanDetection Competition. The common theme behind these attacks is that adaptivityis crucial: different models are vulnerable to different prompting templatese.g. R2D2 is very sensitive to in-context learning prompts some models haveunique vulnerabilities based on their APIs e.g. prefilling for Claude andin some settings it is crucial to restrict the token search space based onprior knowledge e.g. for trojan detection. We provide the code prompts andlogs of the attacks at https://github.com/tml-epfl/llm-adaptive-attacks. |


| Item |Content|
| --- |---|
|idx| 2404.02141v1 |
|title| Robustly estimating heterogeneity in factorial data using Rashomon Partitions |
|authors| Aparajithan VenkateswaranAnirudh SankarArun G. ChandrasekharTyler H. McCormick
|links| http://arxiv.org/abs/2404.02141v1 |
|updated| 2024-04-02 17:53:28 UTC |
|summary| Many statistical analyses in both observational data and randomized controltrials ask: how does the outcome of interest vary with combinations ofobservable covariates How do various drug combinations affect health outcomesor how does technology adoption depend on incentives and demographics Our goalis to partition this factorial space into pools of covariate combinationswhere the outcome differs across the pools but not within a pool. Existingapproaches i search for a single optimal partition under assumptionsabout the association between covariates or ii sample from the entire set ofpossible partitions. Both these approaches ignore the reality that especiallywith correlation structure in covariates many ways to partition the covariatespace may be statistically indistinguishable despite very differentimplications for policy or science. We develop an alternative perspectivecalled Rashomon Partition Sets RPSs. Each item in the RPS partitions thespace of covariates using a tree-like geometry. RPSs incorporate all partitionsthat have posterior values near the maximum a posteriori partition even ifthey offer substantively different explanations and do so using a prior thatmakes no assumptions about associations between covariates. This prior is theell_0 prior which we show is minimax optimal. Given the RPS we calculatethe posterior of any measurable function of the feature effects vector onoutcomes conditional on being in the RPS. We also characterize approximationerror relative to the entire posterior and provide bounds on the size of theRPS. Simulations demonstrate this framework allows for robust conclusionsrelative to conventional regularization techniques. We apply our method tothree empirical settings: price effects on charitable giving chromosomalstructure telomere length and the introduction of microfinance. |


| Item |Content|
| --- |---|
|idx| 2404.01930v1 |
|title| Adaptive Combinatorial Maximization: Beyond Approximate Greedy Policies |
|authors| Shlomi WeitzmanSivan Sabato
|links| http://arxiv.org/abs/2404.01930v1 |
|updated| 2024-04-02 13:23:54 UTC |
|summary| We study adaptive combinatorial maximization which is a core challenge inmachine learning with applications in active learning as well as many otherdomains. We study the Bayesian setting and consider the objectives ofmaximization under a cardinality constraint and minimum cost coverage. Weprovide new comprehensive approximation guarantees that subsume previousresults as well as considerably strengthen them. Our approximation guaranteessimultaneously support the maximal gain ratio as well as near-submodularutility functions and include both maximization under a cardinality constraintand a minimum cost coverage guarantee. In addition we provided anapproximation guarantee for a modified prior which is crucial for obtainingactive learning guarantees that do not depend on the smallest probability inthe prior. Moreover we discover a new parameter of adaptive selectionpolicies which we term the maximal gain ratio. We show that this parameteris strictly less restrictive than the greedy approximation parameter that hasbeen used in previous approximation guarantees and show that it can be used toprovide stronger approximation guarantees than previous results. In particularwe show that the maximal gain ratio is never larger than the greedyapproximation factor of a policy and that it can be considerably smaller. Thisprovides a new insight into the properties that make a policy useful foradaptive combinatorial maximization. |


| Item |Content|
| --- |---|
|idx| 2404.01883v1 |
|title| Adversarial Combinatorial Bandits with Switching Costs |
|authors| Yanyan DongVincent Y. F. Tan
|links| http://dx.doi.org/10.1109/TIT.2024.3384033 |
|updated| 2024-04-02 12:15:37 UTC |
|summary| We study the problem of adversarial combinatorial bandit with a switchingcost lambda for a switch of each selected arm in each round consideringboth the bandit feedback and semi-bandit feedback settings. In the obliviousadversarial case with K base arms and time horizon T we derive lowerbounds for the minimax regret and design algorithms to approach them. To provethese lower bounds we design stochastic loss sequences for both feedbacksettings building on an idea from previous work in Dekel et al. 2014. Thelower bound for bandit feedback is  tildeOmegabig lambdaKfrac13 TIfrac23big while that for semi-bandit feedbackis  tildeOmegabig lambda K Ifrac13 Tfrac23bigwhere I is the number of base arms in the combinatorial arm played in eachround. To approach these lower bounds we design algorithms that operate inbatches by dividing the time horizon into batches to restrict the number ofswitches between actions. For the bandit feedback setting where only the totalloss of the combinatorial arm is observed we introduce the Batched-Exp2algorithm which achieves a regret upper bound of tildeObiglambdaKfrac13Tfrac23Ifrac43big as T tends to infinity.In the semi-bandit feedback setting where all losses for the combinatorial armare observed we propose the Batched-BROAD algorithm which achieves a regretupper bound of tildeObig lambda Kfrac13TIfrac23big. |


| Item |Content|
| --- |---|
|idx| 2404.01866v1 |
|title| Supervised Autoencoder MLP for Financial Time Series Forecasting |
|authors| Bartosz BieganowskiRobert Slepaczuk
|links| http://arxiv.org/abs/2404.01866v1 |
|updated| 2024-04-02 11:44:37 UTC |
|summary| This paper investigates the enhancement of financial time series forecastingwith the use of neural networks through supervised autoencoders aiming toimprove investment strategy performance. It specifically examines the impact ofnoise augmentation and triple barrier labeling on risk-adjusted returns usingthe Sharpe and Information Ratios. The study focuses on the SP 500 indexEUR/USD and BTC/USD as the traded assets from January 1 2010 to April 302022. Findings indicate that supervised autoencoders with balanced noiseaugmentation and bottleneck size significantly boost strategy effectiveness.However excessive noise and large bottleneck sizes can impair performancehighlighting the importance of precise parameter tuning. This paper alsopresents a derivation of a novel optimization metric that can be used withtriple barrier labeling. The results of this study have substantial policyimplications suggesting that financial institutions and regulators couldleverage techniques presented to enhance market stability and investorprotection while also encouraging more informed and strategic investmentapproaches in various financial sectors. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2404.02147v1 |
|title| Harder, Better, Faster, Stronger: Interactive Visualization for Human-Centered AI Tools |
|authors| Md Naimul HoqueSungbok ShinNiklas Elmqvist
|links| http://arxiv.org/abs/2404.02147v1 |
|updated| 2024-04-02 17:57:57 UTC |
|summary| Human-centered AI HCAI rather than replacing the human puts the humanuser in the drivers seat of so-called human-centered AI-infused tools HCAItools: interactive software tools that amplify augment empower and enhancehuman performance using AI models often novel generative or foundation AIones. In this paper we discuss how interactive visualization can be a keyenabling technology for creating such human-centered AI tools. Visualizationhas already been shown to be a fundamental component in explainable AI modelsand coupling this with data-driven semantic and unified interaction feedbackloops will enable a human-centered approach to integrating AI models in theloop with human users. We present several examples of our past and current workon such HCAI tools including for creative writing temporal prediction anduser experience analysis. We then draw parallels between these tools to suggestcommon themes on how interactive visualization can support the design of futureHCAI tools. |


| Item |Content|
| --- |---|
|idx| 2404.02109v1 |
|title| The Effects of Group Sanctions on Participation and Toxicity: Quasi-experimental Evidence from the Fediverse |
|authors| Carl ColglazierNathan TeBlunthuisAaron Shaw
|links| http://arxiv.org/abs/2404.02109v1 |
|updated| 2024-04-02 17:10:33 UTC |
|summary| Online communities often overlap and coexist despite incongruent norms andapproaches to content moderation. When communities diverge decentralized andfederated communities may pursue group-level sanctions including defederationdisconnection to block communication between members of specific communities.We investigate the effects of defederation in the context of the Fediverse aset of decentralized interconnected social networks with independentgovernance. Mastodon and Pleroma the most popular software powering theFediverse allow administrators on one server to defederate from another. Weuse a difference-in-differences approach and matched controls to estimate theeffects of defederation events on participation and message toxicity amongaffected members of the blocked and blocking servers. We find that defederationcauses a drop in activity for accounts on the blocked servers but not on theblocking servers. Also we find no evidence of an effect of defederation onmessage toxicity. |


| Item |Content|
| --- |---|
|idx| 2404.02081v1 |
|title| Explainability in JupyterLab and Beyond: Interactive XAI Systems for Integrated and Collaborative Workflows |
|authors| Grace GuoDustin ArendtAlex Endert
|links| http://arxiv.org/abs/2404.02081v1 |
|updated| 2024-04-02 16:27:44 UTC |
|summary| Explainable AI XAI tools represent a turn to more human-centered andhuman-in-the-loop AI approaches that emphasize user needs and perspectives inmachine learning model development workflows. However while the majority of MLresources available today are developed for Python computational environmentssuch as JupyterLab and Jupyter Notebook the same has not been true ofinteractive XAI systems which are often still implemented as standaloneinterfaces. In this paper we address this mismatch by identifying three designpatterns for embedding front-end XAI interfaces into Jupyter namely: 1One-way communication from Python to JavaScript 2 Two-way datasynchronization and 3 Bi-directional callbacks. We also provide anopen-source toolkit bonXAI that demonstrates how each design pattern might beused to build interactive XAI tools for a Pytorch text classification workflow.Finally we conclude with a discussion of best practices and open questions.Our aims for this paper are to discuss how interactive XAI tools might bedeveloped for computational notebooks and how they can better integrate intoexisting model development workflows to support more collaborativehuman-centered AI. |


| Item |Content|
| --- |---|
|idx| 2404.02009v1 |
|title| Preuve de concept d'un bot vocal dialoguant en wolof |
|authors| Elodie GauthierPapa-Séga WadeThierry MoudencPatrice CollenEmilie De NeefOumar BaNdeye Khoyane CamaCheikh Ahmadou Bamba KebeNdeye Aissatou GningueThomas Mendo'o Aristide
|links| http://arxiv.org/abs/2404.02009v1 |
|updated| 2024-04-02 14:53:41 UTC |
|summary| This paper presents the proof-of-concept of the first automatic voiceassistant ever built in Wolof language the main vehicular language spoken inSenegal. This voicebot is the result of a collaborative research projectbetween Orange Innovation in France Orange Senegal aka Sonatel and ADNCorpa small IT company based in Dakar Senegal. The purpose of the voicebot is toprovide information to Orange customers about the Sargal loyalty program ofOrange Senegal by using the most natural mean to communicate: speech. Thevoicebot receives in input the customers oral request that is then processedby a SLU system to reply to the customers request using audio recordings. Thefirst results of this proof-of-concept are encouraging as we achieved 22 ofWER for the ASR task and 78 of F1-score on the NLU task. |


| Item |Content|
| --- |---|
|idx| 2404.01997v1 |
|title| Cash or Non-Cash? Unveiling Ideators' Incentive Preferences in Crowdsourcing Contests |
|authors| Christoph RiedlJohann FüllerKatja HutterGerard J. Tellis
|links| http://arxiv.org/abs/2404.01997v1 |
|updated| 2024-04-02 14:41:05 UTC |
|summary| Even though research has repeatedly shown that non-cash incentives can beeffective cash incentives are the de facto standard in crowdsourcing contests.In this multi-study research we quantify ideators preferences for non-cashincentives and investigate how allowing ideators to self-select their preferredincentive -- offering ideators a choice between cash and non-cash incentives --affects their creative performance. We further explore whether the marketcontext of the organization hosting the contest -- social non-profit ormonetary for-profit -- moderates incentive preferences and theireffectiveness. We find that individuals exhibit heterogeneous incentivepreferences and often prefer non-cash incentives even in for-profit contexts.Offering ideators a choice of incentives can enhance creative performance.Market context moderates the effect of incentives such that ideators whoreceive non-cash incentives in for-profit contexts tend to exert less effort.We show that heterogeneity of ideators preferences and the ability to satisfydiverse preferences with suitably diverse incentive options is a criticalboundary condition to realizing benefits from offering ideators a choice ofincentives. We provide managers with guidance to design effective incentives byimproving incentive-preference fit for ideators. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2404.01999v1 |
|title| Emergence of Chemotactic Strategies with Multi-Agent Reinforcement Learning |
|authors| Samuel ToveyChristoph LohrmannChristian Holm
|links| http://arxiv.org/abs/2404.01999v1 |
|updated| 2024-04-02 14:42:52 UTC |
|summary| Reinforcement learning RL is a flexible and efficient method forprogramming micro-robots in complex environments. Here we investigate whetherreinforcement learning can provide insights into biological systems whentrained to perform chemotaxis. Namely whether we can learn about howintelligent agents process given information in order to swim towards a target.We run simulations covering a range of agent shapes sizes and swim speeds todetermine if the physical constraints on biological swimmers namely Brownianmotion lead to regions where reinforcement learners training fails. We findthat the RL agents can perform chemotaxis as soon as it is physically possibleand in some cases even before the active swimming overpowers the stochasticenvironment. We study the efficiency of the emergent policy and identifyconvergence in agent size and swim speeds. Finally we study the strategyadopted by the reinforcement learning algorithm to explain how the agentsperform their tasks. To this end we identify three emerging dominantstrategies and several rare approaches taken. These strategies whilstproducing almost identical trajectories in simulation are distinct and giveinsight into the possible mechanisms behind which biological agents exploretheir environment and respond to changing conditions. |


| Item |Content|
| --- |---|
|idx| 2404.01752v1 |
|title| Safe Interval RRT* for Scalable Multi-Robot Path Planning in Continuous Space |
|authors| Joonyeol SimJoonkyung KimChangjoo Nam
|links| http://arxiv.org/abs/2404.01752v1 |
|updated| 2024-04-02 09:07:12 UTC |
|summary| In this paper we consider the problem of Multi-Robot Path Planning MRPP incontinuous space to find conflict-free paths. The difficulty of the problemarises from two primary factors. First the involvement of multiple robotsleads to combinatorial decision-making which escalates the search spaceexponentially. Second the continuous space presents potentially infinitestates and actions. For this problem we propose a two-level approach where thelow level is a sampling-based planner Safe Interval RRT SI-RRT that finds acollision-free trajectory for individual robots. The high level can use anymethod that can resolve inter-robot conflicts where we employ tworepresentative methods that are Prioritized Planning SI-CPP and ConflictBased Search SI-CCBS. Experimental results show that SI-RRT can find ahigh-quality solution quickly with a small number of samples. SI-CPP exhibitsimproved scalability while SI-CCBS produces higher-quality solutions comparedto the state-of-the-art planners for continuous space. Compared to the mostscalable existing algorithm SI-CPP achieves a success rate that is up to 94higher with 100 robots while maintaining solution quality i.e. flowtime thesum of travel times of all robots without significant compromise. SI-CPP alsodecreases the makespan up to 45. SI-CCBS decreases the flowtime by 9 comparedto the competitor albeit exhibiting a 14 lower success rate. |


| Item |Content|
| --- |---|
|idx| 2404.01557v1 |
|title| Distributed Autonomous Swarm Formation for Dynamic Network Bridging |
|authors| Raffaele GallieraThies MöhlenhofAlessandro AmatoDaniel DuranKristen Brent VenableNiranjan Suri
|links| http://arxiv.org/abs/2404.01557v1 |
|updated| 2024-04-02 01:45:03 UTC |
|summary| Effective operation and seamless cooperation of robotic systems are afundamental component of next-generation technologies and applications. Incontexts such as disaster response swarm operations require coordinatedbehavior and mobility control to be handled in a distributed manner with thequality of the agents actions heavily relying on the communication betweenthem and the underlying network. In this paper we formulate the problem ofdynamic network bridging in a novel Decentralized Partially Observable MarkovDecision Process Dec-POMDP where a swarm of agents cooperates to form a linkbetween two distant moving targets. Furthermore we propose a Multi-AgentReinforcement Learning MARL approach for the problem based on GraphConvolutional Reinforcement Learning DGN which naturally applies to thenetworked distributed nature of the task. The proposed method is evaluated ina simulated environment and compared to a centralized heuristic baselineshowing promising results. Moreover a further step in the direction ofsim-to-real transfer is presented by additionally evaluating the proposedapproach in a near Live Virtual Constructive LVC UAV framework. |


| Item |Content|
| --- |---|
|idx| 2404.01551v1 |
|title| Multi-Agent Reinforcement Learning with Control-Theoretic Safety Guarantees for Dynamic Network Bridging |
|authors| Raffaele GallieraKonstantinos MitsopoulosNiranjan SuriRaffaele Romagnoli
|links| http://arxiv.org/abs/2404.01551v1 |
|updated| 2024-04-02 01:30:41 UTC |
|summary| Addressing complex cooperative tasks in safety-critical environments posessignificant challenges for Multi-Agent Systems especially under conditions ofpartial observability. This work introduces a hybrid approach that integratesMulti-Agent Reinforcement Learning with control-theoretic methods to ensuresafe and efficient distributed strategies. Our contributions include a novelsetpoint update algorithm that dynamically adjusts agents positions topreserve safety conditions without compromising the missions objectives.Through experimental validation we demonstrate significant advantages overconventional MARL strategies achieving comparable task performance with zerosafety violations. Our findings indicate that integrating safe control withlearning approaches not only enhances safety compliance but also achieves goodperformance in mission objectives. |


| Item |Content|
| --- |---|
|idx| 2404.01131v1 |
|title| GOV-REK: Governed Reward Engineering Kernels for Designing Robust Multi-Agent Reinforcement Learning Systems |
|authors| Ashish RanaMichael OesterleJannik Brinkmann
|links| http://arxiv.org/abs/2404.01131v1 |
|updated| 2024-04-01 14:19:00 UTC |
|summary| For multi-agent reinforcement learning systems MARLS the problemformulation generally involves investing massive reward engineering effortspecific to a given problem. However this effort often cannot be translated toother problems worse it gets wasted when system dynamics change drastically.This problem is further exacerbated in sparse reward scenarios where ameaningful heuristic can assist in the policy convergence task. We proposeGOVerned Reward Engineering Kernels GOV-REK which dynamically assign rewarddistributions to agents in MARLS during its learning stage. We also introducegovernance kernels which exploit the underlying structure in either state orjoint action space for assigning meaningful agent reward distributions. Duringthe agent learning stage it iteratively explores different reward distributionconfigurations with a Hyperband-like algorithm to learn ideal agent rewardmodels in a problem-agnostic manner. Our experiments demonstrate that ourmeaningful reward priors robustly jumpstart the learning process foreffectively learning different MARL problems. |


