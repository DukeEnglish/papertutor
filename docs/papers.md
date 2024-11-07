# cs.CL 

| Item |Content|
| --- |---|
|idx| 2411.03314v1 |
|title| MME-Finance: A Multimodal Finance Benchmark for Expert-level Understanding and Reasoning |
|authors| Ziliang GanYu LuDong ZhangHaohan LiChe LiuJian LiuJi LiuHaipang WuChaoyou FuZenglin XuRongjunchen ZhangYong Dai
|links| http://arxiv.org/abs/2411.03314v1 |
|updated| 2024-11-05 18:59:51 UTC |
|summary| In recent years multimodal benchmarks for general domains have guided therapid development of multimodal models on general tasks. However the financialfield has its peculiarities. It features unique graphical images e.g.candlestick charts technical indicator charts and possesses a wealth ofspecialized financial knowledge e.g. futures turnover rate. Thereforebenchmarks from general fields often fail to measure the performance ofmultimodal models in the financial domain and thus cannot effectively guidethe rapid development of large financial models. To promote the development oflarge financial multimodal models we propose MME-Finance an bilingualopen-ended and practical usage-oriented Visual Question Answering VQAbenchmark. The characteristics of our benchmark are finance and expertisewhich include constructing charts that reflect the actual usage needs of userse.g. computer screenshots and mobile photography creating questionsaccording to the preferences in financial domain inquiries and annotatingquestions by experts with 10 years of experience in the financial industry.Additionally we have developed a custom-designed financial evaluation systemin which visual information is first introduced in the multi-modal evaluationprocess. Extensive experimental evaluations of 19 mainstream MLLMs areconducted to test their perception reasoning and cognition capabilities. Theresults indicate that models performing well on general benchmarks cannot dowell on MME-Finance for instance the top-performing open-source andclosed-source models obtain 65.69 Qwen2VL-72B and 63.18 GPT-4orespectively. Their performance is particularly poor in categories mostrelevant to finance such as candlestick charts and technical indicator charts.In addition we propose a Chinese version which helps compare performance ofMLLMs under a Chinese context. |


| Item |Content|
| --- |---|
|idx| 2411.03307v1 |
|title| LLMs for Domain Generation Algorithm Detection |
|authors| Reynier Leyva La OCarlos A. CataniaTatiana Parlanti
|links| http://arxiv.org/abs/2411.03307v1 |
|updated| 2024-11-05 18:01:12 UTC |
|summary| This work analyzes the use of large language models LLMs for detectingdomain generation algorithms DGAs. We perform a detailed evaluation of twoimportant techniques: In-Context Learning ICL and Supervised Fine-TuningSFT showing how they can improve detection. SFT increases performance byusing domain-specific data whereas ICL helps the detection model to quicklyadapt to new threats without requiring much retraining. We use Metas Llama3 8Bmodel on a custom dataset with 68 malware families and normal domainscovering several hard-to-detect schemes including recent word-based DGAs.Results proved that LLM-based methods can achieve competitive results in DGAdetection. In particular the SFT-based LLM DGA detector outperformsstate-of-the-art models using attention layers achieving 94 accuracy with a4 false positive rate FPR and excelling at detecting word-based DGA domains. |


| Item |Content|
| --- |---|
|idx| 2411.03300v1 |
|title| VERITAS: A Unified Approach to Reliability Evaluation |
|authors| Rajkumar RamamurthyMeghana Arakkal RajeevOliver MolenschotJames ZouNazneen Rajani
|links| http://arxiv.org/abs/2411.03300v1 |
|updated| 2024-11-05 17:53:25 UTC |
|summary| Large language models LLMs often fail to synthesize information from theircontext to generate an accurate response. This renders them unreliable inknowledge intensive settings where reliability of the output is key. A criticalcomponent for reliable LLMs is the integration of a robust fact-checking systemthat can detect hallucinations across various formats. While severalopen-access fact-checking models are available their functionality is oftenlimited to specific tasks such as grounded question-answering or entailmentverification and they perform less effectively in conversational settings. Onthe other hand closed-access models like GPT-4 and Claude offer greaterflexibility across different contexts including grounded dialogueverification but are hindered by high costs and latency. In this work weintroduce VERITAS a family of hallucination detection models designed tooperate flexibly across diverse contexts while minimizing latency and costs.VERITAS achieves state-of-the-art results considering average performance onall major hallucination detection benchmarks with 10 increase in averageperformance when compared to similar-sized models and get close to theperformance of GPT4 turbo with LLM-as-a-judge setting. |


| Item |Content|
| --- |---|
|idx| 2411.03284v1 |
|title| SMoA: Improving Multi-agent Large Language Models with Sparse Mixture-of-Agents |
|authors| Dawei LiZhen TanPeijia QianYifan LiKumar Satvik ChaudharyLijie HuJiayi Shen
|links| http://arxiv.org/abs/2411.03284v1 |
|updated| 2024-11-05 17:33:39 UTC |
|summary| While multi-agent systems have been shown to significantly enhance theperformance of Large Language Models LLMs across various tasks andapplications the dense interaction between scaling agents potentially hamperstheir efficiency and diversity. To address these challenges we drawinspiration from the sparse mixture-of-agents SMoE and propose a sparsemixture-of-agents SMoA framework to improve the efficiency and diversity ofmulti-agent LLMs. Unlike completely connected structures SMoA introduces novelResponse Selection and Early Stopping mechanisms to sparsify information flowsamong individual LLM agents striking a balance between performance andefficiency. Additionally inspired by the expert diversity principle in SMoEframeworks for workload balance between experts we assign distinct roledescriptions to each LLM agent fostering diverse and divergent thinking.Extensive experiments on reasoning alignment and fairness benchmarksdemonstrate that SMoA achieves performance comparable to traditionalmixture-of-agents approaches but with significantly lower computational costs.Further analysis reveals that SMoA is more stable has a greater capacity toscale and offers considerable potential through hyper-parameter optimization.Code and data will be available at: https://github.com/David-Li0406/SMoA. |


| Item |Content|
| --- |---|
|idx| 2411.03250v1 |
|title| DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models |
|authors| Ying ZhouXinyao WangYulei NiuYaojie ShenLexin TangFan ChenBen HeLe SunLongyin Wen
|links| http://arxiv.org/abs/2411.03250v1 |
|updated| 2024-11-05 16:47:53 UTC |
|summary| Recent advancements in large language models LLMs have significantlyenhanced their knowledge and generative capabilities leading to a surge ofinterest in leveraging LLMs for high-quality data synthesis. However syntheticdata generation via prompting LLMs remains challenging due to LLMs limitedunderstanding of target data distributions and the complexity of promptengineering especially for structured formatted data. To address these issueswe introduce DiffLM a controllable data synthesis framework based onvariational autoencoder VAE which further 1 leverages diffusion models toreserve more information of original distribution and format structure in thelearned latent distribution and 2 decouples the learning of targetdistribution knowledge from the LLMs generative objectives via a plug-and-playlatent feature injection module. As we observed significant discrepanciesbetween the VAEs latent representations and the real data distribution thelatent diffusion module is introduced into our framework to learn a fullyexpressive latent distribution. Evaluations on seven real-world datasets withstructured formatted data i.e. Tabular Code and Tool data demonstrate thatDiffLM generates high-quality data with performance on downstream taskssurpassing that of real data by 2-7 percent in certain cases. The data and codewill be publicly available upon completion of internal review. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2411.03312v1 |
|title| Inference Optimal VLMs Need Only One Visual Token but Larger Models |
|authors| Kevin Y. LiSachin GoyalJoao D. SemedoJ. Zico Kolter
|links| http://arxiv.org/abs/2411.03312v1 |
|updated| 2024-11-05 18:54:21 UTC |
|summary| Vision Language Models VLMs have demonstrated strong capabilities acrossvarious visual understanding and reasoning tasks. However their real-worlddeployment is often constrained by high latency during inference due tosubstantial compute required to process the large number of input tokenspredominantly from the image by the LLM. To reduce inference costs one caneither downsize the LLM or reduce the number of input image-tokens the latterof which has been the focus of many recent works around token compression.However it is unclear what the optimal trade-off is as both the factorsdirectly affect the VLM performance. We first characterize this optimaltrade-off between the number of visual tokens and LLM parameters byestablishing scaling laws that capture variations in performance with these twofactors. Our results reveal a surprising trend: for visual reasoning tasks theinference-optimal behavior in VLMs i.e. minimum downstream error at any givenfixed inference compute is achieved when using the largest LLM that fitswithin the inference budget while minimizing visual token count - often to asingle token. While the token reduction literature has mainly focused onmaintaining base model performance by modestly reducing the token count e.g.5-10times our results indicate that the compute-optimal inference regimerequires operating under even higher token compression ratios. Based on theseinsights we take some initial steps towards building approaches tailored forhigh token compression settings. Code is available athttps://github.com/locuslab/llava-token-compression. |


| Item |Content|
| --- |---|
|idx| 2411.03300v1 |
|title| VERITAS: A Unified Approach to Reliability Evaluation |
|authors| Rajkumar RamamurthyMeghana Arakkal RajeevOliver MolenschotJames ZouNazneen Rajani
|links| http://arxiv.org/abs/2411.03300v1 |
|updated| 2024-11-05 17:53:25 UTC |
|summary| Large language models LLMs often fail to synthesize information from theircontext to generate an accurate response. This renders them unreliable inknowledge intensive settings where reliability of the output is key. A criticalcomponent for reliable LLMs is the integration of a robust fact-checking systemthat can detect hallucinations across various formats. While severalopen-access fact-checking models are available their functionality is oftenlimited to specific tasks such as grounded question-answering or entailmentverification and they perform less effectively in conversational settings. Onthe other hand closed-access models like GPT-4 and Claude offer greaterflexibility across different contexts including grounded dialogueverification but are hindered by high costs and latency. In this work weintroduce VERITAS a family of hallucination detection models designed tooperate flexibly across diverse contexts while minimizing latency and costs.VERITAS achieves state-of-the-art results considering average performance onall major hallucination detection benchmarks with 10 increase in averageperformance when compared to similar-sized models and get close to theperformance of GPT4 turbo with LLM-as-a-judge setting. |


| Item |Content|
| --- |---|
|idx| 2411.03294v2 |
|title| Out-of-Distribution Recovery with Object-Centric Keypoint Inverse Policy For Visuomotor Imitation Learning |
|authors| George Jiayuan GaoTianyu LiNadia Figueroa
|links| http://arxiv.org/abs/2411.03294v2 |
|updated| 2024-11-06 17:53:26 UTC |
|summary| We propose an object-centric recovery policy framework to address thechallenges of out-of-distribution OOD scenarios in visuomotor policylearning. Previous behavior cloning BC methods rely heavily on a large amountof labeled data coverage failing in unfamiliar spatial states. Without relyingon extra data collection our approach learns a recovery policy constructed byan inverse policy inferred from object keypoint manifold gradient in theoriginal training data. The recovery policy serves as a simple add-on to anybase visuomotor BC policy agnostic to a specific method guiding the systemback towards the training distribution to ensure task success even in OODsituations. We demonstrate the effectiveness of our object-centric framework inboth simulation and real robot experiments achieving an improvement of 77.7over the base policy in OOD. Project Website:https://sites.google.com/view/ocr-penn |


| Item |Content|
| --- |---|
|idx| 2411.03292v1 |
|title| Interaction2Code: How Far Are We From Automatic Interactive Webpage Generation? |
|authors| Jingyu XiaoYuxuan WanYintong HuoZhiyao XuMichael R. Lyu
|links| http://arxiv.org/abs/2411.03292v1 |
|updated| 2024-11-05 17:40:03 UTC |
|summary| Converting webpage design into functional UI code is a critical step forbuilding websites which can be labor-intensive and time-consuming. To automatethis design-to-code transformation process various automated methods usinglearning-based networks and multi-modal large language models MLLMs have beenproposed. However these studies were merely evaluated on a narrow range ofstatic web pages and ignored dynamic interaction elements making them lesspractical for real-world website deployment.  To fill in the blank we present the first systematic investigation of MLLMsin generating interactive webpages. Specifically we first formulate theInteraction-to-Code task and build the Interaction2Code benchmark that contains97 unique web pages and 213 distinct interactions spanning 15 webpage typesand 30 interaction categories. We then conduct comprehensive experiments onthree state-of-the-art SOTA MLLMs using both automatic metrics and humanevaluations thereby summarizing six findings accordingly. Our experimentalresults highlight the limitations of MLLMs in generating fine-grainedinteractive features and managing interactions with complex transformations andsubtle visual modifications. We further analyze failure cases and theirunderlying causes identifying 10 common failure types and assessing theirseverity. Additionally our findings reveal three critical influencing factorsi.e. prompts visual saliency and textual descriptions that can enhance theinteraction generation performance of MLLMs. Based on these findings we elicitimplications for researchers and developers providing a foundation for futureadvancements in this field. Datasets and source code are available athttps://github.com/WebPAI/Interaction2Code. |


| Item |Content|
| --- |---|
|idx| 2411.03287v1 |
|title| The Future of Intelligent Healthcare: A Systematic Analysis and Discussion on the Integration and Impact of Robots Using Large Language Models for Healthcare |
|authors| Souren PashangpourGoldie Nejat
|links| http://dx.doi.org/10.3390/robotics13080112 |
|updated| 2024-11-05 17:36:32 UTC |
|summary| The potential use of large language models LLMs in healthcare robotics canhelp address the significant demand put on healthcare systems around the worldwith respect to an aging demographic and a shortage of healthcareprofessionals. Even though LLMs have already been integrated into medicine toassist both clinicians and patients the integration of LLMs within healthcarerobots has not yet been explored for clinical settings. In this perspectivepaper we investigate the groundbreaking developments in robotics and LLMs touniquely identify the needed system requirements for designing health specificLLM based robots in terms of multi modal communication through human robotinteractions HRIs semantic reasoning and task planning. Furthermore wediscuss the ethical issues open challenges and potential future researchdirections for this emerging innovative field. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2411.03312v1 |
|title| Inference Optimal VLMs Need Only One Visual Token but Larger Models |
|authors| Kevin Y. LiSachin GoyalJoao D. SemedoJ. Zico Kolter
|links| http://arxiv.org/abs/2411.03312v1 |
|updated| 2024-11-05 18:54:21 UTC |
|summary| Vision Language Models VLMs have demonstrated strong capabilities acrossvarious visual understanding and reasoning tasks. However their real-worlddeployment is often constrained by high latency during inference due tosubstantial compute required to process the large number of input tokenspredominantly from the image by the LLM. To reduce inference costs one caneither downsize the LLM or reduce the number of input image-tokens the latterof which has been the focus of many recent works around token compression.However it is unclear what the optimal trade-off is as both the factorsdirectly affect the VLM performance. We first characterize this optimaltrade-off between the number of visual tokens and LLM parameters byestablishing scaling laws that capture variations in performance with these twofactors. Our results reveal a surprising trend: for visual reasoning tasks theinference-optimal behavior in VLMs i.e. minimum downstream error at any givenfixed inference compute is achieved when using the largest LLM that fitswithin the inference budget while minimizing visual token count - often to asingle token. While the token reduction literature has mainly focused onmaintaining base model performance by modestly reducing the token count e.g.5-10times our results indicate that the compute-optimal inference regimerequires operating under even higher token compression ratios. Based on theseinsights we take some initial steps towards building approaches tailored forhigh token compression settings. Code is available athttps://github.com/locuslab/llava-token-compression. |


| Item |Content|
| --- |---|
|idx| 2411.03279v1 |
|title| Oblivious Defense in ML Models: Backdoor Removal without Detection |
|authors| Shafi GoldwasserJonathan ShaferNeekon VafaVinod Vaikuntanathan
|links| http://arxiv.org/abs/2411.03279v1 |
|updated| 2024-11-05 17:20:53 UTC |
|summary| As society grows more reliant on machine learning ensuring the security ofmachine learning systems against sophisticated attacks becomes a pressingconcern. A recent result of Goldwasser Kim Vaikuntanathan and Zamir 2022shows that an adversary can plant undetectable backdoors in machine learningmodels allowing the adversary to covertly control the models behavior.Backdoors can be planted in such a way that the backdoored machine learningmodel is computationally indistinguishable from an honest model withoutbackdoors.  In this paper we present strategies for defending against backdoors in MLmodels even if they are undetectable. The key observation is that it issometimes possible to provably mitigate or even remove backdoors withoutneeding to detect them using techniques inspired by the notion of randomself-reducibility. This depends on properties of the ground-truth labelschosen by nature and not of the proposed ML model which may be chosen by anattacker.  We give formal definitions for secure backdoor mitigation and proceed toshow two types of results. First we show a global mitigation techniquewhich removes all backdoors from a machine learning model under the assumptionthat the ground-truth labels are close to a Fourier-heavy function. Second weconsider distributions where the ground-truth labels are close to a linear orpolynomial function in mathbbRn. Here we show local mitigationtechniques which remove backdoors with high probability for every inputs ofinterest and are computationally cheaper than global mitigation. All of ourconstructions are black-box so our techniques work without needing access tothe models representation i.e. its code or parameters. Along the way weprove a simple result for robust mean estimation. |


| Item |Content|
| --- |---|
|idx| 2411.03273v1 |
|title| Graph-Based Semi-Supervised Segregated Lipschitz Learning |
|authors| Farid BozorgniaYassine BelkheiriAbderrahim Elmoataz
|links| http://arxiv.org/abs/2411.03273v1 |
|updated| 2024-11-05 17:16:56 UTC |
|summary| This paper presents an approach to semi-supervised learning for theclassification of data using the Lipschitz Learning on graphs. We develop agraph-based semi-supervised learning framework that leverages the properties ofthe infinity Laplacian to propagate labels in a dataset where only a fewsamples are labeled. By extending the theory of spatial segregation from theLaplace operator to the infinity Laplace operator both in continuum anddiscrete settings our approach provides a robust method for dealing with classimbalance a common challenge in machine learning. Experimental validation onseveral benchmark datasets demonstrates that our method not only improvesclassification accuracy compared to existing methods but also ensures efficientlabel propagation in scenarios with limited labeled data. |


| Item |Content|
| --- |---|
|idx| 2411.03270v1 |
|title| Stable Matching with Ties: Approximation Ratios and Learning |
|authors| Shiyun LinSimon MaurasNadav MerlisVianney Perchet
|links| http://arxiv.org/abs/2411.03270v1 |
|updated| 2024-11-05 17:14:46 UTC |
|summary| We study the problem of matching markets with ties where one side of themarket does not necessarily have strict preferences over members at its otherside. For example workers do not always have strict preferences over jobsstudents can give the same ranking for different schools and more. Inparticular assume w.l.o.g. that workers preferences are determined by theirutility from being matched to each job which might admit ties. Notably incontrast to classical two-sided markets with strict preferences there is nolonger a single stable matching that simultaneously maximizes the utility forall workers.  We aim to guarantee each worker the largest possible share from the utilityin her best possible stable matching. We call the ratio between the workersbest possible stable utility and its assigned utility the emphOptimal StableShare OSS-ratio. We first prove that distributions over stable matchingscannot guarantee an OSS-ratio that is sublinear in the number of workers.Instead randomizing over possibly non-stable matchings we show how to achievea tight logarithmic OSS-ratio. Then we analyze the case where the real utilityis not necessarily known and can only be approximated. In particular weprovide an algorithm that guarantees a similar fraction of the utility comparedto the best possible utility. Finally we move to a bandit setting where weselect a matching at each round and only observe the utilities for matches weperform. We show how to utilize our results for approximate utilities togracefully interpolate between problems without ties and problems withstatistical ties small suboptimality gaps. |


| Item |Content|
| --- |---|
|idx| 2411.03263v1 |
|title| Proxy-informed Bayesian transfer learning with unknown sources |
|authors| Sabina J. SlomanJulien MartinelliSamuel Kaski
|links| http://arxiv.org/abs/2411.03263v1 |
|updated| 2024-11-05 17:02:29 UTC |
|summary| Generalization outside the scope of ones training data requires leveragingprior knowledge about the effects that transfer and the effects that dontbetween different data sources. Bayesian transfer learning is a principledparadigm for specifying this knowledge and refining it on the basis of datafrom the source training and target prediction tasks. We address thechallenging transfer learning setting where the learner i cannot fine-tune inthe target task and ii does not know which source data points correspond tothe same task i.e. the data sources are unknown. We propose a proxy-informedrobust method for probabilistic transfer learning PROMPT which provides aposterior predictive estimate tailored to the structure of the target taskwithout requiring the learner have access to any outcome information from thetarget task. Instead PROMPT relies on the availability of proxy information.PROMPT uses the same proxy information for two purposes: i estimation ofeffects specific to the target task and ii construction of a robustreweighting of the source data for estimation of effects that transfer betweentasks. We provide theoretical results on the effect of this reweighting on therisk of negative transfer and demonstrate application of PROMPT in twosynthetic settings. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2411.03314v1 |
|title| MME-Finance: A Multimodal Finance Benchmark for Expert-level Understanding and Reasoning |
|authors| Ziliang GanYu LuDong ZhangHaohan LiChe LiuJian LiuJi LiuHaipang WuChaoyou FuZenglin XuRongjunchen ZhangYong Dai
|links| http://arxiv.org/abs/2411.03314v1 |
|updated| 2024-11-05 18:59:51 UTC |
|summary| In recent years multimodal benchmarks for general domains have guided therapid development of multimodal models on general tasks. However the financialfield has its peculiarities. It features unique graphical images e.g.candlestick charts technical indicator charts and possesses a wealth ofspecialized financial knowledge e.g. futures turnover rate. Thereforebenchmarks from general fields often fail to measure the performance ofmultimodal models in the financial domain and thus cannot effectively guidethe rapid development of large financial models. To promote the development oflarge financial multimodal models we propose MME-Finance an bilingualopen-ended and practical usage-oriented Visual Question Answering VQAbenchmark. The characteristics of our benchmark are finance and expertisewhich include constructing charts that reflect the actual usage needs of userse.g. computer screenshots and mobile photography creating questionsaccording to the preferences in financial domain inquiries and annotatingquestions by experts with 10 years of experience in the financial industry.Additionally we have developed a custom-designed financial evaluation systemin which visual information is first introduced in the multi-modal evaluationprocess. Extensive experimental evaluations of 19 mainstream MLLMs areconducted to test their perception reasoning and cognition capabilities. Theresults indicate that models performing well on general benchmarks cannot dowell on MME-Finance for instance the top-performing open-source andclosed-source models obtain 65.69 Qwen2VL-72B and 63.18 GPT-4orespectively. Their performance is particularly poor in categories mostrelevant to finance such as candlestick charts and technical indicator charts.In addition we propose a Chinese version which helps compare performance ofMLLMs under a Chinese context. |


| Item |Content|
| --- |---|
|idx| 2411.03313v2 |
|title| Classification Done Right for Vision-Language Pre-Training |
|authors| Zilong HuangQinghao YeBingyi KangJiashi FengHaoqi Fan
|links| http://arxiv.org/abs/2411.03313v2 |
|updated| 2024-11-06 12:07:08 UTC |
|summary| We introduce SuperClass a super simple classification method forvision-language pre-training on image-text data. Unlike its contrastivecounterpart CLIP who contrast with a text encoder SuperClass directly utilizestokenized raw text as supervised classification labels without the need foradditional text filtering or selection. Due to the absence of the text encodingas contrastive target SuperClass does not require a text encoder and does notneed to maintain a large batch size as CLIP does. SuperClass demonstratedsuperior performance on various downstream tasks including classic computervision benchmarks and vision language downstream tasks. We further explored thescaling behavior of SuperClass on model size training length or data sizeand reported encouraging results and comparisons to CLIP.https://github.com/x-cls/superclass |


| Item |Content|
| --- |---|
|idx| 2411.03312v1 |
|title| Inference Optimal VLMs Need Only One Visual Token but Larger Models |
|authors| Kevin Y. LiSachin GoyalJoao D. SemedoJ. Zico Kolter
|links| http://arxiv.org/abs/2411.03312v1 |
|updated| 2024-11-05 18:54:21 UTC |
|summary| Vision Language Models VLMs have demonstrated strong capabilities acrossvarious visual understanding and reasoning tasks. However their real-worlddeployment is often constrained by high latency during inference due tosubstantial compute required to process the large number of input tokenspredominantly from the image by the LLM. To reduce inference costs one caneither downsize the LLM or reduce the number of input image-tokens the latterof which has been the focus of many recent works around token compression.However it is unclear what the optimal trade-off is as both the factorsdirectly affect the VLM performance. We first characterize this optimaltrade-off between the number of visual tokens and LLM parameters byestablishing scaling laws that capture variations in performance with these twofactors. Our results reveal a surprising trend: for visual reasoning tasks theinference-optimal behavior in VLMs i.e. minimum downstream error at any givenfixed inference compute is achieved when using the largest LLM that fitswithin the inference budget while minimizing visual token count - often to asingle token. While the token reduction literature has mainly focused onmaintaining base model performance by modestly reducing the token count e.g.5-10times our results indicate that the compute-optimal inference regimerequires operating under even higher token compression ratios. Based on theseinsights we take some initial steps towards building approaches tailored forhigh token compression settings. Code is available athttps://github.com/locuslab/llava-token-compression. |


| Item |Content|
| --- |---|
|idx| 2411.03286v1 |
|title| DiT4Edit: Diffusion Transformer for Image Editing |
|authors| Kunyu FengYue MaBingyuan WangChenyang QiHaozhe ChenQifeng ChenZeyu Wang
|links| http://arxiv.org/abs/2411.03286v1 |
|updated| 2024-11-05 17:35:41 UTC |
|summary| Despite recent advances in UNet-based image editing methods for shape-awareobject editing in high-resolution images are still lacking. Compared to UNetDiffusion Transformers DiT demonstrate superior capabilities to effectivelycapture the long-range dependencies among patches leading to higher-qualityimage generation. In this paper we propose DiT4Edit the first DiffusionTransformer-based image editing framework. Specifically DiT4Edit uses theDPM-Solver inversion algorithm to obtain the inverted latents reducing thenumber of steps compared to the DDIM inversion algorithm commonly used inUNet-based frameworks. Additionally we design unified attention control andpatches merging tailored for transformer computation streams. This integrationallows our framework to generate higher-quality edited images faster. Ourdesign leverages the advantages of DiT enabling it to surpass UNet structuresin image editing especially in high-resolution and arbitrary-size images.Extensive experiments demonstrate the strong performance of DiT4Edit acrossvarious editing scenarios highlighting the potential of Diffusion Transformersin supporting image editing. |


| Item |Content|
| --- |---|
|idx| 2411.03260v1 |
|title| ShadowMamba: State-Space Model with Boundary-Region Selective Scan for Shadow Removal |
|authors| Xiujin ZhuChee-Onn ChowJoon Huang Chuah
|links| http://arxiv.org/abs/2411.03260v1 |
|updated| 2024-11-05 16:59:06 UTC |
|summary| Image shadow removal is a typical low-level vision problem where thepresence of shadows leads to abrupt changes in brightness in certain regionsaffecting the accuracy of upstream tasks. Current shadow removal methods stillface challenges such as residual boundary artifacts and capturing featureinformation at shadow boundaries is crucial for removing shadows andeliminating residual boundary artifacts. Recently Mamba has achievedremarkable success in computer vision by globally modeling long-sequenceinformation with linear complexity. However when applied to image shadowremoval the original Mamba scanning method overlooks the semantic continuityof shadow boundaries as well as the continuity of semantics within the sameregion. Based on the unique characteristics of shadow images this paperproposes a novel selective scanning method called boundary-region selectivescanning. This method scans boundary regions shadow regions and non-shadowregions independently bringing pixels of the same region type closer togetherin the long sequence especially focusing on the local information at theboundaries which is crucial for shadow removal. This method combines withglobal scanning and channel scanning to jointly accomplish the shadow removal.We name our model ShadowMamba the first Mamba-based model for shadow removal.Extensive experimental results show that our method outperforms currentstate-of-the-art models across most metrics on multiple datasets. The code forShadowMamba is available at Code will be released upon acceptance. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2411.03263v1 |
|title| Proxy-informed Bayesian transfer learning with unknown sources |
|authors| Sabina J. SlomanJulien MartinelliSamuel Kaski
|links| http://arxiv.org/abs/2411.03263v1 |
|updated| 2024-11-05 17:02:29 UTC |
|summary| Generalization outside the scope of ones training data requires leveragingprior knowledge about the effects that transfer and the effects that dontbetween different data sources. Bayesian transfer learning is a principledparadigm for specifying this knowledge and refining it on the basis of datafrom the source training and target prediction tasks. We address thechallenging transfer learning setting where the learner i cannot fine-tune inthe target task and ii does not know which source data points correspond tothe same task i.e. the data sources are unknown. We propose a proxy-informedrobust method for probabilistic transfer learning PROMPT which provides aposterior predictive estimate tailored to the structure of the target taskwithout requiring the learner have access to any outcome information from thetarget task. Instead PROMPT relies on the availability of proxy information.PROMPT uses the same proxy information for two purposes: i estimation ofeffects specific to the target task and ii construction of a robustreweighting of the source data for estimation of effects that transfer betweentasks. We provide theoretical results on the effect of this reweighting on therisk of negative transfer and demonstrate application of PROMPT in twosynthetic settings. |


| Item |Content|
| --- |---|
|idx| 2411.03195v1 |
|title| Online Data Collection for Efficient Semiparametric Inference |
|authors| Shantanu GuptaZachary C. LiptonDavid Childers
|links| http://arxiv.org/abs/2411.03195v1 |
|updated| 2024-11-05 15:40:53 UTC |
|summary| While many works have studied statistical data fusion they typically assumethat the various datasets are given in advance. However in practiceestimation requires difficult data collection decisions like determining theavailable data sources their costs and how many samples to collect from eachsource. Moreover this process is often sequential because the data collectedat a given time can improve collection decisions in the future. In our setupgiven access to multiple data sources and budget constraints the agent mustsequentially decide which data source to query to efficiently estimate a targetparameter. We formalize this task using Online Moment Selection asemiparametric framework that applies to any parameter identified by a set ofmoment conditions. Interestingly the optimal budget allocation depends on theunknown true parameters. We present two online data collection policiesExplore-then-Commit and Explore-then-Greedy that use the parameter estimatesat a given time to optimally allocate the remaining budget in the future steps.We prove that both policies achieve zero regret assessed by asymptotic MSErelative to an oracle policy. We empirically validate our methods on bothsynthetic and real-world causal effect estimation tasks demonstrating that theonline data collection policies outperform their fixed counterparts. |


| Item |Content|
| --- |---|
|idx| 2411.03107v1 |
|title| Near-Optimal Dynamic Regret for Adversarial Linear Mixture MDPs |
|authors| Long-Fei LiPeng ZhaoZhi-Hua Zhou
|links| http://arxiv.org/abs/2411.03107v1 |
|updated| 2024-11-05 13:55:52 UTC |
|summary| We study episodic linear mixture MDPs with the unknown transition andadversarial rewards under full-information feedback employing dynamic regretas the performance measure. We start with in-depth analyses of the strengthsand limitations of the two most popular methods: occupancy-measure-based andpolicy-based methods. We observe that while the occupancy-measure-based methodis effective in addressing non-stationary environments it encountersdifficulties with the unknown transition. In contrast the policy-based methodcan deal with the unknown transition effectively but faces challenges inhandling non-stationary environments. Building on this we propose a novelalgorithm that combines the benefits of both methods. Specifically it employsi an occupancy-measure-based global optimization with a two-layer structureto handle non-stationary environments and ii a policy-based variance-awarevalue-targeted regression to tackle the unknown transition. We bridge these twoparts by a novel conversion. Our algorithm enjoys an widetildemathcalOdsqrtH3 K  sqrtHKH  barP_K dynamic regret where d is thefeature dimension H is the episode length K is the number of episodesbarP_K is the non-stationarity measure. We show it is minimax optimal upto logarithmic factors by establishing a matching lower bound. To the best ofour knowledge this is the first work that achieves near-optimal dynamic regretfor adversarial linear mixture MDPs with the unknown transition without priorknowledge of the non-stationarity measure. |


| Item |Content|
| --- |---|
|idx| 2411.03103v1 |
|title| Benign landscape for Burer-Monteiro factorizations of MaxCut-type semidefinite programs |
|authors| Faniriana Rakoto EndorIrène Waldspurger
|links| http://arxiv.org/abs/2411.03103v1 |
|updated| 2024-11-05 13:47:07 UTC |
|summary| We consider MaxCut-type semidefinite programs SDP which admit a low ranksolution. To numerically leverage the low rank hypothesis a standardalgorithmic approach is the Burer-Monteiro factorization which allows tosignificantly reduce the dimensionality of the problem at the cost of itsconvexity. We give a sharp condition on the conditioning of the Laplacianmatrix associated with the SDP under which any second-order critical point ofthe non-convex problem is a global minimizer. By applying our theorem weimprove on recent results about the correctness of the Burer-Monteiro approachon mathbbZ_2-synchronization problems. |


| Item |Content|
| --- |---|
|idx| 2411.03097v1 |
|title| Correlating Variational Autoencoders Natively For Multi-View Imputation |
|authors| Ella S. C. OrmeMarina EvangelouUlrich Paquet
|links| http://arxiv.org/abs/2411.03097v1 |
|updated| 2024-11-05 13:43:37 UTC |
|summary| Multi-view data from the same source often exhibit correlation. This ismirrored in correlation between the latent spaces of separate variationalautoencoders VAEs trained on each data-view. A multi-view VAE approach isproposed that incorporates a joint prior with a non-zero correlation structurebetween the latent spaces of the VAEs. By enforcing such correlation structuremore strongly correlated latent spaces are uncovered. Using conditionaldistributions to move between these latent spaces missing views can be imputedand used for downstream analysis. Learning this correlation structure involvesmaintaining validity of the prior distribution as well as a successfulparameterization that allows end-to-end learning. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2411.03295v1 |
|title| Examining Human-AI Collaboration for Co-Writing Constructive Comments Online |
|authors| Farhana ShahidMaximilian DittgenMor NaamanAditya Vashistha
|links| http://arxiv.org/abs/2411.03295v1 |
|updated| 2024-11-05 17:42:43 UTC |
|summary| This paper examines how large language models LLMs can help people writeconstructive comments in online debates on divisive social issues and whetherthe notions of constructiveness vary across cultures. Through controlledexperiments with 600 participants from India and the US who reviewed and wroteconstructive comments on online threads on Islamophobia and homophobia wefound potential misalignment in how LLMs and humans perceive constructivenessin online comments. While the LLM was more likely to view dialectical commentsas more constructive participants favored comments that emphasized logic andfacts more than the LLM did. Despite these differences participants ratedLLM-generated and human-AI co-written comments as significantly moreconstructive than those written independently by humans. Our analysis alsorevealed that LLM-generated and human-AI co-written comments exhibited morelinguistic features associated with constructiveness compared to human-writtencomments on divisive topics. When participants used LLMs to refine theircomments the resulting comments were longer more polite positive lesstoxic and more readable with added argumentative features that retained theoriginal intent but occasionally lost nuances. Based on these findings wediscuss ethical and design considerations in using LLMs to facilitateconstructive discourse online. |


| Item |Content|
| --- |---|
|idx| 2411.03292v1 |
|title| Interaction2Code: How Far Are We From Automatic Interactive Webpage Generation? |
|authors| Jingyu XiaoYuxuan WanYintong HuoZhiyao XuMichael R. Lyu
|links| http://arxiv.org/abs/2411.03292v1 |
|updated| 2024-11-05 17:40:03 UTC |
|summary| Converting webpage design into functional UI code is a critical step forbuilding websites which can be labor-intensive and time-consuming. To automatethis design-to-code transformation process various automated methods usinglearning-based networks and multi-modal large language models MLLMs have beenproposed. However these studies were merely evaluated on a narrow range ofstatic web pages and ignored dynamic interaction elements making them lesspractical for real-world website deployment.  To fill in the blank we present the first systematic investigation of MLLMsin generating interactive webpages. Specifically we first formulate theInteraction-to-Code task and build the Interaction2Code benchmark that contains97 unique web pages and 213 distinct interactions spanning 15 webpage typesand 30 interaction categories. We then conduct comprehensive experiments onthree state-of-the-art SOTA MLLMs using both automatic metrics and humanevaluations thereby summarizing six findings accordingly. Our experimentalresults highlight the limitations of MLLMs in generating fine-grainedinteractive features and managing interactions with complex transformations andsubtle visual modifications. We further analyze failure cases and theirunderlying causes identifying 10 common failure types and assessing theirseverity. Additionally our findings reveal three critical influencing factorsi.e. prompts visual saliency and textual descriptions that can enhance theinteraction generation performance of MLLMs. Based on these findings we elicitimplications for researchers and developers providing a foundation for futureadvancements in this field. Datasets and source code are available athttps://github.com/WebPAI/Interaction2Code. |


| Item |Content|
| --- |---|
|idx| 2411.03287v1 |
|title| The Future of Intelligent Healthcare: A Systematic Analysis and Discussion on the Integration and Impact of Robots Using Large Language Models for Healthcare |
|authors| Souren PashangpourGoldie Nejat
|links| http://dx.doi.org/10.3390/robotics13080112 |
|updated| 2024-11-05 17:36:32 UTC |
|summary| The potential use of large language models LLMs in healthcare robotics canhelp address the significant demand put on healthcare systems around the worldwith respect to an aging demographic and a shortage of healthcareprofessionals. Even though LLMs have already been integrated into medicine toassist both clinicians and patients the integration of LLMs within healthcarerobots has not yet been explored for clinical settings. In this perspectivepaper we investigate the groundbreaking developments in robotics and LLMs touniquely identify the needed system requirements for designing health specificLLM based robots in terms of multi modal communication through human robotinteractions HRIs semantic reasoning and task planning. Furthermore wediscuss the ethical issues open challenges and potential future researchdirections for this emerging innovative field. |


| Item |Content|
| --- |---|
|idx| 2411.03275v1 |
|title| Causal Responsibility Attribution for Human-AI Collaboration |
|authors| Yahang QiBernhard SchölkopfZhijing Jin
|links| http://arxiv.org/abs/2411.03275v1 |
|updated| 2024-11-05 17:17:45 UTC |
|summary| As Artificial Intelligence AI systems increasingly influencedecision-making across various fields the need to attribute responsibility forundesirable outcomes has become essential though complicated by the complexinterplay between humans and AI. Existing attribution methods based on actualcausality and Shapley values tend to disproportionately blame agents whocontribute more to an outcome and rely on real-world measures ofblameworthiness that may misalign with responsible AI standards. This paperpresents a causal framework using Structural Causal Models SCMs tosystematically attribute responsibility in human-AI systems measuring overallblameworthiness while employing counterfactual reasoning to account for agentsexpected epistemic levels. Two case studies illustrate the frameworksadaptability in diverse human-AI collaboration scenarios. |


| Item |Content|
| --- |---|
|idx| 2411.03243v1 |
|title| Guidelines para Desenvolvimento de Jogos Mobile Inclusivos |
|authors| Gabriela Panta ZorzoJoão Vitor Dall Agnol FernandesSoraia Raupp Musse
|links| http://arxiv.org/abs/2411.03243v1 |
|updated| 2024-11-05 16:39:46 UTC |
|summary| Games represent a significant part of modern culture which demonstrates theimportance of ensuring that everyone can participate and play in order to feelincluded in our society. However most digital games end up being inaccessibleto people with disabilities. Part of the problem when thinking about inclusivegame design is that there is no single solution for accessibility and whatworks well for one group may not work for another. This work proposes a set ofguidelines for the development of inclusive mobile games considering thewidespread use of smartphones by the population and the need to include peoplewith disabilities in the gaming culture. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2411.03284v1 |
|title| SMoA: Improving Multi-agent Large Language Models with Sparse Mixture-of-Agents |
|authors| Dawei LiZhen TanPeijia QianYifan LiKumar Satvik ChaudharyLijie HuJiayi Shen
|links| http://arxiv.org/abs/2411.03284v1 |
|updated| 2024-11-05 17:33:39 UTC |
|summary| While multi-agent systems have been shown to significantly enhance theperformance of Large Language Models LLMs across various tasks andapplications the dense interaction between scaling agents potentially hamperstheir efficiency and diversity. To address these challenges we drawinspiration from the sparse mixture-of-agents SMoE and propose a sparsemixture-of-agents SMoA framework to improve the efficiency and diversity ofmulti-agent LLMs. Unlike completely connected structures SMoA introduces novelResponse Selection and Early Stopping mechanisms to sparsify information flowsamong individual LLM agents striking a balance between performance andefficiency. Additionally inspired by the expert diversity principle in SMoEframeworks for workload balance between experts we assign distinct roledescriptions to each LLM agent fostering diverse and divergent thinking.Extensive experiments on reasoning alignment and fairness benchmarksdemonstrate that SMoA achieves performance comparable to traditionalmixture-of-agents approaches but with significantly lower computational costs.Further analysis reveals that SMoA is more stable has a greater capacity toscale and offers considerable potential through hyper-parameter optimization.Code and data will be available at: https://github.com/David-Li0406/SMoA. |


| Item |Content|
| --- |---|
|idx| 2411.02983v1 |
|title| Autonomous Decision Making for UAV Cooperative Pursuit-Evasion Game with Reinforcement Learning |
|authors| Yang ZhaoZidong NieKangsheng DongQinghua HuangXuelong Li
|links| http://arxiv.org/abs/2411.02983v1 |
|updated| 2024-11-05 10:45:30 UTC |
|summary| The application of intelligent decision-making in unmanned aerial vehicleUAV is increasing and with the development of UAV 1v1 pursuit-evasion gamemulti-UAV cooperative game has emerged as a new challenge. This paper proposesa deep reinforcement learning-based model for decision-making in multi-role UAVcooperative pursuit-evasion game to address the challenge of enabling UAV toautonomously make decisions in complex game environments. In order to enhancethe training efficiency of the reinforcement learning algorithm in UAVpursuit-evasion game environment that has high-dimensional state-action spacethis paper proposes multi-environment asynchronous double deep Q-network withpriority experience replay algorithm to effectively train the UAVs gamepolicy. Furthermore aiming to improve cooperation ability and task completionefficiency as well as minimize the cost of UAVs in the pursuit-evasion gamethis paper focuses on the allocation of roles and targets within multi-UAVenvironment. The cooperative game decision model with varying numbers of UAVsare obtained by assigning diverse tasks and roles to the UAVs in differentscenarios. The simulation results demonstrate that the proposed method enablesautonomous decision-making of the UAVs in pursuit-evasion game scenarios andexhibits significant capabilities in cooperation. |


| Item |Content|
| --- |---|
|idx| 2411.02820v1 |
|title| DroidSpeak: Enhancing Cross-LLM Communication |
|authors| Yuhan LiuEsha ChoukseShan LuJunchen JiangMadan Musuvathi
|links| http://arxiv.org/abs/2411.02820v1 |
|updated| 2024-11-05 05:41:41 UTC |
|summary| In multi-agent systems utilizing Large Language Models LLMs communicationbetween agents traditionally relies on natural language. This communicationoften includes the full context of the query so far which can introducesignificant prefill-phase latency especially with long contexts.  We introduce DroidSpeak a novel framework to target this cross-LLMcommunication by leveraging the reuse of intermediate data such as inputembeddings E-cache and key-value caches KV-cache. We efficiently bypass theneed to reprocess entire contexts for fine-tuned versions of the samefoundational model. This approach allows faster context integration whilemaintaining the quality of task performance. Experimental evaluationsdemonstrate DroidSpeaks ability to significantly accelerate inter-agentcommunication achieving up to a 2.78x speedup in prefill latency withnegligible loss in accuracy. Our findings underscore the potential to createmore efficient and scalable multi-agent systems. |


| Item |Content|
| --- |---|
|idx| 2411.02584v1 |
|title| Multi-Agent Decision Transformers for Dynamic Dispatching in Material Handling Systems Leveraging Enterprise Big Data |
|authors| Xian Yeow LeeHaiyan WangDaisuke KatsumataTakaharu MatsuiChetan Gupta
|links| http://arxiv.org/abs/2411.02584v1 |
|updated| 2024-11-04 20:26:33 UTC |
|summary| Dynamic dispatching rules that allocate resources to tasks in real-time playa critical role in ensuring efficient operations of many automated materialhandling systems across industries. Traditionally the dispatching rulesdeployed are typically the result of manually crafted heuristics based ondomain experts knowledge. Generating these rules is time-consuming and oftensub-optimal. As enterprises increasingly accumulate vast amounts of operationaldata there is significant potential to leverage this big data to enhance theperformance of automated systems. One promising approach is to use DecisionTransformers which can be trained on existing enterprise data to learn betterdynamic dispatching rules for improving system throughput. In this work westudy the application of Decision Transformers as dynamic dispatching policieswithin an actual multi-agent material handling system and identify scenarioswhere enterprises can effectively leverage Decision Transformers on existingbig data to gain business value. Our empirical results demonstrate thatDecision Transformers can improve the material handling systems throughput bya considerable amount when the heuristic originally used in the enterprise dataexhibits moderate performance and involves no randomness. When the originalheuristic has strong performance Decision Transformers can still improve thethroughput but with a smaller improvement margin. However when the originalheuristics contain an element of randomness or when the performance of thedataset is below a certain threshold Decision Transformers fail to outperformthe original heuristic. These results highlight both the potential andlimitations of Decision Transformers as dispatching policies for automatedindustrial material handling systems. |


| Item |Content|
| --- |---|
|idx| 2411.02524v1 |
|title| SPACE: 3D Spatial Co-operation and Exploration Framework for Robust Mapping and Coverage with Multi-Robot Systems |
|authors| Sai Krishna GhantaRamviyas Parasuraman
|links| http://arxiv.org/abs/2411.02524v1 |
|updated| 2024-11-04 19:04:09 UTC |
|summary| In indoor environments multi-robot visual RGB-D mapping and explorationhold immense potential for application in domains such as domestic service andlogistics where deploying multiple robots in the same environment cansignificantly enhance efficiency. However there are two primary challenges:1 the ghosting trail effect which occurs due to overlapping views ofrobots impacting the accuracy and quality of point cloud reconstruction and2 the oversight of visual reconstructions in selecting the most effectivefrontiers for exploration. Given these challenges are interrelated we addressthem together by proposing a new semi-distributed framework SPACE for spatialcooperation in indoor environments that enables enhanced coverage and 3Dmapping. SPACE leverages geometric techniques including mutual awareness anda dynamic robot filter to overcome spatial mapping constraints.Additionally we introduce a novel spatial frontier detection system and mapmerger integrated with an adaptive frontier assigner for optimal coveragebalancing the exploration and reconstruction objectives. In extensiveROS-Gazebo simulations SPACE demonstrated superior performance overstate-of-the-art approaches in both exploration and mapping metrics. |


