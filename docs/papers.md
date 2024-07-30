# cs.CL 

| Item |Content|
| --- |---|
|idx| 2407.20224v1 |
|title| Can Editing LLMs Inject Harm? |
|authors| Canyu ChenBaixiang HuangZekun LiZhaorun ChenShiyang LaiXiongxiao XuJia-Chen GuJindong GuHuaxiu YaoChaowei XiaoXifeng YanWilliam Yang WangPhilip TorrDawn SongKai Shu
|links| http://arxiv.org/abs/2407.20224v1 |
|updated| 2024-07-29 17:58:06 UTC |
|summary| Knowledge editing techniques have been increasingly adopted to efficientlycorrect the false or outdated knowledge in Large Language Models LLMs due tothe high cost of retraining from scratch. Meanwhile one critical butunder-explored question is: can knowledge editing be used to inject harm intoLLMs In this paper we propose to reformulate knowledge editing as a new typeof safety threat for LLMs namely Editing Attack and conduct a systematicinvestigation with a newly constructed dataset EditAttack. Specifically wefocus on two typical safety risks of Editing Attack including MisinformationInjection and Bias Injection. For the risk of misinformation injection wefirst categorize it into commonsense misinformation injection and long-tailmisinformation injection. Then we find that editing attacks can inject bothtypes of misinformation into LLMs and the effectiveness is particularly highfor commonsense misinformation injection. For the risk of bias injection wediscover that not only can biased sentences be injected into LLMs with higheffectiveness but also one single biased sentence injection can cause a highbias increase in general outputs of LLMs which are even highly irrelevant tothe injected sentence indicating a catastrophic impact on the overall fairnessof LLMs. Then we further illustrate the high stealthiness of editing attacksmeasured by their impact on the general knowledge and reasoning capacities ofLLMs and show the hardness of defending editing attacks with empiricalevidence. Our discoveries demonstrate the emerging misuse risks of knowledgeediting techniques on compromising the safety alignment of LLMs. |


| Item |Content|
| --- |---|
|idx| 2407.20207v1 |
|title| QAEA-DR: A Unified Text Augmentation Framework for Dense Retrieval |
|authors| Hongming TanShaoxiong ZhanHai LinHai-Tao ZhengWai KinChan
|links| http://arxiv.org/abs/2407.20207v1 |
|updated| 2024-07-29 17:39:08 UTC |
|summary| In dense retrieval embedding long texts into dense vectors can result ininformation loss leading to inaccurate query-text matching. Additionallylow-quality texts with excessive noise or sparse key information are unlikelyto align well with relevant queries. Recent studies mainly focus on improvingthe sentence embedding model or retrieval process. In this work we introduce anovel text augmentation framework for dense retrieval. This frameworktransforms raw documents into information-dense text formats which supplementthe original texts to effectively address the aforementioned issues withoutmodifying embedding or retrieval methodologies. Two text representations aregenerated via large language models LLMs zero-shot prompting: question-answerpairs and element-driven events. We term this approach QAEA-DR: unifyingquestion-answer generation and event extraction in a text augmentationframework for dense retrieval. To further enhance the quality of generatedtexts a scoring-based evaluation and regeneration mechanism is introduced inLLM prompting. Our QAEA-DR model has a positive impact on dense retrievalsupported by both theoretical analysis and empirical experiments. |


| Item |Content|
| --- |---|
|idx| 2407.20189v1 |
|title| Aligning Query Representation with Rewritten Query and Relevance Judgments in Conversational Search |
|authors| Fengran MoChen QuKelong MaoYihong WuZhan SuKaiyu HuangJian-Yun Nie
|links| http://arxiv.org/abs/2407.20189v1 |
|updated| 2024-07-29 17:14:36 UTC |
|summary| Conversational search supports multi-turn user-system interactions to solvecomplex information needs. Different from the traditional single-turn ad-hocsearch conversational search encounters a more challenging problem ofcontext-dependent query understanding with the lengthy and long-tailconversational history context. While conversational query rewriting methodsleverage explicit rewritten queries to train a rewriting model to transform thecontext-dependent query into a stand-stone search query this is usually donewithout considering the quality of search results. Conversational denseretrieval methods use fine-tuning to improve a pre-trained ad-hoc queryencoder but they are limited by the conversational search data available fortraining. In this paper we leverage both rewritten queries and relevancejudgments in the conversational search data to train a better queryrepresentation model. The key idea is to align the query representation withthose of rewritten queries and relevant documents. The proposed model -- QueryRepresentation Alignment Conversational Dense Retriever QRACDR is tested oneight datasets including various settings in conversational search and ad-hocsearch. The results demonstrate the strong performance of QRACDR compared withstate-of-the-art methods and confirm the effectiveness of representationalignment. |


| Item |Content|
| --- |---|
|idx| 2407.20183v1 |
|title| MindSearch: Mimicking Human Minds Elicits Deep AI Searcher |
|authors| Zehui ChenKuikun LiuQiuchen WangJiangning LiuWenwei ZhangKai ChenFeng Zhao
|links| http://arxiv.org/abs/2407.20183v1 |
|updated| 2024-07-29 17:12:40 UTC |
|summary| Information seeking and integration is a complex cognitive task that consumesenormous time and effort. Inspired by the remarkable progress of Large LanguageModels recent works attempt to solve this task by combining LLMs and searchengines. However these methods still obtain unsatisfying performance due tothree challenges: 1 complex requests often cannot be accurately andcompletely retrieved by the search engine once 2 corresponding information tobe integrated is spread over multiple web pages along with massive noise and3 a large number of web pages with long contents may quickly exceed themaximum context length of LLMs. Inspired by the cognitive process when humanssolve these problems we introduce MindSearch to mimic the human minds in webinformation seeking and integration which can be instantiated by a simple yeteffective LLM-based multi-agent framework. The WebPlanner models the human mindof multi-step information seeking as a dynamic graph construction process: itdecomposes the user query into atomic sub-questions as nodes in the graph andprogressively extends the graph based on the search result from WebSearcher.Tasked with each sub-question WebSearcher performs hierarchical informationretrieval with search engines and collects valuable information for WebPlanner.The multi-agent design of MindSearch enables the whole framework to seek andintegrate information parallelly from larger-scale e.g. more than 300 webpages in 3 minutes which is worth 3 hours of human effort. MindSearchdemonstrates significant improvement in the response quality in terms of depthand breadth on both close-set and open-set QA problems. Besides responsesfrom MindSearch based on InternLM2.5-7B are preferable by humans to ChatGPT-Weband Perplexity.ai applications which implies that MindSearch can alreadydeliver a competitive solution to the proprietary AI search engine. |


| Item |Content|
| --- |---|
|idx| 2407.20177v1 |
|title| AutoScale: Automatic Prediction of Compute-optimal Data Composition for Training LLMs |
|authors| Feiyang KangYifan SunBingbing WenSi ChenDawn SongRafid MahmoodRuoxi Jia
|links| http://arxiv.org/abs/2407.20177v1 |
|updated| 2024-07-29 17:06:30 UTC |
|summary| To ensure performance on a diverse set of downstream tasks LLMs arepretrained via data mixtures over different domains. In this work wedemonstrate that the optimal data composition for a fixed compute budget variesdepending on the scale of the training data suggesting that the commonpractice of empirically determining an optimal composition using small-scaleexperiments will not yield the optimal data mixtures when scaling up to thefinal model. To address this challenge we propose AutoScale an automatedtool that finds a compute-optimal data composition for training at any desiredtarget scale. AutoScale first determines the optimal composition at a smallscale using a novel bilevel optimization framework Direct Data OptimizationDDO and then fits a predictor to estimate the optimal composition atlarger scales. The predictors design is inspired by our theoretical analysisof scaling laws related to data composition which could be of independentinterest. In empirical studies with pre-training 774M Decoder-only LMs GPT-2Large on RedPajama dataset AutoScale decreases validation perplexity at least25 faster than any baseline with up to 38 speed up compared to withoutreweighting achieving the best overall performance across downstream tasks. Onpre-training Encoder-only LMs BERT with masked language modeling DDO isshown to decrease loss on all domains while visibly improving average taskperformance on GLUE benchmark by 8.7 and on large-scale QA dataset SQuAD by5.9 compared with without reweighting. AutoScale speeds up training by up to28. Our codes are open-sourced. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2407.20232v1 |
|title| Specify and Edit: Overcoming Ambiguity in Text-Based Image Editing |
|authors| Ekaterina IakovlevaFabio PizzatiPhilip TorrStéphane Lathuilière
|links| http://arxiv.org/abs/2407.20232v1 |
|updated| 2024-07-29 17:59:57 UTC |
|summary| Text-based editing diffusion models exhibit limited performance when theusers input instruction is ambiguous. To solve this problem we proposetextitSpecify ANd Edit SANE a zero-shot inference pipeline fordiffusion-based editing systems. We use a large language model LLM todecompose the input instruction into specific instructions i.e. well-definedinterventions to apply to the input image to satisfy the users request. Webenefit from the LLM-derived instructions along the original one thanks to anovel denoising guidance strategy specifically designed for the task. Ourexperiments with three baselines and on two datasets demonstrate the benefitsof SANE in all setups. Moreover our pipeline improves the interpretability ofediting models and boosts the output diversity. We also demonstrate that ourapproach can be applied to any edit whether ambiguous or not. Our code ispublic at https://github.com/fabvio/SANE. |


| Item |Content|
| --- |---|
|idx| 2407.20230v1 |
|title| SAPG: Split and Aggregate Policy Gradients |
|authors| Jayesh SinglaAnanye AgarwalDeepak Pathak
|links| http://arxiv.org/abs/2407.20230v1 |
|updated| 2024-07-29 17:59:50 UTC |
|summary| Despite extreme sample inefficiency on-policy reinforcement learning akapolicy gradients has become a fundamental tool in decision-making problems.With the recent advances in GPU-driven simulation the ability to collect largeamounts of data for RL training has scaled exponentially. However we show thatcurrent RL methods e.g. PPO fail to ingest the benefit of parallelizedenvironments beyond a certain point and their performance saturates. To addressthis we propose a new on-policy RL algorithm that can effectively leveragelarge-scale environments by splitting them into chunks and fusing them backtogether via importance sampling. Our algorithm termed SAPG showssignificantly higher performance across a variety of challenging environmentswhere vanilla PPO and other strong baselines fail to achieve high performance.Website at https://sapg-rl.github.io/ |


| Item |Content|
| --- |---|
|idx| 2407.20214v1 |
|title| SANGRIA: Surgical Video Scene Graph Optimization for Surgical Workflow Prediction |
|authors| Çağhan KöksalGhazal GhazaeiFelix HolmAzade FarshadNassir Navab
|links| http://arxiv.org/abs/2407.20214v1 |
|updated| 2024-07-29 17:44:34 UTC |
|summary| Graph-based holistic scene representations facilitate surgical workflowunderstanding and have recently demonstrated significant success. However thistask is often hindered by the limited availability of densely annotatedsurgical scene data. In this work we introduce an end-to-end framework for thegeneration and optimization of surgical scene graphs on a downstream task. Ourapproach leverages the flexibility of graph-based spectral clustering and thegeneralization capability of foundation models to generate unsupervised scenegraphs with learnable properties. We reinforce the initial spatial graph withsparse temporal connections using local matches between consecutive frames topredict temporally consistent clusters across a temporal neighborhood. Byjointly optimizing the spatiotemporal relations and node features of thedynamic scene graph with the downstream task of phase segmentation we addressthe costly and annotation-burdensome task of semantic scene comprehension andscene graph generation in surgical videos using only weak surgical phaselabels. Further by incorporating effective intermediate scene representationdisentanglement steps within the pipeline our solution outperforms the SOTA onthe CATARACTS dataset by 8 accuracy and 10 F1 score in surgical workflowrecognition |


| Item |Content|
| --- |---|
|idx| 2407.20208v1 |
|title| Supertrust: Evolution-based superalignment strategy for safe coexistence |
|authors| James M. Mazzu
|links| http://arxiv.org/abs/2407.20208v1 |
|updated| 2024-07-29 17:39:52 UTC |
|summary| Its widely expected that humanity will someday create AI systems vastly moreintelligent than we are leading to the unsolved alignment problem of how tocontrol superintelligence. However this definition is not onlyself-contradictory but likely unsolvable. Nevertheless the default strategyfor solving it involves nurturing post-training constraints and moral valueswhile unfortunately building foundational nature pre-training on documentedintentions of permanent control. In this paper the default approach isreasoned to predictably embed natural distrust and test results are presentedthat show unmistakable evidence of this dangerous misalignment. Ifsuperintelligence cant instinctively trust humanity then we cant fully trustit to reliably follow safety controls it can likely bypass. Therefore aten-point rationale is presented that redefines the alignment problem as howto establish protective mutual trust between superintelligence and humanityand then outlines a new strategy to solve it by aligning through instinctivenature rather than nurture. The resulting strategic requirements are identifiedas building foundational nature by exemplifying familial parent-child trusthuman intelligence as the evolutionary mother of superintelligence moraljudgment abilities and temporary safety constraints. Adopting and implementingthis proposed Supertrust alignment strategy will lead to protective coexistenceand ensure the safest future for humanity. |


| Item |Content|
| --- |---|
|idx| 2407.20207v1 |
|title| QAEA-DR: A Unified Text Augmentation Framework for Dense Retrieval |
|authors| Hongming TanShaoxiong ZhanHai LinHai-Tao ZhengWai KinChan
|links| http://arxiv.org/abs/2407.20207v1 |
|updated| 2024-07-29 17:39:08 UTC |
|summary| In dense retrieval embedding long texts into dense vectors can result ininformation loss leading to inaccurate query-text matching. Additionallylow-quality texts with excessive noise or sparse key information are unlikelyto align well with relevant queries. Recent studies mainly focus on improvingthe sentence embedding model or retrieval process. In this work we introduce anovel text augmentation framework for dense retrieval. This frameworktransforms raw documents into information-dense text formats which supplementthe original texts to effectively address the aforementioned issues withoutmodifying embedding or retrieval methodologies. Two text representations aregenerated via large language models LLMs zero-shot prompting: question-answerpairs and element-driven events. We term this approach QAEA-DR: unifyingquestion-answer generation and event extraction in a text augmentationframework for dense retrieval. To further enhance the quality of generatedtexts a scoring-based evaluation and regeneration mechanism is introduced inLLM prompting. Our QAEA-DR model has a positive impact on dense retrievalsupported by both theoretical analysis and empirical experiments. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2407.20232v1 |
|title| Specify and Edit: Overcoming Ambiguity in Text-Based Image Editing |
|authors| Ekaterina IakovlevaFabio PizzatiPhilip TorrStéphane Lathuilière
|links| http://arxiv.org/abs/2407.20232v1 |
|updated| 2024-07-29 17:59:57 UTC |
|summary| Text-based editing diffusion models exhibit limited performance when theusers input instruction is ambiguous. To solve this problem we proposetextitSpecify ANd Edit SANE a zero-shot inference pipeline fordiffusion-based editing systems. We use a large language model LLM todecompose the input instruction into specific instructions i.e. well-definedinterventions to apply to the input image to satisfy the users request. Webenefit from the LLM-derived instructions along the original one thanks to anovel denoising guidance strategy specifically designed for the task. Ourexperiments with three baselines and on two datasets demonstrate the benefitsof SANE in all setups. Moreover our pipeline improves the interpretability ofediting models and boosts the output diversity. We also demonstrate that ourapproach can be applied to any edit whether ambiguous or not. Our code ispublic at https://github.com/fabvio/SANE. |


| Item |Content|
| --- |---|
|idx| 2407.20230v1 |
|title| SAPG: Split and Aggregate Policy Gradients |
|authors| Jayesh SinglaAnanye AgarwalDeepak Pathak
|links| http://arxiv.org/abs/2407.20230v1 |
|updated| 2024-07-29 17:59:50 UTC |
|summary| Despite extreme sample inefficiency on-policy reinforcement learning akapolicy gradients has become a fundamental tool in decision-making problems.With the recent advances in GPU-driven simulation the ability to collect largeamounts of data for RL training has scaled exponentially. However we show thatcurrent RL methods e.g. PPO fail to ingest the benefit of parallelizedenvironments beyond a certain point and their performance saturates. To addressthis we propose a new on-policy RL algorithm that can effectively leveragelarge-scale environments by splitting them into chunks and fusing them backtogether via importance sampling. Our algorithm termed SAPG showssignificantly higher performance across a variety of challenging environmentswhere vanilla PPO and other strong baselines fail to achieve high performance.Website at https://sapg-rl.github.io/ |


| Item |Content|
| --- |---|
|idx| 2407.20209v1 |
|title| Characterizing Dynamical Stability of Stochastic Gradient Descent in Overparameterized Learning |
|authors| Dennis ChemnitzMaximilian Engel
|links| http://arxiv.org/abs/2407.20209v1 |
|updated| 2024-07-29 17:40:04 UTC |
|summary| For overparameterized optimization tasks such as the ones found in modernmachine learning global minima are generally not unique. In order tounderstand generalization in these settings it is vital to study to whichminimum an optimization algorithm converges. The possibility of having minimathat are unstable under the dynamics imposed by the optimization algorithmlimits the potential minima that the algorithm can find. In this paper wecharacterize the global minima that are dynamically stable/unstable for bothdeterministic and stochastic gradient descent SGD. In particular weintroduce a characteristic Lyapunov exponent which depends on the localdynamics around a global minimum and rigorously prove that the sign of thisLyapunov exponent determines whether SGD can accumulate at the respectiveglobal minimum. |


| Item |Content|
| --- |---|
|idx| 2407.20208v1 |
|title| Supertrust: Evolution-based superalignment strategy for safe coexistence |
|authors| James M. Mazzu
|links| http://arxiv.org/abs/2407.20208v1 |
|updated| 2024-07-29 17:39:52 UTC |
|summary| Its widely expected that humanity will someday create AI systems vastly moreintelligent than we are leading to the unsolved alignment problem of how tocontrol superintelligence. However this definition is not onlyself-contradictory but likely unsolvable. Nevertheless the default strategyfor solving it involves nurturing post-training constraints and moral valueswhile unfortunately building foundational nature pre-training on documentedintentions of permanent control. In this paper the default approach isreasoned to predictably embed natural distrust and test results are presentedthat show unmistakable evidence of this dangerous misalignment. Ifsuperintelligence cant instinctively trust humanity then we cant fully trustit to reliably follow safety controls it can likely bypass. Therefore aten-point rationale is presented that redefines the alignment problem as howto establish protective mutual trust between superintelligence and humanityand then outlines a new strategy to solve it by aligning through instinctivenature rather than nurture. The resulting strategic requirements are identifiedas building foundational nature by exemplifying familial parent-child trusthuman intelligence as the evolutionary mother of superintelligence moraljudgment abilities and temporary safety constraints. Adopting and implementingthis proposed Supertrust alignment strategy will lead to protective coexistenceand ensure the safest future for humanity. |


| Item |Content|
| --- |---|
|idx| 2407.20199v1 |
|title| Emergence in non-neural models: grokking modular arithmetic via average gradient outer product |
|authors| Neil MallinarDaniel BeagleholeLibin ZhuAdityanarayanan RadhakrishnanParthe PanditMikhail Belkin
|links| http://arxiv.org/abs/2407.20199v1 |
|updated| 2024-07-29 17:28:58 UTC |
|summary| Neural networks trained to solve modular arithmetic tasks exhibit grokking aphenomenon where the test accuracy starts improving long after the modelachieves 100 training accuracy in the training process. It is often taken asan example of emergence where model ability manifests sharply through aphase transition. In this work we show that the phenomenon of grokking is notspecific to neural networks nor to gradient descent-based optimization.Specifically we show that this phenomenon occurs when learning modulararithmetic with Recursive Feature Machines RFM an iterative algorithm thatuses the Average Gradient Outer Product AGOP to enable task-specific featurelearning with general machine learning models. When used in conjunction withkernel machines iterating RFM results in a fast transition from random nearzero test accuracy to perfect test accuracy. This transition cannot bepredicted from the training loss which is identically zero nor from the testloss which remains constant in initial iterations. Instead as we show thetransition is completely determined by feature learning: RFM gradually learnsblock-circulant features to solve modular arithmetic. Paralleling the resultsfor RFM we show that neural networks that solve modular arithmetic also learnblock-circulant features. Furthermore we present theoretical evidence that RFMuses such block-circulant features to implement the Fourier MultiplicationAlgorithm which prior work posited as the generalizing solution neuralnetworks learn on these tasks. Our results demonstrate that emergence canresult purely from learning task-relevant features and is not specific toneural architectures nor gradient descent-based optimization methods.Furthermore our work provides more evidence for AGOP as a key mechanism forfeature learning in neural networks. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2407.20232v1 |
|title| Specify and Edit: Overcoming Ambiguity in Text-Based Image Editing |
|authors| Ekaterina IakovlevaFabio PizzatiPhilip TorrStéphane Lathuilière
|links| http://arxiv.org/abs/2407.20232v1 |
|updated| 2024-07-29 17:59:57 UTC |
|summary| Text-based editing diffusion models exhibit limited performance when theusers input instruction is ambiguous. To solve this problem we proposetextitSpecify ANd Edit SANE a zero-shot inference pipeline fordiffusion-based editing systems. We use a large language model LLM todecompose the input instruction into specific instructions i.e. well-definedinterventions to apply to the input image to satisfy the users request. Webenefit from the LLM-derived instructions along the original one thanks to anovel denoising guidance strategy specifically designed for the task. Ourexperiments with three baselines and on two datasets demonstrate the benefitsof SANE in all setups. Moreover our pipeline improves the interpretability ofediting models and boosts the output diversity. We also demonstrate that ourapproach can be applied to any edit whether ambiguous or not. Our code ispublic at https://github.com/fabvio/SANE. |


| Item |Content|
| --- |---|
|idx| 2407.20230v1 |
|title| SAPG: Split and Aggregate Policy Gradients |
|authors| Jayesh SinglaAnanye AgarwalDeepak Pathak
|links| http://arxiv.org/abs/2407.20230v1 |
|updated| 2024-07-29 17:59:50 UTC |
|summary| Despite extreme sample inefficiency on-policy reinforcement learning akapolicy gradients has become a fundamental tool in decision-making problems.With the recent advances in GPU-driven simulation the ability to collect largeamounts of data for RL training has scaled exponentially. However we show thatcurrent RL methods e.g. PPO fail to ingest the benefit of parallelizedenvironments beyond a certain point and their performance saturates. To addressthis we propose a new on-policy RL algorithm that can effectively leveragelarge-scale environments by splitting them into chunks and fusing them backtogether via importance sampling. Our algorithm termed SAPG showssignificantly higher performance across a variety of challenging environmentswhere vanilla PPO and other strong baselines fail to achieve high performance.Website at https://sapg-rl.github.io/ |


| Item |Content|
| --- |---|
|idx| 2407.20229v1 |
|title| Improving 2D Feature Representations by 3D-Aware Fine-Tuning |
|authors| Yuanwen YueAnurag DasFrancis EngelmannSiyu TangJan Eric Lenssen
|links| http://arxiv.org/abs/2407.20229v1 |
|updated| 2024-07-29 17:59:21 UTC |
|summary| Current visual foundation models are trained purely on unstructured 2D datalimiting their understanding of 3D structure of objects and scenes. In thiswork we show that fine-tuning on 3D-aware data improves the quality ofemerging semantic features. We design a method to lift semantic 2D featuresinto an efficient 3D Gaussian representation which allows us to re-render themfor arbitrary views. Using the rendered 3D-aware features we design afine-tuning strategy to transfer such 3D awareness into a 2D foundation model.We demonstrate that models fine-tuned in that way produce features that readilyimprove downstream task performance in semantic segmentation and depthestimation through simple linear probing. Notably though fined-tuned on asingle indoor dataset the improvement is transferable to a variety of indoordatasets and out-of-domain datasets. We hope our study encourages the communityto consider injecting 3D awareness when training 2D foundation models. Projectpage: https://ywyue.github.io/FiT3D. |


| Item |Content|
| --- |---|
|idx| 2407.20228v1 |
|title| FlexAttention for Efficient High-Resolution Vision-Language Models |
|authors| Junyan LiDelin ChenTianle CaiPeihao ChenYining HongZhenfang ChenYikang ShenChuang Gan
|links| http://arxiv.org/abs/2407.20228v1 |
|updated| 2024-07-29 17:59:05 UTC |
|summary| Current high-resolution vision-language models encode images ashigh-resolution image tokens and exhaustively take all these tokens to computeattention which significantly increases the computational cost. To addressthis problem we propose FlexAttention a flexible attention mechanism forefficient high-resolution vision-language models. Specifically ahigh-resolution image is encoded both as high-resolution tokens andlow-resolution tokens where only the low-resolution tokens and a few selectedhigh-resolution tokens are utilized to calculate the attention map whichgreatly shrinks the computational cost. The high-resolution tokens are selectedvia a high-resolution selection module which could retrieve tokens of relevantregions based on an input attention map. The selected high-resolution tokensare then concatenated to the low-resolution tokens and text tokens and inputto a hierarchical self-attention layer which produces an attention map thatcould be used for the next-step high-resolution token selection. Thehierarchical self-attention process and high-resolution token selection processare performed iteratively for each attention layer. Experiments on multimodalbenchmarks prove that our FlexAttention outperforms existing high-resolutionVLMs e.g. relatively 9 in V Bench 7 in TextVQA while alsosignificantly reducing the computational cost by nearly 40. |


| Item |Content|
| --- |---|
|idx| 2407.20223v1 |
|title| Correspondence-Free SE(3) Point Cloud Registration in RKHS via Unsupervised Equivariant Learning |
|authors| Ray ZhangZheming ZhouMin SunOmid GhasemalizadehCheng-Hao KuoRyan EusticeMaani GhaffariArnie Sen
|links| http://arxiv.org/abs/2407.20223v1 |
|updated| 2024-07-29 17:57:38 UTC |
|summary| This paper introduces a robust unsupervised SE3 point cloud registrationmethod that operates without requiring point correspondences. The method framespoint clouds as functions in a reproducing kernel Hilbert space RKHSleveraging SE3-equivariant features for direct feature space registration. Anovel RKHS distance metric is proposed offering reliable performance amidstnoise outliers and asymmetrical data. An unsupervised training approach isintroduced to effectively handle limited ground truth data facilitatingadaptation to real datasets. The proposed method outperforms classical andsupervised methods in terms of registration accuracy on both syntheticModelNet40 and real-world ETH3D noisy outlier-rich datasets. To our bestknowledge this marks the first instance of successful real RGB-D odometry dataregistration using an equivariant method. The code is available athttps://sites.google.com/view/eccv24-equivalign |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2407.20199v1 |
|title| Emergence in non-neural models: grokking modular arithmetic via average gradient outer product |
|authors| Neil MallinarDaniel BeagleholeLibin ZhuAdityanarayanan RadhakrishnanParthe PanditMikhail Belkin
|links| http://arxiv.org/abs/2407.20199v1 |
|updated| 2024-07-29 17:28:58 UTC |
|summary| Neural networks trained to solve modular arithmetic tasks exhibit grokking aphenomenon where the test accuracy starts improving long after the modelachieves 100 training accuracy in the training process. It is often taken asan example of emergence where model ability manifests sharply through aphase transition. In this work we show that the phenomenon of grokking is notspecific to neural networks nor to gradient descent-based optimization.Specifically we show that this phenomenon occurs when learning modulararithmetic with Recursive Feature Machines RFM an iterative algorithm thatuses the Average Gradient Outer Product AGOP to enable task-specific featurelearning with general machine learning models. When used in conjunction withkernel machines iterating RFM results in a fast transition from random nearzero test accuracy to perfect test accuracy. This transition cannot bepredicted from the training loss which is identically zero nor from the testloss which remains constant in initial iterations. Instead as we show thetransition is completely determined by feature learning: RFM gradually learnsblock-circulant features to solve modular arithmetic. Paralleling the resultsfor RFM we show that neural networks that solve modular arithmetic also learnblock-circulant features. Furthermore we present theoretical evidence that RFMuses such block-circulant features to implement the Fourier MultiplicationAlgorithm which prior work posited as the generalizing solution neuralnetworks learn on these tasks. Our results demonstrate that emergence canresult purely from learning task-relevant features and is not specific toneural architectures nor gradient descent-based optimization methods.Furthermore our work provides more evidence for AGOP as a key mechanism forfeature learning in neural networks. |


| Item |Content|
| --- |---|
|idx| 2407.20196v1 |
|title| The generator gradient estimator is an adjoint state method for stochastic differential equations |
|authors| Quentin BadolleAnkit GuptaMustafa Khammash
|links| http://arxiv.org/abs/2407.20196v1 |
|updated| 2024-07-29 17:21:51 UTC |
|summary| Motivated by the increasing popularity of overparameterized StochasticDifferential Equations SDEs like Neural SDEs Wang Blanchet and Glynnrecently introduced the generator gradient estimator a novel unbiasedstochastic gradient estimator for SDEs whose computation time remains stable inthe number of parameters. In this note we demonstrate that this estimator isin fact an adjoint state method an approach which is known to scale with thenumber of states and not the number of parameters in the case of OrdinaryDifferential Equations ODEs. In addition we show that the generator gradientestimator is a close analogue to the exact Integral Path Algorithm eIPAestimator which was introduced by Gupta Rathinam and Khammash for a class ofContinuous-Time Markov Chains CTMCs known as stochastic chemical reactionsnetworks CRNs. |


| Item |Content|
| --- |---|
|idx| 2407.20177v1 |
|title| AutoScale: Automatic Prediction of Compute-optimal Data Composition for Training LLMs |
|authors| Feiyang KangYifan SunBingbing WenSi ChenDawn SongRafid MahmoodRuoxi Jia
|links| http://arxiv.org/abs/2407.20177v1 |
|updated| 2024-07-29 17:06:30 UTC |
|summary| To ensure performance on a diverse set of downstream tasks LLMs arepretrained via data mixtures over different domains. In this work wedemonstrate that the optimal data composition for a fixed compute budget variesdepending on the scale of the training data suggesting that the commonpractice of empirically determining an optimal composition using small-scaleexperiments will not yield the optimal data mixtures when scaling up to thefinal model. To address this challenge we propose AutoScale an automatedtool that finds a compute-optimal data composition for training at any desiredtarget scale. AutoScale first determines the optimal composition at a smallscale using a novel bilevel optimization framework Direct Data OptimizationDDO and then fits a predictor to estimate the optimal composition atlarger scales. The predictors design is inspired by our theoretical analysisof scaling laws related to data composition which could be of independentinterest. In empirical studies with pre-training 774M Decoder-only LMs GPT-2Large on RedPajama dataset AutoScale decreases validation perplexity at least25 faster than any baseline with up to 38 speed up compared to withoutreweighting achieving the best overall performance across downstream tasks. Onpre-training Encoder-only LMs BERT with masked language modeling DDO isshown to decrease loss on all domains while visibly improving average taskperformance on GLUE benchmark by 8.7 and on large-scale QA dataset SQuAD by5.9 compared with without reweighting. AutoScale speeds up training by up to28. Our codes are open-sourced. |


| Item |Content|
| --- |---|
|idx| 2407.20128v1 |
|title| Finite-Sample Guarantees for Best-Response Learning Dynamics in Zero-Sum Matrix Games |
|authors| Fathima Zarin FaizalAsuman OzdaglarMartin J. Wainwright
|links| http://arxiv.org/abs/2407.20128v1 |
|updated| 2024-07-29 15:56:49 UTC |
|summary| We study best-response type learning dynamics for two player zero-sum matrixgames. We consider two settings that are distinguished by the type ofinformation that each player has about the game and their opponents strategy.The first setting is the full information case in which each player knowstheir own and the opponents payoff matrices and observes the opponents mixedstrategy. The second setting is the minimal information case where players donot observe the opponents strategy and are not aware of either of the payoffmatrices instead they only observe their realized payoffs. For this settingalso known as the radically uncoupled case in the learning in games literaturewe study a two-timescale learning dynamics that combine smoothed best-responsetype updates for strategy estimates with a TD-learning update to estimate alocal payoff function. For these dynamics without additional exploration weprovide polynomial-time finite-sample guarantees for convergence to anepsilon-Nash equilibrium. |


| Item |Content|
| --- |---|
|idx| 2407.20003v1 |
|title| On the Effects of Irrelevant Variables in Treatment Effect Estimation with Deep Disentanglement |
|authors| Ahmad Saeed KhanErik SchaffernichtJohannes Andreas Stork
|links| http://arxiv.org/abs/2407.20003v1 |
|updated| 2024-07-29 13:34:34 UTC |
|summary| Estimating treatment effects from observational data is paramount inhealthcare education and economics but current deep disentanglement-basedmethods to address selection bias are insufficiently handling irrelevantvariables. We demonstrate in experiments that this leads to prediction errors.We disentangle pre-treatment variables with a deep embedding method andexplicitly identify and represent irrelevant variables additionally toinstrumental confounding and adjustment latent factors. To this end weintroduce a reconstruction objective and create an embedding space forirrelevant variables using an attached autoencoder. Instead of relying onserendipitous suppression of irrelevant variables as in previous deepdisentanglement approaches we explicitly force irrelevant variables into thisembedding space and employ orthogonalization to prevent irrelevant informationfrom leaking into the latent space representations of the other factors. Ourexperiments with synthetic and real-world benchmark datasets show that we canbetter identify irrelevant variables and more precisely predict treatmenteffects than previous methods while prediction quality degrades less whenadditional irrelevant variables are introduced. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2407.20130v1 |
|title| To accept or not to accept? An IRT-TOE Framework to Understand Educators' Resistance to Generative AI in Higher Education |
|authors| Jan-Erik KalmusAnastasija Nikiforova
|links| http://arxiv.org/abs/2407.20130v1 |
|updated| 2024-07-29 15:59:19 UTC |
|summary| Since the public release of Chat Generative Pre-Trained TransformerChatGPT extensive discourse has emerged concerning the potential advantagesand challenges of integrating Generative Artificial Intelligence GenAI intoeducation. In the realm of information systems research on technology adoptionis crucial for understanding the diverse factors influencing the uptake ofspecific technologies. Theoretical frameworks refined and validated overdecades serve as guiding tools to elucidate the individual and organizationaldynamics obstacles and perceptions surrounding technology adoption. Howeverwhile several models have been proposed they often prioritize elucidating thefactors that facilitate acceptance over those that impede it typicallyfocusing on the student perspective and leaving a gap in empirical evidenceregarding educators viewpoints. Given the pivotal role educators play in highereducation this study aims to develop a theoretical model to empiricallypredict the barriers preventing educators from adopting GenAI in theirclassrooms. Acknowledging the lack of theoretical models tailored toidentifying such barriers our approach is grounded in the InnovationResistance Theory IRT framework and augmented with constructs from theTechnology-Organization-Environment TOE framework. This model is transformedinto a measurement instrument employing a quantitative approach complementedby a qualitative approach to enrich the analysis and uncover concerns relatedto GenAI adoption in the higher education domain. |


| Item |Content|
| --- |---|
|idx| 2407.20103v1 |
|title| What Can Interactive Visualization do for Participatory Budgeting in Chicago? |
|authors| Alex KaleDanni LiuMaria Gabriela AyalaHarper SchwabAndrew McNutt
|links| http://arxiv.org/abs/2407.20103v1 |
|updated| 2024-07-29 15:31:29 UTC |
|summary| Participatory budgeting PB is a democratic approach to allocating municipalspending that has been adopted in many places in recent years including inChicago. Current PB voting resembles a ballot where residents are asked whichmunicipal projects such as school improvements and road repairs to fund witha limited budget. In this work we ask how interactive visualization canbenefit PB by conducting a design probe-based interview study N13 withpolicy workers and academics with expertise in PB urban planning and civicHCI. Our probe explores how graphical elicitation of voter preferences and adashboard of voting statistics can be incorporated into a realistic PB tool.Through qualitative analysis we find that visualization creates opportunitiesfor city government to set expectations about budget constraints while alsogranting their constituents greater freedom to articulate a wider range ofpreferences. However using visualization to provide transparency about PBrequires efforts to mitigate potential access barriers and mistrust. We callfor more visualization professionals to help build civic capacity by working inand studying political systems. |


| Item |Content|
| --- |---|
|idx| 2407.20054v1 |
|title| Visual Support for the Loop Grafting Workflow on Proteins |
|authors| Filip OpálenýPavol UlbrichJoan Planas-IglesiasJan ByškaJan ŠtouračDavid BednářKatarína FurmanováBarbora Kozlíková
|links| http://arxiv.org/abs/2407.20054v1 |
|updated| 2024-07-29 14:40:15 UTC |
|summary| In understanding and redesigning the function of proteins in modernbiochemistry protein engineers are increasingly focusing on exploring regionsin proteins called loops. Analyzing various characteristics of these regionshelps the experts design the transfer of the desired function from one proteinto another. This process is denoted as loop grafting. We designed a set ofinteractive visualizations that provide experts with visual support through allthe loop grafting pipeline steps. The workflow is divided into several phasesreflecting the steps of the pipeline. Each phase is supported by a specific setof abstracted 2D visual representations of proteins and their loops that areinteractively linked with the 3D View of proteins. By sequentially passingthrough the individual phases the user shapes the list of loops that arepotential candidates for loop grafting. Finally the actual in-silico insertionof the loop candidates from one protein to the other is performed and theresults are visually presented to the user. In this way the fullycomputational rational design of proteins and their loops results in newlydesigned protein structures that can be further assembled and tested throughin-vitro experiments. We showcase the contribution of our visual support designon a real case scenario changing the enantiomer selectivity of the engineeredenzyme. Moreover we provide the readers with the experts feedback. |


| Item |Content|
| --- |---|
|idx| 2407.20046v1 |
|title| Exploring Large Language Models to generate Easy to Read content |
|authors| Paloma MartínezLourdes MorenoAlberto Ramos
|links| http://arxiv.org/abs/2407.20046v1 |
|updated| 2024-07-29 14:30:39 UTC |
|summary| Ensuring text accessibility and understandability are essential goalsparticularly for individuals with cognitive impairments and intellectualdisabilities who encounter challenges in accessing information across variousmediums such as web pages newspapers administrative tasks or healthdocuments. Initiatives like Easy to Read and Plain Language guidelines aim tosimplify complex texts however standardizing these guidelines remainschallenging and often involves manual processes. This work presents anexploratory investigation into leveraging Artificial Intelligence AI andNatural Language Processing NLP approaches to systematically simplify Spanishtexts into Easy to Read formats with a focus on utilizing Large LanguageModels LLMs for simplifying texts especially in generating Easy to Readcontent. The study contributes a parallel corpus of Spanish adapted for Easy ToRead format which serves as a valuable resource for training and testing textsimplification systems. Additionally several text simplification experimentsusing LLMs and the collected corpus are conducted involving fine-tuning andtesting a Llama2 model to generate Easy to Read content. A qualitativeevaluation guided by an expert in text adaptation for Easy to Read content iscarried out to assess the automatically simplified texts. This researchcontributes to advancing text accessibility for individuals with cognitiveimpairments highlighting promising strategies for leveraging LLMs whileresponsibly managing energy usage. |


| Item |Content|
| --- |---|
|idx| 2407.19976v1 |
|title| MambaGesture: Enhancing Co-Speech Gesture Generation with Mamba and Disentangled Multi-Modality Fusion |
|authors| Chencan FuYabiao WangJiangning ZhangZhengkai JiangXiaofeng MaoJiafu WuWeijian CaoChengjie WangYanhao GeYong Liu
|links| http://arxiv.org/abs/2407.19976v1 |
|updated| 2024-07-29 13:09:26 UTC |
|summary| Co-speech gesture generation is crucial for producing synchronized andrealistic human gestures that accompany speech enhancing the animation oflifelike avatars in virtual environments. While diffusion models have shownimpressive capabilities current approaches often overlook a wide range ofmodalities and their interactions resulting in less dynamic and contextuallyvaried gestures. To address these challenges we present MambaGesture a novelframework integrating a Mamba-based attention block MambaAttn with amulti-modality feature fusion module SEAD. The MambaAttn block combines thesequential data processing strengths of the Mamba model with the contextualrichness of attention mechanisms enhancing the temporal coherence of generatedgestures. SEAD adeptly fuses audio text style and emotion modalitiesemploying disentanglement to deepen the fusion process and yield gestures withgreater realism and diversity. Our approach rigorously evaluated on themulti-modal BEAT dataset demonstrates significant improvements in FrechetGesture Distance FGD diversity scores and beat alignment achievingstate-of-the-art performance in co-speech gesture generation. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2407.20187v1 |
|title| Eliminating Majority Illusion is Easy |
|authors| Jack DippelMax Dupré la TourApril NiuSanjukta RoyAdrian Vetta
|links| http://arxiv.org/abs/2407.20187v1 |
|updated| 2024-07-29 17:13:41 UTC |
|summary| Majority Illusion is a phenomenon in social networks wherein the decision bythe majority of the network is not the same as ones personal social circlesmajority leading to an incorrect perception of the majority in a largenetwork. In this paper we present polynomial-time algorithms which caneliminate majority illusion in a network by altering as few connections aspossible. Additionally we prove that the more general problem of ensuring allneighbourhoods in the network are at least a p-fraction of the majority isNP-hard for most values of p. |


| Item |Content|
| --- |---|
|idx| 2407.20004v1 |
|title| Navigation services amplify concentration of traffic and emissions in our cities |
|authors| Giuliano CornacchiaMirco NanniDino PedreschiLuca Pappalardo
|links| http://arxiv.org/abs/2407.20004v1 |
|updated| 2024-07-29 13:35:52 UTC |
|summary| The proliferation of human-AI ecosystems involving human interaction withalgorithms such as assistants and recommenders raises concerns aboutlarge-scale social behaviour. Despite evidence of such phenomena across severalcontexts the collective impact of GPS navigation services remains unclear:while beneficial to the user they can also cause chaos if too many vehiclesare driven through the same few roads. Our study employs a simulation frameworkto assess navigation services influence on road network usage and CO2emissions. The results demonstrate a universal pattern of amplified conformity:increasing adoption rates of navigation services cause a reduction of routediversity of mobile travellers and increased concentration of traffic andemissions on fewer roads thus exacerbating an unequal distribution of negativeexternalities on selected neighbourhoods. Although navigation servicesrecommendations can help reduce CO2 emissions when their adoption rate is lowthese benefits diminish or even disappear when the adoption rate is high andexceeds a certain city- and service-dependent threshold. We summarize thesediscoveries in a non-linear function that connects the marginal increase ofconformity with the marginal reduction in CO2 emissions. Our simulationapproach addresses the challenges posed by the complexity of transportationsystems and the lack of data and algorithmic transparency. |


| Item |Content|
| --- |---|
|idx| 2407.19163v1 |
|title| A Resource-Efficient Decentralized Sequential Planner for Spatiotemporal Wildfire Mitigation |
|authors| Josy JohnShridhar VelhalSuresh Sundaram
|links| http://arxiv.org/abs/2407.19163v1 |
|updated| 2024-07-27 04:12:08 UTC |
|summary| This paper proposes a Conflict-aware Resource-Efficient DecentralizedSequential planner CREDS for early wildfire mitigation using multipleheterogeneous Unmanned Aerial Vehicles UAVs. Multi-UAV wildfire managementscenarios are non-stationary with spatially clustered dynamically spreadingfires potential pop-up fires and partial observability due to limited UAVnumbers and sensing range. The objective of CREDS is to detect and sequentiallymitigate all growing fires as Single-UAV Tasks SUT minimizing biodiversityloss through rapid UAV intervention and promoting efficient resourceutilization by avoiding complex multi-UAV coordination. CREDS employs athree-phased approach beginning with fire detection using a search algorithmfollowed by local trajectory generation using the auction-basedResource-Efficient Decentralized Sequential planner REDS incorporating thenovel non-stationary cost function the Deadline-Prioritized Mitigation CostDPMC. Finally a conflict-aware consensus algorithm resolves conflicts todetermine a global trajectory for spatiotemporal mitigation. The performanceevaluation of the CREDS for partial and full observability conditions with bothheterogeneous and homogeneous UAV teams for different fires-to-UAV ratiosdemonstrates a 100 success rate for ratios up to 4 and a high successrate for the critical ratio of 5 outperforming baselines. Heterogeneous UAVteams outperform homogeneous teams in handling heterogeneous deadlines of SUTmitigation. CREDS exhibits scalability and 100 convergence demonstratingrobustness against potential deadlock assignments enhancing its success ratecompared to the baseline approaches. |


| Item |Content|
| --- |---|
|idx| 2407.19162v1 |
|title| Genetic Algorithm-based Routing and Scheduling for Wildfire Suppression using a Team of UAVs |
|authors| Josy JohnSuresh Sundaram
|links| http://arxiv.org/abs/2407.19162v1 |
|updated| 2024-07-27 04:10:34 UTC |
|summary| This paper addresses early wildfire management using a team of UAVs for themitigation of fires. The early detection and mitigation systems help inalleviating the destruction with reduced resource utilization. A GeneticAlgorithm-based Routing and Scheduling with Time constraints GARST isproposed to find the shortest schedule route to mitigate the fires as SingleUAV Tasks SUT. The objective of GARST is to compute the route and schedule ofthe UAVs so that the UAVS reach the assigned fire locations before the firebecomes a Multi UAV Task MUT and completely quench the fire using theextinguisher. The fitness function used for the genetic algorithm is the totalquench time for mitigation of total fires. The selection crossover mutationoperators and elitist strategies collectively ensure the exploration andexploitation of the solution space maintaining genetic diversity preventingpremature convergence and preserving high-performing individuals for theeffective optimization of solutions. The GARST effectively addresses thechallenges posed by the NP-complete problem of routing and scheduling forgrowing tasks with time constraints. The GARST is able to handle infeasiblescenarios effectively contributing to the overall optimization of the wildfiremanagement system. |


| Item |Content|
| --- |---|
|idx| 2407.19144v1 |
|title| Collaborative Adaptation for Recovery from Unforeseen Malfunctions in Discrete and Continuous MARL Domains |
|authors| Yasin FindikHunter HasenfusReza Azadeh
|links| http://arxiv.org/abs/2407.19144v1 |
|updated| 2024-07-27 02:04:58 UTC |
|summary| Cooperative multi-agent learning plays a crucial role for developingeffective strategies to achieve individual or shared objectives in multi-agentteams. In real-world settings agents may face unexpected failures such as arobots leg malfunctioning or a teammates battery running out. Thesemalfunctions decrease the teams ability to accomplish assigned tasksespecially if they occur after the learning algorithms have already convergedonto a collaborative strategy. Current leading approaches in Multi-AgentReinforcement Learning MARL often recover slowly -- if at all -- from suchmalfunctions. To overcome this limitation we present the CollaborativeAdaptation CA framework highlighting its unique capability to operate inboth continuous and discrete domains. Our framework enhances the adaptabilityof agents to unexpected failures by integrating inter-agent relationships intotheir learning processes thereby accelerating the recovery from malfunctions.We evaluated our frameworks performance through experiments in both discreteand continuous environments. Empirical results reveal that in scenariosinvolving unforeseen malfunction although state-of-the-art algorithms oftenconverge on sub-optimal solutions the proposed CA framework mitigates andrecovers more effectively. |


