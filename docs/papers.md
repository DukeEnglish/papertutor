# cs.CL 

| Item |Content|
| --- |---|
|idx| 2402.17764v1 |
|title| The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits |
|authors| Shuming MaHongyu WangLingxiao MaLei WangWenhui WangShaohan HuangLi DongRuiping WangJilong XueFuru Wei
|links| http://arxiv.org/abs/2402.17764v1 |
|updated| 2024-02-27 18:56:19 UTC |
|summary| Recent research such as BitNet is paving the way for a new era of 1-bitLarge Language Models LLMs. In this work we introduce a 1-bit LLM variantnamely BitNet b1.58 in which every single parameter or weight of the LLM isternary -1 0 1. It matches the full-precision i.e. FP16 or BF16Transformer LLM with the same model size and training tokens in terms of bothperplexity and end-task performance while being significantly morecost-effective in terms of latency memory throughput and energy consumption.More profoundly the 1.58-bit LLM defines a new scaling law and recipe fortraining new generations of LLMs that are both high-performance andcost-effective. Furthermore it enables a new computation paradigm and opensthe door for designing specific hardware optimized for 1-bit LLMs. |


| Item |Content|
| --- |---|
|idx| 2402.17762v1 |
|title| Massive Activations in Large Language Models |
|authors| Mingjie SunXinlei ChenJ. Zico KolterZhuang Liu
|links| http://arxiv.org/abs/2402.17762v1 |
|updated| 2024-02-27 18:55:17 UTC |
|summary| We observe an empirical phenomenon in Large Language Models LLMs -- veryfew activations exhibit significantly larger values than others e.g. 100000times larger. We call them massive activations. First we demonstrate thewidespread existence of massive activations across various LLMs andcharacterize their locations. Second we find their values largely stayconstant regardless of the input and they function as indispensable bias termsin LLMs. Third these massive activations lead to the concentration ofattention probabilities to their corresponding tokens and further implicitbias terms in the self-attention output. Last we also study massiveactivations in Vision Transformers. |


| Item |Content|
| --- |---|
|idx| 2402.17759v1 |
|title| Towards Optimal Learning of Language Models |
|authors| Yuxian GuLi DongYaru HaoQingxiu DongMinlie HuangFuru Wei
|links| http://arxiv.org/abs/2402.17759v1 |
|updated| 2024-02-27 18:52:19 UTC |
|summary| This work studies the general principles of improving the learning oflanguage models LMs which aims at reducing the necessary training steps forachieving superior performance. Specifically we present a theory for theoptimal learning of LMs. We first propose an objective that optimizes LMlearning by maximizing the data compression ratio in anLM-training-as-lossless-compression view. Then we derive a theorem namedLearning Law to reveal the properties of the dynamics in the optimal learningprocess under our objective. The theorem is then validated by experiments on alinear classification and a real-world language modeling task. Finally weempirically verify that the optimal learning of LMs essentially stems from theimprovement of the coefficients in the scaling law of LMs indicating greatpromise and significance for designing practical learning acceleration methods.Our code can be found at https://aka.ms/LearningLaw. |


| Item |Content|
| --- |---|
|idx| 2402.17753v1 |
|title| Evaluating Very Long-Term Conversational Memory of LLM Agents |
|authors| Adyasha MaharanaDong-Ho LeeSergey TulyakovMohit BansalFrancesco BarbieriYuwei Fang
|links| http://arxiv.org/abs/2402.17753v1 |
|updated| 2024-02-27 18:42:31 UTC |
|summary| Existing works on long-term open-domain dialogues focus on evaluating modelresponses within contexts spanning no more than five chat sessions. Despiteadvancements in long-context large language models LLMs and retrievalaugmented generation RAG techniques their efficacy in very long-termdialogues remains unexplored. To address this research gap we introduce amachine-human pipeline to generate high-quality very long-term dialogues byleveraging LLM-based agent architectures and grounding their dialogues onpersonas and temporal event graphs. Moreover we equip each agent with thecapability of sharing and reacting to images. The generated conversations areverified and edited by human annotators for long-range consistency andgrounding to the event graphs. Using this pipeline we collect LoCoMo adataset of very long-term conversations each encompassing 300 turns and 9Ktokens on avg. over up to 35 sessions. Based on LoCoMo we present acomprehensive evaluation benchmark to measure long-term memory in modelsencompassing question answering event summarization and multi-modal dialoguegeneration tasks. Our experimental results indicate that LLMs exhibitchallenges in understanding lengthy conversations and comprehending long-rangetemporal and causal dynamics within dialogues. Employing strategies likelong-context LLMs or RAG can offer improvements but these models stillsubstantially lag behind human performance. |


| Item |Content|
| --- |---|
|idx| 2402.17733v1 |
|title| Tower: An Open Multilingual Large Language Model for Translation-Related Tasks |
|authors| Duarte M. AlvesJosé PombalNuno M. GuerreiroPedro H. MartinsJoão AlvesAmin FarajianBen PetersRicardo ReiPatrick FernandesSweta AgrawalPierre ColomboJosé G. C. de SouzaAndré F. T. Martins
|links| http://arxiv.org/abs/2402.17733v1 |
|updated| 2024-02-27 18:09:36 UTC |
|summary| While general-purpose large language models LLMs demonstrate proficiency onmultiple tasks within the domain of translation approaches based on open LLMsare competitive only when specializing on a single task. In this paper wepropose a recipe for tailoring LLMs to multiple tasks present in translationworkflows. We perform continued pretraining on a multilingual mixture ofmonolingual and parallel data creating TowerBase followed by finetuning oninstructions relevant for translation processes creating TowerInstruct. Ourfinal model surpasses open alternatives on several tasks relevant totranslation workflows and is competitive with general-purpose closed LLMs. Tofacilitate future research we release the Tower models our specializationdataset an evaluation framework for LLMs focusing on the translationecosystem and a collection of model generations including ours on ourbenchmark. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2402.17768v1 |
|title| Diffusion Meets DAgger: Supercharging Eye-in-hand Imitation Learning |
|authors| Xiaoyu ZhangMatthew ChangPranav KumarSaurabh Gupta
|links| http://arxiv.org/abs/2402.17768v1 |
|updated| 2024-02-27 18:59:18 UTC |
|summary| A common failure mode for policies trained with imitation is compoundingexecution errors at test time. When the learned policy encounters states thatwere not present in the expert demonstrations the policy fails leading todegenerate behavior. The Dataset Aggregation or DAgger approach to thisproblem simply collects more data to cover these failure states. However inpractice this is often prohibitively expensive. In this work we proposeDiffusion Meets DAgger DMD a method to reap the benefits of DAgger withoutthe cost for eye-in-hand imitation learning problems. Instead of collecting newsamples to cover out-of-distribution states DMD uses recent advances indiffusion models to create these samples with diffusion models. This leads torobust performance from few demonstrations. In experiments conducted fornon-prehensile pushing on a Franka Research 3 we show that DMD can achieve asuccess rate of 80 with as few as 8 expert demonstrations where naivebehavior cloning reaches only 20. DMD also outperform competing NeRF-basedaugmentation schemes by 50. |


| Item |Content|
| --- |---|
|idx| 2402.17767v1 |
|title| Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator |
|authors| Arjun GuptaMichelle ZhangRishik SathuaSaurabh Gupta
|links| http://arxiv.org/abs/2402.17767v1 |
|updated| 2024-02-27 18:58:54 UTC |
|summary| Pulling open cabinets and drawers presents many difficult technicalchallenges in perception inferring articulation parameters for objects fromonboard sensors planning producing motion plans that conform to tight taskconstraints and control making and maintaining contact while applying forceson the environment. In this work we build an end-to-end system that enables acommodity mobile manipulator Stretch RE2 to pull open cabinets and drawers indiverse previously unseen real world environments. We conduct 4 days of realworld testing of this system spanning 31 different objects from across 13different real world environments. Our system achieves a success rate of 61 onopening novel cabinets and drawers in unseen environments zero-shot. Ananalysis of the failure modes suggests that errors in perception are the mostsignificant challenge for our system. We will open source code and models forothers to replicate and build upon our system. |


| Item |Content|
| --- |---|
|idx| 2402.17760v1 |
|title| Learning to Program Variational Quantum Circuits with Fast Weights |
|authors| Samuel Yen-Chi Chen
|links| http://arxiv.org/abs/2402.17760v1 |
|updated| 2024-02-27 18:53:18 UTC |
|summary| Quantum Machine Learning QML has surfaced as a pioneering frameworkaddressing sequential control tasks and time-series modeling. It hasdemonstrated empirical quantum advantages notably within domains such asReinforcement Learning RL and time-series prediction. A significantadvancement lies in Quantum Recurrent Neural Networks QRNNs specificallytailored for memory-intensive tasks encompassing partially observableenvironments and non-linear time-series prediction. Nevertheless QRNN-basedmodels encounter challenges notably prolonged training duration stemming fromthe necessity to compute quantum gradients using backpropagation-through-timeBPTT. This predicament exacerbates when executing the complete model onquantum devices primarily due to the substantial demand for circuit evaluationarising from the parameter-shift rule. This paper introduces the Quantum FastWeight Programmers QFWP as a solution to the temporal or sequential learningchallenge. The QFWP leverages a classical neural network referred to as theslow programmer functioning as a quantum programmer to swiftly modify theparameters of a variational quantum circuit termed the fast programmer.Instead of completely overwriting the fast programmer at each time-step theslow programmer generates parameter changes or updates for the quantum circuitparameters. This approach enables the fast programmer to incorporate pastobservations or information. Notably the proposed QFWP model achieves learningof temporal dependencies without necessitating the use of quantum recurrentneural networks. Numerical simulations conducted in this study showcase theefficacy of the proposed QFWP model in both time-series prediction and RLtasks. The model exhibits performance levels either comparable to or surpassingthose achieved by QLSTM-based models. |


| Item |Content|
| --- |---|
|idx| 2402.17753v1 |
|title| Evaluating Very Long-Term Conversational Memory of LLM Agents |
|authors| Adyasha MaharanaDong-Ho LeeSergey TulyakovMohit BansalFrancesco BarbieriYuwei Fang
|links| http://arxiv.org/abs/2402.17753v1 |
|updated| 2024-02-27 18:42:31 UTC |
|summary| Existing works on long-term open-domain dialogues focus on evaluating modelresponses within contexts spanning no more than five chat sessions. Despiteadvancements in long-context large language models LLMs and retrievalaugmented generation RAG techniques their efficacy in very long-termdialogues remains unexplored. To address this research gap we introduce amachine-human pipeline to generate high-quality very long-term dialogues byleveraging LLM-based agent architectures and grounding their dialogues onpersonas and temporal event graphs. Moreover we equip each agent with thecapability of sharing and reacting to images. The generated conversations areverified and edited by human annotators for long-range consistency andgrounding to the event graphs. Using this pipeline we collect LoCoMo adataset of very long-term conversations each encompassing 300 turns and 9Ktokens on avg. over up to 35 sessions. Based on LoCoMo we present acomprehensive evaluation benchmark to measure long-term memory in modelsencompassing question answering event summarization and multi-modal dialoguegeneration tasks. Our experimental results indicate that LLMs exhibitchallenges in understanding lengthy conversations and comprehending long-rangetemporal and causal dynamics within dialogues. Employing strategies likelong-context LLMs or RAG can offer improvements but these models stillsubstantially lag behind human performance. |


| Item |Content|
| --- |---|
|idx| 2402.17747v1 |
|title| When Your AI Deceives You: Challenges with Partial Observability of Human Evaluators in Reward Learning |
|authors| Leon LangDavis FooteStuart RussellAnca DraganErik JennerScott Emmons
|links| http://arxiv.org/abs/2402.17747v1 |
|updated| 2024-02-27 18:32:11 UTC |
|summary| Past analyses of reinforcement learning from human feedback RLHF assumethat the human fully observes the environment. What happens when human feedbackis based only on partial observations We formally define two failure cases:deception and overjustification. Modeling the human as Boltzmann-rationalw.r.t. a belief over trajectories we prove conditions under which RLHF isguaranteed to result in policies that deceptively inflate their performanceoverjustify their behavior to make an impression or both. To help addressthese issues we mathematically characterize how partial observability of theenvironment translates into lack of ambiguity in the learned return function.In some cases accounting for partial observability makes it theoreticallypossible to recover the return function and thus the optimal policy while inother cases there is irreducible ambiguity. We caution against blindlyapplying RLHF in partially observable settings and propose research directionsto help tackle these challenges. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2402.17768v1 |
|title| Diffusion Meets DAgger: Supercharging Eye-in-hand Imitation Learning |
|authors| Xiaoyu ZhangMatthew ChangPranav KumarSaurabh Gupta
|links| http://arxiv.org/abs/2402.17768v1 |
|updated| 2024-02-27 18:59:18 UTC |
|summary| A common failure mode for policies trained with imitation is compoundingexecution errors at test time. When the learned policy encounters states thatwere not present in the expert demonstrations the policy fails leading todegenerate behavior. The Dataset Aggregation or DAgger approach to thisproblem simply collects more data to cover these failure states. However inpractice this is often prohibitively expensive. In this work we proposeDiffusion Meets DAgger DMD a method to reap the benefits of DAgger withoutthe cost for eye-in-hand imitation learning problems. Instead of collecting newsamples to cover out-of-distribution states DMD uses recent advances indiffusion models to create these samples with diffusion models. This leads torobust performance from few demonstrations. In experiments conducted fornon-prehensile pushing on a Franka Research 3 we show that DMD can achieve asuccess rate of 80 with as few as 8 expert demonstrations where naivebehavior cloning reaches only 20. DMD also outperform competing NeRF-basedaugmentation schemes by 50. |


| Item |Content|
| --- |---|
|idx| 2402.17767v1 |
|title| Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator |
|authors| Arjun GuptaMichelle ZhangRishik SathuaSaurabh Gupta
|links| http://arxiv.org/abs/2402.17767v1 |
|updated| 2024-02-27 18:58:54 UTC |
|summary| Pulling open cabinets and drawers presents many difficult technicalchallenges in perception inferring articulation parameters for objects fromonboard sensors planning producing motion plans that conform to tight taskconstraints and control making and maintaining contact while applying forceson the environment. In this work we build an end-to-end system that enables acommodity mobile manipulator Stretch RE2 to pull open cabinets and drawers indiverse previously unseen real world environments. We conduct 4 days of realworld testing of this system spanning 31 different objects from across 13different real world environments. Our system achieves a success rate of 61 onopening novel cabinets and drawers in unseen environments zero-shot. Ananalysis of the failure modes suggests that errors in perception are the mostsignificant challenge for our system. We will open source code and models forothers to replicate and build upon our system. |


| Item |Content|
| --- |---|
|idx| 2402.17764v1 |
|title| The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits |
|authors| Shuming MaHongyu WangLingxiao MaLei WangWenhui WangShaohan HuangLi DongRuiping WangJilong XueFuru Wei
|links| http://arxiv.org/abs/2402.17764v1 |
|updated| 2024-02-27 18:56:19 UTC |
|summary| Recent research such as BitNet is paving the way for a new era of 1-bitLarge Language Models LLMs. In this work we introduce a 1-bit LLM variantnamely BitNet b1.58 in which every single parameter or weight of the LLM isternary -1 0 1. It matches the full-precision i.e. FP16 or BF16Transformer LLM with the same model size and training tokens in terms of bothperplexity and end-task performance while being significantly morecost-effective in terms of latency memory throughput and energy consumption.More profoundly the 1.58-bit LLM defines a new scaling law and recipe fortraining new generations of LLMs that are both high-performance andcost-effective. Furthermore it enables a new computation paradigm and opensthe door for designing specific hardware optimized for 1-bit LLMs. |


| Item |Content|
| --- |---|
|idx| 2402.17762v1 |
|title| Massive Activations in Large Language Models |
|authors| Mingjie SunXinlei ChenJ. Zico KolterZhuang Liu
|links| http://arxiv.org/abs/2402.17762v1 |
|updated| 2024-02-27 18:55:17 UTC |
|summary| We observe an empirical phenomenon in Large Language Models LLMs -- veryfew activations exhibit significantly larger values than others e.g. 100000times larger. We call them massive activations. First we demonstrate thewidespread existence of massive activations across various LLMs andcharacterize their locations. Second we find their values largely stayconstant regardless of the input and they function as indispensable bias termsin LLMs. Third these massive activations lead to the concentration ofattention probabilities to their corresponding tokens and further implicitbias terms in the self-attention output. Last we also study massiveactivations in Vision Transformers. |


| Item |Content|
| --- |---|
|idx| 2402.17760v1 |
|title| Learning to Program Variational Quantum Circuits with Fast Weights |
|authors| Samuel Yen-Chi Chen
|links| http://arxiv.org/abs/2402.17760v1 |
|updated| 2024-02-27 18:53:18 UTC |
|summary| Quantum Machine Learning QML has surfaced as a pioneering frameworkaddressing sequential control tasks and time-series modeling. It hasdemonstrated empirical quantum advantages notably within domains such asReinforcement Learning RL and time-series prediction. A significantadvancement lies in Quantum Recurrent Neural Networks QRNNs specificallytailored for memory-intensive tasks encompassing partially observableenvironments and non-linear time-series prediction. Nevertheless QRNN-basedmodels encounter challenges notably prolonged training duration stemming fromthe necessity to compute quantum gradients using backpropagation-through-timeBPTT. This predicament exacerbates when executing the complete model onquantum devices primarily due to the substantial demand for circuit evaluationarising from the parameter-shift rule. This paper introduces the Quantum FastWeight Programmers QFWP as a solution to the temporal or sequential learningchallenge. The QFWP leverages a classical neural network referred to as theslow programmer functioning as a quantum programmer to swiftly modify theparameters of a variational quantum circuit termed the fast programmer.Instead of completely overwriting the fast programmer at each time-step theslow programmer generates parameter changes or updates for the quantum circuitparameters. This approach enables the fast programmer to incorporate pastobservations or information. Notably the proposed QFWP model achieves learningof temporal dependencies without necessitating the use of quantum recurrentneural networks. Numerical simulations conducted in this study showcase theefficacy of the proposed QFWP model in both time-series prediction and RLtasks. The model exhibits performance levels either comparable to or surpassingthose achieved by QLSTM-based models. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2402.17768v1 |
|title| Diffusion Meets DAgger: Supercharging Eye-in-hand Imitation Learning |
|authors| Xiaoyu ZhangMatthew ChangPranav KumarSaurabh Gupta
|links| http://arxiv.org/abs/2402.17768v1 |
|updated| 2024-02-27 18:59:18 UTC |
|summary| A common failure mode for policies trained with imitation is compoundingexecution errors at test time. When the learned policy encounters states thatwere not present in the expert demonstrations the policy fails leading todegenerate behavior. The Dataset Aggregation or DAgger approach to thisproblem simply collects more data to cover these failure states. However inpractice this is often prohibitively expensive. In this work we proposeDiffusion Meets DAgger DMD a method to reap the benefits of DAgger withoutthe cost for eye-in-hand imitation learning problems. Instead of collecting newsamples to cover out-of-distribution states DMD uses recent advances indiffusion models to create these samples with diffusion models. This leads torobust performance from few demonstrations. In experiments conducted fornon-prehensile pushing on a Franka Research 3 we show that DMD can achieve asuccess rate of 80 with as few as 8 expert demonstrations where naivebehavior cloning reaches only 20. DMD also outperform competing NeRF-basedaugmentation schemes by 50. |


| Item |Content|
| --- |---|
|idx| 2402.17767v1 |
|title| Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator |
|authors| Arjun GuptaMichelle ZhangRishik SathuaSaurabh Gupta
|links| http://arxiv.org/abs/2402.17767v1 |
|updated| 2024-02-27 18:58:54 UTC |
|summary| Pulling open cabinets and drawers presents many difficult technicalchallenges in perception inferring articulation parameters for objects fromonboard sensors planning producing motion plans that conform to tight taskconstraints and control making and maintaining contact while applying forceson the environment. In this work we build an end-to-end system that enables acommodity mobile manipulator Stretch RE2 to pull open cabinets and drawers indiverse previously unseen real world environments. We conduct 4 days of realworld testing of this system spanning 31 different objects from across 13different real world environments. Our system achieves a success rate of 61 onopening novel cabinets and drawers in unseen environments zero-shot. Ananalysis of the failure modes suggests that errors in perception are the mostsignificant challenge for our system. We will open source code and models forothers to replicate and build upon our system. |


| Item |Content|
| --- |---|
|idx| 2402.17766v1 |
|title| ShapeLLM: Universal 3D Object Understanding for Embodied Interaction |
|authors| Zekun QiRunpei DongShaochen ZhangHaoran GengChunrui HanZheng GeLi YiKaisheng Ma
|links| http://arxiv.org/abs/2402.17766v1 |
|updated| 2024-02-27 18:57:12 UTC |
|summary| This paper presents ShapeLLM the first 3D Multimodal Large Language ModelLLM designed for embodied interaction exploring a universal 3D objectunderstanding with 3D point clouds and languages. ShapeLLM is built upon animproved 3D encoder by extending ReCon to ReCon that benefits from multi-viewimage distillation for enhanced geometry understanding. By utilizing ReCon asthe 3D point cloud input encoder for LLMs ShapeLLM is trained on constructedinstruction-following data and tested on our newly human-curated evaluationbenchmark 3D MM-Vet. ReCon and ShapeLLM achieve state-of-the-art performancein 3D geometry understanding and language-unified 3D interaction tasks such asembodied visual grounding. |


| Item |Content|
| --- |---|
|idx| 2402.17758v1 |
|title| ADL4D: Towards A Contextually Rich Dataset for 4D Activities of Daily Living |
|authors| Marsil ZakourPartha Pratim NathLudwig LohmerEmre Faik GökçeMartin PiccolrovazziConstantin PatschYuankai WuRahul ChaudhariEckehard Steinbach
|links| http://arxiv.org/abs/2402.17758v1 |
|updated| 2024-02-27 18:51:52 UTC |
|summary| Hand-Object Interactions HOIs are conditioned on spatial and temporalcontexts like surrounding objects pre- vious actions and future intents forexample grasping and handover actions vary greatly based on objects proximityand trajectory obstruction. However existing datasets for 4D HOI 3D HOI overtime are limited to one subject inter- acting with one object only. Thisrestricts the generalization of learning-based HOI methods trained on thosedatasets. We introduce ADL4D a dataset of up to two subjects inter- actingwith different sets of objects performing Activities of Daily Living ADL likebreakfast or lunch preparation ac- tivities. The transition between multipleobjects to complete a certain task over time introduces a unique contextlacking in existing datasets. Our dataset consists of 75 sequences with a totalof 1.1M RGB-D frames hand and object poses and per-hand fine-grained actionannotations. We develop an automatic system for multi-view multi-hand 3D posean- notation capable of tracking hand poses over time. We inte- grate and testit against publicly available datasets. Finally we evaluate our dataset on thetasks of Hand Mesh Recov- ery HMR and Hand Action Segmentation HAS. |


| Item |Content|
| --- |---|
|idx| 2402.17745v1 |
|title| LoDIP: Low light phase retrieval with deep image prior |
|authors| Raunak ManekarElisa NegriniMinh PhamDaniel JacobsJaideep Srivastava
|links| http://arxiv.org/abs/2402.17745v1 |
|updated| 2024-02-27 18:29:07 UTC |
|summary| Phase retrieval PR is a fundamental challenge in scientific imagingenabling nanoscale techniques like coherent diffractive imaging CDI. Imagingat low radiation doses becomes important in applications where samples aresusceptible to radiation damage. However most PR methods struggle in low dosescenario due to the presence of very high shot noise. Advancements in theoptical data acquisition setup exemplified by in-situ CDI have shownpotential for low-dose imaging. But these depend on a time series ofmeasurements rendering them unsuitable for single-image applications.Similarly on the computational front data-driven phase retrieval techniquesare not readily adaptable to the single-image context. Deep learning basedsingle-image methods such as deep image prior have been effective for variousimaging tasks but have exhibited limited success when applied to PR. In thiswork we propose LoDIP which combines the in-situ CDI setup with the power ofimplicit neural priors to tackle the problem of single-image low-dose phaseretrieval. Quantitative evaluations demonstrate the superior performance ofLoDIP on this task as well as applicability to real experimental scenarios. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2402.17756v1 |
|title| Robustly Learning Single-Index Models via Alignment Sharpness |
|authors| Nikos ZarifisPuqian WangIlias DiakonikolasJelena Diakonikolas
|links| http://arxiv.org/abs/2402.17756v1 |
|updated| 2024-02-27 18:48:07 UTC |
|summary| We study the problem of learning Single-Index Models under the L_22 lossin the agnostic model. We give an efficient learning algorithm achieving aconstant factor approximation to the optimal loss that succeeds under a rangeof distributions including log-concave distributions and a broad class ofmonotone and Lipschitz link functions. This is the first efficient constantfactor approximate agnostic learner even for Gaussian data and for anynontrivial class of link functions. Prior work for the case of unknown linkfunction either works in the realizable setting or does not attain constantfactor approximation. The main technical ingredient enabling our algorithm andanalysis is a novel notion of a local error bound in optimization that we termalignment sharpness and that may be of broader interest. |


| Item |Content|
| --- |---|
|idx| 2402.17747v1 |
|title| When Your AI Deceives You: Challenges with Partial Observability of Human Evaluators in Reward Learning |
|authors| Leon LangDavis FooteStuart RussellAnca DraganErik JennerScott Emmons
|links| http://arxiv.org/abs/2402.17747v1 |
|updated| 2024-02-27 18:32:11 UTC |
|summary| Past analyses of reinforcement learning from human feedback RLHF assumethat the human fully observes the environment. What happens when human feedbackis based only on partial observations We formally define two failure cases:deception and overjustification. Modeling the human as Boltzmann-rationalw.r.t. a belief over trajectories we prove conditions under which RLHF isguaranteed to result in policies that deceptively inflate their performanceoverjustify their behavior to make an impression or both. To help addressthese issues we mathematically characterize how partial observability of theenvironment translates into lack of ambiguity in the learned return function.In some cases accounting for partial observability makes it theoreticallypossible to recover the return function and thus the optimal policy while inother cases there is irreducible ambiguity. We caution against blindlyapplying RLHF in partially observable settings and propose research directionsto help tackle these challenges. |


| Item |Content|
| --- |---|
|idx| 2402.17732v1 |
|title| Batched Nonparametric Contextual Bandits |
|authors| Rong JiangCong Ma
|links| http://arxiv.org/abs/2402.17732v1 |
|updated| 2024-02-27 18:06:20 UTC |
|summary| We study nonparametric contextual bandits under batch constraints where theexpected reward for each action is modeled as a smooth function of covariatesand the policy updates are made at the end of each batch of observations. Weestablish a minimax regret lower bound for this setting and propose BatchedSuccessive Elimination with Dynamic Binning BaSEDB that achieves optimalregret up to logarithmic factors. In essence BaSEDB dynamically splits thecovariate space into smaller bins carefully aligning their widths with thebatch size. We also show the suboptimality of static binning under batchconstraints highlighting the necessity of dynamic binning. Additionally ourresults suggest that a nearly constant number of policy updates can attainoptimal regret in the fully online setting. |


| Item |Content|
| --- |---|
|idx| 2402.17704v1 |
|title| Transfer Learning Bayesian Optimization to Design Competitor DNA Molecules for Use in Diagnostic Assays |
|authors| Ruby SedgwickJohn P. GoertzMolly M. StevensRuth MisenerMark van der Wilk
|links| http://arxiv.org/abs/2402.17704v1 |
|updated| 2024-02-27 17:30:33 UTC |
|summary| With the rise in engineered biomolecular devices there is an increased needfor tailor-made biological sequences. Often many similar biological sequencesneed to be made for a specific application meaning numerous sometimesprohibitively expensive lab experiments are necessary for their optimization.This paper presents a transfer learning design of experiments workflow to makethis development feasible. By combining a transfer learning surrogate modelwith Bayesian optimization we show how the total number of experiments can bereduced by sharing information between optimization tasks. We demonstrate thereduction in the number of experiments using data from the development of DNAcompetitors for use in an amplification-based diagnostic assay. We usecross-validation to compare the predictive accuracy of different transferlearning models and then compare the performance of the models for both singleobjective and penalized optimization tasks. |


| Item |Content|
| --- |---|
|idx| 2402.17699v1 |
|title| Gradient-based Discrete Sampling with Automatic Cyclical Scheduling |
|authors| Patrick PynadathRiddhiman BhattacharyaArun HariharanRuqi Zhang
|links| http://arxiv.org/abs/2402.17699v1 |
|updated| 2024-02-27 17:23:40 UTC |
|summary| Discrete distributions particularly in high-dimensional deep models areoften highly multimodal due to inherent discontinuities. While gradient-baseddiscrete sampling has proven effective it is susceptible to becoming trappedin local modes due to the gradient information. To tackle this challenge wepropose an automatic cyclical scheduling designed for efficient and accuratesampling in multimodal discrete distributions. Our method contains three keycomponents: 1 a cyclical step size schedule where large steps discover newmodes and small steps exploit each mode 2 a cyclical balancing scheduleensuring balanced proposals for given step sizes and high efficiency of theMarkov chain and 3 an automatic tuning scheme for adjusting thehyperparameters in the cyclical schedules allowing adaptability across diversedatasets with minimal tuning. We prove the non-asymptotic convergence andinference guarantee for our method in general discrete distributions. Extensiveexperiments demonstrate the superiority of our method in sampling complexmultimodal discrete distributions. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2402.17751v1 |
|title| An Eye Gaze Heatmap Analysis of Uncertainty Head-Up Display Designs for Conditional Automated Driving |
|authors| Michael A. GerberRonald SchroeterDaniel JohnsonChristian P. JanssenAndry RakotonirainyJonny KuoMike G. Lenne
|links| http://dx.doi.org/10.1145/3613904.3642219 |
|updated| 2024-02-27 18:38:05 UTC |
|summary| This paper reports results from a high-fidelity driving simulator studyN215 about a head-up display HUD that conveys a conditional automatedvehicles dynamic uncertainty about the current situation while fallbackdrivers watch entertaining videos. We compared between-group three designinterventions: display a bar visualisation of uncertainty close to the videointerruption interrupting the video during uncertain situations andcombination a combination of both against a baseline video-only. Wevisualised eye-tracking data to conduct a heatmap analysis of the four groupsgaze behaviour over time. We found interruptions initiated a phase during whichparticipants interleaved their attention between monitoring and entertainment.This improved monitoring behaviour was more pronounced in combination comparedto interruption suggesting pre-warning interruptions have positive effects.The same addition had negative effects without interruptions comparingbaseline  display. Intermittent interruptions may have safety benefits overplacing additional peripheral displays without compromising usability. |


| Item |Content|
| --- |---|
|idx| 2402.17721v1 |
|title| Content-Centric Prototyping of Generative AI Applications: Emerging Approaches and Challenges in Collaborative Software Teams |
|authors| Hari SubramonyamDivy ThakkarJürgen DieberAnoop Sinha
|links| http://arxiv.org/abs/2402.17721v1 |
|updated| 2024-02-27 17:56:10 UTC |
|summary| Generative AI models are increasingly powering software applicationsoffering the capability to produce expressive content across varied contexts.However unlike previous iterations of human-AI design the emerging designprocess for generative capabilities primarily hinges on prompt engineeringstrategies. Given this fundamental shift in approach our work aims tounderstand how collaborative software teams set up and apply design guidelinesand values iteratively prototype prompts and evaluate prompts to achievedesired outcomes. We conducted design studies with 39 industry professionalsincluding designers software engineers and product managers. Our findingsreveal a content-centric prototyping approach in which teams begin with thecontent they want to generate then identify specific attributes constraintsand values and explore methods to give users the ability to influence andinteract with those attributes. Based on associated challenges such as thelack of model interpretability and overfitting the design to examples weoutline considerations for generative AI prototyping. |


| Item |Content|
| --- |---|
|idx| 2402.17553v2 |
|title| OmniACT: A Dataset and Benchmark for Enabling Multimodal Generalist Autonomous Agents for Desktop and Web |
|authors| Raghav KapoorYash Parag ButalaMelisa RussakJing Yu KohKiran KambleWaseem AlshikhRuslan Salakhutdinov
|links| http://arxiv.org/abs/2402.17553v2 |
|updated| 2024-02-28 17:27:39 UTC |
|summary| For decades human-computer interaction has fundamentally been manual. Eventoday almost all productive work done on the computer necessitates human inputat every step. Autonomous virtual agents represent an exciting step inautomating many of these menial tasks. Virtual agents would empower users withlimited technical proficiency to harness the full possibilities of computersystems. They could also enable the efficient streamlining of numerous computertasks ranging from calendar management to complex travel bookings withminimal human intervention. In this paper we introduce OmniACT thefirst-of-a-kind dataset and benchmark for assessing an agents capability togenerate executable programs to accomplish computer tasks. Our scope extendsbeyond traditional web automation covering a diverse range of desktopapplications. The dataset consists of fundamental tasks such as Play the nextsong as well as longer horizon tasks such as Send an email to John Doementioning the time and place to meet. Specifically given a pair of screenimage and a visually-grounded natural language task the goal is to generate ascript capable of fully executing the task. We run several strong baselinelanguage model agents on our benchmark. The strongest baseline GPT-4 performsthe best on our benchmark However its performance level still reaches only 15of the human proficiency in generating executable scripts capable of completingthe task demonstrating the challenge of our task for conventional web agents.Our benchmark provides a platform to measure and evaluate the progress oflanguage model agents in automating computer tasks and motivates future worktowards building multimodal models that bridge large language models and thevisual grounding of computer screens. |


| Item |Content|
| --- |---|
|idx| 2402.17538v1 |
|title| A TDM-based Analog Front-End for Ear-EEG Recording with 83-G$Ω$ Input Impedance, 384-mV DC Tolerance and 0.47-$μ$Vrms Input-Referred Noise |
|authors| Huiyong Zheng
|links| http://arxiv.org/abs/2402.17538v1 |
|updated| 2024-02-27 14:24:55 UTC |
|summary| This paper presents the design of a time-division multiplexedcapacitively-coupled chopper analog front end with a novel impedance boost loopIBL and a novel DC servo loop DSL. The proposed IBL boosts the inputimpedance of the analog front end to up to several tens of GOmega. Itfirstly utilizes an external IBL to prevent the total input impedance fromdegradation caused by parasitic capacitance from the ESD pad and externalinterconnections and secondly relies on an internal IBL to compensate for theleakage current introduced by the chopper. The proposed DSL consists of acoarse DSL driven by square waveforms and a fine DSL driven by fivephase-interleaving PWM waveforms which up modulate the harmonics 5 timeshigher. An edge-pursuit comparator EPC is utilized to monitor the residualelectrode offset voltage EDO at the LNAs output. Designed in a 0.18-mumCMOS process the AFE consumes 4.5 muA from a 1.2-V supply. The simulatedinput referred noise is 0.47 muVrms from 0.5 to 100 Hz in the presence of a384-mV EDO. The proposed AFE achieves a high input impedance of 83 GOmega at1 Hz and 9.3 GOmega at 100 Hz even with the presence of 20-pF parasiticcapacitance. |


| Item |Content|
| --- |---|
|idx| 2402.17456v1 |
|title| A Piece of Theatre: Investigating How Teachers Design LLM Chatbots to Assist Adolescent Cyberbullying Education |
|authors| Michael A. HedderichNatalie N. BazarovaWenting ZouRyun ShimXinda MaQian Yang
|links| http://arxiv.org/abs/2402.17456v1 |
|updated| 2024-02-27 12:27:51 UTC |
|summary| Cyberbullying harms teenagers mental health and teaching them upstandingintervention is crucial. Wizard-of-Oz studies show chatbots can scale uppersonalized and interactive cyberbullying education but implementing suchchatbots is a challenging and delicate task. We created a no-code chatbotdesign tool for K-12 teachers. Using large language models and prompt chainingour tool allows teachers to prototype bespoke dialogue flows and chatbotutterances. In offering this tool we explore teachers distinctive needs whendesigning chatbots to assist their teaching and how chatbot design tools mightbetter support them. Our findings reveal that teachers welcome the toolenthusiastically. Moreover they see themselves as playwrights guiding both thestudents and the chatbots behaviors while allowing for some improvisation.Their goal is to enable students to rehearse both desirable and undesirablereactions to cyberbullying in a safe environment. We discuss the designopportunities LLM-Chains offer for empowering teachers and the researchopportunities this work opens up. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2402.17615v1 |
|title| A Multi-Agent Model for Opinion Evolution under Cognitive Biases |
|authors| Mário S. AlvimArtur Gaspar da SilvaSophia KnightFrank Valencia
|links| http://arxiv.org/abs/2402.17615v1 |
|updated| 2024-02-27 15:44:12 UTC |
|summary| We generalize the DeGroot model for opinion dynamics to better capturerealistic social scenarios. We introduce a model where each agent has their ownindividual cognitive biases. Society is represented as a directed graph whoseedges indicate how much agents influence one another. Biases are represented asthe functions in the square region -112 and categorized into foursub-regions based on the potential reactions they may elicit in an agent duringinstances of opinion disagreement. Under the assumption that each bias of everyagent is a continuous function within the region of receptive but resistantreactions mathbfR we show that the society converges to a consensus ifthe graph is strongly connected. Under the same assumption we also establishthat the entire society converges to a unanimous opinion if and only if thesource components of the graph-namely strongly connected components with noexternal influence-converge to that opinion. We illustrate that convergence isnot guaranteed for strongly connected graphs when biases are eitherdiscontinuous functions in mathbfR or not included in mathbfR. Weshowcase our model through a series of examples and simulations offeringinsights into how opinions form in social networks under cognitive biases. |


| Item |Content|
| --- |---|
|idx| 2402.17270v1 |
|title| Multi-Agent, Human-Agent and Beyond: A Survey on Cooperation in Social Dilemmas |
|authors| Hao GuoChunjiang MuYang ChenChen ShenShuyue HuZhen Wang
|links| http://arxiv.org/abs/2402.17270v1 |
|updated| 2024-02-27 07:31:30 UTC |
|summary| The study of cooperation within social dilemmas has long been a fundamentaltopic across various disciplines including computer science and socialscience. Recent advancements in Artificial Intelligence AI have significantlyreshaped this field offering fresh insights into understanding and enhancingcooperation. This survey examines three key areas at the intersection of AI andcooperation in social dilemmas. First focusing on multi-agent cooperation wereview the intrinsic and external motivations that support cooperation amongrational agents and the methods employed to develop effective strategiesagainst diverse opponents. Second looking into human-agent cooperation wediscuss the current AI algorithms for cooperating with humans and the humanbiases towards AI agents. Third we review the emergent field of leveraging AIagents to enhance cooperation among humans. We conclude by discussing futureresearch avenues such as using large language models establishing unifiedtheoretical frameworks revisiting existing theories of human cooperation andexploring multiple real-world applications. |


| Item |Content|
| --- |---|
|idx| 2402.17161v1 |
|title| Large Language Model for Participatory Urban Planning |
|authors| Zhilun ZhouYuming LinDepeng JinYong Li
|links| http://arxiv.org/abs/2402.17161v1 |
|updated| 2024-02-27 02:47:50 UTC |
|summary| Participatory urban planning is the mainstream of modern urban planning thatinvolves the active engagement of residents. However the traditionalparticipatory paradigm requires experienced planning experts and is oftentime-consuming and costly. Fortunately the emerging Large Language ModelsLLMs have shown considerable ability to simulate human-like agents which canbe used to emulate the participatory process easily. In this work we introducean LLM-based multi-agent collaboration framework for participatory urbanplanning which can generate land-use plans for urban regions considering thediverse needs of residents. Specifically we construct LLM agents to simulate aplanner and thousands of residents with diverse profiles and backgrounds. Wefirst ask the planner to carry out an initial land-use plan. To deal with thedifferent facilities needs of residents we initiate a discussion among theresidents in each community about the plan where residents provide feedbackbased on their profiles. Furthermore to improve the efficiency of discussionwe adopt a fishbowl discussion mechanism where part of the residents discussand the rest of them act as listeners in each round. Finally we let theplanner modify the plan based on residents feedback. We deploy our method ontwo real-world regions in Beijing. Experiments show that our method achievesstate-of-the-art performance in residents satisfaction and inclusion metricsand also outperforms human experts in terms of service accessibility andecology metrics. |


| Item |Content|
| --- |---|
|idx| 2402.17109v1 |
|title| Replicating Electoral Success |
|authors| Kiran TomlinsonTanvi NamjoshiJohan UganderJon Kleinberg
|links| http://arxiv.org/abs/2402.17109v1 |
|updated| 2024-02-27 01:04:07 UTC |
|summary| A core tension in the study of plurality elections is the clash between theclassic Hotelling-Downs model which predicts that two office-seekingcandidates should position themselves at the median voters policy and theempirical observation that real-world democracies often have two major partieswith divergent policies. Motivated by this tension and drawing from boundedrationality we introduce a dynamic model of candidate positioning based on asimple behavioral heuristic: candidates imitate the policy of previous winners.The resulting model is closely connected to evolutionary replicator dynamicsand exhibits complex behavior despite its simplicity. Foruniformly-distributed voters we prove that when there are k  2 3 or 4candidates per election any symmetric candidate distribution converges overtime to a concentration of candidates at the center. With k ge 5 howeverwe prove that the candidate distribution does not converge to the center. Forinitial distributions without any extreme candidates we prove a strongerstatement than non-convergence showing that the density in an interval aroundthe center goes to zero when k ge 5. As a matter of robustness ourconclusions are qualitatively unchanged if a small fraction of candidates arenot winner-copiers and are instead positioned uniformly at random. Beyond ourtheoretical analysis we illustrate our results in simulation for five or morecandidates we find a tendency towards the emergence of two clusters amechanism suggestive of Duvergers Law the empirical finding that pluralityleads to two-party systems. Our simulations also explore several variations ofthe model including non-uniform voter distributions and other forms of noisewhich exhibit similar convergence patterns. Finally we discuss therelationship between our model and prior work on strategic equilibria ofcandidate positioning games. |


| Item |Content|
| --- |---|
|idx| 2402.16823v2 |
|title| Language Agents as Optimizable Graphs |
|authors| Mingchen ZhugeWenyi WangLouis KirschFrancesco FaccioDmitrii KhizbullinJürgen Schmidhuber
|links| http://arxiv.org/abs/2402.16823v2 |
|updated| 2024-02-27 11:03:10 UTC |
|summary| Various human-designed prompt engineering techniques have been proposed toimprove problem solvers based on Large Language Models LLMs yielding manydisparate code bases. We unify these approaches by describing LLM-based agentsas computational graphs. The nodes implement functions to process multimodaldata or query LLMs and the edges describe the information flow betweenoperations. Graphs can be recursively combined into larger composite graphsrepresenting hierarchies of inter-agent collaboration where edges connectoperations of different agents. Our novel automatic graph optimizers 1refine node-level LLM prompts node optimization and 2 improve agentorchestration by changing graph connectivity edge optimization. Experimentsdemonstrate that our framework can be used to efficiently develop integrateand automatically improve various LLM agents. The code can be found athttps://github.com/metauto-ai/gptswarm. |


