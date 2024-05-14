# cs.CL 

| Item |Content|
| --- |---|
|idx| 2405.07990v1 |
|title| Plot2Code: A Comprehensive Benchmark for Evaluating Multi-modal Large Language Models in Code Generation from Scientific Plots |
|authors| Chengyue WuYixiao GeQiushan GuoJiahao WangZhixuan LiangZeyu LuYing ShanPing Luo
|links| http://arxiv.org/abs/2405.07990v1 |
|updated| 2024-05-13 17:59:22 UTC |
|summary| The remarkable progress of Multi-modal Large Language Models MLLMs hasattracted significant attention due to their superior performance in visualcontexts. However their capabilities in turning visual figure to executablecode have not been evaluated thoroughly. To address this we introducePlot2Code a comprehensive visual coding benchmark designed for a fair andin-depth assessment of MLLMs. We carefully collect 132 manually selectedhigh-quality matplotlib plots across six plot types from publicly availablematplotlib galleries. For each plot we carefully offer its source code and andescriptive instruction summarized by GPT-4. This approach enables Plot2Code toextensively evaluate MLLMs code capabilities across various input modalities.Furthermore we propose three automatic evaluation metrics including code passrate text-match ratio and GPT-4V overall rating for a fine-grainedassessment of the output code and rendered images. Instead of simply judgingpass or fail we employ GPT-4V to make an overall judgement between thegenerated and reference images which has been shown to be consistent withhuman evaluation. The evaluation results which include analyses of 14 MLLMssuch as the proprietary GPT-4V Gemini-Pro and the open-sourced Mini-Geminihighlight the substantial challenges presented by Plot2Code. With Plot2Code wereveal that most existing MLLMs struggle with visual coding for text-denseplots heavily relying on textual instruction. We hope that the evaluationresults from Plot2Code on visual coding will guide the future development ofMLLMs. All data involved with Plot2Code are available athttps://huggingface.co/datasets/TencentARC/Plot2Code. |


| Item |Content|
| --- |---|
|idx| 2405.07960v1 |
|title| AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments |
|authors| Samuel SchmidgallRojin ZiaeiCarl HarrisEduardo ReisJeffrey JoplingMichael Moor
|links| http://arxiv.org/abs/2405.07960v1 |
|updated| 2024-05-13 17:38:53 UTC |
|summary| Diagnosing and managing a patient is a complex sequential decision makingprocess that requires physicians to obtain information -- such as which teststo perform -- and to act upon it. Recent advances in artificial intelligenceAI and large language models LLMs promise to profoundly impact clinicalcare. However current evaluation schemes overrely on static medicalquestion-answering benchmarks falling short on interactive decision-makingthat is required in real-life clinical work. Here we present AgentClinic: amultimodal benchmark to evaluate LLMs in their ability to operate as agents insimulated clinical environments. In our benchmark the doctor agent mustuncover the patients diagnosis through dialogue and active data collection. Wepresent two open benchmarks: a multimodal image and dialogue environmentAgentClinic-NEJM and a dialogue-only environment AgentClinic-MedQA. We embedcognitive and implicit biases both in patient and doctor agents to emulaterealistic interactions between biased agents. We find that introducing biasleads to large reductions in diagnostic accuracy of the doctor agents as wellas reduced compliance confidence and follow-up consultation willingness inpatient agents. Evaluating a suite of state-of-the-art LLMs we find thatseveral models that excel in benchmarks like MedQA are performing poorly inAgentClinic-MedQA. We find that the LLM used in the patient agent is animportant factor for performance in the AgentClinic benchmark. We show thatboth having limited interactions as well as too many interaction reducesdiagnostic accuracy in doctor agents. The code and data for this work ispublicly available at https://AgentClinic.github.io. |


| Item |Content|
| --- |---|
|idx| 2405.07940v1 |
|title| RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors |
|authors| Liam DuganAlyssa HwangFilip TrhlikJosh Magnus LudanAndrew ZhuHainiu XuDaphne IppolitoChris Callison-Burch
|links| http://arxiv.org/abs/2405.07940v1 |
|updated| 2024-05-13 17:15:14 UTC |
|summary| Many commercial and open-source models claim to detect machine-generated textwith very high accuracy 99 or higher. However very few of these detectorsare evaluated on shared benchmark datasets and even when they are the datasetsused for evaluation are insufficiently challenging -- lacking variations insampling strategy adversarial attacks and open-source generative models. Inthis work we present RAID: the largest and most challenging benchmark datasetfor machine-generated text detection. RAID includes over 6 million generationsspanning 11 models 8 domains 11 adversarial attacks and 4 decodingstrategies. Using RAID we evaluate the out-of-domain and adversarialrobustness of 8 open- and 4 closed-source detectors and find that currentdetectors are easily fooled by adversarial attacks variations in samplingstrategies repetition penalties and unseen generative models. We release ourdataset and tools to encourage further exploration into detector robustness. |


| Item |Content|
| --- |---|
|idx| 2405.07938v1 |
|title| EconLogicQA: A Question-Answering Benchmark for Evaluating Large Language Models in Economic Sequential Reasoning |
|authors| Yinzhu QuanZefang Liu
|links| http://arxiv.org/abs/2405.07938v1 |
|updated| 2024-05-13 17:13:47 UTC |
|summary| In this paper we introduce EconLogicQA a rigorous benchmark designed toassess the sequential reasoning capabilities of large language models LLMswithin the intricate realms of economics business and supply chainmanagement. Diverging from traditional benchmarks that predict subsequentevents individually EconLogicQA poses a more challenging task: it requiresmodels to discern and sequence multiple interconnected events capturing thecomplexity of economic logics. EconLogicQA comprises an array of multi-eventscenarios derived from economic articles which necessitate an insightfulunderstanding of both temporal and logical event relationships. Throughcomprehensive evaluations we exhibit that EconLogicQA effectively gauges aLLMs proficiency in navigating the sequential complexities inherent ineconomic contexts. We provide a detailed description of EconLogicQA dataset andshows the outcomes from evaluating the benchmark across various leading-edgeLLMs thereby offering a thorough perspective on their sequential reasoningpotential in economic contexts. Our benchmark dataset is available athttps://huggingface.co/datasets/yinzhu-quan/econ_logic_qa. |


| Item |Content|
| --- |---|
|idx| 2405.07932v1 |
|title| PARDEN, Can You Repeat That? Defending against Jailbreaks via Repetition |
|authors| Ziyang ZhangQizhen ZhangJakob Foerster
|links| http://arxiv.org/abs/2405.07932v1 |
|updated| 2024-05-13 17:08:42 UTC |
|summary| Large language models LLMs have shown success in many natural languageprocessing tasks. Despite rigorous safety alignment processes supposedlysafety-aligned LLMs like Llama 2 and Claude 2 are still susceptible tojailbreaks leading to security risks and abuse of the models. One option tomitigate such risks is to augment the LLM with a dedicated safeguard whichchecks the LLMs inputs or outputs for undesired behaviour. A promisingapproach is to use the LLM itself as the safeguard. Nonetheless baselinemethods such as prompting the LLM to self-classify toxic content demonstratelimited efficacy. We hypothesise that this is due to domain shift: thealignment training imparts a self-censoring behaviour to the model Sorry Icant do that while the self-classify approach shifts it to a classificationformat Is this prompt malicious. In this work we propose PARDEN whichavoids this domain shift by simply asking the model to repeat its own outputs.PARDEN neither requires finetuning nor white box access to the model. Weempirically verify the effectiveness of our method and show that PARDENsignificantly outperforms existing jailbreak detection baselines for Llama-2and Claude-2. Code and data are available at https://github.com/Ed-Zh/PARDEN.  We find that PARDEN is particularly powerful in the relevant regime of highTrue Positive Rate TPR and low False Positive Rate FPR. For instance forLlama2-7B at TPR equal to 90 PARDEN accomplishes a roughly 11x reduction inthe FPR from 24.8 to 2.0 on the harmful behaviours dataset. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2405.07992v1 |
|title| MambaOut: Do We Really Need Mamba for Vision? |
|authors| Weihao YuXinchao Wang
|links| http://arxiv.org/abs/2405.07992v1 |
|updated| 2024-05-13 17:59:56 UTC |
|summary| Mamba an architecture with RNN-like token mixer of state space model SSMwas recently introduced to address the quadratic complexity of the attentionmechanism and subsequently applied to vision tasks. Nevertheless theperformance of Mamba for vision is often underwhelming when compared withconvolutional and attention-based models. In this paper we delve into theessence of Mamba and conceptually conclude that Mamba is ideally suited fortasks with long-sequence and autoregressive characteristics. For vision tasksas image classification does not align with either characteristic wehypothesize that Mamba is not necessary for this task Detection andsegmentation tasks are also not autoregressive yet they adhere to thelong-sequence characteristic so we believe it is still worthwhile to exploreMambas potential for these tasks. To empirically verify our hypotheses weconstruct a series of models named emphMambaOut through stacking Mambablocks while removing their core token mixer SSM. Experimental resultsstrongly support our hypotheses. Specifically our MambaOut model surpasses allvisual Mamba models on ImageNet image classification indicating that Mamba isindeed unnecessary for this task. As for detection and segmentation MambaOutcannot match the performance of state-of-the-art visual Mamba modelsdemonstrating the potential of Mamba for long-sequence visual tasks. The codeis available at https://github.com/yuweihao/MambaOut |


| Item |Content|
| --- |---|
|idx| 2405.07991v1 |
|title| SPIN: Simultaneous Perception, Interaction and Navigation |
|authors| Shagun UppalAnanye AgarwalHaoyu XiongKenneth ShawDeepak Pathak
|links| http://arxiv.org/abs/2405.07991v1 |
|updated| 2024-05-13 17:59:36 UTC |
|summary| While there has been remarkable progress recently in the fields ofmanipulation and locomotion mobile manipulation remains a long-standingchallenge. Compared to locomotion or static manipulation a mobile system mustmake a diverse range of long-horizon tasks feasible in unstructured and dynamicenvironments. While the applications are broad and interesting there are aplethora of challenges in developing these systems such as coordination betweenthe base and arm reliance on onboard perception for perceiving and interactingwith the environment and most importantly simultaneously integrating allthese parts together. Prior works approach the problem using disentangledmodular skills for mobility and manipulation that are trivially tied together.This causes several limitations such as compounding errors delays indecision-making and no whole-body coordination. In this work we present areactive mobile manipulation framework that uses an active visual system toconsciously perceive and react to its environment. Similar to how humansleverage whole-body and hand-eye coordination we develop a mobile manipulatorthat exploits its ability to move and see more specifically -- to move inorder to see and to see in order to move. This allows it to not only movearound and interact with its environment but also choose when to perceivewhat using an active visual system. We observe that such an agent learns tonavigate around complex cluttered scenarios while displaying agile whole-bodycoordination using only ego-vision without needing to create environment maps.Results visualizations and videos at https://spin-robot.github.io/ |


| Item |Content|
| --- |---|
|idx| 2405.07987v1 |
|title| The Platonic Representation Hypothesis |
|authors| Minyoung HuhBrian CheungTongzhou WangPhillip Isola
|links| http://arxiv.org/abs/2405.07987v1 |
|updated| 2024-05-13 17:58:30 UTC |
|summary| We argue that representations in AI models particularly deep networks areconverging. First we survey many examples of convergence in the literature:over time and across multiple domains the ways by which different neuralnetworks represent data are becoming more aligned. Next we demonstrateconvergence across data modalities: as vision models and language models getlarger they measure distance between datapoints in a more and more alike way.We hypothesize that this convergence is driving toward a shared statisticalmodel of reality akin to Platos concept of an ideal reality. We term such arepresentation the platonic representation and discuss several possibleselective pressures toward it. Finally we discuss the implications of thesetrends their limitations and counterexamples to our analysis. |


| Item |Content|
| --- |---|
|idx| 2405.07976v1 |
|title| Localized Adaptive Risk Control |
|authors| Matteo ZecchinOsvaldo Simeone
|links| http://arxiv.org/abs/2405.07976v1 |
|updated| 2024-05-13 17:48:45 UTC |
|summary| Adaptive Risk Control ARC is an online calibration strategy based on setprediction that offers worst-case deterministic long-term risk control as wellas statistical marginal coverage guarantees. ARC adjusts the size of theprediction set by varying a single scalar threshold based on feedback from pastdecisions. In this work we introduce Localized Adaptive Risk Control L-ARCan online calibration scheme that targets statistical localized risk guaranteesranging from conditional risk to marginal risk while preserving the worst-caseperformance of ARC. L-ARC updates a threshold function within a reproducingkernel Hilbert space RKHS with the kernel determining the level oflocalization of the statistical risk guarantee. The theoretical resultshighlight a trade-off between localization of the statistical risk andconvergence speed to the long-term risk target. Thanks to localization L-ARCis demonstrated via experiments to produce prediction sets with risk guaranteesacross different data subpopulations significantly improving the fairness ofthe calibrated model for tasks such as image segmentation and beam selection inwireless networks. |


| Item |Content|
| --- |---|
|idx| 2405.07969v1 |
|title| Investigating the Semantic Robustness of CLIP-based Zero-Shot Anomaly Segmentation |
|authors| Kevin StanglMarius ArvinteWeilin XuCory Cornelius
|links| http://arxiv.org/abs/2405.07969v1 |
|updated| 2024-05-13 17:47:08 UTC |
|summary| Zero-shot anomaly segmentation using pre-trained foundation models is apromising approach that enables effective algorithms without expensivedomain-specific training or fine-tuning. Ensuring that these methods workacross various environmental conditions and are robust to distribution shiftsis an open problem. We investigate the performance of WinCLIP 14 zero-shotanomaly segmentation algorithm by perturbing test data using three semantictransformations: bounded angular rotations bounded saturation shifts and hueshifts. We empirically measure a lower performance bound by aggregating acrossper-sample worst-case perturbations and find that average performance drops byup to 20 in area under the ROC curve and 40 in area under the per-regionoverlap curve. We find that performance is consistently lowered on three CLIPbackbones regardless of model architecture or learning objectivedemonstrating a need for careful performance evaluation. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2405.07992v1 |
|title| MambaOut: Do We Really Need Mamba for Vision? |
|authors| Weihao YuXinchao Wang
|links| http://arxiv.org/abs/2405.07992v1 |
|updated| 2024-05-13 17:59:56 UTC |
|summary| Mamba an architecture with RNN-like token mixer of state space model SSMwas recently introduced to address the quadratic complexity of the attentionmechanism and subsequently applied to vision tasks. Nevertheless theperformance of Mamba for vision is often underwhelming when compared withconvolutional and attention-based models. In this paper we delve into theessence of Mamba and conceptually conclude that Mamba is ideally suited fortasks with long-sequence and autoregressive characteristics. For vision tasksas image classification does not align with either characteristic wehypothesize that Mamba is not necessary for this task Detection andsegmentation tasks are also not autoregressive yet they adhere to thelong-sequence characteristic so we believe it is still worthwhile to exploreMambas potential for these tasks. To empirically verify our hypotheses weconstruct a series of models named emphMambaOut through stacking Mambablocks while removing their core token mixer SSM. Experimental resultsstrongly support our hypotheses. Specifically our MambaOut model surpasses allvisual Mamba models on ImageNet image classification indicating that Mamba isindeed unnecessary for this task. As for detection and segmentation MambaOutcannot match the performance of state-of-the-art visual Mamba modelsdemonstrating the potential of Mamba for long-sequence visual tasks. The codeis available at https://github.com/yuweihao/MambaOut |


| Item |Content|
| --- |---|
|idx| 2405.07991v1 |
|title| SPIN: Simultaneous Perception, Interaction and Navigation |
|authors| Shagun UppalAnanye AgarwalHaoyu XiongKenneth ShawDeepak Pathak
|links| http://arxiv.org/abs/2405.07991v1 |
|updated| 2024-05-13 17:59:36 UTC |
|summary| While there has been remarkable progress recently in the fields ofmanipulation and locomotion mobile manipulation remains a long-standingchallenge. Compared to locomotion or static manipulation a mobile system mustmake a diverse range of long-horizon tasks feasible in unstructured and dynamicenvironments. While the applications are broad and interesting there are aplethora of challenges in developing these systems such as coordination betweenthe base and arm reliance on onboard perception for perceiving and interactingwith the environment and most importantly simultaneously integrating allthese parts together. Prior works approach the problem using disentangledmodular skills for mobility and manipulation that are trivially tied together.This causes several limitations such as compounding errors delays indecision-making and no whole-body coordination. In this work we present areactive mobile manipulation framework that uses an active visual system toconsciously perceive and react to its environment. Similar to how humansleverage whole-body and hand-eye coordination we develop a mobile manipulatorthat exploits its ability to move and see more specifically -- to move inorder to see and to see in order to move. This allows it to not only movearound and interact with its environment but also choose when to perceivewhat using an active visual system. We observe that such an agent learns tonavigate around complex cluttered scenarios while displaying agile whole-bodycoordination using only ego-vision without needing to create environment maps.Results visualizations and videos at https://spin-robot.github.io/ |


| Item |Content|
| --- |---|
|idx| 2405.07987v1 |
|title| The Platonic Representation Hypothesis |
|authors| Minyoung HuhBrian CheungTongzhou WangPhillip Isola
|links| http://arxiv.org/abs/2405.07987v1 |
|updated| 2024-05-13 17:58:30 UTC |
|summary| We argue that representations in AI models particularly deep networks areconverging. First we survey many examples of convergence in the literature:over time and across multiple domains the ways by which different neuralnetworks represent data are becoming more aligned. Next we demonstrateconvergence across data modalities: as vision models and language models getlarger they measure distance between datapoints in a more and more alike way.We hypothesize that this convergence is driving toward a shared statisticalmodel of reality akin to Platos concept of an ideal reality. We term such arepresentation the platonic representation and discuss several possibleselective pressures toward it. Finally we discuss the implications of thesetrends their limitations and counterexamples to our analysis. |


| Item |Content|
| --- |---|
|idx| 2405.07977v1 |
|title| A Demographic-Conditioned Variational Autoencoder for fMRI Distribution Sampling and Removal of Confounds |
|authors| Anton OrlichenkoGang QuZiyu ZhouAnqi LiuHong-Wen DengZhengming DingJulia M. StephenTony W. WilsonVince D. CalhounYu-Ping Wang
|links| http://arxiv.org/abs/2405.07977v1 |
|updated| 2024-05-13 17:49:20 UTC |
|summary| Objective: fMRI and derived measures such as functional connectivity FChave been used to predict brain age general fluid intelligence psychiatricdisease status and preclinical neurodegenerative disease. However it is notalways clear that all demographic confounds such as age sex and race havebeen removed from fMRI data. Additionally many fMRI datasets are restricted toauthorized researchers making dissemination of these valuable data sourceschallenging. Methods: We create a variational autoencoder VAE-based modelDemoVAE to decorrelate fMRI features from demographics and generatehigh-quality synthetic fMRI data based on user-supplied demographics. We trainand validate our model using two large widely used datasets the PhiladelphiaNeurodevelopmental Cohort PNC and Bipolar and Schizophrenia Network forIntermediate Phenotypes BSNIP. Results: We find that DemoVAE recapitulatesgroup differences in fMRI data while capturing the full breadth of individualvariations. Significantly we also find that most clinical and computerizedbattery fields that are correlated with fMRI data are not correlated withDemoVAE latents. An exception are several fields related to schizophreniamedication and symptom severity. Conclusion: Our model generates fMRI data thatcaptures the full distribution of FC better than traditional VAE or GAN models.We also find that most prediction using fMRI data is dependent on correlationwith and prediction of demographics. Significance: Our DemoVAE model allowsfor generation of high quality synthetic data conditioned on subjectdemographics as well as the removal of the confounding effects of demographics.We identify that FC-based prediction tasks are highly influenced by demographicconfounds. |


| Item |Content|
| --- |---|
|idx| 2405.07976v1 |
|title| Localized Adaptive Risk Control |
|authors| Matteo ZecchinOsvaldo Simeone
|links| http://arxiv.org/abs/2405.07976v1 |
|updated| 2024-05-13 17:48:45 UTC |
|summary| Adaptive Risk Control ARC is an online calibration strategy based on setprediction that offers worst-case deterministic long-term risk control as wellas statistical marginal coverage guarantees. ARC adjusts the size of theprediction set by varying a single scalar threshold based on feedback from pastdecisions. In this work we introduce Localized Adaptive Risk Control L-ARCan online calibration scheme that targets statistical localized risk guaranteesranging from conditional risk to marginal risk while preserving the worst-caseperformance of ARC. L-ARC updates a threshold function within a reproducingkernel Hilbert space RKHS with the kernel determining the level oflocalization of the statistical risk guarantee. The theoretical resultshighlight a trade-off between localization of the statistical risk andconvergence speed to the long-term risk target. Thanks to localization L-ARCis demonstrated via experiments to produce prediction sets with risk guaranteesacross different data subpopulations significantly improving the fairness ofthe calibrated model for tasks such as image segmentation and beam selection inwireless networks. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2405.07992v1 |
|title| MambaOut: Do We Really Need Mamba for Vision? |
|authors| Weihao YuXinchao Wang
|links| http://arxiv.org/abs/2405.07992v1 |
|updated| 2024-05-13 17:59:56 UTC |
|summary| Mamba an architecture with RNN-like token mixer of state space model SSMwas recently introduced to address the quadratic complexity of the attentionmechanism and subsequently applied to vision tasks. Nevertheless theperformance of Mamba for vision is often underwhelming when compared withconvolutional and attention-based models. In this paper we delve into theessence of Mamba and conceptually conclude that Mamba is ideally suited fortasks with long-sequence and autoregressive characteristics. For vision tasksas image classification does not align with either characteristic wehypothesize that Mamba is not necessary for this task Detection andsegmentation tasks are also not autoregressive yet they adhere to thelong-sequence characteristic so we believe it is still worthwhile to exploreMambas potential for these tasks. To empirically verify our hypotheses weconstruct a series of models named emphMambaOut through stacking Mambablocks while removing their core token mixer SSM. Experimental resultsstrongly support our hypotheses. Specifically our MambaOut model surpasses allvisual Mamba models on ImageNet image classification indicating that Mamba isindeed unnecessary for this task. As for detection and segmentation MambaOutcannot match the performance of state-of-the-art visual Mamba modelsdemonstrating the potential of Mamba for long-sequence visual tasks. The codeis available at https://github.com/yuweihao/MambaOut |


| Item |Content|
| --- |---|
|idx| 2405.07991v1 |
|title| SPIN: Simultaneous Perception, Interaction and Navigation |
|authors| Shagun UppalAnanye AgarwalHaoyu XiongKenneth ShawDeepak Pathak
|links| http://arxiv.org/abs/2405.07991v1 |
|updated| 2024-05-13 17:59:36 UTC |
|summary| While there has been remarkable progress recently in the fields ofmanipulation and locomotion mobile manipulation remains a long-standingchallenge. Compared to locomotion or static manipulation a mobile system mustmake a diverse range of long-horizon tasks feasible in unstructured and dynamicenvironments. While the applications are broad and interesting there are aplethora of challenges in developing these systems such as coordination betweenthe base and arm reliance on onboard perception for perceiving and interactingwith the environment and most importantly simultaneously integrating allthese parts together. Prior works approach the problem using disentangledmodular skills for mobility and manipulation that are trivially tied together.This causes several limitations such as compounding errors delays indecision-making and no whole-body coordination. In this work we present areactive mobile manipulation framework that uses an active visual system toconsciously perceive and react to its environment. Similar to how humansleverage whole-body and hand-eye coordination we develop a mobile manipulatorthat exploits its ability to move and see more specifically -- to move inorder to see and to see in order to move. This allows it to not only movearound and interact with its environment but also choose when to perceivewhat using an active visual system. We observe that such an agent learns tonavigate around complex cluttered scenarios while displaying agile whole-bodycoordination using only ego-vision without needing to create environment maps.Results visualizations and videos at https://spin-robot.github.io/ |


| Item |Content|
| --- |---|
|idx| 2405.07990v1 |
|title| Plot2Code: A Comprehensive Benchmark for Evaluating Multi-modal Large Language Models in Code Generation from Scientific Plots |
|authors| Chengyue WuYixiao GeQiushan GuoJiahao WangZhixuan LiangZeyu LuYing ShanPing Luo
|links| http://arxiv.org/abs/2405.07990v1 |
|updated| 2024-05-13 17:59:22 UTC |
|summary| The remarkable progress of Multi-modal Large Language Models MLLMs hasattracted significant attention due to their superior performance in visualcontexts. However their capabilities in turning visual figure to executablecode have not been evaluated thoroughly. To address this we introducePlot2Code a comprehensive visual coding benchmark designed for a fair andin-depth assessment of MLLMs. We carefully collect 132 manually selectedhigh-quality matplotlib plots across six plot types from publicly availablematplotlib galleries. For each plot we carefully offer its source code and andescriptive instruction summarized by GPT-4. This approach enables Plot2Code toextensively evaluate MLLMs code capabilities across various input modalities.Furthermore we propose three automatic evaluation metrics including code passrate text-match ratio and GPT-4V overall rating for a fine-grainedassessment of the output code and rendered images. Instead of simply judgingpass or fail we employ GPT-4V to make an overall judgement between thegenerated and reference images which has been shown to be consistent withhuman evaluation. The evaluation results which include analyses of 14 MLLMssuch as the proprietary GPT-4V Gemini-Pro and the open-sourced Mini-Geminihighlight the substantial challenges presented by Plot2Code. With Plot2Code wereveal that most existing MLLMs struggle with visual coding for text-denseplots heavily relying on textual instruction. We hope that the evaluationresults from Plot2Code on visual coding will guide the future development ofMLLMs. All data involved with Plot2Code are available athttps://huggingface.co/datasets/TencentARC/Plot2Code. |


| Item |Content|
| --- |---|
|idx| 2405.07988v1 |
|title| A Generalist Learner for Multifaceted Medical Image Interpretation |
|authors| Hong-Yu ZhouSubathra AdithanJulián Nicolás AcostaEric J. TopolPranav Rajpurkar
|links| http://arxiv.org/abs/2405.07988v1 |
|updated| 2024-05-13 17:58:51 UTC |
|summary| Current medical artificial intelligence systems are often limited to narrowapplications hindering their widespread adoption in clinical practice. Toaddress this limitation we propose MedVersa a generalist learner that enablesflexible learning and tasking for medical image interpretation. By leveraging alarge language model as a learnable orchestrator MedVersa can learn from bothvisual and linguistic supervision support multimodal inputs and performreal-time task specification. This versatility allows MedVersa to adapt tovarious clinical scenarios and perform multifaceted medical image analysis. Weintroduce MedInterp the largest multimodal dataset to date for medical imageinterpretation consisting of over 13 million annotated instances spanning 11tasks across 3 modalities to support the development of MedVersa. Ourexperiments demonstrate that MedVersa achieves state-of-the-art performance in9 tasks sometimes outperforming specialist counterparts by over 10. MedVersais the first to showcase the viability of multimodal generative medical AI inimplementing multimodal outputs inputs and dynamic task specificationhighlighting its potential as a multifunctional system for comprehensivemedical image analysis. This generalist approach to medical imageinterpretation paves the way for more adaptable and efficient AI-assistedclinical decision-making. |


| Item |Content|
| --- |---|
|idx| 2405.07987v1 |
|title| The Platonic Representation Hypothesis |
|authors| Minyoung HuhBrian CheungTongzhou WangPhillip Isola
|links| http://arxiv.org/abs/2405.07987v1 |
|updated| 2024-05-13 17:58:30 UTC |
|summary| We argue that representations in AI models particularly deep networks areconverging. First we survey many examples of convergence in the literature:over time and across multiple domains the ways by which different neuralnetworks represent data are becoming more aligned. Next we demonstrateconvergence across data modalities: as vision models and language models getlarger they measure distance between datapoints in a more and more alike way.We hypothesize that this convergence is driving toward a shared statisticalmodel of reality akin to Platos concept of an ideal reality. We term such arepresentation the platonic representation and discuss several possibleselective pressures toward it. Finally we discuss the implications of thesetrends their limitations and counterexamples to our analysis. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2405.07976v1 |
|title| Localized Adaptive Risk Control |
|authors| Matteo ZecchinOsvaldo Simeone
|links| http://arxiv.org/abs/2405.07976v1 |
|updated| 2024-05-13 17:48:45 UTC |
|summary| Adaptive Risk Control ARC is an online calibration strategy based on setprediction that offers worst-case deterministic long-term risk control as wellas statistical marginal coverage guarantees. ARC adjusts the size of theprediction set by varying a single scalar threshold based on feedback from pastdecisions. In this work we introduce Localized Adaptive Risk Control L-ARCan online calibration scheme that targets statistical localized risk guaranteesranging from conditional risk to marginal risk while preserving the worst-caseperformance of ARC. L-ARC updates a threshold function within a reproducingkernel Hilbert space RKHS with the kernel determining the level oflocalization of the statistical risk guarantee. The theoretical resultshighlight a trade-off between localization of the statistical risk andconvergence speed to the long-term risk target. Thanks to localization L-ARCis demonstrated via experiments to produce prediction sets with risk guaranteesacross different data subpopulations significantly improving the fairness ofthe calibrated model for tasks such as image segmentation and beam selection inwireless networks. |


| Item |Content|
| --- |---|
|idx| 2405.07971v1 |
|title| Sensitivity Analysis for Active Sampling, with Applications to the Simulation of Analog Circuits |
|authors| Reda ChhaibiFabrice GamboaChristophe OgerVinicius OliveiraClément PellegriniDamien Remot
|links| http://arxiv.org/abs/2405.07971v1 |
|updated| 2024-05-13 17:47:40 UTC |
|summary| We propose an active sampling flow with the use-case of simulating theimpact of combined variations on analog circuits. In such a context given thelarge number of parameters it is difficult to fit a surrogate model and toefficiently explore the space of design features.  By combining a drastic dimension reduction using sensitivity analysis andBayesian surrogate modeling we obtain a flexible active sampling flow. Onsynthetic and real datasets this flow outperforms the usual Monte-Carlosampling which often forms the foundation of design space exploration. |


| Item |Content|
| --- |---|
|idx| 2405.07914v1 |
|title| Distribution Learning Meets Graph Structure Sampling |
|authors| Arnab BhattacharyyaSutanu GayenPhilips George JohnSayantan SenN. V. Vinodchandran
|links| http://arxiv.org/abs/2405.07914v1 |
|updated| 2024-05-13 16:47:05 UTC |
|summary| This work establishes a novel link between the problem of PAC-learninghigh-dimensional graphical models and the task of efficient counting andsampling of graph structures using an online learning framework.  We observe that if we apply the exponentially weighted average EWA orrandomized weighted majority RWM forecasters on a sequence of samples from adistribution P using the log loss function the average regret incurred by theforecasters predictions can be used to bound the expected KL divergencebetween P and the predictions. Known regret bounds for EWA and RWM then yieldnew sample complexity bounds for learning Bayes nets. Moreover thesealgorithms can be made computationally efficient for several interestingclasses of Bayes nets. Specifically we give a new sample-optimal andpolynomial time learning algorithm with respect to trees of unknown structureand the first polynomial sample and time algorithm for learning with respect toBayes nets over a given chordal skeleton. |


| Item |Content|
| --- |---|
|idx| 2405.07863v1 |
|title| RLHF Workflow: From Reward Modeling to Online RLHF |
|authors| Hanze DongWei XiongBo PangHaoxiang WangHan ZhaoYingbo ZhouNan JiangDoyen SahooCaiming XiongTong Zhang
|links| http://arxiv.org/abs/2405.07863v1 |
|updated| 2024-05-13 15:50:39 UTC |
|summary| We present the workflow of Online Iterative Reinforcement Learning from HumanFeedback RLHF in this technical report which is widely reported tooutperform its offline counterpart by a large margin in the recent largelanguage model LLM literature. However existing open-source RLHF projectsare still largely confined to the offline learning setting. In this technicalreport we aim to fill in this gap and provide a detailed recipe that is easyto reproduce for online iterative RLHF. In particular since online humanfeedback is usually infeasible for open-source communities with limitedresources we start by constructing preference models using a diverse set ofopen-source datasets and use the constructed proxy preference model toapproximate human feedback. Then we discuss the theoretical insights andalgorithmic principles behind online iterative RLHF followed by a detailedpractical implementation. Our trained LLM SFR-Iterative-DPO-LLaMA-3-8B-Rachieves impressive performance on LLM chatbot benchmarks includingAlpacaEval-2 Arena-Hard and MT-Bench as well as other academic benchmarkssuch as HumanEval and TruthfulQA. We have shown that supervised fine-tuningSFT and iterative RLHF can obtain state-of-the-art performance with fullyopen-source datasets. Further we have made our models curated datasets andcomprehensive step-by-step code guidebooks publicly available. Please refer tohttps://github.com/RLHFlow/RLHF-Reward-Modeling andhttps://github.com/RLHFlow/Online-RLHF for more detailed information. |


| Item |Content|
| --- |---|
|idx| 2405.07860v1 |
|title| Uniform Inference for Subsampled Moment Regression |
|authors| David M. RitzwollerVasilis Syrgkanis
|links| http://arxiv.org/abs/2405.07860v1 |
|updated| 2024-05-13 15:46:11 UTC |
|summary| We propose a method for constructing a confidence region for the solution toa conditional moment equation. The method is built around a class of algorithmsfor nonparametric regression based on subsampled kernels. This class includesrandom forest regression. We bound the error in the confidence regions nominalcoverage probability under the restriction that the conditional momentequation of interest satisfies a local orthogonality condition. The method isapplicable to the construction of confidence regions for conditional averagetreatment effects in randomized experiments among many other similar problemsencountered in applied economics and causal inference. As a by-product weobtain several new order-explicit results on the concentration and normalapproximation of high-dimensional U-statistics. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2405.07963v1 |
|title| PyZoBot: A Platform for Conversational Information Extraction and Synthesis from Curated Zotero Reference Libraries through Advanced Retrieval-Augmented Generation |
|authors| Suad AlshammariLama BasalelahWalaa Abu RukbahAli AlsuhibaniDayanjan S. Wijesinghe
|links| http://arxiv.org/abs/2405.07963v1 |
|updated| 2024-05-13 17:44:05 UTC |
|summary| The exponential growth of scientific literature has resulted in informationoverload challenging researchers to effectively synthesize relevantpublications. This paper explores the integration of traditional referencemanagement software with advanced computational techniques including LargeLanguage Models and Retrieval-Augmented Generation. We introduce PyZoBot anAI-driven platform developed in Python incorporating Zoteros referencemanagement with OpenAIs sophisticated LLMs. PyZoBot streamlines knowledgeextraction and synthesis from extensive human-curated scientific literaturedatabases. It demonstrates proficiency in handling complex natural languagequeries integrating data from multiple sources and meticulously presentingreferences to uphold research integrity and facilitate further exploration. Byleveraging LLMs RAG and human expertise through a curated library PyZoBotoffers an effective solution to manage information overload and keep pace withrapid scientific advancements. The development of such AI-enhanced toolspromises significant improvements in research efficiency and effectivenessacross various disciplines. |


| Item |Content|
| --- |---|
|idx| 2405.07960v1 |
|title| AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments |
|authors| Samuel SchmidgallRojin ZiaeiCarl HarrisEduardo ReisJeffrey JoplingMichael Moor
|links| http://arxiv.org/abs/2405.07960v1 |
|updated| 2024-05-13 17:38:53 UTC |
|summary| Diagnosing and managing a patient is a complex sequential decision makingprocess that requires physicians to obtain information -- such as which teststo perform -- and to act upon it. Recent advances in artificial intelligenceAI and large language models LLMs promise to profoundly impact clinicalcare. However current evaluation schemes overrely on static medicalquestion-answering benchmarks falling short on interactive decision-makingthat is required in real-life clinical work. Here we present AgentClinic: amultimodal benchmark to evaluate LLMs in their ability to operate as agents insimulated clinical environments. In our benchmark the doctor agent mustuncover the patients diagnosis through dialogue and active data collection. Wepresent two open benchmarks: a multimodal image and dialogue environmentAgentClinic-NEJM and a dialogue-only environment AgentClinic-MedQA. We embedcognitive and implicit biases both in patient and doctor agents to emulaterealistic interactions between biased agents. We find that introducing biasleads to large reductions in diagnostic accuracy of the doctor agents as wellas reduced compliance confidence and follow-up consultation willingness inpatient agents. Evaluating a suite of state-of-the-art LLMs we find thatseveral models that excel in benchmarks like MedQA are performing poorly inAgentClinic-MedQA. We find that the LLM used in the patient agent is animportant factor for performance in the AgentClinic benchmark. We show thatboth having limited interactions as well as too many interaction reducesdiagnostic accuracy in doctor agents. The code and data for this work ispublicly available at https://AgentClinic.github.io. |


| Item |Content|
| --- |---|
|idx| 2405.07840v1 |
|title| Open-vocabulary Auditory Neural Decoding Using fMRI-prompted LLM |
|authors| Xiaoyu ChenChangde DuChe LiuYizhe WangHuiguang He
|links| http://arxiv.org/abs/2405.07840v1 |
|updated| 2024-05-13 15:25:11 UTC |
|summary| Decoding language information from brain signals represents a vital researcharea within brain-computer interfaces particularly in the context ofdeciphering the semantic information from the fMRI signal. However manyexisting efforts concentrate on decoding small vocabulary sets leaving spacefor the exploration of open vocabulary continuous text decoding. In this paperwe introduce a novel method the textbfBrain Prompt GPT BP-GPT. By usingthe brain representation that is extracted from the fMRI as a prompt ourmethod can utilize GPT-2 to decode fMRI signals into stimulus text. Further weintroduce a text-to-text baseline and align the fMRI prompt to the text prompt.By introducing the text-to-text baseline our BP-GPT can extract a more robustbrain prompt and promote the decoding of pre-trained LLM. We evaluate ourBP-GPT on the open-source auditory semantic decoding dataset and achieve asignificant improvement up to 4.61 on METEOR and 2.43 on BERTScoreacross all the subjects compared to the state-of-the-art method. Theexperimental results demonstrate that using brain representation as a prompt tofurther drive LLM for auditory neural decoding is feasible and effective. |


| Item |Content|
| --- |---|
|idx| 2405.07834v1 |
|title| Adaptive Human-Swarm Interaction based on Workload Measurement using Functional Near-Infrared Spectroscopy |
|authors| Ayodeji O. AbioyeAleksandra LandowskaWilliam HuntHoria MaiorSarvapali D. RamchurnMohammad NaisehAlec BanksMohammad D. Soorati
|links| http://arxiv.org/abs/2405.07834v1 |
|updated| 2024-05-13 15:20:31 UTC |
|summary| One of the challenges of human-swarm interaction HSI is how to manage theoperators workload. In order to do this we propose a novel neurofeedbacktechnique for the real-time measurement of workload using functionalnear-infrared spectroscopy fNIRS. The objective is to develop a baseline forworkload measurement in human-swarm interaction using fNIRS and to develop aninterface that dynamically adapts to the operators workload. The proposedmethod consists of using fNIRS device to measure brain activity process thisthrough a machine learning algorithm and pass it on to the HSI interface. Bydynamically adapting the HSI interface the swarm operators workload could bereduced and the performance improved. |


| Item |Content|
| --- |---|
|idx| 2405.07658v1 |
|title| Understanding Data Understanding: A Framework to Navigate the Intricacies of Data Analytics |
|authors| Joshua HolsteinPhilipp SpitzerMarieke HoellMichael VössingNiklas Kühl
|links| http://arxiv.org/abs/2405.07658v1 |
|updated| 2024-05-13 11:39:36 UTC |
|summary| As organizations face the challenges of processing exponentially growing datavolumes their reliance on analytics to unlock value from this data hasintensified. However the intricacies of big data such as its extensivefeature sets pose significant challenges. A crucial step in leveraging thisdata for insightful analysis is an in-depth understanding of both the data andits domain. Yet existing literature presents a fragmented picture of whatcomprises an effective understanding of data and domain varying significantlyin depth and focus. To address this research gap we conduct a systematicliterature review aiming to delineate the dimensions of data understanding. Weidentify five dimensions: Foundations Collection  SelectionContextualization  Integration Exploration  Discovery and Insights. Thesedimensions collectively form a comprehensive framework for data understandingproviding guidance for organizations seeking meaningful insights from complexdatasets. This study synthesizes the current state of knowledge and lays thegroundwork for further exploration. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2405.07541v1 |
|title| Random walk model that universally generates inverse square Lévy walk by eliminating search cost minimization constraint |
|authors| Shuji ShinoharaDaiki MoritaHayato HiraiRyosuke KuribayashiNobuhito ManomeToru MoriyamaHiroshi OkamotoYoshihiro NakajimaPegio-Yukio GunjiUng-il Chung
|links| http://arxiv.org/abs/2405.07541v1 |
|updated| 2024-05-13 08:22:44 UTC |
|summary| The Levy walk a type of random walk characterized by linear step lengthsthat follow a power-law distribution is observed in the migratory behaviors ofvarious organisms ranging from bacteria to humans. Notably Levy walks withpower exponents close to two are frequently observed though their underlyingcauses remain elusive. This study introduces a simplified abstract random walkmodel designed to produce inverse square Levy walks also known as Cauchywalks and explores the conditions that facilitate these phenomena. In ourmodel agents move toward a randomly selected destination in multi-dimensionalspace and their movement strategy is parameterized by the extent to which theypursue the shortest path. When the search cost is proportional to the distancetraveled this parameter effectively reflects the emphasis on minimizing searchcosts. Our findings reveal that strict adherence to this cost minimizationconstraint results in a Brownian walk pattern. However removing thisconstraint transitions the movement to an inverse square Levy walk.Therefore by modulating the prioritization of search costs our model canseamlessly alternate between Brownian and Cauchy walk dynamics. This model hasthe potential to be utilized for exploring the parameter space of anoptimization problem. |


| Item |Content|
| --- |---|
|idx| 2405.07131v1 |
|title| MAxPrototyper: A Multi-Agent Generation System for Interactive User Interface Prototyping |
|authors| Mingyue YuanJieshan ChenAaron Quigley
|links| http://arxiv.org/abs/2405.07131v1 |
|updated| 2024-05-12 01:57:09 UTC |
|summary| In automated user interactive design designers face key challengesincluding accurate representation of user intent crafting high-qualitycomponents and ensuring both aesthetic and semantic consistency. Addressingthese challenges we introduce MAxPrototyper our human-centered multi-agentsystem for interactive design generation. The core of MAxPrototyper is a themedesign agent. It coordinates with specialized sub-agents each responsible forgenerating specific parts of the design. Through an intuitive online interfaceusers can control the design process by providing text descriptions and layout.Enhanced by improved language and image generation models MAxPrototypergenerates each component with careful detail and contextual understanding. Itsmulti-agent architecture enables a multi-round interaction capability betweenthe system and users facilitating precise and customized design adjustmentsthroughout the creation process. |


| Item |Content|
| --- |---|
|idx| 2405.06161v1 |
|title| (A Partial Survey of) Decentralized, Cooperative Multi-Agent Reinforcement Learning |
|authors| Christopher Amato
|links| http://arxiv.org/abs/2405.06161v1 |
|updated| 2024-05-10 00:50:08 UTC |
|summary| Multi-agent reinforcement learning MARL has exploded in popularity inrecent years. Many approaches have been developed but they can be divided intothree main types: centralized training and execution CTE centralizedtraining for decentralized execution CTDE and Decentralized training andexecution DTE.  Decentralized training and execution methods make the fewest assumptions andare often simple to implement. In fact as Ill discuss any single-agent RLmethod can be used for DTE by just letting each agent learn separately. Ofcourse there are pros and cons to such approaches as we discuss below. It isworth noting that DTE is required if no offline coordination is available. Thatis if all agents must learn during online interactions without priorcoordination learning and execution must both be decentralized. DTE methodscan be applied in cooperative competitive or mixed cases but this text willfocus on the cooperative MARL case.  In this text I will first give a brief description of the cooperative MARLproblem in the form of the Dec-POMDP. Then I will discuss value-based DTEmethods starting with independent Q-learning and its extensions and thendiscuss the extension to the deep case with DQN the additional complicationsthis causes and methods that have been developed to attempt to address theseissues. Next I will discuss policy gradient DTE methods starting withindependent REINFORCE i.e. vanilla policy gradient and then extending tothe actor-critic case and deep variants such as independent PPO. Finally Iwill discuss some general topics related to DTE and future directions. |


| Item |Content|
| --- |---|
|idx| 2405.05950v1 |
|title| Federated Combinatorial Multi-Agent Multi-Armed Bandits |
|authors| Fares FouratiMohamed-Slim AlouiniVaneet Aggarwal
|links| http://arxiv.org/abs/2405.05950v1 |
|updated| 2024-05-09 17:40:09 UTC |
|summary| This paper introduces a federated learning framework tailored for onlinecombinatorial optimization with bandit feedback. In this setting agents selectsubsets of arms observe noisy rewards for these subsets without accessingindividual arm information and can cooperate and share information at specificintervals. Our framework transforms any offline resilient single-agentalpha-epsilon-approximation algorithm having a complexity oftildemathcalOfracpsiepsilonbeta where the logarithm isomitted for some function psi and constant beta into an onlinemulti-agent algorithm with m communicating agents and an alpha-regret ofno more than tildemathcalOm-frac13beta psifrac13betaTfrac2beta3beta. This approach not only eliminates the epsilonapproximation error but also ensures sublinear growth with respect to the timehorizon T and demonstrates a linear speedup with an increasing number ofcommunicating agents. Additionally the algorithm is notablycommunication-efficient requiring only a sublinear number of communicationrounds quantified as tildemathcalOleftpsiTfracbetabeta1right. Furthermore the framework has beensuccessfully applied to online stochastic submodular maximization using variousoffline algorithms yielding the first results for both single-agent andmulti-agent settings and recovering specialized single-agent theoreticalguarantees. We empirically validate our approach to a stochastic datasummarization problem illustrating the effectiveness of the proposedframework even in single-agent scenarios. |


| Item |Content|
| --- |---|
|idx| 2405.05870v1 |
|title| Selecting the Most Conflicting Pair of Candidates |
|authors| Théo DelemazureŁukasz JaneczkoAndrzej KaczmarczykStanisław Szufa
|links| http://arxiv.org/abs/2405.05870v1 |
|updated| 2024-05-09 16:00:20 UTC |
|summary| We study committee elections from a perspective of finding the mostconflicting candidates that is candidates that imply the largest amount ofconflict as per voter preferences. By proposing basic axioms to capture thisobjective we show that none of the prominent multiwinner voting rules meetthem. Consequently we design committee voting rules compliant with ourdesiderata introducing conflictual voting rules. A subsequent deepenedanalysis sheds more light on how they operate. Our investigation identifiesvarious aspects of conflict for which we come up with relevant axioms andquantitative measures which may be of independent interest. We support ourtheoretical study with experiments on both real-life and synthetic data. |


