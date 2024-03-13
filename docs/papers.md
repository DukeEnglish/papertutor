# cs.CL 

| Item |Content|
| --- |---|
|idx| 2403.07872v1 |
|title| Rethinking Generative Large Language Model Evaluation for Semantic Comprehension |
|authors| Fangyun WeiXi ChenLin Luo
|links| http://arxiv.org/abs/2403.07872v1 |
|updated| 2024-03-12 17:59:48 UTC |
|summary| Despite their sophisticated capabilities large language models LLMsencounter a major hurdle in effective assessment. This paper first revisits theprevalent evaluation method-multiple choice question answering MCQA whichallows for straightforward accuracy measurement. Through a comprehensiveevaluation of 24 models across 11 benchmarks we highlight several potentialdrawbacks of MCQA for instance the inconsistency between the MCQA evaluationand the generation of open-ended responses in practical scenarios. In responsewe introduce an RWQ-Elo rating system engaging 24 LLMs such as GPT-4 GPT-3.5Google-Gemini-Pro and LLaMA-1/-2 in a two-player competitive format withGPT-4 serving as the judge. Each LLM receives an Elo rating thereafter. Thissystem is designed to mirror real-world usage and for this purpose we havecompiled a new benchmark called Real-world questions RWQ comprising20772 authentic user inquiries. Additionally we thoroughly analyze thecharacteristics of our system and compare it with prior leaderboards likeAlpacaEval and MT-Bench. Our analysis reveals the stability of our RWQ-Elosystem the feasibility of registering new models and its potential to reshapeLLM leaderboards. |


| Item |Content|
| --- |---|
|idx| 2403.07865v1 |
|title| Exploring Safety Generalization Challenges of Large Language Models via Code |
|authors| Qibing RenChang GaoJing ShaoJunchi YanXin TanWai LamLizhuang Ma
|links| http://arxiv.org/abs/2403.07865v1 |
|updated| 2024-03-12 17:55:38 UTC |
|summary| The rapid advancement of Large Language Models LLMs has brought aboutremarkable capabilities in natural language processing but also raised concernsabout their potential misuse. While strategies like supervised fine-tuning andreinforcement learning from human feedback have enhanced their safety thesemethods primarily focus on natural languages which may not generalize to otherdomains. This paper introduces CodeAttack a framework that transforms naturallanguage inputs into code inputs presenting a novel environment for testingthe safety generalization of LLMs. Our comprehensive studies onstate-of-the-art LLMs including GPT-4 Claude-2 and Llama-2 series reveal acommon safety vulnerability of these models against code input: CodeAttackconsistently bypasses the safety guardrails of all models more than 80 of thetime. Furthermore we find that a larger distribution gap between CodeAttackand natural language leads to weaker safety generalization such as encodingnatural language input with data structures or using less popular programminglanguages. These findings highlight new safety risks in the code domain and theneed for more robust safety alignment algorithms to match the code capabilitiesof LLMs. |


| Item |Content|
| --- |---|
|idx| 2403.07825v1 |
|title| The Missing Piece in Model Editing: A Deep Dive into the Hidden Damage Brought By Model Editing |
|authors| Jianchen WangZhouhong GuZhuozhi XiongHongwei FengYanghua Xiao
|links| http://arxiv.org/abs/2403.07825v1 |
|updated| 2024-03-12 17:04:28 UTC |
|summary| Large Language Models have revolutionized numerous tasks with theirremarkable efficacy.However the editing of these models crucial forrectifying outdated or erroneous information often leads to a complex issueknown as the ripple effect in the hidden space. This effect while difficult todetect can significantly impede the efficacy of model editing tasks anddeteriorate model performance.This paper addresses this scientific challenge byproposing a novel evaluation methodology Graphical Outlier Relation basedAssessmentGORA which quantitatively evaluates the adaptations of the modeland the subsequent impact of editing. Furthermore we introduce the SelectiveOutlier Re-Editing ApproachSORA a model editing method designed to mitigatethis ripple effect. Our comprehensive evaluations reveal that the ripple effectin the hidden space is a significant issue in all current model editingmethods. However our proposed methods GORA and SORA effectively identify andalleviate this issue respectively contributing to the advancement of LLMediting techniques. |


| Item |Content|
| --- |---|
|idx| 2403.07816v1 |
|title| Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM |
|authors| Sainbayar SukhbaatarOlga GolovnevaVasu SharmaHu XuXi Victoria LinBaptiste RozièreJacob KahnDaniel LiWen-tau YihJason WestonXian Li
|links| http://arxiv.org/abs/2403.07816v1 |
|updated| 2024-03-12 16:54:58 UTC |
|summary| We investigate efficient methods for training Large Language Models LLMs topossess capabilities in multiple specialized domains such as coding mathreasoning and world knowledge. Our method named Branch-Train-MiX BTX startsfrom a seed model which is branched to train experts in embarrassinglyparallel fashion with high throughput and reduced communication cost. Afterindividual experts are asynchronously trained BTX brings together theirfeedforward parameters as experts in Mixture-of-Expert MoE layers andaverages the remaining parameters followed by an MoE-finetuning stage to learntoken-level routing. BTX generalizes two special cases the Branch-Train-Mergemethod which does not have the MoE finetuning stage to learn routing andsparse upcycling which omits the stage of training experts asynchronously.Compared to alternative approaches BTX achieves the best accuracy-efficiencytradeoff. |


| Item |Content|
| --- |---|
|idx| 2403.07809v1 |
|title| pyvene: A Library for Understanding and Improving PyTorch Models via Interventions |
|authors| Zhengxuan WuAtticus GeigerAryaman AroraJing HuangZheng WangNoah D. GoodmanChristopher D. ManningChristopher Potts
|links| http://arxiv.org/abs/2403.07809v1 |
|updated| 2024-03-12 16:46:54 UTC |
|summary| Interventions on model-internal states are fundamental operations in manyareas of AI including model editing steering robustness andinterpretability. To facilitate such research we introduce textbfpyvenean open-source Python library that supports customizable interventions on arange of different PyTorch modules. textbfpyvene supports complexintervention schemes with an intuitive configuration format and itsinterventions can be static or include trainable parameters. We show howtextbfpyvene provides a unified and extensible framework for performinginterventions on neural models and sharing the intervened upon models withothers. We illustrate the power of the library via interpretability analysesusing causal abstraction and knowledge localization. We publish our librarythrough Python Package Index PyPI and provide code documentation andtutorials at https://github.com/stanfordnlp/pyvene. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2403.07869v1 |
|title| TeleMoMa: A Modular and Versatile Teleoperation System for Mobile Manipulation |
|authors| Shivin DassWensi AiYuqian JiangSamik SinghJiaheng HuRuohan ZhangPeter StoneBen AbbatematteoRoberto Martin-Martin
|links| http://arxiv.org/abs/2403.07869v1 |
|updated| 2024-03-12 17:58:01 UTC |
|summary| A critical bottleneck limiting imitation learning in robotics is the lack ofdata. This problem is more severe in mobile manipulation where collectingdemonstrations is harder than in stationary manipulation due to the lack ofavailable and easy-to-use teleoperation interfaces. In this work wedemonstrate TeleMoMa a general and modular interface for whole-bodyteleoperation of mobile manipulators. TeleMoMa unifies multiple humaninterfaces including RGB and depth cameras virtual reality controllerskeyboard joysticks etc. and any combination thereof. In its more accessibleversion TeleMoMa works using simply vision e.g. an RGB-D camera loweringthe entry bar for humans to provide mobile manipulation demonstrations. Wedemonstrate the versatility of TeleMoMa by teleoperating several existingmobile manipulators - PAL Tiago Toyota HSR and Fetch - in simulation andthe real world. We demonstrate the quality of the demonstrations collected withTeleMoMa by training imitation learning policies for mobile manipulation tasksinvolving synchronized whole-body motion. Finally we also show that TeleMoMasteleoperation channel enables teleoperation on site looking at the robot orremote sending commands and observations through a computer network andperform user studies to evaluate how easy it is for novice users to learn tocollect demonstrations with different combinations of human interfaces enabledby our system. We hope TeleMoMa becomes a helpful tool for the communityenabling researchers to collect whole-body mobile manipulation demonstrations.For more information and video resultshttps://robin-lab.cs.utexas.edu/telemoma-web. |


| Item |Content|
| --- |---|
|idx| 2403.07865v1 |
|title| Exploring Safety Generalization Challenges of Large Language Models via Code |
|authors| Qibing RenChang GaoJing ShaoJunchi YanXin TanWai LamLizhuang Ma
|links| http://arxiv.org/abs/2403.07865v1 |
|updated| 2024-03-12 17:55:38 UTC |
|summary| The rapid advancement of Large Language Models LLMs has brought aboutremarkable capabilities in natural language processing but also raised concernsabout their potential misuse. While strategies like supervised fine-tuning andreinforcement learning from human feedback have enhanced their safety thesemethods primarily focus on natural languages which may not generalize to otherdomains. This paper introduces CodeAttack a framework that transforms naturallanguage inputs into code inputs presenting a novel environment for testingthe safety generalization of LLMs. Our comprehensive studies onstate-of-the-art LLMs including GPT-4 Claude-2 and Llama-2 series reveal acommon safety vulnerability of these models against code input: CodeAttackconsistently bypasses the safety guardrails of all models more than 80 of thetime. Furthermore we find that a larger distribution gap between CodeAttackand natural language leads to weaker safety generalization such as encodingnatural language input with data structures or using less popular programminglanguages. These findings highlight new safety risks in the code domain and theneed for more robust safety alignment algorithms to match the code capabilitiesof LLMs. |


| Item |Content|
| --- |---|
|idx| 2403.07839v1 |
|title| MoPE-CLIP: Structured Pruning for Efficient Vision-Language Models with Module-wise Pruning Error Metric |
|authors| Haokun LinHaoli BaiZhili LiuLu HouMuyi SunLinqi SongYing WeiZhenan Sun
|links| http://arxiv.org/abs/2403.07839v1 |
|updated| 2024-03-12 17:24:26 UTC |
|summary| Vision-language pre-trained models have achieved impressive performance onvarious downstream tasks. However their large model sizes hinder theirutilization on platforms with limited computational resources. We find thatdirectly using smaller pre-trained models and applying magnitude-based pruningon CLIP models leads to inflexibility and inferior performance. Recent effortsfor VLP compression either adopt uni-modal compression metrics resulting inlimited performance or involve costly mask-search processes with learnablemasks. In this paper we first propose the Module-wise Pruning Error MoPEmetric accurately assessing CLIP module importance by performance decline oncross-modal tasks. Using the MoPE metric we introduce a unified pruningframework applicable to both pre-training and task-specific fine-tuningcompression stages. For pre-training MoPE-CLIP effectively leverages knowledgefrom the teacher model significantly reducing pre-training costs whilemaintaining strong zero-shot capabilities. For fine-tuning consecutive pruningfrom width to depth yields highly competitive task-specific models. Extensiveexperiments in two stages demonstrate the effectiveness of the MoPE metric andMoPE-CLIP outperforms previous state-of-the-art VLP compression methods. |


| Item |Content|
| --- |---|
|idx| 2403.07818v1 |
|title| Label Dropout: Improved Deep Learning Echocardiography Segmentation Using Multiple Datasets With Domain Shift and Partial Labelling |
|authors| Iman IslamEsther Puyol-AntónBram RuijsinkAndrew J. ReaderAndrew P. King
|links| http://arxiv.org/abs/2403.07818v1 |
|updated| 2024-03-12 16:57:56 UTC |
|summary| Echocardiography echo is the first imaging modality used when assessingcardiac function. The measurement of functional biomarkers from echo reliesupon the segmentation of cardiac structures and deep learning models have beenproposed to automate the segmentation process. However in order to translatethese tools to widespread clinical use it is important that the segmentationmodels are robust to a wide variety of images e.g. acquired from differentscanners by operators with different levels of expertise etc.. To achievethis level of robustness it is necessary that the models are trained withmultiple diverse datasets. A significant challenge faced when training withmultiple diverse datasets is the variation in label presence i.e. the combineddata are often partially-labelled. Adaptations of the cross entropy lossfunction have been proposed to deal with partially labelled data. In this paperwe show that training naively with such a loss function and multiple diversedatasets can lead to a form of shortcut learning where the model associateslabel presence with domain characteristics leading to a drop in performance.To address this problem we propose a novel label dropout scheme to break thelink between domain characteristics and the presence or absence of labels. Wedemonstrate that label dropout improves echo segmentation Dice score by 62 and25 on two cardiac structures when training using multiple diverse partiallylabelled datasets. |


| Item |Content|
| --- |---|
|idx| 2403.07816v1 |
|title| Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM |
|authors| Sainbayar SukhbaatarOlga GolovnevaVasu SharmaHu XuXi Victoria LinBaptiste RozièreJacob KahnDaniel LiWen-tau YihJason WestonXian Li
|links| http://arxiv.org/abs/2403.07816v1 |
|updated| 2024-03-12 16:54:58 UTC |
|summary| We investigate efficient methods for training Large Language Models LLMs topossess capabilities in multiple specialized domains such as coding mathreasoning and world knowledge. Our method named Branch-Train-MiX BTX startsfrom a seed model which is branched to train experts in embarrassinglyparallel fashion with high throughput and reduced communication cost. Afterindividual experts are asynchronously trained BTX brings together theirfeedforward parameters as experts in Mixture-of-Expert MoE layers andaverages the remaining parameters followed by an MoE-finetuning stage to learntoken-level routing. BTX generalizes two special cases the Branch-Train-Mergemethod which does not have the MoE finetuning stage to learn routing andsparse upcycling which omits the stage of training experts asynchronously.Compared to alternative approaches BTX achieves the best accuracy-efficiencytradeoff. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2403.07869v1 |
|title| TeleMoMa: A Modular and Versatile Teleoperation System for Mobile Manipulation |
|authors| Shivin DassWensi AiYuqian JiangSamik SinghJiaheng HuRuohan ZhangPeter StoneBen AbbatematteoRoberto Martin-Martin
|links| http://arxiv.org/abs/2403.07869v1 |
|updated| 2024-03-12 17:58:01 UTC |
|summary| A critical bottleneck limiting imitation learning in robotics is the lack ofdata. This problem is more severe in mobile manipulation where collectingdemonstrations is harder than in stationary manipulation due to the lack ofavailable and easy-to-use teleoperation interfaces. In this work wedemonstrate TeleMoMa a general and modular interface for whole-bodyteleoperation of mobile manipulators. TeleMoMa unifies multiple humaninterfaces including RGB and depth cameras virtual reality controllerskeyboard joysticks etc. and any combination thereof. In its more accessibleversion TeleMoMa works using simply vision e.g. an RGB-D camera loweringthe entry bar for humans to provide mobile manipulation demonstrations. Wedemonstrate the versatility of TeleMoMa by teleoperating several existingmobile manipulators - PAL Tiago Toyota HSR and Fetch - in simulation andthe real world. We demonstrate the quality of the demonstrations collected withTeleMoMa by training imitation learning policies for mobile manipulation tasksinvolving synchronized whole-body motion. Finally we also show that TeleMoMasteleoperation channel enables teleoperation on site looking at the robot orremote sending commands and observations through a computer network andperform user studies to evaluate how easy it is for novice users to learn tocollect demonstrations with different combinations of human interfaces enabledby our system. We hope TeleMoMa becomes a helpful tool for the communityenabling researchers to collect whole-body mobile manipulation demonstrations.For more information and video resultshttps://robin-lab.cs.utexas.edu/telemoma-web. |


| Item |Content|
| --- |---|
|idx| 2403.07865v1 |
|title| Exploring Safety Generalization Challenges of Large Language Models via Code |
|authors| Qibing RenChang GaoJing ShaoJunchi YanXin TanWai LamLizhuang Ma
|links| http://arxiv.org/abs/2403.07865v1 |
|updated| 2024-03-12 17:55:38 UTC |
|summary| The rapid advancement of Large Language Models LLMs has brought aboutremarkable capabilities in natural language processing but also raised concernsabout their potential misuse. While strategies like supervised fine-tuning andreinforcement learning from human feedback have enhanced their safety thesemethods primarily focus on natural languages which may not generalize to otherdomains. This paper introduces CodeAttack a framework that transforms naturallanguage inputs into code inputs presenting a novel environment for testingthe safety generalization of LLMs. Our comprehensive studies onstate-of-the-art LLMs including GPT-4 Claude-2 and Llama-2 series reveal acommon safety vulnerability of these models against code input: CodeAttackconsistently bypasses the safety guardrails of all models more than 80 of thetime. Furthermore we find that a larger distribution gap between CodeAttackand natural language leads to weaker safety generalization such as encodingnatural language input with data structures or using less popular programminglanguages. These findings highlight new safety risks in the code domain and theneed for more robust safety alignment algorithms to match the code capabilitiesof LLMs. |


| Item |Content|
| --- |---|
|idx| 2403.07857v1 |
|title| Fairness Feedback Loops: Training on Synthetic Data Amplifies Bias |
|authors| Sierra WyllieIlia ShumailovNicolas Papernot
|links| http://arxiv.org/abs/2403.07857v1 |
|updated| 2024-03-12 17:48:08 UTC |
|summary| Model-induced distribution shifts MIDS occur as previous model outputspollute new model training sets over generations of models. This is known asmodel collapse in the case of generative models and performative prediction orunfairness feedback loops for supervised models. When a model induces adistribution shift it also encodes its mistakes biases and unfairnesses intothe ground truth of its data ecosystem. We introduce a framework that allows usto track multiple MIDS over many generations finding that they can lead toloss in performance fairness and minoritized group representation even ininitially unbiased datasets. Despite these negative consequences we identifyhow models might be used for positive intentional interventions in their dataecosystems providing redress for historical discrimination through a frameworkcalled algorithmic reparation AR. We simulate AR interventions by curatingrepresentative training batches for stochastic gradient descent to demonstratehow AR can improve upon the unfairnesses of models and data ecosystems subjectto other MIDS. Our work takes an important step towards identifyingmitigating and taking accountability for the unfair feedback loops enabled bythe idea that ML systems are inherently neutral and objective. |


| Item |Content|
| --- |---|
|idx| 2403.07856v1 |
|title| Quantum Support Vector Machine for Prostate Cancer Detection: A Performance Analysis |
|authors| Walid El MaouakiTaoufik SaidMohamed Bennai
|links| http://arxiv.org/abs/2403.07856v1 |
|updated| 2024-03-12 17:46:38 UTC |
|summary| This study addresses the urgent need for improved prostate cancer detectionmethods by harnessing the power of advanced technological solutions. Weintroduce the application of Quantum Support Vector Machine QSVM to thiscritical healthcare challenge showcasing an enhancement in diagnosticperformance over the classical Support Vector Machine SVM approach. Our studynot only outlines the remarkable improvements in diagnostic performance made byQSVM over the classic SVM technique but it delves into the advancementsbrought about by the quantum feature map architecture which has been carefullyidentified and evaluated ensuring it aligns seamlessly with the uniquecharacteristics of our prostate cancer dataset. This architecture succeded increating a distinct feature space enabling the detection of complexnon-linear patterns in the data. The findings reveal not only a comparableaccuracy with classical SVM 92 but also a 7.14 increase insensitivity and a notably high F1-Score 93.33. This studys importantcombination of quantum computing in medical diagnostics marks a pivotal stepforward in cancer detection offering promising implications for the future ofhealthcare technology. |


| Item |Content|
| --- |---|
|idx| 2403.07854v1 |
|title| Distilling the Knowledge in Data Pruning |
|authors| Emanuel Ben-BaruchAdam BotachIgor KviatkovskyManoj AggarwalGérard Medioni
|links| http://arxiv.org/abs/2403.07854v1 |
|updated| 2024-03-12 17:44:45 UTC |
|summary| With the increasing size of datasets used for training neural networks datapruning becomes an attractive field of research. However most current datapruning algorithms are limited in their ability to preserve accuracy comparedto models trained on the full data especially in high pruning regimes. In thispaper we explore the application of data pruning while incorporating knowledgedistillation KD when training on a pruned subset. That is rather thanrelying solely on ground-truth labels we also use the soft predictions from ateacher network pre-trained on the complete data. By integrating KD intotraining we demonstrate significant improvement across datasets pruningmethods and on all pruning fractions. We first establish a theoreticalmotivation for employing self-distillation to improve training on pruned data.Then we empirically make a compelling and highly practical observation: usingKD simple random pruning is comparable or superior to sophisticated pruningmethods across all pruning regimes. On ImageNet for example we achievesuperior accuracy despite training on a random subset of only 50 of the data.Additionally we demonstrate a crucial connection between the pruning factorand the optimal knowledge distillation weight. This helps mitigate the impactof samples with noisy labels and low-quality images retained by typical pruningalgorithms. Finally we make an intriguing observation: when using lowerpruning fractions larger teachers lead to accuracy degradation whilesurprisingly employing teachers with a smaller capacity than the students mayimprove results. Our code will be made available. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2403.07874v1 |
|title| Beyond Text: Frozen Large Language Models in Visual Signal Comprehension |
|authors| Lei ZhuFangyun WeiYanye Lu
|links| http://arxiv.org/abs/2403.07874v1 |
|updated| 2024-03-12 17:59:51 UTC |
|summary| In this work we investigate the potential of a large language model LLM todirectly comprehend visual signals without the necessity of fine-tuning onmulti-modal datasets. The foundational concept of our method views an image asa linguistic entity and translates it to a set of discrete words derived fromthe LLMs vocabulary. To achieve this we present the Vision-to-LanguageTokenizer abbreviated as V2T Tokenizer which transforms an image into aforeign language with the combined aid of an encoder-decoder the LLMvocabulary and a CLIP model. With this innovative image encoding the LLMgains the ability not only for visual comprehension but also for imagedenoising and restoration in an auto-regressive fashion-crucially without anyfine-tuning. We undertake rigorous experiments to validate our methodencompassing understanding tasks like image recognition image captioning andvisual question answering as well as image denoising tasks like inpaintingoutpainting deblurring and shift restoration. Code and models are availableat https://github.com/zh460045050/V2L-Tokenizer. |


| Item |Content|
| --- |---|
|idx| 2403.07860v1 |
|title| Bridging Different Language Models and Generative Vision Models for Text-to-Image Generation |
|authors| Shihao ZhaoShaozhe HaoBojia ZiHuaizhe XuKwan-Yee K. Wong
|links| http://arxiv.org/abs/2403.07860v1 |
|updated| 2024-03-12 17:50:11 UTC |
|summary| Text-to-image generation has made significant advancements with theintroduction of text-to-image diffusion models. These models typically consistof a language model that interprets user prompts and a vision model thatgenerates corresponding images. As language and vision models continue toprogress in their respective domains there is a great potential in exploringthe replacement of components in text-to-image diffusion models with moreadvanced counterparts. A broader research objective would therefore be toinvestigate the integration of any two unrelated language and generative visionmodels for text-to-image generation. In this paper we explore this objectiveand propose LaVi-Bridge a pipeline that enables the integration of diversepre-trained language models and generative vision models for text-to-imagegeneration. By leveraging LoRA and adapters LaVi-Bridge offers a flexible andplug-and-play approach without requiring modifications to the original weightsof the language and vision models. Our pipeline is compatible with variouslanguage models and generative vision models accommodating differentstructures. Within this framework we demonstrate that incorporating superiormodules such as more advanced language models or generative vision modelsresults in notable improvements in capabilities like text alignment or imagequality. Extensive evaluations have been conducted to verify the effectivenessof LaVi-Bridge. Code is available athttps://github.com/ShihaoZhaoZSH/LaVi-Bridge. |


| Item |Content|
| --- |---|
|idx| 2403.07854v1 |
|title| Distilling the Knowledge in Data Pruning |
|authors| Emanuel Ben-BaruchAdam BotachIgor KviatkovskyManoj AggarwalGérard Medioni
|links| http://arxiv.org/abs/2403.07854v1 |
|updated| 2024-03-12 17:44:45 UTC |
|summary| With the increasing size of datasets used for training neural networks datapruning becomes an attractive field of research. However most current datapruning algorithms are limited in their ability to preserve accuracy comparedto models trained on the full data especially in high pruning regimes. In thispaper we explore the application of data pruning while incorporating knowledgedistillation KD when training on a pruned subset. That is rather thanrelying solely on ground-truth labels we also use the soft predictions from ateacher network pre-trained on the complete data. By integrating KD intotraining we demonstrate significant improvement across datasets pruningmethods and on all pruning fractions. We first establish a theoreticalmotivation for employing self-distillation to improve training on pruned data.Then we empirically make a compelling and highly practical observation: usingKD simple random pruning is comparable or superior to sophisticated pruningmethods across all pruning regimes. On ImageNet for example we achievesuperior accuracy despite training on a random subset of only 50 of the data.Additionally we demonstrate a crucial connection between the pruning factorand the optimal knowledge distillation weight. This helps mitigate the impactof samples with noisy labels and low-quality images retained by typical pruningalgorithms. Finally we make an intriguing observation: when using lowerpruning fractions larger teachers lead to accuracy degradation whilesurprisingly employing teachers with a smaller capacity than the students mayimprove results. Our code will be made available. |


| Item |Content|
| --- |---|
|idx| 2403.07851v1 |
|title| 12 mJ per Class On-Device Online Few-Shot Class-Incremental Learning |
|authors| Yoga Esa WibowoCristian CioflanThorir Mar IngolfssonMichael HerscheLeo ZhaoAbbas RahimiLuca Benini
|links| http://arxiv.org/abs/2403.07851v1 |
|updated| 2024-03-12 17:43:20 UTC |
|summary| Few-Shot Class-Incremental Learning FSCIL enables machine learning systemsto expand their inference capabilities to new classes using only a few labeledexamples without forgetting the previously learned classes. Classicalbackpropagation-based learning and its variants are often unsuitable forbattery-powered memory-constrained systems at the extreme edge. In this workwe introduce Online Few-Shot Class-Incremental Learning O-FSCIL based on alightweight model consisting of a pretrained and metalearned feature extractorand an expandable explicit memory storing the class prototypes. Thearchitecture is pretrained with a novel feature orthogonality regularizationand metalearned with a multi-margin loss. For learning a new class ourapproach extends the explicit memory with novel class prototypes while theremaining architecture is kept frozen. This allows learning previously unseenclasses based on only a few examples with one single pass hence online.O-FSCIL obtains an average accuracy of 68.62 on the FSCIL CIFAR100 benchmarkachieving state-of-the-art results. Tailored for ultra-low-power platforms weimplement O-FSCIL on the 60 mW GAP9 microcontroller demonstrating onlinelearning capabilities within just 12 mJ per new class. |


| Item |Content|
| --- |---|
|idx| 2403.07839v1 |
|title| MoPE-CLIP: Structured Pruning for Efficient Vision-Language Models with Module-wise Pruning Error Metric |
|authors| Haokun LinHaoli BaiZhili LiuLu HouMuyi SunLinqi SongYing WeiZhenan Sun
|links| http://arxiv.org/abs/2403.07839v1 |
|updated| 2024-03-12 17:24:26 UTC |
|summary| Vision-language pre-trained models have achieved impressive performance onvarious downstream tasks. However their large model sizes hinder theirutilization on platforms with limited computational resources. We find thatdirectly using smaller pre-trained models and applying magnitude-based pruningon CLIP models leads to inflexibility and inferior performance. Recent effortsfor VLP compression either adopt uni-modal compression metrics resulting inlimited performance or involve costly mask-search processes with learnablemasks. In this paper we first propose the Module-wise Pruning Error MoPEmetric accurately assessing CLIP module importance by performance decline oncross-modal tasks. Using the MoPE metric we introduce a unified pruningframework applicable to both pre-training and task-specific fine-tuningcompression stages. For pre-training MoPE-CLIP effectively leverages knowledgefrom the teacher model significantly reducing pre-training costs whilemaintaining strong zero-shot capabilities. For fine-tuning consecutive pruningfrom width to depth yields highly competitive task-specific models. Extensiveexperiments in two stages demonstrate the effectiveness of the MoPE metric andMoPE-CLIP outperforms previous state-of-the-art VLP compression methods. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2403.07862v1 |
|title| Low coordinate degree algorithms I: Universality of computational thresholds for hypothesis testing |
|authors| Dmitriy Kunisky
|links| http://arxiv.org/abs/2403.07862v1 |
|updated| 2024-03-12 17:52:35 UTC |
|summary| We study when low coordinate degree functions LCDF -- linear combinationsof functions depending on small subsets of entries of a vector -- canhypothesis test between high-dimensional probability measures. These functionsare a generalization proposed in Hopkins 2018 thesis but seldom studiedsince of low degree polynomials LDP a class widely used in recentliterature as a proxy for all efficient algorithms for tasks in statistics andoptimization. Instead of the orthogonal polynomial decompositions used in LDPcalculations our analysis of LCDF is based on the Efron-Stein or ANOVAdecomposition making it much more broadly applicable. By way of illustrationwe prove channel universality for the success of LCDF in testing for thepresence of sufficiently dilute random signals through noisy channels: theefficacy of LCDF depends on the channel only through the scalar Fisherinformation for a class of channels including nearly arbitrary additive i.i.d.noise and nearly arbitrary exponential families. As applications we extendlower bounds against LDP for spiked matrix and tensor models under additiveGaussian noise to lower bounds against LCDF under general noisy channels. Wealso give a simple and unified treatment of the effect of censoring models byerasing observations at random and of quantizing models by taking the sign ofthe observations. These results are the first computational lower boundsagainst any large class of algorithms for all of these models when the channelis not one of a few special cases and thereby give the first substantialevidence for the universality of several statistical-to-computational gaps. |


| Item |Content|
| --- |---|
|idx| 2403.07780v1 |
|title| FairRR: Pre-Processing for Group Fairness through Randomized Response |
|authors| Xianli ZengJoshua WardGuang Cheng
|links| http://arxiv.org/abs/2403.07780v1 |
|updated| 2024-03-12 16:08:47 UTC |
|summary| The increasing usage of machine learning models in consequentialdecision-making processes has spurred research into the fairness of thesesystems. While significant work has been done to study group fairness in thein-processing and post-processing setting there has been little thattheoretically connects these results to the pre-processing domain. This paperproposes that achieving group fairness in downstream models can be formulatedas finding the optimal design matrix in which to modify a response variable ina Randomized Response framework. We show that measures of group fairness can bedirectly controlled for with optimal model utility proposing a pre-processingalgorithm called FairRR that yields excellent downstream model utility andfairness. |


| Item |Content|
| --- |---|
|idx| 2403.07745v1 |
|title| Probabilistic Easy Variational Causal Effect |
|authors| Usef FaghihiAmir Saki
|links| http://arxiv.org/abs/2403.07745v1 |
|updated| 2024-03-12 15:28:21 UTC |
|summary| Let X and Z be random vectors and YgXZ. In this paper on the onehand for the case that X and Z are continuous by using the ideas from thetotal variation and the flux of g we develop a point of view in causalinference capable of dealing with a broad domain of causal problems. Indeed wefocus on a function called Probabilistic Easy Variational Causal EffectPEACE which can measure the direct causal effect of X on Y with respectto continuously and interventionally changing the values of X while keepingthe value of Z constant. PEACE is a function of dge 0 which is a degreemanaging the strengths of probability density values fxz. On the otherhand we generalize the above idea for the discrete case and show itscompatibility with the continuous case. Further we investigate some propertiesof PEACE using measure theoretical concepts. Furthermore we provide someidentifiability criteria and several examples showing the generic capability ofPEACE. We note that PEACE can deal with the causal problems for whichmicro-level or just macro-level changes in the value of the input variables areimportant. Finally PEACE is stable under small changes in partialg_in/partial x and the joint distribution of X and Z where g_in isobtained from g by removing all functional relationships defining X andZ. |


| Item |Content|
| --- |---|
|idx| 2403.07735v1 |
|title| The Minimax Rate of HSIC Estimation for Translation-Invariant Kernels |
|authors| Florian KalinkeZoltan Szabo
|links| http://arxiv.org/abs/2403.07735v1 |
|updated| 2024-03-12 15:13:21 UTC |
|summary| Kernel techniques are among the most influential approaches in data scienceand statistics. Under mild conditions the reproducing kernel Hilbert spaceassociated to a kernel is capable of encoding the independence of Mge 2random variables. Probably the most widespread independence measure relying onkernels is the so-called Hilbert-Schmidt independence criterion HSIC alsoreferred to as distance covariance in the statistics literature. Despitevarious existing HSIC estimators designed since its introduction close to twodecades ago the fundamental question of the rate at which HSIC can beestimated is still open. In this work we prove that the minimax optimal rateof HSIC estimation on mathbb Rd for Borel measures containing the Gaussianswith continuous bounded translation-invariant characteristic kernels ismathcal Oleftn-1/2right. Specifically our result implies theoptimality in the minimax sense of many of the most-frequently used estimatorsincluding the U-statistic the V-statistic and the Nystrom-based one onmathbb Rd. |


| Item |Content|
| --- |---|
|idx| 2403.07728v1 |
|title| CAS: A General Algorithm for Online Selective Conformal Prediction with FCR Control |
|authors| Yajie BaoYuyang HuoHaojie RenChangliang Zou
|links| http://arxiv.org/abs/2403.07728v1 |
|updated| 2024-03-12 15:07:20 UTC |
|summary| We study the problem of post-selection predictive inference in an onlinefashion. To avoid devoting resources to unimportant units a preliminaryselection of the current individual before reporting its prediction interval iscommon and meaningful in online predictive tasks. Since the online selectioncauses a temporal multiplicity in the selected prediction intervals it isimportant to control the real-time false coverage-statement rate FCR tomeasure the averaged miscoverage error. We develop a general framework namedCAS Calibration after Adaptive Selection that can wrap around any predictionmodel and online selection rule to output post-selection prediction intervals.If the current individual is selected we first perform an adaptive selectionon historical data to construct a calibration set then output a conformalprediction interval for the unobserved label. We provide tractableconstructions for the calibration set for popular online selection rules. Weproved that CAS can achieve an exact selection-conditional coverage guaranteein the finite-sample and distribution-free regimes. For the decision-drivenselection rule including most online multiple-testing procedures CAS canexactly control the real-time FCR below the target level without anydistributional assumptions. For the online selection with symmetric thresholdswe establish the error bound for the control gap of FCR under milddistributional assumptions. To account for the distribution shift in onlinedata we also embed CAS into some recent dynamic conformal prediction methodsand examine the long-run FCR control. Numerical results on both synthetic andreal data corroborate that CAS can effectively control FCR around the targetlevel and yield more narrowed prediction intervals over existing baselinesacross various settings. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2403.07762v1 |
|title| Supporting Annotators with Affordances for Efficiently Labeling Conversational Data |
|authors| Austin Z. HenleyDavid Piorkowski
|links| http://arxiv.org/abs/2403.07762v1 |
|updated| 2024-03-12 15:51:10 UTC |
|summary| Without well-labeled ground truth data machine learning-based systems wouldnot be as ubiquitous as they are today but these systems rely on substantialamounts of correctly labeled data. Unfortunately crowdsourced labeling is timeconsuming and expensive. To address the concerns of effort and tedium wedesigned CAL a novel interface to aid in data labeling. We made several keydesign decisions for CAL which include preventing inapt labels from beingselected guiding users in selecting an appropriate label when they needassistance incorporating labeling documentation into the interface andproviding an efficient means to view previous labels. We implemented aproduction-quality implementation of CAL and report a user-study evaluationthat compares CAL to a standard spreadsheet. Key findings of our study includeusers using CAL reported lower cognitive load did not increase task timeusers rated CAL to be easier to use and users preferred CAL over thespreadsheet. |


| Item |Content|
| --- |---|
|idx| 2403.07721v1 |
|title| Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion |
|authors| Dongyang LiChen WeiShiying LiJiachen ZouQuanying Liu
|links| http://arxiv.org/abs/2403.07721v1 |
|updated| 2024-03-12 14:58:57 UTC |
|summary| How to decode human vision through neural signals has attracted along-standing interest in neuroscience and machine learning. Modern contrastivelearning and generative models improved the performance of fMRI-based visualdecoding and reconstruction. However the high cost and low temporal resolutionof fMRI limit their applications in brain-computer interfaces BCIs promptinga high need for EEG-based visual reconstruction. In this study we present anEEG-based visual reconstruction framework. It consists of a plug-and-play EEGencoder called the Adaptive Thinking Mapper ATM which is aligned with imageembeddings and a two-stage EEG guidance image generator that first transformsEEG features into image priors and then reconstructs the visual stimuli with apre-trained image generator. Our approach allows EEG embeddings to achievesuperior performance in image classification and retrieval tasks. Our two-stageimage generation strategy vividly reconstructs images seen by humans.Furthermore we analyzed the impact of signals from different time windows andbrain regions on decoding and reconstruction. The versatility of our frameworkis demonstrated in the magnetoencephalogram MEG data modality. We report thatEEG-based visual decoding achieves SOTA performance highlighting theportability low cost and high temporal resolution of EEG enabling a widerange of BCI applications. The code of ATM is available athttps://anonymous.4open.science/status/EEG_Image_decode-DEEF. |


| Item |Content|
| --- |---|
|idx| 2403.07627v1 |
|title| generAItor: Tree-in-the-Loop Text Generation for Language Model Explainability and Adaptation |
|authors| Thilo SpinnerRebecca KehlbeckRita SevastjanovaTobias StähleDaniel A. KeimOliver DeussenMennatallah El-Assady
|links| http://dx.doi.org/10.1145/3652028 |
|updated| 2024-03-12 13:09:15 UTC |
|summary| Large language models LLMs are widely deployed in various downstream taskse.g. auto-completion aided writing or chat-based text generation. Howeverthe considered output candidates of the underlying search algorithm areunder-explored and under-explained. We tackle this shortcoming by proposing atree-in-the-loop approach where a visual representation of the beam searchtree is the central component for analyzing explaining and adapting thegenerated outputs. To support these tasks we present generAItor a visualanalytics technique augmenting the central beam search tree with varioustask-specific widgets providing targeted visualizations and interactionpossibilities. Our approach allows interactions on multiple levels and offersan iterative pipeline that encompasses generating exploring and comparingoutput candidates as well as fine-tuning the model based on adapted data. Ourcase study shows that our tool generates new insights in gender bias analysisbeyond state-of-the-art template-based methods. Additionally we demonstratethe applicability of our approach in a qualitative user study. Finally wequantitatively evaluate the adaptability of the model to few samples asoccurring in text-generation use cases. |


| Item |Content|
| --- |---|
|idx| 2403.07613v1 |
|title| Imagine a dragon made of seaweed: How images enhance learning in Wikipedia |
|authors| Anita SilvaMaria TracyKatharina ReineckeEytan AdarMiriam Redi
|links| http://arxiv.org/abs/2403.07613v1 |
|updated| 2024-03-12 12:50:19 UTC |
|summary| Though images are ubiquitous across Wikipedia it is not obvious that theimage choices optimally support learning. When well selected images canenhance learning by dual coding complementing or supporting articles. Whenchosen poorly images can mislead distract and confuse. We developed a largedataset containing 470 questions  answers to 94 Wikipedia articles with imageson a wide range of topics. Through an online experiment n704 we determinedwhether the images displayed alongside the text of the article are effective inhelping readers understand and learn. For certain tasks such as learning toidentify targets visually e.g. which of these pictures is a gujiaarticle images significantly improve accuracy. Images did not significantlyimprove general knowledge questions e.g. where are gujia from. Mostinterestingly only some images helped with visual knowledge questions e.g.what shape is a gujia. Using our findings we reflect on the implicationsfor editors and tools to support image selection. |


| Item |Content|
| --- |---|
|idx| 2403.07314v1 |
|title| Customizable Avatars with Dynamic Facial Action Coded Expressions (CADyFACE) for Improved User Engagement |
|authors| Megan A. WitherowCrystal ButlerWinston J. ShieldsFurkan IlginNorou DiawaraJanice KeenerJohn W. HarringtonKhan M. Iftekharuddin
|links| http://arxiv.org/abs/2403.07314v1 |
|updated| 2024-03-12 05:00:38 UTC |
|summary| Customizable 3D avatar-based facial expression stimuli may improve userengagement in behavioral biomarker discovery and therapeutic intervention forautism Alzheimers disease facial palsy and more. However there is a lackof customizable avatar-based stimuli with Facial Action Coding System FACSaction unit AU labels. Therefore this study focuses on 1 FACS-labeledcustomizable avatar-based expression stimuli for maintaining subjectsengagement 2 learning-based measurements that quantify subjects facialresponses to such stimuli and 3 validation of constructs represented bystimulus-measurement pairs. We propose Customizable Avatars with Dynamic FacialAction Coded Expressions CADyFACE labeled with AUs by a certified FACSexpert. To measure subjects AUs in response to CADyFACE we propose a novelBeta-guided Correlation and Multi-task Expression learning neural networkBeCoME-Net for multi-label AU detection. The beta-guided correlation lossencourages feature correlation with AUs while discouraging correlation withsubject identities for improved generalization. We train BeCoME-Net forunilateral and bilateral AU detection and compare with state-of-the-artapproaches. To assess construct validity of CADyFACE and BeCoME-Net twentyhealthy adult volunteers complete expression recognition and mimicry tasks inan online feasibility study while webcam-based eye-tracking and video arecollected. We test validity of multiple constructs including face preferenceduring recognition and AUs during mimicry. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2403.07769v1 |
|title| Transforming Competition into Collaboration: The Revolutionary Role of Multi-Agent Systems and Language Models in Modern Organizations |
|authors| Carlos Jose Xavier Cruz
|links| http://arxiv.org/abs/2403.07769v1 |
|updated| 2024-03-12 15:56:10 UTC |
|summary| This article explores the dynamic influence of computational entities basedon multi-agent systems theory SMA combined with large language models LLMwhich are characterized by their ability to simulate complex humaninteractions as a possibility to revolutionize human user interaction from theuse of specialized artificial agents to support everything from operationalorganizational processes to strategic decision making based on appliedknowledge and human orchestration. Previous investigations reveal that thereare limitations particularly in the autonomous approach of artificial agentsespecially when dealing with new challenges and pragmatic tasks such asinducing logical reasoning and problem solving. It is also considered thattraditional techniques such as the stimulation of chains of thoughts requireexplicit human guidance. In our approach we employ agents developed from largelanguage models LLM each with distinct prototyping that considers behavioralelements driven by strategies that stimulate the generation of knowledge basedon the use case proposed in the scenario role-play business using adiscussion approach between agents guided conversation. We demonstrate thepotential of developing agents useful for organizational strategies based onmulti-agent system theories SMA and innovative uses based on large languagemodels LLM based offering a differentiated and adaptable experiment todifferent applications complexities domains and capabilities from LLM. |


| Item |Content|
| --- |---|
|idx| 2403.07748v1 |
|title| Ariadne and Theseus: Exploration and Rendezvous with Two Mobile Agents in an Unknown Graph |
|authors| Romain Cosson
|links| http://arxiv.org/abs/2403.07748v1 |
|updated| 2024-03-12 15:33:09 UTC |
|summary| We investigate two fundamental problems in mobile computing: exploration andrendezvous with two distinct mobile agents in an unknown graph. The agents canread and write information on whiteboards that are located at all nodes. Theyboth move along one adjacent edge at every time-step. In the explorationproblem both agents start from the same node of the graph and must traverseall of its edges. We show that a simple variant of depth-first search achievescollective exploration in m synchronous time-steps where m is the numberof edges of the graph. This improves the competitive ratio of collective graphexploration. In the rendezvous problem the agents start from different nodesof the graph and must meet as fast as possible. We introduce an algorithmguaranteeing rendezvous in at most frac32m time-steps. This improvesover the so-called wait for Mommy algorithm which requires 2m time-steps.All our guarantees are derived from a more general asynchronous setting inwhich the speeds of the agents are controlled by an adversary at all times. Ourguarantees also generalize to weighted graphs if the number of edges m isreplaced by the sum of all edge lengths. |


| Item |Content|
| --- |---|
|idx| 2403.07640v1 |
|title| Asynchronous Approximate Byzantine Consensus: A Multi-hop Relay Method and Tight Graph Conditions |
|authors| Liwei YuanHideaki Ishii
|links| http://arxiv.org/abs/2403.07640v1 |
|updated| 2024-03-12 13:24:19 UTC |
|summary| We study a multi-agent resilient consensus problem where some agents are ofthe Byzantine type and try to prevent the normal ones from reaching consensus.In our setting normal agents communicate with each other asynchronously overmulti-hop relay channels with delays. To solve this asynchronous Byzantineconsensus problem we develop the multi-hop weighted mean subsequence reducedMW-MSR algorithm. The main contribution is that we characterize a tight graphcondition for our algorithm to achieve Byzantine consensus which is expressedin the novel notion of strictly robust graphs. We show that the multi-hopcommunication is effective for enhancing the networks resilience againstByzantine agents. As a result we also obtain novel conditions for resilientconsensus under the malicious attack model which are tighter than those knownin the literature. Furthermore the proposed algorithm can be viewed as ageneralization of the conventional flooding-based algorithms with lesscomputational complexity. Lastly we provide numerical examples to show theeffectiveness of the proposed algorithm. |


| Item |Content|
| --- |---|
|idx| 2403.07559v1 |
|title| Ensembling Prioritized Hybrid Policies for Multi-agent Pathfinding |
|authors| Huijie TangFederico BertoJinkyoo Park
|links| http://arxiv.org/abs/2403.07559v1 |
|updated| 2024-03-12 11:47:12 UTC |
|summary| Multi-Agent Reinforcement Learning MARL based Multi-Agent Path FindingMAPF has recently gained attention due to its efficiency and scalability.Several MARL-MAPF methods choose to use communication to enrich the informationone agent can perceive. However existing works still struggle in structuredenvironments with high obstacle density and a high number of agents. To furtherimprove the performance of the communication-based MARL-MAPF solvers wepropose a new method Ensembling Prioritized Hybrid Policies EPH. We firstpropose a selective communication block to gather richer information for betteragent coordination within multi-agent environments and train the model with aQ-learning-based algorithm. We further introduce three advanced inferencestrategies aimed at bolstering performance during the execution phase. Firstwe hybridize the neural policy with single-agent expert guidance for navigatingconflict-free zones. Secondly we propose Q value-based methods for prioritizedresolution of conflicts as well as deadlock situations. Finally we introduce arobust ensemble method that can efficiently collect the best out of multiplepossible solutions. We empirically evaluate EPH in complex multi-agentenvironments and demonstrate competitive performance against state-of-the-artneural methods for MAPF. |


| Item |Content|
| --- |---|
|idx| 2403.07131v1 |
|title| Bigraph Matching Weighted with Learnt Incentive Function for Multi-Robot Task Allocation |
|authors| Steve PaulNathan MaurerSouma Chowdhury
|links| http://arxiv.org/abs/2403.07131v1 |
|updated| 2024-03-11 19:55:08 UTC |
|summary| Most real-world Multi-Robot Task Allocation MRTA problems require fast andefficient decision-making which is often achieved using heuristics-aidedmethods such as genetic algorithms auction-based methods and bipartite graphmatching methods. These methods often assume a form that lends betterexplainability compared to an end-to-end learnt neural network based policyfor MRTA. However deriving suitable heuristics can be tedious risky and insome cases impractical if problems are too complex. This raises the question:can these heuristics be learned To this end this paper particularly developsa Graph Reinforcement Learning GRL framework to learn the heuristics orincentives for a bipartite graph matching approach to MRTA. Specifically aCapsule Attention policy model is used to learn how to weight task/robotpairings edges in the bipartite graph that connects the set of tasks to theset of robots. The original capsule attention network architecture isfundamentally modified by adding encoding of robots state graph and twoMultihead Attention based decoders whose output are used to construct aLogNormal distribution matrix from which positive bigraph weights can be drawn.The performance of this new bigraph matching approach augmented with aGRL-derived incentive is found to be at par with the original bigraph matchingapproach that used expert-specified heuristics with the former offeringnotable robustness benefits. During training the learned incentive policy isfound to get initially closer to the expert-specified incentive and thenslightly deviate from its trend. |


