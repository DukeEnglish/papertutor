# cs.CL 

| Item |Content|
| --- |---|
|idx| 2408.11051v1 |
|title| FLAME: Learning to Navigate with Multimodal LLM in Urban Environments |
|authors| Yunzhe XuYiyuan PanZhe LiuHesheng Wang
|links| http://arxiv.org/abs/2408.11051v1 |
|updated| 2024-08-20 17:57:46 UTC |
|summary| Large Language Models LLMs have demonstrated potential inVision-and-Language Navigation VLN tasks yet current applications facechallenges. While LLMs excel in general conversation scenarios they strugglewith specialized navigation tasks yielding suboptimal performance compared tospecialized VLN models. We introduce FLAME FLAMingo-Architected EmbodiedAgent a novel Multimodal LLM-based agent and architecture designed for urbanVLN tasks that efficiently handles multiple observations. Our approachimplements a three-phase tuning technique for effective adaptation tonavigation tasks including single perception tuning for street viewdescription multiple perception tuning for trajectory summarization andend-to-end training on VLN datasets. The augmented datasets are synthesizedautomatically. Experimental results demonstrate FLAMEs superiority overexisting methods surpassing state-of-the-art methods by a 7.3 increase intask completion rate on Touchdown dataset. This work showcases the potential ofMultimodal LLMs MLLMs in complex navigation tasks representing anadvancement towards practical applications of MLLMs in embodied AI. Projectpage: https://flame-sjtu.github.io |


| Item |Content|
| --- |---|
|idx| 2408.11049v1 |
|title| MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding |
|authors| Jian ChenVashisth TiwariRanajoy SadhukhanZhuoming ChenJinyuan ShiIan En-Hsu YenBeidi Chen
|links| http://arxiv.org/abs/2408.11049v1 |
|updated| 2024-08-20 17:57:31 UTC |
|summary| Large Language Models LLMs have become more prevalent in long-contextapplications such as interactive chatbots document analysis and agentworkflows but it is challenging to serve long-context requests with lowlatency and high throughput. Speculative decoding SD is a widely usedtechnique to reduce latency without sacrificing performance but theconventional wisdom suggests that its efficacy is limited to small batch sizes.In MagicDec we show that surprisingly SD can achieve speedup even for a highthroughput inference regime for moderate to long sequences. More interestinglyan intelligent drafting strategy can achieve better speedup with increasingbatch size based on our rigorous analysis. MagicDec first identifies thebottleneck shifts with increasing batch size and sequence length and usesthese insights to deploy speculative decoding more effectively for highthroughput inference. Then it leverages draft models with sparse KV cache toaddress the KV bottleneck that scales with both sequence length and batch size. |


| Item |Content|
| --- |---|
|idx| 2408.11046v1 |
|title| Inside the Black Box: Detecting Data Leakage in Pre-trained Language Encoders |
|authors| Yuan XinZheng LiNing YuDingfan ChenMario FritzMichael BackesYang Zhang
|links| http://arxiv.org/abs/2408.11046v1 |
|updated| 2024-08-20 17:55:15 UTC |
|summary| Despite being prevalent in the general field of Natural Language ProcessingNLP pre-trained language models inherently carry privacy and copyrightconcerns due to their nature of training on large-scale web-scraped data. Inthis paper we pioneer a systematic exploration of such risks associated withpre-trained language encoders specifically focusing on the membership leakageof pre-training data exposed through downstream models adapted from pre-trainedlanguage encoders-an aspect largely overlooked in existing literature. Ourstudy encompasses comprehensive experiments across four types of pre-trainedencoder architectures three representative downstream tasks and fivebenchmark datasets. Intriguingly our evaluations reveal for the first timethe existence of membership leakage even when only the black-box output of thedownstream model is exposed highlighting a privacy risk far greater thanpreviously assumed. Alongside we present in-depth analysis and insights towardguiding future researchers and practitioners in addressing the privacyconsiderations in developing pre-trained language models. |


| Item |Content|
| --- |---|
|idx| 2408.11029v1 |
|title| Scaling Law with Learning Rate Annealing |
|authors| Howe TissueVenus WangLu Wang
|links| http://arxiv.org/abs/2408.11029v1 |
|updated| 2024-08-20 17:30:48 UTC |
|summary| We find that the cross-entropy loss curves of neural language modelsempirically adhere to a scaling law with learning rate LR annealing overtraining steps s: Ls  L_0  Acdot S_1-alpha - Ccdot S_2 WhereS_1 is forward area and S_2 is learning rate annealing area. Thisformulation takes into account two factors: 1 The forward scaling defined astypical scaling law and 2 the additional loss drop brought by LR annealing.Therefore this formulation can describe the full loss curve at each steprather than the single loss point at the end of training. Applying the scalinglaw with LR annealing and fitting only one or two training curves we canaccurately predict the loss of language model training at any given step andacross any learning rate scheduler LRS. Furthermore this equation accuratelydescribes the dynamics during training process and provides a theoreticalverification and explanation for numerous experimental findings of previousstudies particularly those focusing on LR schedule and LR annealing. Theresulting insights also serve as a guide for researchers to select criticalLRS in advance by prediction using our equation. Most significantly since allthe points in a full training curve follow the equation we can achieveaccurate loss prediction at any given step across any learning rate schedulerwhile expending less than 1 of the computational cost required by thechinchilla scaling law to fit language modeling loss. This approach extremelydemocratizes scaling law fitting and predicting in developing large languagemodels. |


| Item |Content|
| --- |---|
|idx| 2408.11021v1 |
|title| Athena: Safe Autonomous Agents with Verbal Contrastive Learning |
|authors| Tanmana SadhuAli PesaranghaderYanan ChenDong Hoon Yi
|links| http://arxiv.org/abs/2408.11021v1 |
|updated| 2024-08-20 17:21:10 UTC |
|summary| Due to emergent capabilities large language models LLMs have been utilizedas language-based agents to perform a variety of tasks and make decisions withan increasing degree of autonomy. These autonomous agents can understandhigh-level instructions interact with their environments and execute complextasks using a selection of tools available to them. As the capabilities of theagents expand ensuring their safety and trustworthiness becomes moreimperative. In this study we introduce the Athena framework which leveragesthe concept of verbal contrastive learning where past safe and unsafetrajectories are used as in-context contrastive examples to guide the agenttowards safety while fulfilling a given task. The framework also incorporates acritiquing mechanism to guide the agent to prevent risky actions at every step.Furthermore due to the lack of existing benchmarks on the safety reasoningability of LLM-based agents we curate a set of 80 toolkits across 8 categorieswith 180 scenarios to provide a safety evaluation benchmark. Our experimentalevaluation with both closed- and open-source LLMs indicates verbalcontrastive learning and interaction-level critiquing improve the safety ratesignificantly. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2408.11054v1 |
|title| NeCo: Improving DINOv2's spatial representations in 19 GPU hours with Patch Neighbor Consistency |
|authors| Valentinos ParizaMohammadreza SalehiGertjan BurghoutsFrancesco LocatelloYuki M. Asano
|links| http://arxiv.org/abs/2408.11054v1 |
|updated| 2024-08-20 17:58:59 UTC |
|summary| We propose sorting patch representations across views as a novelself-supervised learning signal to improve pretrained representations. To thisend we introduce NeCo: Patch Neighbor Consistency a novel training loss thatenforces patch-level nearest neighbor consistency across a student and teachermodel relative to reference batches. Our method leverages a differentiablesorting method applied on top of pretrained representations such asDINOv2-registers to bootstrap the learning signal and further improve uponthem. This dense post-pretraining leads to superior performance across variousmodels and datasets despite requiring only 19 hours on a single GPU. Wedemonstrate that this method generates high-quality dense feature encoders andestablish several new state-of-the-art results: 5.5 and  6 fornon-parametric in-context semantic segmentation on ADE20k and Pascal VOC and7.2 and 5.7 for linear segmentation evaluations on COCO-Things and -Stuff. |


| Item |Content|
| --- |---|
|idx| 2408.11053v1 |
|title| Revisiting VerilogEval: Newer LLMs, In-Context Learning, and Specification-to-RTL Tasks |
|authors| Nathaniel PinckneyChristopher BattenMingjie LiuHaoxing RenBrucek Khailany
|links| http://arxiv.org/abs/2408.11053v1 |
|updated| 2024-08-20 17:58:56 UTC |
|summary| The application of large-language models LLMs to digital hardware codegeneration is an emerging field. Most LLMs are primarily trained on naturallanguage and software code. Hardware code such as Verilog represents only asmall portion of the training data and few hardware benchmarks exist. Toaddress this gap the open-source VerilogEval benchmark was released in 2023providing a consistent evaluation framework for LLMs on code completion tasks.It was tested on state-of-the-art models at the time including GPT-4. HoweverVerilogEval and other Verilog generation benchmarks lack failure analysis andin present form are not conducive to exploring prompting techniques. Alsosince VerilogEvals release both commercial and open-source models have seencontinued development.  In this work we evaluate new commercial and open-source models of varyingsizes against an improved VerilogEval benchmark suite. We enhance VerilogEvalsinfrastructure and dataset by automatically classifying failures introduce newprompts for supporting in-context learning ICL examples and extend thesupported tasks to specification-to-RTL translation. We find a measurableimprovement in commercial state-of-the-art models with GPT-4 Turbo achieving a59 pass rate on spec-to-RTL tasks. We also study the performance ofopen-source and domain-specific models that have emerged and demonstrate thatmodels can benefit substantially from ICL. We find that recently-released Llama3.1 405B achieves a pass rate of 58 effectively matching that of GPT-4 Turboand that the much smaller domain-specific RTL-Coder 6.7B models achieve animpressive 37 pass rate. However prompt engineering is key to achieving goodpass rates and varies widely with model and task. A benchmark infrastructurethat allows for prompt engineering and failure analysis is key to continuedmodel development and deployment. |


| Item |Content|
| --- |---|
|idx| 2408.11052v1 |
|title| Accelerating Goal-Conditioned RL Algorithms and Research |
|authors| Michał BortkiewiczWładek PałuckiVivek MyersTadeusz DziarmagaTomasz ArczewskiŁukasz KucińskiBenjamin Eysenbach
|links| http://arxiv.org/abs/2408.11052v1 |
|updated| 2024-08-20 17:58:40 UTC |
|summary| Self-supervision has the potential to transform reinforcement learning RLparalleling the breakthroughs it has enabled in other areas of machinelearning. While self-supervised learning in other domains aims to find patternsin a fixed dataset self-supervised goal-conditioned reinforcement learningGCRL agents discover new behaviors by learning from the goals achieved duringunstructured interaction with the environment. However these methods havefailed to see similar success both due to a lack of data from slowenvironments as well as a lack of stable algorithms. We take a step towardaddressing both of these issues by releasing a high-performance codebase andbenchmark JaxGCRL for self-supervised GCRL enabling researchers to trainagents for millions of environment steps in minutes on a single GPU. The key tothis performance is a combination of GPU-accelerated environments and a stablebatched version of the contrastive reinforcement learning algorithm based onan infoNCE objective that effectively makes use of this increased datathroughput. With this approach we provide a foundation for future research inself-supervised GCRL enabling researchers to quickly iterate on new ideas andevaluate them in a diverse set of challenging environments. Website  Code:https://github.com/MichalBortkiewicz/JaxGCRL |


| Item |Content|
| --- |---|
|idx| 2408.11051v1 |
|title| FLAME: Learning to Navigate with Multimodal LLM in Urban Environments |
|authors| Yunzhe XuYiyuan PanZhe LiuHesheng Wang
|links| http://arxiv.org/abs/2408.11051v1 |
|updated| 2024-08-20 17:57:46 UTC |
|summary| Large Language Models LLMs have demonstrated potential inVision-and-Language Navigation VLN tasks yet current applications facechallenges. While LLMs excel in general conversation scenarios they strugglewith specialized navigation tasks yielding suboptimal performance compared tospecialized VLN models. We introduce FLAME FLAMingo-Architected EmbodiedAgent a novel Multimodal LLM-based agent and architecture designed for urbanVLN tasks that efficiently handles multiple observations. Our approachimplements a three-phase tuning technique for effective adaptation tonavigation tasks including single perception tuning for street viewdescription multiple perception tuning for trajectory summarization andend-to-end training on VLN datasets. The augmented datasets are synthesizedautomatically. Experimental results demonstrate FLAMEs superiority overexisting methods surpassing state-of-the-art methods by a 7.3 increase intask completion rate on Touchdown dataset. This work showcases the potential ofMultimodal LLMs MLLMs in complex navigation tasks representing anadvancement towards practical applications of MLLMs in embodied AI. Projectpage: https://flame-sjtu.github.io |


| Item |Content|
| --- |---|
|idx| 2408.11048v1 |
|title| RP1M: A Large-Scale Motion Dataset for Piano Playing with Bi-Manual Dexterous Robot Hands |
|authors| Yi ZhaoLe ChenJan SchneiderQuankai GaoJuho KannalaBernhard SchölkopfJoni PajarinenDieter Büchler
|links| http://arxiv.org/abs/2408.11048v1 |
|updated| 2024-08-20 17:56:52 UTC |
|summary| It has been a long-standing research goal to endow robot hands withhuman-level dexterity. Bi-manual robot piano playing constitutes a task thatcombines challenges from dynamic tasks such as generating fast while precisemotions with slower but contact-rich manipulation problems. Althoughreinforcement learning based approaches have shown promising results insingle-task performance these methods struggle in a multi-song setting. Ourwork aims to close this gap and thereby enable imitation learning approachesfor robot piano playing at scale. To this end we introduce the Robot Piano 1Million RP1M dataset containing bi-manual robot piano playing motion data ofmore than one million trajectories. We formulate finger placements as anoptimal transport problem thus enabling automatic annotation of vast amountsof unlabeled songs. Benchmarking existing imitation learning approaches showsthat such approaches reach state-of-the-art robot piano playing performance byleveraging RP1M. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2408.11052v1 |
|title| Accelerating Goal-Conditioned RL Algorithms and Research |
|authors| Michał BortkiewiczWładek PałuckiVivek MyersTadeusz DziarmagaTomasz ArczewskiŁukasz KucińskiBenjamin Eysenbach
|links| http://arxiv.org/abs/2408.11052v1 |
|updated| 2024-08-20 17:58:40 UTC |
|summary| Self-supervision has the potential to transform reinforcement learning RLparalleling the breakthroughs it has enabled in other areas of machinelearning. While self-supervised learning in other domains aims to find patternsin a fixed dataset self-supervised goal-conditioned reinforcement learningGCRL agents discover new behaviors by learning from the goals achieved duringunstructured interaction with the environment. However these methods havefailed to see similar success both due to a lack of data from slowenvironments as well as a lack of stable algorithms. We take a step towardaddressing both of these issues by releasing a high-performance codebase andbenchmark JaxGCRL for self-supervised GCRL enabling researchers to trainagents for millions of environment steps in minutes on a single GPU. The key tothis performance is a combination of GPU-accelerated environments and a stablebatched version of the contrastive reinforcement learning algorithm based onan infoNCE objective that effectively makes use of this increased datathroughput. With this approach we provide a foundation for future research inself-supervised GCRL enabling researchers to quickly iterate on new ideas andevaluate them in a diverse set of challenging environments. Website  Code:https://github.com/MichalBortkiewicz/JaxGCRL |


| Item |Content|
| --- |---|
|idx| 2408.11048v1 |
|title| RP1M: A Large-Scale Motion Dataset for Piano Playing with Bi-Manual Dexterous Robot Hands |
|authors| Yi ZhaoLe ChenJan SchneiderQuankai GaoJuho KannalaBernhard SchölkopfJoni PajarinenDieter Büchler
|links| http://arxiv.org/abs/2408.11048v1 |
|updated| 2024-08-20 17:56:52 UTC |
|summary| It has been a long-standing research goal to endow robot hands withhuman-level dexterity. Bi-manual robot piano playing constitutes a task thatcombines challenges from dynamic tasks such as generating fast while precisemotions with slower but contact-rich manipulation problems. Althoughreinforcement learning based approaches have shown promising results insingle-task performance these methods struggle in a multi-song setting. Ourwork aims to close this gap and thereby enable imitation learning approachesfor robot piano playing at scale. To this end we introduce the Robot Piano 1Million RP1M dataset containing bi-manual robot piano playing motion data ofmore than one million trajectories. We formulate finger placements as anoptimal transport problem thus enabling automatic annotation of vast amountsof unlabeled songs. Benchmarking existing imitation learning approaches showsthat such approaches reach state-of-the-art robot piano playing performance byleveraging RP1M. |


| Item |Content|
| --- |---|
|idx| 2408.11032v1 |
|title| Atmospheric Transport Modeling of CO$_2$ with Neural Networks |
|authors| Vitus BensonAna BastosChristian ReimersAlexander J. WinklerFanny YangMarkus Reichstein
|links| http://arxiv.org/abs/2408.11032v1 |
|updated| 2024-08-20 17:33:20 UTC |
|summary| Accurately describing the distribution of CO_2 in the atmosphere withatmospheric tracer transport models is essential for greenhouse gas monitoringand verification support systems to aid implementation of international climateagreements. Large deep neural networks are poised to revolutionize weatherprediction which requires 3D modeling of the atmosphere. While similar in thisregard atmospheric transport modeling is subject to new challenges. Bothstable predictions for longer time horizons and mass conservation throughoutneed to be achieved while IO plays a larger role compared to computationalcosts. In this study we explore four different deep neural networks UNetGraphCast Spherical Fourier Neural Operator and SwinTransformer which haveproven as state-of-the-art in weather prediction to assess their usefulness foratmospheric tracer transport modeling. For this we assemble the CarbonBenchdataset a systematic benchmark tailored for machine learning emulators ofEulerian atmospheric transport. Through architectural adjustments we decouplethe performance of our emulators from the distribution shift caused by a steadyrise in atmospheric CO_2. More specifically we center CO_2 input fields tozero mean and then use an explicit flux scheme and a mass fixer to assure massbalance. This design enables stable and mass conserving transport for over 6months with all four neural network architectures. In our study theSwinTransformer displays particularly strong emulation skill 90-day R2 0.99 with physically plausible emulation even for forward runs of multipleyears. This work paves the way forward towards high resolution forward andinverse modeling of inert trace gases with neural networks. |


| Item |Content|
| --- |---|
|idx| 2408.11019v1 |
|title| An Overlooked Role of Context-Sensitive Dendrites |
|authors| Mohsin RazaAhsan Adeel
|links| http://arxiv.org/abs/2408.11019v1 |
|updated| 2024-08-20 17:18:54 UTC |
|summary| To date most dendritic studies have predominantly focused on the apical zoneof pyramidal two-point neurons TPNs receiving only feedback FB connectionsfrom higher perceptual layers and using them for learning. Recent cellularneurophysiology and computational neuroscience studies suggests that the apicalinput context coming from feedback and lateral connections is multifacetedand far more diverse with greater implications for ongoing learning andprocessing in the brain than previously realized. In addition to the FB theapical tuft receives signals from neighboring cells of the same network asproximal P context other parts of the brain as distal D context andoverall coherent information across the network as universal U context. Theintegrated context C amplifies and suppresses the transmission of coherentand conflicting feedforward FF signals respectively. Specifically we showthat complex context-sensitive CS-TPNs flexibly integrate C moment-by-momentwith the FF somatic current at the soma such that the somatic current isamplified when both feedforward FF and C are coherent otherwise it isattenuated. This generates the event only when the FF and C currents arecoherent which is then translated into a singlet or a burst based on the FBinformation. Spiking simulation results show that this flexible integration ofsomatic and contextual currents enables the propagation of more coherentsignals bursts making learning faster with fewer neurons. Similar behavioris observed when this functioning is used in conventional artificial networkswhere orders of magnitude fewer neurons are required to process vast amounts ofheterogeneous real-world audio-visual AV data trained using backpropagationBP. The computational findings presented here demonstrate the universality ofCS-TPNs suggesting a dendritic narrative that was previously overlooked. |


| Item |Content|
| --- |---|
|idx| 2408.10998v1 |
|title| Audio Match Cutting: Finding and Creating Matching Audio Transitions in Movies and Videos |
|authors| Dennis FedorishinLie LuSrirangaraj SetlurVenu Govindaraju
|links| http://dx.doi.org/10.1109/ICASSP48485.2024.10447306 |
|updated| 2024-08-20 16:46:54 UTC |
|summary| A match cut is a common video editing technique where a pair of shots thathave a similar composition transition fluidly from one to another. Althoughmatch cuts are often visual certain match cuts involve the fluid transition ofaudio where sounds from different sources merge into one indistinguishabletransition between two shots. In this paper we explore the ability toautomatically find and create audio match cuts within videos and movies. Wecreate a self-supervised audio representation for audio match cutting anddevelop a coarse-to-fine audio match pipeline that recommends matching shotsand creates the blended audio. We further annotate a dataset for the proposedaudio match cut task and compare the ability of multiple audio representationsto find audio match cut candidates. Finally we evaluate multiple methods toblend two matching audio candidates with the goal of creating a smoothtransition. Project page and examples are available at:https://denfed.github.io/audiomatchcut/ |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2408.11055v1 |
|title| Prompt-Guided Image-Adaptive Neural Implicit Lookup Tables for Interpretable Image Enhancement |
|authors| Satoshi Kosugi
|links| http://dx.doi.org/10.1145/3664647.3680743 |
|updated| 2024-08-20 17:59:01 UTC |
|summary| In this paper we delve into the concept of interpretable image enhancementa technique that enhances image quality by adjusting filter parameters witheasily understandable names such as Exposure and Contrast. Unlike usingpredefined image editing filters our framework utilizes learnable filters thatacquire interpretable names through training. Our contribution is two-fold.Firstly we introduce a novel filter architecture called an image-adaptiveneural implicit lookup table which uses a multilayer perceptron to implicitlydefine the transformation from input feature space to output color space. Byincorporating image-adaptive parameters directly into the input features weachieve highly expressive filters. Secondly we introduce a prompt guidanceloss to assign interpretable names to each filter. We evaluate visualimpressions of enhancement results such as exposure and contrast using avision and language model along with guiding prompts. We define a constraint toensure that each filter affects only the targeted visual impression withoutinfluencing other attributes which allows us to obtain the desired filtereffects. Experimental results show that our method outperforms existingpredefined filter-based methods thanks to the filters optimized to predicttarget results. Our source code is available athttps://github.com/satoshi-kosugi/PG-IA-NILUT. |


| Item |Content|
| --- |---|
|idx| 2408.11054v1 |
|title| NeCo: Improving DINOv2's spatial representations in 19 GPU hours with Patch Neighbor Consistency |
|authors| Valentinos ParizaMohammadreza SalehiGertjan BurghoutsFrancesco LocatelloYuki M. Asano
|links| http://arxiv.org/abs/2408.11054v1 |
|updated| 2024-08-20 17:58:59 UTC |
|summary| We propose sorting patch representations across views as a novelself-supervised learning signal to improve pretrained representations. To thisend we introduce NeCo: Patch Neighbor Consistency a novel training loss thatenforces patch-level nearest neighbor consistency across a student and teachermodel relative to reference batches. Our method leverages a differentiablesorting method applied on top of pretrained representations such asDINOv2-registers to bootstrap the learning signal and further improve uponthem. This dense post-pretraining leads to superior performance across variousmodels and datasets despite requiring only 19 hours on a single GPU. Wedemonstrate that this method generates high-quality dense feature encoders andestablish several new state-of-the-art results: 5.5 and  6 fornon-parametric in-context semantic segmentation on ADE20k and Pascal VOC and7.2 and 5.7 for linear segmentation evaluations on COCO-Things and -Stuff. |


| Item |Content|
| --- |---|
|idx| 2408.11051v1 |
|title| FLAME: Learning to Navigate with Multimodal LLM in Urban Environments |
|authors| Yunzhe XuYiyuan PanZhe LiuHesheng Wang
|links| http://arxiv.org/abs/2408.11051v1 |
|updated| 2024-08-20 17:57:46 UTC |
|summary| Large Language Models LLMs have demonstrated potential inVision-and-Language Navigation VLN tasks yet current applications facechallenges. While LLMs excel in general conversation scenarios they strugglewith specialized navigation tasks yielding suboptimal performance compared tospecialized VLN models. We introduce FLAME FLAMingo-Architected EmbodiedAgent a novel Multimodal LLM-based agent and architecture designed for urbanVLN tasks that efficiently handles multiple observations. Our approachimplements a three-phase tuning technique for effective adaptation tonavigation tasks including single perception tuning for street viewdescription multiple perception tuning for trajectory summarization andend-to-end training on VLN datasets. The augmented datasets are synthesizedautomatically. Experimental results demonstrate FLAMEs superiority overexisting methods surpassing state-of-the-art methods by a 7.3 increase intask completion rate on Touchdown dataset. This work showcases the potential ofMultimodal LLMs MLLMs in complex navigation tasks representing anadvancement towards practical applications of MLLMs in embodied AI. Projectpage: https://flame-sjtu.github.io |


| Item |Content|
| --- |---|
|idx| 2408.11039v1 |
|title| Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model |
|authors| Chunting ZhouLili YuArun BabuKushal TirumalaMichihiro YasunagaLeonid ShamisJacob KahnXuezhe MaLuke ZettlemoyerOmer Levy
|links| http://arxiv.org/abs/2408.11039v1 |
|updated| 2024-08-20 17:48:20 UTC |
|summary| We introduce Transfusion a recipe for training a multi-modal model overdiscrete and continuous data. Transfusion combines the language modeling lossfunction next token prediction with diffusion to train a single transformerover mixed-modality sequences. We pretrain multiple Transfusion models up to 7Bparameters from scratch on a mixture of text and image data establishingscaling laws with respect to a variety of uni- and cross-modal benchmarks. Ourexperiments show that Transfusion scales significantly better than quantizingimages and training a language model over discrete image tokens. By introducingmodality-specific encoding and decoding layers we can further improve theperformance of Transfusion models and even compress each image to just 16patches. We further demonstrate that scaling our Transfusion recipe to 7Bparameters and 2T multi-modal tokens produces a model that can generate imagesand text on a par with similar scale diffusion models and language modelsreaping the benefits of both worlds. |


| Item |Content|
| --- |---|
|idx| 2408.11032v1 |
|title| Atmospheric Transport Modeling of CO$_2$ with Neural Networks |
|authors| Vitus BensonAna BastosChristian ReimersAlexander J. WinklerFanny YangMarkus Reichstein
|links| http://arxiv.org/abs/2408.11032v1 |
|updated| 2024-08-20 17:33:20 UTC |
|summary| Accurately describing the distribution of CO_2 in the atmosphere withatmospheric tracer transport models is essential for greenhouse gas monitoringand verification support systems to aid implementation of international climateagreements. Large deep neural networks are poised to revolutionize weatherprediction which requires 3D modeling of the atmosphere. While similar in thisregard atmospheric transport modeling is subject to new challenges. Bothstable predictions for longer time horizons and mass conservation throughoutneed to be achieved while IO plays a larger role compared to computationalcosts. In this study we explore four different deep neural networks UNetGraphCast Spherical Fourier Neural Operator and SwinTransformer which haveproven as state-of-the-art in weather prediction to assess their usefulness foratmospheric tracer transport modeling. For this we assemble the CarbonBenchdataset a systematic benchmark tailored for machine learning emulators ofEulerian atmospheric transport. Through architectural adjustments we decouplethe performance of our emulators from the distribution shift caused by a steadyrise in atmospheric CO_2. More specifically we center CO_2 input fields tozero mean and then use an explicit flux scheme and a mass fixer to assure massbalance. This design enables stable and mass conserving transport for over 6months with all four neural network architectures. In our study theSwinTransformer displays particularly strong emulation skill 90-day R2 0.99 with physically plausible emulation even for forward runs of multipleyears. This work paves the way forward towards high resolution forward andinverse modeling of inert trace gases with neural networks. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2408.10996v1 |
|title| Approximation Rates for Shallow ReLU$^k$ Neural Networks on Sobolev Spaces via the Radon Transform |
|authors| Tong MaoJonathan W. SiegelJinchao Xu
|links| http://arxiv.org/abs/2408.10996v1 |
|updated| 2024-08-20 16:43:45 UTC |
|summary| Let Omegasubset mathbbRd be a bounded domain. We consider the problemof how efficiently shallow neural networks with the ReLUk activationfunction can approximate functions from Sobolev spaces WsL_pOmega witherror measured in the L_qOmega-norm. Utilizing the Radon transform andrecent results from discrepancy theory we provide a simple proof of nearlyoptimal approximation rates in a variety of cases including when qleq ppgeq 2 and s leq k  d1/2. The rates we derive are optimal up tologarithmic factors and significantly generalize existing results. Aninteresting consequence is that the adaptivity of shallow ReLUk neuralnetworks enables them to obtain optimal approximation rates for smoothness upto order s  k  d1/2 even though they represent piecewise polynomials offixed degree k. |


| Item |Content|
| --- |---|
|idx| 2408.10976v1 |
|title| Kernel-Based Differentiable Learning of Non-Parametric Directed Acyclic Graphical Models |
|authors| Yurou LiangOleksandr ZadorozhnyiMathias Drton
|links| http://arxiv.org/abs/2408.10976v1 |
|updated| 2024-08-20 16:09:40 UTC |
|summary| Causal discovery amounts to learning a directed acyclic graph DAG thatencodes a causal model. This model selection problem can be challenging due toits large combinatorial search space particularly when dealing withnon-parametric causal models. Recent research has sought to bypass thecombinatorial search by reformulating causal discovery as a continuousoptimization problem employing constraints that ensure the acyclicity of thegraph. In non-parametric settings existing approaches typically rely onfinite-dimensional approximations of the relationships between nodes resultingin a score-based continuous optimization problem with a smooth acyclicityconstraint. In this work we develop an alternative approximation method byutilizing reproducing kernel Hilbert spaces RKHS and applying generalsparsity-inducing regularization terms based on partial derivatives. Withinthis framework we introduce an extended RKHS representer theorem. To enforceacyclicity we advocate the log-determinant formulation of the acyclicityconstraint and show its stability. Finally we assess the performance of ourproposed RKHS-DAGMA procedure through simulations and illustrative dataanalyses. |


| Item |Content|
| --- |---|
|idx| 2408.10939v1 |
|title| Conformalized Interval Arithmetic with Symmetric Calibration |
|authors| Rui LuoZhixin Zhou
|links| http://arxiv.org/abs/2408.10939v1 |
|updated| 2024-08-20 15:27:18 UTC |
|summary| Uncertainty quantification is essential in decision-making especially whenjoint distributions of random variables are involved. While conformalprediction provides distribution-free prediction sets with valid coverageguarantees it traditionally focuses on single predictions. This paperintroduces novel conformal prediction methods for estimating the sum or averageof unknown labels over specific index sets. We develop conformal predictionintervals for single target to the prediction interval for sum of multipletargets. Under permutation invariant assumptions we prove the validity of ourproposed method. We also apply our algorithms on class average estimation andpath cost prediction tasks and we show that our method outperforms existingconformalized approaches as well as non-conformal approaches. |


| Item |Content|
| --- |---|
|idx| 2408.10862v1 |
|title| Feature Selection from Differentially Private Correlations |
|authors| Ryan SwopeAmol KhannaPhilip DoldoSaptarshi RoyEdward Raff
|links| http://arxiv.org/abs/2408.10862v1 |
|updated| 2024-08-20 13:54:07 UTC |
|summary| Data scientists often seek to identify the most important features inhigh-dimensional datasets. This can be done through L_1-regularizedregression but this can become inefficient for very high-dimensional datasets.Additionally high-dimensional regression can leak information about individualdatapoints in a dataset. In this paper we empirically evaluate the establishedbaseline method for feature selection with differential privacy the two-stageselection technique and show that it is not stable under sparsity. This makesit perform poorly on real-world datasets so we consider a different approachto private feature selection. We employ a correlations-based order statistic tochoose important features from a dataset and privatize them to ensure that theresults do not leak information about individual datapoints. We find that ourmethod significantly outperforms the established baseline for private featureselection on many datasets. |


| Item |Content|
| --- |---|
|idx| 2408.10762v1 |
|title| Sparse Regression for Discovery of Constitutive Models from Oscillatory Shear Measurements |
|authors| Sachin ShanbhagGordon Erlebacher
|links| http://arxiv.org/abs/2408.10762v1 |
|updated| 2024-08-20 11:52:21 UTC |
|summary| We propose sparse regression as an alternative to neural networks for thediscovery of parsimonious constitutive models CMs from oscillatory shearexperiments. Symmetry and frame-invariance are strictly imposed by using tensorbasis functions to isolate and describe unknown nonlinear terms in the CMs. Wegenerate synthetic experimental data using the Giesekus and Phan-Thien TannerCMs and consider two different scenarios. In the complete informationscenario we assume that the shear stress along with the first and secondnormal stress differences is measured. This leads to a sparse linearregression problem that can be solved efficiently using l_1 regularization.In the partial information scenario we assume that only shear stress data isavailable. This leads to a more challenging sparse nonlinear regressionproblem for which we propose a greedy two-stage algorithm. In both scenariosthe proposed methods fit and interpolate the training data remarkably well.Predictions of the inferred CMs extrapolate satisfactorily beyond the range oftraining data for oscillatory shear. They also extrapolate reasonably well toflow conditions like startup of steady and uniaxial extension that are not usedin the identification of CMs. We discuss ramifications for experimental designpotential algorithmic improvements and implications of the non-uniqueness ofCMs inferred from partial information. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2408.10937v1 |
|title| Proxona: Leveraging LLM-Driven Personas to Enhance Creators' Understanding of Their Audience |
|authors| Yoonseo ChoiEun Jeong KangSeulgi ChoiMin Kyung LeeJuho Kim
|links| http://arxiv.org/abs/2408.10937v1 |
|updated| 2024-08-20 15:20:30 UTC |
|summary| Creators are nothing without their audience and thereby understanding theiraudience is the cornerstone of their professional achievement. Yet manycreators feel lost while comprehending audiences with existing tools whichoffer insufficient insights for tailoring content to audience needs. To addressthe challenges creators face in understanding their audience we presentProxona a system for defining and extracting representative audience personasfrom the comments. Creators converse with personas to gain insights into theirpreferences and engagement solicit feedback and implement evidence-basedimprovements to their content. Powered by large language models Proxonaanalyzes audience comments distilling the latent characteristics of audiencesinto tangible dimensions classification categories and values categoryattributes. Proxona then clusters these into synthetic personas. Our technicalevaluations demonstrated that our pipelines effectively generated relevant anddistinct dimensions and values enabling the deduction of audience-reflectingpersonas while minimizing the likelihood of hallucinations in personaresponses. Our user evaluation with 11 creators showed that Proxona supportedcreators to gain new insights about their audience make informed decisionsand successfully complete content creation with high confidence. Proxonasdata-driven audience personas empower creators to seamlessly integrate audienceperspectives into their creative processes fostering a collaborative approachto content creation. |


| Item |Content|
| --- |---|
|idx| 2408.10933v1 |
|title| Evaluating Assistive Technologies on a Trade Fair: Methodological Overview and Lessons Learned |
|authors| Annalies BaumeisterFelix GoldauMax PascherJens GerkenUdo FresePatrizia Tolle
|links| http://arxiv.org/abs/2408.10933v1 |
|updated| 2024-08-20 15:17:07 UTC |
|summary| User-centered evaluations are a core requirement in the development of newuser related technologies. However it is often difficult to recruit sufficientparticipants especially if the target population is small particularly busyor in some way restricted in their mobility. We bypassed these problems byconducting studies on trade fairs that were specifically designed for ourtarget population potentially care-receiving individuals in wheelchairs andtherefore provided our users with external incentive to attend our study. Thispaper presents our gathered experiences including methodologicalspecifications and lessons learned and is aimed to guide other researcherswith conducting similar studies. In addition we also discuss chances generatedby this unconventional study environment as well as its limitations. |


| Item |Content|
| --- |---|
|idx| 2408.10908v1 |
|title| Enhancing End-to-End Autonomous Driving Systems Through Synchronized Human Behavior Data |
|authors| Yiqun DuanZhuoli ZhuangJinzhao ZhouYu-Cheng ChangYu-Kai WangChin-Teng Lin
|links| http://arxiv.org/abs/2408.10908v1 |
|updated| 2024-08-20 14:51:51 UTC |
|summary| This paper presents a pioneering exploration into the integration offine-grained human supervision within the autonomous driving domain to enhancesystem performance. The current advances in End-to-End autonomous drivingnormally are data-driven and rely on given expert trials. However thisreliance limits the systems generalizability and their ability to earn humantrust. Addressing this gap our research introduces a novel approach bysynchronously collecting data from human and machine drivers under identicaldriving scenarios focusing on eye-tracking and brainwave data to guide machineperception and decision-making processes. This paper utilizes the Carlasimulation to evaluate the impact brought by human behavior guidance.Experimental results show that using human attention to guide machine attentioncould bring a significant improvement in driving performance. However guidanceby human intention still remains a challenge. This paper pioneers a promisingdirection and potential for utilizing human behavior guidance to enhanceautonomous systems. |


| Item |Content|
| --- |---|
|idx| 2408.10905v1 |
|title| The impact of labeling automotive AI as "trustworthy" or "reliable" on user evaluation and technology acceptance |
|authors| John DorschOphelia Deroy
|links| http://arxiv.org/abs/2408.10905v1 |
|updated| 2024-08-20 14:48:24 UTC |
|summary| This study explores whether labeling AI as trustworthy or reliableinfluences user perceptions and acceptance of automotive AI technologies. Usinga one-way between-subjects design the research involved 478 onlineparticipants who were presented with guidelines for either trustworthy orreliable AI. Participants then evaluated three vignette scenarios and completeda modified version of the Technology Acceptance Model which included variablessuch as perceived ease of use human-like trust and overall attitude. Althoughlabeling AI as trustworthy did not significantly influence judgments onspecific scenarios it increased perceived ease of use and human-like trustparticularly benevolence. This suggests a positive impact on usability and ananthropomorphic effect on user perceptions. The study provides insights intohow specific labels can influence attitudes toward AI technology. |


| Item |Content|
| --- |---|
|idx| 2408.10903v1 |
|title| BEYOND DIALOGUE: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model |
|authors| Yeyong YuRusheng YuHaojie WeiZhanqiu ZhangQuan Qian
|links| http://arxiv.org/abs/2408.10903v1 |
|updated| 2024-08-20 14:47:38 UTC |
|summary| The rapid advancement of large language models LLMs has revolutionizedrole-playing enabling the development of general role-playing models. Howevercurrent role-playing training has two significant issues: I Using apredefined role profile to prompt dialogue training for specific scenariosusually leads to inconsistencies and even conflicts between the dialogue andthe profile resulting in training biases. II The model learns to imitate therole based solely on the profile neglecting profile-dialogue alignment at thesentence level. In this work we propose a simple yet effective frameworkcalled BEYOND DIALOGUE designed to overcome these hurdles. This frameworkinnovatively introduces beyond dialogue tasks to align dialogue with profiletraits based on each specific scenario thereby eliminating biases duringtraining. Furthermore by adopting an innovative prompting mechanism thatgenerates reasoning outcomes for training the framework allows the model toachieve fine-grained alignment between profile and dialogue at the sentencelevel. The aforementioned methods are fully automated and low-cost.Additionally the integration of automated dialogue and objective evaluationmethods forms a comprehensive framework paving the way for generalrole-playing. Experimental results demonstrate that our model excels inadhering to and reflecting various dimensions of role profiles outperformingmost proprietary general and specialized role-playing baselines. All code anddatasets are available at https://github.com/yuyouyu32/BeyondDialogue. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2408.11021v1 |
|title| Athena: Safe Autonomous Agents with Verbal Contrastive Learning |
|authors| Tanmana SadhuAli PesaranghaderYanan ChenDong Hoon Yi
|links| http://arxiv.org/abs/2408.11021v1 |
|updated| 2024-08-20 17:21:10 UTC |
|summary| Due to emergent capabilities large language models LLMs have been utilizedas language-based agents to perform a variety of tasks and make decisions withan increasing degree of autonomy. These autonomous agents can understandhigh-level instructions interact with their environments and execute complextasks using a selection of tools available to them. As the capabilities of theagents expand ensuring their safety and trustworthiness becomes moreimperative. In this study we introduce the Athena framework which leveragesthe concept of verbal contrastive learning where past safe and unsafetrajectories are used as in-context contrastive examples to guide the agenttowards safety while fulfilling a given task. The framework also incorporates acritiquing mechanism to guide the agent to prevent risky actions at every step.Furthermore due to the lack of existing benchmarks on the safety reasoningability of LLM-based agents we curate a set of 80 toolkits across 8 categorieswith 180 scenarios to provide a safety evaluation benchmark. Our experimentalevaluation with both closed- and open-source LLMs indicates verbalcontrastive learning and interaction-level critiquing improve the safety ratesignificantly. |


| Item |Content|
| --- |---|
|idx| 2408.10878v1 |
|title| DBHP: Trajectory Imputation in Multi-Agent Sports Using Derivative-Based Hybrid Prediction |
|authors| Hanjun ChoiHyunsung KimMinho LeeChang-Jo KimJinsung YoonSang-Ki Ko
|links| http://arxiv.org/abs/2408.10878v1 |
|updated| 2024-08-20 14:08:16 UTC |
|summary| Many spatiotemporal domains handle multi-agent trajectory data but inreal-world scenarios collected trajectory data are often partially missing dueto various reasons. While existing approaches demonstrate good performance intrajectory imputation they face challenges in capturing the complex dynamicsand interactions between agents due to a lack of physical constraints thatgovern realistic trajectories leading to suboptimal results. To address thisissue the paper proposes a Derivative-Based Hybrid Prediction DBHP frameworkthat can effectively impute multiple agents missing trajectories. First aneural network equipped with Set Transformers produces a naive prediction ofmissing trajectories while satisfying the permutation-equivariance in terms ofthe order of input agents. Then the framework makes alternative predictionsleveraging velocity and acceleration information and combines all thepredictions with properly determined weights to provide final imputedtrajectories. In this way our proposed framework not only accurately predictsposition velocity and acceleration values but also enforces the physicalrelationship between them eventually improving both the accuracy andnaturalness of the predicted trajectories. Accordingly the experiment resultsabout imputing player trajectories in team sports show that our frameworksignificantly outperforms existing imputation baselines. |


| Item |Content|
| --- |---|
|idx| 2408.10790v1 |
|title| Multi-Agent Based Simulation for Decentralized Electric Vehicle Charging Strategies and their Impacts |
|authors| Kristoffer ChristensenBo Nørregaard JørgensenZheng Grace Ma
|links| http://arxiv.org/abs/2408.10790v1 |
|updated| 2024-08-20 12:31:43 UTC |
|summary| The growing shift towards a Smart Grid involves integrating numerous newdigital energy solutions into the energy ecosystems to address problems arisingfrom the transition to carbon neutrality particularly in linking theelectricity and transportation sectors. Yet this shift brings challenges dueto mass electric vehicle adoption and the lack of methods to adequately assessvarious EV charging algorithms and their ecosystem impacts. This paperintroduces a multi-agent based simulation model validated through a case studyof a Danish radial distribution network serving 126 households. The studyreveals that traditional charging leads to grid overload by 2031 at 67 EVpenetration while decentralized strategies like Real-Time Pricing could causeoverloads as early as 2028. The developed multi-agent based simulationdemonstrates its ability to offer detailed hourly analysis of future loadprofiles in distribution grids and therefore can be applied to otherprospective scenarios in similar energy systems. |


| Item |Content|
| --- |---|
|idx| 2408.10783v1 |
|title| Multi-agent based modeling for investigating excess heat utilization from electrolyzer production to district heating network |
|authors| Kristoffer ChristensenBo Nørregaard JørgensenZheng Grace Ma
|links| http://arxiv.org/abs/2408.10783v1 |
|updated| 2024-08-20 12:21:21 UTC |
|summary| Power-to-Hydrogen is crucial for the renewable energy transition yetexisting literature lacks business models for the significant excess heat itgenerates. This study addresses this by evaluating three models for sellingelectrolyzer-generated heat to district heating grids: constant flexible andrenewable-source hydrogen production with and without heat sales. Usingagent-based modeling and multi-criteria decision-making methods VIKOR TOPSISPROMETHEE it finds that selling excess heat can cut hydrogen production costsby 5.6. The optimal model operates flexibly with electricity spot pricesincludes heat sales and maintains a hydrogen price of 3.3 EUR/kg.Environmentally hydrogen production from grid electricity could emit up to13783.8 tons of CO2 over four years from 2023. The best economic andenvironmental model uses renewable sources and sells heat at 3.5 EUR/kg |


| Item |Content|
| --- |---|
|idx| 2408.10773v1 |
|title| Multi-Agent Based Simulation for Investigating Centralized Charging Strategies and their Impact on Electric Vehicle Home Charging Ecosystem |
|authors| Kristoffer ChristensenBo Nørregaard JørgensenZheng Grace Ma
|links| http://arxiv.org/abs/2408.10773v1 |
|updated| 2024-08-20 12:12:35 UTC |
|summary| This paper addresses the critical integration of electric vehicles EVs intothe electricity grid which is essential for achieving carbon neutrality by2050. The rapid increase in EV adoption poses significant challenges to theexisting grid infrastructure particularly in managing the increasingelectricity demand and mitigating the risk of grid overloads. Centralized EVcharging strategies are investigated due to their potential to optimize gridstability and efficiency compared to decentralized approaches that mayexacerbate grid stress. Utilizing a multi-agent based simulation model thestudy provides a realistic representation of the electric vehicle home chargingecosystem in a case study of Strib Denmark. The findings show that theEarliest-deadline-first and Round Robin perform best with 100 EV adoption interms of EV user satisfaction. The simulation considers a realistic adoptioncurve EV charging strategies EV models and driving patterns to capture thefull ecosystem dynamics over a long-term period with high resolution hourly.Additionally the study offers detailed load profiles for future distributiongrids demonstrating how centralized charging strategies can efficiently managegrid loads and prevent overloads. |


