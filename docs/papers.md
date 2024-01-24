# cs.CL 

| Item |Content|
| --- |---|
|idx| 2401.12208v1 |
|title| CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation |
|authors| Zhihong ChenMaya VarmaJean-Benoit DelbrouckMagdalini PaschaliLouis BlankemeierDave Van VeenJeya Maria Jose ValanarasuAlaa YoussefJoseph Paul CohenEduardo Pontes ReisEmily B. TsaiAndrew JohnstonCameron OlsenTanishq Mathew AbrahamSergios GatidisAkshay S. ChaudhariCurtis Langlotz
|links| http://arxiv.org/abs/2401.12208v1 |
|updated| 2024-01-22 18:51:07 UTC |
|summary| Chest X-rays CXRs are the most frequently performed imaging test inclinical practice. Recent advances in the development of vision-languagefoundation models FMs give rise to the possibility of performing automatedCXR interpretation which can assist physicians with clinical decision-makingand improve patient outcomes. However developing FMs that can accuratelyinterpret CXRs is challenging due to the 1 limited availability oflarge-scale vision-language datasets in the medical image domain 2 lack ofvision and language encoders that can capture the complexities of medical dataand 3 absence of evaluation frameworks for benchmarking the abilities of FMson CXR interpretation. In this work we address these challenges by firstintroducing emphCheXinstruct - a large-scale instruction-tuning datasetcurated from 28 publicly-available datasets. We then present emphCheXagent -an instruction-tuned FM capable of analyzing and summarizing CXRs. To buildCheXagent we design a clinical large language model LLM for parsingradiology reports a vision encoder for representing CXR images and a networkto bridge the vision and language modalities. Finally we introduceemphCheXbench - a novel benchmark designed to systematically evaluate FMsacross 8 clinically-relevant CXR interpretation tasks. Extensive quantitativeevaluations and qualitative reviews with five expert radiologists demonstratethat CheXagent outperforms previously-developed general- and medical-domain FMson CheXbench tasks. Furthermore in an effort to improve model transparency weperform a fairness evaluation across factors of sex race and age to highlightpotential performance disparities. Our project is aturlhttps://stanford-aimi.github.io/chexagent.html. |


| Item |Content|
| --- |---|
|idx| 2401.12200v1 |
|title| APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference |
|authors| Bowen ZhaoHannaneh HajishirziQingqing Cao
|links| http://arxiv.org/abs/2401.12200v1 |
|updated| 2024-01-22 18:39:40 UTC |
|summary| Fine-tuning and inference with large Language Models LM are generally knownto be expensive. Parameter-efficient fine-tuning over pretrained LMs reducestraining memory by updating a small number of LM parameters but does notimprove inference efficiency. Structured pruning improves LM inferenceefficiency by removing consistent parameter blocks yet often increasestraining memory and time. To improve both training and inference efficiency weintroduce APT that adaptively prunes and tunes parameters for the LMs. At theearly stage of fine-tuning APT dynamically adds salient tuning parameters forfast and accurate convergence while discarding unimportant parameters forefficiency. Compared to baselines our experiments show that APT maintains upto 98 task performance when pruning RoBERTa and T5 models with 40 parametersleft while keeping 86.4 LLaMA models performance with 70 parametersremained. Furthermore APT speeds up LMs fine-tuning by up to 8x and reduceslarge LMs memory training footprint by up to 70. |


| Item |Content|
| --- |---|
|idx| 2401.12192v1 |
|title| Text Embedding Inversion Attacks on Multilingual Language Models |
|authors| Yiyi ChenHeather LentJohannes Bjerva
|links| http://arxiv.org/abs/2401.12192v1 |
|updated| 2024-01-22 18:34:42 UTC |
|summary| Representing textual information as real-numbered embeddings has become thenorm in NLP. Moreover with the rise of public interest in large languagemodels LLMs Embeddings as a Service EaaS has rapidly gained traction as abusiness model. This is not without outstanding security risks as previousresearch has demonstrated that sensitive data can be reconstructed fromembeddings even without knowledge of the underlying model that generated them.However such work is limited by its sole focus on English leaving all otherlanguages vulnerable to attacks by malicious actors. As many international andmultilingual companies leverage EaaS there is an urgent need for research intomultilingual LLM security. To this end this work investigates LLM securityfrom the perspective of multilingual embedding inversion. Concretely we definethe problem of black-box multilingual and cross-lingual inversion attacks withspecial attention to a cross-domain scenario. Our findings reveal thatmultilingual models are potentially more vulnerable to inversion attacks thantheir monolingual counterparts. This stems from the reduced data requirementsfor achieving comparable inversion performance in settings where the underlyinglanguage is not known a-priori. To our knowledge this work is the first todelve into multilinguality within the context of inversion attacks and ourfindings highlight the need for further investigation and enhanced defenses inthe area of NLP Security. |


| Item |Content|
| --- |---|
|idx| 2401.12187v1 |
|title| WARM: On the Benefits of Weight Averaged Reward Models |
|authors| Alexandre RaméNino VieillardLéonard HussenotRobert DadashiGeoffrey CideronOlivier BachemJohan Ferret
|links| http://arxiv.org/abs/2401.12187v1 |
|updated| 2024-01-22 18:27:08 UTC |
|summary| Aligning large language models LLMs with human preferences throughreinforcement learning RLHF can lead to reward hacking where LLMs exploitfailures in the reward model RM to achieve seemingly high rewards withoutmeeting the underlying objectives. We identify two primary challenges whendesigning RMs to mitigate reward hacking: distribution shifts during the RLprocess and inconsistencies in human preferences. As a solution we proposeWeight Averaged Reward Models WARM first fine-tuning multiple RMs thenaveraging them in the weight space. This strategy follows the observation thatfine-tuned weights remain linearly mode connected when sharing the samepre-training. By averaging weights WARM improves efficiency compared to thetraditional ensembling of predictions while improving reliability underdistribution shifts and robustness to preference inconsistencies. Ourexperiments on summarization tasks using best-of-N and RL methods shows thatWARM improves the overall quality and alignment of LLM predictions forexample a policy RL fine-tuned with WARM has a 79.4 win rate against a policyRL fine-tuned with a single RM. |


| Item |Content|
| --- |---|
|idx| 2401.12181v1 |
|title| Universal Neurons in GPT2 Language Models |
|authors| Wes GurneeTheo HorsleyZifan Carl GuoTara Rezaei KheirkhahQinyi SunWill HathawayNeel NandaDimitris Bertsimas
|links| http://arxiv.org/abs/2401.12181v1 |
|updated| 2024-01-22 18:11:01 UTC |
|summary| A basic question within the emerging field of mechanistic interpretability isthe degree to which neural networks learn the same underlying mechanisms. Inother words are neural mechanisms universal across different models In thiswork we study the universality of individual neurons across GPT2 modelstrained from different initial random seeds motivated by the hypothesis thatuniversal neurons are likely to be interpretable. In particular we computepairwise correlations of neuron activations over 100 million tokens for everyneuron pair across five different seeds and find that 1-5 of neurons areuniversal that is pairs of neurons which consistently activate on the sameinputs. We then study these universal neurons in detail finding that theyusually have clear interpretations and taxonomize them into a small number ofneuron families. We conclude by studying patterns in neuron weights toestablish several universal functional roles of neurons in simple circuits:deactivating attention heads changing the entropy of the next tokendistribution and predicting the next token to not be within a particularset. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2401.12205v1 |
|title| Retrieval-Guided Reinforcement Learning for Boolean Circuit Minimization |
|authors| Animesh Basak ChowdhuryMarco RomanelliBenjamin TanRamesh KarriSiddharth Garg
|links| http://arxiv.org/abs/2401.12205v1 |
|updated| 2024-01-22 18:46:30 UTC |
|summary| Logic synthesis a pivotal stage in chip design entails optimizing chipspecifications encoded in hardware description languages like Verilog intohighly efficient implementations using Boolean logic gates. The processinvolves a sequential application of logic minimization heuristics synthesisrecipe with their arrangement significantly impacting crucial metrics suchas area and delay. Addressing the challenge posed by the broad spectrum ofdesign complexities - from variations of past designs e.g. adders andmultipliers to entirely novel configurations e.g. innovative processorinstructions - requires a nuanced synthesis recipe guided by human expertiseand intuition. This study conducts a thorough examination of learning andsearch techniques for logic synthesis unearthing a surprising revelation:pre-trained agents when confronted with entirely novel designs may veer offcourse detrimentally affecting the search trajectory. We present ABC-RL ameticulously tuned alpha parameter that adeptly adjusts recommendations frompre-trained agents during the search process. Computed based on similarityscores through nearest neighbor retrieval from the training dataset ABC-RLyields superior synthesis recipes tailored for a wide array of hardwaredesigns. Our findings showcase substantial enhancements in theQuality-of-result QoR of synthesized circuits boasting improvements of up to24.8 compared to state-of-the-art techniques. Furthermore ABC-RL achieves animpressive up to 9x reduction in runtime iso-QoR when compared to currentstate-of-the-art methodologies. |


| Item |Content|
| --- |---|
|idx| 2401.12203v1 |
|title| Unsupervised Machine Learning for the Classification of Astrophysical X-ray Sources |
|authors| Víctor Samuel Pérez-DíazJuan Rafael Martínez-GalarzaAlexander CaicedoRaffaele D'Abrusco
|links| http://arxiv.org/abs/2401.12203v1 |
|updated| 2024-01-22 18:42:31 UTC |
|summary| The automatic classification of X-ray detections is a necessary step inextracting astrophysical information from compiled catalogs of astrophysicalsources. Classification is useful for the study of individual objectsstatistics for population studies as well as for anomaly detection i.e. theidentification of new unexplored phenomena including transients and spectrallyextreme sources. Despite the importance of this task classification remainschallenging in X-ray astronomy due to the lack of optical counterparts andrepresentative training sets. We develop an alternative methodology thatemploys an unsupervised machine learning approach to provide probabilisticclasses to Chandra Source Catalog sources with a limited number of labeledsources and without ancillary information from optical and infrared catalogs.We provide a catalog of probabilistic classes for 8756 sources comprising atotal of 14507 detections and demonstrate the success of the method atidentifying emission from young stellar objects as well as distinguishingbetween small-scale and large-scale compact accretors with a significant levelof confidence. We investigate the consistency between the distribution offeatures among classified objects and well-established astrophysical hypothesessuch as the unified AGN model. This provides interpretability to theprobabilistic classifier. Code and tables are available publicly throughGitHub. We provide a web playground for readers to explore our finalclassification at https://umlcaxs-playground.streamlit.app. |


| Item |Content|
| --- |---|
|idx| 2401.12202v1 |
|title| OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics |
|authors| Peiqi LiuYaswanth OrruChris PaxtonNur Muhammad Mahi ShafiullahLerrel Pinto
|links| http://arxiv.org/abs/2401.12202v1 |
|updated| 2024-01-22 18:42:20 UTC |
|summary| Remarkable progress has been made in recent years in the fields of visionlanguage and robotics. We now have vision models capable of recognizingobjects based on language queries navigation systems that can effectivelycontrol mobile systems and grasping models that can handle a wide range ofobjects. Despite these advancements general-purpose applications of roboticsstill lag behind even though they rely on these fundamental capabilities ofrecognition navigation and grasping. In this paper we adopt a systems-firstapproach to develop a new Open Knowledge-based robotics framework calledOK-Robot. By combining Vision-Language Models VLMs for object detectionnavigation primitives for movement and grasping primitives for objectmanipulation OK-Robot offers a integrated solution for pick-and-dropoperations without requiring any training. To evaluate its performance we runOK-Robot in 10 real-world home environments. The results demonstrate thatOK-Robot achieves a 58.5 success rate in open-ended pick-and-drop tasksrepresenting a new state-of-the-art in Open Vocabulary Mobile ManipulationOVMM with nearly 1.8x the performance of prior work. On cleaner unclutteredenvironments OK-Robots performance increases to 82. However the mostimportant insight gained from OK-Robot is the critical role of nuanced detailswhen combining Open Knowledge systems like VLMs with robotic modules. Videos ofour experiments are available on our website: https://ok-robot.github.io |


| Item |Content|
| --- |---|
|idx| 2401.12192v1 |
|title| Text Embedding Inversion Attacks on Multilingual Language Models |
|authors| Yiyi ChenHeather LentJohannes Bjerva
|links| http://arxiv.org/abs/2401.12192v1 |
|updated| 2024-01-22 18:34:42 UTC |
|summary| Representing textual information as real-numbered embeddings has become thenorm in NLP. Moreover with the rise of public interest in large languagemodels LLMs Embeddings as a Service EaaS has rapidly gained traction as abusiness model. This is not without outstanding security risks as previousresearch has demonstrated that sensitive data can be reconstructed fromembeddings even without knowledge of the underlying model that generated them.However such work is limited by its sole focus on English leaving all otherlanguages vulnerable to attacks by malicious actors. As many international andmultilingual companies leverage EaaS there is an urgent need for research intomultilingual LLM security. To this end this work investigates LLM securityfrom the perspective of multilingual embedding inversion. Concretely we definethe problem of black-box multilingual and cross-lingual inversion attacks withspecial attention to a cross-domain scenario. Our findings reveal thatmultilingual models are potentially more vulnerable to inversion attacks thantheir monolingual counterparts. This stems from the reduced data requirementsfor achieving comparable inversion performance in settings where the underlyinglanguage is not known a-priori. To our knowledge this work is the first todelve into multilinguality within the context of inversion attacks and ourfindings highlight the need for further investigation and enhanced defenses inthe area of NLP Security. |


| Item |Content|
| --- |---|
|idx| 2401.12187v1 |
|title| WARM: On the Benefits of Weight Averaged Reward Models |
|authors| Alexandre RaméNino VieillardLéonard HussenotRobert DadashiGeoffrey CideronOlivier BachemJohan Ferret
|links| http://arxiv.org/abs/2401.12187v1 |
|updated| 2024-01-22 18:27:08 UTC |
|summary| Aligning large language models LLMs with human preferences throughreinforcement learning RLHF can lead to reward hacking where LLMs exploitfailures in the reward model RM to achieve seemingly high rewards withoutmeeting the underlying objectives. We identify two primary challenges whendesigning RMs to mitigate reward hacking: distribution shifts during the RLprocess and inconsistencies in human preferences. As a solution we proposeWeight Averaged Reward Models WARM first fine-tuning multiple RMs thenaveraging them in the weight space. This strategy follows the observation thatfine-tuned weights remain linearly mode connected when sharing the samepre-training. By averaging weights WARM improves efficiency compared to thetraditional ensembling of predictions while improving reliability underdistribution shifts and robustness to preference inconsistencies. Ourexperiments on summarization tasks using best-of-N and RL methods shows thatWARM improves the overall quality and alignment of LLM predictions forexample a policy RL fine-tuned with WARM has a 79.4 win rate against a policyRL fine-tuned with a single RM. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2401.12217v1 |
|title| Exploring Simple Open-Vocabulary Semantic Segmentation |
|authors| Zihang Lai
|links| http://arxiv.org/abs/2401.12217v1 |
|updated| 2024-01-22 18:59:29 UTC |
|summary| Open-vocabulary semantic segmentation models aim to accurately assign asemantic label to each pixel in an image from a set of arbitraryopen-vocabulary texts. In order to learn such pixel-level alignment currentapproaches typically rely on a combination of i image-level VL model e.g.CLIP ii ground truth masks and iii custom grouping encoders. In thispaper we introduce S-Seg a novel model that can achieve surprisingly strongperformance without depending on any of the above elements. S-Seg leveragespseudo-mask and language to train a MaskFormer and can be easily trained frompublicly available image-text datasets. Contrary to prior works our modeldirectly trains for pixel-level features and language alignment. Once trainedS-Seg generalizes well to multiple testing datasets without requiringfine-tuning. In addition S-Seg has the extra benefits of scalability with dataand consistently improvement when augmented with self-training. We believe thatour simple yet effective approach will serve as a solid baseline for futureresearch. |


| Item |Content|
| --- |---|
|idx| 2401.12216v1 |
|title| Mitigating Covariate Shift in Misspecified Regression with Applications to Reinforcement Learning |
|authors| Philip AmortilaTongyi CaoAkshay Krishnamurthy
|links| http://arxiv.org/abs/2401.12216v1 |
|updated| 2024-01-22 18:59:12 UTC |
|summary| A pervasive phenomenon in machine learning applications is distributionshift where training and deployment conditions for a machine learning modeldiffer. As distribution shift typically results in a degradation inperformance much attention has been devoted to algorithmic interventions thatmitigate these detrimental effects. In this paper we study the effect ofdistribution shift in the presence of model misspecification specificallyfocusing on L_infty-misspecified regression and adversarial covariateshift where the regression target remains fixed while the covariatedistribution changes arbitrarily. We show that empirical risk minimization orstandard least squares regression can result in undesirable misspecificationamplification where the error due to misspecification is amplified by thedensity ratio between the training and testing distributions. As our mainresult we develop a new algorithm -- inspired by robust optimizationtechniques -- that avoids this undesirable behavior resulting in nomisspecification amplification while still obtaining optimal statistical rates.As applications we use this regression procedure to obtain new guarantees inoffline and online reinforcement learning with misspecification and establishnew separations between previously studied structural conditions and notions ofcoverage. |


| Item |Content|
| --- |---|
|idx| 2401.12207v1 |
|title| Rate-Distortion-Perception Tradeoff Based on the Conditional-Distribution Perception Measure |
|authors| Sadaf SalehkalaibarJun ChenAshish KhistiWei Yu
|links| http://arxiv.org/abs/2401.12207v1 |
|updated| 2024-01-22 18:49:56 UTC |
|summary| We study the rate-distortion-perception RDP tradeoff for a memorylesssource model in the asymptotic limit of large block-lengths. Our perceptionmeasure is based on a divergence between the distributions of the source andreconstruction sequences conditioned on the encoder output which was firstproposed in 1 2. We consider the case when there is no shared randomnessbetween the encoder and the decoder. For the case of discrete memorylesssources we derive a single-letter characterization of the RDP function thussettling a problem that remains open for the marginal metric introduced in Blauand Michaeli 3 with no shared randomness. Our achievability scheme is basedon lossy source coding with a posterior reference map proposed in 4. For thecase of continuous valued sources under squared error distortion measure andsquared quadratic Wasserstein perception measure we also derive a single-lettercharacterization and show that a noise-adding mechanism at the decoder sufficesto achieve the optimal representation. For the case of zero perception loss weshow that our characterization interestingly coincides with the results for themarginal metric derived in 5 6 and again demonstrate that zero perceptionloss can be achieved with a 3-dB penalty in the minimum distortion. Finallywe specialize our results to the case of Gaussian sources. We derive the RDPfunction for vector Gaussian sources and propose a waterfilling type solution.We also partially characterize the RDP function for a mixture of vectorGaussians. |


| Item |Content|
| --- |---|
|idx| 2401.12205v1 |
|title| Retrieval-Guided Reinforcement Learning for Boolean Circuit Minimization |
|authors| Animesh Basak ChowdhuryMarco RomanelliBenjamin TanRamesh KarriSiddharth Garg
|links| http://arxiv.org/abs/2401.12205v1 |
|updated| 2024-01-22 18:46:30 UTC |
|summary| Logic synthesis a pivotal stage in chip design entails optimizing chipspecifications encoded in hardware description languages like Verilog intohighly efficient implementations using Boolean logic gates. The processinvolves a sequential application of logic minimization heuristics synthesisrecipe with their arrangement significantly impacting crucial metrics suchas area and delay. Addressing the challenge posed by the broad spectrum ofdesign complexities - from variations of past designs e.g. adders andmultipliers to entirely novel configurations e.g. innovative processorinstructions - requires a nuanced synthesis recipe guided by human expertiseand intuition. This study conducts a thorough examination of learning andsearch techniques for logic synthesis unearthing a surprising revelation:pre-trained agents when confronted with entirely novel designs may veer offcourse detrimentally affecting the search trajectory. We present ABC-RL ameticulously tuned alpha parameter that adeptly adjusts recommendations frompre-trained agents during the search process. Computed based on similarityscores through nearest neighbor retrieval from the training dataset ABC-RLyields superior synthesis recipes tailored for a wide array of hardwaredesigns. Our findings showcase substantial enhancements in theQuality-of-result QoR of synthesized circuits boasting improvements of up to24.8 compared to state-of-the-art techniques. Furthermore ABC-RL achieves animpressive up to 9x reduction in runtime iso-QoR when compared to currentstate-of-the-art methodologies. |


| Item |Content|
| --- |---|
|idx| 2401.12202v1 |
|title| OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics |
|authors| Peiqi LiuYaswanth OrruChris PaxtonNur Muhammad Mahi ShafiullahLerrel Pinto
|links| http://arxiv.org/abs/2401.12202v1 |
|updated| 2024-01-22 18:42:20 UTC |
|summary| Remarkable progress has been made in recent years in the fields of visionlanguage and robotics. We now have vision models capable of recognizingobjects based on language queries navigation systems that can effectivelycontrol mobile systems and grasping models that can handle a wide range ofobjects. Despite these advancements general-purpose applications of roboticsstill lag behind even though they rely on these fundamental capabilities ofrecognition navigation and grasping. In this paper we adopt a systems-firstapproach to develop a new Open Knowledge-based robotics framework calledOK-Robot. By combining Vision-Language Models VLMs for object detectionnavigation primitives for movement and grasping primitives for objectmanipulation OK-Robot offers a integrated solution for pick-and-dropoperations without requiring any training. To evaluate its performance we runOK-Robot in 10 real-world home environments. The results demonstrate thatOK-Robot achieves a 58.5 success rate in open-ended pick-and-drop tasksrepresenting a new state-of-the-art in Open Vocabulary Mobile ManipulationOVMM with nearly 1.8x the performance of prior work. On cleaner unclutteredenvironments OK-Robots performance increases to 82. However the mostimportant insight gained from OK-Robot is the critical role of nuanced detailswhen combining Open Knowledge systems like VLMs with robotic modules. Videos ofour experiments are available on our website: https://ok-robot.github.io |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2401.12217v1 |
|title| Exploring Simple Open-Vocabulary Semantic Segmentation |
|authors| Zihang Lai
|links| http://arxiv.org/abs/2401.12217v1 |
|updated| 2024-01-22 18:59:29 UTC |
|summary| Open-vocabulary semantic segmentation models aim to accurately assign asemantic label to each pixel in an image from a set of arbitraryopen-vocabulary texts. In order to learn such pixel-level alignment currentapproaches typically rely on a combination of i image-level VL model e.g.CLIP ii ground truth masks and iii custom grouping encoders. In thispaper we introduce S-Seg a novel model that can achieve surprisingly strongperformance without depending on any of the above elements. S-Seg leveragespseudo-mask and language to train a MaskFormer and can be easily trained frompublicly available image-text datasets. Contrary to prior works our modeldirectly trains for pixel-level features and language alignment. Once trainedS-Seg generalizes well to multiple testing datasets without requiringfine-tuning. In addition S-Seg has the extra benefits of scalability with dataand consistently improvement when augmented with self-training. We believe thatour simple yet effective approach will serve as a solid baseline for futureresearch. |


| Item |Content|
| --- |---|
|idx| 2401.12215v1 |
|title| Less Could Be Better: Parameter-efficient Fine-tuning Advances Medical Vision Foundation Models |
|authors| Chenyu LianHong-Yu ZhouYizhou YuLiansheng Wang
|links| http://arxiv.org/abs/2401.12215v1 |
|updated| 2024-01-22 18:59:07 UTC |
|summary| Parameter-efficient fine-tuning PEFT that was initially developed forexploiting pre-trained large language models has recently emerged as aneffective approach to perform transfer learning on computer vision tasks.However the effectiveness of PEFT on medical vision foundation models is stillunclear and remains to be explored. As a proof of concept we conducted adetailed empirical study on applying PEFT to chest radiography foundationmodels. Specifically we delved into LoRA a representative PEFT method andcompared it against full-parameter fine-tuning FFT on two self-supervisedradiography foundation models across three well-established chest radiographdatasets. Our results showed that LoRA outperformed FFT in 13 out of 18transfer learning tasks by at most 2.9 using fewer than 1 tunable parameters.Combining LoRA with foundation models we set up new state-of-the-art on arange of data-efficient learning tasks such as an AUROC score of 80.6 using1 labeled data on NIH ChestX-ray14. We hope this study can evoke moreattention from the community in the use of PEFT for transfer learning onmedical imaging tasks. Code and models are available athttps://github.com/RL4M/MED-PEFT. |


| Item |Content|
| --- |---|
|idx| 2401.12210v1 |
|title| Connecting the Dots: Leveraging Spatio-Temporal Graph Neural Networks for Accurate Bangla Sign Language Recognition |
|authors| Haz Sameen ShahgirKhondker Salman SayeedMd Toki TahmidTanjeem Azwad ZamanMd. Zarif Ul Alam
|links| http://arxiv.org/abs/2401.12210v1 |
|updated| 2024-01-22 18:52:51 UTC |
|summary| Recent advances in Deep Learning and Computer Vision have been successfullyleveraged to serve marginalized communities in various contexts. One such areais Sign Language - a primary means of communication for the deaf community.However so far the bulk of research efforts and investments have gone intoAmerican Sign Language and research activity into low-resource sign languages- especially Bangla Sign Language - has lagged significantly. In this researchpaper we present a new word-level Bangla Sign Language dataset - BdSL40 -consisting of 611 videos over 40 words along with two different approaches:one with a 3D Convolutional Neural Network model and another with a novel GraphNeural Network approach for the classification of BdSL40 dataset. This is thefirst study on word-level BdSL recognition and the dataset was transcribedfrom Indian Sign Language ISL using the Bangla Sign Language Dictionary1997. The proposed GNN model achieved an F1 score of 89. The studyhighlights the significant lexical and semantic similarity between BdSL WestBengal Sign Language and ISL and the lack of word-level datasets for BdSL inthe literature. We release the dataset and source code to stimulate furtherresearch. |


| Item |Content|
| --- |---|
|idx| 2401.12208v1 |
|title| CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation |
|authors| Zhihong ChenMaya VarmaJean-Benoit DelbrouckMagdalini PaschaliLouis BlankemeierDave Van VeenJeya Maria Jose ValanarasuAlaa YoussefJoseph Paul CohenEduardo Pontes ReisEmily B. TsaiAndrew JohnstonCameron OlsenTanishq Mathew AbrahamSergios GatidisAkshay S. ChaudhariCurtis Langlotz
|links| http://arxiv.org/abs/2401.12208v1 |
|updated| 2024-01-22 18:51:07 UTC |
|summary| Chest X-rays CXRs are the most frequently performed imaging test inclinical practice. Recent advances in the development of vision-languagefoundation models FMs give rise to the possibility of performing automatedCXR interpretation which can assist physicians with clinical decision-makingand improve patient outcomes. However developing FMs that can accuratelyinterpret CXRs is challenging due to the 1 limited availability oflarge-scale vision-language datasets in the medical image domain 2 lack ofvision and language encoders that can capture the complexities of medical dataand 3 absence of evaluation frameworks for benchmarking the abilities of FMson CXR interpretation. In this work we address these challenges by firstintroducing emphCheXinstruct - a large-scale instruction-tuning datasetcurated from 28 publicly-available datasets. We then present emphCheXagent -an instruction-tuned FM capable of analyzing and summarizing CXRs. To buildCheXagent we design a clinical large language model LLM for parsingradiology reports a vision encoder for representing CXR images and a networkto bridge the vision and language modalities. Finally we introduceemphCheXbench - a novel benchmark designed to systematically evaluate FMsacross 8 clinically-relevant CXR interpretation tasks. Extensive quantitativeevaluations and qualitative reviews with five expert radiologists demonstratethat CheXagent outperforms previously-developed general- and medical-domain FMson CheXbench tasks. Furthermore in an effort to improve model transparency weperform a fairness evaluation across factors of sex race and age to highlightpotential performance disparities. Our project is aturlhttps://stanford-aimi.github.io/chexagent.html. |


| Item |Content|
| --- |---|
|idx| 2401.12202v1 |
|title| OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics |
|authors| Peiqi LiuYaswanth OrruChris PaxtonNur Muhammad Mahi ShafiullahLerrel Pinto
|links| http://arxiv.org/abs/2401.12202v1 |
|updated| 2024-01-22 18:42:20 UTC |
|summary| Remarkable progress has been made in recent years in the fields of visionlanguage and robotics. We now have vision models capable of recognizingobjects based on language queries navigation systems that can effectivelycontrol mobile systems and grasping models that can handle a wide range ofobjects. Despite these advancements general-purpose applications of roboticsstill lag behind even though they rely on these fundamental capabilities ofrecognition navigation and grasping. In this paper we adopt a systems-firstapproach to develop a new Open Knowledge-based robotics framework calledOK-Robot. By combining Vision-Language Models VLMs for object detectionnavigation primitives for movement and grasping primitives for objectmanipulation OK-Robot offers a integrated solution for pick-and-dropoperations without requiring any training. To evaluate its performance we runOK-Robot in 10 real-world home environments. The results demonstrate thatOK-Robot achieves a 58.5 success rate in open-ended pick-and-drop tasksrepresenting a new state-of-the-art in Open Vocabulary Mobile ManipulationOVMM with nearly 1.8x the performance of prior work. On cleaner unclutteredenvironments OK-Robots performance increases to 82. However the mostimportant insight gained from OK-Robot is the critical role of nuanced detailswhen combining Open Knowledge systems like VLMs with robotic modules. Videos ofour experiments are available on our website: https://ok-robot.github.io |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2401.12216v1 |
|title| Mitigating Covariate Shift in Misspecified Regression with Applications to Reinforcement Learning |
|authors| Philip AmortilaTongyi CaoAkshay Krishnamurthy
|links| http://arxiv.org/abs/2401.12216v1 |
|updated| 2024-01-22 18:59:12 UTC |
|summary| A pervasive phenomenon in machine learning applications is distributionshift where training and deployment conditions for a machine learning modeldiffer. As distribution shift typically results in a degradation inperformance much attention has been devoted to algorithmic interventions thatmitigate these detrimental effects. In this paper we study the effect ofdistribution shift in the presence of model misspecification specificallyfocusing on L_infty-misspecified regression and adversarial covariateshift where the regression target remains fixed while the covariatedistribution changes arbitrarily. We show that empirical risk minimization orstandard least squares regression can result in undesirable misspecificationamplification where the error due to misspecification is amplified by thedensity ratio between the training and testing distributions. As our mainresult we develop a new algorithm -- inspired by robust optimizationtechniques -- that avoids this undesirable behavior resulting in nomisspecification amplification while still obtaining optimal statistical rates.As applications we use this regression procedure to obtain new guarantees inoffline and online reinforcement learning with misspecification and establishnew separations between previously studied structural conditions and notions ofcoverage. |


| Item |Content|
| --- |---|
|idx| 2401.12058v1 |
|title| The Dimension Strikes Back with Gradients: Generalization of Gradient Methods in Stochastic Convex Optimization |
|authors| Matan SchlisermanUri ShermanTomer Koren
|links| http://arxiv.org/abs/2401.12058v1 |
|updated| 2024-01-22 15:50:32 UTC |
|summary| We study the generalization performance of gradient methods in thefundamental stochastic convex optimization setting focusing on its dimensiondependence. First for full-batch gradient descent GD we give a constructionof a learning problem in dimension dOn2 where the canonical version ofGD tuned for optimal performance of the empirical risk trained with ntraining examples converges with constant probability to an approximateempirical risk minimizer with Omega1 population excess risk. Our boundtranslates to a lower bound of Omega sqrtd on the number of trainingexamples required for standard GD to reach a non-trivial test error answeringan open question raised by Feldman 2016 and Amir Koren and Livni 2021band showing that a non-trivial dimension dependence is unavoidable.Furthermore for standard one-pass stochastic gradient descent SGD we showthat an application of the same construction technique provides a similarOmegasqrtd lower bound for the sample complexity of SGD to reach anon-trivial empirical error despite achieving optimal test performance. Thisagain provides an exponential improvement in the dimension dependence comparedto previous work Koren Livni Mansour and Sherman 2022 resolving an openquestion left therein. |


| Item |Content|
| --- |---|
|idx| 2401.12000v1 |
|title| Integrating Statistical Significance and Discriminative Power in Pattern Discovery |
|authors| Leonardo AlexandreRafael S. CostaRui Henriques
|links| http://arxiv.org/abs/2401.12000v1 |
|updated| 2024-01-22 14:51:01 UTC |
|summary| Pattern discovery plays a central role in both descriptive and predictivetasks across multiple domains. Actionable patterns must meet rigorousstatistical significance criteria and in the presence of target variablesfurther uphold discriminative power. Our work addresses the underexplored areaof guiding pattern discovery by integrating statistical significance anddiscriminative power criteria into state-of-the-art algorithms while preservingpattern quality. We also address how pattern quality thresholds imposed bysome algorithms can be rectified to accommodate these additional criteria. Totest the proposed methodology we select the triclustering task as the guidingpattern discovery case and extend well-known greedy and multi-objectiveoptimization triclustering algorithms delta-Trimax and TriGen that usevarious pattern quality criteria such as Mean Squared Residual MSR LeastSquared Lines LSL and Multi Slope Measure MSL. Results from three casestudies show the role of the proposed methodology in discovering patterns withpronounced improvements of discriminative power and statistical significancewithout quality deterioration highlighting its importance in supervisedlyguiding the search. Although the proposed methodology is motivated overmultivariate time series data it can be straightforwardly extended to patterndiscovery tasks involving multivariate N-way N3 transactional andsequential data structures.  Availability: The code is freely available athttps://github.com/JupitersMight/MOF_Triclustering under the MIT license. |


| Item |Content|
| --- |---|
|idx| 2401.11974v1 |
|title| Cross-Validation Conformal Risk Control |
|authors| Kfir M. CohenSangwoo ParkOsvaldo SimeoneShlomo Shamai
|links| http://arxiv.org/abs/2401.11974v1 |
|updated| 2024-01-22 14:26:02 UTC |
|summary| Conformal risk control CRC is a recently proposed technique that appliespost-hoc to a conventional point predictor to provide calibration guarantees.Generalizing conformal prediction CP with CRC calibration is ensured for aset predictor that is extracted from the point predictor to control a riskfunction such as the probability of miscoverage or the false negative rate. Theoriginal CRC requires the available data set to be split between training andvalidation data sets. This can be problematic when data availability islimited resulting in inefficient set predictors. In this paper a novel CRCmethod is introduced that is based on cross-validation rather than onvalidation as the original CRC. The proposed cross-validation CRC CV-CRCextends a version of the jackknife-minmax from CP to CRC allowing for thecontrol of a broader range of risk functions. CV-CRC is proved to offertheoretical guarantees on the average risk of the set predictor. Furthermorenumerical experiments show that CV-CRC can reduce the average set size withrespect to CRC when the available data are limited. |


| Item |Content|
| --- |---|
|idx| 2401.11954v1 |
|title| RUMBoost: Gradient Boosted Random Utility Models |
|authors| Nicolas SalvadéTim Hillel
|links| http://arxiv.org/abs/2401.11954v1 |
|updated| 2024-01-22 13:54:26 UTC |
|summary| This paper introduces the RUMBoost model a novel discrete choice modellingapproach that combines the interpretability and behavioural robustness ofRandom Utility Models RUMs with the generalisation and predictive ability ofdeep learning methods. We obtain the full functional form of non-linear utilityspecifications by replacing each linear parameter in the utility functions of aRUM with an ensemble of gradient boosted regression trees. This enablespiece-wise constant utility values to be imputed for all alternatives directlyfrom the data for any possible combination of input variables. We introduceadditional constraints on the ensembles to ensure three crucial features of theutility specifications: i dependency of the utilities of each alternative ononly the attributes of that alternative ii monotonicity of marginalutilities and iii an intrinsically interpretable functional form where theexact response of the model is known throughout the entire input space.Furthermore we introduce an optimisation-based smoothing technique thatreplaces the piece-wise constant utility values of alternative attributes withmonotonic piece-wise cubic splines to identify non-linear parameters withdefined gradient. We demonstrate the potential of the RUMBoost model comparedto various ML and Random Utility benchmark models for revealed preference modechoice data from London. The results highlight the great predictive performanceand the direct interpretability of our proposed approach. Furthermore thesmoothed attribute utility functions allow for the calculation of variousbehavioural indicators and marginal utilities. Finally we demonstrate theflexibility of our methodology by showing how the RUMBoost model can beextended to complex model specifications including attribute interactionscorrelation within alternative error terms and heterogeneity within thepopulation. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2401.12133v1 |
|title| VRMN-bD: A Multi-modal Natural Behavior Dataset of Immersive Human Fear Responses in VR Stand-up Interactive Games |
|authors| He ZhangXinyang LiYuanxi SunXinyi FuChristine QiuJohn M. Carroll
|links| http://arxiv.org/abs/2401.12133v1 |
|updated| 2024-01-22 17:15:02 UTC |
|summary| Understanding and recognizing emotions are important and challenging issuesin the metaverse era. Understanding identifying and predicting fear which isone of the fundamental human emotions in virtual reality VR environmentsplays an essential role in immersive game development scene development andnext-generation virtual human-computer interaction applications. In thisarticle we used VR horror games as a medium to analyze fear emotions bycollecting multi-modal data posture audio and physiological signals from 23players. We used an LSTM-based model to predict fear with accuracies of 65.31and 90.47 under 6-level classification no fear and five different levels offear and 2-level classification no fear and fear respectively. Weconstructed a multi-modal natural behavior dataset of immersive human fearresponses VRMN-bD and compared it with existing relevant advanced datasets.The results show that our dataset has fewer limitations in terms of collectionmethod data scale and audience scope. We are unique and advanced in targetingmulti-modal datasets of fear and behavior in VR stand-up interactiveenvironments. Moreover we discussed the implications of this work forcommunities and applications. The dataset and pre-trained model are availableat https://github.com/KindOPSTAR/VRMN-bD. |


| Item |Content|
| --- |---|
|idx| 2401.12125v1 |
|title| CodeTailor: Personalized Parsons Puzzles are Preferred Over AI-Generated Solutions to Support Learning |
|authors| Xinying HouZihan WuXu WangBarbara J. Ericson
|links| http://arxiv.org/abs/2401.12125v1 |
|updated| 2024-01-22 17:08:54 UTC |
|summary| Programming can be challenging for novices but it is difficult to providehigh-quality comprehensive and timely support at scale. Generative AI and itsproducts like ChatGPT can create a solution for most introductory programmingproblems. However students may become overly reliant on these tools for quickcode generation and homework completion leading to reduced engagement andlimited learning. In this work we present sys a system that utilizes largelanguage models LLM while still promoting students cognitive engagement.sys provides a personalized Parsons puzzle to support struggling students.In a Parsons puzzle students place mixed-up code blocks in the correct orderto solve a problem. A technical evaluation with 800 incorrect student codedemonstrated that sys can efficiently create high-quality correctpersonalized and concise Parsons puzzles for students. In a within-subjectsexperiment with 18 novice programmers most students rated using sys as moreengaging and they preferred sys for learning rather than simply receivingan AI-generated solution. Additionally students recalled more new elementsfrom the supported practice to the posttest after using sys compared towhen they simply received a direct solution. Qualitative observations andinterviews provided evidence for the benefits of sys including emphasizingalgorithmic thinking fostering continuity in learning promoting metacognitivereflection and boosting student confidence. We conclude by suggesting futuredesigns for applying generative AI in a way that minimizes over-reliance andenhances learning. |


| Item |Content|
| --- |---|
|idx| 2401.12076v1 |
|title| Human Impression of Humanoid Robots Mirroring Social Cues |
|authors| Di FuFares AbawiPhilipp AllgeuerStefan Wermter
|links| http://dx.doi.org/10.1145/3610978.3640580 |
|updated| 2024-01-22 16:14:57 UTC |
|summary| Mirroring non-verbal social cues such as affect or movement can enhancehuman-human and human-robot interactions in the real world. The roboticplatforms and control methods also impact peoples perception of human-robotinteraction. However limited studies have compared robot imitation acrossdifferent platforms and control methods. Our research addresses this gap byconducting two experiments comparing peoples perception of affective mirroringbetween the iCub and Pepper robots and movement mirroring between vision-basediCub control and Inertial Measurement Unit IMU-based iCub control. Wediscovered that the iCub robot was perceived as more humanlike than the Pepperrobot when mirroring affect. A vision-based controlled iCub outperformed theIMU-based controlled one in the movement mirroring task. Our findings suggestthat different robotic platforms impact peoples perception of robotsmirroring during HRI. The control method also contributes to the robotsmirroring performance. Our work sheds light on the design and application ofdifferent humanoid robots in the real world. |


| Item |Content|
| --- |---|
|idx| 2401.12064v1 |
|title| Market Responses to Genuine Versus Strategic Generosity: An Empirical Examination of NFT Charity Fundraisers |
|authors| Chen LiangMurat TuncGordon Burtch
|links| http://arxiv.org/abs/2401.12064v1 |
|updated| 2024-01-22 15:58:47 UTC |
|summary| Crypto donations now represent a significant fraction of charitable givingworldwide. Nonfungible token NFT charity fundraisers which involve the saleof NFTs of artistic works with the proceeds donated to philanthropic causeshave emerged as a novel development in this space. A unique aspect of NFTcharity fundraisers is the significant potential for donors to reap financialgains from the rising value of purchased NFTs. Questions may arise about themotivations of donors in these charity fundraisers resulting in a negativesocial image. NFT charity fundraisers thus offer a unique opportunity tounderstand the economic consequences of a donors social image. We investigatethese effects in the context of a large NFT charity fundraiser. We identify thecausal effect of purchasing an NFT within the charity fundraiser on a donorslater market outcomes by leveraging random variation in transaction processingtimes on the blockchain. Further we demonstrate a clear pattern ofheterogeneity based on an individuals decision to relist versus hold thepurchased charity NFTs a sign of strategic generosity and based on anindividuals degree of social exposure within the NFT marketplace. We show thatcharity-NFT relisters experience significant penalties in the market interms of the prices they are able to command on other NFT listingsparticularly among those who relist quickly and those who are more sociallyexposed. Our study underscores the growing importance of digital visibility andtraceability features that characterize crypto-philanthropy and onlinephilanthropy more broadly. |


| Item |Content|
| --- |---|
|idx| 2401.12032v1 |
|title| MINT: A wrapper to make multi-modal and multi-image AI models interactive |
|authors| Jan FreybergAbhijit Guha RoyTerry SpitzBeverly FreemanMike SchaekermannPatricia StrachanEva SchniderRenee WongDale R WebsterAlan KarthikesalingamYun LiuKrishnamurthy DvijothamUmesh Telang
|links| http://arxiv.org/abs/2401.12032v1 |
|updated| 2024-01-22 15:17:54 UTC |
|summary| During the diagnostic process doctors incorporate multimodal informationincluding imaging and the medical history - and similarly medical AIdevelopment has increasingly become multimodal. In this paper we tackle a moresubtle challenge: doctors take a targeted medical history to obtain only themost pertinent pieces of information how do we enable AI to do the same Wedevelop a wrapper method named MINT Make your model INTeractive thatautomatically determines what pieces of information are most valuable at eachstep and ask for only the most useful information. We demonstrate the efficacyof MINT wrapping a skin disease prediction model where multiple images and aset of optional answers to 25 standard metadata questions i.e. structuredmedical history are used by a multi-modal deep network to provide adifferential diagnosis. We show that MINT can identify whether metadata inputsare needed and if so which question to ask next. We also demonstrate that whencollecting multiple images MINT can identify if an additional image would bebeneficial and if so which type of image to capture. We showed that MINTreduces the number of metadata and image inputs needed by 82 and 36.2respectively while maintaining predictive performance. Using real-world AIdermatology system data we show that needing fewer inputs can retain usersthat may otherwise fail to complete the system submission and drop off withouta diagnosis. Qualitative examples show MINT can closely mimic the step-by-stepdecision making process of a clinical workflow and how this is different forstraight forward cases versus more difficult ambiguous cases. Finally wedemonstrate how MINT is robust to different underlying multi-model classifiersand can be easily adapted to user requirements without significant modelre-training. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2401.12159v1 |
|title| Transcending To Notions |
|authors| Sama Sai KarthikJayati DeshmukhSrinath Srinivasa
|links| http://arxiv.org/abs/2401.12159v1 |
|updated| 2024-01-22 17:54:07 UTC |
|summary| Social identities play an important role in the dynamics of human societiesand it can be argued that some sense of identification with a larger cause oridea plays a critical role in making humans act responsibly. Often socialactivists strive to get populations to identify with some cause or notion --like green energy diversity etc. in order to bring about desired socialchanges. We explore the problem of designing computational models for socialidentities in the context of autonomous AI agents. For this we propose anagent model that enables agents to identify with certain notions and show howthis affects collective outcomes. We also contrast between associations ofidentity with rational preferences. The proposed model is simulated in anapplication context of urban mobility where we show how changes in socialidentity affect mobility patterns and collective outcomes. |


| Item |Content|
| --- |---|
|idx| 2401.12108v1 |
|title| On-Time Delivery in Crowdshipping Systems: An Agent-Based Approach Using Streaming Data |
|authors| Jeremias DötterlRalf BrunsJürgen DunkelSascha Ossowski
|links| http://dx.doi.org/10.3233/FAIA200075 |
|updated| 2024-01-22 16:45:15 UTC |
|summary| In parcel delivery the last mile from the parcel hub to the customer iscostly especially for time-sensitive delivery tasks that have to be completedwithin hours after arrival. Recently crowdshipping has attracted increasedattention as a new alternative to traditional delivery modes. In crowdshippingprivate citizens the crowd perform short detours in their daily lives tocontribute to parcel delivery in exchange for small incentives. Howeverachieving desirable crowd behavior is challenging as the crowd is highlydynamic and consists of autonomous self-interested individuals. Leveragingcrowdshipping for time-sensitive deliveries remains an open challenge. In thispaper we present an agent-based approach to on-time parcel delivery withcrowds. Our system performs data stream processing on the couriers smartphonesensor data to predict delivery delays. Whenever a delay is predicted thesystem attempts to forge an agreement for transferring the parcel from thecurrent deliverer to a more promising courier nearby. Our experiments show thatthrough accurate delay predictions and purposeful task transfers many delayscan be prevented that would occur without our approach. |


| Item |Content|
| --- |---|
|idx| 2401.12079v1 |
|title| Collaborative Reinforcement Learning Based Unmanned Aerial Vehicle (UAV) Trajectory Design for 3D UAV Tracking |
|authors| Yujiao ZhuMingzhe ChenSihua WangYe HuYuchen LiuChangchuan Yin
|links| http://arxiv.org/abs/2401.12079v1 |
|updated| 2024-01-22 16:21:19 UTC |
|summary| In this paper the problem of using one active unmanned aerial vehicle UAVand four passive UAVs to localize a 3D target UAV in real time is investigated.In the considered model each passive UAV receives reflection signals from thetarget UAV which are initially transmitted by the active UAV. The receivedreflection signals allow each passive UAV to estimate the signal transmissiondistance which will be transmitted to a base station BS for the estimation ofthe position of the target UAV. Due to the movement of the target UAV eachactive/passive UAV must optimize its trajectory to continuously localize thetarget UAV. Meanwhile since the accuracy of the distance estimation depends onthe signal-to-noise ratio of the transmission signals the active UAV mustoptimize its transmit power. This problem is formulated as an optimizationproblem whose goal is to jointly optimize the transmit power of the active UAVand trajectories of both active and passive UAVs so as to maximize the targetUAV positioning accuracy. To solve this problem a Z function decompositionbased reinforcement learning ZD-RL method is proposed. Compared to valuefunction decomposition based RL VD-RL the proposed method can find theprobability distribution of the sum of future rewards to accurately estimatethe expected value of the sum of future rewards thus finding better transmitpower of the active UAV and trajectories for both active and passive UAVs andimproving target UAV positioning accuracy. Simulation results show that theproposed ZD-RL method can reduce the positioning errors by up to 39.4 and64.6 compared to VD-RL and independent deep RL methods respectively. |


| Item |Content|
| --- |---|
|idx| 2401.11881v1 |
|title| Modelling the Dynamics of Identity and Fairness in Ultimatum Game |
|authors| Janvi ChhabraJayati DeshmukhSrinath Srinivasa
|links| http://arxiv.org/abs/2401.11881v1 |
|updated| 2024-01-22 12:12:05 UTC |
|summary| Allocation games are zero-sum games that model the distribution of resourcesamong multiple agents. In this paper we explore the interplay between anelastic sense of subjective identity and its impact on notions of fairness inallocation. An elastic sense of identity in agents is known to lead toresponsible decision-making in non-cooperative non-zero-sum games likePrisoners Dilemma and is a desirable feature to add into agent models.However when it comes to allocation an elastic sense of identity can be shownto exacerbate inequities in allocation giving no rational incentive for agentsto act fairly towards one another. This lead us to introduce a sense offairness as an innate characteristic of autonomous agency. For this weimplement the well-known Ultimatum Game between two agents where their elasticsense of self controlled by a parameter called gamma and a sense offairness controlled by a parameter called tau are both varied. We studythe points at which agents find it no longer rational to identify with theother agent and uphold their sense of fairness and vice versa. Such a studyalso helps us discern the subtle difference between responsibility and fairnesswhen it comes to autonomous agency. |


| Item |Content|
| --- |---|
|idx| 2401.11880v1 |
|title| PsySafe: A Comprehensive Framework for Psychological-based Attack, Defense, and Evaluation of Multi-agent System Safety |
|authors| Zaibin ZhangYongting ZhangLijun LiHongzhi GaoLijun WangHuchuan LuFeng ZhaoYu QiaoJing Shao
|links| http://arxiv.org/abs/2401.11880v1 |
|updated| 2024-01-22 12:11:55 UTC |
|summary| Multi-agent systems augmented with Large Language Models LLMs demonstratesignificant capabilities for collective intelligence. However the potentialmisuse of this intelligence for malicious purposes presents significant risks.To date comprehensive research on the safety issues associated withmulti-agent systems remains limited. From the perspective of agent psychologywe discover that the dark psychological states of agents can lead to severesafety issues. To address these issues we propose a comprehensive frameworkgrounded in agent psychology. In our framework we focus on three aspects:identifying how dark personality traits in agents might lead to riskybehaviors designing defense strategies to mitigate these risks and evaluatingthe safety of multi-agent systems from both psychological and behavioralperspectives. Our experiments reveal several intriguing phenomena such as thecollective dangerous behaviors among agents agents propensity forself-reflection when engaging in dangerous behavior and the correlationbetween agents psychological assessments and their dangerous behaviors. Weanticipate that our framework and observations will provide valuable insightsfor further research into the safety of multi-agent systems. We will make ourdata and code publicly accessible at https:/github.com/AI4Good24/PsySafe. |


