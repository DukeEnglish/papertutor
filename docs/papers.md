# cs.CL 

| Item |Content|
| --- |---|
|idx| 2401.14400v1 |
|title| Modular Adaptation of Multilingual Encoders to Written Swiss German Dialect |
|authors| Jannis VamvasNoëmi AepliRico Sennrich
|links| http://arxiv.org/abs/2401.14400v1 |
|updated| 2024-01-25 18:59:32 UTC |
|summary| Creating neural text encoders for written Swiss German is challenging due toa dearth of training data combined with dialectal variation. In this paper webuild on several existing multilingual encoders and adapt them to Swiss Germanusing continued pre-training. Evaluation on three diverse downstream tasksshows that simply adding a Swiss German adapter to a modular encoder achieves97.5 of fully monolithic adaptation performance. We further find that for thetask of retrieving Swiss German sentences given Standard German queriesadapting a character-level model is more effective than the other adaptationstrategies. We release our code and the models trained for our experiments athttps://github.com/ZurichNLP/swiss-german-text-encoders |


| Item |Content|
| --- |---|
|idx| 2401.14373v1 |
|title| TURNA: A Turkish Encoder-Decoder Language Model for Enhanced Understanding and Generation |
|authors| Gökçe UludoğanZeynep Yirmibeşoğlu BalalFurkan AkkurtMelikşah TürkerOnur GüngörSusan Üsküdarlı
|links| http://arxiv.org/abs/2401.14373v1 |
|updated| 2024-01-25 18:24:13 UTC |
|summary| The recent advances in natural language processing have predominantly favoredwell-resourced English-centric models resulting in a significant gap withlow-resource languages. In this work we introduce the language model TURNAwhich is developed for the low-resource language Turkish and is capable of bothnatural language understanding and generation tasks. TURNA is pretrained withan encoder-decoder architecture based on the unified framework UL2 with adiverse corpus that we specifically curated for this purpose. We evaluatedTURNA with three generation tasks and five understanding tasks for Turkish. Theresults show that TURNA outperforms several multilingual models in bothunderstanding and generation tasks and competes with monolingual Turkishmodels in understanding tasks. TURNA is made available athttps://huggingface.co/boun-tabi-LMG/TURNA . |


| Item |Content|
| --- |---|
|idx| 2401.14367v1 |
|title| Genie: Achieving Human Parity in Content-Grounded Datasets Generation |
|authors| Asaf YehudaiBoaz CarmeliYosi MassOfir ArvivNathaniel MillsAssaf ToledoEyal ShnarchLeshem Choshen
|links| http://arxiv.org/abs/2401.14367v1 |
|updated| 2024-01-25 18:14:57 UTC |
|summary| The lack of high-quality data for content-grounded generation tasks has beenidentified as a major obstacle to advancing these tasks. To address this gapwe propose Genie a novel method for automatically generating high-qualitycontent-grounded data. It consists of three stages: a Content Preparationb Generation: creating task-specific examples from the content e.g.question-answer pairs or summaries. c Filtering mechanism aiming to ensurethe quality and faithfulness of the generated data. We showcase thismethodology by generating three large-scale synthetic data making wishes forLong-Form Question-Answering LFQA summarization and information extraction.In a human evaluation our generated data was found to be natural and of highquality. Furthermore we compare models trained on our data with models trainedon human-written data -- ELI5 and ASQA for LFQA and CNN-DailyMail forSummarization. We show that our models are on par with or outperforming modelstrained on human-generated data and consistently outperforming them infaithfulness. Finally we applied our method to create LFQA data within themedical domain and compared a model trained on it with models trained on otherdomains. |


| Item |Content|
| --- |---|
|idx| 2401.14360v1 |
|title| A Comparative Analysis of Noise Reduction Methods in Sentiment Analysis on Noisy Bengali Texts |
|authors| Kazi Toufique ElahiTasnuva Binte RahmanShakil ShahriarSamir SarkerMd. Tanvir Rouf ShawonG. M. Shahariar
|links| http://arxiv.org/abs/2401.14360v1 |
|updated| 2024-01-25 18:06:19 UTC |
|summary| While Bengali is considered a language with limited resources sentimentanalysis has been a subject of extensive research in the literature.Nevertheless there is a scarcity of exploration into sentiment analysisspecifically in the realm of noisy Bengali texts. In this paper we introduce adataset NC-SentNoB that we annotated manually to identify ten different typesof noise found in a pre-existing sentiment analysis dataset comprising ofaround 15K noisy Bengali texts. At first given an input noisy text weidentify the noise type addressing this as a multi-label classification task.Then we introduce baseline noise reduction methods to alleviate noise prior toconducting sentiment analysis. Finally we assess the performance of fine-tunedsentiment analysis models with both noisy and noise-reduced texts to makecomparisons. The experimental findings indicate that the noise reductionmethods utilized are not satisfactory highlighting the need for more suitablenoise reduction methods in future research endeavors. We have made theimplementation and dataset presented in this paper publicly available athttps://github.com/ktoufiquee/A-Comparative-Analysis-of-Noise-Reduction-Methods-in-Sentiment-Analysis-on-Noisy-Bengali-Texts |


| Item |Content|
| --- |---|
|idx| 2401.14295v1 |
|title| Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts |
|authors| Maciej BestaFlorim MemediZhenyu ZhangRobert GerstenbergerNils BlachPiotr NyczykMarcin CopikGrzegorz KwaśniewskiJürgen MüllerLukas GianinazziAles KubicekHubert NiewiadomskiOnur MutluTorsten Hoefler
|links| http://arxiv.org/abs/2401.14295v1 |
|updated| 2024-01-25 16:34:00 UTC |
|summary| The field of natural language processing NLP has witnessed significantprogress in recent years with a notable focus on improving large languagemodels LLM performance through innovative prompting techniques. Among theseprompt engineering coupled with structures has emerged as a promising paradigmwith designs such as Chain-of-Thought Tree of Thoughts or Graph of Thoughtsin which the overall LLM reasoning is guided by a structure such as a graph. Asillustrated with numerous examples this paradigm significantly enhances theLLMs capability to solve numerous tasks ranging from logical or mathematicalreasoning to planning or creative writing. To facilitate the understanding ofthis growing field and pave the way for future developments we devise ageneral blueprint for effective and efficient LLM reasoning schemes. For thiswe conduct an in-depth analysis of the prompt execution pipeline clarifyingand clearly defining different concepts. We then build the first taxonomy ofstructure-enhanced LLM reasoning schemes. We focus on identifying fundamentalclasses of harnessed structures and we analyze the representations of thesestructures algorithms executed with these structures and many others. Werefer to these structures as reasoning topologies because their representationbecomes to a degree spatial as they are contained within the LLM context. Ourstudy compares existing prompting schemes using the proposed taxonomydiscussing how certain design choices lead to different patterns in performanceand cost. We also outline theoretical underpinnings relationships betweenprompting and others parts of the LLM ecosystem such as knowledge bases andthe associated research challenges. Our work will help to advance future promptengineering techniques. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2401.14405v1 |
|title| Multimodal Pathway: Improve Transformers with Irrelevant Data from Other Modalities |
|authors| Yiyuan ZhangXiaohan DingKaixiong GongYixiao GeYing ShanXiangyu Yue
|links| http://arxiv.org/abs/2401.14405v1 |
|updated| 2024-01-25 18:59:58 UTC |
|summary| We propose to improve transformers of a specific modality with irrelevantdata from other modalities e.g. improve an ImageNet model with audio or pointcloud datasets. We would like to highlight that the data samples of the targetmodality are irrelevant to the other modalities which distinguishes our methodfrom other works utilizing paired e.g. CLIP or interleaved data of differentmodalities. We propose a methodology named Multimodal Pathway - given a targetmodality and a transformer designed for it we use an auxiliary transformertrained with data of another modality and construct pathways to connectcomponents of the two models so that data of the target modality can beprocessed by both models. In this way we utilize the universalsequence-to-sequence modeling abilities of transformers obtained from twomodalities. As a concrete implementation we use a modality-specific tokenizerand task-specific head as usual but utilize the transformer blocks of theauxiliary model via a proposed method named Cross-Modal Re-parameterizationwhich exploits the auxiliary weights without any inference costs. On the imagepoint cloud video and audio recognition tasks we observe significant andconsistent performance improvements with irrelevant data from other modalities.The code and models are available at https://github.com/AILab-CVC/M2PT. |


| Item |Content|
| --- |---|
|idx| 2401.14403v1 |
|title| Adaptive Mobile Manipulation for Articulated Objects In the Open World |
|authors| Haoyu XiongRussell MendoncaKenneth ShawDeepak Pathak
|links| http://arxiv.org/abs/2401.14403v1 |
|updated| 2024-01-25 18:59:44 UTC |
|summary| Deploying robots in open-ended unstructured environments such as homes hasbeen a long-standing research problem. However robots are often studied onlyin closed-off lab settings and prior mobile manipulation work is restricted topick-move-place which is arguably just the tip of the iceberg in this area. Inthis paper we introduce Open-World Mobile Manipulation System a full-stackapproach to tackle realistic articulated object operation e.g. real-worlddoors cabinets drawers and refrigerators in open-ended unstructuredenvironments. The robot utilizes an adaptive learning framework to initiallylearns from a small set of data through behavior cloning followed by learningfrom online practice on novel objects that fall outside the trainingdistribution. We also develop a low-cost mobile manipulation hardware platformcapable of safe and autonomous online adaptation in unstructured environmentswith a cost of around 20000 USD. In our experiments we utilize 20 articulateobjects across 4 buildings in the CMU campus. With less than an hour of onlinelearning for each object the system is able to increase success rate from 50of BC pre-training to 95 using online adaptation. Video results athttps://open-world-mobilemanip.github.io/ |


| Item |Content|
| --- |---|
|idx| 2401.14373v1 |
|title| TURNA: A Turkish Encoder-Decoder Language Model for Enhanced Understanding and Generation |
|authors| Gökçe UludoğanZeynep Yirmibeşoğlu BalalFurkan AkkurtMelikşah TürkerOnur GüngörSusan Üsküdarlı
|links| http://arxiv.org/abs/2401.14373v1 |
|updated| 2024-01-25 18:24:13 UTC |
|summary| The recent advances in natural language processing have predominantly favoredwell-resourced English-centric models resulting in a significant gap withlow-resource languages. In this work we introduce the language model TURNAwhich is developed for the low-resource language Turkish and is capable of bothnatural language understanding and generation tasks. TURNA is pretrained withan encoder-decoder architecture based on the unified framework UL2 with adiverse corpus that we specifically curated for this purpose. We evaluatedTURNA with three generation tasks and five understanding tasks for Turkish. Theresults show that TURNA outperforms several multilingual models in bothunderstanding and generation tasks and competes with monolingual Turkishmodels in understanding tasks. TURNA is made available athttps://huggingface.co/boun-tabi-LMG/TURNA . |


| Item |Content|
| --- |---|
|idx| 2401.14371v1 |
|title| Efficient Optimisation of Physical Reservoir Computers using only a Delayed Input |
|authors| Enrico PiccoLina JaurigueKathy LüdgeSerge Massar
|links| http://arxiv.org/abs/2401.14371v1 |
|updated| 2024-01-25 18:20:37 UTC |
|summary| We present an experimental validation of a recently proposed optimizationtechnique for reservoir computing using an optoelectronic setup. Reservoircomputing is a robust framework for signal processing applications and thedevelopment of efficient optimization approaches remains a key challenge. Thetechnique we address leverages solely a delayed version of the input signal toidentify the optimal operational region of the reservoir simplifying thetraditionally time-consuming task of hyperparameter tuning. We verify theeffectiveness of this approach on different benchmark tasks and reservoiroperating conditions. |


| Item |Content|
| --- |---|
|idx| 2401.14367v1 |
|title| Genie: Achieving Human Parity in Content-Grounded Datasets Generation |
|authors| Asaf YehudaiBoaz CarmeliYosi MassOfir ArvivNathaniel MillsAssaf ToledoEyal ShnarchLeshem Choshen
|links| http://arxiv.org/abs/2401.14367v1 |
|updated| 2024-01-25 18:14:57 UTC |
|summary| The lack of high-quality data for content-grounded generation tasks has beenidentified as a major obstacle to advancing these tasks. To address this gapwe propose Genie a novel method for automatically generating high-qualitycontent-grounded data. It consists of three stages: a Content Preparationb Generation: creating task-specific examples from the content e.g.question-answer pairs or summaries. c Filtering mechanism aiming to ensurethe quality and faithfulness of the generated data. We showcase thismethodology by generating three large-scale synthetic data making wishes forLong-Form Question-Answering LFQA summarization and information extraction.In a human evaluation our generated data was found to be natural and of highquality. Furthermore we compare models trained on our data with models trainedon human-written data -- ELI5 and ASQA for LFQA and CNN-DailyMail forSummarization. We show that our models are on par with or outperforming modelstrained on human-generated data and consistently outperforming them infaithfulness. Finally we applied our method to create LFQA data within themedical domain and compared a model trained on it with models trained on otherdomains. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2401.14405v1 |
|title| Multimodal Pathway: Improve Transformers with Irrelevant Data from Other Modalities |
|authors| Yiyuan ZhangXiaohan DingKaixiong GongYixiao GeYing ShanXiangyu Yue
|links| http://arxiv.org/abs/2401.14405v1 |
|updated| 2024-01-25 18:59:58 UTC |
|summary| We propose to improve transformers of a specific modality with irrelevantdata from other modalities e.g. improve an ImageNet model with audio or pointcloud datasets. We would like to highlight that the data samples of the targetmodality are irrelevant to the other modalities which distinguishes our methodfrom other works utilizing paired e.g. CLIP or interleaved data of differentmodalities. We propose a methodology named Multimodal Pathway - given a targetmodality and a transformer designed for it we use an auxiliary transformertrained with data of another modality and construct pathways to connectcomponents of the two models so that data of the target modality can beprocessed by both models. In this way we utilize the universalsequence-to-sequence modeling abilities of transformers obtained from twomodalities. As a concrete implementation we use a modality-specific tokenizerand task-specific head as usual but utilize the transformer blocks of theauxiliary model via a proposed method named Cross-Modal Re-parameterizationwhich exploits the auxiliary weights without any inference costs. On the imagepoint cloud video and audio recognition tasks we observe significant andconsistent performance improvements with irrelevant data from other modalities.The code and models are available at https://github.com/AILab-CVC/M2PT. |


| Item |Content|
| --- |---|
|idx| 2401.14404v1 |
|title| Deconstructing Denoising Diffusion Models for Self-Supervised Learning |
|authors| Xinlei ChenZhuang LiuSaining XieKaiming He
|links| http://arxiv.org/abs/2401.14404v1 |
|updated| 2024-01-25 18:59:57 UTC |
|summary| In this study we examine the representation learning abilities of DenoisingDiffusion Models DDM that were originally purposed for image generation. Ourphilosophy is to deconstruct a DDM gradually transforming it into a classicalDenoising Autoencoder DAE. This deconstructive procedure allows us to explorehow various components of modern DDMs influence self-supervised representationlearning. We observe that only a very few modern components are critical forlearning good representations while many others are nonessential. Our studyultimately arrives at an approach that is highly simplified and to a largeextent resembles a classical DAE. We hope our study will rekindle interest in afamily of classical methods within the realm of modern self-supervisedlearning. |


| Item |Content|
| --- |---|
|idx| 2401.14403v1 |
|title| Adaptive Mobile Manipulation for Articulated Objects In the Open World |
|authors| Haoyu XiongRussell MendoncaKenneth ShawDeepak Pathak
|links| http://arxiv.org/abs/2401.14403v1 |
|updated| 2024-01-25 18:59:44 UTC |
|summary| Deploying robots in open-ended unstructured environments such as homes hasbeen a long-standing research problem. However robots are often studied onlyin closed-off lab settings and prior mobile manipulation work is restricted topick-move-place which is arguably just the tip of the iceberg in this area. Inthis paper we introduce Open-World Mobile Manipulation System a full-stackapproach to tackle realistic articulated object operation e.g. real-worlddoors cabinets drawers and refrigerators in open-ended unstructuredenvironments. The robot utilizes an adaptive learning framework to initiallylearns from a small set of data through behavior cloning followed by learningfrom online practice on novel objects that fall outside the trainingdistribution. We also develop a low-cost mobile manipulation hardware platformcapable of safe and autonomous online adaptation in unstructured environmentswith a cost of around 20000 USD. In our experiments we utilize 20 articulateobjects across 4 buildings in the CMU campus. With less than an hour of onlinelearning for each object the system is able to increase success rate from 50of BC pre-training to 95 using online adaptation. Video results athttps://open-world-mobilemanip.github.io/ |


| Item |Content|
| --- |---|
|idx| 2401.14398v1 |
|title| pix2gestalt: Amodal Segmentation by Synthesizing Wholes |
|authors| Ege OzgurogluRuoshi LiuDídac SurísDian ChenAchal DavePavel TokmakovCarl Vondrick
|links| http://arxiv.org/abs/2401.14398v1 |
|updated| 2024-01-25 18:57:36 UTC |
|summary| We introduce pix2gestalt a framework for zero-shot amodal segmentationwhich learns to estimate the shape and appearance of whole objects that areonly partially visible behind occlusions. By capitalizing on large-scalediffusion models and transferring their representations to this task we learna conditional diffusion model for reconstructing whole objects in challengingzero-shot cases including examples that break natural and physical priorssuch as art. As training data we use a synthetically curated datasetcontaining occluded objects paired with their whole counterparts. Experimentsshow that our approach outperforms supervised baselines on establishedbenchmarks. Our model can furthermore be used to significantly improve theperformance of existing object recognition and 3D reconstruction methods in thepresence of occlusions. |


| Item |Content|
| --- |---|
|idx| 2401.14388v1 |
|title| Smooth Ranking SVM via Cutting-Plane Method |
|authors| Erhan Can OzcanBerk GörgülüMustafa G. BaydoganIoannis Ch. Paschalidis
|links| http://arxiv.org/abs/2401.14388v1 |
|updated| 2024-01-25 18:47:23 UTC |
|summary| The most popular classification algorithms are designed to maximizeclassification accuracy during training. However this strategy may fail in thepresence of class imbalance since it is possible to train models with highaccuracy by overfitting to the majority class. On the other hand the AreaUnder the Curve AUC is a widely used metric to compare classificationperformance of different algorithms when there is a class imbalance andvarious approaches focusing on the direct optimization of this metric duringtraining have been proposed. Among them SVM-based formulations are especiallypopular as this formulation allows incorporating different regularizationstrategies easily. In this work we develop a prototype learning approach thatrelies on cutting-plane method similar to Ranking SVM to maximize AUC. Ouralgorithm learns simpler models by iteratively introducing cutting planes thusoverfitting is prevented in an unconventional way. Furthermore it penalizesthe changes in the weights at each iteration to avoid large jumps that might beobserved in the test performance thus facilitating a smooth learning process.Based on the experiments conducted on 73 binary classification datasets ourmethod yields the best test AUC in 25 datasets among its relevant competitors. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2401.14405v1 |
|title| Multimodal Pathway: Improve Transformers with Irrelevant Data from Other Modalities |
|authors| Yiyuan ZhangXiaohan DingKaixiong GongYixiao GeYing ShanXiangyu Yue
|links| http://arxiv.org/abs/2401.14405v1 |
|updated| 2024-01-25 18:59:58 UTC |
|summary| We propose to improve transformers of a specific modality with irrelevantdata from other modalities e.g. improve an ImageNet model with audio or pointcloud datasets. We would like to highlight that the data samples of the targetmodality are irrelevant to the other modalities which distinguishes our methodfrom other works utilizing paired e.g. CLIP or interleaved data of differentmodalities. We propose a methodology named Multimodal Pathway - given a targetmodality and a transformer designed for it we use an auxiliary transformertrained with data of another modality and construct pathways to connectcomponents of the two models so that data of the target modality can beprocessed by both models. In this way we utilize the universalsequence-to-sequence modeling abilities of transformers obtained from twomodalities. As a concrete implementation we use a modality-specific tokenizerand task-specific head as usual but utilize the transformer blocks of theauxiliary model via a proposed method named Cross-Modal Re-parameterizationwhich exploits the auxiliary weights without any inference costs. On the imagepoint cloud video and audio recognition tasks we observe significant andconsistent performance improvements with irrelevant data from other modalities.The code and models are available at https://github.com/AILab-CVC/M2PT. |


| Item |Content|
| --- |---|
|idx| 2401.14404v1 |
|title| Deconstructing Denoising Diffusion Models for Self-Supervised Learning |
|authors| Xinlei ChenZhuang LiuSaining XieKaiming He
|links| http://arxiv.org/abs/2401.14404v1 |
|updated| 2024-01-25 18:59:57 UTC |
|summary| In this study we examine the representation learning abilities of DenoisingDiffusion Models DDM that were originally purposed for image generation. Ourphilosophy is to deconstruct a DDM gradually transforming it into a classicalDenoising Autoencoder DAE. This deconstructive procedure allows us to explorehow various components of modern DDMs influence self-supervised representationlearning. We observe that only a very few modern components are critical forlearning good representations while many others are nonessential. Our studyultimately arrives at an approach that is highly simplified and to a largeextent resembles a classical DAE. We hope our study will rekindle interest in afamily of classical methods within the realm of modern self-supervisedlearning. |


| Item |Content|
| --- |---|
|idx| 2401.14403v1 |
|title| Adaptive Mobile Manipulation for Articulated Objects In the Open World |
|authors| Haoyu XiongRussell MendoncaKenneth ShawDeepak Pathak
|links| http://arxiv.org/abs/2401.14403v1 |
|updated| 2024-01-25 18:59:44 UTC |
|summary| Deploying robots in open-ended unstructured environments such as homes hasbeen a long-standing research problem. However robots are often studied onlyin closed-off lab settings and prior mobile manipulation work is restricted topick-move-place which is arguably just the tip of the iceberg in this area. Inthis paper we introduce Open-World Mobile Manipulation System a full-stackapproach to tackle realistic articulated object operation e.g. real-worlddoors cabinets drawers and refrigerators in open-ended unstructuredenvironments. The robot utilizes an adaptive learning framework to initiallylearns from a small set of data through behavior cloning followed by learningfrom online practice on novel objects that fall outside the trainingdistribution. We also develop a low-cost mobile manipulation hardware platformcapable of safe and autonomous online adaptation in unstructured environmentswith a cost of around 20000 USD. In our experiments we utilize 20 articulateobjects across 4 buildings in the CMU campus. With less than an hour of onlinelearning for each object the system is able to increase success rate from 50of BC pre-training to 95 using online adaptation. Video results athttps://open-world-mobilemanip.github.io/ |


| Item |Content|
| --- |---|
|idx| 2401.14401v1 |
|title| Range-Agnostic Multi-View Depth Estimation With Keyframe Selection |
|authors| Andrea ContiMatteo PoggiValerio CambareriStefano Mattoccia
|links| http://arxiv.org/abs/2401.14401v1 |
|updated| 2024-01-25 18:59:42 UTC |
|summary| Methods for 3D reconstruction from posed frames require prior knowledge aboutthe scene metric range usually to recover matching cues along the epipolarlines and narrow the search range. However such prior might not be directlyavailable or estimated inaccurately in real scenarios -- e.g. outdoor 3Dreconstruction from video sequences -- therefore heavily hampering performance.In this paper we focus on multi-view depth estimation without requiring priorknowledge about the metric range of the scene by proposing RAMDepth anefficient and purely 2D framework that reverses the depth estimation andmatching steps order. Moreover we demonstrate the capability of our frameworkto provide rich insights about the quality of the views used for prediction.Additional material can be found on our project pagehttps://andreaconti.github.io/projects/range_agnostic_multi_view_depth. |


| Item |Content|
| --- |---|
|idx| 2401.14398v1 |
|title| pix2gestalt: Amodal Segmentation by Synthesizing Wholes |
|authors| Ege OzgurogluRuoshi LiuDídac SurísDian ChenAchal DavePavel TokmakovCarl Vondrick
|links| http://arxiv.org/abs/2401.14398v1 |
|updated| 2024-01-25 18:57:36 UTC |
|summary| We introduce pix2gestalt a framework for zero-shot amodal segmentationwhich learns to estimate the shape and appearance of whole objects that areonly partially visible behind occlusions. By capitalizing on large-scalediffusion models and transferring their representations to this task we learna conditional diffusion model for reconstructing whole objects in challengingzero-shot cases including examples that break natural and physical priorssuch as art. As training data we use a synthetically curated datasetcontaining occluded objects paired with their whole counterparts. Experimentsshow that our approach outperforms supervised baselines on establishedbenchmarks. Our model can furthermore be used to significantly improve theperformance of existing object recognition and 3D reconstruction methods in thepresence of occlusions. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2401.14343v1 |
|title| Class-attribute Priors: Adapting Optimization to Heterogeneity and Fairness Objective |
|authors| Xuechen ZhangMingchen LiJiasi ChenChristos ThrampoulidisSamet Oymak
|links| http://arxiv.org/abs/2401.14343v1 |
|updated| 2024-01-25 17:43:39 UTC |
|summary| Modern classification problems exhibit heterogeneities across individualclasses: Each class may have unique attributes such as sample size labelquality or predictability easy vs difficult and variable importance attest-time. Without care these heterogeneities impede the learning processmost notably when optimizing fairness objectives. Confirming this under agaussian mixture setting we show that the optimal SVM classifier for balancedaccuracy needs to be adaptive to the class attributes. This motivates us topropose CAP: An effective and general method that generates a class-specificlearning strategy e.g. hyperparameter based on the attributes of that class.This way optimization process better adapts to heterogeneities. CAP leads tosubstantial improvements over the naive approach of assigning separatehyperparameters to each class. We instantiate CAP for loss function design andpost-hoc logit adjustment with emphasis on label-imbalanced problems. We showthat CAP is competitive with prior art and its flexibility unlocks clearbenefits for fairness objectives beyond balanced accuracy. Finally we evaluateCAP on problems with label noise as well as weighted test objectives toshowcase how CAP can jointly adapt to different heterogeneities. |


| Item |Content|
| --- |---|
|idx| 2401.14340v1 |
|title| Estimation of partially known Gaussian graphical models with score-based structural priors |
|authors| Martín SevillaAntonio García MarquesSantiago Segarra
|links| http://arxiv.org/abs/2401.14340v1 |
|updated| 2024-01-25 17:39:47 UTC |
|summary| We propose a novel algorithm for the support estimation of partially knownGaussian graphical models that incorporates prior information about theunderlying graph. In contrast to classical approaches that provide a pointestimate based on a maximum likelihood or a maximum a posteriori criterionusing simple priors on the precision matrix we consider a prior on the graphand rely on annealed Langevin diffusion to generate samples from the posteriordistribution. Since the Langevin sampler requires access to the score functionof the underlying graph prior we use graph neural networks to effectivelyestimate the score from a graph dataset either available beforehand orgenerated from a known distribution. Numerical experiments demonstrate thebenefits of our approach. |


| Item |Content|
| --- |---|
|idx| 2401.14283v1 |
|title| Information Leakage Detection through Approximate Bayes-optimal Prediction |
|authors| Pritha GuptaMarcel WeverEyke Hüllermeier
|links| http://arxiv.org/abs/2401.14283v1 |
|updated| 2024-01-25 16:15:27 UTC |
|summary| In todays data-driven world the proliferation of publicly availableinformation intensifies the challenge of information leakage IL raisingsecurity concerns. IL involves unintentionally exposing secret sensitiveinformation to unauthorized parties via systems observable information.Conventional statistical approaches which estimate mutual information MIbetween observable and secret information for detecting IL face challengessuch as the curse of dimensionality convergence computational complexity andMI misestimation. Furthermore emerging supervised machine learning MLmethods though effective are limited to binary system-sensitive informationand lack a comprehensive theoretical framework. To address these limitationswe establish a theoretical framework using statistical learning theory andinformation theory to accurately quantify and detect IL. We demonstrate that MIcan be accurately estimated by approximating the log-loss and accuracy of theBayes predictor. As the Bayes predictor is typically unknown in practice wepropose to approximate it with the help of automated machine learning AutoML.First we compare our MI estimation approaches against current baselines usingsynthetic data sets generated using the multivariate normal MVN distributionwith known MI. Second we introduce a cut-off technique using one-sidedstatistical tests to detect IL employing the Holm-Bonferroni correction toincrease confidence in detection decisions. Our study evaluates IL detectionperformance on real-world data sets highlighting the effectiveness of theBayes predictors log-loss estimation and finds our proposed method toeffectively estimate MI on synthetic data sets and thus detect ILs accurately. |


| Item |Content|
| --- |---|
|idx| 2401.14210v1 |
|title| At the junction between deep learning and statistics of extremes: formalizing the landslide hazard definition |
|authors| Ashok DahalRaphaël HuserLuigi Lombardo
|links| http://arxiv.org/abs/2401.14210v1 |
|updated| 2024-01-25 14:48:08 UTC |
|summary| The most adopted definition of landslide hazard combines spatial informationabout landslide location susceptibility threat intensity and frequencyreturn period. Only the first two elements are usually considered andestimated when working over vast areas. Even then separate models constitutethe standard with frequency being rarely investigated. Frequency and intensityare intertwined and depend on each other because larger events occur lessfrequently and vice versa. However due to the lack of multi-temporalinventories and joint statistical models modelling such properties via aunified hazard model has always been challenging and has yet to be attempted.Here we develop a unified model to estimate landslide hazard at the slope unitlevel to address such gaps. We employed deep learning combined with a modelmotivated by extreme-value theory to analyse an inventory of 30 years ofobserved rainfall-triggered landslides in Nepal and assess landslide hazard formultiple return periods. We also use our model to further explore landslidehazard for the same return periods under different climate change scenarios upto the end of the century. Our results show that the proposed model performsexcellently and can be used to model landslide hazard in a unified manner.Geomorphologically we find that under both climate change scenarios SSP245and SSP885 landslide hazard is likely to increase up to two times on averagein the lower Himalayan regions while remaining the same in the middle Himalayanregion whilst decreasing slightly in the upper Himalayan region areas. |


| Item |Content|
| --- |---|
|idx| 2401.14161v1 |
|title| Adapting tree-based multiple imputation methods for multi-level data? A simulation study |
|authors| Ketevan GurtskaiaJakob SchwerterPhilipp Doebler
|links| http://arxiv.org/abs/2401.14161v1 |
|updated| 2024-01-25 13:12:50 UTC |
|summary| This simulation study evaluates the effectiveness of multiple imputation MItechniques for multilevel data. It compares the performance of traditionalMultiple Imputation by Chained Equations MICE with tree-based methods such asChained Random Forests with Predictive Mean Matching and Extreme GradientBoosting. Adapted versions that include dummy variables for cluster membershipare also included for the tree-based methods. Methods are evaluated forcoefficient estimation bias statistical power and type I error rates onsimulated hierarchical data with different cluster sizes 25 and 50 and levelsof missingness 10 and 50. Coefficients are estimated using randomintercept and random slope models. The results show that while MICE ispreferred for accurate rejection rates Extreme Gradient Boosting isadvantageous for reducing bias. Furthermore the study finds that bias levelsare similar across different cluster sizes but rejection rates tend to be lessfavorable with fewer clusters lower power higher type I error. In additionthe inclusion of cluster dummies in tree-based methods improves estimation forLevel 1 variables but is less effective for Level 2 variables. When databecome too complex and MICE is too slow extreme gradient boosting is a goodalternative for hierarchical data.  Keywords: Multiple imputation multi-level data MICE missRanger mixgb |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2401.14362v1 |
|title| The Typing Cure: Experiences with Large Language Model Chatbots for Mental Health Support |
|authors| Inhwa SongSachin R. PendseNeha KumarMunmun De Choudhury
|links| http://arxiv.org/abs/2401.14362v1 |
|updated| 2024-01-25 18:08:53 UTC |
|summary| People experiencing severe distress increasingly use Large Language ModelLLM chatbots as mental health support tools. Discussions on social media havedescribed how engagements were lifesaving for some but evidence suggests thatgeneral-purpose LLM chatbots also have notable risks that could endanger thewelfare of users if not designed responsibly. In this study we investigate thelived experiences of people who have used LLM chatbots for mental healthsupport. We build on interviews with 21 individuals from globally diversebackgrounds to analyze how users create unique support roles for theirchatbots fill in gaps in everyday care and navigate associated culturallimitations when seeking support from chatbots. We ground our analysis inpsychotherapy literature around effective support and introduce the concept oftherapeutic alignment or aligning AI with therapeutic values for mental healthcontexts. Our study offers recommendations for how designers can approach theethical and effective use of LLM chatbots and other AI mental health supporttools in mental health care. |


| Item |Content|
| --- |---|
|idx| 2401.14268v1 |
|title| GPTVoiceTasker: LLM-Powered Virtual Assistant for Smartphone |
|authors| Minh Duc VuHan WangZhuang LiJieshan ChenShengdong ZhaoZhenchang XingChunyang Chen
|links| http://arxiv.org/abs/2401.14268v1 |
|updated| 2024-01-25 16:02:56 UTC |
|summary| Virtual assistants have the potential to play an important role in helpingusers achieves different tasks. However these systems face challenges in theirreal-world usability characterized by inefficiency and struggles in graspinguser intentions. Leveraging recent advances in Large Language Models LLMs weintroduce GptVoiceTasker a virtual assistant poised to enhance userexperiences and task efficiency on mobile devices. GptVoiceTasker excels atintelligently deciphering user commands and executing relevant deviceinteractions to streamline task completion. The system continually learns fromhistorical user commands to automate subsequent usages further enhancingexecution efficiency. Our experiments affirm GptVoiceTaskers exceptionalcommand interpretation abilities and the precision of its task automationmodule. In our user study GptVoiceTasker boosted task efficiency in real-worldscenarios by 34.85 accompanied by positive participant feedback. We madeGptVoiceTasker open-source inviting further research into LLMs utilization fordiverse tasks through prompt engineering and leveraging user usage data toimprove efficiency. |


| Item |Content|
| --- |---|
|idx| 2401.14095v1 |
|title| Evaluating User Experience and Data Quality in a Gamified Data Collection for Appearance-Based Gaze Estimation |
|authors| Mingtao YueTomomi SayudaMiles PenningtonYusuke Sugano
|links| http://arxiv.org/abs/2401.14095v1 |
|updated| 2024-01-25 11:16:14 UTC |
|summary| Appearance-based gaze estimation which uses only a regular camera toestimate human gaze is important in various application fields. While thetechnique faces data bias issues data collection protocol is often demandingand collecting data from a wide range of participants is difficult. It is animportant challenge to design opportunities that allow a diverse range ofpeople to participate while ensuring the quality of the training data. Totackle this challenge we introduce a novel gamified approach for collectingtraining data. In this game two players communicate words via eye gaze througha transparent letter board. Images captured during gameplay serve as valuabletraining data for gaze estimation models. The game is designed as a physicalinstallation that involves communication between players and it is expected toattract the interest of diverse participants. We assess the games significanceon data quality and user experience through a comparative user study. |


| Item |Content|
| --- |---|
|idx| 2401.14078v1 |
|title| The Adaptive Architectural Layout: How the Control of a Semi-Autonomous Mobile Robotic Partition was Shared to Mediate the Environmental Demands and Resources of an Open-Plan Office |
|authors| Binh Vinh Duc NguyenAndrew Vande Moere
|links| http://dx.doi.org/10.1145/3613904.3642465 |
|updated| 2024-01-25 10:55:39 UTC |
|summary| A typical open-plan office layout is unable to optimally host multiplecollocated work activities personal needs and situational events as itsspace exerts a range of environmental demands on workers in terms ofmaintaining their acoustic visual or privacy comfort. As we hypothesise thatthese demands could be coped by optimising the environmental resources of thearchitectural layout we deployed a mobile robotic partition that autonomouslymanoeuvres between predetermined locations. During a five-weeks in-the-wildstudy within a real-world open-plan office we studied how 13 workers adoptedfour distinct adaptation strategies when sharing the spatiotemporal control ofthe robotic partition. Based on their logged and self-reported reasoning wepresent six initiation regulating factors that determine the appropriateness ofeach adaptation strategy. This study thus contributes to how futurehuman-building interaction could autonomously improve the experience comfortperformance and even the health and wellbeing of multiple workers that sharethe same workplace. |


| Item |Content|
| --- |---|
|idx| 2401.14010v2 |
|title| Leveraging Large Models for Crafting Narrative Visualization: A Survey |
|authors| Yi HeShixiong CaoYang ShiQing ChenKe XuNan Cao
|links| http://arxiv.org/abs/2401.14010v2 |
|updated| 2024-01-26 04:09:21 UTC |
|summary| Narrative visualization effectively transforms data into engaging storiesmaking complex information accessible to a broad audience. Large modelsessential for narrative visualization inherently facilitate this processthrough their superior ability to handle natural language queries and answersgenerate cohesive narratives and enhance visual communication. Inspired byprevious work in narrative visualization and recent advances in large modelswe synthesized potential tasks and opportunities for large models at variousstages of narrative visualization. In our study we surveyed 79 papers toexplore the role of large models in automating narrative visualizationcreation. We propose a comprehensive pipeline that leverages large models forcrafting narrative visualization categorizing the reviewed literature intofour essential phases: Data Narration Visualization and Presentation.Additionally we identify nine specific tasks where large models are appliedacross these stages. This study maps out the landscape of challenges andopportunities in the LM4NV process providing insightful directions for futureresearch and valuable guidance for scholars in the field. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2401.13947v1 |
|title| Networked Multiagent Reinforcement Learning for Peer-to-Peer Energy Trading |
|authors| Chen FengAndrew L. Liu
|links| http://arxiv.org/abs/2401.13947v1 |
|updated| 2024-01-25 05:05:55 UTC |
|summary| Utilizing distributed renewable and energy storage resources in localdistribution networks via peer-to-peer P2P energy trading has long beentouted as a solution to improve energy systems resilience and sustainability.Consumers and prosumers those who have energy generation resources howeverdo not have the expertise to engage in repeated P2P trading and thezero-marginal costs of renewables present challenges in determining fair marketprices. To address these issues we propose multi-agent reinforcement learningMARL frameworks to help automate consumers bidding and management of theirsolar PV and energy storage resources under a specific P2P clearing mechanismthat utilizes the so-called supply-demand ratio. In addition we show how theMARL frameworks can integrate physical network constraints to realize voltagecontrol hence ensuring physical feasibility of the P2P energy trading andpaving way for real-world implementations. |


| Item |Content|
| --- |---|
|idx| 2401.13945v1 |
|title| General Automatic Solution Generation of Social Problems |
|authors| Tong NiuHaoyu HuangYu DuWeihao ZhangLuping ShiRong Zhao
|links| http://dx.doi.org/10.1007/s11633-024-1396-5 |
|updated| 2024-01-25 05:00:46 UTC |
|summary| Given the escalating intricacy and multifaceted nature of contemporary socialsystems manually generating solutions to address pertinent social issues hasbecome a formidable task. In response to this challenge the rapid developmentof artificial intelligence has spurred the exploration of computationalmethodologies aimed at automatically generating solutions. However currentmethods for auto-generation of solutions mainly concentrate on local socialregulations that pertain to specific scenarios. Here we report an automaticsocial operating system ASOS designed for general social solution generationwhich is built upon agent-based models enabling both global and local analysesand regulations of social problems across spatial and temporal dimensions. ASOSadopts a hypergraph with extensible social semantics for a comprehensive andstructured representation of social dynamics. It also incorporates ageneralized protocol for standardized hypergraph operations and a symbolichybrid framework that delivers interpretable solutions yielding a balancebetween regulatory efficacy and function viability. To demonstrate theeffectiveness of ASOS we apply it to the domain of averting extreme eventswithin international oil futures markets. By generating a new trading rolesupplemented by new mechanisms ASOS can adeptly discern precarious marketconditions and make front-running interventions for non-profit purposes. Thisstudy demonstrates that ASOS provides an efficient and systematic approach forgenerating solutions for enhancing our society. |


| Item |Content|
| --- |---|
|idx| 2401.13604v1 |
|title| Stream-based perception for cognitive agents in mobile ecosystems |
|authors| Jeremias DötterlRalf BrunsJürgen DunkelSascha Ossowski
|links| http://dx.doi.org/10.3233/AIC-190614 |
|updated| 2024-01-24 17:14:50 UTC |
|summary| Cognitive agent abstractions can help to engineer intelligent systems acrossmobile devices. On smartphones the data obtained from onboard sensors can givevaluable insights into the users current situation. Unfortunately todayscognitive agent frameworks cannot cope well with the challengingcharacteristics of sensor data. Sensor data is located on a low abstractionlevel and the individual data elements are not meaningful when observed inisolation. In contrast cognitive agents operate on high-level percepts andlack the means to effectively detect complex spatio-temporal patterns insequences of multiple percepts. In this paper we present a stream-basedperception approach that enables the agents to perceive meaningful situationsin low-level sensor data streams. We present a crowdshipping case study whereautonomous self-interested agents collaborate to deliver parcels to theirdestinations. We show how situations derived from smartphone sensor data cantrigger and guide auctions which the agents use to reach agreements.Experiments with real smartphone data demonstrate the benefits of stream-basedagent perception. |


| Item |Content|
| --- |---|
|idx| 2401.13460v1 |
|title| Multi-Agent Diagnostics for Robustness via Illuminated Diversity |
|authors| Mikayel SamvelyanDavide PaglieriMinqi JiangJack Parker-HolderTim Rocktäschel
|links| http://arxiv.org/abs/2401.13460v1 |
|updated| 2024-01-24 14:02:09 UTC |
|summary| In the rapidly advancing field of multi-agent systems ensuring robustness inunfamiliar and adversarial settings is crucial. Notwithstanding theiroutstanding performance in familiar environments these systems often falter innew situations due to overfitting during the training phase. This is especiallypronounced in settings where both cooperative and competitive behaviours arepresent encapsulating a dual nature of overfitting and generalisationchallenges. To address this issue we present Multi-Agent Diagnostics forRobustness via Illuminated Diversity MADRID a novel approach for generatingdiverse adversarial scenarios that expose strategic vulnerabilities inpre-trained multi-agent policies. Leveraging the concepts from open-endedlearning MADRID navigates the vast space of adversarial settings employing atarget policys regret to gauge the vulnerabilities of these settings. Weevaluate the effectiveness of MADRID on the 11vs11 version of Google ResearchFootball one of the most complex environments for multi-agent reinforcementlearning. Specifically we employ MADRID for generating a diverse array ofadversarial settings for TiZero the state-of-the-art approach which mastersthe game through 45 days of training on a large-scale distributedinfrastructure. We expose key shortcomings in TiZeros tacticaldecision-making underlining the crucial importance of rigorous evaluation inmulti-agent systems. |


| Item |Content|
| --- |---|
|idx| 2401.13127v1 |
|title| Generalization of Heterogeneous Multi-Robot Policies via Awareness and Communication of Capabilities |
|authors| Pierce HowellMax RudolphReza TorbatiKevin FuHarish Ravichandar
|links| http://arxiv.org/abs/2401.13127v1 |
|updated| 2024-01-23 22:31:34 UTC |
|summary| Recent advances in multi-agent reinforcement learning MARL are enablingimpressive coordination in heterogeneous multi-robot teams. However existingapproaches often overlook the challenge of generalizing learned policies toteams of new compositions sizes and robots. While such generalization mightnot be important in teams of virtual agents that can retrain policieson-demand it is pivotal in multi-robot systems that are deployed in thereal-world and must readily adapt to inevitable changes. As such multi-robotpolicies must remain robust to team changes -- an ability we call adaptiveteaming. In this work we investigate if awareness and communication of robotcapabilities can provide such generalization by conducting detailed experimentsinvolving an established multi-robot test bed. We demonstrate that shareddecentralized policies that enable robots to be both aware of and communicatetheir capabilities can achieve adaptive teaming by implicitly capturing thefundamental relationship between collective capabilities and effectivecoordination. Videos of trained policies can be viewed at:https://sites.google.com/view/cap-comm |


