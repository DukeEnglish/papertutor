# cs.CL 

| Item |Content|
| --- |---|
|idx| 2401.10882v1 |
|title| Reinforcement learning for question answering in programming domain using public community scoring as a human feedback |
|authors| Alexey GorbatovskiSergey Kovalchuk
|links| http://arxiv.org/abs/2401.10882v1 |
|updated| 2024-01-19 18:49:36 UTC |
|summary| In this study we investigate the enhancement of the GPT Neo 125M performancein Community Question Answering CQA with a focus on programming through theintegration of Reinforcement Learning from Human Feedback RLHF and theutilization of scores from Stack Overflow. Two distinct reward model trainingstrategies are employed for fine-tuning with Proximal Policy OptimizationPPO. Notably the improvements in performance achieved through this methodare comparable to those of GPT Neo 2.7B parameter variant. Additionally anauxiliary scoring mechanism is introduced which demonstrates the limitationsof conventional linguistic metrics in evaluating responses in the programmingdomain. Through accurate analysis this paper looks at the divergence betweentraditional linguistic metrics and our human-preferences-based reward modelunderscoring the imperative for domain-specific evaluation methods. Byelucidating the complexities involved in applying RLHF to programming CQA andaccentuating the significance of context-aware evaluation this studycontributes to the ongoing efforts in refining Large Language Models throughfocused human feedback. |


| Item |Content|
| --- |---|
|idx| 2401.10862v1 |
|title| Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning |
|authors| Adib HasanIleana RuginaAlex Wang
|links| http://arxiv.org/abs/2401.10862v1 |
|updated| 2024-01-19 18:05:34 UTC |
|summary| Large Language Models LLMs are vulnerable to Jailbreaking prompts a typeof attack that can coax these models into generating harmful and illegalcontent. In this paper we show that pruning up to 20 of LLM parametersmarkedly increases their resistance to such attacks without additional trainingand without sacrificing their performance in standard benchmarks. Intriguinglywe discovered that the enhanced safety observed post-pruning correlates to theinitial safety training level of the model hinting that the effect of pruningcould be more general and may hold for other LLM behaviors beyond safety.Additionally we introduce a curated dataset of 225 harmful tasks across fivecategories inserted into ten different Jailbreaking prompts showing thatpruning aids LLMs in concentrating attention on task-relevant tokens injailbreaking prompts. Lastly our experiments reveal that the prominent chatmodels such as LLaMA-2 Chat Vicuna and Mistral Instruct exhibit highsusceptibility to jailbreaking attacks with some categories achieving nearly70-100 success rate. These insights underline the potential of pruning as ageneralizable approach for improving LLM safety reliability and potentiallyother desired behaviors. |


| Item |Content|
| --- |---|
|idx| 2401.10850v1 |
|title| Advancements in eHealth Data Analytics through Natural Language Processing and Deep Learning |
|authors| Elena-Simona ApostolCiprian-Octavian Truică
|links| http://arxiv.org/abs/2401.10850v1 |
|updated| 2024-01-19 17:51:11 UTC |
|summary| The healthcare environment is commonly referred to as information-rich butalso knowledge poor. Healthcare systems collect huge amounts of data fromvarious sources: lab reports medical letters logs of medical tools orprograms medical prescriptions etc. These massive sets of data can providegreat knowledge and information that can improve the medical services andoverall the healthcare domain such as disease prediction by analyzing thepatients symptoms or disease prevention by facilitating the discovery ofbehavioral factors for diseases. Unfortunately only a relatively small volumeof the textual eHealth data is processed and interpreted an important factorbeing the difficulty in efficiently performing Big Data operations. In themedical field detecting domain-specific multi-word terms is a crucial task asthey can define an entire concept with a few words. A term can be defined as alinguistic structure or a concept and it is composed of one or more words witha specific meaning to a domain. All the terms of a domain create itsterminology. This chapter offers a critical study of the current mostperformant solutions for analyzing unstructured image and textual eHealthdata. This study also provides a comparison of the current Natural LanguageProcessing and Deep Learning techniques in the eHealth context. Finally weexamine and discuss some of the current issues and we define a set of researchdirections in this area. |


| Item |Content|
| --- |---|
|idx| 2401.10841v1 |
|title| Using LLMs to discover emerging coded antisemitic hate-speech emergence in extremist social media |
|authors| Dhanush KikkisettiRaza Ul MustafaWendy MelilloRoberto CorizzoZois BoukouvalasJeff GillNathalie Japkowicz
|links| http://arxiv.org/abs/2401.10841v1 |
|updated| 2024-01-19 17:40:50 UTC |
|summary| Online hate speech proliferation has created a difficult problem for socialmedia platforms. A particular challenge relates to the use of coded language bygroups interested in both creating a sense of belonging for its users andevading detection. Coded language evolves quickly and its use varies over time.This paper proposes a methodology for detecting emerging coded hate-ladenterminology. The methodology is tested in the context of online antisemiticdiscourse. The approach considers posts scraped from social media platformsoften used by extremist users. The posts are scraped using seed expressionsrelated to previously known discourse of hatred towards Jews. The method beginsby identifying the expressions most representative of each post and calculatingtheir frequency in the whole corpus. It filters out grammatically incoherentexpressions as well as previously encountered ones so as to focus on emergentwell-formed terminology. This is followed by an assessment of semanticsimilarity to known antisemitic terminology using a fine-tuned large languagemodel and subsequent filtering out of the expressions that are too distantfrom known expressions of hatred. Emergent antisemitic expressions containingterms clearly relating to Jewish topics are then removed to return only codedexpressions of hatred. |


| Item |Content|
| --- |---|
|idx| 2401.10825v1 |
|title| A survey on recent advances in named entity recognition |
|authors| Imed KeraghelStanislas MorbieuMohamed Nadif
|links| http://arxiv.org/abs/2401.10825v1 |
|updated| 2024-01-19 17:21:05 UTC |
|summary| Named Entity Recognition seeks to extract substrings within a text that namereal-world objects and to determine their type for example whether they referto persons or organizations. In this survey we first present an overview ofrecent popular approaches but we also look at graph- and transformer- basedmethods including Large Language Models LLMs that have not had much coveragein other surveys. Second we focus on methods designed for datasets with scarceannotations. Third we evaluate the performance of the main NER implementationson a variety of datasets with differing characteristics as regards theirdomain their size and their number of classes. We thus provide a deepcomparison of algorithms that are never considered together. Our experimentsshed some light on how the characteristics of datasets affect the behavior ofthe methods that we compare. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2401.10889v1 |
|title| Synthesizing Moving People with 3D Control |
|authors| Boyi LiJathushan RajasegaranYossi GandelsmanAlexei A. EfrosJitendra Malik
|links| http://arxiv.org/abs/2401.10889v1 |
|updated| 2024-01-19 18:59:11 UTC |
|summary| In this paper we present a diffusion model-based framework for animatingpeople from a single image for a given target 3D motion sequence. Our approachhas two core components: a learning priors about invisible parts of the humanbody and clothing and b rendering novel body poses with proper clothing andtexture. For the first part we learn an in-filling diffusion model tohallucinate unseen parts of a person given a single image. We train this modelon texture map space which makes it more sample-efficient since it isinvariant to pose and viewpoint. Second we develop a diffusion-based renderingpipeline which is controlled by 3D human poses. This produces realisticrenderings of novel poses of the person including clothing hair andplausible in-filling of unseen regions. This disentangled approach allows ourmethod to generate a sequence of images that are faithful to the target motionin the 3D pose and to the input image in terms of visual similarity. Inaddition to that the 3D control allows various synthetic camera trajectoriesto render a person. Our experiments show that our method is resilient ingenerating prolonged motions and varied challenging and complex poses comparedto prior methods. Please check our website for more details:https://boyiliee.github.io/3DHM.github.io/. |


| Item |Content|
| --- |---|
|idx| 2401.10886v1 |
|title| SCENES: Subpixel Correspondence Estimation With Epipolar Supervision |
|authors| Dominik A. KloepferJoão F. HenriquesDylan Campbell
|links| http://arxiv.org/abs/2401.10886v1 |
|updated| 2024-01-19 18:57:46 UTC |
|summary| Extracting point correspondences from two or more views of a scene is afundamental computer vision problem with particular importance for relativecamera pose estimation and structure-from-motion. Existing local featurematching approaches trained with correspondence supervision on large-scaledatasets obtain highly-accurate matches on the test sets. However they do notgeneralise well to new datasets with different characteristics to those theywere trained on unlike classic feature extractors. Instead they requirefinetuning which assumes that ground-truth correspondences or ground-truthcamera poses and 3D structure are available. We relax this assumption byremoving the requirement of 3D structure e.g. depth maps or point clouds andonly require camera pose information which can be obtained from odometry. Wedo so by replacing correspondence losses with epipolar losses which encourageputative matches to lie on the associated epipolar line. While weaker thancorrespondence supervision we observe that this cue is sufficient forfinetuning existing models on new data. We then further relax the assumption ofknown camera poses by using pose estimates in a novel bootstrapping approach.We evaluate on highly challenging datasets including an indoor drone datasetand an outdoor smartphone camera dataset and obtain state-of-the-art resultswithout strong supervision. |


| Item |Content|
| --- |---|
|idx| 2401.10882v1 |
|title| Reinforcement learning for question answering in programming domain using public community scoring as a human feedback |
|authors| Alexey GorbatovskiSergey Kovalchuk
|links| http://arxiv.org/abs/2401.10882v1 |
|updated| 2024-01-19 18:49:36 UTC |
|summary| In this study we investigate the enhancement of the GPT Neo 125M performancein Community Question Answering CQA with a focus on programming through theintegration of Reinforcement Learning from Human Feedback RLHF and theutilization of scores from Stack Overflow. Two distinct reward model trainingstrategies are employed for fine-tuning with Proximal Policy OptimizationPPO. Notably the improvements in performance achieved through this methodare comparable to those of GPT Neo 2.7B parameter variant. Additionally anauxiliary scoring mechanism is introduced which demonstrates the limitationsof conventional linguistic metrics in evaluating responses in the programmingdomain. Through accurate analysis this paper looks at the divergence betweentraditional linguistic metrics and our human-preferences-based reward modelunderscoring the imperative for domain-specific evaluation methods. Byelucidating the complexities involved in applying RLHF to programming CQA andaccentuating the significance of context-aware evaluation this studycontributes to the ongoing efforts in refining Large Language Models throughfocused human feedback. |


| Item |Content|
| --- |---|
|idx| 2401.10862v1 |
|title| Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning |
|authors| Adib HasanIleana RuginaAlex Wang
|links| http://arxiv.org/abs/2401.10862v1 |
|updated| 2024-01-19 18:05:34 UTC |
|summary| Large Language Models LLMs are vulnerable to Jailbreaking prompts a typeof attack that can coax these models into generating harmful and illegalcontent. In this paper we show that pruning up to 20 of LLM parametersmarkedly increases their resistance to such attacks without additional trainingand without sacrificing their performance in standard benchmarks. Intriguinglywe discovered that the enhanced safety observed post-pruning correlates to theinitial safety training level of the model hinting that the effect of pruningcould be more general and may hold for other LLM behaviors beyond safety.Additionally we introduce a curated dataset of 225 harmful tasks across fivecategories inserted into ten different Jailbreaking prompts showing thatpruning aids LLMs in concentrating attention on task-relevant tokens injailbreaking prompts. Lastly our experiments reveal that the prominent chatmodels such as LLaMA-2 Chat Vicuna and Mistral Instruct exhibit highsusceptibility to jailbreaking attacks with some categories achieving nearly70-100 success rate. These insights underline the potential of pruning as ageneralizable approach for improving LLM safety reliability and potentiallyother desired behaviors. |


| Item |Content|
| --- |---|
|idx| 2401.10850v1 |
|title| Advancements in eHealth Data Analytics through Natural Language Processing and Deep Learning |
|authors| Elena-Simona ApostolCiprian-Octavian Truică
|links| http://arxiv.org/abs/2401.10850v1 |
|updated| 2024-01-19 17:51:11 UTC |
|summary| The healthcare environment is commonly referred to as information-rich butalso knowledge poor. Healthcare systems collect huge amounts of data fromvarious sources: lab reports medical letters logs of medical tools orprograms medical prescriptions etc. These massive sets of data can providegreat knowledge and information that can improve the medical services andoverall the healthcare domain such as disease prediction by analyzing thepatients symptoms or disease prevention by facilitating the discovery ofbehavioral factors for diseases. Unfortunately only a relatively small volumeof the textual eHealth data is processed and interpreted an important factorbeing the difficulty in efficiently performing Big Data operations. In themedical field detecting domain-specific multi-word terms is a crucial task asthey can define an entire concept with a few words. A term can be defined as alinguistic structure or a concept and it is composed of one or more words witha specific meaning to a domain. All the terms of a domain create itsterminology. This chapter offers a critical study of the current mostperformant solutions for analyzing unstructured image and textual eHealthdata. This study also provides a comparison of the current Natural LanguageProcessing and Deep Learning techniques in the eHealth context. Finally weexamine and discuss some of the current issues and we define a set of researchdirections in this area. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2401.10886v1 |
|title| SCENES: Subpixel Correspondence Estimation With Epipolar Supervision |
|authors| Dominik A. KloepferJoão F. HenriquesDylan Campbell
|links| http://arxiv.org/abs/2401.10886v1 |
|updated| 2024-01-19 18:57:46 UTC |
|summary| Extracting point correspondences from two or more views of a scene is afundamental computer vision problem with particular importance for relativecamera pose estimation and structure-from-motion. Existing local featurematching approaches trained with correspondence supervision on large-scaledatasets obtain highly-accurate matches on the test sets. However they do notgeneralise well to new datasets with different characteristics to those theywere trained on unlike classic feature extractors. Instead they requirefinetuning which assumes that ground-truth correspondences or ground-truthcamera poses and 3D structure are available. We relax this assumption byremoving the requirement of 3D structure e.g. depth maps or point clouds andonly require camera pose information which can be obtained from odometry. Wedo so by replacing correspondence losses with epipolar losses which encourageputative matches to lie on the associated epipolar line. While weaker thancorrespondence supervision we observe that this cue is sufficient forfinetuning existing models on new data. We then further relax the assumption ofknown camera poses by using pose estimates in a novel bootstrapping approach.We evaluate on highly challenging datasets including an indoor drone datasetand an outdoor smartphone camera dataset and obtain state-of-the-art resultswithout strong supervision. |


| Item |Content|
| --- |---|
|idx| 2401.10874v1 |
|title| Applications of flow models to the generation of correlated lattice QCD ensembles |
|authors| Ryan AbbottAleksandar BotevDenis BoydaDaniel C. HackettGurtej KanwarSébastien RacanièreDanilo J. RezendeFernando Romero-LópezPhiala E. ShanahanJulian M. Urban
|links| http://arxiv.org/abs/2401.10874v1 |
|updated| 2024-01-19 18:33:52 UTC |
|summary| Machine-learned normalizing flows can be used in the context of latticequantum field theory to generate statistically correlated ensembles of latticegauge fields at different action parameters. This work demonstrates how thesecorrelations can be exploited for variance reduction in the computation ofobservables. Three different proof-of-concept applications are demonstratedusing a novel residual flow architecture: continuum limits of gauge theoriesthe mass dependence of QCD observables and hadronic matrix elements based onthe Feynman-Hellmann approach. In all three cases it is shown that statisticaluncertainties are significantly reduced when machine-learned flows areincorporated as compared with the same calculations performed with uncorrelatedensembles or direct reweighting. |


| Item |Content|
| --- |---|
|idx| 2401.10862v1 |
|title| Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning |
|authors| Adib HasanIleana RuginaAlex Wang
|links| http://arxiv.org/abs/2401.10862v1 |
|updated| 2024-01-19 18:05:34 UTC |
|summary| Large Language Models LLMs are vulnerable to Jailbreaking prompts a typeof attack that can coax these models into generating harmful and illegalcontent. In this paper we show that pruning up to 20 of LLM parametersmarkedly increases their resistance to such attacks without additional trainingand without sacrificing their performance in standard benchmarks. Intriguinglywe discovered that the enhanced safety observed post-pruning correlates to theinitial safety training level of the model hinting that the effect of pruningcould be more general and may hold for other LLM behaviors beyond safety.Additionally we introduce a curated dataset of 225 harmful tasks across fivecategories inserted into ten different Jailbreaking prompts showing thatpruning aids LLMs in concentrating attention on task-relevant tokens injailbreaking prompts. Lastly our experiments reveal that the prominent chatmodels such as LLaMA-2 Chat Vicuna and Mistral Instruct exhibit highsusceptibility to jailbreaking attacks with some categories achieving nearly70-100 success rate. These insights underline the potential of pruning as ageneralizable approach for improving LLM safety reliability and potentiallyother desired behaviors. |


| Item |Content|
| --- |---|
|idx| 2401.10859v1 |
|title| Ensembler: Combating model inversion attacks using model ensemble during collaborative inference |
|authors| Dancheng LiuJinjun Xiong
|links| http://arxiv.org/abs/2401.10859v1 |
|updated| 2024-01-19 18:03:21 UTC |
|summary| Deep learning models have exhibited remarkable performance across variousdomains. Nevertheless the burgeoning model sizes compel edge devices tooffload a significant portion of the inference process to the cloud. While thispractice offers numerous advantages it also raises critical concerns regardinguser data privacy. In scenarios where the cloud servers trustworthiness is inquestion the need for a practical and adaptable method to safeguard dataprivacy becomes imperative. In this paper we introduce Ensembler anextensible framework designed to substantially increase the difficulty ofconducting model inversion attacks for adversarial parties. Ensembler leveragesmodel ensembling on the adversarial server running in parallel with existingapproaches that introduce perturbations to sensitive data during colloborativeinference. Our experiments demonstrate that when combined with even basicGaussian noise Ensembler can effectively shield images from reconstructionattacks achieving recognition levels that fall below human performance in somestrict settings significantly outperforming baseline methods lacking theEnsembler framework. |


| Item |Content|
| --- |---|
|idx| 2401.10841v1 |
|title| Using LLMs to discover emerging coded antisemitic hate-speech emergence in extremist social media |
|authors| Dhanush KikkisettiRaza Ul MustafaWendy MelilloRoberto CorizzoZois BoukouvalasJeff GillNathalie Japkowicz
|links| http://arxiv.org/abs/2401.10841v1 |
|updated| 2024-01-19 17:40:50 UTC |
|summary| Online hate speech proliferation has created a difficult problem for socialmedia platforms. A particular challenge relates to the use of coded language bygroups interested in both creating a sense of belonging for its users andevading detection. Coded language evolves quickly and its use varies over time.This paper proposes a methodology for detecting emerging coded hate-ladenterminology. The methodology is tested in the context of online antisemiticdiscourse. The approach considers posts scraped from social media platformsoften used by extremist users. The posts are scraped using seed expressionsrelated to previously known discourse of hatred towards Jews. The method beginsby identifying the expressions most representative of each post and calculatingtheir frequency in the whole corpus. It filters out grammatically incoherentexpressions as well as previously encountered ones so as to focus on emergentwell-formed terminology. This is followed by an assessment of semanticsimilarity to known antisemitic terminology using a fine-tuned large languagemodel and subsequent filtering out of the expressions that are too distantfrom known expressions of hatred. Emergent antisemitic expressions containingterms clearly relating to Jewish topics are then removed to return only codedexpressions of hatred. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2401.10891v1 |
|title| Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data |
|authors| Lihe YangBingyi KangZilong HuangXiaogang XuJiashi FengHengshuang Zhao
|links| http://arxiv.org/abs/2401.10891v1 |
|updated| 2024-01-19 18:59:52 UTC |
|summary| This work presents Depth Anything a highly practical solution for robustmonocular depth estimation. Without pursuing novel technical modules we aim tobuild a simple yet powerful foundation model dealing with any images under anycircumstances. To this end we scale up the dataset by designing a data engineto collect and automatically annotate large-scale unlabeled data 62M whichsignificantly enlarges the data coverage and thus is able to reduce thegeneralization error. We investigate two simple yet effective strategies thatmake data scaling-up promising. First a more challenging optimization targetis created by leveraging data augmentation tools. It compels the model toactively seek extra visual knowledge and acquire robust representations.Second an auxiliary supervision is developed to enforce the model to inheritrich semantic priors from pre-trained encoders. We evaluate its zero-shotcapabilities extensively including six public datasets and randomly capturedphotos. It demonstrates impressive generalization ability. Further throughfine-tuning it with metric depth information from NYUv2 and KITTI new SOTAsare set. Our better depth model also results in a better depth-conditionedControlNet. Our models are released athttps://github.com/LiheYoung/Depth-Anything. |


| Item |Content|
| --- |---|
|idx| 2401.10890v1 |
|title| Event detection from novel data sources: Leveraging satellite imagery alongside GPS traces |
|authors| Ekin UgurelSteffen CoenenMinda Zhou ChenCynthia Chen
|links| http://arxiv.org/abs/2401.10890v1 |
|updated| 2024-01-19 18:59:37 UTC |
|summary| Rapid identification and response to breaking events particularly those thatpose a threat to human life such as natural disasters or conflicts is ofparamount importance. The prevalence of mobile devices and the ubiquity ofnetwork connectivity has generated a massive amount of temporally- andspatially-stamped data. Numerous studies have used mobile data to deriveindividual human mobility patterns for various applications. Similarly theincreasing number of orbital satellites has made it easier to gatherhigh-resolution images capturing a snapshot of a geographical area in sub-dailytemporal frequency. We propose a novel data fusion methodology integratingsatellite imagery with privacy-enhanced mobile data to augment the eventinference task whether in real-time or historical. In the absence of boots onthe ground mobile data is able to give an approximation of human mobilityproximity to one another and the built environment. On the other handsatellite imagery can provide visual information on physical changes to thebuilt and natural environment. The expected use cases for our methodologyinclude small-scale disaster detection i.e. tornadoes wildfires and floodsin rural regions search and rescue operation augmentation for lost hikers inremote wilderness areas and identification of active conflict areas andpopulation displacement in war-torn states. Our implementation is open-sourceon GitHub: https://github.com/ekinugurel/SatMobFusion. |


| Item |Content|
| --- |---|
|idx| 2401.10889v1 |
|title| Synthesizing Moving People with 3D Control |
|authors| Boyi LiJathushan RajasegaranYossi GandelsmanAlexei A. EfrosJitendra Malik
|links| http://arxiv.org/abs/2401.10889v1 |
|updated| 2024-01-19 18:59:11 UTC |
|summary| In this paper we present a diffusion model-based framework for animatingpeople from a single image for a given target 3D motion sequence. Our approachhas two core components: a learning priors about invisible parts of the humanbody and clothing and b rendering novel body poses with proper clothing andtexture. For the first part we learn an in-filling diffusion model tohallucinate unseen parts of a person given a single image. We train this modelon texture map space which makes it more sample-efficient since it isinvariant to pose and viewpoint. Second we develop a diffusion-based renderingpipeline which is controlled by 3D human poses. This produces realisticrenderings of novel poses of the person including clothing hair andplausible in-filling of unseen regions. This disentangled approach allows ourmethod to generate a sequence of images that are faithful to the target motionin the 3D pose and to the input image in terms of visual similarity. Inaddition to that the 3D control allows various synthetic camera trajectoriesto render a person. Our experiments show that our method is resilient ingenerating prolonged motions and varied challenging and complex poses comparedto prior methods. Please check our website for more details:https://boyiliee.github.io/3DHM.github.io/. |


| Item |Content|
| --- |---|
|idx| 2401.10886v1 |
|title| SCENES: Subpixel Correspondence Estimation With Epipolar Supervision |
|authors| Dominik A. KloepferJoão F. HenriquesDylan Campbell
|links| http://arxiv.org/abs/2401.10886v1 |
|updated| 2024-01-19 18:57:46 UTC |
|summary| Extracting point correspondences from two or more views of a scene is afundamental computer vision problem with particular importance for relativecamera pose estimation and structure-from-motion. Existing local featurematching approaches trained with correspondence supervision on large-scaledatasets obtain highly-accurate matches on the test sets. However they do notgeneralise well to new datasets with different characteristics to those theywere trained on unlike classic feature extractors. Instead they requirefinetuning which assumes that ground-truth correspondences or ground-truthcamera poses and 3D structure are available. We relax this assumption byremoving the requirement of 3D structure e.g. depth maps or point clouds andonly require camera pose information which can be obtained from odometry. Wedo so by replacing correspondence losses with epipolar losses which encourageputative matches to lie on the associated epipolar line. While weaker thancorrespondence supervision we observe that this cue is sufficient forfinetuning existing models on new data. We then further relax the assumption ofknown camera poses by using pose estimates in a novel bootstrapping approach.We evaluate on highly challenging datasets including an indoor drone datasetand an outdoor smartphone camera dataset and obtain state-of-the-art resultswithout strong supervision. |


| Item |Content|
| --- |---|
|idx| 2401.10877v1 |
|title| The Cadaver in the Machine: The Social Practices of Measurement and Validation in Motion Capture Technology |
|authors| Emma HarveyHauke SandhausAbigail Z. JacobsEmanuel MossMona Sloane
|links| http://arxiv.org/abs/2401.10877v1 |
|updated| 2024-01-19 18:41:53 UTC |
|summary| Motion capture systems used across various domains make bodyrepresentations concrete through technical processes. We argue that themeasurement of bodies and the validation of measurements for motion capturesystems can be understood as social practices. By analyzing the findings of asystematic literature review N278 through the lens of social practicetheory we show how these practices and their varying attention to errorsbecome ingrained in motion capture design and innovation over time. Moreoverwe show how contemporary motion capture systems perpetuate assumptions abouthuman bodies and their movements. We suggest that social practices ofmeasurement and validation are ubiquitous in the development of data- andsensor-driven systems more broadly and provide this work as a basis forinvestigating hidden design assumptions and their potential negativeconsequences in human-computer interaction. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2401.10811v1 |
|title| Simulation Based Bayesian Optimization |
|authors| Roi NaveiroBecky Tang
|links| http://arxiv.org/abs/2401.10811v1 |
|updated| 2024-01-19 16:56:11 UTC |
|summary| Bayesian Optimization BO is a powerful method for optimizing black-boxfunctions by combining prior knowledge with ongoing function evaluations. BOconstructs a probabilistic surrogate model of the objective function given thecovariates which is in turn used to inform the selection of future evaluationpoints through an acquisition function. For smooth continuous search spacesGaussian Processes GPs are commonly used as the surrogate model as they offeranalytical access to posterior predictive distributions thus facilitating thecomputation and optimization of acquisition functions. However in complexscenarios involving optimizations over categorical or mixed covariate spacesGPs may not be ideal.  This paper introduces Simulation Based Bayesian Optimization SBBO as anovel approach to optimizing acquisition functions that only requiresemphsampling-based access to posterior predictive distributions. SBBO allowsthe use of surrogate probabilistic models tailored for combinatorial spaceswith discrete variables. Any Bayesian model in which posterior inference iscarried out through Markov chain Monte Carlo can be selected as the surrogatemodel in SBBO. In applications involving combinatorial optimization wedemonstrate empirically the effectiveness of SBBO method using various choicesof surrogate models. |


| Item |Content|
| --- |---|
|idx| 2401.10566v1 |
|title| Robust Multi-Modal Density Estimation |
|authors| Anna MészárosJulian F. SchumannJavier Alonso-MoraArkady ZgonnikovJens Kober
|links| http://arxiv.org/abs/2401.10566v1 |
|updated| 2024-01-19 09:10:58 UTC |
|summary| Development of multi-modal probabilistic prediction models has lead to aneed for comprehensive evaluation metrics. While several metrics cancharacterize the accuracy of machine-learned models e.g. negativelog-likelihood Jensen-Shannon divergence these metrics typically operate onprobability densities. Applying them to purely sample-based prediction modelsthus requires that the underlying density function is estimated. Howevercommon methods such as kernel density estimation KDE have been demonstratedto lack robustness while more complex methods have not been evaluated inmulti-modal estimation problems. In this paper we present ROME RObustMulti-modal density Estimator a non-parametric approach for densityestimation which addresses the challenge of estimating multi-modal non-normaland highly correlated distributions. ROME utilizes clustering to segment amulti-modal set of samples into multiple uni-modal ones and then combinessimple KDE estimates obtained for individual clusters in a single multi-modalestimate. We compared our approach to state-of-the-art methods for densityestimation as well as ablations of ROME showing that it not only outperformsestablished methods but is also more robust to a variety of distributions. Ourresults demonstrate that ROME can overcome the issues of over-fitting andover-smoothing exhibited by other estimators promising a more robustevaluation of probabilistic machine learning models. |


| Item |Content|
| --- |---|
|idx| 2401.10474v1 |
|title| LDReg: Local Dimensionality Regularized Self-Supervised Learning |
|authors| Hanxun HuangRicardo J. G. B. CampelloSarah Monazam ErfaniXingjun MaMichael E. HouleJames Bailey
|links| http://arxiv.org/abs/2401.10474v1 |
|updated| 2024-01-19 03:50:19 UTC |
|summary| Representations learned via self-supervised learning SSL can be susceptibleto dimensional collapse where the learned representation subspace is ofextremely low dimensionality and thus fails to represent the full datadistribution and modalities. Dimensional collapse also known as theunderfilling phenomenon is one of the major causes of degraded performance ondownstream tasks. Previous work has investigated the dimensional collapseproblem of SSL at a global level. In this paper we demonstrate thatrepresentations can span over high dimensional space globally but collapselocally. To address this we propose a method called textitlocaldimensionality regularization LDReg. Our formulation is based on thederivation of the Fisher-Rao metric to compare and optimize local distancedistributions at an asymptotically small radius for each data point. Byincreasing the local intrinsic dimensionality we demonstrate through a rangeof experiments that LDReg improves the representation quality of SSL. Theresults also show that LDReg can regularize dimensionality at both local andglobal levels. |


| Item |Content|
| --- |---|
|idx| 2401.10383v1 |
|title| Cooperative Multi-Agent Graph Bandits: UCB Algorithm and Regret Analysis |
|authors| Phevos PaschalidisRunyu ZhangNa Li
|links| http://arxiv.org/abs/2401.10383v1 |
|updated| 2024-01-18 21:36:17 UTC |
|summary| In this paper we formulate the multi-agent graph bandit problem as amulti-agent extension of the graph bandit problem introduced by ZhangJohansson and Li CISS 57 1-6 2023. In our formulation N cooperativeagents travel on a connected graph G with K nodes. Upon arrival at eachnode agents observe a random reward drawn from a node-dependent probabilitydistribution. The reward of the system is modeled as a weighted sum of therewards the agents observe where the weights capture the decreasing marginalreward associated with multiple agents sampling the same node at the same time.We propose an Upper Confidence Bound UCB-based learning algorithmMulti-G-UCB and prove that its expected regret over T steps is bounded byONlogTsqrtKT  DK where D is the diameter of graph G. Lastlywe numerically test our algorithm by comparing it to alternative methods. |


| Item |Content|
| --- |---|
|idx| 2401.10204v1 |
|title| Maximal-Capacity Discrete Memoryless Channel Identification |
|authors| Maximilian EggerRawad BitarAntonia Wachter-ZehDeniz GündüzNir Weinberger
|links| http://arxiv.org/abs/2401.10204v1 |
|updated| 2024-01-18 18:44:10 UTC |
|summary| The problem of identifying the channel with the highest capacity amongseveral discrete memoryless channels DMCs is considered. The problem is castas a pure-exploration multi-armed bandit problem which follows the practicaluse of training sequences to sense the communication channel statistics. Acapacity estimator is proposed and tight confidence bounds on the estimatorerror are derived. Based on this capacity estimator a gap-eliminationalgorithm termed BestChanID is proposed which is oblivious to thecapacity-achieving input distribution and is guaranteed to output the DMC withthe largest capacity with a desired confidence. Furthermore two additionalalgorithms NaiveChanSel and MedianChanEl that output with certain confidence aDMC with capacity close to the maximal are introduced. Each of thosealgorithms is beneficial in a different regime and can be used as a subroutinein BestChanID. The sample complexity of all algorithms is analyzed as afunction of the desired confidence parameter the number of channels and thechannels input and output alphabet sizes. The cost of best channelidentification is shown to scale quadratically with the alphabet size and afundamental lower bound for the required number of channel senses to identifythe best channel with a certain confidence is derived. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2401.10883v1 |
|title| RetinaVR: Democratizing Vitreoretinal Surgery Training with a Portable and Affordable Virtual Reality Simulator in the Metaverse |
|authors| Fares AntakiCédryk DoucetDaniel MiladCharles-Édouard GiguèreBenoit OzellKarim Hammamji
|links| http://arxiv.org/abs/2401.10883v1 |
|updated| 2024-01-19 18:54:10 UTC |
|summary| We developed and validated RetinaVR an affordable and immersive virtualreality simulator for vitreoretinal surgery training using the Meta Quest 2 VRheadset. We focused on four core fundamental skills: core vitrectomyperipheral shaving membrane peeling and endolaser application. The validationstudy involved 10 novice ophthalmology residents and 10 expert vitreoretinalsurgeons. We demonstrated construct validity as shown by the varying userperformance in a way that correlates with experimental runs age sex andexpertise. RetinaVR shows promise as a portable and affordable simulator withpotential to democratize surgical simulation access especially in developingcountries. |


| Item |Content|
| --- |---|
|idx| 2401.10882v1 |
|title| Reinforcement learning for question answering in programming domain using public community scoring as a human feedback |
|authors| Alexey GorbatovskiSergey Kovalchuk
|links| http://arxiv.org/abs/2401.10882v1 |
|updated| 2024-01-19 18:49:36 UTC |
|summary| In this study we investigate the enhancement of the GPT Neo 125M performancein Community Question Answering CQA with a focus on programming through theintegration of Reinforcement Learning from Human Feedback RLHF and theutilization of scores from Stack Overflow. Two distinct reward model trainingstrategies are employed for fine-tuning with Proximal Policy OptimizationPPO. Notably the improvements in performance achieved through this methodare comparable to those of GPT Neo 2.7B parameter variant. Additionally anauxiliary scoring mechanism is introduced which demonstrates the limitationsof conventional linguistic metrics in evaluating responses in the programmingdomain. Through accurate analysis this paper looks at the divergence betweentraditional linguistic metrics and our human-preferences-based reward modelunderscoring the imperative for domain-specific evaluation methods. Byelucidating the complexities involved in applying RLHF to programming CQA andaccentuating the significance of context-aware evaluation this studycontributes to the ongoing efforts in refining Large Language Models throughfocused human feedback. |


| Item |Content|
| --- |---|
|idx| 2401.10880v1 |
|title| DynaVis: Dynamically Synthesized UI Widgets for Visualization Editing |
|authors| Priyan VaithilingamElena L. GlassmanJeevana Priya InalaChenglong Wang
|links| http://arxiv.org/abs/2401.10880v1 |
|updated| 2024-01-19 18:49:03 UTC |
|summary| Users often rely on GUIs to edit and interact with visualizations - adaunting task due to the large space of editing options. As a result users areeither overwhelmed by a complex UI or constrained by a custom UI with atailored fixed subset of options with limited editing flexibility. NaturalLanguage Interfaces NLIs are emerging as a feasible alternative for users tospecify edits. However NLIs forgo the advantages of traditional GUI: theability to explore and repeat edits and see instant visual feedback.  We introduce DynaVis which blends natural language and dynamicallysynthesized UI widgets. As the user describes an editing task in naturallanguage DynaVis performs the edit and synthesizes a persistent widget thatthe user can interact with to make further modifications. Study participantsn24 preferred DynaVis over the NLI-only interface citing ease of furtheredits and editing confidence due to immediate visual feedback. |


| Item |Content|
| --- |---|
|idx| 2401.10877v1 |
|title| The Cadaver in the Machine: The Social Practices of Measurement and Validation in Motion Capture Technology |
|authors| Emma HarveyHauke SandhausAbigail Z. JacobsEmanuel MossMona Sloane
|links| http://arxiv.org/abs/2401.10877v1 |
|updated| 2024-01-19 18:41:53 UTC |
|summary| Motion capture systems used across various domains make bodyrepresentations concrete through technical processes. We argue that themeasurement of bodies and the validation of measurements for motion capturesystems can be understood as social practices. By analyzing the findings of asystematic literature review N278 through the lens of social practicetheory we show how these practices and their varying attention to errorsbecome ingrained in motion capture design and innovation over time. Moreoverwe show how contemporary motion capture systems perpetuate assumptions abouthuman bodies and their movements. We suggest that social practices ofmeasurement and validation are ubiquitous in the development of data- andsensor-driven systems more broadly and provide this work as a basis forinvestigating hidden design assumptions and their potential negativeconsequences in human-computer interaction. |


| Item |Content|
| --- |---|
|idx| 2401.10873v1 |
|title| An AI-Resilient Text Rendering Technique for Reading and Skimming Documents |
|authors| Ziwei GuIan ArawjoKenneth LiJonathan K. KummerfeldElena L. Glassman
|links| http://arxiv.org/abs/2401.10873v1 |
|updated| 2024-01-19 18:33:30 UTC |
|summary| Readers find text difficult to consume for many reasons. Summarization canaddress some of these difficulties but introduce others such as omittingmisrepresenting or hallucinating information which can be hard for a readerto notice. One approach to addressing this problem is to instead modify how theoriginal text is rendered to make important information more salient. Weintroduce Grammar-Preserving Text Saliency Modulation GP-TSM a textrendering method with a novel means of identifying what to de-emphasize.Specifically GP-TSM uses a recursive sentence compression method to identifysuccessive levels of detail beyond the core meaning of a passage which arede-emphasized by rendering words in successively lighter but still legible graytext. In a lab study n18 participants preferred GP-TSM over pre-existingword-level text rendering methods and were able to answer GRE readingcomprehension questions more efficiently. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2401.10383v1 |
|title| Cooperative Multi-Agent Graph Bandits: UCB Algorithm and Regret Analysis |
|authors| Phevos PaschalidisRunyu ZhangNa Li
|links| http://arxiv.org/abs/2401.10383v1 |
|updated| 2024-01-18 21:36:17 UTC |
|summary| In this paper we formulate the multi-agent graph bandit problem as amulti-agent extension of the graph bandit problem introduced by ZhangJohansson and Li CISS 57 1-6 2023. In our formulation N cooperativeagents travel on a connected graph G with K nodes. Upon arrival at eachnode agents observe a random reward drawn from a node-dependent probabilitydistribution. The reward of the system is modeled as a weighted sum of therewards the agents observe where the weights capture the decreasing marginalreward associated with multiple agents sampling the same node at the same time.We propose an Upper Confidence Bound UCB-based learning algorithmMulti-G-UCB and prove that its expected regret over T steps is bounded byONlogTsqrtKT  DK where D is the diameter of graph G. Lastlywe numerically test our algorithm by comparing it to alternative methods. |


| Item |Content|
| --- |---|
|idx| 2401.10149v1 |
|title| Multi-Agent Reinforcement Learning for Maritime Operational Technology Cyber Security |
|authors| Alec WilsonRyan MenziesNeela MorarjiDavid FosterMarco Casassa MontEsin TurkbeylerLisa Gralewski
|links| http://arxiv.org/abs/2401.10149v1 |
|updated| 2024-01-18 17:22:22 UTC |
|summary| This paper demonstrates the potential for autonomous cyber defence to beapplied on industrial control systems and provides a baseline environment tofurther explore Multi-Agent Reinforcement Learnings MARL application to thisproblem domain. It introduces a simulation environment IPMSRL of a genericIntegrated Platform Management System IPMS and explores the use of MARL forautonomous cyber defence decision-making on generic maritime based IPMSOperational Technology OT. OT cyber defensive actions are less mature thanthey are for Enterprise IT. This is due to the relatively brittle nature of OTinfrastructure originating from the use of legacy systems design-timeengineering assumptions and lack of full-scale modern security controls. Thereare many obstacles to be tackled across the cyber landscape due to continuallyincreasing cyber-attack sophistication and the limitations of traditionalIT-centric cyber defence solutions. Traditional IT controls are rarely deployedon OT infrastructure and where they are some threats arent fully addressed.In our experiments a shared critic implementation of Multi Agent ProximalPolicy Optimisation MAPPO outperformed Independent Proximal PolicyOptimisation IPPO. MAPPO reached an optimal policy episode outcome mean of1 after 800K timesteps whereas IPPO was only able to reach an episode outcomemean of 0.966 after one million timesteps. Hyperparameter tuning greatlyimproved training performance. Across one million timesteps the tunedhyperparameters reached an optimal policy whereas the default hyperparametersonly managed to win sporadically with most simulations resulting in a draw. Wetested a real-world constraint attack detection alert success and found thatwhen alert success probability is reduced to 0.75 or 0.9 the MARL defenderswere still able to win in over 97.5 or 99.5 of episodes respectively. |


| Item |Content|
| --- |---|
|idx| 2401.10300v1 |
|title| A Hierarchical Framework with Spatio-Temporal Consistency Learning for Emergence Detection in Complex Adaptive Systems |
|authors| Siyuan ChenXin DuJiahai Wang
|links| http://arxiv.org/abs/2401.10300v1 |
|updated| 2024-01-18 08:55:05 UTC |
|summary| Emergence a global property of complex adaptive systems CASs constitutedby interactive agents is prevalent in real-world dynamic systems e.g.network-level traffic congestions. Detecting its formation and evaporationhelps to monitor the state of a system allowing to issue a warning signal forharmful emergent phenomena. Since there is no centralized controller of CASdetecting emergence based on each agents local observation is desirable butchallenging. Existing works are unable to capture emergence-related spatialpatterns and fail to model the nonlinear relationships among agents. Thispaper proposes a hierarchical framework with spatio-temporal consistencylearning to solve these two problems by learning the system representation andagent representations respectively. Especially spatio-temporal encoders aretailored to capture agents nonlinear relationships and the systems complexevolution. Representations of the agents and the system are learned bypreserving the intrinsic spatio-temporal consistency in a self-supervisedmanner. Our method achieves more accurate detection than traditional methodsand deep learning methods on three datasets with well-known yet hard-to-detectemergent behaviors. Notably our hierarchical framework is generic which canemploy other deep learning methods for agent-level and system-level detection. |


| Item |Content|
| --- |---|
|idx| 2401.09666v1 |
|title| Traffic Smoothing Controllers for Autonomous Vehicles Using Deep Reinforcement Learning and Real-World Trajectory Data |
|authors| Nathan LichtléKathy JangAdit ShahEugene VinitskyJonathan W. LeeAlexandre M. Bayen
|links| http://arxiv.org/abs/2401.09666v1 |
|updated| 2024-01-18 00:50:41 UTC |
|summary| Designing traffic-smoothing cruise controllers that can be deployed ontoautonomous vehicles is a key step towards improving traffic flow reducingcongestion and enhancing fuel efficiency in mixed autonomy traffic. We bypassthe common issue of having to carefully fine-tune a large trafficmicrosimulator by leveraging real-world trajectory data from the I-24 highwayin Tennessee replayed in a one-lane simulation. Using standard deepreinforcement learning methods we train energy-reducing wave-smoothingpolicies. As an input to the agent we observe the speed and distance of onlythe vehicle in front which are local states readily available on most recentvehicles as well as non-local observations about the downstream state of thetraffic. We show that at a low 4 autonomous vehicle penetration rate weachieve significant fuel savings of over 15 on trajectories exhibiting manystop-and-go waves. Finally we analyze the smoothing effect of the controllersand demonstrate robustness to adding lane-changing into the simulation as wellas the removal of downstream information. |


| Item |Content|
| --- |---|
|idx| 2401.09032v1 |
|title| Improved Consensus ADMM for Cooperative Motion Planning of Large-Scale Connected Autonomous Vehicles with Limited Communication |
|authors| Haichao LiuZhenmin HuangZicheng ZhuYulin LiShaojie ShenJun Ma
|links| http://arxiv.org/abs/2401.09032v1 |
|updated| 2024-01-17 07:58:48 UTC |
|summary| This paper investigates a cooperative motion planning problem for large-scaleconnected autonomous vehicles CAVs under limited communications whichaddresses the challenges of high communication and computing resourcerequirements. Our proposed methodology incorporates a parallel optimizationalgorithm with improved consensus ADMM considering a more realistic locallyconnected topology network and time complexity of ON is achieved byexploiting the sparsity in the dual update process. To further enhance thecomputational efficiency we employ a lightweight evolution strategy for thedynamic connectivity graph of CAVs and each sub-problem split from theconsensus ADMM only requires managing a small group of CAVs. The proposedmethod implemented with the receding horizon scheme is validated thoroughlyand comparisons with existing numerical solvers and approaches demonstrate theefficiency of our proposed algorithm. Also simulations on large-scalecooperative driving tasks involving 80 vehicles are performed in thehigh-fidelity CARLA simulator which highlights the remarkable computationalefficiency scalability and effectiveness of our proposed development.Demonstration videos are available athttps://henryhcliu.github.io/icadmm_cmp_carla. |


