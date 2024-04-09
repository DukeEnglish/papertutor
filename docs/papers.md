# cs.CL 

| Item |Content|
| --- |---|
|idx| 2404.05720v1 |
|title| Language-Independent Representations Improve Zero-Shot Summarization |
|authors| Vladimir SolovyevDanni LiuJan Niehues
|links| http://arxiv.org/abs/2404.05720v1 |
|updated| 2024-04-08 17:56:43 UTC |
|summary| Finetuning pretrained models on downstream generation tasks often leads tocatastrophic forgetting in zero-shot conditions. In this work we focus onsummarization and tackle the problem through the lens of language-independentrepresentations. After training on monolingual summarization we performzero-shot transfer to new languages or language pairs. We first show naivelyfinetuned models are highly language-specific in both output behavior andinternal representations resulting in poor zero-shot performance. Next wepropose query-key QK finetuning to decouple task-specific knowledge from thepretrained language generation abilities. Then after showing downsides of thestandard adversarial language classifier we propose a balanced variant thatmore directly enforces language-agnostic representations. Moreover ourqualitative analyses show removing source language identity correlates tozero-shot summarization performance. Our code is openly available. |


| Item |Content|
| --- |---|
|idx| 2404.05719v1 |
|title| Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs |
|authors| Keen YouHaotian ZhangEldon SchoopFloris WeersAmanda SwearnginJeffrey NicholsYinfei YangZhe Gan
|links| http://arxiv.org/abs/2404.05719v1 |
|updated| 2024-04-08 17:55:44 UTC |
|summary| Recent advancements in multimodal large language models MLLMs have beennoteworthy yet these general-domain MLLMs often fall short in their abilityto comprehend and interact effectively with user interface UI screens. Inthis paper we present Ferret-UI a new MLLM tailored for enhancedunderstanding of mobile UI screens equipped with referring grounding andreasoning capabilities. Given that UI screens typically exhibit a moreelongated aspect ratio and contain smaller objects of interest e.g. iconstexts than natural images we incorporate any resolution on top of Ferret tomagnify details and leverage enhanced visual features. Specifically eachscreen is divided into 2 sub-images based on the original aspect ratio i.e.horizontal division for portrait screens and vertical division for landscapescreens. Both sub-images are encoded separately before being sent to LLMs. Wemeticulously gather training samples from an extensive range of elementary UItasks such as icon recognition find text and widget listing. These samplesare formatted for instruction-following with region annotations to facilitateprecise referring and grounding. To augment the models reasoning ability wefurther compile a dataset for advanced tasks including detailed descriptionperception/interaction conversations and function inference. After training onthe curated datasets Ferret-UI exhibits outstanding comprehension of UIscreens and the capability to execute open-ended instructions. For modelevaluation we establish a comprehensive benchmark encompassing all theaforementioned tasks. Ferret-UI excels not only beyond most open-source UIMLLMs but also surpasses GPT-4V on all the elementary UI tasks. |


| Item |Content|
| --- |---|
|idx| 2404.05694v1 |
|title| Comprehensive Study on German Language Models for Clinical and Biomedical Text Understanding |
|authors| Ahmad Idrissi-YaghirAmin DadaHenning SchäferKamyar ArzidehGiulia BaldiniJan TrienesMax HasinJeanette BewersdorffCynthia S. SchmidtMarie BauerKaleb E. SmithJiang BianYonghui WuJörg SchlöttererTorsten ZeschPeter A. HornChristin SeifertFelix NensaJens KleesiekChristoph M. Friedrich
|links| http://arxiv.org/abs/2404.05694v1 |
|updated| 2024-04-08 17:24:04 UTC |
|summary| Recent advances in natural language processing NLP can be largelyattributed to the advent of pre-trained language models such as BERT andRoBERTa. While these models demonstrate remarkable performance on generaldatasets they can struggle in specialized domains such as medicine whereunique domain-specific terminologies domain-specific abbreviations andvarying document structures are common. This paper explores strategies foradapting these models to domain-specific requirements primarily throughcontinuous pre-training on domain-specific data. We pre-trained several Germanmedical language models on 2.4B tokens derived from translated public Englishmedical data and 3B tokens of German clinical data. The resulting models wereevaluated on various German downstream tasks including named entityrecognition NER multi-label classification and extractive questionanswering. Our results suggest that models augmented by clinical andtranslation-based pre-training typically outperform general domain models inmedical contexts. We conclude that continuous pre-training has demonstrated theability to match or even exceed the performance of clinical models trained fromscratch. Furthermore pre-training on clinical data or leveraging translatedtexts have proven to be reliable methods for domain adaptation in medical NLPtasks. |


| Item |Content|
| --- |---|
|idx| 2404.05692v1 |
|title| Evaluating Mathematical Reasoning Beyond Accuracy |
|authors| Shijie XiaXuefeng LiYixin LiuTongshuang WuPengfei Liu
|links| http://arxiv.org/abs/2404.05692v1 |
|updated| 2024-04-08 17:18:04 UTC |
|summary| The leaderboard of Large Language Models LLMs in mathematical tasks hasbeen continuously updated. However the majority of evaluations focus solely onthe final results neglecting the quality of the intermediate steps. Thisoversight can mask underlying problems such as logical errors or unnecessarysteps in the reasoning process. To measure reasoning beyond final-answeraccuracy we introduce ReasonEval a new methodology for evaluating the qualityof reasoning steps. ReasonEval employs textitvalidity andtextitredundancy to characterize the reasoning quality as well asaccompanying LLMs to assess them automatically. Instantiated by base modelsthat possess strong mathematical knowledge and trained with high-qualitylabeled data ReasonEval achieves state-of-the-art performance on human-labeleddatasets and can accurately detect different types of errors generated byperturbation. When applied to evaluate LLMs specialized in math we find thatan increase in final-answer accuracy does not necessarily guarantee animprovement in the overall quality of the reasoning steps for challengingmathematical problems. Additionally we observe that ReasonEval can play asignificant role in data selection. We release the best-performing modelmeta-evaluation script and all evaluation results athttps://github.com/GAIR-NLP/ReasonEval. |


| Item |Content|
| --- |---|
|idx| 2404.05659v1 |
|title| VietMed: A Dataset and Benchmark for Automatic Speech Recognition of Vietnamese in the Medical Domain |
|authors| Khai Le-Duc
|links| http://arxiv.org/abs/2404.05659v1 |
|updated| 2024-04-08 16:43:52 UTC |
|summary| Due to privacy restrictions theres a shortage of publicly available speechrecognition datasets in the medical domain. In this work we present VietMed -a Vietnamese speech recognition dataset in the medical domain comprising 16h oflabeled medical speech 1000h of unlabeled medical speech and 1200h ofunlabeled general-domain speech. To our best knowledge VietMed is by far theworlds largest public medical speech recognition dataset in 7 aspects: totalduration number of speakers diseases recording conditions speaker rolesunique medical terms and accents. VietMed is also by far the largest publicVietnamese speech dataset in terms of total duration. Additionally we are thefirst to present a medical ASR dataset covering all ICD-10 disease groups andall accents within a country. Moreover we release the first public large-scalepre-trained models for Vietnamese ASR w2v2-Viet and XLSR-53-Viet along withthe first public large-scale fine-tuned models for medical ASR. Even withoutany medical data in unsupervised pre-training our best pre-trained modelXLSR-53-Viet generalizes very well to the medical domain by outperformingstate-of-the-art XLSR-53 from 51.8 to 29.6 WER on test set a relativereduction of more than 40. All code data and models are made publiclyavailable here: https://github.com/leduckhai/MultiMed. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2404.05720v1 |
|title| Language-Independent Representations Improve Zero-Shot Summarization |
|authors| Vladimir SolovyevDanni LiuJan Niehues
|links| http://arxiv.org/abs/2404.05720v1 |
|updated| 2024-04-08 17:56:43 UTC |
|summary| Finetuning pretrained models on downstream generation tasks often leads tocatastrophic forgetting in zero-shot conditions. In this work we focus onsummarization and tackle the problem through the lens of language-independentrepresentations. After training on monolingual summarization we performzero-shot transfer to new languages or language pairs. We first show naivelyfinetuned models are highly language-specific in both output behavior andinternal representations resulting in poor zero-shot performance. Next wepropose query-key QK finetuning to decouple task-specific knowledge from thepretrained language generation abilities. Then after showing downsides of thestandard adversarial language classifier we propose a balanced variant thatmore directly enforces language-agnostic representations. Moreover ourqualitative analyses show removing source language identity correlates tozero-shot summarization performance. Our code is openly available. |


| Item |Content|
| --- |---|
|idx| 2404.05717v1 |
|title| SwapAnything: Enabling Arbitrary Object Swapping in Personalized Visual Editing |
|authors| Jing GuYilin WangNanxuan ZhaoWei XiongQing LiuZhifei ZhangHe ZhangJianming ZhangHyunJoon JungXin Eric Wang
|links| http://arxiv.org/abs/2404.05717v1 |
|updated| 2024-04-08 17:52:29 UTC |
|summary| Effective editing of personal content holds a pivotal role in enablingindividuals to express their creativity weaving captivating narratives withintheir visual stories and elevate the overall quality and impact of theirvisual content. Therefore in this work we introduce SwapAnything a novelframework that can swap any objects in an image with personalized conceptsgiven by the reference while keeping the context unchanged. Compared withexisting methods for personalized subject swapping SwapAnything has threeunique advantages: 1 precise control of arbitrary objects and parts ratherthan the main subject 2 more faithful preservation of context pixels 3better adaptation of the personalized concept to the image. First we proposetargeted variable swapping to apply region control over latent feature maps andswap masked variables for faithful context preservation and initial semanticconcept swapping. Then we introduce appearance adaptation to seamlessly adaptthe semantic concept into the original image in terms of target locationshape style and content during the image generation process. Extensiveresults on both human and automatic evaluation demonstrate significantimprovements of our approach over baseline methods on personalized swapping.Furthermore SwapAnything shows its precise and faithful swapping abilitiesacross single object multiple objects partial object and cross-domainswapping tasks. SwapAnything also achieves great performance on text-basedswapping and tasks beyond swapping such as object insertion. |


| Item |Content|
| --- |---|
|idx| 2404.05695v1 |
|title| Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer |
|authors| Xinyang GuYen-Jen WangJianyu Chen
|links| http://arxiv.org/abs/2404.05695v1 |
|updated| 2024-04-08 17:26:28 UTC |
|summary| Humanoid-Gym is an easy-to-use reinforcement learning RL framework based onNvidia Isaac Gym designed to train locomotion skills for humanoid robotsemphasizing zero-shot transfer from simulation to the real-world environment.Humanoid-Gym also integrates a sim-to-sim framework from Isaac Gym to Mujocothat allows users to verify the trained policies in different physicalsimulations to ensure the robustness and generalization of the policies. Thisframework is verified by RobotEras XBot-S 1.2-meter tall humanoid robot andXBot-L 1.65-meter tall humanoid robot in a real-world environment withzero-shot sim-to-real transfer. The project website and source code can befound at: https://sites.google.com/view/humanoid-gym/. |


| Item |Content|
| --- |---|
|idx| 2404.05694v1 |
|title| Comprehensive Study on German Language Models for Clinical and Biomedical Text Understanding |
|authors| Ahmad Idrissi-YaghirAmin DadaHenning SchäferKamyar ArzidehGiulia BaldiniJan TrienesMax HasinJeanette BewersdorffCynthia S. SchmidtMarie BauerKaleb E. SmithJiang BianYonghui WuJörg SchlöttererTorsten ZeschPeter A. HornChristin SeifertFelix NensaJens KleesiekChristoph M. Friedrich
|links| http://arxiv.org/abs/2404.05694v1 |
|updated| 2024-04-08 17:24:04 UTC |
|summary| Recent advances in natural language processing NLP can be largelyattributed to the advent of pre-trained language models such as BERT andRoBERTa. While these models demonstrate remarkable performance on generaldatasets they can struggle in specialized domains such as medicine whereunique domain-specific terminologies domain-specific abbreviations andvarying document structures are common. This paper explores strategies foradapting these models to domain-specific requirements primarily throughcontinuous pre-training on domain-specific data. We pre-trained several Germanmedical language models on 2.4B tokens derived from translated public Englishmedical data and 3B tokens of German clinical data. The resulting models wereevaluated on various German downstream tasks including named entityrecognition NER multi-label classification and extractive questionanswering. Our results suggest that models augmented by clinical andtranslation-based pre-training typically outperform general domain models inmedical contexts. We conclude that continuous pre-training has demonstrated theability to match or even exceed the performance of clinical models trained fromscratch. Furthermore pre-training on clinical data or leveraging translatedtexts have proven to be reliable methods for domain adaptation in medical NLPtasks. |


| Item |Content|
| --- |---|
|idx| 2404.05689v1 |
|title| Automated discovery of symbolic laws governing skill acquisition from naturally occurring data |
|authors| Sannyuya LiuQing LiXiaoxuan ShenJianwen SunZongkai Yang
|links| http://arxiv.org/abs/2404.05689v1 |
|updated| 2024-04-08 17:15:37 UTC |
|summary| Skill acquisition is a key area of research in cognitive psychology as itencompasses multiple psychological processes. The laws discovered underexperimental paradigms are controversial and lack generalizability. This paperaims to unearth the laws of skill learning from large-scale training log data.A two-stage algorithm was developed to tackle the issues of unobservablecognitive states and algorithmic explosion in searching. Initially a deeplearning model is employed to determine the learners cognitive state andassess the feature importance. Subsequently symbolic regression algorithms areutilized to parse the neural network model into algebraic equations. Theexperimental results of simulated data demonstrate that the proposed algorithmcan accurately restore various preset laws within a certain range of noise incontinues feedback setting. Application of proposed method to Lumosity trainingdata demonstrates superior performance compared to traditional and latestmodels in terms of fitness. The results indicate the discovery of two new formsof skill acquisition laws while some previous findings have been reaffirmed. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2404.05728v1 |
|title| A Large-Scale Exploration of $μ$-Transfer |
|authors| Lucas Lingle
|links| http://arxiv.org/abs/2404.05728v1 |
|updated| 2024-04-08 17:59:44 UTC |
|summary| Large neural network models have become a mainstay of natural languageprocessing and computer vision yet their initialization and learning rates areset in a largely heuristic fashion potentially varying from paper to paper andone model size to the next. The mu-Parameterization muP offers apotential solution to these challenges yielding scaling rules for modelinitialization and learning rates and reportedly enabling zero-shothyperparameter transfer from small to large models in a variety of cases.  Despite the evident promise the muP scaling rules are not yet widelyadopted perhaps due to higher implementation complexity many variations orcomplex theoretical background. This work investigates muP empiricallyfocusing on the ubiquitous transformer architecture and aims to answer asimple question: does mu-Transfer yield optimal learning rates in practiceFrom models with 2M to 10B parameters we show that mu-Transfer works asintended for the majority of important cases but also identify some surprisingcases where it may not. |


| Item |Content|
| --- |---|
|idx| 2404.05723v1 |
|title| Predicting Overtakes in Trucks Using CAN Data |
|authors| Talha Hanif ButtPrayag TiwariFernando Alonso-Fernandez
|links| http://arxiv.org/abs/2404.05723v1 |
|updated| 2024-04-08 17:58:22 UTC |
|summary| Safe overtakes in trucks are crucial to prevent accidents reduce congestionand ensure efficient traffic flow making early prediction essential for timelyand informed driving decisions. Accordingly we investigate the detection oftruck overtakes from CAN data. Three classifiers Artificial Neural NetworksANN Random Forest and Support Vector Machines SVM are employed for thetask. Our analysis covers up to 10 seconds before the overtaking event usingan overlapping sliding window of 1 second to extract CAN features. We observethat the prediction scores of the overtake class tend to increase as weapproach the overtake trigger while the no-overtake class remain stable oroscillates depending on the classifier. Thus the best accuracy is achievedwhen approaching the trigger making early overtaking prediction challenging.The classifiers show good accuracy in classifying overtakes Recall/TPR  93but accuracy is suboptimal in classifying no-overtakes TNR typically 80-90and below 60 for one SVM variant. We further combine two classifiers RandomForest and linear SVM by averaging their output scores. The fusion is observedto improve no-overtake classification TNR  92 at the expense of reducingovertake accuracy TPR. However the latter is kept above 91 near theovertake trigger. Therefore the fusion balances TPR and TNR providing moreconsistent performance than individual classifiers. |


| Item |Content|
| --- |---|
|idx| 2404.05695v1 |
|title| Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer |
|authors| Xinyang GuYen-Jen WangJianyu Chen
|links| http://arxiv.org/abs/2404.05695v1 |
|updated| 2024-04-08 17:26:28 UTC |
|summary| Humanoid-Gym is an easy-to-use reinforcement learning RL framework based onNvidia Isaac Gym designed to train locomotion skills for humanoid robotsemphasizing zero-shot transfer from simulation to the real-world environment.Humanoid-Gym also integrates a sim-to-sim framework from Isaac Gym to Mujocothat allows users to verify the trained policies in different physicalsimulations to ensure the robustness and generalization of the policies. Thisframework is verified by RobotEras XBot-S 1.2-meter tall humanoid robot andXBot-L 1.65-meter tall humanoid robot in a real-world environment withzero-shot sim-to-real transfer. The project website and source code can befound at: https://sites.google.com/view/humanoid-gym/. |


| Item |Content|
| --- |---|
|idx| 2404.05694v1 |
|title| Comprehensive Study on German Language Models for Clinical and Biomedical Text Understanding |
|authors| Ahmad Idrissi-YaghirAmin DadaHenning SchäferKamyar ArzidehGiulia BaldiniJan TrienesMax HasinJeanette BewersdorffCynthia S. SchmidtMarie BauerKaleb E. SmithJiang BianYonghui WuJörg SchlöttererTorsten ZeschPeter A. HornChristin SeifertFelix NensaJens KleesiekChristoph M. Friedrich
|links| http://arxiv.org/abs/2404.05694v1 |
|updated| 2024-04-08 17:24:04 UTC |
|summary| Recent advances in natural language processing NLP can be largelyattributed to the advent of pre-trained language models such as BERT andRoBERTa. While these models demonstrate remarkable performance on generaldatasets they can struggle in specialized domains such as medicine whereunique domain-specific terminologies domain-specific abbreviations andvarying document structures are common. This paper explores strategies foradapting these models to domain-specific requirements primarily throughcontinuous pre-training on domain-specific data. We pre-trained several Germanmedical language models on 2.4B tokens derived from translated public Englishmedical data and 3B tokens of German clinical data. The resulting models wereevaluated on various German downstream tasks including named entityrecognition NER multi-label classification and extractive questionanswering. Our results suggest that models augmented by clinical andtranslation-based pre-training typically outperform general domain models inmedical contexts. We conclude that continuous pre-training has demonstrated theability to match or even exceed the performance of clinical models trained fromscratch. Furthermore pre-training on clinical data or leveraging translatedtexts have proven to be reliable methods for domain adaptation in medical NLPtasks. |


| Item |Content|
| --- |---|
|idx| 2404.05693v1 |
|title| Evaluating the Efficacy of Cut-and-Paste Data Augmentation in Semantic Segmentation for Satellite Imagery |
|authors| Ionut M. MotoiLeonardo SaraceniDaniele NardiThomas A. Ciarfuglia
|links| http://arxiv.org/abs/2404.05693v1 |
|updated| 2024-04-08 17:18:30 UTC |
|summary| Satellite imagery is crucial for tasks like environmental monitoring andurban planning. Typically it relies on semantic segmentation or Land Use LandCover LULC classification to categorize each pixel. Despite the advancementsbrought about by Deep Neural Networks DNNs their performance in segmentationtasks is hindered by challenges such as limited availability of labeled dataclass imbalance and the inherent variability and complexity of satelliteimages. In order to mitigate those issues our study explores the effectivenessof a Cut-and-Paste augmentation technique for semantic segmentation insatellite images. We adapt this augmentation which usually requires labeledinstances to the case of semantic segmentation. By leveraging the connectedcomponents in the semantic segmentation labels we extract instances that arethen randomly pasted during training. Using the DynamicEarthNet dataset and aU-Net model for evaluation we found that this augmentation significantlyenhances the mIoU score on the test set from 37.9 to 44.1. This findinghighlights the potential of the Cut-and-Paste augmentation to improve thegeneralization capabilities of semantic segmentation models in satelliteimagery. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2404.05729v1 |
|title| Finding Visual Task Vectors |
|authors| Alberto HojelYutong BaiTrevor DarrellAmir GlobersonAmir Bar
|links| http://arxiv.org/abs/2404.05729v1 |
|updated| 2024-04-08 17:59:46 UTC |
|summary| Visual Prompting is a technique for teaching models to perform a visual taskvia in-context examples without any additional training. In this work weanalyze the activations of MAE-VQGAN a recent Visual Prompting model and findtask vectors activations that encode task-specific information. Equipped withthis insight we demonstrate that it is possible to identify the task vectorsand use them to guide the network towards performing different tasks withoutproviding any input-output examples. To find task vectors we compute theaverage intermediate activations per task and use the REINFORCE algorithm tosearch for the subset of task vectors. The resulting task vectors guide themodel towards performing a task better than the original model without the needfor input-output examples. |


| Item |Content|
| --- |---|
|idx| 2404.05726v1 |
|title| MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding |
|authors| Bo HeHengduo LiYoung Kyun JangMenglin JiaXuefei CaoAshish ShahAbhinav ShrivastavaSer-Nam Lim
|links| http://arxiv.org/abs/2404.05726v1 |
|updated| 2024-04-08 17:59:24 UTC |
|summary| With the success of large language models LLMs integrating the visionmodel into LLMs to build vision-language foundation models has gained much moreinterest recently. However existing LLM-based large multimodal models e.g.Video-LLaMA VideoChat can only take in a limited number of frames for shortvideo understanding. In this study we mainly focus on designing an efficientand effective model for long-term video understanding. Instead of trying toprocess more frames simultaneously like most existing work we propose toprocess videos in an online manner and store past video information in a memorybank. This allows our model to reference historical video content for long-termanalysis without exceeding LLMs context length constraints or GPU memorylimits. Our memory bank can be seamlessly integrated into current multimodalLLMs in an off-the-shelf manner. We conduct extensive experiments on variousvideo understanding tasks such as long-video understanding video questionanswering and video captioning and our model can achieve state-of-the-artperformances across multiple datasets. Code available athttps://boheumd.github.io/MA-LMM/. |


| Item |Content|
| --- |---|
|idx| 2404.05719v1 |
|title| Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs |
|authors| Keen YouHaotian ZhangEldon SchoopFloris WeersAmanda SwearnginJeffrey NicholsYinfei YangZhe Gan
|links| http://arxiv.org/abs/2404.05719v1 |
|updated| 2024-04-08 17:55:44 UTC |
|summary| Recent advancements in multimodal large language models MLLMs have beennoteworthy yet these general-domain MLLMs often fall short in their abilityto comprehend and interact effectively with user interface UI screens. Inthis paper we present Ferret-UI a new MLLM tailored for enhancedunderstanding of mobile UI screens equipped with referring grounding andreasoning capabilities. Given that UI screens typically exhibit a moreelongated aspect ratio and contain smaller objects of interest e.g. iconstexts than natural images we incorporate any resolution on top of Ferret tomagnify details and leverage enhanced visual features. Specifically eachscreen is divided into 2 sub-images based on the original aspect ratio i.e.horizontal division for portrait screens and vertical division for landscapescreens. Both sub-images are encoded separately before being sent to LLMs. Wemeticulously gather training samples from an extensive range of elementary UItasks such as icon recognition find text and widget listing. These samplesare formatted for instruction-following with region annotations to facilitateprecise referring and grounding. To augment the models reasoning ability wefurther compile a dataset for advanced tasks including detailed descriptionperception/interaction conversations and function inference. After training onthe curated datasets Ferret-UI exhibits outstanding comprehension of UIscreens and the capability to execute open-ended instructions. For modelevaluation we establish a comprehensive benchmark encompassing all theaforementioned tasks. Ferret-UI excels not only beyond most open-source UIMLLMs but also surpasses GPT-4V on all the elementary UI tasks. |


| Item |Content|
| --- |---|
|idx| 2404.05717v1 |
|title| SwapAnything: Enabling Arbitrary Object Swapping in Personalized Visual Editing |
|authors| Jing GuYilin WangNanxuan ZhaoWei XiongQing LiuZhifei ZhangHe ZhangJianming ZhangHyunJoon JungXin Eric Wang
|links| http://arxiv.org/abs/2404.05717v1 |
|updated| 2024-04-08 17:52:29 UTC |
|summary| Effective editing of personal content holds a pivotal role in enablingindividuals to express their creativity weaving captivating narratives withintheir visual stories and elevate the overall quality and impact of theirvisual content. Therefore in this work we introduce SwapAnything a novelframework that can swap any objects in an image with personalized conceptsgiven by the reference while keeping the context unchanged. Compared withexisting methods for personalized subject swapping SwapAnything has threeunique advantages: 1 precise control of arbitrary objects and parts ratherthan the main subject 2 more faithful preservation of context pixels 3better adaptation of the personalized concept to the image. First we proposetargeted variable swapping to apply region control over latent feature maps andswap masked variables for faithful context preservation and initial semanticconcept swapping. Then we introduce appearance adaptation to seamlessly adaptthe semantic concept into the original image in terms of target locationshape style and content during the image generation process. Extensiveresults on both human and automatic evaluation demonstrate significantimprovements of our approach over baseline methods on personalized swapping.Furthermore SwapAnything shows its precise and faithful swapping abilitiesacross single object multiple objects partial object and cross-domainswapping tasks. SwapAnything also achieves great performance on text-basedswapping and tasks beyond swapping such as object insertion. |


| Item |Content|
| --- |---|
|idx| 2404.05705v1 |
|title| Learning 3D-Aware GANs from Unposed Images with Template Feature Field |
|authors| Xinya ChenHanlei GuoYanrui BinShangzhan ZhangYuanbo YangYue WangYujun ShenYiyi Liao
|links| http://arxiv.org/abs/2404.05705v1 |
|updated| 2024-04-08 17:42:08 UTC |
|summary| Collecting accurate camera poses of training images has been shown to wellserve the learning of 3D-aware generative adversarial networks GANs yet canbe quite expensive in practice. This work targets learning 3D-aware GANs fromunposed images for which we propose to perform on-the-fly pose estimation oftraining images with a learned template feature field TeFF. Concretely inaddition to a generative radiance field as in previous approaches we ask thegenerator to also learn a field from 2D semantic features while sharing thedensity from the radiance field. Such a framework allows us to acquire acanonical 3D feature template leveraging the dataset mean discovered by thegenerative model and further efficiently estimate the pose parameters on realdata. Experimental results on various challenging datasets demonstrate thesuperiority of our approach over state-of-the-art alternatives from both thequalitative and the quantitative perspectives. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2404.05678v1 |
|title| Flexible Fairness Learning via Inverse Conditional Permutation |
|authors| Yuheng LaiLeying Guan
|links| http://arxiv.org/abs/2404.05678v1 |
|updated| 2024-04-08 16:57:44 UTC |
|summary| Equalized odds as a popular notion of algorithmic fairness aims to ensurethat sensitive variables such as race and gender do not unfairly influencethe algorithm prediction when conditioning on the true outcome. Despite rapidadvancements most of the current research focuses on the violation ofequalized odds caused by one sensitive attribute leaving the challenge ofsimultaneously accounting for multiple attributes under-addressed. We addressthis gap by introducing a fairness learning approach that integratesadversarial learning with a novel inverse conditional permutation. Thisapproach effectively and flexibly handles multiple sensitive attributespotentially of mixed data types. The efficacy and flexibility of our method aredemonstrated through both simulation studies and empirical analysis ofreal-world datasets. |


| Item |Content|
| --- |---|
|idx| 2404.05555v1 |
|title| On the Convergence of Continual Learning with Adaptive Methods |
|authors| Seungyub HanYeongmo KimTaehyun ChoJungwoo Lee
|links| http://arxiv.org/abs/2404.05555v1 |
|updated| 2024-04-08 14:28:27 UTC |
|summary| One of the objectives of continual learning is to prevent catastrophicforgetting in learning multiple tasks sequentially and the existing solutionshave been driven by the conceptualization of the plasticity-stability dilemma.However the convergence of continual learning for each sequential task is lessstudied so far. In this paper we provide a convergence analysis ofmemory-based continual learning with stochastic gradient descent and empiricalevidence that training current tasks causes the cumulative degradation ofprevious tasks. We propose an adaptive method for nonconvex continual learningNCCL which adjusts step sizes of both previous and current tasks with thegradients. The proposed method can achieve the same convergence rate as the SGDmethod when the catastrophic forgetting term which we define in the paper issuppressed at each iteration. Further we demonstrate that the proposedalgorithm improves the performance of continual learning over existing methodsfor several image classification tasks. |


| Item |Content|
| --- |---|
|idx| 2404.05209v1 |
|title| Maximally Forward-Looking Core Inflation |
|authors| Philippe Goulet CoulombeKarin KlieberChristophe BarretteMaximilian Goebel
|links| http://arxiv.org/abs/2404.05209v1 |
|updated| 2024-04-08 05:39:41 UTC |
|summary| Timely monetary policy decision-making requires timely core inflationmeasures. We create a new core inflation series that is explicitly designed tosucceed at that goal. Precisely we introduce the Assemblage Regression ageneralized nonnegative ridge regression problem that optimizes the priceindexs subcomponent weights such that the aggregate is maximally predictive offuture headline inflation. Ordering subcomponents according to their rank ineach period switches the algorithm to be learning supervised trimmed inflation- or put differently the maximally forward-looking summary statistic of therealized price changes distribution. In an extensive out-of-sample forecastingexperiment for the US and the euro area we find substantial improvements forsignaling medium-term inflation developments in both the pre- and post-Covidyears. Those coming from the supervised trimmed version are particularlystriking and are attributable to a highly asymmetric trimming which contrastswith conventional indicators. We also find that this metric was indicatingfirst upward pressures on inflation as early as mid-2020 and quickly capturedthe turning point in 2022. We also consider extensions like assemblinginflation from geographical regions trimmed temporal aggregation and buildingcore measures specialized for either upside or downside inflation risks. |


| Item |Content|
| --- |---|
|idx| 2404.05185v1 |
|title| Convergence analysis of controlled particle systems arising in deep learning: from finite to infinite sample size |
|authors| Huafu LiaoAlpár R. MészárosChenchen MouChao Zhou
|links| http://arxiv.org/abs/2404.05185v1 |
|updated| 2024-04-08 04:22:55 UTC |
|summary| This paper deals with a class of neural SDEs and studies the limitingbehavior of the associated sampled optimal control problems as the sample sizegrows to infinity. The neural SDEs with N samples can be linked to theN-particle systems with centralized control. We analyze theHamilton--Jacobi--Bellman equation corresponding to the N-particle system andestablish regularity results which are uniform in N. The uniform regularityestimates are obtained by the stochastic maximum principle and the analysis ofa backward stochastic Riccati equation. Using these uniform regularity resultswe show the convergence of the minima of objective functionals and optimalparameters of the neural SDEs as the sample size N tends to infinity. Thelimiting objects can be identified with suitable functions defined on theWasserstein space of Borel probability measures. Furthermore quantitativealgebraic convergence rates are also obtained. |


| Item |Content|
| --- |---|
|idx| 2404.05155v1 |
|title| On the price of exact truthfulness in incentive-compatible online learning with bandit feedback: A regret lower bound for WSU-UX |
|authors| Ali MortazaviJunhao LinNishant A. Mehta
|links| http://arxiv.org/abs/2404.05155v1 |
|updated| 2024-04-08 02:41:32 UTC |
|summary| In one view of the classical game of prediction with expert advice withbinary outcomes in each round each expert maintains an adversarially chosenbelief and honestly reports this belief. We consider a recently introducedstrategic variant of this problem with selfish reputation-seeking expertswhere each expert strategically reports in order to maximize their expectedfuture reputation based on their belief. In this work our goal is to design analgorithm for the selfish experts problem that is incentive-compatible IC oremphtruthful meaning each experts best strategy is to report truthfullywhile also ensuring the algorithm enjoys sublinear regret with respect to theexpert with the best belief. Freeman et al. 2020 recently studied thisproblem in the full information and bandit settings and obtained truthfulno-regret algorithms by leveraging prior work on wagering mechanisms. Whiletheir results under full information match the minimax rate for the classicalhonest experts problem the best-known regret for their bandit algorithmWSU-UX is OT2/3 which does not match the minimax rate for the classicalhonest bandits setting. It was unclear whether the higher regret was anartifact of their analysis or a limitation of WSU-UX. We show via explicitconstruction of loss sequences that the algorithm suffers a worst-caseOmegaT2/3 lower bound. Left open is the possibility that a different ICalgorithm obtains OsqrtT regret. Yet WSU-UX was a natural choice forsuch an algorithm owing to the limited design room for IC algorithms in thissetting. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2404.05719v1 |
|title| Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs |
|authors| Keen YouHaotian ZhangEldon SchoopFloris WeersAmanda SwearnginJeffrey NicholsYinfei YangZhe Gan
|links| http://arxiv.org/abs/2404.05719v1 |
|updated| 2024-04-08 17:55:44 UTC |
|summary| Recent advancements in multimodal large language models MLLMs have beennoteworthy yet these general-domain MLLMs often fall short in their abilityto comprehend and interact effectively with user interface UI screens. Inthis paper we present Ferret-UI a new MLLM tailored for enhancedunderstanding of mobile UI screens equipped with referring grounding andreasoning capabilities. Given that UI screens typically exhibit a moreelongated aspect ratio and contain smaller objects of interest e.g. iconstexts than natural images we incorporate any resolution on top of Ferret tomagnify details and leverage enhanced visual features. Specifically eachscreen is divided into 2 sub-images based on the original aspect ratio i.e.horizontal division for portrait screens and vertical division for landscapescreens. Both sub-images are encoded separately before being sent to LLMs. Wemeticulously gather training samples from an extensive range of elementary UItasks such as icon recognition find text and widget listing. These samplesare formatted for instruction-following with region annotations to facilitateprecise referring and grounding. To augment the models reasoning ability wefurther compile a dataset for advanced tasks including detailed descriptionperception/interaction conversations and function inference. After training onthe curated datasets Ferret-UI exhibits outstanding comprehension of UIscreens and the capability to execute open-ended instructions. For modelevaluation we establish a comprehensive benchmark encompassing all theaforementioned tasks. Ferret-UI excels not only beyond most open-source UIMLLMs but also surpasses GPT-4V on all the elementary UI tasks. |


| Item |Content|
| --- |---|
|idx| 2404.05572v1 |
|title| Eye Tracking on Text Reading with Visual Enhancements |
|authors| Franziska HuthMaurice KochMiriam AwadDaniel WeiskopfKuno Kurzhals
|links| http://dx.doi.org/10.1145/3649902.3653521 |
|updated| 2024-04-08 14:49:26 UTC |
|summary| The interplay between text and visualization is gaining importance for mediawhere traditional text is enriched by visual elements to improve readabilityand emphasize facts. In two controlled eye-tracking experiments N12 weapproach answers to the question: How do visualization techniques influencereading behavior We compare plain text to that marked with highlights iconsand word-sized data visualizations. We assess quantitative metricseyemovement completion time error rate and subjective feedbackpersonalpreference and ratings. The results indicate that visualization techniquesespecially in the first experiment show promising trends for improved readingbehavior. The results also show the need for further research to make readingmore effective and inform suggestions for future studies. |


| Item |Content|
| --- |---|
|idx| 2404.05462v1 |
|title| Interactive Formal Specification for Mathematical Problems of Engineers |
|authors| Walther Neuper
|links| http://dx.doi.org/10.4204/EPTCS.400.8 |
|updated| 2024-04-08 12:41:22 UTC |
|summary| The paper presents the second part of a precise description of the prototypethat has been developed in the course of the ISAC project over the last twodecades. This part describes the specify-phase while the first partdescribing the solve-phase is already published.  In the specify-phase a student interactively constructs a formalspecification. The ISAC prototype implements formal specifications asestablished in theoretical computer science however the input language forthe construction avoids requiring users to have knowledge of logic this makesthe system useful for various engineering faculties and also for high school.  The paper discusses not only ISACs design of the specify-phase in detailbut also gives a brief introduction to implementation with the aim ofadvertising the re-use of formal frameworks inclusive respective front-endswith their generic tools for language definition and their rich pool ofsoftware components for formal mathematics. |


| Item |Content|
| --- |---|
|idx| 2404.05442v1 |
|title| Unlocking Adaptive User Experience with Generative AI |
|authors| Yutan HuangTanjila KanijAnuradha MadugallaShruti MahajanChetan AroraJohn Grundy
|links| http://arxiv.org/abs/2404.05442v1 |
|updated| 2024-04-08 12:22:39 UTC |
|summary| Developing user-centred applications that address diverse user needs requiresrigorous user research. This is time effort and cost-consuming. With therecent rise of generative AI techniques based on Large Language Models LLMsthere is a possibility that these powerful tools can be used to developadaptive interfaces. This paper presents a novel approach to develop userpersonas and adaptive interface candidates for a specific domain using ChatGPT.We develop user personas and adaptive interfaces using both ChatGPT and atraditional manual process and compare these outcomes. To obtain data for thepersonas we collected data from 37 survey participants and 4 interviews incollaboration with a not-for-profit organisation. The comparison of ChatGPTgenerated content and manual content indicates promising results that encourageusing LLMs in the adaptive interfaces design process. |


| Item |Content|
| --- |---|
|idx| 2404.05429v1 |
|title| Re-Ranking News Comments by Constructiveness and Curiosity Significantly Increases Perceived Respect, Trustworthiness, and Interest |
|authors| Emily SaltzZaria HowardTin Acosta
|links| http://arxiv.org/abs/2404.05429v1 |
|updated| 2024-04-08 11:56:22 UTC |
|summary| Online commenting platforms have commonly developed systems to address onlineharms by removing and down-ranking content. An alternative under-exploredapproach is to focus on up-ranking content to proactively prioritize prosocialcommentary and set better conversational norms. We present a study with 460English-speaking US-based news readers to understand the effects of re-rankingcomments by constructiveness curiosity and personal stories on a variety ofoutcomes related to willingness to participate and engage as well as perceivedcredibility and polarization in a comment section. In our rich-media surveyexperiment participants across these four ranking conditions and a controlgroup reviewed prototypes of comment sections of a Politics op-ed and Diningarticle. We found that outcomes varied significantly by article type.Up-ranking curiosity and constructiveness improved a number of measures for thePolitics article including perceived textitRespecttextitTrustworthiness and textitInterestingness of the comment section.Constructiveness also increased perceptions that the comments were favorable toRepublicans with no condition worsening perceptions of partisans.Additionally in the Dining article personal stories and constructivenessrankings significantly improved the perceived informativeness of the comments.Overall these findings indicate that incorporating prosocial qualities ofspeech into ranking could be a promising approach to promote healthier lesspolarized dialogue in online comment sections. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2404.05569v1 |
|title| 360°REA: Towards A Reusable Experience Accumulation with 360° Assessment for Multi-Agent System |
|authors| Shen GaoHao LiZhengliang ShiChengrui HuangQuan TuZhiliang TianMinlie HuangShuo Shang
|links| http://arxiv.org/abs/2404.05569v1 |
|updated| 2024-04-08 14:43:13 UTC |
|summary| Large language model agents have demonstrated remarkable advancements acrossvarious complex tasks. Recent works focus on optimizing the agent team oremploying self-reflection to iteratively solve complex tasks. Since theseagents are all based on the same LLM only conducting self-evaluation orremoving underperforming agents does not substantively enhance the capabilityof the agents. We argue that a comprehensive evaluation and accumulatingexperience from evaluation feedback is an effective approach to improvingsystem performance. In this paper we propose Reusable Experience Accumulationwith 360deg Assessment 360degREA a hierarchical multi-agent frameworkinspired by corporate organizational practices. The framework employs a novel360deg performance assessment method for multi-perspective performanceevaluation with fine-grained assessment. To enhance the capability of agents inaddressing complex tasks we introduce dual-level experience pool for agents toaccumulate experience through fine-grained assessment. Extensive experiments oncomplex task datasets demonstrate the effectiveness of 360degREA. |


| Item |Content|
| --- |---|
|idx| 2404.04912v1 |
|title| Opinion Dynamics for Utility Maximizing Agents: Exploring the Impact of Resource Penalty |
|authors| Prashil WankhedeNirabhra MandalSonia MartínezPavankumar Tallapragada
|links| http://arxiv.org/abs/2404.04912v1 |
|updated| 2024-04-07 10:39:53 UTC |
|summary| We propose a continuous-time nonlinear model of opinion dynamics withutility-maximizing agents connected via a social influence network. Adistinguishing feature of the proposed model is the inclusion of anopinion-dependent resource-penalty term in the utilities which limits theagents from holding opinions of large magnitude. The proposed utility functionsalso account for how the relative resources within the social group affect bothan agents stubbornness and social influence. Each agent myopically seeks tomaximize its utility by revising its opinion in the gradient ascent directionof its utility function thus leading to the proposed opinion dynamics. We showthat for any arbitrary social influence network opinions are ultimatelybounded. For networks with weak antagonistic relations we show that thereexists a globally exponentially stable equilibrium using contraction theory. Weestablish conditions for the existence of consensus equilibrium and analyze therelative dominance of the agents at consensus. We also conduct a game-theoreticanalysis of the underlying opinion formation game including on Nash equilibriaand on prices of anarchy in terms of satisfaction ratios. Additionally we alsoinvestigate the oscillatory behavior of opinions in a two-agent scenario.Finally simulations illustrate our findings. |


| Item |Content|
| --- |---|
|idx| 2404.04678v1 |
|title| Automatic Gradient Estimation for Calibrating Crowd Models with Discrete Decision Making |
|authors| Philipp AndelfingerJustin N. Kreikemeyer
|links| http://arxiv.org/abs/2404.04678v1 |
|updated| 2024-04-06 16:48:12 UTC |
|summary| Recently proposed gradient estimators enable gradient descent over stochasticprograms with discrete jumps in the response surface which are not covered byautomatic differentiation AD alone. Although these estimators capability toguide a swift local search has been shown for certain problems theirapplicability to models relevant to real-world applications remains largelyunexplored. As the gradients governing the choice in candidate solutions arecalculated from sampled simulation trajectories the optimization procedurebears similarities to metaheuristics such as particle swarm optimization whichputs the focus on the different methods calibration progress per functionevaluation. Here we consider the calibration of force-based crowd evacuationmodels based on the popular Social Force model augmented by discrete decisionmaking. After studying the ability of an AD-based estimator for branchingprograms to capture the simulations rugged response surface calibrationproblems are tackled using gradient descent and two metaheuristics. As our maininsights we find 1 that the estimations fidelity benefits from disregardingjumps of large magnitude inherent to the Social Force model and 2 that thecommon problem of calibration by adjusting a simulation input distributionobviates the need for AD across the Social Force calculations allowinggradient descent to excel. |


| Item |Content|
| --- |---|
|idx| 2404.04497v1 |
|title| Self-organizing Multiagent Target Enclosing under Limited Information and Safety Guarantees |
|authors| Praveen Kumar RanjanAbhinav SinhaYongcan Cao
|links| http://arxiv.org/abs/2404.04497v1 |
|updated| 2024-04-06 04:15:05 UTC |
|summary| This paper introduces an approach to address the target enclosing problemusing non-holonomic multiagent systems where agents autonomously self-organizethemselves in the desired formation around a fixed target. Our approachcombines global enclosing behavior and local collision avoidance mechanisms bydevising a novel potential function and sliding manifold. In our approachagents independently move toward the desired enclosing geometry when apart andactivate the collision avoidance mechanism when a collision is imminentthereby guaranteeing inter-agent safety. We rigorously show that an agent doesnot need to ensure safety with every other agent and put forth a concept of thenearest colliding agent for any arbitrary agent with whom ensuring safety issufficient to avoid collisions in the entire swarm. The proposed controleliminates the need for a fixed or pre-established agent arrangement around thetarget and requires only relative information between an agent and the target.This makes our design particularly appealing for scenarios with limited globalinformation hence significantly reducing communication requirements. Wefinally present simulation results to vindicate the efficacy of the proposedmethod. |


| Item |Content|
| --- |---|
|idx| 2404.03984v1 |
|title| ROMA-iQSS: An Objective Alignment Approach via State-Based Value Learning and ROund-Robin Multi-Agent Scheduling |
|authors| Chi-Hui LinJoewie J. KohAlessandro RonconeLijun Chen
|links| http://arxiv.org/abs/2404.03984v1 |
|updated| 2024-04-05 09:39:47 UTC |
|summary| Effective multi-agent collaboration is imperative for solving complexdistributed problems. In this context two key challenges must be addressed:first autonomously identifying optimal objectives for collective outcomessecond aligning these objectives among agents. Traditional frameworks oftenreliant on centralized learning struggle with scalability and efficiency inlarge multi-agent systems. To overcome these issues we introduce adecentralized state-based value learning algorithm that enables agents toindependently discover optimal states. Furthermore we introduce a novelmechanism for multi-agent interaction wherein less proficient agents followand adopt policies from more experienced ones thereby indirectly guiding theirlearning process. Our theoretical analysis shows that our approach leadsdecentralized agents to an optimal collective policy. Empirical experimentsfurther demonstrate that our method outperforms existing decentralizedstate-based and action-based value learning strategies by effectivelyidentifying and aligning optimal objectives. |


