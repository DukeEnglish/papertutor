# cs.CL 

| Item |Content|
| --- |---|
|idx| 2405.08784v1 |
|title| Refinement of an Epilepsy Dictionary through Human Annotation of Health-related posts on Instagram |
|authors| Aehong MinXuan WangRion Brattig CorreiaJordan RozumWendy R. MillerLuis M. Rocha
|links| http://arxiv.org/abs/2405.08784v1 |
|updated| 2024-05-14 17:27:59 UTC |
|summary| We used a dictionary built from biomedical terminology extracted from varioussources such as DrugBank MedDRA MedlinePlus TCMGeneDIT to tag more than 8million Instagram posts by users who have mentioned an epilepsy-relevant drugat least once between 2010 and early 2016. A random sample of 1771 posts with2947 term matches was evaluated by human annotators to identifyfalse-positives. OpenAIs GPT series models were compared against humanannotation. Frequent terms with a high false-positive rate were removed fromthe dictionary. Analysis of the estimated false-positive rates of the annotatedterms revealed 8 ambiguous terms plus synonyms used in Instagram posts whichwere removed from the original dictionary. To study the effect of removingthose terms we constructed knowledge networks using the refined and theoriginal dictionaries and performed an eigenvector-centrality analysis on bothnetworks. We show that the refined dictionary thus produced leads to asignificantly different rank of important terms as measured by theireigenvector-centrality of the knowledge networks. Furthermore the mostimportant terms obtained after refinement are of greater medical relevance. Inaddition we show that OpenAIs GPT series models fare worse than humanannotators in this task. |


| Item |Content|
| --- |---|
|idx| 2405.08760v1 |
|title| Is the Pope Catholic? Yes, the Pope is Catholic. Generative Evaluation of Intent Resolution in LLMs |
|authors| Akhila YerukolaSaujas VaduguruDaniel FriedMaarten Sap
|links| http://arxiv.org/abs/2405.08760v1 |
|updated| 2024-05-14 16:48:56 UTC |
|summary| Humans often express their communicative intents indirectly or non-literallywhich requires their interlocutors -- human or AI -- to understand beyond theliteral meaning of words. While most existing work has focused ondiscriminative evaluations we present a new approach to generatively evaluatelarge language models LLMs intention understanding by examining theirresponses to non-literal utterances. Ideally an LLM should respond in linewith the true intention of a non-literal utterance not its literalinterpretation. Our findings show that LLMs struggle to generate pragmaticallyrelevant responses to non-literal language achieving only 50-55 accuracy onaverage. While explicitly providing oracle intentions significantly improvesperformance e.g. 75 for Mistral-Instruct this still indicates challengesin leveraging given intentions to produce appropriate responses. Usingchain-of-thought to make models spell out intentions yields much smaller gains60 for Mistral-Instruct. These findings suggest that LLMs are not yeteffective pragmatic interlocutors highlighting the need for better approachesfor modeling intentions and utilizing them for pragmatic generation. |


| Item |Content|
| --- |---|
|idx| 2405.08751v1 |
|title| From Text to Context: An Entailment Approach for News Stakeholder Classification |
|authors| Alapan KuilaSudeshna Sarkar
|links| http://arxiv.org/abs/2405.08751v1 |
|updated| 2024-05-14 16:35:21 UTC |
|summary| Navigating the complex landscape of news articles involves understanding thevarious actors or entities involved referred to as news stakeholders. Thesestakeholders ranging from policymakers to opposition figures citizens andmore play pivotal roles in shaping news narratives. Recognizing theirstakeholder types reflecting their roles political alignments socialstanding and more is paramount for a nuanced comprehension of news content.Despite existing works focusing on salient entity extraction coveragevariations and political affiliations through social media data the automateddetection of stakeholder roles within news content remains an underexploreddomain. In this paper we bridge this gap by introducing an effective approachto classify stakeholder types in news articles. Our method involvestransforming the stakeholder classification problem into a natural languageinference task utilizing contextual information from news articles andexternal knowledge to enhance the accuracy of stakeholder type detection.Moreover our proposed model showcases efficacy in zero-shot settings furtherextending its applicability to diverse news contexts. |


| Item |Content|
| --- |---|
|idx| 2405.08729v1 |
|title| Targeted Augmentation for Low-Resource Event Extraction |
|authors| Sijia WangLifu Huang
|links| http://arxiv.org/abs/2405.08729v1 |
|updated| 2024-05-14 16:15:31 UTC |
|summary| Addressing the challenge of low-resource information extraction remains anongoing issue due to the inherent information scarcity within limited trainingexamples. Existing data augmentation methods considered potential solutionsstruggle to strike a balance between weak augmentation e.g. synonymaugmentation and drastic augmentation e.g. conditional generation withoutproper guidance. This paper introduces a novel paradigm that employs targetedaugmentation and back validation to produce augmented examples with enhanceddiversity polarity accuracy and coherence. Extensive experimental resultsdemonstrate the effectiveness of the proposed paradigm. Furthermore identifiedlimitations are discussed shedding light on areas for future improvement. |


| Item |Content|
| --- |---|
|idx| 2405.08644v1 |
|title| Thinking Tokens for Language Modeling |
|authors| David HerelTomas Mikolov
|links| http://arxiv.org/abs/2405.08644v1 |
|updated| 2024-05-14 14:21:43 UTC |
|summary| How much is 56 times 37 Language models often make mistakes in these typesof difficult calculations. This is usually explained by their inability toperform complex reasoning. Since language models rely on large training setsand great memorization capability naturally they are not equipped to runcomplex calculations. However one can argue that humans also cannot performthis calculation immediately and require a considerable amount of time toconstruct the solution. In order to enhance the generalization capability oflanguage models and as a parallel to human behavior we propose to use specialthinking tokens which allow the model to perform much more calculationswhenever a complex problem is encountered. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2405.08792v1 |
|title| Towards Enhanced RAC Accessibility: Leveraging Datasets and LLMs |
|authors| Edison Jair Bejarano SepulvedaNicolai Potes HectorSantiago Pineda MontoyaFelipe Ivan RodriguezJaime Enrique OrduyAlec Rosales CabezasDanny Traslaviña NavarreteSergio Madrid Farfan
|links| http://arxiv.org/abs/2405.08792v1 |
|updated| 2024-05-14 17:41:07 UTC |
|summary| This paper explores the potential of large language models LLMs to make theAeronautical Regulations of Colombia RAC more accessible. Given thecomplexity and extensive technicality of the RAC this study introduces a novelapproach to simplifying these regulations for broader understanding. Bydeveloping the first-ever RAC database which contains 24478 expertly labeledquestion-and-answer pairs and fine-tuning LLMs specifically for RACapplications the paper outlines the methodology for dataset assemblyexpert-led annotation and model training. Utilizing the Gemma1.1 2b modelalong with advanced techniques like Unsloth for efficient VRAM usage and flashattention mechanisms the research aims to expedite training processes. Thisinitiative establishes a foundation to enhance the comprehensibility andaccessibility of RAC potentially benefiting novices and reducing dependence onexpert consultations for navigating the aviation industrys regulatorylandscape.  You can visit the datasethttps://huggingface.co/somosnlp/gemma-1.1-2b-it_ColombiaRAC_FullyCurated_format_chatML_V1and the modelhttps://huggingface.co/datasets/somosnlp/ColombiaRAC_FullyCurated here. |


| Item |Content|
| --- |---|
|idx| 2405.08790v1 |
|title| Kolmogorov-Arnold Networks (KANs) for Time Series Analysis |
|authors| Cristian J. Vaca-RubioLuis BlancoRoberto PereiraMàrius Caus
|links| http://arxiv.org/abs/2405.08790v1 |
|updated| 2024-05-14 17:38:17 UTC |
|summary| This paper introduces a novel application of Kolmogorov-Arnold NetworksKANs to time series forecasting leveraging their adaptive activationfunctions for enhanced predictive modeling. Inspired by the Kolmogorov-Arnoldrepresentation theorem KANs replace traditional linear weights withspline-parametrized univariate functions allowing them to learn activationpatterns dynamically. We demonstrate that KANs outperforms conventionalMulti-Layer Perceptrons MLPs in a real-world satellite traffic forecastingtask providing more accurate results with considerably fewer number oflearnable parameters. We also provide an ablation study of KAN-specificparameters impact on performance. The proposed approach opens new avenues foradaptive forecasting models emphasizing the potential of KANs as a powerfultool in predictive analytics. |


| Item |Content|
| --- |---|
|idx| 2405.08780v1 |
|title| Harnessing the power of longitudinal medical imaging for eye disease prognosis using Transformer-based sequence modeling |
|authors| Gregory HolsteMingquan LinRuiwen ZhouFei WangLei LiuQi YanSarah H. Van TasselKyle KovacsEmily Y. ChewZhiyong LuZhangyang WangYifan Peng
|links| http://arxiv.org/abs/2405.08780v1 |
|updated| 2024-05-14 17:15:28 UTC |
|summary| Deep learning has enabled breakthroughs in automated diagnosis from medicalimaging with many successful applications in ophthalmology. However standardmedical image classification approaches only assess disease presence at thetime of acquisition neglecting the common clinical setting of longitudinalimaging. For slow progressive eye diseases like age-related maculardegeneration AMD and primary open-angle glaucoma POAG patients undergorepeated imaging over time to track disease progression and forecasting thefuture risk of developing disease is critical to properly plan treatment. Ourproposed Longitudinal Transformer for Survival Analysis LTSA enables dynamicdisease prognosis from longitudinal medical imaging modeling the time todisease from sequences of fundus photography images captured over longirregular time periods. Using longitudinal imaging data from the Age-RelatedEye Disease Study AREDS and Ocular Hypertension Treatment Study OHTS LTSAsignificantly outperformed a single-image baseline in 19/20 head-to-headcomparisons on late AMD prognosis and 18/20 comparisons on POAG prognosis. Atemporal attention analysis also suggested that while the most recent image istypically the most influential prior imaging still provides additionalprognostic value. |


| Item |Content|
| --- |---|
|idx| 2405.08768v1 |
|title| EfficientTrain++: Generalized Curriculum Learning for Efficient Visual Backbone Training |
|authors| Yulin WangYang YueRui LuYizeng HanShiji SongGao Huang
|links| http://arxiv.org/abs/2405.08768v1 |
|updated| 2024-05-14 17:00:43 UTC |
|summary| The superior performance of modern visual backbones usually comes with acostly training procedure. We contribute to this issue by generalizing the ideaof curriculum learning beyond its original formulation i.e. training modelsusing easier-to-harder data. Specifically we reformulate the trainingcurriculum as a soft-selection function which uncovers progressively moredifficult patterns within each example during training instead of performingeasier-to-harder sample selection. Our work is inspired by an intriguingobservation on the learning dynamics of visual backbones: during the earlierstages of training the model predominantly learns to recognize someeasier-to-learn discriminative patterns in the data. These patterns whenobserved through frequency and spatial domains incorporate lower-frequencycomponents and the natural image contents without distortion or dataaugmentation. Motivated by these findings we propose a curriculum where themodel always leverages all the training data at every learning stage yet theexposure to the easier-to-learn patterns of each example is initiated firstwith harder patterns gradually introduced as training progresses. To implementthis idea in a computationally efficient way we introduce a cropping operationin the Fourier spectrum of the inputs enabling the model to learn from onlythe lower-frequency components. Then we show that exposing the contents ofnatural images can be readily achieved by modulating the intensity of dataaugmentation. Finally we integrate these aspects and design curriculumschedules with tailored search algorithms. The resulting methodEfficientTrain is simple general yet surprisingly effective. It reducesthe training time of a wide variety of popular models by 1.5-3.0x onImageNet-1K/22K without sacrificing accuracy. It also demonstrates efficacy inself-supervised learning e.g. MAE. |


| Item |Content|
| --- |---|
|idx| 2405.08760v1 |
|title| Is the Pope Catholic? Yes, the Pope is Catholic. Generative Evaluation of Intent Resolution in LLMs |
|authors| Akhila YerukolaSaujas VaduguruDaniel FriedMaarten Sap
|links| http://arxiv.org/abs/2405.08760v1 |
|updated| 2024-05-14 16:48:56 UTC |
|summary| Humans often express their communicative intents indirectly or non-literallywhich requires their interlocutors -- human or AI -- to understand beyond theliteral meaning of words. While most existing work has focused ondiscriminative evaluations we present a new approach to generatively evaluatelarge language models LLMs intention understanding by examining theirresponses to non-literal utterances. Ideally an LLM should respond in linewith the true intention of a non-literal utterance not its literalinterpretation. Our findings show that LLMs struggle to generate pragmaticallyrelevant responses to non-literal language achieving only 50-55 accuracy onaverage. While explicitly providing oracle intentions significantly improvesperformance e.g. 75 for Mistral-Instruct this still indicates challengesin leveraging given intentions to produce appropriate responses. Usingchain-of-thought to make models spell out intentions yields much smaller gains60 for Mistral-Instruct. These findings suggest that LLMs are not yeteffective pragmatic interlocutors highlighting the need for better approachesfor modeling intentions and utilizing them for pragmatic generation. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2405.08813v1 |
|title| CinePile: A Long Video Question Answering Dataset and Benchmark |
|authors| Ruchit RawalKhalid SaifullahRonen BasriDavid JacobsGowthami SomepalliTom Goldstein
|links| http://arxiv.org/abs/2405.08813v1 |
|updated| 2024-05-14 17:59:02 UTC |
|summary| Current datasets for long-form video understanding often fall short ofproviding genuine long-form comprehension challenges as many tasks derivedfrom these datasets can be successfully tackled by analyzing just one or a fewrandom frames from a video. To address this issue we present a novel datasetand benchmark CinePile specifically designed for authentic long-form videounderstanding. This paper details our innovative approach for creating aquestion-answer dataset utilizing advanced LLMs with human-in-the-loop andbuilding upon human-generated raw data. Our comprehensive dataset comprises305000 multiple-choice questions MCQs covering various visual andmultimodal aspects including temporal comprehension understandinghuman-object interactions and reasoning about events or actions within ascene. Additionally we evaluate recent video-centric LLMs both open-sourceand proprietary on the test split of our dataset. The findings reveal thateven state-of-the-art video-centric LLMs significantly lag behind humanperformance in these tasks highlighting the complexity and challenge inherentin video understanding. The dataset is available athttps://hf.co/datasets/tomg-group-umd/cinepile |


| Item |Content|
| --- |---|
|idx| 2405.08801v1 |
|title| Prospects of Privacy Advantage in Quantum Machine Learning |
|authors| Jamie HeredgeNiraj KumarDylan HermanShouvanik ChakrabartiRomina YalovetzkyShree Hari SureshbabuMarco Pistoia
|links| http://arxiv.org/abs/2405.08801v1 |
|updated| 2024-05-14 17:49:18 UTC |
|summary| Ensuring data privacy in machine learning models is critical particularly indistributed settings where model gradients are typically shared among multipleparties to allow collaborative learning. Motivated by the increasing success ofrecovering input data from the gradients of classical models this studyaddresses a central question: How hard is it to recover the input data from thegradients of quantum machine learning models Focusing on variational quantumcircuits VQC as learning models we uncover the crucial role played by thedynamical Lie algebra DLA of the VQC ansatz in determining privacyvulnerabilities. While the DLA has previously been linked to the classicalsimulatability and trainability of VQC models this work for the first timeestablishes its connection to the privacy of VQC models. In particular we showthat properties conducive to the trainability of VQCs such as apolynomial-sized DLA also facilitate the extraction of detailed snapshots ofthe input. We term this a weak privacy breach as the snapshots enable trainingVQC models for distinct learning tasks without direct access to the originalinput. Further we investigate the conditions for a strong privacy breach wherethe original input data can be recovered from these snapshots by classical orquantum-assisted polynomial time methods. We establish conditions on theencoding map such as classical simulatability overlap with DLA basis and itsFourier frequency characteristics that enable such a privacy breach of VQCmodels. Our findings thus play a crucial role in detailing the prospects ofquantum privacy advantage by guiding the requirements for designing quantummachine learning models that balance trainability with robust privacyprotection. |


| Item |Content|
| --- |---|
|idx| 2405.08793v1 |
|title| A Brief Introduction to Causal Inference in Machine Learning |
|authors| Kyunghyun Cho
|links| http://arxiv.org/abs/2405.08793v1 |
|updated| 2024-05-14 17:41:55 UTC |
|summary| This is a lecture note produced for DS-GA 3001.003 Special Topics in DS -Causal Inference in Machine Learning at the Center for Data Science New YorkUniversity in Spring 2024. This course was created to target masters and PhDlevel students with basic background in machine learning but who were notexposed to causal inference or causal reasoning in general previously. Inparticular this course focuses on introducing such students to expand theirview and knowledge of machine learning to incorporate causal reasoning as thisaspect is at the core of so-called out-of-distribution generalization or lackthereof. |


| Item |Content|
| --- |---|
|idx| 2405.08792v1 |
|title| Towards Enhanced RAC Accessibility: Leveraging Datasets and LLMs |
|authors| Edison Jair Bejarano SepulvedaNicolai Potes HectorSantiago Pineda MontoyaFelipe Ivan RodriguezJaime Enrique OrduyAlec Rosales CabezasDanny Traslaviña NavarreteSergio Madrid Farfan
|links| http://arxiv.org/abs/2405.08792v1 |
|updated| 2024-05-14 17:41:07 UTC |
|summary| This paper explores the potential of large language models LLMs to make theAeronautical Regulations of Colombia RAC more accessible. Given thecomplexity and extensive technicality of the RAC this study introduces a novelapproach to simplifying these regulations for broader understanding. Bydeveloping the first-ever RAC database which contains 24478 expertly labeledquestion-and-answer pairs and fine-tuning LLMs specifically for RACapplications the paper outlines the methodology for dataset assemblyexpert-led annotation and model training. Utilizing the Gemma1.1 2b modelalong with advanced techniques like Unsloth for efficient VRAM usage and flashattention mechanisms the research aims to expedite training processes. Thisinitiative establishes a foundation to enhance the comprehensibility andaccessibility of RAC potentially benefiting novices and reducing dependence onexpert consultations for navigating the aviation industrys regulatorylandscape.  You can visit the datasethttps://huggingface.co/somosnlp/gemma-1.1-2b-it_ColombiaRAC_FullyCurated_format_chatML_V1and the modelhttps://huggingface.co/datasets/somosnlp/ColombiaRAC_FullyCurated here. |


| Item |Content|
| --- |---|
|idx| 2405.08790v1 |
|title| Kolmogorov-Arnold Networks (KANs) for Time Series Analysis |
|authors| Cristian J. Vaca-RubioLuis BlancoRoberto PereiraMàrius Caus
|links| http://arxiv.org/abs/2405.08790v1 |
|updated| 2024-05-14 17:38:17 UTC |
|summary| This paper introduces a novel application of Kolmogorov-Arnold NetworksKANs to time series forecasting leveraging their adaptive activationfunctions for enhanced predictive modeling. Inspired by the Kolmogorov-Arnoldrepresentation theorem KANs replace traditional linear weights withspline-parametrized univariate functions allowing them to learn activationpatterns dynamically. We demonstrate that KANs outperforms conventionalMulti-Layer Perceptrons MLPs in a real-world satellite traffic forecastingtask providing more accurate results with considerably fewer number oflearnable parameters. We also provide an ablation study of KAN-specificparameters impact on performance. The proposed approach opens new avenues foradaptive forecasting models emphasizing the potential of KANs as a powerfultool in predictive analytics. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2405.08816v1 |
|title| The RoboDrive Challenge: Drive Anytime Anywhere in Any Condition |
|authors| Lingdong KongShaoyuan XieHanjiang HuYaru NiuWei Tsang OoiBenoit R. CottereauLai Xing NgYuexin MaWenwei ZhangLiang PanKai ChenZiwei LiuWeichao QiuWei ZhangXu CaoHao LuYing-Cong ChenCaixin KangXinning ZhouChengyang YingWentao ShangXingxing WeiYinpeng DongBo YangShengyin JiangZeliang MaDengyi JiHaiwen LiXingliang HuangYu TianGenghua KouFan JiaYingfei LiuTiancai WangYing LiXiaoshuai HaoYifan YangHui ZhangMengchuan WeiYi ZhouHaimei ZhaoJing ZhangJinke LiXiao HeXiaoqiang ChengBingyang ZhangLirong ZhaoDianlei DingFangsheng LiuYixiang YanHongming WangNanfei YeLun LuoYubo TianYiwei ZuoZhe CaoYi RenYunfan LiWenjie LiuXun WuYifan MaoMing LiJian LiuJiayang LiuZihan QinCunxi ChuJialei XuWenbo ZhaoJunjun JiangXianming LiuZiyan WangChiwei LiShilong LiChendong YuanSongyue YangWentao LiuPeng ChenBin ZhouYubo WangChi ZhangJianhang SunHai ChenXiao YangLizhong WangDongyi FuYongchun LinHuitong YangHaoang LiYadan LuoXianjing ChengYong Xu
|links| http://arxiv.org/abs/2405.08816v1 |
|updated| 2024-05-14 17:59:57 UTC |
|summary| In the realm of autonomous driving robust perception underout-of-distribution conditions is paramount for the safe deployment ofvehicles. Challenges such as adverse weather sensor malfunctions andenvironmental unpredictability can severely impact the performance ofautonomous systems. The 2024 RoboDrive Challenge was crafted to propel thedevelopment of driving perception technologies that can withstand and adapt tothese real-world variabilities. Focusing on four pivotal tasks -- BEVdetection map segmentation semantic occupancy prediction and multi-viewdepth estimation -- the competition laid down a gauntlet to innovate andenhance system resilience against typical and atypical disturbances. Thisyears challenge consisted of five distinct tracks and attracted 140 registeredteams from 93 institutes across 11 countries resulting in nearly one thousandsubmissions evaluated through our servers. The competition culminated in 15top-performing solutions which introduced a range of innovative approachesincluding advanced data augmentation multi-sensor fusion self-supervisedlearning for error correction and new algorithmic strategies to enhance sensorrobustness. These contributions significantly advanced the state of the artparticularly in handling sensor inconsistencies and environmental variability.Participants through collaborative efforts pushed the boundaries of currenttechnologies showcasing their potential in real-world scenarios. Extensiveevaluations and analyses provided insights into the effectiveness of thesesolutions highlighting key trends and successful strategies for improving theresilience of driving perception systems. This challenge has set a newbenchmark in the field providing a rich repository of techniques expected toguide future research in this field. |


| Item |Content|
| --- |---|
|idx| 2405.08815v1 |
|title| Efficient Vision-Language Pre-training by Cluster Masking |
|authors| Zihao WeiZixuan PanAndrew Owens
|links| http://arxiv.org/abs/2405.08815v1 |
|updated| 2024-05-14 17:59:40 UTC |
|summary| We propose a simple strategy for masking image patches during visual-languagecontrastive learning that improves the quality of the learned representationsand the training speed. During each iteration of training we randomly maskclusters of visually similar image patches as measured by their raw pixelintensities. This provides an extra learning signal beyond the contrastivetraining itself since it forces a model to predict words for masked visualstructures solely from context. It also speeds up training by reducing theamount of data used in each image. We evaluate the effectiveness of our modelby pre-training on a number of benchmarks finding that it outperforms othermasking strategies such as FLIP on the quality of the learned representation. |


| Item |Content|
| --- |---|
|idx| 2405.08813v1 |
|title| CinePile: A Long Video Question Answering Dataset and Benchmark |
|authors| Ruchit RawalKhalid SaifullahRonen BasriDavid JacobsGowthami SomepalliTom Goldstein
|links| http://arxiv.org/abs/2405.08813v1 |
|updated| 2024-05-14 17:59:02 UTC |
|summary| Current datasets for long-form video understanding often fall short ofproviding genuine long-form comprehension challenges as many tasks derivedfrom these datasets can be successfully tackled by analyzing just one or a fewrandom frames from a video. To address this issue we present a novel datasetand benchmark CinePile specifically designed for authentic long-form videounderstanding. This paper details our innovative approach for creating aquestion-answer dataset utilizing advanced LLMs with human-in-the-loop andbuilding upon human-generated raw data. Our comprehensive dataset comprises305000 multiple-choice questions MCQs covering various visual andmultimodal aspects including temporal comprehension understandinghuman-object interactions and reasoning about events or actions within ascene. Additionally we evaluate recent video-centric LLMs both open-sourceand proprietary on the test split of our dataset. The findings reveal thateven state-of-the-art video-centric LLMs significantly lag behind humanperformance in these tasks highlighting the complexity and challenge inherentin video understanding. The dataset is available athttps://hf.co/datasets/tomg-group-umd/cinepile |


| Item |Content|
| --- |---|
|idx| 2405.08807v1 |
|title| SciFIBench: Benchmarking Large Multimodal Models for Scientific Figure Interpretation |
|authors| Jonathan RobertsKai HanNeil HoulsbySamuel Albanie
|links| http://arxiv.org/abs/2405.08807v1 |
|updated| 2024-05-14 17:54:17 UTC |
|summary| Large multimodal models LMMs have proven flexible and generalisable acrossmany tasks and fields. Although they have strong potential to aid scientificresearch their capabilities in this domain are not well characterised. A keyaspect of scientific research is the ability to understand and interpretfigures which serve as a rich compressed source of complex information. Inthis work we present SciFIBench a scientific figure interpretation benchmark.Our main benchmark consists of a 1000-question gold set of multiple-choicequestions split between two tasks across 12 categories. The questions arecurated from CS arXiv paper figures and captions using adversarial filteringto find hard negatives and human verification for quality control. We evaluate26 LMMs on SciFIBench finding it to be a challenging benchmark. Finally weinvestigate the alignment and reasoning faithfulness of the LMMs on augmentedquestion sets from our benchmark. We release SciFIBench to encourage progressin this domain. |


| Item |Content|
| --- |---|
|idx| 2405.08794v1 |
|title| Ambiguous Annotations: When is a Pedestrian not a Pedestrian? |
|authors| Luisa SchwirtenJannes ScholzDaniel KondermannJanis Keuper
|links| http://arxiv.org/abs/2405.08794v1 |
|updated| 2024-05-14 17:44:34 UTC |
|summary| Datasets labelled by human annotators are widely used in the training andtesting of machine learning models. In recent years researchers areincreasingly paying attention to label quality. However it is not alwayspossible to objectively determine whether an assigned label is correct or not.The present work investigates this ambiguity in the annotation of autonomousdriving datasets as an important dimension of data quality. Our experimentsshow that excluding highly ambiguous data from the training improves modelperformance of a state-of-the-art pedestrian detector in terms of LAMRprecision and F1 score thereby saving training time and annotation costs.Furthermore we demonstrate that in order to safely remove ambiguous instancesand ensure the retained representativeness of the training data anunderstanding of the properties of the dataset and class under investigation iscrucial. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2405.08719v1 |
|title| Addressing Misspecification in Simulation-based Inference through Data-driven Calibration |
|authors| Antoine WehenkelJuan L. GamellaOzan SenerJens BehrmannGuillermo SapiroMarco CuturiJörn-Henrik Jacobsen
|links| http://arxiv.org/abs/2405.08719v1 |
|updated| 2024-05-14 16:04:39 UTC |
|summary| Driven by steady progress in generative modeling simulation-based inferenceSBI has enabled inference over stochastic simulators. However recent workhas demonstrated that model misspecification can harm SBIs reliability. Thiswork introduces robust posterior estimation ROPE a framework that overcomesmodel misspecification with a small real-world calibration set of ground truthparameter measurements. We formalize the misspecification gap as the solutionof an optimal transport problem between learned representations of real-worldand simulated observations. Assuming the prior distribution over the parametersof interest is known and well-specified our method offers a controllablebalance between calibrated uncertainty and informative inference under allpossible misspecifications of the simulator. Our empirical results on foursynthetic tasks and two real-world problems demonstrate that ROPE outperformsbaselines and consistently returns informative and calibrated credibleintervals. |


| Item |Content|
| --- |---|
|idx| 2405.08699v1 |
|title| Weakly-supervised causal discovery based on fuzzy knowledge and complex data complementarity |
|authors| Wenrui LiWei ZhangQinghao ZhangXuegong ZhangXiaowo Wang
|links| http://arxiv.org/abs/2405.08699v1 |
|updated| 2024-05-14 15:39:22 UTC |
|summary| Causal discovery based on observational data is important for deciphering thecausal mechanism behind complex systems. However the effectiveness of existingcausal discovery methods is limited due to inferior prior knowledge domaininconsistencies and the challenges of high-dimensional datasets with smallsample sizes. To address this gap we propose a novel weakly-supervised fuzzyknowledge and data co-driven causal discovery method named KEEL. KEEL adopts afuzzy causal knowledge schema to encapsulate diverse types of fuzzy knowledgeand forms corresponding weakened constraints. This schema not only lessens thedependency on expertise but also allows various types of limited anderror-prone fuzzy knowledge to guide causal discovery. It can enhance thegeneralization and robustness of causal discovery especially inhigh-dimensional and small-sample scenarios. In addition we integrate theextended linear causal model ELCM into KEEL for dealing with themulti-distribution and incomplete data. Extensive experiments with differentdatasets demonstrate the superiority of KEEL over several state-of-the-artmethods in accuracy robustness and computational efficiency. For causaldiscovery in real protein signal transduction processes KEEL outperforms thebenchmark method with limited data. In summary KEEL is effective to tackle thecausal discovery tasks with higher accuracy while alleviating the requirementfor extensive domain expertise. |


| Item |Content|
| --- |---|
|idx| 2405.08675v1 |
|title| Simplifying Debiased Inference via Automatic Differentiation and Probabilistic Programming |
|authors| Alex Luedtke
|links| http://arxiv.org/abs/2405.08675v1 |
|updated| 2024-05-14 14:56:54 UTC |
|summary| We introduce an algorithm that simplifies the construction of efficientestimators making them accessible to a broader audience. Dimple takes asinput computer code representing a parameter of interest and outputs anefficient estimator. Unlike standard approaches it does not require users toderive a functional derivative known as the efficient influence function.Dimple avoids this task by applying automatic differentiation to thestatistical functional of interest. Doing so requires expressing thisfunctional as a composition of primitives satisfying a novel differentiabilitycondition. Dimple also uses this composition to determine the nuisances it mustestimate. In software primitives can be implemented independently of oneanother and reused across different estimation problems. We provide aproof-of-concept Python implementation and showcase through examples how itallows users to go from parameter specification to efficient estimation withjust a few lines of code. |


| Item |Content|
| --- |---|
|idx| 2405.08498v1 |
|title| Learning Decision Policies with Instrumental Variables through Double Machine Learning |
|authors| Daqian ShaoAshkan SoleymaniFrancesco QuinzanMarta Kwiatkowska
|links| http://arxiv.org/abs/2405.08498v1 |
|updated| 2024-05-14 10:55:04 UTC |
|summary| A common issue in learning decision-making policies in data-rich settings isspurious correlations in the offline dataset which can be caused by hiddenconfounders. Instrumental variable IV regression which utilises a keyunconfounded variable known as the instrument is a standard technique forlearning causal relationships between confounded action outcome and contextvariables. Most recent IV regression algorithms use a two-stage approach wherea deep neural network DNN estimator learnt in the first stage is directlyplugged into the second stage in which another DNN is used to estimate thecausal effect. Naively plugging the estimator can cause heavy bias in thesecond stage especially when regularisation bias is present in the first stageestimator. We propose DML-IV a non-linear IV regression method that reducesthe bias in two-stage IV regressions and effectively learns high-performingpolicies. We derive a novel learning objective to reduce bias and design theDML-IV algorithm following the double/debiased machine learning DMLframework. The learnt DML-IV estimator has strong convergence rate andON-1/2 suboptimality guarantees that match those when the dataset isunconfounded. DML-IV outperforms state-of-the-art IV regression methods on IVregression benchmarks and learns high-performing policies in the presence ofinstruments. |


| Item |Content|
| --- |---|
|idx| 2405.08484v1 |
|title| Universal replication of chaotic characteristics by classical and quantum machine learning |
|authors| Sheng-Chen BaiShi-Ju Ran
|links| http://arxiv.org/abs/2405.08484v1 |
|updated| 2024-05-14 10:12:47 UTC |
|summary| Replicating chaotic characteristics of non-linear dynamics by machinelearning ML has recently drawn wide attentions. In this work we propose thata ML model trained to predict the state one-step-ahead from several latesthistoric states can accurately replicate the bifurcation diagram and theLyapunov exponents of discrete dynamic systems. The characteristics fordifferent values of the hyper-parameters are captured universally by a singleML model while the previous works considered training the ML modelindependently by fixing the hyper-parameters to be specific values. Ourbenchmarks on the one- and two-dimensional Logistic maps show that variationalquantum circuit can reproduce the long-term characteristics with higheraccuracy than the long short-term memory a well-recognized classical MLmodel. Our work reveals an essential difference between the ML for the chaoticcharacteristics and that for standard tasks from the perspective of therelation between performance and model complexity. Our results suggest thatquantum circuit model exhibits potential advantages on mitigating over-fittingachieving higher accuracy and stability. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2405.08573v1 |
|title| ViSTooth: A Visualization Framework for Tooth Segmentation on Panoramic Radiograph |
|authors| Shenji ZhuMiaoxin HuTianya PanYue HongBin LiZhiguang ZhouTing Xu
|links| http://arxiv.org/abs/2405.08573v1 |
|updated| 2024-05-14 13:10:54 UTC |
|summary| Tooth segmentation is a key step for computer aided diagnosis of dentaldiseases. Numerous machine learning models have been employed for toothsegmentation on dental panoramic radiograph. However it is a difficult task toachieve accurate tooth segmentation due to complex tooth shapes diverse toothcategories and incomplete sample set for machine learning. In this paper wepropose ViSTooth a visualization framework for tooth segmentation on dentalpanoramic radiograph. First we employ Mask R-CNN to conduct preliminary toothsegmentation and a set of domain metrics are proposed to estimate the accuracyof the segmented teeth including tooth shape tooth position and tooth angle.Then we represent the teeth with high-dimensional vectors and visualize theirdistribution in a low-dimensional space in which experts can easily observethose teeth with specific metrics. Further we expand the sample set with theexpert-specified teeth and train the tooth segmentation model iteratively.Finally we conduct case study and expert study to demonstrate theeffectiveness and usability of our ViSTooth in aiding experts to implementaccurate tooth segmentation guided by expert knowledge. |


| Item |Content|
| --- |---|
|idx| 2405.08527v1 |
|title| EEG-Features for Generalized Deepfake Detection |
|authors| Arian BeckmannTilman StephaniFelix KlotzscheYonghao ChenSimon M. HofmannArno VillringerMichael GaeblerVadim NikulinSebastian BossePeter EisertAnna Hilsmann
|links| http://arxiv.org/abs/2405.08527v1 |
|updated| 2024-05-14 12:06:44 UTC |
|summary| Since the advent of Deepfakes in digital media the development of robust andreliable detection mechanism is urgently called for. In this study we explorea novel approach to Deepfake detection by utilizing electroencephalographyEEG measured from the neural processing of a human participant who viewed andcategorized Deepfake stimuli from the FaceForensics datset. Thesemeasurements serve as input features to a binary support vector classifiertrained to discriminate between real and manipulated facial images. We examinewhether EEG data can inform Deepfake detection and also if it can provide ageneralized representation capable of identifying Deepfakes beyond the trainingdomain. Our preliminary results indicate that human neural processing signalscan be successfully integrated into Deepfake detection frameworks and hint atthe potential for a generalized neural representation of artifacts in computergenerated faces. Moreover our study provides next steps towards theunderstanding of how digital realism is embedded in the human cognitive systempossibly enabling the development of more realistic digital avatars in thefuture. |


| Item |Content|
| --- |---|
|idx| 2405.08526v1 |
|title| Why Larp?! A Synthesis Paper on Live Action Roleplay in Relation to HCI Research and Practice |
|authors| Karin JohanssonRaquel Breejon RobinsonJon BackSarah Lynne BowmanJames FeyElena Márquez SeguraAnnika WaernKatherine Isbister
|links| http://arxiv.org/abs/2405.08526v1 |
|updated| 2024-05-14 12:06:00 UTC |
|summary| Live action roleplay larp has a wide range of applications and can berelevant in relation to HCI. While there has been research about larp inrelation to topics such as embodied interaction playfulness and futuringpublished in HCI venues since the early 2000s there is not yet a compilationof this knowledge. In this paper we synthesise knowledge about larp andlarp-adjacent work within the domain of HCI. We present a practitioner overviewfrom an expert group of larp researchers the results of a literature reviewand highlight particular larp research exemplars which all work together toshowcase the diverse set of ways that larp can be utilised in relation to HCItopics and research. This paper identifies the need for further discussionstoward establishing best practices for utilising larp in relation to HCIresearch as well as advocating for increased engagement with larps outsideacademia. |


| Item |Content|
| --- |---|
|idx| 2405.08515v1 |
|title| Precarious Experiences: Citizens' Frustrations, Anxieties and Burdens of an Online Welfare Benefit System |
|authors| Colin WatsonAdam W ParnabyAhmed Kharrufa
|links| http://arxiv.org/abs/2405.08515v1 |
|updated| 2024-05-14 11:37:29 UTC |
|summary| There is a significant overlap between people who are supported byincome-related social welfare benefits often in precarious situations andthose who experience greater digital exclusion. We report on a study ofclaimants using the UKs Universal Credit online welfare benefit systemdesigned as and still digital by default. Through data collection involvingremote interviews n11 and online surveys n66 we expose claimants ownlived experiences interacting with this system. The claimants explain howdigital channels can contribute to an imbalance of power and agency at a timewhen their own circumstances mean they have reduced abilities resources andcapacities and where design choices can adversely affect peoples utility toleverage help from their own wider socio-technical ecosystems. We contributeeight recommendations from these accounts to inform the future design anddevelopment of digital welfare benefit systems for this population to reducedigital barriers and harms. |


| Item |Content|
| --- |---|
|idx| 2405.08447v1 |
|title| AI-Resilient Interfaces |
|authors| Elena L. GlassmanZiwei GuJonathan K. Kummerfeld
|links| http://arxiv.org/abs/2405.08447v1 |
|updated| 2024-05-14 09:12:30 UTC |
|summary| AI is powerful but it can make choices that result in objective errorscontextually inappropriate outputs and disliked options. We need AI-resilientinterfaces that help people be resilient to the AI choices that are not rightor not right for them. To support this goal interfaces need to help usersnotice and have the context to appropriately judge those AI choices. Existinghuman-AI interaction guidelines recommend efficient user dismissalmodification or otherwise efficient recovery from AI choices that a user doesnot like. However in order to recover from AI choices the user must noticethem first. This can be difficult. For example when generating summaries oflong documents a systems exclusion of a detail that is critically importantto the user is hard for the user to notice. That detail can be hiding in a wallof text in the original document and the existence of a summary may tempt theuser not to read the original document as carefully. Once noticed judging AIchoices well can also be challenging. The interface may provide very littleinformation that contextualizes the choices and the user may fall back onassumptions when deciding whether to dismiss modify or otherwise recover froman AI choice. Building on prior work this paper defines key aspects ofAI-resilient interfaces illustrated with examples. Designing interfaces forincreased AI-resilience of users will improve AI safety usability andutility. This is especially critical where AI-powered systems are used forcontext- and preference-dominated open-ended AI-assisted tasks like ideatingsummarizing searching sensemaking and the reading and writing of text orcode. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2405.08206v1 |
|title| Beyond Theorems: A Counterexample to Potential Markov Game Criteria |
|authors| Fatemeh FardnoSeyed Majid Zahedi
|links| http://arxiv.org/abs/2405.08206v1 |
|updated| 2024-05-13 21:49:15 UTC |
|summary| There are only limited classes of multi-player stochastic games in whichindependent learning is guaranteed to converge to a Nash equilibrium. Markovpotential games are a key example of such classes. Prior work has outlined setsof sufficient conditions for a stochastic game to qualify as a Markov potentialgame. However these conditions often impose strict limitations on the gamesstructure and tend to be challenging to verify. To address these limitationsMguni et al. 12 introduce a relaxed notion of Markov potential games andoffer an alternative set of necessary conditions for categorizing stochasticgames as potential games. Under these conditions the authors claim that adeterministic Nash equilibrium can be computed efficiently by solving a dualMarkov decision process. In this paper we offer evidence refuting this claimby presenting a counterexample. |


| Item |Content|
| --- |---|
|idx| 2405.07541v2 |
|title| Random walk model that universally generates inverse square Lévy walk by eliminating search cost minimization constraint |
|authors| Shuji ShinoharaDaiki MoritaHayato HiraiRyosuke KuribayashiNobuhito ManomeToru MoriyamaHiroshi OkamotoYoshihiro NakajimaPegio-Yukio GunjiUng-il Chung
|links| http://arxiv.org/abs/2405.07541v2 |
|updated| 2024-05-14 01:44:36 UTC |
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


