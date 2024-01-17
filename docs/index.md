# cs.CL 

| Item |Content|
| --- |---|
|idx| 1 |
|title| Machine Translation Models are Zero-Shot Detectors of Translation Direction |
|authors| Michelle WastlJannis VamvasRico Sennrich
|links| http://arxiv.org/abs/2401.06769v1 |
|updated| 2024-01-12 18:59:02 UTC |
|summary| Detecting the translation direction of parallel text has applications formachine translation training and evaluation but also has forensic applicationssuch as resolving plagiarism or forgery allegations. In this work we explorean unsupervised approach to translation direction detection based on the simplehypothesis thatptexttranslationtextoriginalptextoriginaltexttranslationmotivated by the well-known simplification effect in translationese ormachine-translationese. In experiments with massively multilingual machinetranslation models across 20 translation directions we confirm theeffectiveness of the approach for high-resource language pairs achievingdocument-level accuracies of 82-96 for NMT-produced translations and 60-81for human translations depending on the model used. Code and demo areavailable at https://github.com/ZurichNLP/translation-direction-detection |


| Item |Content|
| --- |---|
|idx| 2 |
|title| Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements |
|authors| Anton VoronovLena WolfMax Ryabinin
|links| http://arxiv.org/abs/2401.06766v1 |
|updated| 2024-01-12 18:58:26 UTC |
|summary| Large language models demonstrate a remarkable capability for learning tosolve new tasks from a few examples. The prompt template or the way the inputexamples are formatted to obtain the prompt is an important yet oftenoverlooked aspect of in-context learning. In this work we conduct acomprehensive study of the template formats influence on the in-contextlearning performance. We evaluate the impact of the prompt template acrossmodels from 770M to 70B parameters and 4 standard classification datasets. Weshow that a poor choice of the template can reduce the performance of thestrongest models and inference methods to a random guess level. Moreimportantly the best templates do not transfer between different setups andeven between models of the same family. Our findings show that the currentlyprevalent approach to evaluation which ignores template selection may givemisleading results due to different templates in different works. As a firststep towards mitigating this issue we propose Template Ensembles thataggregate model predictions across several templates. This simple test-timeaugmentation boosts average performance while being robust to the choice ofrandom set of templates. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| APAR: LLMs Can Do Auto-Parallel Auto-Regressive Decoding |
|authors| Mingdao LiuAohan ZengBowen WangPeng ZhangJie TangYuxiao Dong
|links| http://arxiv.org/abs/2401.06761v1 |
|updated| 2024-01-12 18:50:36 UTC |
|summary| The massive adoption of large language models LLMs demands efficientdeployment strategies. However the auto-regressive decoding process which isfundamental to how most LLMs generate text poses challenges to achieveefficient serving. In this work we introduce a parallel auto-regressivegeneration method. By instruct-tuning on general domain data that containshierarchical structures we enable LLMs to independently plan their generationprocess and perform auto-parallel auto-regressive APAR generationsignificantly reducing the number of generation steps. APAR alone can achieveup to 2x speed-up and when combined with speculative decoding the speed-upcan reach up to 4x. In addition APAR reduces the key-value cache consumptionand attention computation during generation. This leads to a throughputincrease of 20-70 and a latency reduce of 20-35 in high-throughput scenarioscompared to state-of-the-art serving frameworks. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| Navigating the Metrics Maze: Reconciling Score Magnitudes and Accuracies |
|authors| Tom KocmiVilém ZouharChristian FedermannMatt Post
|links| http://arxiv.org/abs/2401.06760v1 |
|updated| 2024-01-12 18:47:40 UTC |
|summary| Ten years ago a single metric BLEU governed progress in machine translationresearch. For better or worse there is no such consensus today andconsequently it is difficult for researchers to develop and retain the kinds ofheuristic intuitions about metric deltas that drove earlier research anddeployment decisions. This paper investigates the dynamic range of a numberof modern metrics in an effort to provide a collective understanding of themeaning of differences in scores both within and among metrics in other wordswe ask what point difference X in metric Y is required between two systems forhumans to notice We conduct our evaluation on a new large dataset ToShip23using it to discover deltas at which metrics achieve system-level differencesthat are meaningful to humans which we measure by pairwise system accuracy. Weadditionally show that this method of establishing delta-accuracy is morestable than the standard use of statistical p-values in regards to testsetsize. Where data size permits we also explore the effect of metric deltas andaccuracy across finer-grained features such as translation direction domainand system closeness. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| Stylometry Analysis of Multi-authored Documents for Authorship and Author Style Change Detection |
|authors| Muhammad Tayyab ZamirMuhammad Asif AyubAsma GulNasir AhmadKashif Ahmad
|links| http://arxiv.org/abs/2401.06752v1 |
|updated| 2024-01-12 18:36:41 UTC |
|summary| In recent years the increasing use of Artificial Intelligence based textgeneration tools has posed new challenges in document provenanceauthentication and authorship detection. However advancements in stylometryhave provided opportunities for automatic authorship and author changedetection in multi-authored documents using style analysis techniques. Styleanalysis can serve as a primary step toward document provenance andauthentication through authorship detection. This paper investigates three keytasks of style analysis: i classification of single and multi-authoreddocuments ii single change detection which involves identifying the pointwhere the author switches and iii multiple author-switching detection inmulti-authored documents. We formulate all three tasks as classificationproblems and propose a merit-based fusion framework that integrates severalstate-of-the-art natural language processing NLP algorithms and weightoptimization techniques. We also explore the potential of special characterswhich are typically removed during pre-processing in NLP applications on theperformance of the proposed methods for these tasks by conducting extensiveexperiments on both cleaned and raw datasets. Experimental results demonstratesignificant improvements over existing solutions for all three tasks on abenchmark dataset. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 1 |
|title| Synthetic Data Generation Framework, Dataset, and Efficient Deep Model for Pedestrian Intention Prediction |
|authors| Muhammad Naveed RiazMaciej WielgoszAbel Garcia RomeraAntonio M. Lopez
|links| http://arxiv.org/abs/2401.06757v1 |
|updated| 2024-01-12 18:44:01 UTC |
|summary| Pedestrian intention prediction is crucial for autonomous driving. Inparticular knowing if pedestrians are going to cross in front of theego-vehicle is core to performing safe and comfortable maneuvers. Creatingaccurate and fast models that predict such intentions from sequential images ischallenging. A factor contributing to this is the lack of datasets with diversecrossing and non-crossing C/NC scenarios. We address this scarceness byintroducing a framework named ARCANE which allows programmatically generatingsynthetic datasets consisting of C/NC video clip samples. As an example we useARCANE to generate a large and diverse dataset named PedSynth. We will show howPedSynth complements widely used real-world datasets such as JAAD and PIE soenabling more accurate models for C/NC prediction. Considering the onboarddeployment of C/NC prediction models we also propose a deep model namedPedGNN which is fast and has a very low memory footprint. PedGNN is based on aGNN-GRU architecture that takes a sequence of pedestrian skeletons as input topredict crossing intentions. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| The Unreasonable Effectiveness of Easy Training Data for Hard Tasks |
|authors| Peter HaseMohit BansalPeter ClarkSarah Wiegreffe
|links| http://arxiv.org/abs/2401.06751v1 |
|updated| 2024-01-12 18:36:29 UTC |
|summary| How can we train models to perform well on hard test data when hard trainingdata is by definition difficult to label correctly This question has beentermed the scalable oversight problem and has drawn increasing attention aslanguage models have continually improved. In this paper we present thesurprising conclusion that current language models often generalize relativelywell from easy to hard data even performing as well as oracle models trainedon hard data. We demonstrate this kind of easy-to-hard generalization usingsimple training methods like in-context learning linear classifier heads andQLoRA for seven different measures of datapoint hardness including sixempirically diverse human hardness measures like grade level and onemodel-based measure loss-based. Furthermore we show that even if one caresmost about model performance on hard data it can be better to collect andtrain on easy data rather than hard data since hard data is generally noisierand costlier to collect. Our experiments use open models up to 70b in size andfour publicly available question-answering datasets with questions ranging indifficulty from 3rd grade science questions to college level STEM questions andgeneral-knowledge trivia. We conclude that easy-to-hard generalization in LMsis surprisingly strong for the tasks studied suggesting the scalable oversightproblem may be easier than previously thought. Our code is available athttps://github.com/allenai/easy-to-hard-generalization |


| Item |Content|
| --- |---|
|idx| 3 |
|title| Using Natural Language Inference to Improve Persona Extraction from Dialogue in a New Domain |
|authors| Alexandra DeLuciaMengjie ZhaoYoshinori MaedaMakoto YodaKeiichi YamadaHiromi Wakaki
|links| http://arxiv.org/abs/2401.06742v1 |
|updated| 2024-01-12 18:25:03 UTC |
|summary| While valuable datasets such as PersonaChat provide a foundation for trainingpersona-grounded dialogue agents they lack diversity in conversational andnarrative settings primarily existing in the real world. To develop dialogueagents with unique personas models are trained to converse given a specificpersona but hand-crafting these persona can be time-consuming thus methodsexist to automatically extract persona information from existingcharacter-specific dialogue. However these persona-extraction models are alsotrained on datasets derived from PersonaChat and struggle to providehigh-quality persona information from conversational settings that do not takeplace in the real world such as the fantasy-focused dataset LIGHT. Creatingnew data to train models on a specific setting is human-intensive thusprohibitively expensive. To address both these issues we introduce a naturallanguage inference method for post-hoc adapting a trained persona extractionmodel to a new setting. We draw inspiration from the literature of dialognatural language inference NLI and devise NLI-reranking methods to extractstructured persona information from dialogue. Compared to existing personaextraction models our method returns higher-quality extracted persona andrequires less human annotation. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| Relying on the Unreliable: The Impact of Language Models' Reluctance to Express Uncertainty |
|authors| Kaitlyn ZhouJena D. HwangXiang RenMaarten Sap
|links| http://arxiv.org/abs/2401.06730v1 |
|updated| 2024-01-12 18:03:30 UTC |
|summary| As natural language becomes the default interface for human-AI interactionthere is a critical need for LMs to appropriately communicate uncertainties indownstream applications. In this work we investigate how LMs incorporateconfidence about their responses via natural language and how downstream usersbehave in response to LM-articulated uncertainties. We examine publiclydeployed models and find that LMs are unable to express uncertainties whenanswering questions even when they produce incorrect responses. LMs can beexplicitly prompted to express confidences but tend to be overconfidentresulting in high error rates on average 47 among confident responses. Wetest the risks of LM overconfidence by running human experiments and show thatusers rely heavily on LM generations whether or not they are marked bycertainty. Lastly we investigate the preference-annotated datasets used inRLHF alignment and find that humans have a bias against texts with uncertainty.Our work highlights a new set of safety harms facing human-LM interactions andproposes design recommendations and mitigating strategies moving forward. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| Reframing Tax Law Entailment as Analogical Reasoning |
|authors| Xinrui ZouMing ZhangNathaniel WeirBenjamin Van DurmeNils Holzenberger
|links| http://arxiv.org/abs/2401.06715v1 |
|updated| 2024-01-12 17:37:07 UTC |
|summary| Statutory reasoning refers to the application of legislative provisions to aseries of case facts described in natural language. We re-frame statutoryreasoning as an analogy task where each instance of the analogy task involvesa combination of two instances of statutory reasoning. This increases thedataset size by two orders of magnitude and introduces an element ofinterpretability. We show that this task is roughly as difficult to NaturalLanguage Processing models as the original task. Finally we come back tostatutory reasoning solving it with a combination of a retrieval mechanism andanalogy models and showing some progress on prior comparable work. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 1 |
|title| Seeing the roads through the trees: A benchmark for modeling spatial dependencies with aerial imagery |
|authors| Caleb RobinsonIsaac CorleyAnthony OrtizRahul DodhiaJuan M. Lavista FerresPeyman Najafirad
|links| http://arxiv.org/abs/2401.06762v1 |
|updated| 2024-01-12 18:50:43 UTC |
|summary| Fully understanding a complex high-resolution satellite or aerial imageryscene often requires spatial reasoning over a broad relevant context. The humanobject recognition system is able to understand object in a scene over along-range relevant context. For example if a human observes an aerial scenethat shows sections of road broken up by tree canopy then they will beunlikely to conclude that the road has actually been broken up into disjointpieces by trees and instead think that the canopy of nearby trees is occludingthe road. However there is limited research being conducted to understandlong-range context understanding of modern machine learning models. In thiswork we propose a road segmentation benchmark dataset Chesapeake Roads SpatialContext RSC for evaluating the spatial long-range context understanding ofgeospatial machine learning models and show how commonly used semanticsegmentation models can fail at this task. For example we show that a U-Nettrained to segment roads from background in aerial imagery achieves an 84recall on unoccluded roads but just 63.5 recall on roads covered by treecanopy despite being trained to model both the same way. We further analyze howthe performance of models changes as the relevant context for a decisionunoccluded roads in our case varies in distance. We release the code toreproduce our experiments and dataset of imagery and masks to encourage futureresearch in this direction -- https://github.com/isaaccorley/ChesapeakeRSC. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| Synthetic Data Generation Framework, Dataset, and Efficient Deep Model for Pedestrian Intention Prediction |
|authors| Muhammad Naveed RiazMaciej WielgoszAbel Garcia RomeraAntonio M. Lopez
|links| http://arxiv.org/abs/2401.06757v1 |
|updated| 2024-01-12 18:44:01 UTC |
|summary| Pedestrian intention prediction is crucial for autonomous driving. Inparticular knowing if pedestrians are going to cross in front of theego-vehicle is core to performing safe and comfortable maneuvers. Creatingaccurate and fast models that predict such intentions from sequential images ischallenging. A factor contributing to this is the lack of datasets with diversecrossing and non-crossing C/NC scenarios. We address this scarceness byintroducing a framework named ARCANE which allows programmatically generatingsynthetic datasets consisting of C/NC video clip samples. As an example we useARCANE to generate a large and diverse dataset named PedSynth. We will show howPedSynth complements widely used real-world datasets such as JAAD and PIE soenabling more accurate models for C/NC prediction. Considering the onboarddeployment of C/NC prediction models we also propose a deep model namedPedGNN which is fast and has a very low memory footprint. PedGNN is based on aGNN-GRU architecture that takes a sequence of pedestrian skeletons as input topredict crossing intentions. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| Solving the Discretised Multiphase Flow Equations with Interface Capturing on Structured Grids Using Machine Learning Libraries |
|authors| Boyang ChenClaire E. HeaneyJefferson L. M. A. GomesOmar K. MatarChristopher C. Pain
|links| http://arxiv.org/abs/2401.06755v1 |
|updated| 2024-01-12 18:42:42 UTC |
|summary| This paper solves the multiphase flow equations with interface capturingusing the AI4PDEs approach Artificial Intelligence for Partial DifferentialEquations. The solver within AI4PDEs uses tools from machine learning MLlibraries to solve exactly partial differential equations PDEs that havebeen discretised using numerical methods. Convolutional layers can be used toexpress the discretisations as a neural network whose weights are determinedby the numerical method rather than by training. To solve the system amultigrid solver is implemented through a neural network with a U-Netarchitecture. Immiscible two-phase flow is modelled by the 3D incompressibleNavier-Stokes equations with surface tension and advection of a volume fractionfield which describes the interface between the fluids. A new compressivealgebraic volume-of-fluids method is introduced based on a residualformulation using Petrov-Galerkin for accuracy and designed with AI4PDEs inmind. High-order finite-element based schemes are chosen to model a collapsingwater column and a rising bubble. Results compare well with experimental dataand other numerical results from the literature demonstrating that for thefirst time finite element discretisations of multiphase flows can be solvedusing the neural network solver from the AI4PDEs approach. A benefit ofexpressing numerical discretisations as neural networks is that the code canrun without modification on CPUs GPUs or the latest accelerators designedespecially to run AI codes. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| The Unreasonable Effectiveness of Easy Training Data for Hard Tasks |
|authors| Peter HaseMohit BansalPeter ClarkSarah Wiegreffe
|links| http://arxiv.org/abs/2401.06751v1 |
|updated| 2024-01-12 18:36:29 UTC |
|summary| How can we train models to perform well on hard test data when hard trainingdata is by definition difficult to label correctly This question has beentermed the scalable oversight problem and has drawn increasing attention aslanguage models have continually improved. In this paper we present thesurprising conclusion that current language models often generalize relativelywell from easy to hard data even performing as well as oracle models trainedon hard data. We demonstrate this kind of easy-to-hard generalization usingsimple training methods like in-context learning linear classifier heads andQLoRA for seven different measures of datapoint hardness including sixempirically diverse human hardness measures like grade level and onemodel-based measure loss-based. Furthermore we show that even if one caresmost about model performance on hard data it can be better to collect andtrain on easy data rather than hard data since hard data is generally noisierand costlier to collect. Our experiments use open models up to 70b in size andfour publicly available question-answering datasets with questions ranging indifficulty from 3rd grade science questions to college level STEM questions andgeneral-knowledge trivia. We conclude that easy-to-hard generalization in LMsis surprisingly strong for the tasks studied suggesting the scalable oversightproblem may be easier than previously thought. Our code is available athttps://github.com/allenai/easy-to-hard-generalization |


| Item |Content|
| --- |---|
|idx| 5 |
|title| A deep implicit-explicit minimizing movement method for option pricing in jump-diffusion models |
|authors| Emmanuil H. GeorgoulisAntonis PapapantoleonCostas Smaragdakis
|links| http://arxiv.org/abs/2401.06740v1 |
|updated| 2024-01-12 18:21:01 UTC |
|summary| We develop a novel deep learning approach for pricing European basket optionswritten on assets that follow jump-diffusion dynamics. The option pricingproblem is formulated as a partial integro-differential equation which isapproximated via a new implicit-explicit minimizing movement time-steppingapproach involving approximation by deep residual-type Artificial NeuralNetworks ANNs for each time step. The integral operator is discretized viatwo different approaches: a a sparse-grid Gauss--Hermite approximationfollowing localised coordinate axes arising from singular value decompositionsand b an ANN-based high-dimensional special-purpose quadrature rule.Crucially the proposed ANN is constructed to ensure the asymptotic behavior ofthe solution for large values of the underlyings and also leads to consistentoutputs with respect to a priori known qualitative properties of the solution.The performance and robustness with respect to the dimension of the methods areassessed in a series of numerical experiments involving the Mertonjump-diffusion model. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 1 |
|title| Seeing the roads through the trees: A benchmark for modeling spatial dependencies with aerial imagery |
|authors| Caleb RobinsonIsaac CorleyAnthony OrtizRahul DodhiaJuan M. Lavista FerresPeyman Najafirad
|links| http://arxiv.org/abs/2401.06762v1 |
|updated| 2024-01-12 18:50:43 UTC |
|summary| Fully understanding a complex high-resolution satellite or aerial imageryscene often requires spatial reasoning over a broad relevant context. The humanobject recognition system is able to understand object in a scene over along-range relevant context. For example if a human observes an aerial scenethat shows sections of road broken up by tree canopy then they will beunlikely to conclude that the road has actually been broken up into disjointpieces by trees and instead think that the canopy of nearby trees is occludingthe road. However there is limited research being conducted to understandlong-range context understanding of modern machine learning models. In thiswork we propose a road segmentation benchmark dataset Chesapeake Roads SpatialContext RSC for evaluating the spatial long-range context understanding ofgeospatial machine learning models and show how commonly used semanticsegmentation models can fail at this task. For example we show that a U-Nettrained to segment roads from background in aerial imagery achieves an 84recall on unoccluded roads but just 63.5 recall on roads covered by treecanopy despite being trained to model both the same way. We further analyze howthe performance of models changes as the relevant context for a decisionunoccluded roads in our case varies in distance. We release the code toreproduce our experiments and dataset of imagery and masks to encourage futureresearch in this direction -- https://github.com/isaaccorley/ChesapeakeRSC. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| Synthetic Data Generation Framework, Dataset, and Efficient Deep Model for Pedestrian Intention Prediction |
|authors| Muhammad Naveed RiazMaciej WielgoszAbel Garcia RomeraAntonio M. Lopez
|links| http://arxiv.org/abs/2401.06757v1 |
|updated| 2024-01-12 18:44:01 UTC |
|summary| Pedestrian intention prediction is crucial for autonomous driving. Inparticular knowing if pedestrians are going to cross in front of theego-vehicle is core to performing safe and comfortable maneuvers. Creatingaccurate and fast models that predict such intentions from sequential images ischallenging. A factor contributing to this is the lack of datasets with diversecrossing and non-crossing C/NC scenarios. We address this scarceness byintroducing a framework named ARCANE which allows programmatically generatingsynthetic datasets consisting of C/NC video clip samples. As an example we useARCANE to generate a large and diverse dataset named PedSynth. We will show howPedSynth complements widely used real-world datasets such as JAAD and PIE soenabling more accurate models for C/NC prediction. Considering the onboarddeployment of C/NC prediction models we also propose a deep model namedPedGNN which is fast and has a very low memory footprint. PedGNN is based on aGNN-GRU architecture that takes a sequence of pedestrian skeletons as input topredict crossing intentions. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| Scalable 3D Panoptic Segmentation With Superpoint Graph Clustering |
|authors| Damien RobertHugo RaguetLoic Landrieu
|links| http://arxiv.org/abs/2401.06704v1 |
|updated| 2024-01-12 17:10:52 UTC |
|summary| We introduce a highly efficient method for panoptic segmentation of large 3Dpoint clouds by redefining this task as a scalable graph clustering problem.This approach can be trained using only local auxiliary tasks therebyeliminating the resource-intensive instance-matching step during training.Moreover our formulation can easily be adapted to the superpoint paradigmfurther increasing its efficiency. This allows our model to process scenes withmillions of points and thousands of objects in a single inference. Our methodcalled SuperCluster achieves a new state-of-the-art panoptic segmentationperformance for two indoor scanning datasets: 50.1 PQ 7.8 for S3DISArea5 and 58.7 PQ 25.2 for ScanNetV2. We also set the firststate-of-the-art for two large-scale mobile mapping benchmarks: KITTI-360 andDALES. With only 209k parameters our model is over 30 times smaller thanthe best-competing method and trains up to 15 times faster. Our code andpretrained models are available athttps://github.com/drprojects/superpoint_transformer. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| Embedded Planogram Compliance Control System |
|authors| M. Erkin YücelSerkan TopaloğluCem Ünsalan
|links| http://arxiv.org/abs/2401.06690v1 |
|updated| 2024-01-12 16:54:26 UTC |
|summary| The retail sector presents several open and challenging problems that couldbenefit from advanced pattern recognition and computer vision techniques. Onesuch critical challenge is planogram compliance control. In this study wepropose a complete embedded system to tackle this issue. Our system consists offour key components as image acquisition and transfer via stand-alone embeddedcamera module object detection via computer vision and deep learning methodsworking on single board computers planogram compliance control method againworking on single board computers and energy harvesting and power managementblock to accompany the embedded camera modules. The image acquisition andtransfer block is implemented on the ESP-EYE camera module. The objectdetection block is based on YOLOv5 as the deep learning method and localfeature extraction. We implement these methods on Raspberry Pi 4 NVIDIA JetsonOrin Nano and NVIDIA Jetson AGX Orin as single board computers. The planogramcompliance control block utilizes sequence alignment through a modifiedNeedleman-Wunsch algorithm. This block is also working along with the objectdetection block on the same single board computers. The energy harvesting andpower management block consists of solar and RF energy harvesting modules withsuitable battery pack for operation. We tested the proposed embedded planogramcompliance control system on two different datasets to provide valuableinsights on its strengths and weaknesses. The results show that our methodachieves F1 scores of 0.997 and 1.0 in object detection and planogramcompliance control blocks respectively. Furthermore we calculated that thecomplete embedded system can work in stand-alone form up to two years based onbattery. This duration can be further extended with the integration of theproposed solar and RF energy harvesting options. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| Decoupling Pixel Flipping and Occlusion Strategy for Consistent XAI Benchmarks |
|authors| Stefan BlücherJohanna VielhabenNils Strodthoff
|links| http://arxiv.org/abs/2401.06654v1 |
|updated| 2024-01-12 16:01:17 UTC |
|summary| Feature removal is a central building block for eXplainable AI XAI bothfor occlusion-based explanations Shapley values as well as their evaluationpixel flipping PF. However occlusion strategies can vary significantly fromsimple mean replacement up to inpainting with state-of-the-art diffusionmodels. This ambiguity limits the usefulness of occlusion-based approaches. Forexample PF benchmarks lead to contradicting rankings. This is amplified bycompeting PF measures: Features are either removed starting with mostinfluential first MIF or least influential first LIF. This study proposestwo complementary perspectives to resolve this disagreement problem. Firstlywe address the common criticism of occlusion-based XAI that artificial sampleslead to unreliable model evaluations. We propose to measure the reliability bythe Reference-Out-of-Model-Scope OMS score. The R-OMS score enables asystematic comparison of occlusion strategies and resolves the disagreementproblem by grouping consistent PF rankings. Secondly we show that theinsightfulness of MIF and LIF is conversely dependent on the R-OMS score. Toleverage this we combine the MIF and LIF measures into the symmetric relevancegain SRG measure. This breaks the inherent connection to the underlyingocclusion strategy and leads to consistent rankings. This resolves thedisagreement problem which we verify for a set of 40 different occlusionstrategies. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 1 |
|title| Noise-adaptive (Accelerated) Stochastic Heavy-Ball Momentum |
|authors| Anh DangReza BabanezhadSharan Vaswani
|links| http://arxiv.org/abs/2401.06738v1 |
|updated| 2024-01-12 18:17:28 UTC |
|summary| We analyze the convergence of stochastic heavy ball SHB momentum in thesmooth strongly-convex setting. Kidambi et al. 2018 show that SHB withsmall mini-batches cannot attain an accelerated rate of convergence even forquadratics and conjecture that the practical gain of SHB is a by-product ofmini-batching. We substantiate this claim by showing that SHB can obtain anaccelerated rate when the mini-batch size is larger than some threshold. Inparticular for strongly-convex quadratics with condition number kappa weprove that SHB with the standard step-size and momentum parameters results inan Oleftexp-fracTsqrtkappa  sigma right convergence ratewhere T is the number of iterations and sigma2 is the variance in thestochastic gradients. To ensure convergence to the minimizer we propose amulti-stage approach that results in a noise-adaptiveOleftexpleft-fracTsqrtkappa right  fracsigmaTrightrate. For general strongly-convex functions we use the averaginginterpretation of SHB along with exponential step-sizes to prove anOleftexpleft-fracTkappa right  fracsigma2T rightconvergence to the minimizer in a noise-adaptive manner. Finally weempirically demonstrate the effectiveness of the proposed algorithms. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| Valid causal inference with unobserved confounding in high-dimensional settings |
|authors| Niloofar MoosaviTetiana GorbachXavier de Luna
|links| http://arxiv.org/abs/2401.06564v1 |
|updated| 2024-01-12 13:21:20 UTC |
|summary| Various methods have recently been proposed to estimate causal effects withconfidence intervals that are uniformly valid over a set of data generatingprocesses when high-dimensional nuisance models are estimated bypost-model-selection or machine learning estimators. These methods typicallyrequire that all the confounders are observed to ensure identification of theeffects. We contribute by showing how valid semiparametric inference can beobtained in the presence of unobserved confounders and high-dimensionalnuisance models. We propose uncertainty intervals which allow for unobservedconfounding and show that the resulting inference is valid when the amount ofunobserved confounding is small relative to the sample size the latter isformalized in terms of convergence rates. Simulation experiments illustrate thefinite sample properties of the proposed intervals and investigate analternative procedure that improves the empirical coverage of the intervalswhen the amount of unobserved confounding is large. Finally a case study onthe effect of smoking during pregnancy on birth weight is used to illustratethe use of the methods introduced to perform a sensitivity analysis tounobserved confounding. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| Boosting Causal Additive Models |
|authors| Maximilian KertelNadja Klein
|links| http://arxiv.org/abs/2401.06523v1 |
|updated| 2024-01-12 11:43:11 UTC |
|summary| We present a boosting-based method to learn additive Structural EquationModels SEMs from observational data with a focus on the theoretical aspectsof determining the causal order among variables. We introduce a family of scorefunctions based on arbitrary regression techniques for which we establishnecessary conditions to consistently favor the true causal ordering. Ouranalysis reveals that boosting with early stopping meets these criteria andthus offers a consistent score function for causal orderings. To address thechallenges posed by high-dimensional data sets we adapt our approach through acomponent-wise gradient descent in the space of additive SEMs. Our simulationstudy underlines our theoretical results for lower dimensions and demonstratesthat our high-dimensional adaptation is competitive with state-of-the-artmethods. In addition it exhibits robustness with respect to the choice of thehyperparameters making the procedure easy to tune. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| A comprehensive framework for multi-fidelity surrogate modeling with noisy data: a gray-box perspective |
|authors| Katerina GiannoukouStefano MarelliBruno Sudret
|links| http://arxiv.org/abs/2401.06447v1 |
|updated| 2024-01-12 08:37:41 UTC |
|summary| Computer simulations a.k.a. white-box models are more indispensable thanever to model intricate engineering systems. However computational modelsalone often fail to fully capture the complexities of reality. When physicalexperiments are accessible though it is of interest to enhance the incompleteinformation offered by computational models. Gray-box modeling is concernedwith the problem of merging information from data-driven a.k.a. black-boxmodels and white-box i.e. physics-based models. In this paper we propose toperform this task by using multi-fidelity surrogate models MFSMs. A MFSMintegrates information from models with varying computational fidelity into anew surrogate model. The multi-fidelity surrogate modeling framework we proposehandles noise-contaminated data and is able to estimate the underlyingnoise-free high-fidelity function. Our methodology emphasizes on deliveringprecise estimates of the uncertainty in its predictions in the form ofconfidence and prediction intervals by quantitatively incorporating thedifferent types of uncertainty that affect the problem arising frommeasurement noise and from lack of knowledge due to the limited experimentaldesign budget on both the high- and low-fidelity models. Applied to gray-boxmodeling our MFSM framework treats noisy experimental data as thehigh-fidelity and the white-box computational models as their low-fidelitycounterparts. The effectiveness of our methodology is showcased throughsynthetic examples and a wind turbine application. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| Faster Sampling without Isoperimetry via Diffusion-based Monte Carlo |
|authors| Xunpeng HuangDifan ZouHanze DongYian MaTong Zhang
|links| http://arxiv.org/abs/2401.06325v1 |
|updated| 2024-01-12 02:33:57 UTC |
|summary| To sample from a general target distribution p_propto e-f_ beyond theisoperimetric condition Huang et al. 2023 proposed to perform samplingthrough reverse diffusion giving rise to Diffusion-based Monte Carlo DMC.Specifically DMC follows the reverse SDE of a diffusion process thattransforms the target distribution to the standard Gaussian utilizing anon-parametric score estimation. However the original DMC algorithmencountered high gradient complexity resulting in an exponential dependency onthe error tolerance epsilon of the obtained samples. In this paper wedemonstrate that the high complexity of DMC originates from its redundantdesign of score estimation and proposed a more efficient algorithm calledRS-DMC based on a novel recursive score estimation method. In particular wefirst divide the entire diffusion process into multiple segments and thenformulate the score estimation step at any time step as a series ofinterconnected mean estimation and sampling subproblems accordingly which arecorrelated in a recursive manner. Importantly we show that with a properdesign of the segment decomposition all sampling subproblems will only need totackle a strongly log-concave distribution which can be very efficient tosolve using the Langevin-based samplers with a provably rapid convergence rate.As a result we prove that the gradient complexity of RS-DMC only has aquasi-polynomial dependency on epsilon which significantly improvesexponential gradient complexity in Huang et al. 2023. Furthermore undercommonly used dissipative conditions our algorithm is provably much fasterthan the popular Langevin-based algorithms. Our algorithm design andtheoretical framework illuminate a novel direction for addressing samplingproblems which could be of broader applicability in the community. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 1 |
|title| Relying on the Unreliable: The Impact of Language Models' Reluctance to Express Uncertainty |
|authors| Kaitlyn ZhouJena D. HwangXiang RenMaarten Sap
|links| http://arxiv.org/abs/2401.06730v1 |
|updated| 2024-01-12 18:03:30 UTC |
|summary| As natural language becomes the default interface for human-AI interactionthere is a critical need for LMs to appropriately communicate uncertainties indownstream applications. In this work we investigate how LMs incorporateconfidence about their responses via natural language and how downstream usersbehave in response to LM-articulated uncertainties. We examine publiclydeployed models and find that LMs are unable to express uncertainties whenanswering questions even when they produce incorrect responses. LMs can beexplicitly prompted to express confidences but tend to be overconfidentresulting in high error rates on average 47 among confident responses. Wetest the risks of LM overconfidence by running human experiments and show thatusers rely heavily on LM generations whether or not they are marked bycertainty. Lastly we investigate the preference-annotated datasets used inRLHF alignment and find that humans have a bias against texts with uncertainty.Our work highlights a new set of safety harms facing human-LM interactions andproposes design recommendations and mitigating strategies moving forward. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| Resource-Efficient Gesture Recognition using Low-Resolution Thermal Camera via Spiking Neural Networks and Sparse Segmentation |
|authors| Ali SafaWout MommenLars Keuninckx
|links| http://arxiv.org/abs/2401.06563v1 |
|updated| 2024-01-12 13:20:01 UTC |
|summary| This work proposes a novel approach for hand gesture recognition using aninexpensive low-resolution 24 x 32 thermal sensor processed by a SpikingNeural Network SNN followed by Sparse Segmentation and feature-based gestureclassification via Robust Principal Component Analysis R-PCA. Compared to theuse of standard RGB cameras the proposed system is insensitive to lightingvariations while being significantly less expensive compared to high-frequencyradars time-of-flight cameras and high-resolution thermal sensors previouslyused in literature. Crucially this paper shows that the innovative use of therecently proposed Monostable Multivibrator MMV neural networks as a new classof SNN achieves more than one order of magnitude smaller memory and computecomplexity compared to deep learning approaches while reaching a top gesturerecognition accuracy of 93.9 using a 5-class thermal camera dataset acquiredin a car cabin within an automotive context. Our dataset is released forhelping future research. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| AttributionScanner: A Visual Analytics System for Metadata-Free Data-Slicing Based Model Validation |
|authors| Xiwei XuanJorge Piazentin OnoLiang GouKwan-Liu MaLiu Ren
|links| http://arxiv.org/abs/2401.06462v1 |
|updated| 2024-01-12 09:17:32 UTC |
|summary| Data slice-finding is an emerging technique for evaluating machine learningmodels. It works by identifying subgroups within a specified dataset thatexhibit poor performance often defined by distinct feature sets ormeta-information. However in the context of unstructured image data dataslice-finding poses two notable challenges: it requires additional metadata --a laborious and costly requirement and also demands non-trivial efforts forinterpreting the root causes of the underperformance within data slices. Toaddress these challenges we introduce AttributionScanner an innovativehuman-in-the-loop Visual Analytics VA system designed for data-slicing-basedmachine learning ML model validation. Our approach excels in identifyinginterpretable data slices employing explainable features extracted through thelens of Explainable AI XAI techniques and removing the necessity foradditional metadata of textual annotations or cross-model embeddings.AttributionScanner demonstrates proficiency in pinpointing critical modelissues including spurious correlations and mislabeled data. Our novel VAinterface visually summarizes data slices enabling users to gather insightsinto model behavior patterns effortlessly. Furthermore our framework closesthe ML Development Cycle by empowering domain experts to address model issuesby using a cutting-edge neural network regularization technique. The efficacyof AttributionScanner is underscored through two prototype use caseselucidating its substantial effectiveness in model validation forvision-centric tasks. Our approach paves the way for ML researchers andpractitioners to drive interpretable model validation in a data-efficient wayultimately leading to more reliable and accurate models. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| Why Doesn't Microsoft Let Me Sleep? How Automaticity of Windows Updates Impacts User Autonomy |
|authors| Sanju AhujaRidhi JainJyoti Kumar
|links| http://arxiv.org/abs/2401.06413v1 |
|updated| 2024-01-12 07:18:22 UTC |
|summary| Automating the user away has been designated as a dark pattern inliterature for performing tasks without user consent or confirmation. Howeverlimited studies have been reported on how users experience the sense ofautonomy when digital systems fully or partially bypass consent. More researchis required to understand what makes automaticity a threat to autonomy. Toaddress this gap a qualitative interview study with 10 users was conducted toinvestigate the user experience of Microsoft Windows updates. It was found thatten design features of Windows updates impact the autonomy experience. For eachdesign feature the contextual factors which influence its impact on autonomywere also noted. The findings of this paper can help designers understand theethical concerns posed by automaticity in design and identify measures tomitigate these concerns. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| Understanding whole-body inter-personal dynamics between two players using neural Granger causality as the explainable AI (XAI) |
|authors| Ryota TakamidoChiharu SuzukiJun OtaHiroki Nakamoto
|links| http://arxiv.org/abs/2401.06412v1 |
|updated| 2024-01-12 07:17:56 UTC |
|summary| Background: Simultaneously focusing on intra- and inter-individual bodydynamics and elucidating how these affect each other will help understand humaninter-personal coordination behavior. However this association has not beeninvestigated previously owing to difficulties in analyzing complex causalrelations among several body components.To address this issue this studyproposes a new analytical framework that attempts to understand the underlyingcausal structures behind each joint movement of individual baseball playersusing neural Granger causality NGC as the explainable AI. Methods: In the NGCanalysis causal relationships were defined as the size of the weightparameters of the first layer of a machine-learning model trained to predictthe future state of a specific time-series variable. To verify the approach ina practical context we conducted an experiment with 16 pairs of expertbaseball pitchers and batters input datasets with 27 joint resultant velocitydata joints of 13 pitchers and 14 batters were generated and used for modeltraining.Results: NGC analysis revealed significant causal relations amongintra- and inter-individual body components such as the batters hands having acausal effect from the pitchers throwing arm. Remarkably although thecausality from the batters body to pitchers body is much lower than thereverse it is significantly correlated with batter performance outcomes.Conclusions: The above results suggest the effectiveness of NGC analysis forunderstanding whole-body inter-personal coordination dynamics and that of theAI technique as a new approach for analyzing complex human behavior from adifferent perspective than conventional techniques. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 1 |
|title| A Semantic-Aware Multiple Access Scheme for Distributed, Dynamic 6G-Based Applications |
|authors| Hamidreza MazandaraniMasoud ShokrnezhadTarik Taleb
|links| http://arxiv.org/abs/2401.06308v1 |
|updated| 2024-01-12 00:32:38 UTC |
|summary| The emergence of the semantic-aware paradigm presents opportunities forinnovative services especially in the context of 6G-based applications.Although significant progress has been made in semantic extraction techniquesthe incorporation of semantic information into resource allocationdecision-making is still in its early stages lacking consideration of therequirements and characteristics of future systems. In response this paperintroduces a novel formulation for the problem of multiple access to thewireless spectrum. It aims to optimize the utilization-fairness trade-offusing the alpha-fairness metric while accounting for user data correlationby introducing the concepts of self- and assisted throughputs. Initially theproblem is analyzed to identify its optimal solution. Subsequently aSemantic-Aware Multi-Agent Double and Dueling Deep Q-Learning SAMA-D3QLtechnique is proposed. This method is grounded in Model-free Multi-Agent DeepReinforcement Learning MADRL enabling the user equipment to autonomouslymake decisions regarding wireless spectrum access based solely on their localindividual observations. The efficiency of the proposed technique is evaluatedthrough two scenarios: single-channel and multi-channel. The findingsillustrate that across a spectrum of alpha values association matricesand channels SAMA-D3QL consistently outperforms alternative approaches. Thisestablishes it as a promising candidate for facilitating the realization offuture federated dynamically evolving applications. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| XGBoost Learning of Dynamic Wager Placement for In-Play Betting on an Agent-Based Model of a Sports Betting Exchange |
|authors| Chawin TerawongDave Cliff
|links| http://arxiv.org/abs/2401.06086v1 |
|updated| 2024-01-11 18:03:17 UTC |
|summary| We present first results from the use of XGBoost a highly effective machinelearning ML method within the Bristol Betting Exchange BBE an open-sourceagent-based model ABM designed to simulate a contemporary sports-bettingexchange with in-play betting during track-racing events such as horse races.We use the BBE ABM and its array of minimally-simple bettor-agents as asynthetic data generator which feeds into our XGBoost ML system with theintention that XGBoost discovers profitable dynamic betting strategies bylearning from the more profitable bets made by the BBE bettor-agents. Afterthis XGBoost training which results in one or more decision trees abettor-agent with a betting strategy determined by the XGBoost-learned decisiontrees is added to the BBE ABM and made to bet on a sequence of races undervarious conditions and betting-market scenarios with profitability serving asthe primary metric of comparison and evaluation. Our initial findings presentedhere show that XGBoost trained in this way can indeed learn profitable bettingstrategies and can generalise to learn strategies that outperform each of theset of strategies used for creation of the training data. To foster furtherresearch and enhancements the complete version of our extended BBE includingthe XGBoost integration has been made freely available as an open-sourcerelease on GitHub. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| Confidence-Based Curriculum Learning for Multi-Agent Path Finding |
|authors| Thomy PhanJoseph DriscollJustin RombergSven Koenig
|links| http://arxiv.org/abs/2401.05860v1 |
|updated| 2024-01-11 12:11:24 UTC |
|summary| A wide range of real-world applications can be formulated as Multi-Agent PathFinding MAPF problem where the goal is to find collision-free paths formultiple agents with individual start and goal locations. State-of-the-art MAPFsolvers are mainly centralized and depend on global information which limitstheir scalability and flexibility regarding changes or new maps that wouldrequire expensive replanning. Multi-agent reinforcement learning MARL offersan alternative way by learning decentralized policies that can generalize overa variety of maps. While there exist some prior works that attempt to connectboth areas the proposed techniques are heavily engineered and very complex dueto the integration of many mechanisms that limit generality and are expensiveto use. We argue that much simpler and general approaches are needed to bringthe areas of MARL and MAPF closer together with significantly lower costs. Inthis paper we propose Confidence-based Auto-Curriculum for Team UpdateStability CACTUS as a lightweight MARL approach to MAPF. CACTUS defines asimple reverse curriculum scheme where the goal of each agent is randomlyplaced within an allocation radius around the agents start location. Theallocation radius increases gradually as all agents improve which is assessedby a confidence-based measure. We evaluate CACTUS in various maps of differentsizes obstacle densities and numbers of agents. Our experiments demonstratebetter performance and generalization capabilities than state-of-the-art MARLapproaches with less than 600000 trainable parameters which is less than 5of the neural network size of current MARL approaches to MAPF. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| Multi-Agent Based Simulation for Investigating Electric Vehicle Adoption and Its Impacts on Electricity Distribution Grids and CO2 Emissions |
|authors| Kristoffer ChristensenZheng Grace MaBo Nørregaard Jørgensen
|links| http://dx.doi.org/10.1007/978-3-031-48652-4_1 |
|updated| 2024-01-11 11:59:13 UTC |
|summary| Electric vehicles are expected to significantly contribute to CO2-eq.emissions reduction but the increasing number of EVs also introduceschal-lenges to the energy system and to what extent it contributes toachieving cli-mate goals remains unknown. Static modeling and assumption-basedsimula-tions have been used for such investigation but they cannot capture therealistic ecosystem dynamics. To fill the gap this paper investigates theimpacts of two adoption curves of private EVs on the electricity distributiongrids and national climate goals. This paper develops a multi-agent basedsimulation with two adoption curves the Traditional EV charging strategyvarious EV models driv-ing patterns and CO2-eq. emission data to capture thefull ecosystem dynamics during a long-term period from 2020 to 2032. The Danish2030 climate goal and a Danish distribution network with 126 residentialconsumers are chosen as the case study. The results show that both EV adoptioncurves of 1 million and 775k EVs by 2030 will not satisfy the Danish climategoal of reducing transport sector emissions by 30 by 2030. The results alsoshow that the current resi-dential electricity distribution grids cannot handlethe load from increasing EVs. The first grid overload will occur in 2031around 16 and 24 months later for the 1 million and 775k EVs adopted by 2030with a 67 share of EVs in the grid. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| Designing Heterogeneous LLM Agents for Financial Sentiment Analysis |
|authors| Frank Xing
|links| http://arxiv.org/abs/2401.05799v1 |
|updated| 2024-01-11 10:06:42 UTC |
|summary| Large language models LLMs have drastically changed the possible ways todesign intelligent systems shifting the focuses from massive data acquisitionand new modeling training to human alignment and strategical elicitation of thefull potential of existing pre-trained models. This paradigm shift however isnot fully realized in financial sentiment analysis FSA due to thediscriminative nature of this task and a lack of prescriptive knowledge of howto leverage generative models in such a context. This study investigates theeffectiveness of the new paradigm i.e. using LLMs without fine-tuning forFSA. Rooted in Minskys theory of mind and emotions a design framework withheterogeneous LLM agents is proposed. The framework instantiates specializedagents using prior domain knowledge of the types of FSA errors and reasons onthe aggregated agent discussions. Comprehensive evaluation on FSA datasets showthat the framework yields better accuracies especially when the discussionsare substantial. This study contributes to the design foundations and paves newavenues for LLMs-based FSA. Implications on business and management are alsodiscussed. |


