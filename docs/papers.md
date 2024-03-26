# cs.CL 

| Item |Content|
| --- |---|
|idx| 2403.16804v1 |
|title| TEI2GO: A Multilingual Approach for Fast Temporal Expression Identification |
|authors| Hugo SousaRicardo CamposAlípio Jorge
|links| http://dx.doi.org/10.1145/3583780.3615130 |
|updated| 2024-03-25 14:23:03 UTC |
|summary| Temporal expression identification is crucial for understanding texts writtenin natural language. Although highly effective systems such as HeidelTimeexist their limited runtime performance hampers adoption in large-scaleapplications and production environments. In this paper we introduce theTEI2GO models matching HeidelTimes effectiveness but with significantlyimproved runtime supporting six languages and achieving state-of-the-artresults in four of them. To train the TEI2GO models we used a combination ofmanually annotated reference corpus and developed Professor HeidelTime acomprehensive weakly labeled corpus of news texts annotated with HeidelTime.This corpus comprises a total of 138069 documents over six languages with1050921 temporal expressions the largest open-source annotated dataset fortemporal expression identification to date. By describing how the models wereproduced we aim to encourage the research community to further explorerefine and extend the set of models to additional languages and domains. Codeannotations and models are openly available for community exploration and use.The models are conveniently on HuggingFace for seamless integration andapplication. |


| Item |Content|
| --- |---|
|idx| 2403.16792v1 |
|title| Iterative Refinement of Project-Level Code Context for Precise Code Generation with Compiler Feedback |
|authors| Zhangqian BiYao WanZheng WangHongyu ZhangBatu GuanFangxin LuZili ZhangYulei SuiXuanhua ShiHai Jin
|links| http://arxiv.org/abs/2403.16792v1 |
|updated| 2024-03-25 14:07:27 UTC |
|summary| Large language models LLMs have shown remarkable progress in automated codegeneration. Yet incorporating LLM-based code generation into real-lifesoftware projects poses challenges as the generated code may contain errors inAPI usage class data structure or missing project-specific information. Asmuch of this project-specific context cannot fit into the prompts of LLMs wemust find ways to allow the model to explore the project-level code context. Tothis end this paper puts forward a novel approach termed ProCoder whichiteratively refines the project-level code context for precise code generationguided by the compiler feedback. In particular ProCoder first leveragescompiler techniques to identify a mismatch between the generated code and theprojects context. It then iteratively aligns and fixes the identified errorsusing information extracted from the code repository. We integrate ProCoderwith two representative LLMs i.e. GPT-3.5-Turbo and Code Llama 13B andapply it to Python code generation. Experimental results show that ProCodersignificantly improves the vanilla LLMs by over 80 in generating codedependent on project context and consistently outperforms the existingretrieval-based code generation baselines. |


| Item |Content|
| --- |---|
|idx| 2403.16777v1 |
|title| Can Machine Translation Bridge Multilingual Pretraining and Cross-lingual Transfer Learning? |
|authors| Shaoxiong JiTimothee MickusVincent SegonneJörg Tiedemann
|links| http://arxiv.org/abs/2403.16777v1 |
|updated| 2024-03-25 13:53:04 UTC |
|summary| Multilingual pretraining and fine-tuning have remarkably succeeded in variousnatural language processing tasks. Transferring representations from onelanguage to another is especially crucial for cross-lingual learning. One canexpect machine translation objectives to be well suited to fostering suchcapabilities as they involve the explicit alignment of semantically equivalentsentences from different languages. This paper investigates the potentialbenefits of employing machine translation as a continued training objective toenhance language representation learning bridging multilingual pretraining andcross-lingual applications. We study this question through two lenses: aquantitative evaluation of the performance of existing models and an analysisof their latent representations. Our results show that contrary toexpectations machine translation as the continued training fails to enhancecross-lingual representation learning in multiple cross-lingual naturallanguage understanding tasks. We conclude that explicit sentence-levelalignment in the cross-lingual scenario is detrimental to cross-lingualtransfer pretraining which has important implications for future cross-lingualtransfer studies. We furthermore provide evidence through similarity measuresand investigation of parameters that this lack of positive influence is due tooutput separability -- which we argue is of use for machine translation butdetrimental elsewhere. |


| Item |Content|
| --- |---|
|idx| 2403.16771v1 |
|title| Synthetic Data Generation and Joint Learning for Robust Code-Mixed Translation |
|authors| KartikSanjana SoniAnoop KunchukuttanTanmoy ChakrabortyMd Shad Akhtar
|links| http://arxiv.org/abs/2403.16771v1 |
|updated| 2024-03-25 13:50:11 UTC |
|summary| The widespread online communication in a modern multilingual world hasprovided opportunities to blend more than one language aka code-mixedlanguage in a single utterance. This has resulted a formidable challenge forthe computational models due to the scarcity of annotated data and presence ofnoise. A potential solution to mitigate the data scarcity problem inlow-resource setup is to leverage existing data in resource-rich languagethrough translation. In this paper we tackle the problem of code-mixedHinglish and Bengalish to English machine translation. First wesynthetically develop HINMIX a parallel corpus of Hinglish to English with4.2M sentence pairs. Subsequently we propose RCMT a robust perturbationbased joint-training model that learns to handle noise in the real-worldcode-mixed text by parameter sharing across clean and noisy words. Further weshow the adaptability of RCMT in a zero-shot setup for Bengalish to Englishtranslation. Our evaluation and comprehensive analyses qualitatively andquantitatively demonstrate the superiority of RCMT over state-of-the-artcode-mixed and robust translation methods. |


| Item |Content|
| --- |---|
|idx| 2403.16702v1 |
|title| ProCQA: A Large-scale Community-based Programming Question Answering Dataset for Code Search |
|authors| Zehan LiJianfei ZhangChuantao YinYuanxin OuyangWenge Rong
|links| http://arxiv.org/abs/2403.16702v1 |
|updated| 2024-03-25 12:34:33 UTC |
|summary| Retrieval-based code question answering seeks to match user queries innatural language to relevant code snippets. Previous approaches typically relyon pretraining models using crafted bi-modal and uni-modal datasets to aligntext and code representations. In this paper we introduce ProCQA alarge-scale programming question answering dataset extracted from theStackOverflow community offering naturally structured mixed-modal QA pairs. Tovalidate its effectiveness we propose a modality-agnostic contrastivepre-training approach to improve the alignment of text and code representationsof current code language models. Compared to previous models that primarilyemploy bimodal and unimodal pairs extracted from CodeSearchNet forpre-training our model exhibits significant performance improvements across awide range of code retrieval benchmarks. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2403.16812v1 |
|title| Towards Human-AI Deliberation: Design and Evaluation of LLM-Empowered Deliberative AI for AI-Assisted Decision-Making |
|authors| Shuai MaQiaoyi ChenXinru WangChengbo ZhengZhenhui PengMing YinXiaojuan Ma
|links| http://arxiv.org/abs/2403.16812v1 |
|updated| 2024-03-25 14:34:06 UTC |
|summary| In AI-assisted decision-making humans often passively review AIs suggestionand decide whether to accept or reject it as a whole. In such a paradigmhumans are found to rarely trigger analytical thinking and face difficulties incommunicating the nuances of conflicting opinions to the AI when disagreementsoccur. To tackle this challenge we propose Human-AI Deliberation a novelframework to promote human reflection and discussion on conflicting human-AIopinions in decision-making. Based on theories in human deliberation thisframework engages humans and AI in dimension-level opinion elicitationdeliberative discussion and decision updates. To empower AI with deliberativecapabilities we designed Deliberative AI which leverages large languagemodels LLMs as a bridge between humans and domain-specific models to enableflexible conversational interactions and faithful information provision. Anexploratory evaluation on a graduate admissions task shows that Deliberative AIoutperforms conventional explainable AI XAI assistants in improving humansappropriate reliance and task performance. Based on a mixed-methods analysis ofparticipant behavior perception user experience and open-ended feedback wedraw implications for future AI-assisted decision tool design. |


| Item |Content|
| --- |---|
|idx| 2403.16809v1 |
|title| An LLM-Based Digital Twin for Optimizing Human-in-the Loop Systems |
|authors| Hanqing YangMarie SiewCarlee Joe-Wong
|links| http://arxiv.org/abs/2403.16809v1 |
|updated| 2024-03-25 14:32:28 UTC |
|summary| The increasing prevalence of Cyber-Physical Systems and the Internet ofThings CPS-IoT applications and Foundation Models are enabling newapplications that leverage real-time control of the environment. For examplereal-time control of Heating Ventilation and Air-Conditioning HVAC systemscan reduce its usage when not needed for the comfort of human occupants hencereducing energy consumption. Collecting real-time feedback on human preferencesin such human-in-the-loop HITL systems however is difficult in practice. Wepropose the use of large language models LLMs to deal with the challenges ofdynamic environments and difficult-to-obtain data in CPS optimization. In thispaper we present a case study that employs LLM agents to mimic the behaviorsand thermal preferences of various population groups e.g. young families theelderly in a shopping mall. The aggregated thermal preferences are integratedinto an agent-in-the-loop based reinforcement learning algorithm AitL-RL whichemploys the LLM as a dynamic simulation of the physical environment to learnhow to balance between energy savings and occupant comfort. Our results showthat LLMs are capable of simulating complex population movements within largeopen spaces. Besides AitL-RL demonstrates superior performance compared to thepopular existing policy of set point control suggesting that adaptive andpersonalized decision-making is critical for efficient optimization in CPS-IoTapplications. Through this case study we demonstrate the potential ofintegrating advanced Foundation Models like LLMs into CPS-IoT to enhance systemadaptability and efficiency. The projects code can be found on our GitHubrepository. |


| Item |Content|
| --- |---|
|idx| 2403.16808v1 |
|title| Navigating the EU AI Act: A Methodological Approach to Compliance for Safety-critical Products |
|authors| J. KellyS. Ali ZafarL. HeidemannJ. ZacchiD. EspinozaN. Mata
|links| http://arxiv.org/abs/2403.16808v1 |
|updated| 2024-03-25 14:32:18 UTC |
|summary| In December 2023 the European Parliament provisionally agreed on the EU AIAct. This unprecedented regulatory framework for AI systems lays out guidelinesto ensure the safety legality and trustworthiness of AI products. This paperpresents a methodology for interpreting the EU AI Act requirements forhigh-risk AI systems by leveraging product quality models. We first propose anextended product quality model for AI systems incorporating attributesrelevant to the Act not covered by current quality models. We map the Actrequirements to relevant quality attributes with the goal of refining them intomeasurable characteristics. We then propose a contract-based approach to derivetechnical requirements at the stakeholder level. This facilitates thedevelopment and assessment of AI systems that not only adhere to establishedquality standards but also comply with the regulatory requirements outlined inthe Act for high-risk including safety-critical AI systems. We demonstratethe applicability of this methodology on an exemplary automotive supply chainuse case where several stakeholders interact to achieve EU AI Act compliance. |


| Item |Content|
| --- |---|
|idx| 2403.16798v1 |
|title| Cluster-Based Normalization Layer for Neural Networks |
|authors| Bilal FayeHanane AzzagMustapha Lebbah
|links| http://arxiv.org/abs/2403.16798v1 |
|updated| 2024-03-25 14:17:38 UTC |
|summary| Deep learning faces significant challenges during the training of neuralnetworks including internal covariate shift label shift vanishing/explodinggradients overfitting and computational complexity. While conventionalnormalization methods such as Batch Normalization aim to tackle some of theseissues they often depend on assumptions that constrain their adaptability.Mixture Normalization faces computational hurdles in its pursuit of handlingmultiple Gaussian distributions.  This paper introduces Cluster-Based Normalization CB-Norm in two variants -Supervised Cluster-Based Normalization SCB-Norm and UnsupervisedCluster-Based Normalization UCB-Norm - proposing a groundbreaking one-stepnormalization approach. CB-Norm leverages a Gaussian mixture model tospecifically address challenges related to gradient stability and learningacceleration.  For SCB-Norm a supervised variant the novel mechanism involves introducingpredefined data partitioning termed clusters to normalize activations basedon the assigned cluster. This cluster-driven approach creates a space thatconforms to a Gaussian mixture model. On the other hand UCB-Norm anunsupervised counterpart dynamically clusters neuron activations duringtraining adapting to task-specific challenges without relying on predefineddata partitions clusters. This dual approach ensures flexibility inaddressing diverse learning scenarios.  CB-Norm innovatively uses a one-step normalization approach where parametersof each mixture component cluster in activation space serve as weights fordeep neural networks. This adaptive clustering process tackles both clusteringand resolution of deep neural network tasks concurrently during trainingsignifying a notable advancement in the field. |


| Item |Content|
| --- |---|
|idx| 2403.16782v1 |
|title| The Anatomy of Adversarial Attacks: Concept-based XAI Dissection |
|authors| Georgii MikriukovGesina SchwalbeFranz MotzkusKorinna Bade
|links| http://arxiv.org/abs/2403.16782v1 |
|updated| 2024-03-25 13:57:45 UTC |
|summary| Adversarial attacks AAs pose a significant threat to the reliability androbustness of deep neural networks. While the impact of these attacks on modelpredictions has been extensively studied their effect on the learnedrepresentations and concepts within these models remains largely unexplored. Inthis work we perform an in-depth analysis of the influence of AAs on theconcepts learned by convolutional neural networks CNNs using eXplainableartificial intelligence XAI techniques. Through an extensive set ofexperiments across various network architectures and targeted AA techniques weunveil several key findings. First AAs induce substantial alterations in theconcept composition within the feature space introducing new concepts ormodifying existing ones. Second the adversarial perturbation itself can belinearly decomposed into a set of latent vector components with a subset ofthese being responsible for the attacks success. Notably we discover thatthese components are target-specific i.e. are similar for a given targetclass throughout different AA techniques and starting classes. Our findingsprovide valuable insights into the nature of AAs and their impact on learnedrepresentations paving the way for the development of more robust andinterpretable deep learning models as well as effective defenses againstadversarial threats. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2403.16809v1 |
|title| An LLM-Based Digital Twin for Optimizing Human-in-the Loop Systems |
|authors| Hanqing YangMarie SiewCarlee Joe-Wong
|links| http://arxiv.org/abs/2403.16809v1 |
|updated| 2024-03-25 14:32:28 UTC |
|summary| The increasing prevalence of Cyber-Physical Systems and the Internet ofThings CPS-IoT applications and Foundation Models are enabling newapplications that leverage real-time control of the environment. For examplereal-time control of Heating Ventilation and Air-Conditioning HVAC systemscan reduce its usage when not needed for the comfort of human occupants hencereducing energy consumption. Collecting real-time feedback on human preferencesin such human-in-the-loop HITL systems however is difficult in practice. Wepropose the use of large language models LLMs to deal with the challenges ofdynamic environments and difficult-to-obtain data in CPS optimization. In thispaper we present a case study that employs LLM agents to mimic the behaviorsand thermal preferences of various population groups e.g. young families theelderly in a shopping mall. The aggregated thermal preferences are integratedinto an agent-in-the-loop based reinforcement learning algorithm AitL-RL whichemploys the LLM as a dynamic simulation of the physical environment to learnhow to balance between energy savings and occupant comfort. Our results showthat LLMs are capable of simulating complex population movements within largeopen spaces. Besides AitL-RL demonstrates superior performance compared to thepopular existing policy of set point control suggesting that adaptive andpersonalized decision-making is critical for efficient optimization in CPS-IoTapplications. Through this case study we demonstrate the potential ofintegrating advanced Foundation Models like LLMs into CPS-IoT to enhance systemadaptability and efficiency. The projects code can be found on our GitHubrepository. |


| Item |Content|
| --- |---|
|idx| 2403.16798v1 |
|title| Cluster-Based Normalization Layer for Neural Networks |
|authors| Bilal FayeHanane AzzagMustapha Lebbah
|links| http://arxiv.org/abs/2403.16798v1 |
|updated| 2024-03-25 14:17:38 UTC |
|summary| Deep learning faces significant challenges during the training of neuralnetworks including internal covariate shift label shift vanishing/explodinggradients overfitting and computational complexity. While conventionalnormalization methods such as Batch Normalization aim to tackle some of theseissues they often depend on assumptions that constrain their adaptability.Mixture Normalization faces computational hurdles in its pursuit of handlingmultiple Gaussian distributions.  This paper introduces Cluster-Based Normalization CB-Norm in two variants -Supervised Cluster-Based Normalization SCB-Norm and UnsupervisedCluster-Based Normalization UCB-Norm - proposing a groundbreaking one-stepnormalization approach. CB-Norm leverages a Gaussian mixture model tospecifically address challenges related to gradient stability and learningacceleration.  For SCB-Norm a supervised variant the novel mechanism involves introducingpredefined data partitioning termed clusters to normalize activations basedon the assigned cluster. This cluster-driven approach creates a space thatconforms to a Gaussian mixture model. On the other hand UCB-Norm anunsupervised counterpart dynamically clusters neuron activations duringtraining adapting to task-specific challenges without relying on predefineddata partitions clusters. This dual approach ensures flexibility inaddressing diverse learning scenarios.  CB-Norm innovatively uses a one-step normalization approach where parametersof each mixture component cluster in activation space serve as weights fordeep neural networks. This adaptive clustering process tackles both clusteringand resolution of deep neural network tasks concurrently during trainingsignifying a notable advancement in the field. |


| Item |Content|
| --- |---|
|idx| 2403.16790v1 |
|title| Iso-Diffusion: Improving Diffusion Probabilistic Models Using the Isotropy of the Additive Gaussian Noise |
|authors| Dilum FernandoDhananjaya jayasundaraRoshan GodaliyaddaChaminda BandaraParakrama EkanayakeVijitha Herath
|links| http://arxiv.org/abs/2403.16790v1 |
|updated| 2024-03-25 14:05:52 UTC |
|summary| Denoising Diffusion Probabilistic Models DDPMs have accomplished much inthe realm of generative AI. Despite their high performance there is room forimprovement especially in terms of sample fidelity by utilizing statisticalproperties that impose structural integrity such as isotropy. Minimizing themean squared error between the additive and predicted noise alone does notimpose constraints on the predicted noise to be isotropic. Thus we weremotivated to utilize the isotropy of the additive noise as a constraint on theobjective function to enhance the fidelity of DDPMs. Our approach is simple andcan be applied to any DDPM variant. We validate our approach by presentingexperiments conducted on four synthetic 2D datasets as well as on unconditionalimage generation. As demonstrated by the results the incorporation of thisconstraint improves the fidelity metrics Precision and Density for the 2Ddatasets as well as for the unconditional image generation. |


| Item |Content|
| --- |---|
|idx| 2403.16782v1 |
|title| The Anatomy of Adversarial Attacks: Concept-based XAI Dissection |
|authors| Georgii MikriukovGesina SchwalbeFranz MotzkusKorinna Bade
|links| http://arxiv.org/abs/2403.16782v1 |
|updated| 2024-03-25 13:57:45 UTC |
|summary| Adversarial attacks AAs pose a significant threat to the reliability androbustness of deep neural networks. While the impact of these attacks on modelpredictions has been extensively studied their effect on the learnedrepresentations and concepts within these models remains largely unexplored. Inthis work we perform an in-depth analysis of the influence of AAs on theconcepts learned by convolutional neural networks CNNs using eXplainableartificial intelligence XAI techniques. Through an extensive set ofexperiments across various network architectures and targeted AA techniques weunveil several key findings. First AAs induce substantial alterations in theconcept composition within the feature space introducing new concepts ormodifying existing ones. Second the adversarial perturbation itself can belinearly decomposed into a set of latent vector components with a subset ofthese being responsible for the attacks success. Notably we discover thatthese components are target-specific i.e. are similar for a given targetclass throughout different AA techniques and starting classes. Our findingsprovide valuable insights into the nature of AAs and their impact on learnedrepresentations paving the way for the development of more robust andinterpretable deep learning models as well as effective defenses againstadversarial threats. |


| Item |Content|
| --- |---|
|idx| 2403.16776v1 |
|title| Diff-Def: Diffusion-Generated Deformation Fields for Conditional Atlases |
|authors| Sophie StarckVasiliki Sideri-LampretsaBernhard KainzMartin MentenTamara MuellerDaniel Rueckert
|links| http://arxiv.org/abs/2403.16776v1 |
|updated| 2024-03-25 13:52:48 UTC |
|summary| Anatomical atlases are widely used for population analysis. Conditionalatlases target a particular sub-population defined via certain conditions e.g.demographics or pathologies and allow for the investigation of fine-grainedanatomical differences - such as morphological changes correlated with age.Existing approaches use either registration-based methods that are unable tohandle large anatomical variations or generative models which can suffer fromtraining instabilities and hallucinations. To overcome these limitations weuse latent diffusion models to generate deformation fields which transform ageneral population atlas into one representing a specific sub-population. Bygenerating a deformation field and registering the conditional atlas to aneighbourhood of images we ensure structural plausibility and avoidhallucinations which can occur during direct image synthesis. We compare ourmethod to several state-of-the-art atlas generation methods in experimentsusing 5000 brain as well as whole-body MR images from UK Biobank. Our methodgenerates highly realistic atlases with smooth transformations and highanatomical fidelity outperforming the baselines. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2403.16803v1 |
|title| Exploiting Priors from 3D Diffusion Models for RGB-Based One-Shot View Planning |
|authors| Sicong PanLiren JinXuying HuangCyrill StachnissMarija PopovićMaren Bennewitz
|links| http://arxiv.org/abs/2403.16803v1 |
|updated| 2024-03-25 14:21:49 UTC |
|summary| Object reconstruction is relevant for many autonomous robotic tasks thatrequire interaction with the environment. A key challenge in such scenarios isplanning view configurations to collect informative measurements forreconstructing an initially unknown object. One-shot view planning enablesefficient data collection by predicting view configurations and planning theglobally shortest path connecting all views at once. However geometric priorsabout the object are required to conduct one-shot view planning. In this workwe propose a novel one-shot view planning approach that utilizes the powerful3D generation capabilities of diffusion models as priors. By incorporating suchgeometric priors into our pipeline we achieve effective one-shot view planningstarting with only a single RGB image of the object to be reconstructed. Ourplanning experiments in simulation and real-world setups indicate that ourapproach balances well between object reconstruction quality and movement cost. |


| Item |Content|
| --- |---|
|idx| 2403.16794v1 |
|title| CurbNet: Curb Detection Framework Based on LiDAR Point Cloud Segmentation |
|authors| Guoyang ZhaoFulong MaYuxuan LiuWeiqing QiMing Liu
|links| http://arxiv.org/abs/2403.16794v1 |
|updated| 2024-03-25 14:13:09 UTC |
|summary| Curb detection is an important function in intelligent driving and can beused to determine drivable areas of the road. However curbs are difficult todetect due to the complex road environment. This paper introduces CurbNet anovel framework for curb detection leveraging point cloud segmentation.Addressing the dearth of comprehensive curb datasets and the absence of 3Dannotations we have developed the 3D-Curb dataset encompassing 7100 frameswhich represents the largest and most categorically diverse collection of curbpoint clouds currently available. Recognizing that curbs are primarilycharacterized by height variations our approach harnesses spatially-rich 3Dpoint clouds for training. To tackle the challenges presented by the unevendistribution of curb features on the xy-plane and their reliance on z-axishigh-frequency features we introduce the multi-scale and channel attentionMSCA module a bespoke solution designed to optimize detection performance.Moreover we propose an adaptive weighted loss function group specificallyformulated to counteract the imbalance in the distribution of curb point cloudsrelative to other categories. Our extensive experimentation on 2 major datasetshas yielded results that surpass existing benchmarks set by leading curbdetection and point cloud segmentation models. By integrating multi-clusteringand curve fitting techniques in our post-processing stage we havesubstantially reduced noise in curb detection thereby enhancing precision to0.8744. Notably CurbNet has achieved an exceptional average metrics of over0.95 at a tolerance of just 0.15m thereby establishing a new benchmark.Furthermore corroborative real-world experiments and dataset analyzes mutuallyvalidate each other solidifying CurbNets superior detection proficiency andits robust generalizability. |


| Item |Content|
| --- |---|
|idx| 2403.16788v1 |
|title| HPL-ESS: Hybrid Pseudo-Labeling for Unsupervised Event-based Semantic Segmentation |
|authors| Linglin JingYiming DingYunpeng GaoZhigang WangXu YanDong WangGerald SchaeferHui FangBin ZhaoXuelong Li
|links| http://arxiv.org/abs/2403.16788v1 |
|updated| 2024-03-25 14:02:33 UTC |
|summary| Event-based semantic segmentation has gained popularity due to its capabilityto deal with scenarios under high-speed motion and extreme lighting conditionswhich cannot be addressed by conventional RGB cameras. Since it is hard toannotate event data previous approaches rely on event-to-image reconstructionto obtain pseudo labels for training. However this will inevitably introducenoise and learning from noisy pseudo labels especially when generated from asingle source may reinforce the errors. This drawback is also calledconfirmation bias in pseudo-labeling. In this paper we propose a novel hybridpseudo-labeling framework for unsupervised event-based semantic segmentationHPL-ESS to alleviate the influence of noisy pseudo labels. In particular wefirst employ a plain unsupervised domain adaptation framework as our baselinewhich can generate a set of pseudo labels through self-training. Then weincorporate offline event-to-image reconstruction into the framework andobtain another set of pseudo labels by predicting segmentation maps on thereconstructed images. A noisy label learning strategy is designed to mix thetwo sets of pseudo labels and enhance the quality. Moreover we propose a softprototypical alignment module to further improve the consistency of targetdomain features. Extensive experiments show that our proposed methodoutperforms existing state-of-the-art methods by a large margin on theDSEC-Semantic dataset 5.88 accuracy 10.32 mIoU which even surpassesseveral supervised methods. |


| Item |Content|
| --- |---|
|idx| 2403.16782v1 |
|title| The Anatomy of Adversarial Attacks: Concept-based XAI Dissection |
|authors| Georgii MikriukovGesina SchwalbeFranz MotzkusKorinna Bade
|links| http://arxiv.org/abs/2403.16782v1 |
|updated| 2024-03-25 13:57:45 UTC |
|summary| Adversarial attacks AAs pose a significant threat to the reliability androbustness of deep neural networks. While the impact of these attacks on modelpredictions has been extensively studied their effect on the learnedrepresentations and concepts within these models remains largely unexplored. Inthis work we perform an in-depth analysis of the influence of AAs on theconcepts learned by convolutional neural networks CNNs using eXplainableartificial intelligence XAI techniques. Through an extensive set ofexperiments across various network architectures and targeted AA techniques weunveil several key findings. First AAs induce substantial alterations in theconcept composition within the feature space introducing new concepts ormodifying existing ones. Second the adversarial perturbation itself can belinearly decomposed into a set of latent vector components with a subset ofthese being responsible for the attacks success. Notably we discover thatthese components are target-specific i.e. are similar for a given targetclass throughout different AA techniques and starting classes. Our findingsprovide valuable insights into the nature of AAs and their impact on learnedrepresentations paving the way for the development of more robust andinterpretable deep learning models as well as effective defenses againstadversarial threats. |


| Item |Content|
| --- |---|
|idx| 2403.16776v1 |
|title| Diff-Def: Diffusion-Generated Deformation Fields for Conditional Atlases |
|authors| Sophie StarckVasiliki Sideri-LampretsaBernhard KainzMartin MentenTamara MuellerDaniel Rueckert
|links| http://arxiv.org/abs/2403.16776v1 |
|updated| 2024-03-25 13:52:48 UTC |
|summary| Anatomical atlases are widely used for population analysis. Conditionalatlases target a particular sub-population defined via certain conditions e.g.demographics or pathologies and allow for the investigation of fine-grainedanatomical differences - such as morphological changes correlated with age.Existing approaches use either registration-based methods that are unable tohandle large anatomical variations or generative models which can suffer fromtraining instabilities and hallucinations. To overcome these limitations weuse latent diffusion models to generate deformation fields which transform ageneral population atlas into one representing a specific sub-population. Bygenerating a deformation field and registering the conditional atlas to aneighbourhood of images we ensure structural plausibility and avoidhallucinations which can occur during direct image synthesis. We compare ourmethod to several state-of-the-art atlas generation methods in experimentsusing 5000 brain as well as whole-body MR images from UK Biobank. Our methodgenerates highly realistic atlases with smooth transformations and highanatomical fidelity outperforming the baselines. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2403.16688v1 |
|title| Optimal convex $M$-estimation via score matching |
|authors| Oliver Y. FengYu-Chun KaoMin XuRichard J. Samworth
|links| http://arxiv.org/abs/2403.16688v1 |
|updated| 2024-03-25 12:23:19 UTC |
|summary| In the context of linear regression we construct a data-driven convex lossfunction with respect to which empirical risk minimisation yields optimalasymptotic variance in the downstream estimation of the regressioncoefficients. Our semiparametric approach targets the best decreasingapproximation of the derivative of the log-density of the noise distribution.At the population level this fitting process is a nonparametric extension ofscore matching corresponding to a log-concave projection of the noisedistribution with respect to the Fisher divergence. The procedure iscomputationally efficient and we prove that our procedure attains the minimalasymptotic covariance among all convex M-estimators. As an example of anon-log-concave setting for Cauchy errors the optimal convex loss function isHuber-like and our procedure yields an asymptotic efficiency greater than 0.87relative to the oracle maximum likelihood estimator of the regressioncoefficients that uses knowledge of this error distribution in this sense weobtain robustness without sacrificing much efficiency. Numerical experimentsconfirm the practical merits of our proposal. |


| Item |Content|
| --- |---|
|idx| 2403.16681v1 |
|title| A note on generalization bounds for losses with finite moments |
|authors| Borja Rodríguez-GálvezOmar RivasplataRagnar ThobabenMikael Skoglund
|links| http://arxiv.org/abs/2403.16681v1 |
|updated| 2024-03-25 12:15:55 UTC |
|summary| This paper studies the truncation method from Alquier 1 to derivehigh-probability PAC-Bayes bounds for unbounded losses with heavy tails.Assuming that the p-th moment is bounded the resulting bounds interpolatebetween a slow rate 1 / sqrtn when p2 and a fast rate 1 / n when pto infty and the loss is essentially bounded. Moreover the paper derives ahigh-probability PAC-Bayes bound for losses with a bounded variance. This boundhas an exponentially better dependence on the confidence parameter and thedependency measure than previous bounds in the literature. Finally the paperextends all results to guarantees in expectation and single-draw PAC-Bayes. Inorder to so it obtains analogues of the PAC-Bayes fast rate bound for boundedlosses from 2 in these settings. |


| Item |Content|
| --- |---|
|idx| 2403.16523v1 |
|title| Causal Discovery from Poisson Branching Structural Causal Model Using High-Order Cumulant with Path Analysis |
|authors| Jie QiaoYu XiangZhengming ChenRuichu CaiZhifeng Hao
|links| http://arxiv.org/abs/2403.16523v1 |
|updated| 2024-03-25 08:06:08 UTC |
|summary| Count data naturally arise in many fields such as finance neuroscience andepidemiology and discovering causal structure among count data is a crucialtask in various scientific and industrial scenarios. One of the most commoncharacteristics of count data is the inherent branching structure described bya binomial thinning operator and an independent Poisson distribution thatcaptures both branching and noise. For instance in a population countscenario mortality and immigration contribute to the count where survivalfollows a Bernoulli distribution and immigration follows a Poissondistribution. However causal discovery from such data is challenging due tothe non-identifiability issue: a single causal pair is Markov equivalent i.e.Xrightarrow Y and Yrightarrow X are distributed equivalent. Fortunatelyin this work we found that the causal order from X to its child Y isidentifiable if X is a root vertex and has at least two directed paths toY or the ancestor of X with the most directed path to X has a directedpath to Y without passing X. Specifically we propose a Poisson BranchingStructure Causal Model PB-SCM and perform a path analysis on PB-SCM usinghigh-order cumulants. Theoretical results establish the connection between thepath and cumulant and demonstrate that the path information can be obtainedfrom the cumulant. With the path information causal order is identifiableunder some graphical conditions. A practical algorithm for learning causalstructure under PB-SCM is proposed and the experiments demonstrate and verifythe effectiveness of the proposed method. |


| Item |Content|
| --- |---|
|idx| 2403.16459v1 |
|title| On the rates of convergence for learning with convolutional neural networks |
|authors| Yunfei YangHan FengDing-Xuan Zhou
|links| http://arxiv.org/abs/2403.16459v1 |
|updated| 2024-03-25 06:42:02 UTC |
|summary| We study the approximation and learning capacities of convolutional neuralnetworks CNNs. Our first result proves a new approximation bound for CNNswith certain constraint on the weights. Our second result gives a new analysison the covering number of feed-forward neural networks which include CNNs asspecial cases. The analysis carefully takes into account the size of theweights and hence gives better bounds than existing literature in somesituations. Using these two results we are able to derive rates of convergencefor estimators based on CNNs in many learning problems. In particular weestablish minimax optimal convergence rates of the least squares based on CNNsfor learning smooth functions in the nonparametric regression setting. Forbinary classification we derive convergence rates for CNN classifiers withhinge loss and logistic loss. It is also shown that the obtained rates areminimax optimal in several settings. |


| Item |Content|
| --- |---|
|idx| 2403.16377v1 |
|title| Real-time Adaptation for Condition Monitoring Signal Prediction using Label-aware Neural Processes |
|authors| Seokhyun ChungRaed Al Kontar
|links| http://arxiv.org/abs/2403.16377v1 |
|updated| 2024-03-25 02:47:29 UTC |
|summary| Building a predictive model that rapidly adapts to real-time conditionmonitoring CM signals is critical for engineering systems/units.Unfortunately many current methods suffer from a trade-off betweenrepresentation power and agility in online settings. For instance parametricmethods that assume an underlying functional form for CM signals facilitateefficient online prediction updates. However this simplification leads tovulnerability to model specifications and an inability to capture complexsignals. On the other hand approaches based on over-parameterized ornon-parametric models can excel at explaining complex nonlinear signals butreal-time updates for such models pose a challenging task. In this paper wepropose a neural process-based approach that addresses this trade-off. Itencodes available observations within a CM signal into a representation spaceand then reconstructs the signals history and evolution for prediction. Oncetrained the model can encode an arbitrary number of observations withoutrequiring retraining enabling on-the-spot real-time predictions along withquantified uncertainty and can be readily updated as more online data isgathered. Furthermore our model is designed to incorporate qualitativeinformation i.e. labels from individual units. This integration not onlyenhances individualized predictions for each unit but also enables jointinference for both signals and their associated labels. Numerical studies onboth synthetic and real-world data in reliability engineering highlight theadvantageous features of our model in real-time adaptation enhanced signalprediction with uncertainty quantification and joint prediction for labels andsignals. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2403.16812v1 |
|title| Towards Human-AI Deliberation: Design and Evaluation of LLM-Empowered Deliberative AI for AI-Assisted Decision-Making |
|authors| Shuai MaQiaoyi ChenXinru WangChengbo ZhengZhenhui PengMing YinXiaojuan Ma
|links| http://arxiv.org/abs/2403.16812v1 |
|updated| 2024-03-25 14:34:06 UTC |
|summary| In AI-assisted decision-making humans often passively review AIs suggestionand decide whether to accept or reject it as a whole. In such a paradigmhumans are found to rarely trigger analytical thinking and face difficulties incommunicating the nuances of conflicting opinions to the AI when disagreementsoccur. To tackle this challenge we propose Human-AI Deliberation a novelframework to promote human reflection and discussion on conflicting human-AIopinions in decision-making. Based on theories in human deliberation thisframework engages humans and AI in dimension-level opinion elicitationdeliberative discussion and decision updates. To empower AI with deliberativecapabilities we designed Deliberative AI which leverages large languagemodels LLMs as a bridge between humans and domain-specific models to enableflexible conversational interactions and faithful information provision. Anexploratory evaluation on a graduate admissions task shows that Deliberative AIoutperforms conventional explainable AI XAI assistants in improving humansappropriate reliance and task performance. Based on a mixed-methods analysis ofparticipant behavior perception user experience and open-ended feedback wedraw implications for future AI-assisted decision tool design. |


| Item |Content|
| --- |---|
|idx| 2403.16795v1 |
|title| "We Have No Idea How Models will Behave in Production until Production": How Engineers Operationalize Machine Learning |
|authors| Shreya ShankarRolando GarciaJoseph M HellersteinAditya G Parameswaran
|links| http://dx.doi.org/10.1145/3653697 |
|updated| 2024-03-25 14:13:43 UTC |
|summary| Organizations rely on machine learning engineers MLEs to deploy models andmaintain ML pipelines in production. Due to models extensive reliance on freshdata the operationalization of machine learning or MLOps requires MLEs tohave proficiency in data science and engineering. When considered holisticallythe job seems staggering -- how do MLEs do MLOps and what are theirunaddressed challenges To address these questions we conductedsemi-structured ethnographic interviews with 18 MLEs working on variousapplications including chatbots autonomous vehicles and finance. We findthat MLEs engage in a workflow of i data preparation ii experimentationiii evaluation throughout a multi-staged deployment and iv continualmonitoring and response. Throughout this workflow MLEs collaborate extensivelywith data scientists product stakeholders and one another supplementingroutine verbal exchanges with communication tools ranging from Slack toorganization-wide ticketing and reporting systems. We introduce the 3Vs ofMLOps: velocity visibility and versioning -- three virtues of successful MLdeployments that MLEs learn to balance and grow as they mature. Finally wediscuss design implications and opportunities for future work. |


| Item |Content|
| --- |---|
|idx| 2403.16760v1 |
|title| As Good As A Coin Toss Human detection of AI-generated images, videos, audio, and audiovisual stimuli |
|authors| Di CookeAbigail EdwardsSophia BarkoffKathryn Kelly
|links| http://arxiv.org/abs/2403.16760v1 |
|updated| 2024-03-25 13:39:33 UTC |
|summary| As synthetic media becomes progressively more realistic and barriers to usingit continue to lower the technology has been increasingly utilized formalicious purposes from financial fraud to nonconsensual pornography. Todaythe principal defense against being misled by synthetic media relies on theability of the human observer to visually and auditorily discern between realand fake. However it remains unclear just how vulnerable people actually areto deceptive synthetic media in the course of their day to day lives. Weconducted a perceptual study with 1276 participants to assess how accuratepeople were at distinguishing synthetic images audio only video only andaudiovisual stimuli from authentic. To reflect the circumstances under whichpeople would likely encounter synthetic media in the wild testing conditionsand stimuli emulated a typical online platform while all synthetic media usedin the survey was sourced from publicly accessible generative AI technology.  We find that overall participants struggled to meaningfully discern betweensynthetic and authentic content. We also find that detection performanceworsens when the stimuli contains synthetic content as compared to authenticcontent images featuring human faces as compared to non face objects a singlemodality as compared to multimodal stimuli mixed authenticity as compared tobeing fully synthetic for audiovisual stimuli and features foreign languagesas compared to languages the observer is fluent in. Finally we also find thatprior knowledge of synthetic media does not meaningfully impact their detectionperformance. Collectively these results indicate that people are highlysusceptible to being tricked by synthetic media in their daily lives and thathuman perceptual detection capabilities can no longer be relied upon as aneffective counterdefense. |


| Item |Content|
| --- |---|
|idx| 2403.16653v1 |
|title| Instantaneous Visual Analysis of Blood Flow in Stenoses Using Morphological Similarity |
|authors| Pepe EulzerKevin RichterAnna HundertmarkRalf WickenhöferCarsten M. KlingnerKai Lawonn
|links| http://arxiv.org/abs/2403.16653v1 |
|updated| 2024-03-25 11:40:47 UTC |
|summary| The emergence of computational fluid dynamics CFD enabled the simulation ofintricate transport processes including flow in physiological structures suchas blood vessels. While these so-called hemodynamic simulations offergroundbreaking opportunities to solve problems at the clinical forefront asuccessful translation of CFD to clinical decision-making is challenging.Hemodynamic simulations are intrinsically complex time-consuming andresource-intensive which conflicts with the time-sensitive nature of clinicalworkflows and the fact that hospitals usually do not have the necessaryresources or infrastructure to support CFD simulations. To address thesetransfer challenges we propose a novel visualization system which enablesinstant flow exploration without performing on-site simulation. To gaininsights into the viability of the approach we focus on hemodynamicsimulations of the carotid bifurcation which is a highly relevant arterialsubtree in stroke diagnostics and prevention. We created an initial database of120 high-resolution carotid bifurcation flow models and developed a set ofsimilarity metrics used to place a new carotid surface model into aneighborhood of simulated cases with the highest geometric similarity. Theneighborhood can be immediately explored and the flow fields analyzed. We foundthat if the artery models are similar enough in the regions of interest a newsimulation leads to coinciding results allowing the user to circumventindividual flow simulations. We conclude that similarity-based visual analysisis a promising approach toward the usability of CFD in medical practice. |


| Item |Content|
| --- |---|
|idx| 2403.16645v1 |
|title| Virtual Co-Pilot: Multimodal Large Language Model-enabled Quick-access Procedures for Single Pilot Operations |
|authors| Fan LiShanshan FengYuqi YanChing-Hung LeeYew Soon Ong
|links| http://arxiv.org/abs/2403.16645v1 |
|updated| 2024-03-25 11:31:45 UTC |
|summary| Advancements in technology pilot shortages and cost pressures are driving atrend towards single-pilot and even remote operations in aviation. Consideringthe extensive workload and huge risks associated with single-pilot operationsthe development of a Virtual Co-Pilot V-CoP is expected to be a potential wayto ensure aviation safety. This study proposes a V-CoP concept and explores howhumans and virtual assistants can effectively collaborate. A preliminary casestudy is conducted to explore a critical role of V-CoP namely automated quickprocedures searching using the multimodal large language model LLM. TheLLM-enabled V-CoP integrates the pilot instruction and real-time cockpitinstrumental data to prompt applicable aviation manuals and operationprocedures. The results showed that the LLM-enabled V-CoP achieved highaccuracy in situational analysis and effective retrieval of procedureinformation. The results showed that the LLM-enabled V-CoP achieved highaccuracy in situational analysis 90.5 and effective retrieval of procedureinformation 86.5. The proposed V-CoP is expected to provide a foundation forfuture virtual intelligent assistant development improve the performance ofsingle pilots and reduce the risk of human errors in aviation. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2403.16719v1 |
|title| Towards a Formalisation of Value-based Actions and Consequentialist Ethics |
|authors| Adam WynerTomasz ZurekDOrota Stachura-Zurek
|links| http://arxiv.org/abs/2403.16719v1 |
|updated| 2024-03-25 12:56:48 UTC |
|summary| Agents act to bring about a state of the world that is more compatible withtheir personal or institutional values. To formalise this intuition the paperproposes an action framework based on the STRIPS formalisation. Technicallythe contribution expresses actions in terms of Value-based Formal ReasoningVFR which provides a set of propositions derived from an Agents valueprofile and the Agents assessment of propositions with respect to the profile.Conceptually the contribution provides a computational framework for a form ofconsequentialist ethics which is satisficing luralistic act-based andpreferential. |


| Item |Content|
| --- |---|
|idx| 2403.16517v1 |
|title| Norm Violation Detection in Multi-Agent Systems using Large Language Models: A Pilot Study |
|authors| Shawn HeSurangika RanathungaStephen CranefieldBastin Tony Roy Savarimuthu
|links| http://arxiv.org/abs/2403.16517v1 |
|updated| 2024-03-25 08:01:33 UTC |
|summary| Norms are an important component of the social fabric of society byprescribing expected behaviour. In Multi-Agent Systems MAS agentsinteracting within a society are equipped to possess social capabilities suchas reasoning about norms and trust. Norms have long been of interest within theNormative Multi-Agent Systems community with researchers studying topics suchas norm emergence norm violation detection and sanctioning. However thesestudies have some limitations: they are often limited to simple domains normshave been represented using a variety of representations with no standardapproach emerging and the symbolic reasoning mechanisms generally used maysuffer from a lack of extensibility and robustness. In contrast Large LanguageModels LLMs offer opportunities to discover and reason about norms across alarge range of social situations. This paper evaluates the capability of LLMsto detecting norm violations. Based on simulated data from 80 stories in ahousehold context with varying complexities we investigated whether 10 normsare violated. For our evaluations we first obtained the ground truth from threehuman evaluators for each story. Then the majority result was compared againstthe results from three well-known LLM models Llama 2 7B Mixtral 7B andChatGPT-4. Our results show the promise of ChatGPT-4 for detecting normviolations with Mixtral some distance behind. Also we identify areas wherethese models perform poorly and discuss implications for future work. |


| Item |Content|
| --- |---|
|idx| 2403.16329v1 |
|title| Social Deliberation vs. Social Contracts in Self-Governing Voluntary Organisations |
|authors| Matthew ScottAsimina MertzaniCiske SmitStefan SarkadiJeremy Pitt
|links| http://arxiv.org/abs/2403.16329v1 |
|updated| 2024-03-24 23:34:40 UTC |
|summary| Self-organising multi-agent systems regulate their components behaviourvoluntarily according to a set of socially-constructed mutually-agreed andmutable social arrangements. In some systems these arrangements may be appliedwith a frequency at a scale and within implicit cost constraints such thatperformance becomes a pressing issue. This paper introduces thetextitMegabike Scenario which consists of a negotiated agreement on arelatively large set of conventional rules frequent democraticdecision-making according to those rules and a resource-bounded imperative toreach correct decisions. A formalism is defined for effective rulerepresentation and processing in the scenario and is evaluated against fiveinterleaved socio-functional requirements. System performance is also evaluatedempirically through simulation. We conclude that to self-organise their socialarrangements agents need some awareness of their own limitations and the valueof compromise. |


| Item |Content|
| --- |---|
|idx| 2403.16151v1 |
|title| Ultra Low-Cost Two-Stage Multimodal System for Non-Normative Behavior Detection |
|authors| Albert LuStephen Cranefield
|links| http://arxiv.org/abs/2403.16151v1 |
|updated| 2024-03-24 13:44:32 UTC |
|summary| The online community has increasingly been inundated by a toxic wave ofharmful comments. In response to this growing challenge we introduce atwo-stage ultra-low-cost multimodal harmful behavior detection method designedto identify harmful comments and images with high precision and recall rates.We first utilize the CLIP-ViT model to transform tweets and images intoembeddings effectively capturing the intricate interplay of semantic meaningand subtle contextual clues within texts and images. Then in the second stagethe system feeds these embeddings into a conventional machine learningclassifier like SVM or logistic regression enabling the system to be trainedrapidly and to perform inference at an ultra-low cost. By converting tweetsinto rich multimodal embeddings through the CLIP-ViT model and utilizing themto train conventional machine learning classifiers our system is not onlycapable of detecting harmful textual information with near-perfect performanceachieving precision and recall rates above 99 but also demonstrates theability to zero-shot harmful images without additional training thanks to itsmultimodal embedding input. This capability empowers our system to identifyunseen harmful images without requiring extensive and costly image datasets.Additionally our system quickly adapts to new harmful content if a newharmful content pattern is identified we can fine-tune the classifier with thecorresponding tweets embeddings to promptly update the system. This makes itwell suited to addressing the ever-evolving nature of online harmfulnessproviding online communities with a robust generalizable and cost-effectivetool to safeguard their communities. |


| Item |Content|
| --- |---|
|idx| 2403.15946v1 |
|title| Team Coordination on Graphs: Problem, Analysis, and Algorithms |
|authors| Manshi LimbuYanlin ZhouGregory SteinXuan WangDaigo ShishikaXuesu Xiao
|links| http://arxiv.org/abs/2403.15946v1 |
|updated| 2024-03-23 22:31:17 UTC |
|summary| Team Coordination on Graphs with Risky Edges TCGRE is a recently emergedproblem in which a robot team collectively reduces graph traversal costthrough support from one robot to another when the latter traverses a riskyedge. Resembling the traditional Multi-Agent Path Finding MAPF problem bothclassical and learning-based methods have been proposed to solve TCGREhowever they lacked either computation efficiency or optimality assurance. Inthis paper we reformulate TCGRE as a constrained optimization and performrigorous mathematical analysis. Our theoretical analysis shows the NP-hardnessof TCGRE by reduction from the Maximum 3D Matching problem and that efficientdecomposition is a key to tackle this combinatorial optimization problem.Further more we design three classes of algorithms to solve TCGRE i.e. JointState Graph JSG based coordination based and receding-horizon sub-teambased solutions. Each of these proposed algorithms enjoy different provableoptimality and efficiency characteristics that are demonstrated in ourextensive experiments. |


