# cs.CL 

| Item |Content|
| --- |---|
|idx| 1 |
|title| RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture |
|authors| Aman GuptaAnup ShirgaonkarAngels de Luis BalaguerBruno SilvaDaniel HolsteinDawei LiJennifer MarsmanLeonardo O. NunesMahsa RouzbahmanMorris SharpNick MecklenburgRafael PadilhaRanveer ChandraRenato Luiz de Freitas CunhaRoberto de M. Estevão FilhoRyan TsangSara MalvarSwati SharmaTodd HendryVijay AskiVijetha VijayendranVinamra Benara
|links| http://arxiv.org/abs/2401.08406v1 |
|updated| 2024-01-16 14:44:47 UTC |
|summary| There are two common ways in which developers are incorporating proprietaryand domain-specific data when building applications of Large Language ModelsLLMs: Retrieval-Augmented Generation RAG and Fine-Tuning. RAG augments theprompt with the external data while fine-Tuning incorporates the additionalknowledge into the model itself. However the pros and cons of both approachesare not well understood. In this paper we propose a pipeline for fine-tuningand RAG and present the tradeoffs of both for multiple popular LLMs includingLlama2-13B GPT-3.5 and GPT-4. Our pipeline consists of multiple stagesincluding extracting information from PDFs generating questions and answersusing them for fine-tuning and leveraging GPT-4 for evaluating the results. Wepropose metrics to assess the performance of different stages of the RAG andfine-Tuning pipeline. We conduct an in-depth study on an agricultural dataset.Agriculture as an industry has not seen much penetration of AI and we study apotentially disruptive application - what if we could provide location-specificinsights to a farmer Our results show the effectiveness of our datasetgeneration pipeline in capturing geographic-specific knowledge and thequantitative and qualitative benefits of RAG and fine-tuning. We see anaccuracy increase of over 6 p.p. when fine-tuning the model and this iscumulative with RAG which increases accuracy by 5 p.p. further. In oneparticular experiment we also demonstrate that the fine-tuned model leveragesinformation from across geographies to answer specific questions increasinganswer similarity from 47 to 72. Overall the results point to how systemsbuilt using LLMs can be adapted to respond and incorporate knowledge across adimension that is critical for a specific industry paving the way for furtherapplications of LLMs in other industrial domains. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| Hidden Flaws Behind Expert-Level Accuracy of GPT-4 Vision in Medicine |
|authors| Qiao JinFangyuan ChenYiliang ZhouZiyang XuJustin M. CheungRobert ChenRonald M. SummersJustin F. RousseauPeiyun NiMarc J LandsmanSally L. BaxterSubhi J. Al'ArefYijia LiMichael F. ChiangYifan PengZhiyong Lu
|links| http://arxiv.org/abs/2401.08396v1 |
|updated| 2024-01-16 14:41:20 UTC |
|summary| Recent studies indicate that Generative Pre-trained Transformer 4 with VisionGPT-4V outperforms human physicians in medical challenge tasks. Howeverthese evaluations primarily focused on the accuracy of multi-choice questionsalone. Our study extends the current scope by conducting a comprehensiveanalysis of GPT-4Vs rationales of image comprehension recall of medicalknowledge and step-by-step multimodal reasoning when solving New EnglandJournal of Medicine NEJM Image Challenges - an imaging quiz designed to testthe knowledge and diagnostic capabilities of medical professionals. Evaluationresults confirmed that GPT-4V outperforms human physicians regardingmulti-choice accuracy 88.0 vs. 77.0 p0.034. GPT-4V also performs well incases where physicians incorrectly answer with over 80 accuracy. However wediscovered that GPT-4V frequently presents flawed rationales in cases where itmakes the correct final choices 27.3 most prominent in image comprehension21.6. Regardless of GPT-4Vs high accuracy in multi-choice questions ourfindings emphasize the necessity for further in-depth evaluations of itsrationales before integrating such models into clinical workflows. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models |
|authors| Zongxin YangGuikun ChenXiaodi LiWenguan WangYi Yang
|links| http://arxiv.org/abs/2401.08392v1 |
|updated| 2024-01-16 14:33:09 UTC |
|summary| The field of AI agents is advancing at an unprecedented rate due to thecapabilities of large language models LLMs. However LLM-driven visual agentsmainly focus on solving tasks for the image modality which limits theirability to understand the dynamic nature of the real world making it still farfrom real-life applications e.g. guiding students in laboratory experimentsand identifying their mistakes. Considering the video modality better reflectsthe ever-changing and perceptually intensive nature of real-world scenarios wedevise DoraemonGPT a comprehensive and conceptually elegant system driven byLLMs to handle dynamic video tasks. Given a video with a question/taskDoraemonGPT begins by converting the input video with massive content into asymbolic memory that stores textittask-related attributes. This structuredrepresentation allows for spatial-temporal querying and reasoning by sub-tasktools resulting in concise and relevant intermediate results. Recognizing thatLLMs have limited internal knowledge when it comes to specialized domainse.g. analyzing the scientific principles underlying experiments weincorporate plug-and-play tools to assess external knowledge and address tasksacross different domains. Moreover we introduce a novel LLM-driven plannerbased on Monte Carlo Tree Search to efficiently explore the large planningspace for scheduling various tools. The planner iteratively finds feasiblesolutions by backpropagating the results reward and multiple solutions can besummarized into an improved final answer. We extensively evaluate DoraemonGPTin dynamic scenes and provide in-the-wild showcases demonstrating its abilityto handle more complex questions than previous studies. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| Cross-lingual neural fuzzy matching for exploiting target-language monolingual corpora in computer-aided translation |
|authors| Miquel Esplà-GomisVíctor M. Sánchez-CartagenaJuan Antonio Pérez-OrtizFelipe Sánchez-Martínez
|links| http://dx.doi.org/10.18653/v1/2022.emnlp-main.511 |
|updated| 2024-01-16 14:00:28 UTC |
|summary| Computer-aided translation CAT tools based on translation memories MTplay a prominent role in the translation workflow of professional translators.However the reduced availability of in-domain TMs as compared to in-domainmonolingual corpora limits its adoption for a number of translation tasks. Inthis paper we introduce a novel neural approach aimed at overcoming thislimitation by exploiting not only TMs but also in-domain target-language TLmonolingual corpora and still enabling a similar functionality to that offeredby conventional TM-based CAT tools. Our approach relies on cross-lingualsentence embeddings to retrieve translation proposals from TL monolingualcorpora and on a neural model to estimate their post-editing effort. The paperpresents an automatic evaluation of these techniques on four language pairsthat shows that our approach can successfully exploit monolingual texts in aTM-based CAT environment increasing the amount of useful translationproposals and that our neural model for estimating the post-editing effortenables the combination of translation proposals obtained from monolingualcorpora and from TMs in the usual way. A human evaluation performed on a singlelanguage pair confirms the results of the automatic evaluation and seems toindicate that the translation proposals retrieved with our approach are moreuseful than what the automatic evaluation shows. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| Morphology and Syntax of the Tamil Language |
|authors| Kengatharaiyer Sarveswaran
|links| http://arxiv.org/abs/2401.08367v1 |
|updated| 2024-01-16 13:52:25 UTC |
|summary| This paper provides an overview of the morphology and syntax of the Tamillanguage focusing on its contemporary usage. The paper also highlights thecomplexity and richness of Tamil in terms of its morphological and syntacticfeatures which will be useful for linguists analysing the language andconducting comparative studies. In addition the paper will be useful for thosedeveloping computational resources for the Tamil language. It is proven as arule-based morphological analyser cum generator and a computational grammar forTamil have already been developed based on this paper. To enhance accessibilityfor a broader audience the analysis is conducted without relying on anyspecific grammatical formalism. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 1 |
|title| Interrogating AI: Characterizing Emergent Playful Interactions with ChatGPT |
|authors| Mohammad Ronagh NikghalbJinghui Cheng
|links| http://arxiv.org/abs/2401.08405v1 |
|updated| 2024-01-16 14:44:13 UTC |
|summary| In an era of AIs growing capabilities and influences recent advancementsare reshaping HCI and CSCWs view of AI as mere tools. Playful interactionswith AI systems naturally emerged as a way for users to make sense of theever-changing technology. However these emergent and playful interactions areunderexamined. We target this gap by investigating playful interactionsexhibited by users of a recently trending powerful AI technology ChatGPT.Through a thematic analysis of 372 user-generated posts on the ChatGPTsubreddit we found that a substantial portion of user discourse revolvesaround playful interactions. The analysis further allowed us to construct apreliminary taxonomy to describe these interactions categorizing them into sixtypes: reflecting jesting imitating challenging tricking and contrivingeach included sub-categories. Overall this study contributes to the field ofHCI and CSCW by illuminating the multifaceted nature of playful interactionswith AI underlining their significance in shaping the human-AI relationship. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| Hidden Flaws Behind Expert-Level Accuracy of GPT-4 Vision in Medicine |
|authors| Qiao JinFangyuan ChenYiliang ZhouZiyang XuJustin M. CheungRobert ChenRonald M. SummersJustin F. RousseauPeiyun NiMarc J LandsmanSally L. BaxterSubhi J. Al'ArefYijia LiMichael F. ChiangYifan PengZhiyong Lu
|links| http://arxiv.org/abs/2401.08396v1 |
|updated| 2024-01-16 14:41:20 UTC |
|summary| Recent studies indicate that Generative Pre-trained Transformer 4 with VisionGPT-4V outperforms human physicians in medical challenge tasks. Howeverthese evaluations primarily focused on the accuracy of multi-choice questionsalone. Our study extends the current scope by conducting a comprehensiveanalysis of GPT-4Vs rationales of image comprehension recall of medicalknowledge and step-by-step multimodal reasoning when solving New EnglandJournal of Medicine NEJM Image Challenges - an imaging quiz designed to testthe knowledge and diagnostic capabilities of medical professionals. Evaluationresults confirmed that GPT-4V outperforms human physicians regardingmulti-choice accuracy 88.0 vs. 77.0 p0.034. GPT-4V also performs well incases where physicians incorrectly answer with over 80 accuracy. However wediscovered that GPT-4V frequently presents flawed rationales in cases where itmakes the correct final choices 27.3 most prominent in image comprehension21.6. Regardless of GPT-4Vs high accuracy in multi-choice questions ourfindings emphasize the necessity for further in-depth evaluations of itsrationales before integrating such models into clinical workflows. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| A Micro Architectural Events Aware Real-Time Embedded System Fault Injector |
|authors| Enrico MaglianoAlessio CarpegnaAlessadro SavinoStefano Di Carlo
|links| http://arxiv.org/abs/2401.08397v1 |
|updated| 2024-01-16 14:41:20 UTC |
|summary| In contemporary times the increasing complexity of the system posessignificant challenges to the reliability trustworthiness and security of theSACRES. Key issues include the susceptibility to phenomena such asinstantaneous voltage spikes electromagnetic interference neutron strikesand out-of-range temperatures. These factors can induce switch state changes intransistors resulting in bit-flipping soft errors and transient corruptionof stored data in memory. The occurrence of soft errors in turn may lead tosystem faults that can propel the system into a hazardous state. Particularlyin critical sectors like automotive avionics or aerospace such malfunctionscan have real-world implications potentially causing harm to individuals.  This paper introduces a novel fault injector designed to facilitate themonitoring aggregation and examination of micro-architectural events. This isachieved by harnessing the microprocessors PMU and the debugging interfacespecifically focusing on ensuring the repeatability of fault injections. Thefault injection methodology targets bit-flipping within the memory systemaffecting CPU registers and RAM. The outcomes of these fault injections enablea thorough analysis of the impact of soft errors and establish a robustcorrelation between the identified faults and the essential timingpredictability demanded by SACRES. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| Deep Learning-based Group Causal Inference in Multivariate Time-series |
|authors| Wasim AhmadMaha ShadaydehJoachim Denzler
|links| http://arxiv.org/abs/2401.08386v1 |
|updated| 2024-01-16 14:19:28 UTC |
|summary| Causal inference in a nonlinear system of multivariate timeseries isinstrumental in disentangling the intricate web of relationships amongvariables enabling us to make more accurate predictions and gain deeperinsights into real-world complex systems. Causality methods typically identifythe causal structure of a multivariate system by considering the cause-effectrelationship of each pair of variables while ignoring the collective effect ofa group of variables or interactions involving more than two-time seriesvariables. In this work we test model invariance by group-level interventionson the trained deep networks to infer causal direction in groups of variablessuch as climate and ecosystem brain networks etc. Extensive testing withsynthetic and real-world time series data shows a significant improvement ofour method over other applied group causality methods and provides us insightsinto real-world time series. The code for our method can be foundat:https://github.com/wasimahmadpk/gCause. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference |
|authors| Jinghan YaoQuentin AnthonyAamir ShafiHari SubramoniDhabaleswar K.Panda
|links| http://arxiv.org/abs/2401.08383v1 |
|updated| 2024-01-16 14:16:47 UTC |
|summary| In large language models like the Generative Pre-trained Transformer theMixture of Experts paradigm has emerged as a powerful technique for enhancingmodel expressiveness and accuracy. However deploying GPT MoE models forparallel inference on distributed systems presents significant challengesprimarily due to the extensive Alltoall communication required for expertrouting and aggregation. This communication bottleneck exacerbates the alreadycomplex computational landscape hindering the efficient utilization ofhigh-performance computing resources. In this paper we propose a lightweightoptimization technique called ExFlow to largely accelerate the inference ofthese MoE models. We take a new perspective on alleviating the communicationoverhead by exploiting the inter-layer expert affinity. Unlike previousmethods our solution can be directly applied to pre-trained MoE models withoutany fine-tuning or accuracy degradation. By proposing a context-coherent expertparallelism on distributed systems our design only uses one Alltoallcommunication to deliver the same functionality while previous methods allrequire two Alltoalls. By carefully examining the conditional probability intokens routing across multiple layers we proved that pre-trained GPT MoEmodels implicitly exhibit a strong inter-layer expert affinity. We then designan efficient integer programming model to capture such features and show thatby properly placing the experts on corresponding GPUs we can reduce up to 67cross-GPU routing latency. Our solution beats the cutting-edge MoEimplementations with experts from 8 to 64 with up to 2.2x improvement ininference throughput. We further provide a detailed study of how the modelimplicitly acquires this expert affinity at the very early training stage andhow this affinity evolves and stabilizes during training. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 1 |
|title| Faster ISNet for Background Bias Mitigation on Deep Neural Networks |
|authors| Pedro R. A. S. BassiSergio DecherchiAndrea Cavalli
|links| http://arxiv.org/abs/2401.08409v1 |
|updated| 2024-01-16 14:49:26 UTC |
|summary| Image background features can constitute background bias spuriouscorrelations and impact deep classifiers decisions causing shortcut learningClever Hans effect and reducing the generalization skill on real-world data.The concept of optimizing Layer-wise Relevance Propagation LRP heatmaps toimprove classifier behavior was recently introduced by a neural networkarchitecture named ISNet. It minimizes background relevance in LRP maps tomitigate the influence of image background features on deep classifiersdecisions hindering shortcut learning and improving generalization. For eachtraining image the original ISNet produces one heatmap per possible class inthe classification task hence its training time scales linearly with thenumber of classes. Here we introduce reformulated architectures that allow thetraining time to become independent from this number rendering theoptimization process much faster. We challenged the enhanced models utilizingthe MNIST dataset with synthetic background bias and COVID-19 detection inchest X-rays an application that is prone to shortcut learning due tobackground bias. The trained models minimized background attention and hinderedshortcut learning while retaining high accuracy. Considering externalout-of-distribution test datasets they consistently proved more accuratethan multiple state-of-the-art deep neural network architectures including adedicated image semantic segmenter followed by a classifier. The architecturespresented here represent a potentially massive improvement in training speedover the original ISNet thus introducing LRP optimization into a gamut ofapplications that could not be feasibly handled by the original model. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture |
|authors| Aman GuptaAnup ShirgaonkarAngels de Luis BalaguerBruno SilvaDaniel HolsteinDawei LiJennifer MarsmanLeonardo O. NunesMahsa RouzbahmanMorris SharpNick MecklenburgRafael PadilhaRanveer ChandraRenato Luiz de Freitas CunhaRoberto de M. Estevão FilhoRyan TsangSara MalvarSwati SharmaTodd HendryVijay AskiVijetha VijayendranVinamra Benara
|links| http://arxiv.org/abs/2401.08406v1 |
|updated| 2024-01-16 14:44:47 UTC |
|summary| There are two common ways in which developers are incorporating proprietaryand domain-specific data when building applications of Large Language ModelsLLMs: Retrieval-Augmented Generation RAG and Fine-Tuning. RAG augments theprompt with the external data while fine-Tuning incorporates the additionalknowledge into the model itself. However the pros and cons of both approachesare not well understood. In this paper we propose a pipeline for fine-tuningand RAG and present the tradeoffs of both for multiple popular LLMs includingLlama2-13B GPT-3.5 and GPT-4. Our pipeline consists of multiple stagesincluding extracting information from PDFs generating questions and answersusing them for fine-tuning and leveraging GPT-4 for evaluating the results. Wepropose metrics to assess the performance of different stages of the RAG andfine-Tuning pipeline. We conduct an in-depth study on an agricultural dataset.Agriculture as an industry has not seen much penetration of AI and we study apotentially disruptive application - what if we could provide location-specificinsights to a farmer Our results show the effectiveness of our datasetgeneration pipeline in capturing geographic-specific knowledge and thequantitative and qualitative benefits of RAG and fine-tuning. We see anaccuracy increase of over 6 p.p. when fine-tuning the model and this iscumulative with RAG which increases accuracy by 5 p.p. further. In oneparticular experiment we also demonstrate that the fine-tuned model leveragesinformation from across geographies to answer specific questions increasinganswer similarity from 47 to 72. Overall the results point to how systemsbuilt using LLMs can be adapted to respond and incorporate knowledge across adimension that is critical for a specific industry paving the way for furtherapplications of LLMs in other industrial domains. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| Training and Comparison of nnU-Net and DeepMedic Methods for Autosegmentation of Pediatric Brain Tumors |
|authors| Arastoo VossoughNastaran KhaliliAriana M. FamiliarDeep GandhiKarthik ViswanathanWenxin TuDebanjan HaldarSina BagheriHannah AndersonShuvanjan HaldarPhillip B. StormAdam ResnickJeffrey B. WareAli NabavizadehAnahita Fathi Kazerooni
|links| http://arxiv.org/abs/2401.08404v1 |
|updated| 2024-01-16 14:44:06 UTC |
|summary| Brain tumors are the most common solid tumors and the leading cause ofcancer-related death among children. Tumor segmentation is essential insurgical and treatment planning and response assessment and monitoring.However manual segmentation is time-consuming and has high inter-operatorvariability underscoring the need for more efficient methods. We compared twodeep learning-based 3D segmentation models DeepMedic and nnU-Net aftertraining with pediatric-specific multi-institutional brain tumor data usingbased on multi-parametric MRI scans.Multi-parametric preoperative MRI scans of339 pediatric patients n293 internal and n46 external cohorts with avariety of tumor subtypes were preprocessed and manually segmented into fourtumor subregions i.e. enhancing tumor ET non-enhancing tumor NET cysticcomponents CC and peritumoral edema ED. After training performance of thetwo models on internal and external test sets was evaluated using Dice scoressensitivity and Hausdorff distance with reference to ground truth manualsegmentations. Dice score for nnU-Net internal test sets was mean /- SDmedian 0.9/-0.07 0.94 for WT 0.77/-0.29 for ET 0.66/-0.32 for NET0.71/-0.33 for CC and 0.71/-0.40 for ED respectively. For DeepMedic theDice scores were 0.82/-0.16 for WT 0.66/-0.32 for ET 0.48/-0.27 for NET0.48/-0.36 for CC and 0.19/-0.33 for ED respectively. Dice scores weresignificantly higher for nnU-Net p0.01. External validation of the trainednnU-Net model on the multi-institutional BraTS-PEDs 2023 dataset revealed highgeneralization capability in segmentation of whole tumor and tumor core withDice scores of 0.87/-0.13 0.91 and 0.83/-0.18 0.89 respectively.Pediatric-specific data trained nnU-Net model is superior to DeepMedic forwhole tumor and subregion segmentation of pediatric brain tumors. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| Deep Learning-based Group Causal Inference in Multivariate Time-series |
|authors| Wasim AhmadMaha ShadaydehJoachim Denzler
|links| http://arxiv.org/abs/2401.08386v1 |
|updated| 2024-01-16 14:19:28 UTC |
|summary| Causal inference in a nonlinear system of multivariate timeseries isinstrumental in disentangling the intricate web of relationships amongvariables enabling us to make more accurate predictions and gain deeperinsights into real-world complex systems. Causality methods typically identifythe causal structure of a multivariate system by considering the cause-effectrelationship of each pair of variables while ignoring the collective effect ofa group of variables or interactions involving more than two-time seriesvariables. In this work we test model invariance by group-level interventionson the trained deep networks to infer causal direction in groups of variablessuch as climate and ecosystem brain networks etc. Extensive testing withsynthetic and real-world time series data shows a significant improvement ofour method over other applied group causality methods and provides us insightsinto real-world time series. The code for our method can be foundat:https://github.com/wasimahmadpk/gCause. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference |
|authors| Jinghan YaoQuentin AnthonyAamir ShafiHari SubramoniDhabaleswar K.Panda
|links| http://arxiv.org/abs/2401.08383v1 |
|updated| 2024-01-16 14:16:47 UTC |
|summary| In large language models like the Generative Pre-trained Transformer theMixture of Experts paradigm has emerged as a powerful technique for enhancingmodel expressiveness and accuracy. However deploying GPT MoE models forparallel inference on distributed systems presents significant challengesprimarily due to the extensive Alltoall communication required for expertrouting and aggregation. This communication bottleneck exacerbates the alreadycomplex computational landscape hindering the efficient utilization ofhigh-performance computing resources. In this paper we propose a lightweightoptimization technique called ExFlow to largely accelerate the inference ofthese MoE models. We take a new perspective on alleviating the communicationoverhead by exploiting the inter-layer expert affinity. Unlike previousmethods our solution can be directly applied to pre-trained MoE models withoutany fine-tuning or accuracy degradation. By proposing a context-coherent expertparallelism on distributed systems our design only uses one Alltoallcommunication to deliver the same functionality while previous methods allrequire two Alltoalls. By carefully examining the conditional probability intokens routing across multiple layers we proved that pre-trained GPT MoEmodels implicitly exhibit a strong inter-layer expert affinity. We then designan efficient integer programming model to capture such features and show thatby properly placing the experts on corresponding GPUs we can reduce up to 67cross-GPU routing latency. Our solution beats the cutting-edge MoEimplementations with experts from 8 to 64 with up to 2.2x improvement ininference throughput. We further provide a detailed study of how the modelimplicitly acquires this expert affinity at the very early training stage andhow this affinity evolves and stabilizes during training. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 1 |
|title| Faster ISNet for Background Bias Mitigation on Deep Neural Networks |
|authors| Pedro R. A. S. BassiSergio DecherchiAndrea Cavalli
|links| http://arxiv.org/abs/2401.08409v1 |
|updated| 2024-01-16 14:49:26 UTC |
|summary| Image background features can constitute background bias spuriouscorrelations and impact deep classifiers decisions causing shortcut learningClever Hans effect and reducing the generalization skill on real-world data.The concept of optimizing Layer-wise Relevance Propagation LRP heatmaps toimprove classifier behavior was recently introduced by a neural networkarchitecture named ISNet. It minimizes background relevance in LRP maps tomitigate the influence of image background features on deep classifiersdecisions hindering shortcut learning and improving generalization. For eachtraining image the original ISNet produces one heatmap per possible class inthe classification task hence its training time scales linearly with thenumber of classes. Here we introduce reformulated architectures that allow thetraining time to become independent from this number rendering theoptimization process much faster. We challenged the enhanced models utilizingthe MNIST dataset with synthetic background bias and COVID-19 detection inchest X-rays an application that is prone to shortcut learning due tobackground bias. The trained models minimized background attention and hinderedshortcut learning while retaining high accuracy. Considering externalout-of-distribution test datasets they consistently proved more accuratethan multiple state-of-the-art deep neural network architectures including adedicated image semantic segmenter followed by a classifier. The architecturespresented here represent a potentially massive improvement in training speedover the original ISNet thus introducing LRP optimization into a gamut ofapplications that could not be feasibly handled by the original model. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| Cross-Domain Few-Shot Segmentation via Iterative Support-Query Correspondence Mining |
|authors| Jiahao NieYun XingGongjie ZhangPei YanAoran XiaoYap-Peng TanAlex C. KotShijian Lu
|links| http://arxiv.org/abs/2401.08407v1 |
|updated| 2024-01-16 14:45:41 UTC |
|summary| Cross-Domain Few-Shot Segmentation CD-FSS poses the challenge of segmentingnovel categories from a distinct domain using only limited exemplars. In thispaper we undertake a comprehensive study of CD-FSS and uncover two crucialinsights: i the necessity of a fine-tuning stage to effectively transfer thelearned meta-knowledge across domains and ii the overfitting risk during thenaive fine-tuning due to the scarcity of novel category examples. With theseinsights we propose a novel cross-domain fine-tuning strategy that addressesthe challenging CD-FSS tasks. We first design Bi-directional Few-shotPrediction BFP which establishes support-query correspondence in abi-directional manner crafting augmented supervision to reduce the overfittingrisk. Then we further extend BFP into Iterative Few-shot Adaptor IFA whichis a recursive framework to capture the support-query correspondenceiteratively targeting maximal exploitation of supervisory signals from thesparse novel category samples. Extensive empirical evaluations show that ourmethod significantly outperforms the state-of-the-arts 7.8 which verifiesthat IFA tackles the cross-domain challenges and mitigates the overfittingsimultaneously. Code will be made available. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| Training and Comparison of nnU-Net and DeepMedic Methods for Autosegmentation of Pediatric Brain Tumors |
|authors| Arastoo VossoughNastaran KhaliliAriana M. FamiliarDeep GandhiKarthik ViswanathanWenxin TuDebanjan HaldarSina BagheriHannah AndersonShuvanjan HaldarPhillip B. StormAdam ResnickJeffrey B. WareAli NabavizadehAnahita Fathi Kazerooni
|links| http://arxiv.org/abs/2401.08404v1 |
|updated| 2024-01-16 14:44:06 UTC |
|summary| Brain tumors are the most common solid tumors and the leading cause ofcancer-related death among children. Tumor segmentation is essential insurgical and treatment planning and response assessment and monitoring.However manual segmentation is time-consuming and has high inter-operatorvariability underscoring the need for more efficient methods. We compared twodeep learning-based 3D segmentation models DeepMedic and nnU-Net aftertraining with pediatric-specific multi-institutional brain tumor data usingbased on multi-parametric MRI scans.Multi-parametric preoperative MRI scans of339 pediatric patients n293 internal and n46 external cohorts with avariety of tumor subtypes were preprocessed and manually segmented into fourtumor subregions i.e. enhancing tumor ET non-enhancing tumor NET cysticcomponents CC and peritumoral edema ED. After training performance of thetwo models on internal and external test sets was evaluated using Dice scoressensitivity and Hausdorff distance with reference to ground truth manualsegmentations. Dice score for nnU-Net internal test sets was mean /- SDmedian 0.9/-0.07 0.94 for WT 0.77/-0.29 for ET 0.66/-0.32 for NET0.71/-0.33 for CC and 0.71/-0.40 for ED respectively. For DeepMedic theDice scores were 0.82/-0.16 for WT 0.66/-0.32 for ET 0.48/-0.27 for NET0.48/-0.36 for CC and 0.19/-0.33 for ED respectively. Dice scores weresignificantly higher for nnU-Net p0.01. External validation of the trainednnU-Net model on the multi-institutional BraTS-PEDs 2023 dataset revealed highgeneralization capability in segmentation of whole tumor and tumor core withDice scores of 0.87/-0.13 0.91 and 0.83/-0.18 0.89 respectively.Pediatric-specific data trained nnU-Net model is superior to DeepMedic forwhole tumor and subregion segmentation of pediatric brain tumors. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding |
|authors| Yun LiuHaolin YangXu SiLing LiuZipeng LiYuxiang ZhangYebin LiuLi Yi
|links| http://arxiv.org/abs/2401.08399v1 |
|updated| 2024-01-16 14:41:42 UTC |
|summary| Humans commonly work with multiple objects in daily life and can intuitivelytransfer manipulation skills to novel objects by understanding objectfunctional regularities. However existing technical approaches for analyzingand synthesizing hand-object manipulation are mostly limited to handling asingle hand and object due to the lack of data support. To address this weconstruct TACO an extensive bimanual hand-object-interaction dataset spanninga large variety of tool-action-object compositions for daily human activities.TACO contains 2.5K motion sequences paired with third-person and egocentricviews precise hand-object 3D meshes and action labels. To rapidly expand thedata scale we present a fully-automatic data acquisition pipeline combiningmulti-view sensing with an optical motion capture system. With the vastresearch fields provided by TACO we benchmark three generalizablehand-object-interaction tasks: compositional action recognition generalizablehand-object motion forecasting and cooperative grasp synthesis. Extensiveexperiments reveal new insights challenges and opportunities for advancingthe studies of generalizable hand-object motion analysis and synthesis. Ourdata and code are available at https://taco2024.github.io. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| High-Quality Mesh Blendshape Generation from Face Videos via Neural Inverse Rendering |
|authors| Xin MingJiawei LiJingwang LingLibo ZhangFeng Xu
|links| http://arxiv.org/abs/2401.08398v1 |
|updated| 2024-01-16 14:41:31 UTC |
|summary| Readily editable mesh blendshapes have been widely used in animationpipelines while recent advancements in neural geometry and appearancerepresentations have enabled high-quality inverse rendering. Building uponthese observations we introduce a novel technique that reconstructs mesh-basedblendshape rigs from single or sparse multi-view videos leveragingstate-of-the-art neural inverse rendering. We begin by constructing adeformation representation that parameterizes vertex displacements intodifferential coordinates with tetrahedral connections allowing forhigh-quality vertex deformation on high-resolution meshes. By constructing aset of semantic regulations in this representation we achieve jointoptimization of blendshapes and expression coefficients. Furthermore to enablea user-friendly multi-view setup with unsynchronized cameras we propose aneural regressor to model time-varying motion parameters. This approachimplicitly considers the time difference across multiple cameras enhancing theaccuracy of motion modeling. Experiments demonstrate that with the flexibleinput of single or sparse multi-view videos we reconstruct personalizedhigh-fidelity blendshapes. These blendshapes are both geometrically andsemantically accurate and they are compatible with industrial animationpipelines. Code and data will be released. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 1 |
|title| Sparse PCA with False Discovery Rate Controlled Variable Selection |
|authors| Jasin MachkourArnaud BreloyMichael MumaDaniel P. PalomarFrédéric Pascal
|links| http://arxiv.org/abs/2401.08375v1 |
|updated| 2024-01-16 14:07:36 UTC |
|summary| Sparse principal component analysis PCA aims at mapping large dimensionaldata to a linear subspace of lower dimension. By imposing loading vectors to besparse it performs the double duty of dimension reduction and variableselection. Sparse PCA algorithms are usually expressed as a trade-off betweenexplained variance and sparsity of the loading vectors i.e. number ofselected variables. As a high explained variance is not necessarily synonymouswith relevant information these methods are prone to select irrelevantvariables. To overcome this issue we propose an alternative formulation ofsparse PCA driven by the false discovery rate FDR. We then leverage theTerminating-Random Experiments T-Rex selector to automatically determine anFDR-controlled support of the loading vectors. A major advantage of theresulting T-Rex PCA is that no sparsity parameter tuning is required. Numericalexperiments and a stock market data example demonstrate a significantperformance improvement. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| Causal Machine Learning for Moderation Effects |
|authors| Nora BearthMichael Lechner
|links| http://arxiv.org/abs/2401.08290v1 |
|updated| 2024-01-16 11:34:59 UTC |
|summary| It is valuable for any decision maker to know the impact of decisionstreatments on average and for subgroups. The causal machine learningliterature has recently provided tools for estimating group average treatmenteffects GATE to understand treatment heterogeneity better. This paperaddresses the challenge of interpreting such differences in treatment effectsbetween groups while accounting for variations in other covariates. We proposea new parameter the balanced group average treatment effect BGATE whichmeasures a GATE with a specific distribution of a priori-determined covariates.By taking the difference of two BGATEs we can analyse heterogeneity moremeaningfully than by comparing two GATEs. The estimation strategy for thisparameter is based on double/debiased machine learning for discrete treatmentsin an unconfoundedness setting and the estimator is shown to besqrtN-consistent and asymptotically normal under standard conditions.Adding additional identifying assumptions allows specific balanced differencesin treatment effects between groups to be interpreted causally leading to thecausal balanced group average treatment effect. We explore the finite sampleproperties in a small-scale simulation study and demonstrate the usefulness ofthese parameters in an empirical example. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| Statistical Test for Attention Map in Vision Transformer |
|authors| Tomohiro ShiraishiDaiki MiwaTeruyuki KatsuokaVo Nguyen Le DuyKoichi TajiIchiro Takeuchi
|links| http://arxiv.org/abs/2401.08169v1 |
|updated| 2024-01-16 07:18:47 UTC |
|summary| The Vision Transformer ViT demonstrates exceptional performance in variouscomputer vision tasks. Attention is crucial for ViT to capture complexwide-ranging relationships among image patches allowing the model to weigh theimportance of image patches and aiding our understanding of the decision-makingprocess. However when utilizing the attention of ViT as evidence inhigh-stakes decision-making tasks such as medical diagnostics a challengearises due to the potential of attention mechanisms erroneously focusing onirrelevant regions. In this study we propose a statistical test for ViTsattentions enabling us to use the attentions as reliable quantitative evidenceindicators for ViTs decision-making with a rigorously controlled error rate.Using the framework called selective inference we quantify the statisticalsignificance of attentions in the form of p-values which enables thetheoretically grounded quantification of the false positive detectionprobability of attentions. We demonstrate the validity and the effectiveness ofthe proposed method through numerical experiments and applications to brainimage diagnoses. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| Fundamental limits of community detection from multi-view data: multi-layer, dynamic and partially labeled block models |
|authors| Xiaodong YangBuyu LinSubhabrata Sen
|links| http://arxiv.org/abs/2401.08167v1 |
|updated| 2024-01-16 07:13:32 UTC |
|summary| Multi-view data arises frequently in modern network analysis e.g. relationsof multiple types among individuals in social network analysis longitudinalmeasurements of interactions among observational units annotated networks withnoisy partial labeling of vertices etc. We study community detection in thesedisparate settings via a unified theoretical framework and investigate thefundamental thresholds for community recovery. We characterize the mutualinformation between the data and the latent parameters provided the degreesare sufficiently large. Based on this general result i we derive a sharpthreshold for community detection in an inhomogeneous multilayer block modelcitepchen2022global ii characterize a sharp threshold for weak recoveryin a dynamic stochastic block model citepmatias2017statistical and iiiidentify the limiting mutual information in an unbalanced partially labeledblock model. Our first two results are derived modulo coordinate-wise convexityassumptions on specific functions -- we provide extensive numerical evidencefor their correctness. Finally we introduce iterative algorithms based onApproximate Message Passing for community detection in these problems. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| Differentially Private Sliced Inverse Regression: Minimax Optimality and Algorithm |
|authors| Xintao XiaLinjun ZhangZhanrui Cai
|links| http://arxiv.org/abs/2401.08150v1 |
|updated| 2024-01-16 06:47:43 UTC |
|summary| Privacy preservation has become a critical concern in high-dimensional dataanalysis due to the growing prevalence of data-driven applications. Proposed byLi 1991 sliced inverse regression has emerged as a widely utilizedstatistical technique for reducing covariate dimensionality while maintainingsufficient statistical information. In this paper we propose optimallydifferentially private algorithms specifically designed to address privacyconcerns in the context of sufficient dimension reduction. We proceed toestablish lower bounds for differentially private sliced inverse regression inboth the low and high-dimensional settings. Moreover we develop differentiallyprivate algorithms that achieve the minimax lower bounds up to logarithmicfactors. Through a combination of simulations and real data analysis weillustrate the efficacy of these differentially private algorithms insafeguarding privacy while preserving vital information within the reduceddimension space. As a natural extension we can readily offer analogous lowerand upper bounds for differentially private sparse principal componentanalysis a topic that may also be of potential interest to the statistical andmachine learning community. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 1 |
|title| Interrogating AI: Characterizing Emergent Playful Interactions with ChatGPT |
|authors| Mohammad Ronagh NikghalbJinghui Cheng
|links| http://arxiv.org/abs/2401.08405v1 |
|updated| 2024-01-16 14:44:13 UTC |
|summary| In an era of AIs growing capabilities and influences recent advancementsare reshaping HCI and CSCWs view of AI as mere tools. Playful interactionswith AI systems naturally emerged as a way for users to make sense of theever-changing technology. However these emergent and playful interactions areunderexamined. We target this gap by investigating playful interactionsexhibited by users of a recently trending powerful AI technology ChatGPT.Through a thematic analysis of 372 user-generated posts on the ChatGPTsubreddit we found that a substantial portion of user discourse revolvesaround playful interactions. The analysis further allowed us to construct apreliminary taxonomy to describe these interactions categorizing them into sixtypes: reflecting jesting imitating challenging tricking and contrivingeach included sub-categories. Overall this study contributes to the field ofHCI and CSCW by illuminating the multifaceted nature of playful interactionswith AI underlining their significance in shaping the human-AI relationship. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| Understanding User Experience in Large Language Model Interactions |
|authors| Jiayin WangWeizhi MaPeijie SunMin ZhangJian-Yun Nie
|links| http://arxiv.org/abs/2401.08329v1 |
|updated| 2024-01-16 12:49:00 UTC |
|summary| In the rapidly evolving landscape of large language models LLMs mostresearch has primarily viewed them as independent individuals focusing onassessing their capabilities through standardized benchmarks and enhancingtheir general intelligence. This perspective however tends to overlook thevital role of LLMs as user-centric services in human-AI collaboration. This gapin research becomes increasingly critical as LLMs become more integrated intopeoples everyday and professional interactions. This study addresses theimportant need to understand user satisfaction with LLMs by exploring four keyaspects: comprehending user intents scrutinizing user experiences addressingmajor user concerns about current LLM services and charting future researchpaths to bolster human-AI collaborations. Our study develops a taxonomy of 7user intents in LLM interactions grounded in analysis of real-world userinteraction logs and human verification. Subsequently we conduct a user surveyto gauge their satisfaction with LLM services encompassing usage frequencyexperiences across intents and predominant concerns. This survey compiling411 anonymous responses uncovers 11 first-hand insights into the current stateof user engagement with LLMs. Based on this empirical analysis we pinpoint 6future research directions prioritizing the user perspective in LLMdevelopments. This user-centered approach is essential for crafting LLMs thatare not just technologically advanced but also resonate with the intricaterealities of human interactions and real-world applications. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| Adapt/Exchange decisions or generic choices: Does framing influence how people integrate qualitatively different risks? |
|authors| Romy MüllerAlexander Blunk
|links| http://arxiv.org/abs/2401.08241v1 |
|updated| 2024-01-16 09:53:39 UTC |
|summary| In complex systems decision makers often have to consider qualitativelydifferent risks when choosing between options. Do their strategies ofintegrating these risks depend on the framing of problem contents In thepresent study participants were either instructed that they were choosingbetween two ways of solving a complex problem or between two generic options.The former was framed as a modular plant scenario that required choices betweenmodifying parameter settings in a current module Adapt and replacing themodule by another one Exchange. The risk was higher for Adapt to harm theproduct and for Exchange to harm the plant. These risks were presented asprobabilities and participants were either told that the consequences of bothrisks were equally severe content-same group or that harming the plant wasmuch worse content-different group. A third group made decisions based on thesame probabilities but received a generic task framing no-content group. Weexpected framing to affect risk integration leading the content-same group tomake different choices than the no-content group. Contrary to this hypothesisthese two groups were strikingly similar in their decision outcomes andstrategies but clearly differed from the content-different group. Thesefindings question whether ecological validity can be enhanced merely by framinga task in terms of real-world problem contents. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| EEG-based Cognitive Load Estimation of Acoustic Parameters for Data Sonification |
|authors| Gulshan SharmaSurbhi MadanManeesh BilalpurAbhinav DhallRamanathan Subramanian
|links| http://arxiv.org/abs/2401.08164v1 |
|updated| 2024-01-16 07:11:14 UTC |
|summary| Sonification is a data visualization technique which expresses dataattributes via psychoacoustic parameters which are non-speech audio signalsused to convey information. This paper investigates the binary estimation ofcognitive load induced by psychoacoustic parameters conveying the focus levelof an astronomical image via Electroencephalogram EEG embeddings. Employingmachine learning and deep learning methodologies we demonstrate that EEGsignals are reliable for a binary estimation of cognitive load b isolatingeasy vs difficult visual-to-auditory perceptual mappings and c capturingperceptual similarities among psychoacoustic parameters. Our key findingsreveal that 1 EEG embeddings can reliably measure cognitive load achieving apeak F1-score of 0.98 2 Extreme focus levels are easier to detect viaauditory mappings than intermediate ones and 3 psychoacoustic parametersinducing comparable cognitive load levels tend to generate similar EEGencodings. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| 'One Style Does Not Regulate All': Moderation Practices in Public and Private WhatsApp Groups |
|authors| Farhana ShahidDhruv AgarwalAditya Vashistha
|links| http://arxiv.org/abs/2401.08091v1 |
|updated| 2024-01-16 03:32:15 UTC |
|summary| WhatsApp is the largest social media platform in the Global South and is avirulent force in global misinformation and political propaganda. Due toend-to-end encryption WhatsApp can barely review any content and this oftenpushes the responsibility of moderation towards group admins. Yet little isknown about how WhatsApp group admins manage their groups what factors andvalues influence moderation decisions and what challenges they face inmoderating their groups. To fill this gap we interviewed admins of 32 diversegroups and reviewed content from 30 public groups in India and Bangladesh. Weobserved notable differences in the formation members behavior andmoderation of public versus private groups as well as in how WhatsApp adminsoperate compared to those on other platforms. We used Baumrinds typology ofparenting styles as a lens to explore moderation practices in WhatsApp groupsand identified four moderation styles based on how responsive and controllingthe admins were and discuss design recommendations to help them better manageproblematic content in WhatsApp groups. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 1 |
|title| A Day-to-Day Dynamical Approach to the Most Likely User Equilibrium Problem |
|authors| Jiayang LiQianni WangLiyang FengJun XieYu Marco Nie
|links| http://arxiv.org/abs/2401.08013v1 |
|updated| 2024-01-15 23:43:41 UTC |
|summary| The lack of a unique user equilibrium UE route flow in traffic assignmenthas posed a significant challenge to many transportation applications. Themaximum-entropy principle which advocates for the consistent selection of themost likely solution as a representative is often used to address thechallenge. Built on a recently proposed day-to-day DTD discrete-timedynamical model called cumulative logit CULO this study provides a newbehavioral underpinning for the maximum-entropy UE MEUE route flow. It hasbeen proven that CULO can reach a UE state without presuming travelers areperfectly rational. Here we further establish that CULO always converges tothe MEUE route flow if i travelers have zero prior information about routesand thus are forced to give all routes an equal choice probability or ii alltravelers gather information from the same source such that the so-calledgeneral proportionality condition is satisfied. Thus CULO may be used as apractical solution algorithm for the MEUE problem. To put this idea intopractice we propose to eliminate the route enumeration requirement of theoriginal CULO model through an iterative route discovery scheme. We alsoexamine the discrete-time versions of four popular continuous-time dynamicalmodels and compare them to CULO. The analysis shows that the replicator dynamicis the only one that has the potential to reach the MEUE solution with someregularity. The analytical results are confirmed through numerical experiments. |


| Item |Content|
| --- |---|
|idx| 2 |
|title| Emergency Localization for Mobile Ground Users: An Adaptive UAV Trajectory Planning Method |
|authors| Zhihao ZhuJiafan HeLuyang HouLianming XuWendi ZhuLi Wang
|links| http://arxiv.org/abs/2401.07256v1 |
|updated| 2024-01-14 11:20:20 UTC |
|summary| In emergency search and rescue scenarios the quick location of trappedpeople is essential. However disasters can render the Global PositioningSystem GPS unusable. Unmanned aerial vehicles UAVs with localizationdevices can serve as mobile anchors due to their agility and high line-of-sightLoS probability. Nonetheless the number of available UAVs during the initialstages of disaster relief is limited and innovative methods are needed toquickly plan UAV trajectories to locate non-uniformly distributed dynamictargets while ensuring localization accuracy. To address this challenge wedesign a single UAV localization method without hovering use the maximumlikelihood estimation MLE method to estimate the location of mobile users anddefine the upper bound of the localization error by considering usersmovement.Combining this localization method and localization error-index weutilize the enhanced particle swarm optimization EPSO algorithm and edgeaccess strategy to develop a low complexity localization-oriented adaptivetrajectory planning algorithm. Simulation results demonstrate that our methodoutperforms other baseline algorithms enabling faster localization withoutcompromising localization accuracy. |


| Item |Content|
| --- |---|
|idx| 3 |
|title| Trust from Ethical Point of View: Exploring Dynamics Through Multiagent-Driven Cognitive Modeling |
|authors| Abbas Tariverdi
|links| http://arxiv.org/abs/2401.07255v1 |
|updated| 2024-01-14 11:19:18 UTC |
|summary| The paper begins by exploring the rationality of ethical trust as afoundational concept. This involves distinguishing between trust andtrustworthiness and delving into scenarios where trust is both rational andmoral. It lays the groundwork for understanding the complexities of trustdynamics in decision-making scenarios. Following this theoretical groundworkwe introduce an agent-based simulation framework that investigates thesedynamics of ethical trust specifically in the context of a disaster responsescenario. These agents utilizing emotional models like Plutchiks Wheel ofEmotions and memory learning mechanisms are tasked with allocating limitedresources in disaster-affected areas. The model which embodies the principlesdiscussed in the first section integrates cognitive load management Big Fivepersonality traits and structured interactions within networked orhierarchical settings. It also includes feedback loops and simulates externalevents to evaluate their impact on the formation and evolution of trust amongagents. Through our simulations we demonstrate the intricate interplay ofcognitive emotional and social factors in ethical decision-making. Theseinsights shed light on the behaviors and resilience of trust networks in crisissituations emphasizing the role of rational and moral considerations in thedevelopment of trust among autonomous agents. This study contributes to thefield by offering an understanding of trust dynamics in socio-technical systemsand by providing a robust adaptable framework capable of addressing ethicaldilemmas in disaster response and beyond. The implementation of the algorithmspresented in this paper is available at this GitHub repository:urlhttps://github.com/abbas-tari/ethical-trust-cognitive-modeling. |


| Item |Content|
| --- |---|
|idx| 4 |
|title| A Dynamic Agent Based Model of the Real Economy with Monopolistic Competition, Perfect Product Differentiation, Heterogeneous Agents, Increasing Returns to Scale and Trade in Disequilibrium |
|authors| Subhamon SupanthaNaresh Kumar Sharma
|links| http://arxiv.org/abs/2401.07070v1 |
|updated| 2024-01-13 13:10:20 UTC |
|summary| We have used agent-based modeling as our numerical method to artificiallysimulate a dynamic real economy where agents are rational maximizers of anobjective function of Cobb-Douglas type. The economy is characterised byheterogeneous agents acting out of local or imperfect informationmonopolistic competition perfect product differentiation allowance forincreasing returns to scale technology and trade in disequilibrium. Analgorithm for economic activity in each period is devised and a general purposeopen source agent-based model is developed which allows for counterfactualinquiries testing out treatments analysing causality of various economicprocesses outcomes and studying emergent properties. 10000 simulations with10 firms and 80 consumers are run with varying parameters and the results showthat from only a few initial conditions the economy reaches equilibrium whilein most of the other cases it remains in perpetual disequilibrium. It alsoshows that from a few initial conditions the economy reaches a disaster whereall the consumer wealth falls to zero or only a single producer remains.Furthermore from some initial conditions an ideal economy with high wagerate high consumer utility and no unemployment is also reached. It was alsoobserved that starting from an equal endowment of wealth in consumers and inproducers inequality emerged in the economy. In majority of the cases most ofthe firms6-7 shut down because they were not profitable enough and only a fewfirms remained. Our results highlight that all these varying outcomes arepossible for a decentralized market economy with rational optimizing agents. |


| Item |Content|
| --- |---|
|idx| 5 |
|title| Aquarium: A Comprehensive Framework for Exploring Predator-Prey Dynamics through Multi-Agent Reinforcement Learning Algorithms |
|authors| Michael KölleYannick ErpeldingFabian RitzThomy PhanSteffen IlliumClaudia Linnhoff-Popien
|links| http://arxiv.org/abs/2401.07056v1 |
|updated| 2024-01-13 12:09:49 UTC |
|summary| Recent advances in Multi-Agent Reinforcement Learning have prompted themodeling of intricate interactions between agents in simulated environments. Inparticular the predator-prey dynamics have captured substantial interest andvarious simulations been tailored to unique requirements. To prevent furthertime-intensive developments we introduce Aquarium a comprehensive Multi-AgentReinforcement Learning environment for predator-prey interaction enabling thestudy of emergent behavior. Aquarium is open source and offers a seamlessintegration of the PettingZoo framework allowing a quick start with provenalgorithm implementations. It features physics-based agent movement on atwo-dimensional edge-wrapping plane. The agent-environment interactionobservations actions rewards and the environment settings agent speedprey reproduction predator starvation and others are fully customizable.Besides a resource-efficient visualization Aquarium supports to record videofiles providing a visual comprehension of agent behavior. To demonstrate theenvironments capabilities we conduct preliminary studies which use PPO totrain multiple prey agents to evade a predator. In accordance to theliterature we find Individual Learning to result in worse performance thanParameter Sharing which significantly improves coordination andsample-efficiency. |

