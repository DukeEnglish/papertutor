# cs.CL 

| Item |Content|
| --- |---|
|idx| 2403.04746v1 |
|title| LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error |
|authors| Boshi WangHao FangJason EisnerBenjamin Van DurmeYu Su
|links| http://arxiv.org/abs/2403.04746v1 |
|updated| 2024-03-07 18:50:51 UTC |
|summary| Tools are essential for large language models LLMs to acquire up-to-dateinformation and take consequential actions in external environments. Existingwork on tool-augmented LLMs primarily focuses on the broad coverage of toolsand the flexibility of adding new tools. However a critical aspect that hassurprisingly been understudied is simply how accurately an LLM uses tools forwhich it has been trained. We find that existing LLMs including GPT-4 andopen-source LLMs specifically fine-tuned for tool use only reach a correctnessrate in the range of 30 to 60 far from reliable use in practice. We proposea biologically inspired method for tool-augmented LLMs simulated trial anderror STE that orchestrates three key mechanisms for successful tool usebehaviors in the biological system: trial and error imagination and memory.Specifically STE leverages an LLMs imagination to simulate plausiblescenarios for using a tool after which the LLM interacts with the tool tolearn from its execution feedback. Both short-term and long-term memory areemployed to improve the depth and breadth of the exploration respectively.Comprehensive experiments on ToolBench show that STE substantially improvestool learning for LLMs under both in-context learning and fine-tuning settingsbringing a boost of 46.7 to Mistral-Instruct-7B and enabling it to outperformGPT-4. We also show effective continual learning of tools via a simpleexperience replay strategy. |


| Item |Content|
| --- |---|
|idx| 2403.04732v1 |
|title| How Far Are We from Intelligent Visual Deductive Reasoning? |
|authors| Yizhe ZhangHe BaiRuixiang ZhangJiatao GuShuangfei ZhaiJosh SusskindNavdeep Jaitly
|links| http://arxiv.org/abs/2403.04732v1 |
|updated| 2024-03-07 18:35:54 UTC |
|summary| Vision-Language Models VLMs such as GPT-4V have recently demonstratedincredible strides on diverse vision language tasks. We dig into vision-baseddeductive reasoning a more sophisticated but less explored realm and findpreviously unexposed blindspots in the current SOTA VLMs. Specifically weleverage Ravens Progressive Matrices RPMs to assess VLMs abilities toperform multi-hop relational and deductive reasoning relying solely on visualclues. We perform comprehensive evaluations of several popular VLMs employingstandard strategies such as in-context learning self-consistency andChain-of-thoughts CoT on three diverse datasets including the Mensa IQ testIntelligenceTest and RAVEN. The results reveal that despite the impressivecapabilities of LLMs in text-based reasoning we are still far from achievingcomparable proficiency in visual deductive reasoning. We found that certainstandard strategies that are effective when applied to LLMs do not seamlesslytranslate to the challenges presented by visual reasoning tasks. Moreover adetailed analysis reveals that VLMs struggle to solve these tasks mainlybecause they are unable to perceive and comprehend multiple confoundingabstract patterns in RPM examples. |


| Item |Content|
| --- |---|
|idx| 2403.04706v1 |
|title| Common 7B Language Models Already Possess Strong Math Capabilities |
|authors| Chen LiWeiqi WangJingcheng HuYixuan WeiNanning ZhengHan HuZheng ZhangHouwen Peng
|links| http://arxiv.org/abs/2403.04706v1 |
|updated| 2024-03-07 18:00:40 UTC |
|summary| Mathematical capabilities were previously believed to emerge in commonlanguage models only at a very large scale or require extensive math-relatedpre-training. This paper shows that the LLaMA-2 7B model with commonpre-training already exhibits strong mathematical abilities as evidenced byits impressive accuracy of 97.7 and 72.0 on the GSM8K and MATH benchmarksrespectively when selecting the best response from 256 random generations. Theprimary issue with the current base model is the difficulty in consistentlyeliciting its inherent mathematical capabilities. Notably the accuracy for thefirst answer drops to 49.5 and 7.9 on the GSM8K and MATH benchmarksrespectively. We find that simply scaling up the SFT data can significantlyenhance the reliability of generating correct answers. However the potentialfor extensive scaling is constrained by the scarcity of publicly available mathquestions. To overcome this limitation we employ synthetic data which provesto be nearly as effective as real data and shows no clear saturation whenscaled up to approximately one million samples. This straightforward approachachieves an accuracy of 82.6 on GSM8K and 40.6 on MATH using LLaMA-2 7Bmodels surpassing previous models by 14.2 and 20.8 respectively. We alsoprovide insights into scaling behaviors across different reasoning complexitiesand error types. |


| Item |Content|
| --- |---|
|idx| 2403.04696v1 |
|title| Fact-Checking the Output of Large Language Models via Token-Level Uncertainty Quantification |
|authors| Ekaterina FadeevaAleksandr RubashevskiiArtem ShelmanovSergey PetrakovHaonan LiHamdy MubarakEvgenii TsymbalovGleb KuzminAlexander PanchenkoTimothy BaldwinPreslav NakovMaxim Panov
|links| http://arxiv.org/abs/2403.04696v1 |
|updated| 2024-03-07 17:44:17 UTC |
|summary| Large language models LLMs are notorious for hallucinating i.e. producingerroneous claims in their output. Such hallucinations can be dangerous asoccasional factual inaccuracies in the generated text might be obscured by therest of the output being generally factual making it extremely hard for theusers to spot them. Current services that leverage LLMs usually do not provideany means for detecting unreliable generations. Here we aim to bridge thisgap. In particular we propose a novel fact-checking and hallucinationdetection pipeline based on token-level uncertainty quantification. Uncertaintyscores leverage information encapsulated in the output of a neural network orits layers to detect unreliable predictions and we show that they can be usedto fact-check the atomic claims in the LLM output. Moreover we present a noveltoken-level uncertainty quantification method that removes the impact ofuncertainty about what claim to generate on the current step and what surfaceform to use. Our method Claim Conditioned Probability CCP measures only theuncertainty of particular claim value expressed by the model. Experiments onthe task of biography generation demonstrate strong improvements for CCPcompared to the baselines for six different LLMs and three languages. Humanevaluation reveals that the fact-checking pipeline based on uncertaintyquantification is competitive with a fact-checking tool that leverages externalknowledge. |


| Item |Content|
| --- |---|
|idx| 2403.04671v1 |
|title| Greater than the sum of its parts: The role of minority and majority status in collaborative problem-solving communication |
|authors| Jacqueline G. CavazosNia Nixon
|links| http://arxiv.org/abs/2403.04671v1 |
|updated| 2024-03-07 17:17:20 UTC |
|summary| Collaborative problem-solving CPS is a vital skill used both in theworkplace and in educational environments. CPS is useful in tacklingincreasingly complex global economic and political issues and is considered acentral 21st century skill. The increasingly connected global communitypresents a fruitful opportunity for creative and collaborative problem-solvinginteractions and solutions that involve diverse perspectives. Unfortunatelywomen and underrepresented minorities URMs often face obstacles duringcollaborative interactions that hinder their key participation in theseproblem-solving conversations. Here we explored the communication patterns ofminority and non-minority individuals working together in a CPS task. GroupCommunication Analysis GCA a temporally-sensitive computational linguistictool was used to examine how URM status impacts individuals sociocognitivelinguistic patterns. Results show differences across racial/ethnic groups inkey sociocognitive features that indicate fruitful collaborative interactions.We also investigated how the groups racial/ethnic composition impacts bothindividual and group communication patterns. In general individuals in moredemographically diverse groups displayed more productive communicationbehaviors than individuals who were in majority-dominated groups. We discussthe implications of individual and group diversity on communication patternsthat emerge during CPS and how these patterns can impact collaborativeoutcomes. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2403.04760v1 |
|title| iScore: Visual Analytics for Interpreting How Language Models Automatically Score Summaries |
|authors| Adam CosciaLangdon HolmesWesley MorrisJoon Suh ChoiScott CrossleyAlex Endert
|links| http://dx.doi.org/10.1145/3640543.3645142 |
|updated| 2024-03-07 18:56:39 UTC |
|summary| The recent explosion in popularity of large language models LLMs hasinspired learning engineers to incorporate them into adaptive educational toolsthat automatically score summary writing. Understanding and evaluating LLMs isvital before deploying them in critical learning environments yet theirunprecedented size and expanding number of parameters inhibits transparency andimpedes trust when they underperform. Through a collaborative user-centereddesign process with several learning engineers building and deploying summaryscoring LLMs we characterized fundamental design challenges and goals aroundinterpreting their models including aggregating large text inputs trackingscore provenance and scaling LLM interpretability methods. To address theirconcerns we developed iScore an interactive visual analytics tool forlearning engineers to upload score and compare multiple summariessimultaneously. Tightly integrated views allow users to iteratively revise thelanguage in summaries track changes in the resulting LLM scores and visualizemodel weights at multiple levels of abstraction. To validate our approach wedeployed iScore with three learning engineers over the course of a month. Wepresent a case study where interacting with iScore led a learning engineer toimprove their LLMs score accuracy by three percentage points. Finally weconducted qualitative interviews with the learning engineers that revealed howiScore enabled them to understand evaluate and build trust in their LLMsduring deployment. |


| Item |Content|
| --- |---|
|idx| 2403.04758v1 |
|title| KnowledgeVIS: Interpreting Language Models by Comparing Fill-in-the-Blank Prompts |
|authors| Adam CosciaAlex Endert
|links| http://dx.doi.org/10.1109/TVCG.2023.3346713 |
|updated| 2024-03-07 18:56:31 UTC |
|summary| Recent growth in the popularity of large language models has led to theirincreased usage for summarizing predicting and generating text making itvital to help researchers and engineers understand how and why they work. Wepresent KnowledgeVis a human-in-the-loop visual analytics system forinterpreting language models using fill-in-the-blank sentences as prompts. Bycomparing predictions between sentences KnowledgeVis reveals learnedassociations that intuitively connect what language models learn duringtraining to natural language tasks downstream helping users create and testmultiple prompt variations analyze predicted words using a novel semanticclustering technique and discover insights using interactive visualizations.Collectively these visualizations help users identify the likelihood anduniqueness of individual predictions compare sets of predictions betweenprompts and summarize patterns and relationships between predictions acrossall prompts. We demonstrate the capabilities of KnowledgeVis with feedback fromsix NLP experts as well as three different use cases: 1 probing biomedicalknowledge in two domain-adapted models and 2 evaluating harmful identitystereotypes and 3 discovering facts and relationships between threegeneral-purpose models. |


| Item |Content|
| --- |---|
|idx| 2403.04747v1 |
|title| GNN-VPA: A Variance-Preserving Aggregation Strategy for Graph Neural Networks |
|authors| Lisa SchneckenreiterRichard FreinschlagFlorian SestakJohannes BrandstetterGünter KlambauerAndreas Mayr
|links| http://arxiv.org/abs/2403.04747v1 |
|updated| 2024-03-07 18:52:27 UTC |
|summary| Graph neural networks GNNs and especially message-passing neural networksexcel in various domains such as physics drug discovery and molecularmodeling. The expressivity of GNNs with respect to their ability todiscriminate non-isomorphic graphs critically depends on the functions employedfor message aggregation and graph-level readout. By applying signal propagationtheory we propose a variance-preserving aggregation function VPA thatmaintains expressivity but yields improved forward and backward dynamics.Experiments demonstrate that VPA leads to increased predictive performance forpopular GNN architectures as well as improved learning dynamics. Our resultscould pave the way towards normalizer-free or self-normalizing GNNs. |


| Item |Content|
| --- |---|
|idx| 2403.04746v1 |
|title| LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error |
|authors| Boshi WangHao FangJason EisnerBenjamin Van DurmeYu Su
|links| http://arxiv.org/abs/2403.04746v1 |
|updated| 2024-03-07 18:50:51 UTC |
|summary| Tools are essential for large language models LLMs to acquire up-to-dateinformation and take consequential actions in external environments. Existingwork on tool-augmented LLMs primarily focuses on the broad coverage of toolsand the flexibility of adding new tools. However a critical aspect that hassurprisingly been understudied is simply how accurately an LLM uses tools forwhich it has been trained. We find that existing LLMs including GPT-4 andopen-source LLMs specifically fine-tuned for tool use only reach a correctnessrate in the range of 30 to 60 far from reliable use in practice. We proposea biologically inspired method for tool-augmented LLMs simulated trial anderror STE that orchestrates three key mechanisms for successful tool usebehaviors in the biological system: trial and error imagination and memory.Specifically STE leverages an LLMs imagination to simulate plausiblescenarios for using a tool after which the LLM interacts with the tool tolearn from its execution feedback. Both short-term and long-term memory areemployed to improve the depth and breadth of the exploration respectively.Comprehensive experiments on ToolBench show that STE substantially improvestool learning for LLMs under both in-context learning and fine-tuning settingsbringing a boost of 46.7 to Mistral-Instruct-7B and enabling it to outperformGPT-4. We also show effective continual learning of tools via a simpleexperience replay strategy. |


| Item |Content|
| --- |---|
|idx| 2403.04732v1 |
|title| How Far Are We from Intelligent Visual Deductive Reasoning? |
|authors| Yizhe ZhangHe BaiRuixiang ZhangJiatao GuShuangfei ZhaiJosh SusskindNavdeep Jaitly
|links| http://arxiv.org/abs/2403.04732v1 |
|updated| 2024-03-07 18:35:54 UTC |
|summary| Vision-Language Models VLMs such as GPT-4V have recently demonstratedincredible strides on diverse vision language tasks. We dig into vision-baseddeductive reasoning a more sophisticated but less explored realm and findpreviously unexposed blindspots in the current SOTA VLMs. Specifically weleverage Ravens Progressive Matrices RPMs to assess VLMs abilities toperform multi-hop relational and deductive reasoning relying solely on visualclues. We perform comprehensive evaluations of several popular VLMs employingstandard strategies such as in-context learning self-consistency andChain-of-thoughts CoT on three diverse datasets including the Mensa IQ testIntelligenceTest and RAVEN. The results reveal that despite the impressivecapabilities of LLMs in text-based reasoning we are still far from achievingcomparable proficiency in visual deductive reasoning. We found that certainstandard strategies that are effective when applied to LLMs do not seamlesslytranslate to the challenges presented by visual reasoning tasks. Moreover adetailed analysis reveals that VLMs struggle to solve these tasks mainlybecause they are unable to perceive and comprehend multiple confoundingabstract patterns in RPM examples. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2403.04764v1 |
|title| Minimizing the Thompson Sampling Regret-to-Sigma Ratio (TS-RSR): a provably efficient algorithm for batch Bayesian Optimization |
|authors| Zhaolin RenNa Li
|links| http://arxiv.org/abs/2403.04764v1 |
|updated| 2024-03-07 18:58:26 UTC |
|summary| This paper presents a new approach for batch Bayesian Optimization BOwhere the sampling takes place by minimizing a Thompson Sampling approximationof a regret to uncertainty ratio. Our objective is able to coordinate theactions chosen in each batch in a way that minimizes redundancy between pointswhilst focusing on points with high predictive means or high uncertainty. Weprovide high-probability theoretical guarantees on the regret of our algorithm.Finally numerically we demonstrate that our method attains state-of-the-artperformance on a range of nonconvex test functions where it outperformsseveral competitive benchmark batch BO algorithms by an order of magnitude onaverage. |


| Item |Content|
| --- |---|
|idx| 2403.04763v1 |
|title| BloomGML: Graph Machine Learning through the Lens of Bilevel Optimization |
|authors| Amber Yijia ZhengTong HeYixuan QiuMinjie WangDavid Wipf
|links| http://arxiv.org/abs/2403.04763v1 |
|updated| 2024-03-07 18:57:46 UTC |
|summary| Bilevel optimization refers to scenarios whereby the optimal solution of alower-level energy function serves as input features to an upper-levelobjective of interest. These optimal features typically depend on tunableparameters of the lower-level energy in such a way that the entire bilevelpipeline can be trained end-to-end. Although not generally presented as suchthis paper demonstrates how a variety of graph learning techniques can berecast as special cases of bilevel optimization or simplifications thereof. Inbrief building on prior work we first derive a more flexible class of energyfunctions that when paired with various descent steps e.g. gradient descentproximal methods momentum etc. form graph neural network GNNmessage-passing layers critically we also carefully unpack where any residualapproximation error lies with respect to the underlying constituentmessage-passing functions. We then probe several simplifications of thisframework to derive close connections with non-GNN-based graph learningapproaches including knowledge graph embeddings various forms of labelpropagation and efficient graph-regularized MLP models. And finally wepresent supporting empirical results that demonstrate the versatility of theproposed bilevel lens which we refer to as BloomGML referencing that BiLevelOptimization Offers More Graph Machine Learning. Our code is available athttps://github.com/amberyzheng/BloomGML. Let graph ML bloom. |


| Item |Content|
| --- |---|
|idx| 2403.04760v1 |
|title| iScore: Visual Analytics for Interpreting How Language Models Automatically Score Summaries |
|authors| Adam CosciaLangdon HolmesWesley MorrisJoon Suh ChoiScott CrossleyAlex Endert
|links| http://dx.doi.org/10.1145/3640543.3645142 |
|updated| 2024-03-07 18:56:39 UTC |
|summary| The recent explosion in popularity of large language models LLMs hasinspired learning engineers to incorporate them into adaptive educational toolsthat automatically score summary writing. Understanding and evaluating LLMs isvital before deploying them in critical learning environments yet theirunprecedented size and expanding number of parameters inhibits transparency andimpedes trust when they underperform. Through a collaborative user-centereddesign process with several learning engineers building and deploying summaryscoring LLMs we characterized fundamental design challenges and goals aroundinterpreting their models including aggregating large text inputs trackingscore provenance and scaling LLM interpretability methods. To address theirconcerns we developed iScore an interactive visual analytics tool forlearning engineers to upload score and compare multiple summariessimultaneously. Tightly integrated views allow users to iteratively revise thelanguage in summaries track changes in the resulting LLM scores and visualizemodel weights at multiple levels of abstraction. To validate our approach wedeployed iScore with three learning engineers over the course of a month. Wepresent a case study where interacting with iScore led a learning engineer toimprove their LLMs score accuracy by three percentage points. Finally weconducted qualitative interviews with the learning engineers that revealed howiScore enabled them to understand evaluate and build trust in their LLMsduring deployment. |


| Item |Content|
| --- |---|
|idx| 2403.04759v1 |
|title| Lifelong Intelligence Beyond the Edge using Hyperdimensional Computing |
|authors| Xiaofan YuAnthony ThomasIvannia Gomez MorenoLouis GutierrezTajana Rosing
|links| http://arxiv.org/abs/2403.04759v1 |
|updated| 2024-03-07 18:56:33 UTC |
|summary| On-device learning has emerged as a prevailing trend that avoids the slowresponse time and costly communication of cloud-based learning. The ability tolearn continuously and indefinitely in a changing environment and withresource constraints is critical for real sensor deployments. Howeverexisting designs are inadequate for practical scenarios with i streaming datainput ii lack of supervision and iii limited on-board resources. In thispaper we design and deploy the first on-device lifelong learning system calledLifeHD for general IoT applications with limited supervision. LifeHD isdesigned based on a novel neurally-inspired and lightweight learning paradigmcalled Hyperdimensional Computing HDC. We utilize a two-tier associativememory organization to intelligently store and manage high-dimensionallow-precision vectors which represent the historical patterns as clustercentroids. We additionally propose two variants of LifeHD to cope with scarcelabeled inputs and power constraints. We implement LifeHD on off-the-shelf edgeplatforms and perform extensive evaluations across three scenarios. Ourmeasurements show that LifeHD improves the unsupervised clustering accuracy byup to 74.8 compared to the state-of-the-art NN-based unsupervised lifelonglearning baselines with as much as 34.3x better energy efficiency. Our code isavailable at https://github.com/Orienfish/LifeHD. |


| Item |Content|
| --- |---|
|idx| 2403.04758v1 |
|title| KnowledgeVIS: Interpreting Language Models by Comparing Fill-in-the-Blank Prompts |
|authors| Adam CosciaAlex Endert
|links| http://dx.doi.org/10.1109/TVCG.2023.3346713 |
|updated| 2024-03-07 18:56:31 UTC |
|summary| Recent growth in the popularity of large language models has led to theirincreased usage for summarizing predicting and generating text making itvital to help researchers and engineers understand how and why they work. Wepresent KnowledgeVis a human-in-the-loop visual analytics system forinterpreting language models using fill-in-the-blank sentences as prompts. Bycomparing predictions between sentences KnowledgeVis reveals learnedassociations that intuitively connect what language models learn duringtraining to natural language tasks downstream helping users create and testmultiple prompt variations analyze predicted words using a novel semanticclustering technique and discover insights using interactive visualizations.Collectively these visualizations help users identify the likelihood anduniqueness of individual predictions compare sets of predictions betweenprompts and summarize patterns and relationships between predictions acrossall prompts. We demonstrate the capabilities of KnowledgeVis with feedback fromsix NLP experts as well as three different use cases: 1 probing biomedicalknowledge in two domain-adapted models and 2 evaluating harmful identitystereotypes and 3 discovering facts and relationships between threegeneral-purpose models. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2403.04765v1 |
|title| Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed |
|authors| Yifan WangXingyi HeSida PengDongli TanXiaowei Zhou
|links| http://arxiv.org/abs/2403.04765v1 |
|updated| 2024-03-07 18:58:40 UTC |
|summary| We present a novel method for efficiently producing semi-dense matches acrossimages. Previous detector-free matcher LoFTR has shown remarkable matchingcapability in handling large-viewpoint change and texture-poor scenarios butsuffers from low efficiency. We revisit its design choices and derive multipleimprovements for both efficiency and accuracy. One key observation is thatperforming the transformer over the entire feature map is redundant due toshared local information therefore we propose an aggregated attentionmechanism with adaptive token selection for efficiency. Furthermore we findspatial variance exists in LoFTRs fine correlation module which is adverse tomatching accuracy. A novel two-stage correlation layer is proposed to achieveaccurate subpixel correspondences for accuracy improvement. Our efficiencyoptimized model is sim 2.5times faster than LoFTR which can even surpassstate-of-the-art efficient sparse matching pipeline SuperPoint  LightGlue.Moreover extensive experiments show that our method can achieve higheraccuracy compared with competitive semi-dense matchers with considerableefficiency benefits. This opens up exciting prospects for large-scale orlatency-sensitive applications such as image retrieval and 3D reconstruction.Project page: https://zju3dv.github.io/efficientloftr. |


| Item |Content|
| --- |---|
|idx| 2403.04755v1 |
|title| That's My Point: Compact Object-centric LiDAR Pose Estimation for Large-scale Outdoor Localisation |
|authors| Georgi PramatarovMatthew GaddPaul NewmanDaniele De Martini
|links| http://arxiv.org/abs/2403.04755v1 |
|updated| 2024-03-07 18:55:30 UTC |
|summary| This paper is about 3D pose estimation on LiDAR scans with extremely minimalstorage requirements to enable scalable mapping and localisation. We achievethis by clustering all points of segmented scans into semantic objects andrepresenting them only with their respective centroid and semantic class. Inthis way each LiDAR scan is reduced to a compact collection of four-numbervectors. This abstracts away important structural information from the sceneswhich is crucial for traditional registration approaches. To mitigate this weintroduce an object-matching network based on self- and cross-correlation thatcaptures geometric and semantic relationships between entities. The respectivematches allow us to recover the relative transformation between scans throughweighted Singular Value Decomposition SVD and RANdom SAmple ConsensusRANSAC. We demonstrate that such representation is sufficient for metriclocalisation by registering point clouds taken under different viewpoints onthe KITTI dataset and at different periods of time localising between KITTIand KITTI-360. We achieve accurate metric estimates comparable withstate-of-the-art methods with almost half the representation size specifically1.33 kB on average. |


| Item |Content|
| --- |---|
|idx| 2403.04739v1 |
|title| I Can't Believe It's Not Scene Flow! |
|authors| Ishan KhatriKyle VedderNeehar PeriDeva RamananJames Hays
|links| http://arxiv.org/abs/2403.04739v1 |
|updated| 2024-03-07 18:46:01 UTC |
|summary| Current scene flow methods broadly fail to describe motion on small objectsand current scene flow evaluation protocols hide this failure by averaging overmany points with most drawn larger objects. To fix this evaluation failure wepropose a new evaluation protocol Bucket Normalized EPE which is class-awareand speed-normalized enabling contextualized error comparisons between objecttypes that move at vastly different speeds. To highlight current methodfailures we propose a frustratingly simple supervised scene flow baselineTrackFlow built by bolting a high-quality pretrained detector trained usingmany class rebalancing techniques onto a simple tracker that producesstate-of-the-art performance on current standard evaluations and largeimprovements over prior art on our new evaluation. Our results make it clearthat all scene flow evaluations must be class and speed aware and supervisedscene flow methods must address point class imbalances. We release theevaluation code publicly athttps://github.com/kylevedder/BucketedSceneFlowEval. |


| Item |Content|
| --- |---|
|idx| 2403.04735v1 |
|title| SnapNTell: Enhancing Entity-Centric Visual Question Answering with Retrieval Augmented Multimodal LLM |
|authors| Jielin QiuAndrea MadottoZhaojiang LinPaul A. CrookYifan Ethan XuXin Luna DongChristos FaloutsosLei LiBabak DamavandiSeungwhan Moon
|links| http://arxiv.org/abs/2403.04735v1 |
|updated| 2024-03-07 18:38:17 UTC |
|summary| Vision-extended LLMs have made significant strides in Visual QuestionAnswering VQA. Despite these advancements VLLMs still encounter substantialdifficulties in handling queries involving long-tail entities with a tendencyto produce erroneous or hallucinated responses. In this work we introduce anovel evaluative benchmark named textbfSnapNTell specifically tailored forentity-centric VQA. This task aims to test the models capabilities inidentifying entities and providing detailed entity-specific knowledge. We havedeveloped the textbfSnapNTell Dataset distinct from traditional VQAdatasets: 1 It encompasses a wide range of categorized entities eachrepresented by images and explicitly named in the answers 2 It features QApairs that require extensive knowledge for accurate responses. The dataset isorganized into 22 major categories containing 7568 unique entities in total.For each entity we curated 10 illustrative images and crafted 10knowledge-intensive QA pairs. To address this novel task we devised ascalable efficient and transparent retrieval-augmented multimodal LLM. Ourapproach markedly outperforms existing methods on the SnapNTell datasetachieving a 66.5 improvement in the BELURT score. We will soon make thedataset and the source code publicly accessible. |


| Item |Content|
| --- |---|
|idx| 2403.04732v1 |
|title| How Far Are We from Intelligent Visual Deductive Reasoning? |
|authors| Yizhe ZhangHe BaiRuixiang ZhangJiatao GuShuangfei ZhaiJosh SusskindNavdeep Jaitly
|links| http://arxiv.org/abs/2403.04732v1 |
|updated| 2024-03-07 18:35:54 UTC |
|summary| Vision-Language Models VLMs such as GPT-4V have recently demonstratedincredible strides on diverse vision language tasks. We dig into vision-baseddeductive reasoning a more sophisticated but less explored realm and findpreviously unexposed blindspots in the current SOTA VLMs. Specifically weleverage Ravens Progressive Matrices RPMs to assess VLMs abilities toperform multi-hop relational and deductive reasoning relying solely on visualclues. We perform comprehensive evaluations of several popular VLMs employingstandard strategies such as in-context learning self-consistency andChain-of-thoughts CoT on three diverse datasets including the Mensa IQ testIntelligenceTest and RAVEN. The results reveal that despite the impressivecapabilities of LLMs in text-based reasoning we are still far from achievingcomparable proficiency in visual deductive reasoning. We found that certainstandard strategies that are effective when applied to LLMs do not seamlesslytranslate to the challenges presented by visual reasoning tasks. Moreover adetailed analysis reveals that VLMs struggle to solve these tasks mainlybecause they are unable to perceive and comprehend multiple confoundingabstract patterns in RPM examples. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2403.04764v1 |
|title| Minimizing the Thompson Sampling Regret-to-Sigma Ratio (TS-RSR): a provably efficient algorithm for batch Bayesian Optimization |
|authors| Zhaolin RenNa Li
|links| http://arxiv.org/abs/2403.04764v1 |
|updated| 2024-03-07 18:58:26 UTC |
|summary| This paper presents a new approach for batch Bayesian Optimization BOwhere the sampling takes place by minimizing a Thompson Sampling approximationof a regret to uncertainty ratio. Our objective is able to coordinate theactions chosen in each batch in a way that minimizes redundancy between pointswhilst focusing on points with high predictive means or high uncertainty. Weprovide high-probability theoretical guarantees on the regret of our algorithm.Finally numerically we demonstrate that our method attains state-of-the-artperformance on a range of nonconvex test functions where it outperformsseveral competitive benchmark batch BO algorithms by an order of magnitude onaverage. |


| Item |Content|
| --- |---|
|idx| 2403.04747v1 |
|title| GNN-VPA: A Variance-Preserving Aggregation Strategy for Graph Neural Networks |
|authors| Lisa SchneckenreiterRichard FreinschlagFlorian SestakJohannes BrandstetterGünter KlambauerAndreas Mayr
|links| http://arxiv.org/abs/2403.04747v1 |
|updated| 2024-03-07 18:52:27 UTC |
|summary| Graph neural networks GNNs and especially message-passing neural networksexcel in various domains such as physics drug discovery and molecularmodeling. The expressivity of GNNs with respect to their ability todiscriminate non-isomorphic graphs critically depends on the functions employedfor message aggregation and graph-level readout. By applying signal propagationtheory we propose a variance-preserving aggregation function VPA thatmaintains expressivity but yields improved forward and backward dynamics.Experiments demonstrate that VPA leads to increased predictive performance forpopular GNN architectures as well as improved learning dynamics. Our resultscould pave the way towards normalizer-free or self-normalizing GNNs. |


| Item |Content|
| --- |---|
|idx| 2403.04744v1 |
|title| SQ Lower Bounds for Non-Gaussian Component Analysis with Weaker Assumptions |
|authors| Ilias DiakonikolasDaniel KaneLisheng RenYuxin Sun
|links| http://arxiv.org/abs/2403.04744v1 |
|updated| 2024-03-07 18:49:32 UTC |
|summary| We study the complexity of Non-Gaussian Component Analysis NGCA in theStatistical Query SQ model. Prior work developed a general methodology toprove SQ lower bounds for this task that have been applicable to a wide rangeof contexts. In particular it was known that for any univariate distributionA satisfying certain conditions distinguishing between a standardmultivariate Gaussian and a distribution that behaves like A in a randomhidden direction and like a standard Gaussian in the orthogonal complement isSQ-hard. The required conditions were that 1 A matches many low-ordermoments with the standard univariate Gaussian and 2 the chi-squared norm ofA with respect to the standard Gaussian is finite. While the moment-matchingcondition is necessary for hardness the chi-squared condition was onlyrequired for technical reasons. In this work we establish that the lattercondition is indeed not necessary. In particular we prove near-optimal SQlower bounds for NGCA under the moment-matching condition only. Our resultnaturally generalizes to the setting of a hidden subspace. Leveraging ourgeneral SQ lower bound we obtain near-optimal SQ lower bounds for a range ofconcrete estimation tasks where existing techniques provide sub-optimal or evenvacuous guarantees. |


| Item |Content|
| --- |---|
|idx| 2403.04726v1 |
|title| A Sub-Quadratic Time Algorithm for Robust Sparse Mean Estimation |
|authors| Ankit Pensia
|links| http://arxiv.org/abs/2403.04726v1 |
|updated| 2024-03-07 18:23:51 UTC |
|summary| We study the algorithmic problem of sparse mean estimation in the presence ofadversarial outliers. Specifically the algorithm observes a emphcorruptedset of samples from mathcalNmumathbfI_d where the unknown meanmu in mathbbRd is constrained to be k-sparse. A series of prior workshas developed efficient algorithms for robust sparse mean estimation withsample complexity mathrmpolyklog d 1/epsilon and runtime d2mathrmpolyklog d1/epsilon where epsilon is the fraction ofcontamination. In particular the fastest runtime of existing algorithms isquadratic Omegad2 which can be prohibitive in high dimensions. Thisquadratic barrier in the runtime stems from the reliance of these algorithms onthe sample covariance matrix which is of size d2. Our main contribution isan algorithm for robust sparse mean estimation which runs inemphsubquadratic time using mathrmpolyklog d1/epsilon samples. Wealso provide analogous results for robust sparse PCA. Our results build onalgorithmic advances in detecting weak correlations a generalized version ofthe light-bulb problem by Valiant. |


| Item |Content|
| --- |---|
|idx| 2403.04629v1 |
|title| Explaining Bayesian Optimization by Shapley Values Facilitates Human-AI Collaboration |
|authors| Julian RodemannFederico CroppiPhilipp ArensYusuf SaleJulia HerbingerBernd BischlEyke HüllermeierThomas AugustinConor J. WalshGiuseppe Casalicchio
|links| http://arxiv.org/abs/2403.04629v1 |
|updated| 2024-03-07 16:13:32 UTC |
|summary| Bayesian optimization BO with Gaussian processes GP has become anindispensable algorithm for black box optimization problems. Not without a dashof irony BO is often considered a black box itself lacking ways to providereasons as to why certain parameters are proposed to be evaluated. This isparticularly relevant in human-in-the-loop applications of BO such as inrobotics. We address this issue by proposing ShapleyBO a framework forinterpreting BOs proposals by game-theoretic Shapley values.They quantify eachparameters contribution to BOs acquisition function. Exploiting the linearityof Shapley values we are further able to identify how strongly each parameterdrives BOs exploration and exploitation for additive acquisition functionslike the confidence bound. We also show that ShapleyBO can disentangle thecontributions to exploration into those that explore aleatoric and epistemicuncertainty. Moreover our method gives rise to a ShapleyBO-assisted humanmachine interface HMI allowing users to interfere with BO in case proposalsdo not align with human reasoning. We demonstrate this HMIs benefits for theuse case of personalizing wearable robotic devices assistive back exosuits byhuman-in-the-loop BO. Results suggest human-BO teams with access to ShapleyBOcan achieve lower regret than teams without. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2403.04761v1 |
|title| DeepSee: Multidimensional Visualizations of Seabed Ecosystems |
|authors| Adam CosciaHaley M. SapersNoah DeutschMalika KhuranaJohn S. MagyarSergio A. ParraDaniel R. UtterRebecca L. WipflerDavid W. CaressEric J. MartinJennifer B. PaduanMaggie HendrieSantiago LombeydaHillary MushkinAlex EndertScott DavidoffVictoria J. Orphan
|links| http://dx.doi.org/10.1145/3613904.3642001 |
|updated| 2024-03-07 18:56:47 UTC |
|summary| Scientists studying deep ocean microbial ecosystems use limited numbers ofsediment samples collected from the seafloor to characterize importantlife-sustaining biogeochemical cycles in the environment. Yet conductingfieldwork to sample these extreme remote environments is both expensive andtime consuming requiring tools that enable scientists to explore the samplinghistory of field sites and predict where taking new samples is likely tomaximize scientific return. We conducted a collaborative user-centered designstudy with a team of scientific researchers to develop DeepSee an interactivedata workspace that visualizes 2D and 3D interpolations of biogeochemical andmicrobial processes in context together with sediment sampling history overlaidon 2D seafloor maps. Based on a field deployment and qualitative interviews wefound that DeepSee increased the scientific return from limited sample sizescatalyzed new research workflows reduced long-term costs of sharing data andsupported teamwork and communication between team members with diverse researchgoals. |


| Item |Content|
| --- |---|
|idx| 2403.04760v1 |
|title| iScore: Visual Analytics for Interpreting How Language Models Automatically Score Summaries |
|authors| Adam CosciaLangdon HolmesWesley MorrisJoon Suh ChoiScott CrossleyAlex Endert
|links| http://dx.doi.org/10.1145/3640543.3645142 |
|updated| 2024-03-07 18:56:39 UTC |
|summary| The recent explosion in popularity of large language models LLMs hasinspired learning engineers to incorporate them into adaptive educational toolsthat automatically score summary writing. Understanding and evaluating LLMs isvital before deploying them in critical learning environments yet theirunprecedented size and expanding number of parameters inhibits transparency andimpedes trust when they underperform. Through a collaborative user-centereddesign process with several learning engineers building and deploying summaryscoring LLMs we characterized fundamental design challenges and goals aroundinterpreting their models including aggregating large text inputs trackingscore provenance and scaling LLM interpretability methods. To address theirconcerns we developed iScore an interactive visual analytics tool forlearning engineers to upload score and compare multiple summariessimultaneously. Tightly integrated views allow users to iteratively revise thelanguage in summaries track changes in the resulting LLM scores and visualizemodel weights at multiple levels of abstraction. To validate our approach wedeployed iScore with three learning engineers over the course of a month. Wepresent a case study where interacting with iScore led a learning engineer toimprove their LLMs score accuracy by three percentage points. Finally weconducted qualitative interviews with the learning engineers that revealed howiScore enabled them to understand evaluate and build trust in their LLMsduring deployment. |


| Item |Content|
| --- |---|
|idx| 2403.04758v1 |
|title| KnowledgeVIS: Interpreting Language Models by Comparing Fill-in-the-Blank Prompts |
|authors| Adam CosciaAlex Endert
|links| http://dx.doi.org/10.1109/TVCG.2023.3346713 |
|updated| 2024-03-07 18:56:31 UTC |
|summary| Recent growth in the popularity of large language models has led to theirincreased usage for summarizing predicting and generating text making itvital to help researchers and engineers understand how and why they work. Wepresent KnowledgeVis a human-in-the-loop visual analytics system forinterpreting language models using fill-in-the-blank sentences as prompts. Bycomparing predictions between sentences KnowledgeVis reveals learnedassociations that intuitively connect what language models learn duringtraining to natural language tasks downstream helping users create and testmultiple prompt variations analyze predicted words using a novel semanticclustering technique and discover insights using interactive visualizations.Collectively these visualizations help users identify the likelihood anduniqueness of individual predictions compare sets of predictions betweenprompts and summarize patterns and relationships between predictions acrossall prompts. We demonstrate the capabilities of KnowledgeVis with feedback fromsix NLP experts as well as three different use cases: 1 probing biomedicalknowledge in two domain-adapted models and 2 evaluating harmful identitystereotypes and 3 discovering facts and relationships between threegeneral-purpose models. |


| Item |Content|
| --- |---|
|idx| 2403.04757v1 |
|title| Preliminary Guidelines For Combining Data Integration and Visual Data Analysis |
|authors| Adam CosciaAshley SuhRemco ChangAlex Endert
|links| http://dx.doi.org/10.1109/TVCG.2023.3334513 |
|updated| 2024-03-07 18:56:16 UTC |
|summary| Data integration is often performed to consolidate information from multipledisparate data sources during visual data analysis. However integrationoperations are usually separate from visual analytics operations such as encodeand filter in both interface design and empirical research. We conducted apreliminary user study to investigate whether and how data integration shouldbe incorporated directly into the visual analytics process. We used twointerface alternatives featuring contrasting approaches to the data preparationand analysis workflow: manual file-based ex-situ integration as a separate stepfrom visual analytics operations and automatic UI-based in-situ integrationmerged with visual analytics operations. Participants were asked to completespecific and free-form tasks with each interface browsing for patternsgenerating insights and summarizing relationships between attributesdistributed across multiple files. Analyzing participants interactions andfeedback we found both task completion time and total interactions to besimilar across interfaces and tasks as well as unique integration strategiesbetween interfaces and emergent behaviors related to satisficing and cognitivebias. Participants time spent and interactions revealed that in-situintegration enabled users to spend more time on analysis tasks compared withex-situ integration. Participants integration strategies and analyticalbehaviors revealed differences in interface usage for generating and trackinghypotheses and insights. With these results we synthesized preliminaryguidelines for designing future visual analytics interfaces that can supportintegrating attributes throughout an active analysis process. |


| Item |Content|
| --- |---|
|idx| 2403.04716v1 |
|title| QRtree -- Decision Tree dialect specification of QRscript |
|authors| Stefano ScanzioMatteo RosaniMattia ScamuzziGianluca Cena
|links| http://arxiv.org/abs/2403.04716v1 |
|updated| 2024-03-07 18:14:02 UTC |
|summary| This specification document specifies the syntax and semantics of QRtreewhich is a specific dialect of QRscript particularly suited to representdecision trees without chance nodes. The term dialect identifies one of thepossible sub-languages that can be encoded inside of an eQR code via QRscript.This specification will describe an intermediate representation of QRtree madethrough a language derived by the three-address code. It will then define thetransformation rules from the intermediate representation to a binary code. Thelatter is a binary representation called eQRtreebytecode. These rules can alsobe applied inversely to transform the eQRtreeBytecode into the intermediaterepresentation. This specification document will pay particular attention tothe creation of a compact eQRtreebytecode as the maximum number of bits thatcan be stored in a QR code is at the time of writing equal to 2953 bytes inthe case of QR code version 40 with a low error correction level. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2403.04627v1 |
|title| Distributed Multi-objective Optimization in Cyber-Physical Energy Systems |
|authors| Sanja StarkEmilie FrostMarvin Nebel-Wenner
|links| http://arxiv.org/abs/2403.04627v1 |
|updated| 2024-03-07 16:12:54 UTC |
|summary| Managing complex Cyber-Physical Energy Systems CPES requires solvingvarious optimization problems with multiple objectives and constraints. Asdistributed control architectures are becoming more popular in CPES for certaintasks due to their flexibility robustness and privacy protectionmulti-objective optimization must also be distributed. For this purpose wepresent MO-COHDA a fully distributed agent-based algorithm for solvingmulti-objective optimization problems of CPES. MO-COHDA allows an easy andflexible adaptation to different use cases and integration of customfunctionality. To evaluate the effectiveness of MO-COHDA we compare it to acentral NSGA-2 algorithm using multi-objective benchmark functions from the ZDTproblem suite. The results show that MO-COHDA can approximate the referencefront of the benchmark problems well and is suitable for solvingmulti-objective optimization problems. In addition an example use case ofscheduling a group of generation units while optimizing three differentobjectives was evaluated to show how MO-COHDA can be easily applied toreal-world optimization problems in CPES. |


| Item |Content|
| --- |---|
|idx| 2403.04442v1 |
|title| Cooperative Bayesian Optimization for Imperfect Agents |
|authors| Ali KhoshvishkaiePetrus MikkolaPierre-Alexandre MurenaSamuel Kaski
|links| http://dx.doi.org/10.1007/978-3-031-43412-9_28 |
|updated| 2024-03-07 12:16:51 UTC |
|summary| We introduce a cooperative Bayesian optimization problem for optimizingblack-box functions of two variables where two agents choose together at whichpoints to query the function but have only control over one variable each. Thissetting is inspired by human-AI teamwork where an AI-assistant helps its humanuser solve a problem in this simplest case collaborative optimization. Weformulate the solution as sequential decision-making where the agent wecontrol models the user as a computationally rational agent with priorknowledge about the function. We show that strategic planning of the queriesenables better identification of the global maximum of the function as long asthe user avoids excessive exploration. This planning is made possible by usingBayes Adaptive Monte Carlo planning and by endowing the agent with a user modelthat accounts for conservative belief updates and exploratory sampling of thepoints to query. |


| Item |Content|
| --- |---|
|idx| 2403.04370v1 |
|title| Cooperative Task Execution in Multi-Agent Systems |
|authors| KarishmaShrisha Rao
|links| http://arxiv.org/abs/2403.04370v1 |
|updated| 2024-03-07 09:58:59 UTC |
|summary| We propose a multi-agent system that enables groups of agents to collaborateand work autonomously to execute tasks. Groups can work in a decentralizedmanner and can adapt to dynamic changes in the environment. Groups of agentssolve assigned tasks by exploring the solution space cooperatively based on thehighest reward first. The tasks have a dependency structure associated withthem. We rigorously evaluated the performance of the system and the individualgroup performance using centralized and decentralized control approaches fortask distribution. Based on the results the centralized approach is moreefficient for systems with a less-dependent system G_18 while thedecentralized approach performs better for systems with a highly-dependentsystem G_40. We also evaluated task allocation to groups that do not haveinterdependence. Our findings reveal that there was significantly lessdifference in the number of tasks allocated to each group in a less-dependentsystem than in a highly-dependent one. The experimental results showed that alarge number of small-size cooperative groups of agents unequivocally improvedthe systems performance compared to a small number of large-size cooperativegroups of agents. Therefore it is essential to identify the optimal group sizefor a system to enhance its performance. |


| Item |Content|
| --- |---|
|idx| 2403.04232v1 |
|title| Generalizing Cooperative Eco-driving via Multi-residual Task Learning |
|authors| Vindula JayawardanaSirui LiCathy WuYashar FaridKentaro Oguchi
|links| http://arxiv.org/abs/2403.04232v1 |
|updated| 2024-03-07 05:25:34 UTC |
|summary| Conventional control such as model-based control is commonly utilized inautonomous driving due to its efficiency and reliability. However real-worldautonomous driving contends with a multitude of diverse traffic scenarios thatare challenging for these planning algorithms. Model-free Deep ReinforcementLearning DRL presents a promising avenue in this direction but learning DRLcontrol policies that generalize to multiple traffic scenarios is still achallenge. To address this we introduce Multi-residual Task Learning MRTL ageneric learning framework based on multi-task learning that for a set of taskscenarios decomposes the control into nominal components that are effectivelysolved by conventional control methods and residual terms which are solvedusing learning. We employ MRTL for fleet-level emission reduction in mixedtraffic using autonomous vehicles as a means of system control. By analyzingthe performance of MRTL across nearly 600 signalized intersections and 1200traffic scenarios we demonstrate that it emerges as a promising approach tosynergize the strengths of DRL and conventional methods in generalizablecontrol. |


| Item |Content|
| --- |---|
|idx| 2403.04202v1 |
|title| Dynamics of Moral Behavior in Heterogeneous Populations of Learning Agents |
|authors| Elizaveta TennantStephen HailesMirco Musolesi
|links| http://arxiv.org/abs/2403.04202v1 |
|updated| 2024-03-07 04:12:24 UTC |
|summary| Growing concerns about safety and alignment of AI systems highlight theimportance of embedding moral capabilities in artificial agents. A promisingsolution is the use of learning from experience i.e. Reinforcement Learning.In multi-agent social environments complex population-level phenomena mayemerge from interactions between individual learning agents. Many of theexisting studies rely on simulated social dilemma environments to study theinteractions of independent learning agents. However they tend to ignore themoral heterogeneity that is likely to be present in societies of agents inpractice. For example at different points in time a single learning agent mayface opponents who are consequentialist i.e. caring about maximizing someoutcome over time or norm-based i.e. focusing on conforming to a specificnorm here and now. The extent to which agents co-development may be impactedby such moral heterogeneity in populations is not well understood. In thispaper we present a study of the learning dynamics of morally heterogeneouspopulations interacting in a social dilemma setting. Using a Prisoners Dilemmaenvironment with a partner selection mechanism we investigate the extent towhich the prevalence of diverse moral agents in populations affects individualagents learning behaviors and emergent population-level outcomes. We observeseveral types of non-trivial interactions between pro-social and anti-socialagents and find that certain classes of moral agents are able to steer selfishagents towards more cooperative behavior. |


