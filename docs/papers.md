# cs.CL 

| Item |Content|
| --- |---|
|idx| 2402.01629v1 |
|title| Position Paper: Generalized grammar rules and structure-based generalization beyond classical equivariance for lexical tasks and transduction |
|authors| Mircea PetracheShubhendu Trivedi
|links| http://arxiv.org/abs/2402.01629v1 |
|updated| 2024-02-02 18:44:37 UTC |
|summary| Compositional generalization is one of the main properties whichdifferentiates lexical learning in humans from state-of-art neural networks. Wepropose a general framework for building models that can generalizecompositionally using the concept of Generalized Grammar Rules GGRs a classof symmetry-based compositional constraints for transduction tasks which weview as a transduction analogue of equivariance constraints in physics-inspiredtasks. Besides formalizing generalized notions of symmetry for languagetransduction our framework is general enough to contain many existing works asspecial cases. We present ideas on how GGRs might be implemented and in theprocess draw connections to reinforcement learning and other areas of research. |


| Item |Content|
| --- |---|
|idx| 2402.01622v2 |
|title| TravelPlanner: A Benchmark for Real-World Planning with Language Agents |
|authors| Jian XieKai ZhangJiangjie ChenTinghui ZhuRenze LouYuandong TianYanghua XiaoYu Su
|links| http://arxiv.org/abs/2402.01622v2 |
|updated| 2024-02-05 06:48:01 UTC |
|summary| Planning has been part of the core pursuit for artificial intelligence sinceits conception but earlier AI agents mostly focused on constrained settingsbecause many of the cognitive substrates necessary for human-level planninghave been lacking. Recently language agents powered by large language modelsLLMs have shown interesting capabilities such as tool use and reasoning. Arethese language agents capable of planning in more complex settings that are outof the reach of prior AI agents To advance this investigation we proposeTravelPlanner a new planning benchmark that focuses on travel planning acommon real-world planning scenario. It provides a rich sandbox environmentvarious tools for accessing nearly four million data records and 1225meticulously curated planning intents and reference plans. Comprehensiveevaluations show that the current language agents are not yet capable ofhandling such complex planning tasks-even GPT-4 only achieves a success rate of0.6. Language agents struggle to stay on task use the right tools to collectinformation or keep track of multiple constraints. However we note that themere possibility for language agents to tackle such a complex problem is initself non-trivial progress. TravelPlanner provides a challenging yetmeaningful testbed for future language agents. |


| Item |Content|
| --- |---|
|idx| 2402.01620v1 |
|title| MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models |
|authors| Justin Chih-Yao ChenSwarnadeep SahaElias Stengel-EskinMohit Bansal
|links| http://arxiv.org/abs/2402.01620v1 |
|updated| 2024-02-02 18:35:14 UTC |
|summary| Multi-agent interactions between Large Language Model LLM agents have shownmajor improvements on diverse reasoning tasks. However these involve longgenerations from multiple models across several rounds making them expensive.Moreover these multi-agent approaches fail to provide a final single modelfor efficient inference. To address this we introduce MAGDi a new method forstructured distillation of the reasoning interactions between multiple LLMsinto smaller LMs. MAGDi teaches smaller models by representing multi-agentinteractions as graphs augmenting a base student model with a graph encoderand distilling knowledge using three objective functions: next-tokenprediction a contrastive loss between correct and incorrect reasoning and agraph-based objective to model the interaction structure. Experiments on sevenwidely-used commonsense and math reasoning benchmarks show that MAGDi improvesthe reasoning capabilities of smaller models outperforming several methodsthat distill from a single teacher and multiple teachers. Moreover MAGDi alsodemonstrates an order of magnitude higher efficiency over its teachers. Weconduct extensive analyses to show that MAGDi 1 enhances the generalizabilityto out-of-domain tasks 2 scales positively with the size and strength of thebase student model and 3 obtains larger improvements via our multi-teachertraining when applying self-consistency - an inference technique that relieson model diversity. |


| Item |Content|
| --- |---|
|idx| 2402.01619v1 |
|title| KB-Plugin: A Plug-and-play Framework for Large Language Models to Induce Programs over Low-resourced Knowledge Bases |
|authors| Jiajie ZhangShulin CaoLinmei HuLing FengLei HouJuanzi Li
|links| http://arxiv.org/abs/2402.01619v1 |
|updated| 2024-02-02 18:32:24 UTC |
|summary| Program induction PI has become a promising paradigm for using knowledgebases KBs to help large language models LLMs answer complexknowledge-intensive questions. Nonetheless PI typically relies on a largenumber of parallel question-program pairs to make the LLM aware of the schemaof the given KB and is thus challenging for many low-resourced KBs that lackannotated data. To this end we propose KB-Plugin a plug-and-play frameworkthat enables LLMs to induce programs over any low-resourced KB. FirstlyKB-Plugin adopts self-supervised learning to encode the detailed schemainformation of a given KB into a pluggable module namely schema plugin.Secondly KB-Plugin utilizes abundant annotated data from a rich-resourced KBto train another pluggable module namely PI plugin which can help the LLMextract question-relevant schema information from the schema plugin of any KBand utilize this information to induce programs over this KB. Experiments onfive heterogeneous KBQA datasets show that KB-Plugin achieves better orcomparable performance with 25times smaller backbone LLM compared to SoTA PImethods for low-resourced KBs and even approaches the performance ofsupervised methods. Our code and data are available athttps://github.com/THU-KEG/KB-Plugin. |


| Item |Content|
| --- |---|
|idx| 2402.01618v1 |
|title| Style Vectors for Steering Generative Large Language Model |
|authors| Kai KonenSophie JentzschDiaoulé DialloPeer SchüttOliver BenschRoxanne El BaffDominik OpitzTobias Hecking
|links| http://arxiv.org/abs/2402.01618v1 |
|updated| 2024-02-02 18:31:15 UTC |
|summary| This research explores strategies for steering the output of large languagemodels LLMs towards specific styles such as sentiment emotion or writingstyle by adding style vectors to the activations of hidden layers during textgeneration. We show that style vectors can be simply computed from recordedlayer activations for input texts in a specific style in contrast to morecomplex training-based approaches. Through a series of experiments wedemonstrate the effectiveness of activation engineering using such stylevectors to influence the style of generated text in a nuanced andparameterisable way distinguishing it from prompt engineering. The presentedresearch constitutes a significant step towards developing more adaptive andeffective AI-empowered interactive systems. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2402.01614v1 |
|title| L2G2G: a Scalable Local-to-Global Network Embedding with Graph Autoencoders |
|authors| Ruikang OuyangAndrew ElliottStratis LimniosMihai CucuringuGesine Reinert
|links| http://arxiv.org/abs/2402.01614v1 |
|updated| 2024-02-02 18:24:37 UTC |
|summary| For analysing real-world networks graph representation learning is a populartool. These methods such as a graph autoencoder GAE typically rely onlow-dimensional representations also called embeddings which are obtainedthrough minimising a loss function these embeddings are used with a decoderfor downstream tasks such as node classification and edge prediction. WhileGAEs tend to be fairly accurate they suffer from scalability issues. Forimproved speed a Local2Global approach which combines graph patch embeddingsbased on eigenvector synchronisation was shown to be fast and achieve goodaccuracy. Here we propose L2G2G a Local2Global method which improves GAEaccuracy without sacrificing scalability. This improvement is achieved bydynamically synchronising the latent node representations while training theGAEs. It also benefits from the decoder computing an only local patch loss.Hence aligning the local embeddings in each epoch utilises more informationfrom the graph than a single post-training alignment does while maintainingscalability. We illustrate on synthetic benchmarks as well as real-worldexamples that L2G2G achieves higher accuracy than the standard Local2Globalapproach and scales efficiently on the larger data sets. We find that for largeand dense networks it even outperforms the slow but assumed more accurateGAEs. |


| Item |Content|
| --- |---|
|idx| 2402.01613v1 |
|title| Nomic Embed: Training a Reproducible Long Context Text Embedder |
|authors| Zach NussbaumJohn X. MorrisBrandon DuderstadtAndriy Mulyar
|links| http://arxiv.org/abs/2402.01613v1 |
|updated| 2024-02-02 18:23:18 UTC |
|summary| This technical report describes the training of nomic-embed-text-v1 thefirst fully reproducible open-source open-weights open-data 8192 contextlength English text embedding model that outperforms both OpenAI Ada-002 andOpenAI text-embedding-3-small on short and long-context tasks. We release thetraining code and model weights under an Apache 2 license. In contrast withother open-source models we release a training data loader with 235 millioncurated text pairs that allows for the full replication of nomic-embed-text-v1.You can find code and data to replicate the model athttps://github.com/nomic-ai/contrastors |


| Item |Content|
| --- |---|
|idx| 2402.01607v1 |
|title| Natural Counterfactuals With Necessary Backtracking |
|authors| Guang-Yuan HaoJiji ZhangBiwei HuangHao WangKun Zhang
|links| http://arxiv.org/abs/2402.01607v1 |
|updated| 2024-02-02 18:11:43 UTC |
|summary| Counterfactual reasoning is pivotal in human cognition and especiallyimportant for providing explanations and making decisions. While Judea Pearlsinfluential approach is theoretically elegant its generation of acounterfactual scenario often requires interventions that are too detached fromthe real scenarios to be feasible. In response we propose a framework ofnatural counterfactuals and a method for generating counterfactuals that arenatural with respect to the actual worlds data distribution. Our methodologyrefines counterfactual reasoning allowing changes in causally precedingvariables to minimize deviations from realistic scenarios. To generate naturalcounterfactuals we introduce an innovative optimization framework that permitsbut controls the extent of backtracking with a naturalness criterion. Empiricalexperiments indicate the effectiveness of our method. |


| Item |Content|
| --- |---|
|idx| 2402.01602v1 |
|title| Foundation Model Sherpas: Guiding Foundation Models through Knowledge and Reasoning |
|authors| Debarun BhattacharjyaJunkyu LeeDon Joven AgravanteBalaji GanesanRadu Marinescu
|links| http://arxiv.org/abs/2402.01602v1 |
|updated| 2024-02-02 18:00:35 UTC |
|summary| Foundation models FMs such as large language models have revolutionized thefield of AI by showing remarkable performance in various tasks. However theyexhibit numerous limitations that prevent their broader adoption in manyreal-world systems which often require a higher bar for trustworthiness andusability. Since FMs are trained using loss functions aimed at reconstructingthe training corpus in a self-supervised manner there is no guarantee that themodels output aligns with users preferences for a specific task at hand. Inthis survey paper we propose a conceptual framework that encapsulatesdifferent modes by which agents could interact with FMs and guide them suitablyfor a set of tasks particularly through knowledge augmentation and reasoning.Our framework elucidates agent role categories such as updating the underlyingFM assisting with prompting the FM and evaluating the FM output. We alsocategorize several state-of-the-art approaches into agent interactionprotocols highlighting the nature and extent of involvement of the variousagent roles. The proposed framework provides guidance for future directions tofurther realize the power of FMs in practical AI systems. |


| Item |Content|
| --- |---|
|idx| 2402.01591v1 |
|title| BAT: Learning to Reason about Spatial Sounds with Large Language Models |
|authors| Zhisheng ZhengPuyuan PengZiyang MaXie ChenEunsol ChoiDavid Harwath
|links| http://arxiv.org/abs/2402.01591v1 |
|updated| 2024-02-02 17:34:53 UTC |
|summary| Spatial sound reasoning is a fundamental human skill enabling us to navigateand interpret our surroundings based on sound. In this paper we present BATwhich combines the spatial sound perception ability of a binaural acousticscene analysis model with the natural language reasoning capabilities of alarge language model LLM to replicate this innate ability. To address thelack of existing datasets of in-the-wild spatial sounds we synthesized abinaural audio dataset using AudioSet and SoundSpaces 2.0. Next we developedSpatialSoundQA a spatial sound-based question-answering dataset offering arange of QA tasks that train BAT in various aspects of spatial sound perceptionand reasoning. The acoustic front end encoder of BAT is a novel spatial audioencoder named Spatial Audio Spectrogram Transformer or Spatial-AST which byitself achieves strong performance across sound event detection spatiallocalization and distance estimation. By integrating Spatial-AST with LLaMA-27B model BAT transcends standard Sound Event Localization and Detection SELDtasks enabling the model to reason about the relationships between the soundsin its environment. Our experiments demonstrate BATs superior performance onboth spatial sound perception and reasoning showcasing the immense potentialof LLMs in navigating and interpreting complex spatial audio environments. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2402.01635v1 |
|title| kNN Algorithm for Conditional Mean and Variance Estimation with Automated Uncertainty Quantification and Variable Selection |
|authors| Marcos MatabuenaJuan C. VidalOscar Hernan Madrid PadillaJukka-Pekka Onnela
|links| http://arxiv.org/abs/2402.01635v1 |
|updated| 2024-02-02 18:54:18 UTC |
|summary| In this paper we introduce a kNN-based regression method that synergizes thescalability and adaptability of traditional non-parametric kNN models with anovel variable selection technique. This method focuses on accuratelyestimating the conditional mean and variance of random response variablesthereby effectively characterizing conditional distributions across diversescenarios.Our approach incorporates a robust uncertainty quantificationmechanism leveraging our prior estimation work on conditional mean andvariance. The employment of kNN ensures scalable computational efficiency inpredicting intervals and statistical accuracy in line with optimalnon-parametric rates. Additionally we introduce a new kNN semi-parametricalgorithm for estimating ROC curves accounting for covariates. For selectingthe smoothing parameter k we propose an algorithm with theoreticalguarantees.Incorporation of variable selection enhances the performance of themethod significantly over conventional kNN techniques in various modelingtasks. We validate the approach through simulations in low moderate andhigh-dimensional covariate spaces. The algorithms effectiveness isparticularly notable in biomedical applications as demonstrated in two casestudies. Concluding with a theoretical analysis we highlight the consistencyand convergence rate of our method over traditional kNN models particularlywhen the underlying regression model takes values in a low-dimensional space. |


| Item |Content|
| --- |---|
|idx| 2402.01632v1 |
|title| Beyond Lengthscales: No-regret Bayesian Optimisation With Unknown Hyperparameters Of Any Type |
|authors| Juliusz ZiomekMasaki AdachiMichael A. Osborne
|links| http://arxiv.org/abs/2402.01632v1 |
|updated| 2024-02-02 18:52:16 UTC |
|summary| Bayesian optimisation requires fitting a Gaussian process model which inturn requires specifying hyperparameters - most of the theoretical literatureassumes those hyperparameters are known. The commonly used maximum likelihoodestimator for hyperparameters of the Gaussian process is consistent only if thedata fills the space uniformly which does not have to be the case in Bayesianoptimisation. Since no guarantees exist regarding the correctness ofhyperparameter estimation and those hyperparameters can significantly affectthe Gaussian process fit theoretical analysis of Bayesian optimisation withunknown hyperparameters is very challenging. Previously proposed algorithmswith the no-regret property were only able to handle the special case ofunknown lengthscales reproducing kernel Hilbert space norm and applied only tothe frequentist case. We propose a novel algorithm HE-GP-UCB which is thefirst algorithm enjoying the no-regret property in the case of unknownhyperparameters of arbitrary form and which supports both Bayesian andfrequentist settings. Our proof idea is novel and can easily be extended toother variants of Bayesian optimisation. We show this by extending ouralgorithm to the adversarially robust optimisation setting under unknownhyperparameters. Finally we empirically evaluate our algorithm on a set of toyproblems and show that it can outperform the maximum likelihood estimator. |


| Item |Content|
| --- |---|
|idx| 2402.01629v1 |
|title| Position Paper: Generalized grammar rules and structure-based generalization beyond classical equivariance for lexical tasks and transduction |
|authors| Mircea PetracheShubhendu Trivedi
|links| http://arxiv.org/abs/2402.01629v1 |
|updated| 2024-02-02 18:44:37 UTC |
|summary| Compositional generalization is one of the main properties whichdifferentiates lexical learning in humans from state-of-art neural networks. Wepropose a general framework for building models that can generalizecompositionally using the concept of Generalized Grammar Rules GGRs a classof symmetry-based compositional constraints for transduction tasks which weview as a transduction analogue of equivariance constraints in physics-inspiredtasks. Besides formalizing generalized notions of symmetry for languagetransduction our framework is general enough to contain many existing works asspecial cases. We present ideas on how GGRs might be implemented and in theprocess draw connections to reinforcement learning and other areas of research. |


| Item |Content|
| --- |---|
|idx| 2402.01621v1 |
|title| Stochastic Two Points Method for Deep Model Zeroth-order Optimization |
|authors| Yijiang PangJiayu Zhou
|links| http://arxiv.org/abs/2402.01621v1 |
|updated| 2024-02-02 18:39:40 UTC |
|summary| Large foundation models such as large language models have performedexceptionally well in various application scenarios. Building or fullyfine-tuning such large models is usually prohibitive due to either hardwarebudget or lack of access to backpropagation. The zeroth-order methods offer apromising direction for tackling this challenge where only forward passes areneeded to update the model. This paper introduces an efficient StochasticTwo-Point S2P approach within the gradient-free regime. We present thetheoretical convergence properties of S2P under the general and relaxedsmoothness assumptions. The theoretical properties also shed light on a fasterand more stable S2P variant Accelerated S2P AS2P through exploiting our newconvergence properties that better represent the dynamics of deep models intraining. Our comprehensive empirical results show that AS2P is highlyeffective in optimizing objectives for large deep models including languagemodels and outperforms standard methods across various model types and scaleswith 2 times speed-up in training over most conducted tasks. |


| Item |Content|
| --- |---|
|idx| 2402.01617v1 |
|title| A GP-based Robust Motion Planning Framework for Agile Autonomous Robot Navigation and Recovery in Unknown Environments |
|authors| Nicholas MohammadJacob HigginsNicola Bezzo
|links| http://arxiv.org/abs/2402.01617v1 |
|updated| 2024-02-02 18:27:21 UTC |
|summary| For autonomous mobile robots uncertainties in the environment and systemmodel can lead to failure in the motion planning pipeline resulting inpotential collisions. In order to achieve a high level of robust autonomythese robots should be able to proactively predict and recover from suchfailures. To this end we propose a Gaussian Process GP based model forproactively detecting the risk of future motion planning failure. When thisrisk exceeds a certain threshold a recovery behavior is triggered thatleverages the same GP model to find a safe state from which the robot maycontinue towards the goal. The proposed approach is trained in simulation onlyand can generalize to real world environments on different robotic platforms.Simulations and physical experiments demonstrate that our framework is capableof both predicting planner failures and recovering the robot to states whereplanner success is likely all while producing agile motion. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2402.01596v1 |
|title| Immersive Video Compression using Implicit Neural Representations |
|authors| Ho Man KwanFan ZhangAndrew GowerDavid Bull
|links| http://arxiv.org/abs/2402.01596v1 |
|updated| 2024-02-02 17:49:31 UTC |
|summary| Recent work on implicit neural representations INRs has evidenced theirpotential for efficiently representing and encoding conventional video content.In this paper we for the first time extend their application to immersivemulti-view videos by proposing MV-HiNeRV a new INR-based immersive videocodec. MV-HiNeRV is an enhanced version of a state-of-the-art INR-based videocodec HiNeRV which was developed for single-view video compression. We havemodified the model to learn a different group of feature grids for each viewand share the learnt network parameters among all views. This enables the modelto effectively exploit the spatio-temporal and the inter-view redundancy thatexists within multi-view videos. The proposed codec was used to compressmulti-view texture and depth video sequences in the MPEG Immersive Video MIVCommon Test Conditions and tested against the MIV Test model TMIV that usesthe VVenC video codec. The results demonstrate the superior performance ofMV-HiNeRV with significant coding gains up to 72.33 over TMIV. Theimplementation of MV-HiNeRV will be published for further development andevaluation. |


| Item |Content|
| --- |---|
|idx| 2402.01590v1 |
|title| NeuroCine: Decoding Vivid Video Sequences from Human Brain Activties |
|authors| Jingyuan SunMingxiao LiZijiao ChenMarie-Francine Moens
|links| http://arxiv.org/abs/2402.01590v1 |
|updated| 2024-02-02 17:34:25 UTC |
|summary| In the pursuit to understand the intricacies of human brains visualprocessing reconstructing dynamic visual experiences from brain activitiesemerges as a challenging yet fascinating endeavor. While recent advancementshave achieved success in reconstructing static images from non-invasive brainrecordings the domain of translating continuous brain activities into videoformat remains underexplored. In this work we introduce NeuroCine a noveldual-phase framework to targeting the inherent challenges of decoding fMRIdata such as noises spatial redundancy and temporal lags. This frameworkproposes spatial masking and temporal interpolation-based augmentation forcontrastive learning fMRI representations and a diffusion model enhanced bydependent prior noise for video generation. Tested on a publicly available fMRIdataset our method shows promising results outperforming the previousstate-of-the-art models by a notable margin of 20.97 31.00 and12.30 respectively on decoding the brain activities of three subjects inthe fMRI dataset as measured by SSIM. Additionally our attention analysissuggests that the model aligns with existing brain structures and functionsindicating its biological plausibility and interpretability. |


| Item |Content|
| --- |---|
|idx| 2402.01566v1 |
|title| Boximator: Generating Rich and Controllable Motions for Video Synthesis |
|authors| Jiawei WangYuchen ZhangJiaxin ZouYan ZengGuoqiang WeiLiping YuanHang Li
|links| http://arxiv.org/abs/2402.01566v1 |
|updated| 2024-02-02 16:59:48 UTC |
|summary| Generating rich and controllable motion is a pivotal challenge in videosynthesis. We propose Boximator a new approach for fine-grained motioncontrol. Boximator introduces two constraint types: hard box and soft box.Users select objects in the conditional frame using hard boxes and then useeither type of boxes to roughly or rigorously define the objects positionshape or motion path in future frames. Boximator functions as a plug-in forexisting video diffusion models. Its training process preserves the basemodels knowledge by freezing the original weights and training only thecontrol module. To address training challenges we introduce a novelself-tracking technique that greatly simplifies the learning of box-objectcorrelations. Empirically Boximator achieves state-of-the-art video qualityFVD scores improving on two base models and further enhanced afterincorporating box constraints. Its robust motion controllability is validatedby drastic increases in the bounding box alignment metric. Human evaluationalso shows that users favor Boximator generation results over the base model. |


| Item |Content|
| --- |---|
|idx| 2402.01557v1 |
|title| Deep Continuous Networks |
|authors| Nergis TomenSilvia L. PinteaJan C. van Gemert
|links| http://arxiv.org/abs/2402.01557v1 |
|updated| 2024-02-02 16:50:18 UTC |
|summary| CNNs and computational models of biological vision share some fundamentalprinciples which opened new avenues of research. However fruitful cross-fieldresearch is hampered by conventional CNN architectures being based on spatiallyand depthwise discrete representations which cannot accommodate certainaspects of biological complexity such as continuously varying receptive fieldsizes and dynamics of neuronal responses. Here we propose deep continuousnetworks DCNs which combine spatially continuous filters with thecontinuous depth framework of neural ODEs. This allows us to learn the spatialsupport of the filters during training as well as model the continuousevolution of feature maps linking DCNs closely to biological models. We showthat DCNs are versatile and highly applicable to standard image classificationand reconstruction problems where they improve parameter and data efficiencyand allow for meta-parametrization. We illustrate the biological plausibilityof the scale distributions learned by DCNs and explore their performance in aneuroscientifically inspired pattern completion task. Finally we investigatean efficient implementation of DCNs by changing input contrast. |


| Item |Content|
| --- |---|
|idx| 2402.01555v1 |
|title| SLYKLatent, a Learning Framework for Facial Features Estimation |
|authors| Samuel AdebayoJoost C. DessingSeán McLoone
|links| http://arxiv.org/abs/2402.01555v1 |
|updated| 2024-02-02 16:47:18 UTC |
|summary| In this research we present SLYKLatent a novel approach for enhancing gazeestimation by addressing appearance instability challenges in datasets due toaleatoric uncertainties covariant shifts and test domain generalization.SLYKLatent utilizes Self-Supervised Learning for initial training with facialexpression datasets followed by refinement with a patch-based tri-branchnetwork and an inverse explained variance-weighted training loss function. Ourevaluation on benchmark datasets achieves an 8.7 improvement on Gaze360rivals top MPIIFaceGaze results and leads on a subset of ETH-XGaze by 13surpassing existing methods by significant margins. Adaptability tests onRAF-DB and Affectnet show 86.4 and 60.9 accuracies respectively. Ablationstudies confirm the effectiveness of SLYKLatents novel components. Thisapproach has strong potential in human-robot interaction. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2402.01635v1 |
|title| kNN Algorithm for Conditional Mean and Variance Estimation with Automated Uncertainty Quantification and Variable Selection |
|authors| Marcos MatabuenaJuan C. VidalOscar Hernan Madrid PadillaJukka-Pekka Onnela
|links| http://arxiv.org/abs/2402.01635v1 |
|updated| 2024-02-02 18:54:18 UTC |
|summary| In this paper we introduce a kNN-based regression method that synergizes thescalability and adaptability of traditional non-parametric kNN models with anovel variable selection technique. This method focuses on accuratelyestimating the conditional mean and variance of random response variablesthereby effectively characterizing conditional distributions across diversescenarios.Our approach incorporates a robust uncertainty quantificationmechanism leveraging our prior estimation work on conditional mean andvariance. The employment of kNN ensures scalable computational efficiency inpredicting intervals and statistical accuracy in line with optimalnon-parametric rates. Additionally we introduce a new kNN semi-parametricalgorithm for estimating ROC curves accounting for covariates. For selectingthe smoothing parameter k we propose an algorithm with theoreticalguarantees.Incorporation of variable selection enhances the performance of themethod significantly over conventional kNN techniques in various modelingtasks. We validate the approach through simulations in low moderate andhigh-dimensional covariate spaces. The algorithms effectiveness isparticularly notable in biomedical applications as demonstrated in two casestudies. Concluding with a theoretical analysis we highlight the consistencyand convergence rate of our method over traditional kNN models particularlywhen the underlying regression model takes values in a low-dimensional space. |


| Item |Content|
| --- |---|
|idx| 2402.01632v1 |
|title| Beyond Lengthscales: No-regret Bayesian Optimisation With Unknown Hyperparameters Of Any Type |
|authors| Juliusz ZiomekMasaki AdachiMichael A. Osborne
|links| http://arxiv.org/abs/2402.01632v1 |
|updated| 2024-02-02 18:52:16 UTC |
|summary| Bayesian optimisation requires fitting a Gaussian process model which inturn requires specifying hyperparameters - most of the theoretical literatureassumes those hyperparameters are known. The commonly used maximum likelihoodestimator for hyperparameters of the Gaussian process is consistent only if thedata fills the space uniformly which does not have to be the case in Bayesianoptimisation. Since no guarantees exist regarding the correctness ofhyperparameter estimation and those hyperparameters can significantly affectthe Gaussian process fit theoretical analysis of Bayesian optimisation withunknown hyperparameters is very challenging. Previously proposed algorithmswith the no-regret property were only able to handle the special case ofunknown lengthscales reproducing kernel Hilbert space norm and applied only tothe frequentist case. We propose a novel algorithm HE-GP-UCB which is thefirst algorithm enjoying the no-regret property in the case of unknownhyperparameters of arbitrary form and which supports both Bayesian andfrequentist settings. Our proof idea is novel and can easily be extended toother variants of Bayesian optimisation. We show this by extending ouralgorithm to the adversarially robust optimisation setting under unknownhyperparameters. Finally we empirically evaluate our algorithm on a set of toyproblems and show that it can outperform the maximum likelihood estimator. |


| Item |Content|
| --- |---|
|idx| 2402.01629v1 |
|title| Position Paper: Generalized grammar rules and structure-based generalization beyond classical equivariance for lexical tasks and transduction |
|authors| Mircea PetracheShubhendu Trivedi
|links| http://arxiv.org/abs/2402.01629v1 |
|updated| 2024-02-02 18:44:37 UTC |
|summary| Compositional generalization is one of the main properties whichdifferentiates lexical learning in humans from state-of-art neural networks. Wepropose a general framework for building models that can generalizecompositionally using the concept of Generalized Grammar Rules GGRs a classof symmetry-based compositional constraints for transduction tasks which weview as a transduction analogue of equivariance constraints in physics-inspiredtasks. Besides formalizing generalized notions of symmetry for languagetransduction our framework is general enough to contain many existing works asspecial cases. We present ideas on how GGRs might be implemented and in theprocess draw connections to reinforcement learning and other areas of research. |


| Item |Content|
| --- |---|
|idx| 2402.01614v1 |
|title| L2G2G: a Scalable Local-to-Global Network Embedding with Graph Autoencoders |
|authors| Ruikang OuyangAndrew ElliottStratis LimniosMihai CucuringuGesine Reinert
|links| http://arxiv.org/abs/2402.01614v1 |
|updated| 2024-02-02 18:24:37 UTC |
|summary| For analysing real-world networks graph representation learning is a populartool. These methods such as a graph autoencoder GAE typically rely onlow-dimensional representations also called embeddings which are obtainedthrough minimising a loss function these embeddings are used with a decoderfor downstream tasks such as node classification and edge prediction. WhileGAEs tend to be fairly accurate they suffer from scalability issues. Forimproved speed a Local2Global approach which combines graph patch embeddingsbased on eigenvector synchronisation was shown to be fast and achieve goodaccuracy. Here we propose L2G2G a Local2Global method which improves GAEaccuracy without sacrificing scalability. This improvement is achieved bydynamically synchronising the latent node representations while training theGAEs. It also benefits from the decoder computing an only local patch loss.Hence aligning the local embeddings in each epoch utilises more informationfrom the graph than a single post-training alignment does while maintainingscalability. We illustrate on synthetic benchmarks as well as real-worldexamples that L2G2G achieves higher accuracy than the standard Local2Globalapproach and scales efficiently on the larger data sets. We find that for largeand dense networks it even outperforms the slow but assumed more accurateGAEs. |


| Item |Content|
| --- |---|
|idx| 2402.01599v1 |
|title| Hyperparameter tuning via trajectory predictions: Stochastic prox-linear methods in matrix sensing |
|authors| Mengqi LouKabir Aladin VerchandAshwin Pananjady
|links| http://arxiv.org/abs/2402.01599v1 |
|updated| 2024-02-02 17:55:12 UTC |
|summary| Motivated by the desire to understand stochastic algorithms for nonconvexoptimization that are robust to their hyperparameter choices we analyze amini-batched prox-linear iterative algorithm for the problem of recovering anunknown rank-1 matrix from rank-1 Gaussian measurements corrupted by noise. Wederive a deterministic recursion that predicts the error of this method andshow using a non-asymptotic framework that this prediction is accurate forany batch-size and a large range of step-sizes. In particular our analysisreveals that this method though stochastic converges linearly from a localinitialization with a fixed step-size to a statistical error floor. Ouranalysis also exposes how the batch-size step-size and noise level affect thelinear convergence rate and the eventual statistical estimation error and wedemonstrate how to use our deterministic predictions to perform hyperparametertuning e.g. step-size and batch-size selection without ever running themethod. On a technical level our analysis is enabled in part by showing thatthe fluctuations of the empirical iterates around our deterministic predictionsscale with the error of the previous iterate. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2402.01555v1 |
|title| SLYKLatent, a Learning Framework for Facial Features Estimation |
|authors| Samuel AdebayoJoost C. DessingSeán McLoone
|links| http://arxiv.org/abs/2402.01555v1 |
|updated| 2024-02-02 16:47:18 UTC |
|summary| In this research we present SLYKLatent a novel approach for enhancing gazeestimation by addressing appearance instability challenges in datasets due toaleatoric uncertainties covariant shifts and test domain generalization.SLYKLatent utilizes Self-Supervised Learning for initial training with facialexpression datasets followed by refinement with a patch-based tri-branchnetwork and an inverse explained variance-weighted training loss function. Ourevaluation on benchmark datasets achieves an 8.7 improvement on Gaze360rivals top MPIIFaceGaze results and leads on a subset of ETH-XGaze by 13surpassing existing methods by significant margins. Adaptability tests onRAF-DB and Affectnet show 86.4 and 60.9 accuracies respectively. Ablationstudies confirm the effectiveness of SLYKLatents novel components. Thisapproach has strong potential in human-robot interaction. |


| Item |Content|
| --- |---|
|idx| 2402.01536v1 |
|title| Homogenization Effects of Large Language Models on Human Creative Ideation |
|authors| Barrett R. AndersonJash Hemant ShahMax Kreminski
|links| http://arxiv.org/abs/2402.01536v1 |
|updated| 2024-02-02 16:27:11 UTC |
|summary| Large language models LLMs are now being used in a wide variety ofcontexts including as creativity support tools CSTs intended to help theirusers come up with new ideas. But do LLMs actually support user creativity Wehypothesized that the use of an LLM as a CST might make the LLMs users feelmore creative and even broaden the range of ideas suggested by each individualuser but also homogenize the ideas suggested by different users. We conducteda 36-participant comparative user study and found in accordance with thehomogenization hypothesis that different users tended to produce lesssemantically distinct ideas with ChatGPT than with an alternative CST.Additionally ChatGPT users generated a greater number of more detailed ideasbut felt less responsible for the ideas they generated. We discuss potentialimplications of these findings for users designers and developers ofLLM-based CSTs. |


| Item |Content|
| --- |---|
|idx| 2402.01292v1 |
|title| Towards the new XAI: A Hypothesis-Driven Approach to Decision Support Using Evidence |
|authors| Thao LeTim MillerRonal SinghLiz Sonenberg
|links| http://arxiv.org/abs/2402.01292v1 |
|updated| 2024-02-02 10:28:24 UTC |
|summary| Prior research on AI-assisted human decision-making has explored severaldifferent explainable AI XAI approaches. A recent paper has proposed aparadigm shift calling for hypothesis-driven XAI through a conceptual frameworkcalled evaluative AI that gives people evidence that supports or refuteshypotheses without necessarily giving a decision-aid recommendation. In thispaper we describe and evaluate an approach for hypothesis-driven XAI based onthe Weight of Evidence WoE framework which generates both positive andnegative evidence for a given hypothesis. Through human behaviouralexperiments we show that our hypothesis-driven approach increases decisionaccuracy reduces reliance compared to a recommendation-driven approach and anAI-explanation-only baseline but with a small increase in under-reliancecompared to the recommendation-driven approach. Further we show thatparticipants used our hypothesis-driven approach in a materially different wayto the two baselines. |


| Item |Content|
| --- |---|
|idx| 2402.01227v1 |
|title| STAA-Net: A Sparse and Transferable Adversarial Attack for Speech Emotion Recognition |
|authors| Yi ChangZhao RenZixing ZhangXin JingKun QianXi ShaoBin HuTanja SchultzBjörn W. Schuller
|links| http://arxiv.org/abs/2402.01227v1 |
|updated| 2024-02-02 08:46:57 UTC |
|summary| Speech contains rich information on the emotions of humans and SpeechEmotion Recognition SER has been an important topic in the area ofhuman-computer interaction. The robustness of SER models is crucialparticularly in privacy-sensitive and reliability-demanding domains likeprivate healthcare. Recently the vulnerability of deep neural networks in theaudio domain to adversarial attacks has become a popular area of research.However prior works on adversarial attacks in the audio domain primarily relyon iterative gradient-based techniques which are time-consuming and prone tooverfitting the specific threat model. Furthermore the exploration of sparseperturbations which have the potential for better stealthiness remainslimited in the audio domain. To address these challenges we propose agenerator-based attack method to generate sparse and transferable adversarialexamples to deceive SER models in an end-to-end and efficient manner. Weevaluate our method on two widely-used SER datasets Database of Elicited Moodin Speech DEMoS and Interactive Emotional dyadic MOtion CAPture IEMOCAPand demonstrate its ability to generate successful sparse adversarial examplesin an efficient manner. Moreover our generated adversarial examples exhibitmodel-agnostic transferability enabling effective adversarial attacks onadvanced victim models. |


| Item |Content|
| --- |---|
|idx| 2402.01117v1 |
|title| DTS-SQL: Decomposed Text-to-SQL with Small Large Language Models |
|authors| Mohammadreza PourrezaDavood Rafiei
|links| http://arxiv.org/abs/2402.01117v1 |
|updated| 2024-02-02 03:21:00 UTC |
|summary| Leading models for the text-to-SQL task heavily rely on proprietary LargeLanguage Models LLMs posing concerns over data privacy. Closing theperformance gap between small open-source models and large proprietary modelsis crucial to mitigate this reliance. To this end we introduce a noveltwo-stage fine-tuning approach that decomposes the task into two simpler tasks.Through comprehensive evaluation on two large cross-domain datasets and twosmall LLMs we show that this approach improves execution accuracy by 3 to 7percent effectively aligning the performance of open-source models with theirproprietary counterparts. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2402.01586v1 |
|title| TrustAgent: Towards Safe and Trustworthy LLM-based Agents through Agent Constitution |
|authors| Wenyue HuaXianjun YangZelong LiCheng WeiYongfeng Zhang
|links| http://arxiv.org/abs/2402.01586v1 |
|updated| 2024-02-02 17:26:23 UTC |
|summary| The emergence of LLM-based agents has garnered considerable attention yettheir trustworthiness remains an under-explored area. As agents can directlyinteract with the physical environment their reliability and safety iscritical. This paper presents an Agent-Constitution-based agent frameworkTrustAgent an initial investigation into improving the safety dimension oftrustworthiness in LLM-based agents. This framework consists of threefoldstrategies: pre-planning strategy which injects safety knowledge to the modelprior to plan generation in-planning strategy which bolsters safety duringplan generation and post-planning strategy which ensures safety bypost-planning inspection. Through experimental analysis we demonstrate howthese approaches can effectively elevate an LLM agents safety by identifyingand preventing potential dangers. Furthermore we explore the intricaterelationships between safety and helpfulness and between the models reasoningability and its efficacy as a safe agent. This paper underscores the imperativeof integrating safety awareness and trustworthiness into the design anddeployment of LLM-based agents not only to enhance their performance but alsoto ensure their responsible integration into human-centric environments. Dataand code are available at https://github.com/agiresearch/TrustAgent. |


| Item |Content|
| --- |---|
|idx| 2402.01546v1 |
|title| Privacy-Preserving Distributed Learning for Residential Short-Term Load Forecasting |
|authors| Yi DongYingjie WangMariana GamaMustafa A. MustafaGeert DeconinckXiaowei Huang
|links| http://arxiv.org/abs/2402.01546v1 |
|updated| 2024-02-02 16:39:08 UTC |
|summary| In the realm of power systems the increasing involvement of residentialusers in load forecasting applications has heightened concerns about dataprivacy. Specifically the load data can inadvertently reveal the dailyroutines of residential users thereby posing a risk to their propertysecurity. While federated learning FL has been employed to safeguard userprivacy by enabling model training without the exchange of raw data these FLmodels have shown vulnerabilities to emerging attack techniques such as DeepLeakage from Gradients and poisoning attacks. To counteract these we initiallyemploy a Secure-Aggregation SecAgg algorithm that leverages multipartycomputation cryptographic techniques to mitigate the risk of gradient leakage.However the introduction of SecAgg necessitates the deployment of additionalsub-center servers for executing the multiparty computation protocol therebyescalating computational complexity and reducing system robustness especiallyin scenarios where one or more sub-centers are unavailable. To address thesechallenges we introduce a Markovian Switching-based distributed trainingframework the convergence of which is substantiated through rigoroustheoretical analysis. The Distributed Markovian Switching DMS topology showsstrong robustness towards the poisoning attacks as well. Case studies employingreal-world power system load data validate the efficacy of our proposedalgorithm. It not only significantly minimizes communication complexity butalso maintains accuracy levels comparable to traditional FL methods therebyenhancing the scalability of our load forecasting algorithm. |


| Item |Content|
| --- |---|
|idx| 2402.01446v1 |
|title| Guidance Graph Optimization for Lifelong Multi-Agent Path Finding |
|authors| Yulun ZhangHe JiangVarun BhattStefanos NikolaidisJiaoyang Li
|links| http://arxiv.org/abs/2402.01446v1 |
|updated| 2024-02-02 14:38:04 UTC |
|summary| We study how to use guidance to improve the throughput of lifelongMulti-Agent Path Finding MAPF. Previous studies have demonstrated that whileincorporating guidance such as highways can accelerate MAPF algorithms thisoften results in a trade-off with solution quality. In addition how togenerate good guidance automatically remains largely unexplored with currentmethods falling short of surpassing manually designed ones. In this work weintroduce the directed guidance graph as a versatile representation of guidancefor lifelong MAPF framing Guidance Graph Optimization GGO as the task ofoptimizing its edge weights. We present two GGO algorithms to automaticallygenerate guidance for arbitrary lifelong MAPF algorithms and maps. The firstmethod directly solves GGO by employing CMA-ES a black-box optimizationalgorithm. The second method PIU optimizes an update model capable ofgenerating guidance demonstrating the ability to transfer optimized guidancegraphs to larger maps with similar layouts. Empirically we show that 1 ourguidance graphs improve the throughput of three representative lifelong MAPFalgorithms in four benchmark maps and 2 our update model can generateguidance graphs for as large as 93 times 91 maps and as many as 3000 agents. |


| Item |Content|
| --- |---|
|idx| 2402.01302v1 |
|title| A Unified Framework for Gradient-based Clustering of Distributed Data |
|authors| Aleksandar ArmackiDragana BajovićDušan JakovetićSoummya Kar
|links| http://arxiv.org/abs/2402.01302v1 |
|updated| 2024-02-02 10:44:42 UTC |
|summary| We develop a family of distributed clustering algorithms that work overnetworks of users. In the proposed scenario users contain a local dataset andcommunicate only with their immediate neighbours with the aim of finding aclustering of the full joint data. The proposed family termed DistributedGradient Clustering DGC-mathcalF_rho is parametrized by rho geq 1controling the proximity of users center estimates with mathcalFdetermining the clustering loss. Specialized to popular clustering losses likeK-means and Huber loss DGC-mathcalF_rho gives rise to noveldistributed clustering algorithms DGC-KM_rho and DGC-HL_rho while anovel clustering loss based on the logistic function leads to DGC-LL_rho. Weprovide a unified analysis and establish several strong results under mildassumptions. First the sequence of centers generated by the methods convergesto a well-defined notion of fixed point under any center initialization andvalue of rho. Second as rho increases the family of fixed pointsproduced by DGC-mathcalF_rho converges to a notion of consensus fixedpoints. We show that consensus fixed points of DGC-mathcalF_rho areequivalent to fixed points of gradient clustering over the full dataguaranteeing a clustering of the full data is produced. For the special case ofBregman losses we show that our fixed points converge to the set of Lloydpoints. Numerical experiments on real data confirm our theoretical findings anddemonstrate strong performance of the methods. |


| Item |Content|
| --- |---|
|idx| 2402.01294v1 |
|title| Minimizing Regret in Billboard Advertisement under Zonal Influence Constraint |
|authors| Dildar AliSuman BanerjeeYamuna Prasad
|links| http://arxiv.org/abs/2402.01294v1 |
|updated| 2024-02-02 10:31:28 UTC |
|summary| In a typical billboard advertisement technique a number of digitalbillboards are owned by an influence provider and many advertisers approachthe influence provider for a specific number of views of their advertisementcontent on a payment basis. If the influence provider provides the demanded ormore influence then he will receive the full payment or else a partialpayment. In the context of an influence provider if he provides more or lessthan an advertisers demanded influence it is a loss for him. This isformalized as Regret and naturally in the context of the influenceprovider the goal will be to allocate the billboard slots among theadvertisers such that the total regret is minimized. In this paper we studythis problem as a discrete optimization problem and propose four solutionapproaches. The first one selects the billboard slots from the available onesin an incremental greedy manner and we call this method the Budget EffectiveGreedy approach. In the second one we introduce randomness with the first onewhere we perform the marginal gain computation for a sample of randomly chosenbillboard slots. The remaining two approaches are further improvements over thesecond one. We analyze all the algorithms to understand their time and spacecomplexity. We implement them with real-life trajectory and billboard datasetsand conduct a number of experiments. It has been observed that the randomizedbudget effective greedy approach takes reasonable computational time whileminimizing the regret. |


