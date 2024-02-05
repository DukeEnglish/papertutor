# cs.CL 

| Item |Content|
| --- |---|
|idx| 2402.00861v1 |
|title| Evaluating Large Language Models for Generalization and Robustness via Data Compression |
|authors| Yucheng LiYunhao GuoFrank GuerinChenghua Lin
|links| http://arxiv.org/abs/2402.00861v1 |
|updated| 2024-02-01 18:56:18 UTC |
|summary| Existing methods for evaluating large language models face challenges such asdata contamination sensitivity to prompts and the high cost of benchmarkcreation. To address this we propose a lossless data compression basedevaluation approach that tests how models predictive abilities generalizeafter their training cutoff. Specifically we collect comprehensive test dataspanning 83 months from 2017 to 2023 and split the data into training andtesting periods according to models training data cutoff. We measure: 1 thecompression performance on the testing period as a measure of generalization onunseen data and 2 the performance gap between the training and testing periodas a measure of robustness. Our experiments test 14 representative largelanguage models with various sizes on sources including Wikipedia newsarticles code arXiv papers and multi-modal data. We find that thecompression rate of many models reduces significantly after their cutoff datebut models such as Mistral and Llama-2 demonstrate a good balance betweenperformance and robustness. Results also suggest that models struggle togeneralize on news and code data but work especially well on arXiv papers. Wealso find the context size and tokenization implementation have a big impact ofon the overall compression performance. |


| Item |Content|
| --- |---|
|idx| 2402.00858v1 |
|title| Can Large Language Models Understand Context? |
|authors| Yilun ZhuJoel Ruben Antony MonizShruti BhargavaJiarui LuDhivya PiraviperumalSite LiYuan ZhangHong YuBo-Hsiang Tseng
|links| http://arxiv.org/abs/2402.00858v1 |
|updated| 2024-02-01 18:55:29 UTC |
|summary| Understanding context is key to understanding human language an abilitywhich Large Language Models LLMs have been increasingly seen to demonstrateto an impressive extent. However though the evaluation of LLMs encompassesvarious domains within the realm of Natural Language Processing limitedattention has been paid to probing their linguistic capability of understandingcontextual features. This paper introduces a context understanding benchmark byadapting existing datasets to suit the evaluation of generative models. Thisbenchmark comprises of four distinct tasks and nine datasets all featuringprompts designed to assess the models ability to understand context. First weevaluate the performance of LLMs under the in-context learning pretrainingscenario. Experimental results indicate that pre-trained dense models strugglewith understanding more nuanced contextual features when compared tostate-of-the-art fine-tuned models. Second as LLM compression holds growingsignificance in both research and real-world applications we assess thecontext understanding of quantized models under in-context-learning settings.We find that 3-bit post-training quantization leads to varying degrees ofperformance reduction on our benchmark. We conduct an extensive analysis ofthese scenarios to substantiate our experimental results. |


| Item |Content|
| --- |---|
|idx| 2402.00856v2 |
|title| Towards Efficient and Exact Optimization of Language Model Alignment |
|authors| Haozhe JiCheng LuYilin NiuPei KeHongning WangJun ZhuJie TangMinlie Huang
|links| http://arxiv.org/abs/2402.00856v2 |
|updated| 2024-02-02 15:50:10 UTC |
|summary| The alignment of language models with human preferences is vital for theirapplication in real-world tasks. The problem is formulated as optimizing themodels policy to maximize the expected reward that reflects human preferenceswith minimal deviation from the initial policy. While considered as astraightforward solution reinforcement learning RL suffers from highvariance in policy updates which impedes efficient policy improvement.Recently direct preference optimization DPO was proposed to directlyoptimize the policy from preference data. Though simple to implement DPO isderived based on the optimal policy that is not assured to be achieved inpractice which undermines its convergence to the intended solution.  In this paper we propose efficient exact optimization EXO of the alignmentobjective. We prove that EXO is guaranteed to optimize in the same direction asthe RL algorithms asymptotically for arbitary parametrization of the policywhile enables efficient optimization by circumventing the complexitiesassociated with RL algorithms. We compare our method to DPO with boththeoretical and empirical analyses and further demonstrate the advantages ofour method over existing approaches on realistic human preference data. |


| Item |Content|
| --- |---|
|idx| 2402.00841v1 |
|title| Tiny Titans: Can Smaller Large Language Models Punch Above Their Weight in the Real World for Meeting Summarization? |
|authors| Xue-Yong FuMd Tahmid Rahman LaskarElena KhasanovaCheng ChenShashi Bhushan TN
|links| http://arxiv.org/abs/2402.00841v1 |
|updated| 2024-02-01 18:31:34 UTC |
|summary| Large Language Models LLMs have demonstrated impressive capabilities tosolve a wide range of tasks without being explicitly fine-tuned ontask-specific datasets. However deploying LLMs in the real world is nottrivial as it requires substantial computing resources. In this paper weinvestigate whether smaller compact LLMs are a good alternative to thecomparatively Larger LLMs2 to address significant costs associated withutilizing LLMs in the real world. In this regard we study the meetingsummarization task in a real-world industrial environment and conduct extensiveexperiments by comparing the performance of fine-tuned compact LLMs e.g.FLAN-T5 TinyLLaMA LiteLLaMA with zero-shot larger LLMs e.g. LLaMA-2GPT-3.5 PaLM-2. We observe that most smaller LLMs even after fine-tuningfail to outperform larger zero-shot LLMs in meeting summarization datasets.However a notable exception is FLAN-T5 780M parameters which performs onpar or even better than many zero-shot Larger LLMs from 7B to above 70Bparameters while being significantly smaller. This makes compact LLMs likeFLAN-T5 a suitable cost-efficient solution for real-world industrialdeployment. |


| Item |Content|
| --- |---|
|idx| 2402.00838v1 |
|title| OLMo: Accelerating the Science of Language Models |
|authors| Dirk GroeneveldIz BeltagyPete WalshAkshita BhagiaRodney KinneyOyvind TafjordAnanya Harsh JhaHamish IvisonIan MagnussonYizhong WangShane AroraDavid AtkinsonRussell AuthurKhyathi Raghavi ChanduArman CohanJennifer DumasYanai ElazarYuling GuJack HesselTushar KhotWilliam MerrillJacob MorrisonNiklas MuennighoffAakanksha NaikCrystal NamMatthew E. PetersValentina PyatkinAbhilasha RavichanderDustin SchwenkSaurabh ShahWill SmithEmma StrubellNishant SubramaniMitchell WortsmanPradeep DasigiNathan LambertKyle RichardsonLuke ZettlemoyerJesse DodgeKyle LoLuca SoldainiNoah A. SmithHannaneh Hajishirzi
|links| http://arxiv.org/abs/2402.00838v1 |
|updated| 2024-02-01 18:28:55 UTC |
|summary| Language models LMs have become ubiquitous in both NLP research and incommercial product offerings. As their commercial importance has surged themost powerful models have become closed off gated behind proprietaryinterfaces with important details of their training data architectures anddevelopment undisclosed. Given the importance of these details inscientifically studying these models including their biases and potentialrisks we believe it is essential for the research community to have access topowerful truly open LMs. To this end this technical report details the firstrelease of OLMo a state-of-the-art truly Open Language Model and itsframework to build and study the science of language modeling. Unlike mostprior efforts that have only released model weights and inference code werelease OLMo and the whole framework including training data and training andevaluation code. We hope this release will empower and strengthen the openresearch community and inspire a new wave of innovation. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2402.00861v1 |
|title| Evaluating Large Language Models for Generalization and Robustness via Data Compression |
|authors| Yucheng LiYunhao GuoFrank GuerinChenghua Lin
|links| http://arxiv.org/abs/2402.00861v1 |
|updated| 2024-02-01 18:56:18 UTC |
|summary| Existing methods for evaluating large language models face challenges such asdata contamination sensitivity to prompts and the high cost of benchmarkcreation. To address this we propose a lossless data compression basedevaluation approach that tests how models predictive abilities generalizeafter their training cutoff. Specifically we collect comprehensive test dataspanning 83 months from 2017 to 2023 and split the data into training andtesting periods according to models training data cutoff. We measure: 1 thecompression performance on the testing period as a measure of generalization onunseen data and 2 the performance gap between the training and testing periodas a measure of robustness. Our experiments test 14 representative largelanguage models with various sizes on sources including Wikipedia newsarticles code arXiv papers and multi-modal data. We find that thecompression rate of many models reduces significantly after their cutoff datebut models such as Mistral and Llama-2 demonstrate a good balance betweenperformance and robustness. Results also suggest that models struggle togeneralize on news and code data but work especially well on arXiv papers. Wealso find the context size and tokenization implementation have a big impact ofon the overall compression performance. |


| Item |Content|
| --- |---|
|idx| 2402.00854v1 |
|title| SymbolicAI: A framework for logic-based approaches combining generative models and solvers |
|authors| Marius-Constantin DinuClaudiu Leoveanu-CondreiMarkus HolzleitnerWerner ZellingerSepp Hochreiter
|links| http://arxiv.org/abs/2402.00854v1 |
|updated| 2024-02-01 18:50:50 UTC |
|summary| We introduce SymbolicAI a versatile and modular framework employing alogic-based approach to concept learning and flow management in generativeprocesses. SymbolicAI enables the seamless integration of generative modelswith a diverse range of solvers by treating large language models LLMs assemantic parsers that execute tasks based on both natural and formal languageinstructions thus bridging the gap between symbolic reasoning and generativeAI. We leverage probabilistic programming principles to tackle complex tasksand utilize differentiable and classical programming paradigms with theirrespective strengths. The framework introduces a set of polymorphiccompositional and self-referential operations for data stream manipulationaligning LLM outputs with user objectives. As a result we can transitionbetween the capabilities of various foundation models endowed with zero- andfew-shot learning capabilities and specialized fine-tuned models or solversproficient in addressing specific problems. In turn the framework facilitatesthe creation and evaluation of explainable computational graphs. We conclude byintroducing a quality measure and its empirical score for evaluating thesecomputational graphs and propose a benchmark that compares variousstate-of-the-art LLMs across a set of complex workflows. We refer to theempirical score as the Vector Embedding for Relational Trajectory Evaluationthrough Cross-similarity or VERTEX score for short. The framework codebaseand benchmark are linked below. |


| Item |Content|
| --- |---|
|idx| 2402.00839v1 |
|title| X-CBA: Explainability Aided CatBoosted Anomal-E for Intrusion Detection System |
|authors| Kiymet KayaElif AkSumeyye BasBerk CanberkSule Gunduz Oguducu
|links| http://arxiv.org/abs/2402.00839v1 |
|updated| 2024-02-01 18:29:16 UTC |
|summary| The effectiveness of Intrusion Detection Systems IDS is critical in an erawhere cyber threats are becoming increasingly complex. Machine learning MLand deep learning DL models provide an efficient and accurate solution foridentifying attacks and anomalies in computer networks. However using ML andDL models in IDS has led to a trust deficit due to their non-transparentdecision-making. This transparency gap in IDS research is significantaffecting confidence and accountability. To address this paper introduces anovel Explainable IDS approach called X-CBA that leverages the structuraladvantages of Graph Neural Networks GNNs to effectively process networktraffic data while also adapting a new Explainable AI XAI methodology.Unlike most GNN-based IDS that depend on labeled network traffic and nodefeatures thereby overlooking critical packet-level information our approachleverages a broader range of traffic data through network flows including edgeattributes to improve detection capabilities and adapt to novel threats.Through empirical testing we establish that our approach not only achieveshigh accuracy with 99.47 in threat detection but also advances the field byproviding clear actionable explanations of its analytical outcomes. Thisresearch also aims to bridge the current gap and facilitate the broaderintegration of ML/DL technologies in cybersecurity defenses by offering a localand global explainability solution that is both precise and interpretable. |


| Item |Content|
| --- |---|
|idx| 2402.00835v1 |
|title| ALISON: Fast and Effective Stylometric Authorship Obfuscation |
|authors| Eric XingSaranya VenkatramanThai LeDongwon Lee
|links| http://arxiv.org/abs/2402.00835v1 |
|updated| 2024-02-01 18:22:32 UTC |
|summary| Authorship Attribution AA and Authorship Obfuscation AO are two competingtasks of increasing importance in privacy research. Modern AA leverages anauthors consistent writing style to match a text to its author using an AAclassifier. AO is the corresponding adversarial task aiming to modify a textin such a way that its semantics are preserved yet an AA model cannotcorrectly infer its authorship. To address privacy concerns raised bystate-of-the-art SOTA AA methods new AO methods have been proposed butremain largely impractical to use due to their prohibitively slow training andobfuscation speed often taking hours. To this challenge we propose apractical AO method ALISON that 1 dramatically reduces training/obfuscationtime demonstrating more than 10x faster obfuscation than SOTA AO methods 2achieves better obfuscation success through attacking three transformer-basedAA methods on two benchmark datasets typically performing 15 better thancompeting methods 3 does not require direct signals from a target AAclassifier during obfuscation and 4 utilizes unique stylometric featuresallowing sound model interpretation for explainable obfuscation. We alsodemonstrate that ALISON can effectively prevent four SOTA AA methods fromaccurately determining the authorship of ChatGPT-generated texts all whileminimally changing the original text semantics. To ensure the reproducibilityof our findings our code and data are available at:https://github.com/EricX003/ALISON. |


| Item |Content|
| --- |---|
|idx| 2402.00831v1 |
|title| A YANG-aided Unified Strategy for Black Hole Detection for Backbone Networks |
|authors| Elif AkKiymet KayaEren OzaltunSule Gunduz OguducuBerk Canberk
|links| http://arxiv.org/abs/2402.00831v1 |
|updated| 2024-02-01 18:17:37 UTC |
|summary| Despite the crucial importance of addressing Black Hole failures in Internetbackbone networks effective detection strategies in backbone networks arelacking. This is largely because previous research has been centered on MobileAd-hoc Networks MANETs which operate under entirely different dynamicsprotocols and topologies making their findings not directly transferable tobackbone networks. Furthermore detecting Black Hole failures in backbonenetworks is particularly challenging. It requires a comprehensive range ofnetwork data due to the wide variety of conditions that need to be consideredmaking data collection and analysis far from straightforward. Addressing thisgap our study introduces a novel approach for Black Hole detection in backbonenetworks using specialized Yet Another Next Generation YANG data models withBlack Hole-sensitive Metric Matrix BHMM analysis. This paper details ourmethod of selecting and analyzing four YANG models relevant to Black Holedetection in ISP networks focusing on routing protocols and ISP-specificconfigurations. Our BHMM approach derived from these models demonstrates a 10improvement in detection accuracy and a 13 increase in packet delivery ratehighlighting the efficiency of our approach. Additionally we evaluate theMachine Learning approach leveraged with BHMM analysis in two different networksettings a commercial ISP network and a scientific research-only networktopology. This evaluation also demonstrates the practical applicability of ourmethod yielding significantly improved prediction outcomes in bothenvironments. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2402.00865v1 |
|title| Towards Optimal Feature-Shaping Methods for Out-of-Distribution Detection |
|authors| Qinyu ZhaoMing XuKartik GuptaAkshay AsthanaLiang ZhengStephen Gould
|links| http://arxiv.org/abs/2402.00865v1 |
|updated| 2024-02-01 18:59:22 UTC |
|summary| Feature shaping refers to a family of methods that exhibit state-of-the-artperformance for out-of-distribution OOD detection. These approachesmanipulate the feature representation typically from the penultimate layer ofa pre-trained deep learning model so as to better differentiate betweenin-distribution ID and OOD samples. However existing feature-shaping methodsusually employ rules manually designed for specific model architectures and OODdatasets which consequently limit their generalization ability. To addressthis gap we first formulate an abstract optimization framework for studyingfeature-shaping methods. We then propose a concrete reduction of the frameworkwith a simple piecewise constant shaping function and show that existingfeature-shaping methods approximate the optimal solution to the concreteoptimization problem. Further assuming that OOD data is inaccessible wepropose a formulation that yields a closed-form solution for the piecewiseconstant shaping function utilizing solely the ID data. Through extensiveexperiments we show that the feature-shaping function optimized by our methodimproves the generalization ability of OOD detection across a large variety ofdatasets and model architectures. |


| Item |Content|
| --- |---|
|idx| 2402.00857v1 |
|title| Early Time Classification with Accumulated Accuracy Gap Control |
|authors| Liran RingelRegev CohenDaniel FreedmanMichael EladYaniv Romano
|links| http://arxiv.org/abs/2402.00857v1 |
|updated| 2024-02-01 18:54:34 UTC |
|summary| Early time classification algorithms aim to label a stream of featureswithout processing the full input stream while maintaining accuracy comparableto that achieved by applying the classifier to the entire input. In this paperwe introduce a statistical framework that can be applied to any sequentialclassifier formulating a calibrated stopping rule. This data-driven ruleattains finite-sample distribution-free control of the accuracy gap betweenfull and early-time classification. We start by presenting a novel method thatbuilds on the Learn-then-Test calibration framework to control this gapmarginally on average over i.i.d. instances. As this algorithm tends to yieldan excessively high accuracy gap for early halt times our main contribution isthe proposal of a framework that controls a stronger notion of error where theaccuracy gap is controlled conditionally on the accumulated halt times.Numerical experiments demonstrate the effectiveness applicability andusefulness of our method. We show that our proposed early stopping mechanismreduces up to 94 of timesteps used for classification while achieving rigorousaccuracy gap control. |


| Item |Content|
| --- |---|
|idx| 2402.00854v1 |
|title| SymbolicAI: A framework for logic-based approaches combining generative models and solvers |
|authors| Marius-Constantin DinuClaudiu Leoveanu-CondreiMarkus HolzleitnerWerner ZellingerSepp Hochreiter
|links| http://arxiv.org/abs/2402.00854v1 |
|updated| 2024-02-01 18:50:50 UTC |
|summary| We introduce SymbolicAI a versatile and modular framework employing alogic-based approach to concept learning and flow management in generativeprocesses. SymbolicAI enables the seamless integration of generative modelswith a diverse range of solvers by treating large language models LLMs assemantic parsers that execute tasks based on both natural and formal languageinstructions thus bridging the gap between symbolic reasoning and generativeAI. We leverage probabilistic programming principles to tackle complex tasksand utilize differentiable and classical programming paradigms with theirrespective strengths. The framework introduces a set of polymorphiccompositional and self-referential operations for data stream manipulationaligning LLM outputs with user objectives. As a result we can transitionbetween the capabilities of various foundation models endowed with zero- andfew-shot learning capabilities and specialized fine-tuned models or solversproficient in addressing specific problems. In turn the framework facilitatesthe creation and evaluation of explainable computational graphs. We conclude byintroducing a quality measure and its empirical score for evaluating thesecomputational graphs and propose a benchmark that compares variousstate-of-the-art LLMs across a set of complex workflows. We refer to theempirical score as the Vector Embedding for Relational Trajectory Evaluationthrough Cross-similarity or VERTEX score for short. The framework codebaseand benchmark are linked below. |


| Item |Content|
| --- |---|
|idx| 2402.00853v1 |
|title| LTAU-FF: Loss Trajectory Analysis for Uncertainty in Atomistic Force Fields |
|authors| Joshua A. VitaAmit SamantaFei ZhouVincenzo Lordi
|links| http://arxiv.org/abs/2402.00853v1 |
|updated| 2024-02-01 18:50:42 UTC |
|summary| Model ensembles are simple and effective tools for estimating the predictionuncertainty of deep learning atomistic force fields. Despite this widespreadadoption of ensemble-based uncertainty quantification UQ techniques islimited by the high computational costs incurred by ensembles during bothtraining and inference. In this work we leverage the cumulative distributionfunctions CDFs of per-sample errors obtained over the course of training toefficiently represent the model ensemble and couple them with a distance-basedsimilarity search in the model latent space. Using these tools we develop asimple UQ metric which we call LTAU that leverages the strengths ofensemble-based techniques without requiring the evaluation of multiple modelsduring either training or inference. As an initial test we apply our methodtowards estimating the epistemic uncertainty in atomistic force fieldsLTAU-FF and demonstrate that it can be easily calibrated to accuratelypredict test errors on multiple datasets from the literature. We thenillustrate the utility of LTAU-FF in two practical applications: 1 tuning thetraining-validation gap for an example dataset and 2 predicting errors inrelaxation trajectories on the OC20 IS2RS task. Though in this work we focus onthe use of LTAU with deep learning atomistic force fields we emphasize that itcan be readily applied to any regression task or any ensemble-generationtechnique to provide a reliable and easy-to-implement UQ metric. |


| Item |Content|
| --- |---|
|idx| 2402.00851v1 |
|title| Data Augmentation Scheme for Raman Spectra with Highly Correlated Annotations |
|authors| Christoph LangeIsabel ThieleLara SantolinSebastian L. RiedelMaxim BorisyakPeter NeubauerM. Nicolas Cruz Bournazou
|links| http://arxiv.org/abs/2402.00851v1 |
|updated| 2024-02-01 18:46:28 UTC |
|summary| In biotechnology Raman Spectroscopy is rapidly gaining popularity as aprocess analytical technology PAT that measures cell densities substrate-and product concentrations. As it records vibrational modes of molecules itprovides that information non-invasively in a single spectrum. Typicallypartial least squares PLS is the model of choice to infer information aboutvariables of interest from the spectra. However biological processes are knownfor their complexity where convolutional neural networks CNN present apowerful alternative. They can handle non-Gaussian noise and account for beammisalignment pixel malfunctions or the presence of additional substances.However they require a lot of data during model training and they pick upnon-linear dependencies in the process variables. In this work we exploit theadditive nature of spectra in order to generate additional data points from agiven dataset that have statistically independent labels so that a networktrained on such data exhibits low correlations between the model predictions.We show that training a CNN on these generated data points improves theperformance on datasets where the annotations do not bear the same correlationas the dataset that was used for model training. This data augmentationtechnique enables us to reuse spectra as training data for new contexts thatexhibit different correlations. The additional data allows for building abetter and more robust model. This is of interest in scenarios where largeamounts of historical data are available but are currently not used for modeltraining. We demonstrate the capabilities of the proposed method usingsynthetic spectra of Ralstonia eutropha batch cultivations to monitorsubstrate biomass and polyhydroxyalkanoate PHA biopolymer concentrationsduring of the experiments. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2402.00867v1 |
|title| AToM: Amortized Text-to-Mesh using 2D Diffusion |
|authors| Guocheng QianJunli CaoAliaksandr SiarohinYash KantChaoyang WangMichael VasilkovskyHsin-Ying LeeYuwei FangIvan SkorokhodovPeiye ZhuangIgor GilitschenskiJian RenBernard GhanemKfir AbermanSergey Tulyakov
|links| http://arxiv.org/abs/2402.00867v1 |
|updated| 2024-02-01 18:59:56 UTC |
|summary| We introduce Amortized Text-to-Mesh AToM a feed-forward text-to-meshframework optimized across multiple text prompts simultaneously. In contrast toexisting text-to-3D methods that often entail time-consuming per-promptoptimization and commonly output representations other than polygonal meshesAToM directly generates high-quality textured meshes in less than 1 second witharound 10 times reduction in the training cost and generalizes to unseenprompts. Our key idea is a novel triplane-based text-to-mesh architecture witha two-stage amortized optimization strategy that ensures stable training andenables scalability. Through extensive experiments on various promptbenchmarks AToM significantly outperforms state-of-the-art amortizedapproaches with over 4 times higher accuracy in DF415 dataset and producesmore distinguishable and higher-quality 3D outputs. AToM demonstrates stronggeneralizability offering finegrained 3D assets for unseen interpolatedprompts without further optimization during inference unlike per-promptsolutions. |


| Item |Content|
| --- |---|
|idx| 2402.00868v1 |
|title| We're Not Using Videos Effectively: An Updated Domain Adaptive Video Segmentation Baseline |
|authors| Simar KareerVivek VijaykumarHarsh MaheshwariPrithvijit ChattopadhyayJudy HoffmanViraj Prabhu
|links| http://arxiv.org/abs/2402.00868v1 |
|updated| 2024-02-01 18:59:56 UTC |
|summary| There has been abundant work in unsupervised domain adaptation for semanticsegmentation DAS seeking to adapt a model trained on images from a labeledsource domain to an unlabeled target domain. While the vast majority of priorwork has studied this as a frame-level Image-DAS problem a few Video-DAS workshave sought to additionally leverage the temporal signal present in adjacentframes. However Video-DAS works have historically studied a distinct set ofbenchmarks from Image-DAS with minimal cross-benchmarking. In this work weaddress this gap. Surprisingly we find that 1 even after carefullycontrolling for data and model architecture state-of-the-art Image-DAS methodsHRDA and HRDAMIC outperform Video-DAS methods on established Video-DASbenchmarks 14.5 mIoU on ViperrightarrowCityscapesSeq 19.0 mIoU onSynthiarightarrowCityscapesSeq and 2 naive combinations of Image-DAS andVideo-DAS techniques only lead to marginal improvements across datasets. Toavoid siloed progress between Image-DAS and Video-DAS we open-source ourcodebase with support for a comprehensive set of Video-DAS and Image-DASmethods on a common benchmark. Code available athttps://github.com/SimarKareer/UnifiedVideoDA |


| Item |Content|
| --- |---|
|idx| 2402.00865v1 |
|title| Towards Optimal Feature-Shaping Methods for Out-of-Distribution Detection |
|authors| Qinyu ZhaoMing XuKartik GuptaAkshay AsthanaLiang ZhengStephen Gould
|links| http://arxiv.org/abs/2402.00865v1 |
|updated| 2024-02-01 18:59:22 UTC |
|summary| Feature shaping refers to a family of methods that exhibit state-of-the-artperformance for out-of-distribution OOD detection. These approachesmanipulate the feature representation typically from the penultimate layer ofa pre-trained deep learning model so as to better differentiate betweenin-distribution ID and OOD samples. However existing feature-shaping methodsusually employ rules manually designed for specific model architectures and OODdatasets which consequently limit their generalization ability. To addressthis gap we first formulate an abstract optimization framework for studyingfeature-shaping methods. We then propose a concrete reduction of the frameworkwith a simple piecewise constant shaping function and show that existingfeature-shaping methods approximate the optimal solution to the concreteoptimization problem. Further assuming that OOD data is inaccessible wepropose a formulation that yields a closed-form solution for the piecewiseconstant shaping function utilizing solely the ID data. Through extensiveexperiments we show that the feature-shaping function optimized by our methodimproves the generalization ability of OOD detection across a large variety ofdatasets and model architectures. |


| Item |Content|
| --- |---|
|idx| 2402.00864v1 |
|title| ViCA-NeRF: View-Consistency-Aware 3D Editing of Neural Radiance Fields |
|authors| Jiahua DongYu-Xiong Wang
|links| http://arxiv.org/abs/2402.00864v1 |
|updated| 2024-02-01 18:59:09 UTC |
|summary| We introduce ViCA-NeRF the first view-consistency-aware method for 3Dediting with text instructions. In addition to the implicit neural radiancefield NeRF modeling our key insight is to exploit two sources ofregularization that explicitly propagate the editing information acrossdifferent views thus ensuring multi-view consistency. For geometricregularization we leverage the depth information derived from NeRF toestablish image correspondences between different views. For learnedregularization we align the latent codes in the 2D diffusion model betweenedited and unedited images enabling us to edit key views and propagate theupdate throughout the entire scene. Incorporating these two strategies ourViCA-NeRF operates in two stages. In the initial stage we blend edits fromdifferent views to create a preliminary 3D edit. This is followed by a secondstage of NeRF training dedicated to further refining the scenes appearance.Experimental results demonstrate that ViCA-NeRF provides more flexibleefficient 3 times faster editing with higher levels of consistency anddetails compared with the state of the art. Our code is publicly available. |


| Item |Content|
| --- |---|
|idx| 2402.00863v2 |
|title| Geometry Transfer for Stylizing Radiance Fields |
|authors| Hyunyoung JungSeonghyeon NamNikolaos SarafianosSungjoo YooAlexander Sorkine-HornungRakesh Ranjan
|links| http://arxiv.org/abs/2402.00863v2 |
|updated| 2024-02-02 07:39:54 UTC |
|summary| Shape and geometric patterns are essential in defining stylistic identity.However current 3D style transfer methods predominantly focus on transferringcolors and textures often overlooking geometric aspects. In this paper weintroduce Geometry Transfer a novel method that leverages geometricdeformation for 3D style transfer. This technique employs depth maps to extracta style guide subsequently applied to stylize the geometry of radiance fields.Moreover we propose new techniques that utilize geometric cues from the 3Dscene thereby enhancing aesthetic expressiveness and more accuratelyreflecting intended styles. Our extensive experiments show that GeometryTransfer enables a broader and more expressive range of stylizations therebysignificantly expanding the scope of 3D style transfer. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2402.00857v1 |
|title| Early Time Classification with Accumulated Accuracy Gap Control |
|authors| Liran RingelRegev CohenDaniel FreedmanMichael EladYaniv Romano
|links| http://arxiv.org/abs/2402.00857v1 |
|updated| 2024-02-01 18:54:34 UTC |
|summary| Early time classification algorithms aim to label a stream of featureswithout processing the full input stream while maintaining accuracy comparableto that achieved by applying the classifier to the entire input. In this paperwe introduce a statistical framework that can be applied to any sequentialclassifier formulating a calibrated stopping rule. This data-driven ruleattains finite-sample distribution-free control of the accuracy gap betweenfull and early-time classification. We start by presenting a novel method thatbuilds on the Learn-then-Test calibration framework to control this gapmarginally on average over i.i.d. instances. As this algorithm tends to yieldan excessively high accuracy gap for early halt times our main contribution isthe proposal of a framework that controls a stronger notion of error where theaccuracy gap is controlled conditionally on the accumulated halt times.Numerical experiments demonstrate the effectiveness applicability andusefulness of our method. We show that our proposed early stopping mechanismreduces up to 94 of timesteps used for classification while achieving rigorousaccuracy gap control. |


| Item |Content|
| --- |---|
|idx| 2402.00849v1 |
|title| Score-based Causal Representation Learning: Linear and General Transformations |
|authors| Burak VarıcıEmre AcartürkKarthikeyan ShanmugamAbhishek KumarAli Tajer
|links| http://arxiv.org/abs/2402.00849v1 |
|updated| 2024-02-01 18:40:03 UTC |
|summary| This paper addresses intervention-based causal representation learning CRLunder a general nonparametric latent causal model and an unknown transformationthat maps the latent variables to the observed variables. Linear and generaltransformations are investigated. The paper addresses both theemphidentifiability and emphachievability aspects. Identifiability refersto determining algorithm-agnostic conditions that ensure recovering the truelatent causal variables and the latent causal graph underlying them.Achievability refers to the algorithmic aspects and addresses designingalgorithms that achieve identifiability guarantees. By drawing novelconnections between emphscore functions i.e. the gradients of thelogarithm of density functions and CRL this paper designs a emphscore-basedclass of algorithms that ensures both identifiability and achievability.First the paper focuses on emphlinear transformations and shows that onestochastic hard intervention per node suffices to guarantee identifiability. Italso provides partial identifiability guarantees for soft interventionsincluding identifiability up to ancestors for general causal models and perfectlatent graph recovery for sufficiently non-linear causal models. Secondly itfocuses on emphgeneral transformations and shows that two stochastic hardinterventions per node suffice for identifiability. Notably one doesemphnot need to know which pair of interventional environments have the samenode intervened. |


| Item |Content|
| --- |---|
|idx| 2402.00847v1 |
|title| BootsTAP: Bootstrapped Training for Tracking-Any-Point |
|authors| Carl DoerschYi YangDilara GokayPauline LucSkanda KoppulaAnkush GuptaJoseph HeywardRoss GoroshinJoão CarreiraAndrew Zisserman
|links| http://arxiv.org/abs/2402.00847v1 |
|updated| 2024-02-01 18:38:55 UTC |
|summary| To endow models with greater understanding of physics and motion it isuseful to enable them to perceive how solid surfaces move and deform in realscenes. This can be formalized as Tracking-Any-Point TAP which requires thealgorithm to be able to track any point corresponding to a solid surface in avideo potentially densely in space and time. Large-scale ground-truth trainingdata for TAP is only available in simulation which currently has limitedvariety of objects and motion. In this work we demonstrate how large-scaleunlabeled uncurated real-world data can improve a TAP model with minimalarchitectural changes using a self-supervised student-teacher setup. Wedemonstrate state-of-the-art performance on the TAP-Vid benchmark surpassingprevious results by a wide margin: for example TAP-Vid-DAVIS performanceimproves from 61.3 to 66.4 and TAP-Vid-Kinetics from 57.2 to 61.5. |


| Item |Content|
| --- |---|
|idx| 2402.00809v1 |
|title| Position Paper: Bayesian Deep Learning in the Age of Large-Scale AI |
|authors| Theodore PapamarkouMaria SkoularidouKonstantina PallaLaurence AitchisonJulyan ArbelDavid DunsonMaurizio FilipponeVincent FortuinPhilipp HennigAliaksandr HubinAlexander ImmerTheofanis KaraletsosMohammad Emtiyaz KhanAgustinus KristiadiYingzhen LiJose Miguel Hernandez LobatoStephan MandtChristopher NemethMichael A. OsborneTim G. J. RudnerDavid RügamerYee Whye TehMax WellingAndrew Gordon WilsonRuqi Zhang
|links| http://arxiv.org/abs/2402.00809v1 |
|updated| 2024-02-01 17:45:26 UTC |
|summary| In the current landscape of deep learning research there is a predominantemphasis on achieving high predictive accuracy in supervised tasks involvinglarge image and language datasets. However a broader perspective reveals amultitude of overlooked metrics tasks and data types such as uncertaintyactive and continual learning and scientific data that demand attention.Bayesian deep learning BDL constitutes a promising avenue offeringadvantages across these diverse settings. This paper posits that BDL canelevate the capabilities of deep learning. It revisits the strengths of BDLacknowledges existing challenges and highlights some exciting research avenuesaimed at addressing these obstacles. Looking ahead the discussion focuses onpossible ways to combine large-scale foundation models with BDL to unlock theirfull potential. |


| Item |Content|
| --- |---|
|idx| 2402.00776v1 |
|title| Hybrid Quantum Vision Transformers for Event Classification in High Energy Physics |
|authors| Eyup B. UnluMarçal Comajoan CaraGopal Ramesh DahaleZhongtian DongRoy T. ForestanoSergei GleyzerDaniel JusticeKyoungchul KongTom MagorschKonstantin T. MatchevKatia Matcheva
|links| http://arxiv.org/abs/2402.00776v1 |
|updated| 2024-02-01 17:05:37 UTC |
|summary| Models based on vision transformer architectures are consideredstate-of-the-art when it comes to image classification tasks. However theyrequire extensive computational resources both for training and deployment. Theproblem is exacerbated as the amount and complexity of the data increases.Quantum-based vision transformer models could potentially alleviate this issueby reducing the training and operating time while maintaining the samepredictive power. Although current quantum computers are not yet able toperform high-dimensional tasks yet they do offer one of the most efficientsolutions for the future. In this work we construct several variations of aquantum hybrid vision transformer for a classification problem in high energyphysics distinguishing photons and electrons in the electromagneticcalorimeter. We test them against classical vision transformer architectures.Our findings indicate that the hybrid models can achieve comparable performanceto their classical analogues with a similar number of parameters. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2402.00822v1 |
|title| WiOpen: A Robust Wi-Fi-based Open-set Gesture Recognition Framework |
|authors| Xiang ZhangJingyang HuangHuan YanPeng ZhaoGuohang ZhuangZhi LiuBin Liu
|links| http://arxiv.org/abs/2402.00822v1 |
|updated| 2024-02-01 18:05:38 UTC |
|summary| Recent years have witnessed a growing interest in Wi-Fi-based gesturerecognition. However existing works have predominantly focused on closed-setparadigms where all testing gestures are predefined during training. Thisposes a significant challenge in real-world applications as unseen gesturesmight be misclassified as known classes during testing. To address this issuewe propose WiOpen a robust Wi-Fi-based Open-Set Gesture Recognition OSGRframework. Implementing OSGR requires addressing challenges caused by theunique uncertainty in Wi-Fi sensing. This uncertainty resulting from noise anddomains leads to widely scattered and irregular data distributions incollected Wi-Fi sensing data. Consequently data ambiguity between classes andchallenges in defining appropriate decision boundaries to identify unknownsarise. To tackle these challenges WiOpen adopts a two-fold approach toeliminate uncertainty and define precise decision boundaries. Initially itaddresses uncertainty induced by noise during data preprocessing by utilizingthe CSI ratio. Next it designs the OSGR network based on an uncertaintyquantification method. Throughout the learning process this networkeffectively mitigates uncertainty stemming from domains. Ultimately thenetwork leverages relationships among samples neighbors to dynamically defineopen-set decision boundaries successfully realizing OSGR. Comprehensiveexperiments on publicly accessible datasets confirm WiOpens effectiveness.Notably WiOpen also demonstrates superiority in cross-domain tasks whencompared to state-of-the-art approaches. |


| Item |Content|
| --- |---|
|idx| 2402.00812v1 |
|title| Examining the Influence of Digital Phantom Models in Virtual Imaging Trials for Tomographic Breast Imaging |
|authors| Amar KavuriMini Das
|links| http://arxiv.org/abs/2402.00812v1 |
|updated| 2024-02-01 17:49:51 UTC |
|summary| Purpose: Digital phantoms are one of the key components of virtual imagingtrials VITs that aim to assess and optimize new medical imaging systems andalgorithms. However these phantoms vary in their voxel resolution appearanceand structural details. This study aims to examine whether and how variationsbetween digital phantoms influence system optimization with digital breasttomosynthesis DBT as a chosen modality. Methods: We selected widely used andopen-access digital breast phantoms generated with different methods. For eachphantom type we created an ensemble of DBT images to test acquisitionstrategies. Human observer localization ROC LROC was used to assess observerperformance studies for each case. Noise power spectrum NPS was estimated tocompare the phantom structural components. Further we computed several gazemetrics to quantify the gaze pattern when viewing images generated fromdifferent phantom types. Results: Our LROC results show that the arc samplingsfor peak performance were approximately 2.5 degrees and 6 degrees in Bakic andXCAT breast phantoms respectively for 3-mm lesion detection tasks and indicatethat system optimization outcomes from VITs can vary with phantom types andstructural frequency components. Additionally a significant correlation p0.01 between gaze metrics and diagnostic performance suggests that gazeanalysis can be used to understand and evaluate task difficulty in VITs. |


| Item |Content|
| --- |---|
|idx| 2402.00808v1 |
|title| Exploring the Dynamics between Cobot's Production Rhythm, Locus of Control and Emotional State in a Collaborative Assembly Scenario |
|authors| Marta MondelliniMatteo Lavit NicoraPooja PrajodElisabeth AndréRocco VertechyAlessandro AntoniettiMatteo Malosio
|links| http://arxiv.org/abs/2402.00808v1 |
|updated| 2024-02-01 17:44:46 UTC |
|summary| In industrial scenarios there is widespread use of collaborative robotscobots and growing interest is directed at evaluating and measuring theimpact of some characteristics of the cobot on the human factor. In the presentpilot study the effect that the production rhythm C1 - Slow C2 - Fast C3 -Adapted to the participants pace of a cobot has on the Experiential Locus ofControl ELoC and the emotional state of 31 participants has been examined.The operators performance the degree of basic internal Locus of Control andthe attitude towards the robots were also considered. No difference was foundregarding the emotional state and the ELoC in the three conditions butconsidering the other psychological variables a more complex situationemerges. Overall results seem to indicate a need to consider the personspsychological characteristics to offer a differentiated and optimal interactionexperience. |


| Item |Content|
| --- |---|
|idx| 2402.00793v1 |
|title| Distinguishing the Indistinguishable: Human Expertise in Algorithmic Prediction |
|authors| Rohan AlurManish RaghavanDevavrat Shah
|links| http://arxiv.org/abs/2402.00793v1 |
|updated| 2024-02-01 17:23:54 UTC |
|summary| We introduce a novel framework for incorporating human expertise intoalgorithmic predictions. Our approach focuses on the use of human judgment todistinguish inputs which look the same to any feasible predictive algorithm.We argue that this framing clarifies the problem of human/AI collaboration inprediction tasks as experts often have access to information -- particularlysubjective information -- which is not encoded in the algorithms trainingdata. We use this insight to develop a set of principled algorithms forselectively incorporating human feedback only when it improves the performanceof any feasible predictor. We find empirically that although algorithms oftenoutperform their human counterparts on average human judgment cansignificantly improve algorithmic predictions on specific instances which canbe identified ex-ante. In an X-ray classification task we find that thissubset constitutes nearly 30 of the patient population. Our approach providesa natural way of uncovering this heterogeneity and thus enabling effectivehuman-AI collaboration. |


| Item |Content|
| --- |---|
|idx| 2402.00764v1 |
|title| To Search or To Gen? Exploring the Synergy between Generative AI and Web Search in Programming |
|authors| Ryan YenNicole SultanumJian Zhao
|links| http://arxiv.org/abs/2402.00764v1 |
|updated| 2024-02-01 16:53:15 UTC |
|summary| The convergence of generative AI and web search is reshaping problem-solvingfor programmers. However the lack of understanding regarding their interplayin the information-seeking process often leads programmers to perceive them asalternatives rather than complementary tools. To analyze this interaction andexplore their synergy we conducted an interview study with eight experiencedprogrammers. Drawing from the results and literature we have identified threemajor challenges and proposed three decision-making stages each with its ownrelevant factors. Additionally we present a comprehensive process model thatcaptures programmers interaction patterns. This model encompassesdecision-making stages the information-foraging loop and cognitive activitiesduring system interaction offering a holistic framework to comprehend andoptimize the use of these convergent tools in programming. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2402.00787v1 |
|title| Learning and Calibrating Heterogeneous Bounded Rational Market Behaviour with Multi-Agent Reinforcement Learning |
|authors| Benjamin Patrick EvansSumitra Ganesh
|links| http://arxiv.org/abs/2402.00787v1 |
|updated| 2024-02-01 17:21:45 UTC |
|summary| Agent-based models ABMs have shown promise for modelling various real worldphenomena incompatible with traditional equilibrium analysis. However acritical concern is the manual definition of behavioural rules in ABMs. Recentdevelopments in multi-agent reinforcement learning MARL offer a way toaddress this issue from an optimisation perspective where agents strive tomaximise their utility eliminating the need for manual rule specification.This learning-focused approach aligns with established economic and financialmodels through the use of rational utility-maximising agents. However thisrepresentation departs from the fundamental motivation for ABMs: that realisticdynamics emerging from bounded rationality and agent heterogeneity can bemodelled. To resolve this apparent disparity between the two approaches wepropose a novel technique for representing heterogeneous processing-constrainedagents within a MARL framework. The proposed approach treats agents asconstrained optimisers with varying degrees of strategic skills permittingdeparture from strict utility maximisation. Behaviour is learnt throughrepeated simulations with policy gradients to adjust action likelihoods. Toallow efficient computation we use parameterised shared policy learning withdistributions of agent skill levels. Shared policy learning avoids the need foragents to learn individual policies yet still enables a spectrum of boundedrational behaviours. We validate our models effectiveness using real-worlddata on a range of canonical n-agent settings demonstrating significantlyimproved predictive capability. |


| Item |Content|
| --- |---|
|idx| 2402.00598v1 |
|title| A Promise Theory Perspective on the Role of Intent in Group Dynamics |
|authors| M. BurgessR. I. M. Dunbar
|links| http://arxiv.org/abs/2402.00598v1 |
|updated| 2024-02-01 13:51:35 UTC |
|summary| We present a simple argument using Promise Theory and dimensional analysisfor the Dunbar scaling hierarchy supported by recent data from group formationin Wikipedia editing. We show how the assumption of a common priority seedsgroup alignment until the costs associated with attending to the group outweighthe benefits in a detailed balance scenario. Subject to partial efficiency ofimplementing promised intentions we can reproduce a series of compatible ratesthat balance growth with entropy. |


| Item |Content|
| --- |---|
|idx| 2402.00595v1 |
|title| Group Related Phenomena in Wikipedia Edits |
|authors| M. BurgessR. I. M. Dunbar
|links| http://arxiv.org/abs/2402.00595v1 |
|updated| 2024-02-01 13:45:12 UTC |
|summary| Human communities have self-organizing properties that give rise to veryspecific natural grouping patterns reflected in the Dunbar Number and itslayered structure a Dunbar Graph. Since work-groups are necessarily alsosocial groups we might expect the same principles to apply here as well. Onefactor likely to be important in limiting the size of groups is that conflictstypically escalate with the number of people involved. Here we analyseWikipedia editing histories across a wide range of topics to show that there isan emergent coherence in the size of groups formed transiently to edit thecontent of subject texts with two peaks averaging at around N8 for the sizecorresponding to maximal contention and at around N4 as a regular team.These values are consistent with the observed sizes of conversational groupsas well as the hierarchical structuring of Dunbar graphs. We use the PromiseTheory of trust to suggest a scaling law that may apply to all groupdistributions based on seeded attraction. In addition to providing furtherevidence that even natural communities of strangers are self-organising theresults have important implications for the governance of the Wikipedia commonsand for the security of all online social platforms and associations. |


| Item |Content|
| --- |---|
|idx| 2402.00588v1 |
|title| BrainSLAM: SLAM on Neural Population Activity Data |
|authors| Kipp FreudNathan LeporaMatt W. JonesCian O'Donnell
|links| http://arxiv.org/abs/2402.00588v1 |
|updated| 2024-02-01 13:34:59 UTC |
|summary| Simultaneous localisation and mapping SLAM algorithms are commonly used inrobotic systems for learning maps of novel environments. Brains also appear tolearn maps but the mechanisms are not known and it is unclear how to inferthese maps from neural activity data. We present BrainSLAM a method forperforming SLAM using only population activity local field potential LFPdata simultaneously recorded from three brain regions in rats: hippocampusprefrontal cortex and parietal cortex. This system uses a convolutional neuralnetwork CNN to decode velocity and familiarity information from waveletscalograms of neural local field potential data recorded from rats as theynavigate a 2D maze. The CNNs output drives a RatSLAM-inspired architecturepowering an attractor network which performs path integration plus a separatesystem which performs loop closure detecting previously visited locationsand correcting map aliasing errors. Together these three components canconstruct faithful representations of the environment while simultaneouslytracking the animals location. This is the first demonstration of inference ofa spatial map from brain recordings. Our findings expand SLAM to a newmodality enabling a new method of mapping environments and facilitating abetter understanding of the role of cognitive maps in navigation and decisionmaking. |


| Item |Content|
| --- |---|
|idx| 2402.00334v1 |
|title| Multi-agent Path Finding for Cooperative Autonomous Driving |
|authors| Zhongxia YanHan ZhengCathy Wu
|links| http://arxiv.org/abs/2402.00334v1 |
|updated| 2024-02-01 04:39:15 UTC |
|summary| Anticipating possible future deployment of connected and automated vehiclesCAVs cooperative autonomous driving at intersections has been studied bymany works in control theory and intelligent transportation across decades.Simultaneously recent parallel works in robotics have devised efficientalgorithms for multi-agent path finding MAPF though often in environmentswith simplified kinematics. In this work we hybridize insights and algorithmsfrom MAPF with the structure and heuristics of optimizing the crossing order ofCAVs at signal-free intersections. We devise an optimal and complete algorithmOrder-based Search with Kinematics Arrival Time Scheduling OBS-KATS whichsignificantly outperforms existing algorithms fixed heuristics andprioritized planning with KATS. The performance is maintained under differentvehicle arrival rates lane lengths crossing speeds and control horizon.Through ablations and dissections we offer insight on the contributing factorsto OBS-KATSs performance. Our work is directly applicable to many similarlyscaled traffic and multi-robot scenarios with directed lanes. |


