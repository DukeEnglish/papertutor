# cs.CL 

| Item |Content|
| --- |---|
|idx| 2403.01342v1 |
|title| LM4OPT: Unveiling the Potential of Large Language Models in Formulating Mathematical Optimization Problems |
|authors| Tasnim AhmedSalimur Choudhury
|links| http://arxiv.org/abs/2403.01342v1 |
|updated| 2024-03-02 23:32:33 UTC |
|summary| In the rapidly evolving field of natural language processing the translationof linguistic descriptions into mathematical formulation of optimizationproblems presents a formidable challenge demanding intricate understanding andprocessing capabilities from Large Language Models LLMs. This study comparesprominent LLMs including GPT-3.5 GPT-4 and Llama-2-7b in zero-shot andone-shot settings for this task. Our findings show GPT-4s superiorperformance particularly in the one-shot scenario. A central part of thisresearch is the introduction of LM4OPT a progressive fine-tuning frameworkfor Llama-2-7b that utilizes noisy embeddings and specialized datasets.However this research highlights a notable gap in the contextual understandingcapabilities of smaller models such as Llama-2-7b compared to largercounterparts especially in processing lengthy and complex input contexts. Ourempirical investigation utilizing the NL4Opt dataset unveils that GPT-4surpasses the baseline performance established by previous research achievingan F1-score of 0.63 solely based on the problem description in naturallanguage and without relying on any additional named entity information.GPT-3.5 follows closely both outperforming the fine-tuned Llama-2-7b. Thesefindings not only benchmark the current capabilities of LLMs in a novelapplication area but also lay the groundwork for future improvements inmathematical formulation of optimization problems from natural language input. |


| Item |Content|
| --- |---|
|idx| 2403.01309v1 |
|title| VNLP: Turkish NLP Package |
|authors| Meliksah TurkerMehmet Erdi AriAydin Han
|links| http://arxiv.org/abs/2403.01309v1 |
|updated| 2024-03-02 20:46:56 UTC |
|summary| In this work we present VNLP: the first dedicated complete open-sourcewell-documented lightweight production-ready state-of-the-art NaturalLanguage Processing NLP package for the Turkish language. It contains a widevariety of tools ranging from the simplest tasks such as sentence splittingand text normalization to the more advanced ones such as text and tokenclassification models. Its token classification models are based on ContextModel a novel architecture that is both an encoder and an auto-regressivemodel. NLP tasks solved by VNLP models include but are not limited to SentimentAnalysis Named Entity Recognition Morphological Analysis  Disambiguationand Part-of-Speech Tagging. Moreover it comes with pre-trained word embeddingsand corresponding SentencePiece Unigram tokenizers. VNLP has an open-sourceGitHub repository ReadtheDocs documentation PyPi package for convenientinstallation Python and command-line API and a demo page to test all thefunctionality. Consequently our main contribution is a complete compacteasy-to-install and easy-to-use NLP package for Turkish. |


| Item |Content|
| --- |---|
|idx| 2403.01308v1 |
|title| VBART: The Turkish LLM |
|authors| Meliksah TurkerMehmet Erdi AriAydin Han
|links| http://arxiv.org/abs/2403.01308v1 |
|updated| 2024-03-02 20:40:11 UTC |
|summary| We present VBART the first Turkish sequence-to-sequence Large LanguageModels LLMs pre-trained on a large corpus from scratch. VBART are compactLLMs based on good ideas leveraged from BART and mBART models and come in twosizes Large and XLarge. Fine-tuned VBART models surpass the priorstate-of-the-art results in abstractive text summarization title generationtext paraphrasing question answering and question generation tasks. They allowfine-tuning for future text generation tasks and datasets carving a new pathfor Turkish Natural Language Processing NLP research. Our work shows thathaving a pre-trained LLM for Turkish outperforms up to 3x multilingual modelsimproving existing results and providing efficient models for training andinference. Moreover we show that our monolingual tokenizer is 7x moreefficient than OpenAIs multilingual tokenizer. Last but not least weintroduce a method to enlarge an existing pre-trained LLM and question therelevancy of Chinchilla Scaling Law to sequence-to-sequence masked languagemodels. Our fine-tuned models tokenizer and cleaned web corpus of 135 GB arepublicly available at huggingface.co/vngrs-ai. |


| Item |Content|
| --- |---|
|idx| 2403.01304v1 |
|title| Improving the Validity of Automatically Generated Feedback via Reinforcement Learning |
|authors| Alexander ScarlatosDigory SmithSimon WoodheadAndrew Lan
|links| http://arxiv.org/abs/2403.01304v1 |
|updated| 2024-03-02 20:25:50 UTC |
|summary| Automatically generating feedback via large language models LLMs inintelligent tutoring systems and online learning platforms has the potential toimprove the learning outcomes of many students. However both feedbackgeneration and evaluation are challenging: feedback content has to be validespecially in subjects like math which requires models to understand theproblem the solution and where the students error lies. Feedback also has tobe pedagogically valid to reflect effective tutoring strategies such asexplaining possible misconceptions and encouraging the student among otherdesirable features. In this work we address both problems of automaticallygenerating and evaluating feedback while considering both correctness andalignment. First we propose a rubric for evaluating math feedback and showthat GPT-4 is able to effectively use it to annotate human-written andLLM-generated feedback. Second we propose a framework for feedback generationthat optimizes both correctness and alignment using reinforcement learningRL. Specifically we use GPT-4s annotations to create preferences overfeedback pairs in an augmented dataset for training via direct preferenceoptimization DPO. We show that our methods significantly increase thecorrectness and alignment of generated feedback with Llama 2 an open-sourceLLM qualitatively analyze our generation and evaluation systems using casestudies and outline several areas for future work. |


| Item |Content|
| --- |---|
|idx| 2403.01289v1 |
|title| Greed is All You Need: An Evaluation of Tokenizer Inference Methods |
|authors| Omri UzanCraig W. SchmidtChris TannerYuval Pinter
|links| http://arxiv.org/abs/2403.01289v1 |
|updated| 2024-03-02 19:01:40 UTC |
|summary| While subword tokenizers such as BPE and WordPiece are typically used tobuild vocabularies for NLP models the method of decoding text into a sequenceof tokens from these vocabularies is often left unspecified or ill-suited tothe method in which they were constructed. We provide a controlled analysis ofseven tokenizer inference methods across four different algorithms and threevocabulary sizes performed on a novel intrinsic evaluation suite we curatedfor English combining measures rooted in morphology cognition andinformation theory. We show that for the most commonly used tokenizers greedyinference performs surprisingly well and that SaGe a recently-introducedcontextually-informed tokenizer outperforms all others on morphologicalalignment. |


# cs.AI 

| Item |Content|
| --- |---|
|idx| 2403.01348v1 |
|title| SANGRIA: Stacked Autoencoder Neural Networks with Gradient Boosting for Indoor Localization |
|authors| Danish GufranSaideep TikuSudeep Pasricha
|links| http://dx.doi.org/10.1109/LES.2023.3279017 |
|updated| 2024-03-03 00:01:29 UTC |
|summary| Indoor localization is a critical task in many embedded applications such asasset tracking emergency response and realtime navigation. In this articlewe propose a novel fingerprintingbased framework for indoor localization calledSANGRIA that uses stacked autoencoder neural networks with gradient boostedtrees. Our approach is designed to overcome the device heterogeneity challengethat can create uncertainty in wireless signal measurements across embeddeddevices used for localization. We compare SANGRIA to several state-of-the-artframeworks and demonstrate 42.96 lower average localization error acrossdiverse indoor locales and heterogeneous devices. |


| Item |Content|
| --- |---|
|idx| 2403.01332v1 |
|title| Chaining thoughts and LLMs to learn DNA structural biophysics |
|authors| Tyler D. RossAshwin Gopinath
|links| http://arxiv.org/abs/2403.01332v1 |
|updated| 2024-03-02 22:38:01 UTC |
|summary| The future development of an AI scientist a tool that is capable ofintegrating a variety of experimental data and generating testable hypothesesholds immense potential. So far bespoke machine learning models have beencreated to specialize in singular scientific tasks but otherwise lack theflexibility of a general purpose model. Here we show that a general purposelarge language model chatGPT 3.5-turbo can be fine-tuned to learn thestructural biophysics of DNA. We find that both fine-tuning models to returnchain-of-thought responses and chaining together models fine-tuned for subtaskshave an enhanced ability to analyze and design DNA sequences and theirstructures. |


| Item |Content|
| --- |---|
|idx| 2403.01329v1 |
|title| Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models |
|authors| Neta ShaulUriel SingerRicky T. Q. ChenMatthew LeAli ThabetAlbert PumarolaYaron Lipman
|links| http://arxiv.org/abs/2403.01329v1 |
|updated| 2024-03-02 22:27:44 UTC |
|summary| This paper introduces Bespoke Non-Stationary BNS Solvers a solverdistillation approach to improve sample efficiency of Diffusion and Flowmodels. BNS solvers are based on a family of non-stationary solvers thatprovably subsumes existing numerical ODE solvers and consequently demonstrateconsiderable improvement in sample approximation PSNR over these baselines.Compared to model distillation BNS solvers benefit from a tiny parameter space200 parameters fast optimization two orders of magnitude fastermaintain diversity of samples and in contrast to previous solver distillationapproaches nearly close the gap from standard distillation methods such asProgressive Distillation in the low-medium NFE regime. For example BNS solverachieves 45 PSNR / 1.76 FID using 16 NFE in class-conditional ImageNet-64. Weexperimented with BNS solvers for conditional image generation text-to-imagegeneration and text-2-audio generation showing significant improvement insample approximation PSNR in all. |


| Item |Content|
| --- |---|
|idx| 2403.01309v1 |
|title| VNLP: Turkish NLP Package |
|authors| Meliksah TurkerMehmet Erdi AriAydin Han
|links| http://arxiv.org/abs/2403.01309v1 |
|updated| 2024-03-02 20:46:56 UTC |
|summary| In this work we present VNLP: the first dedicated complete open-sourcewell-documented lightweight production-ready state-of-the-art NaturalLanguage Processing NLP package for the Turkish language. It contains a widevariety of tools ranging from the simplest tasks such as sentence splittingand text normalization to the more advanced ones such as text and tokenclassification models. Its token classification models are based on ContextModel a novel architecture that is both an encoder and an auto-regressivemodel. NLP tasks solved by VNLP models include but are not limited to SentimentAnalysis Named Entity Recognition Morphological Analysis  Disambiguationand Part-of-Speech Tagging. Moreover it comes with pre-trained word embeddingsand corresponding SentencePiece Unigram tokenizers. VNLP has an open-sourceGitHub repository ReadtheDocs documentation PyPi package for convenientinstallation Python and command-line API and a demo page to test all thefunctionality. Consequently our main contribution is a complete compacteasy-to-install and easy-to-use NLP package for Turkish. |


| Item |Content|
| --- |---|
|idx| 2403.01308v1 |
|title| VBART: The Turkish LLM |
|authors| Meliksah TurkerMehmet Erdi AriAydin Han
|links| http://arxiv.org/abs/2403.01308v1 |
|updated| 2024-03-02 20:40:11 UTC |
|summary| We present VBART the first Turkish sequence-to-sequence Large LanguageModels LLMs pre-trained on a large corpus from scratch. VBART are compactLLMs based on good ideas leveraged from BART and mBART models and come in twosizes Large and XLarge. Fine-tuned VBART models surpass the priorstate-of-the-art results in abstractive text summarization title generationtext paraphrasing question answering and question generation tasks. They allowfine-tuning for future text generation tasks and datasets carving a new pathfor Turkish Natural Language Processing NLP research. Our work shows thathaving a pre-trained LLM for Turkish outperforms up to 3x multilingual modelsimproving existing results and providing efficient models for training andinference. Moreover we show that our monolingual tokenizer is 7x moreefficient than OpenAIs multilingual tokenizer. Last but not least weintroduce a method to enlarge an existing pre-trained LLM and question therelevancy of Chinchilla Scaling Law to sequence-to-sequence masked languagemodels. Our fine-tuned models tokenizer and cleaned web corpus of 135 GB arepublicly available at huggingface.co/vngrs-ai. |


# cs.LG 

| Item |Content|
| --- |---|
|idx| 2403.01361v1 |
|title| Bandit Profit-maximization for Targeted Marketing |
|authors| Joon Suk HuhEllen VitercikKirthevasan Kandasamy
|links| http://arxiv.org/abs/2403.01361v1 |
|updated| 2024-03-03 01:33:47 UTC |
|summary| We study a sequential profit-maximization problem optimizing for both priceand ancillary variables like marketing expenditures. Specifically we aim tomaximize profit over an arbitrary sequence of multiple demand curves eachdependent on a distinct ancillary variable but sharing the same price. Aprototypical example is targeted marketing where a firm seller wishes tosell a product over multiple markets. The firm may invest different marketingexpenditures for different markets to optimize customer acquisition but mustmaintain the same price across all markets. Moreover markets may haveheterogeneous demand curves each responding to prices and marketingexpenditures differently. The firms objective is to maximize its gross profitthe total revenue minus marketing costs.  Our results are near-optimal algorithms for this class of problems in anadversarial bandit setting where demand curves are arbitrary non-adaptivesequences and the firm observes only noisy evaluations of chosen points on thedemand curves. We prove a regret upper bound ofwidetildemathcalObignT3/4big and a lower bound ofOmegabignT3/4big for monotonic demand curves and a regret bound ofwidetildeThetabignT2/3big for demands curves that are monotonic inprice and concave in the ancillary variables. |


| Item |Content|
| --- |---|
|idx| 2403.01355v1 |
|title| a-DCF: an architecture agnostic metric with application to spoofing-robust speaker verification |
|authors| Hye-jin ShimJee-weon JungTomi KinnunenNicholas EvansJean-Francois BonastreItshak Lapidot
|links| http://arxiv.org/abs/2403.01355v1 |
|updated| 2024-03-03 00:58:27 UTC |
|summary| Spoofing detection is today a mainstream research topic. Standard metrics canbe applied to evaluate the performance of isolated spoofing detection solutionsand others have been proposed to support their evaluation when they arecombined with speaker detection. These either have well-known deficiencies orrestrict the architectural approach to combine speaker and spoof detectors. Inthis paper we propose an architecture-agnostic detection cost functiona-DCF. A generalisation of the original DCF used widely for the assessment ofautomatic speaker verification ASV the a-DCF is designed for the evaluationof spoofing-robust ASV. Like the DCF the a-DCF reflects the cost of decisionsin a Bayes risk sense with explicitly defined class priors and detection costmodel. We demonstrate the merit of the a-DCF through the benchmarkingevaluation of architecturally-heterogeneous spoofing-robust ASV solutions. |


| Item |Content|
| --- |---|
|idx| 2403.01352v1 |
|title| Improving Uncertainty Sampling with Bell Curve Weight Function |
|authors| Zan-Kai ChongHiroyuki OhsakiBok-Min Goi
|links| http://dx.doi.org/10.17706/ijapm.2023.13.4.44-52 |
|updated| 2024-03-03 00:14:12 UTC |
|summary| Typically a supervised learning model is trained using passive learning byrandomly selecting unlabelled instances to annotate. This approach is effectivefor learning a model but can be costly in cases where acquiring labelledinstances is expensive. For example it can be time-consuming to manuallyidentify spam mails labelled instances from thousands of emails unlabelledinstances flooding an inbox during initial data collection. Generally weanswer the above scenario with uncertainty sampling an active learning methodthat improves the efficiency of supervised learning by using fewer labelledinstances than passive learning. Given an unlabelled data pool uncertaintysampling queries the labels of instances where the predicted probabilities pfall into the uncertainty region i.e. p approx 0.5. The newly acquiredlabels are then added to the existing labelled data pool to learn a new model.Nonetheless the performance of uncertainty sampling is susceptible to the areaof unpredictable responses AUR and the nature of the dataset. It is difficultto determine whether to use passive learning or uncertainty sampling withoutprior knowledge of a new dataset. To address this issue we propose bell curvesampling which employs a bell curve weight function to acquire new labels.With the bell curve centred at p0.5 bell curve sampling selects instanceswhose predicted values are in the uncertainty area most of the time withoutneglecting the rest. Simulation results show that most of the time bell curvesampling outperforms uncertainty sampling and passive learning in datasets ofdifferent natures and with AUR. |


| Item |Content|
| --- |---|
|idx| 2403.01348v1 |
|title| SANGRIA: Stacked Autoencoder Neural Networks with Gradient Boosting for Indoor Localization |
|authors| Danish GufranSaideep TikuSudeep Pasricha
|links| http://dx.doi.org/10.1109/LES.2023.3279017 |
|updated| 2024-03-03 00:01:29 UTC |
|summary| Indoor localization is a critical task in many embedded applications such asasset tracking emergency response and realtime navigation. In this articlewe propose a novel fingerprintingbased framework for indoor localization calledSANGRIA that uses stacked autoencoder neural networks with gradient boostedtrees. Our approach is designed to overcome the device heterogeneity challengethat can create uncertainty in wireless signal measurements across embeddeddevices used for localization. We compare SANGRIA to several state-of-the-artframeworks and demonstrate 42.96 lower average localization error acrossdiverse indoor locales and heterogeneous devices. |


| Item |Content|
| --- |---|
|idx| 2403.01346v1 |
|title| Improve Cost Efficiency of Active Learning over Noisy Dataset |
|authors| Zan-Kai ChongHiroyuki OhsakiBryan Ng
|links| http://arxiv.org/abs/2403.01346v1 |
|updated| 2024-03-02 23:53:24 UTC |
|summary| Active learning is a learning strategy whereby the machine learning algorithmactively identifies and labels data points to optimize its learning. Thisstrategy is particularly effective in domains where an abundance of unlabeleddata exists but the cost of labeling these data points is prohibitivelyexpensive. In this paper we consider cases of binary classification whereacquiring a positive instance incurs a significantly higher cost compared tothat of negative instances. For example in the financial industry such as inmoney-lending businesses a defaulted loan constitutes a positive event leadingto substantial financial loss. To address this issue we propose a shiftednormal distribution sampling function that samples from a wider range thantypical uncertainty sampling. Our simulation underscores that our proposedsampling function limits both noisy and positive label selection deliveringbetween 20 and 32 improved cost efficiency over different test datasets. |


# cs.CV 

| Item |Content|
| --- |---|
|idx| 2403.01362v1 |
|title| Enhancing Retinal Vascular Structure Segmentation in Images With a Novel Design Two-Path Interactive Fusion Module Model |
|authors| Rui YangShunpu Zhang
|links| http://arxiv.org/abs/2403.01362v1 |
|updated| 2024-03-03 01:36:11 UTC |
|summary| Precision in identifying and differentiating micro and macro blood vessels inthe retina is crucial for the diagnosis of retinal diseases although it posesa significant challenge. Current autoencoding-based segmentation approachesencounter limitations as they are constrained by the encoder and undergo areduction in resolution during the encoding stage. The inability to recoverlost information in the decoding phase further impedes these approaches.Consequently their capacity to extract the retinal microvascular structure isrestricted. To address this issue we introduce Swin-Res-Net a specializedmodule designed to enhance the precision of retinal vessel segmentation.Swin-Res-Net utilizes the Swin transformer which uses shifted windows withdisplacement for partitioning to reduce network complexity and acceleratemodel convergence. Additionally the model incorporates interactive fusion witha functional module in the Res2Net architecture. The Res2Net leveragesmulti-scale techniques to enlarge the receptive field of the convolutionalkernel enabling the extraction of additional semantic information from theimage. This combination creates a new module that enhances the localization andseparation of micro vessels in the retina. To improve the efficiency ofprocessing vascular information weve added a module to eliminate redundantinformation between the encoding and decoding steps.  Our proposed architecture produces outstanding results either meeting orsurpassing those of other published models. The AUC reflects significantenhancements achieving values of 0.9956 0.9931 and 0.9946 in pixel-wisesegmentation of retinal vessels across three widely utilized datasets:CHASE-DB1 DRIVE and STARE respectively. Moreover Swin-Res-Net outperformsalternative architectures demonstrating superior performance in both IOU andF1 measure metrics. |


| Item |Content|
| --- |---|
|idx| 2403.01345v1 |
|title| ShapeBoost: Boosting Human Shape Estimation with Part-Based Parameterization and Clothing-Preserving Augmentation |
|authors| Siyuan BianJiefeng LiJiasheng TangCewu Lu
|links| http://arxiv.org/abs/2403.01345v1 |
|updated| 2024-03-02 23:40:23 UTC |
|summary| Accurate human shape recovery from a monocular RGB image is a challengingtask because humans come in different shapes and sizes and wear differentclothes. In this paper we propose ShapeBoost a new human shape recoveryframework that achieves pixel-level alignment even for rare body shapes andhigh accuracy for people wearing different types of clothes. Unlike previousapproaches that rely on the use of PCA-based shape coefficients we adopt a newhuman shape parameterization that decomposes the human shape into bone lengthsand the mean width of each part slice. This part-based parameterizationtechnique achieves a balance between flexibility and validity using asemi-analytical shape reconstruction algorithm. Based on this newparameterization a clothing-preserving data augmentation module is proposed togenerate realistic images with diverse body shapes and accurate annotations.Experimental results show that our method outperforms other state-of-the-artmethods in diverse body shape situations as well as in varied clothingsituations. |


| Item |Content|
| --- |---|
|idx| 2403.01344v1 |
|title| Mitigating the Bias in the Model for Continual Test-Time Adaptation |
|authors| Inseop ChungKyomin HwangJayeon YooNojun Kwak
|links| http://arxiv.org/abs/2403.01344v1 |
|updated| 2024-03-02 23:37:16 UTC |
|summary| Continual Test-Time Adaptation CTA is a challenging task that aims to adapta source pre-trained model to continually changing target domains. In the CTAsetting a model does not know when the target domain changes thus facing adrastic change in the distribution of streaming inputs during the test-time.The key challenge is to keep adapting the model to the continually changingtarget domains in an online manner. We find that a model shows highly biasedpredictions as it constantly adapts to the chaining distribution of the targetdata. It predicts certain classes more often than other classes makinginaccurate over-confident predictions. This paper mitigates this issue toimprove performance in the CTA scenario. To alleviate the bias issue we makeclass-wise exponential moving average target prototypes with reliable targetsamples and exploit them to cluster the target features class-wisely. Moreoverwe aim to align the target distributions to the source distribution byanchoring the target feature to its corresponding source prototype. Withextensive experiments our proposed method achieves noteworthy performance gainwhen applied on top of existing CTA methods without substantial adaptation timeoverhead. |


| Item |Content|
| --- |---|
|idx| 2403.01329v1 |
|title| Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models |
|authors| Neta ShaulUriel SingerRicky T. Q. ChenMatthew LeAli ThabetAlbert PumarolaYaron Lipman
|links| http://arxiv.org/abs/2403.01329v1 |
|updated| 2024-03-02 22:27:44 UTC |
|summary| This paper introduces Bespoke Non-Stationary BNS Solvers a solverdistillation approach to improve sample efficiency of Diffusion and Flowmodels. BNS solvers are based on a family of non-stationary solvers thatprovably subsumes existing numerical ODE solvers and consequently demonstrateconsiderable improvement in sample approximation PSNR over these baselines.Compared to model distillation BNS solvers benefit from a tiny parameter space200 parameters fast optimization two orders of magnitude fastermaintain diversity of samples and in contrast to previous solver distillationapproaches nearly close the gap from standard distillation methods such asProgressive Distillation in the low-medium NFE regime. For example BNS solverachieves 45 PSNR / 1.76 FID using 16 NFE in class-conditional ImageNet-64. Weexperimented with BNS solvers for conditional image generation text-to-imagegeneration and text-2-audio generation showing significant improvement insample approximation PSNR in all. |


| Item |Content|
| --- |---|
|idx| 2403.01326v1 |
|title| DNA Family: Boosting Weight-Sharing NAS with Block-Wise Supervisions |
|authors| Guangrun WangChanglin LiLiuchun YuanJiefeng PengXiaoyu XianXiaodan LiangXiaojun ChangLiang Lin
|links| http://arxiv.org/abs/2403.01326v1 |
|updated| 2024-03-02 22:16:47 UTC |
|summary| Neural Architecture Search NAS aiming at automatically designing neuralarchitectures by machines has been considered a key step toward automaticmachine learning. One notable NAS branch is the weight-sharing NAS whichsignificantly improves search efficiency and allows NAS algorithms to run onordinary computers. Despite receiving high expectations this category ofmethods suffers from low search effectiveness. By employing a generalizationboundedness tool we demonstrate that the devil behind this drawback is theuntrustworthy architecture rating with the oversized search space of thepossible architectures. Addressing this problem we modularize a large searchspace into blocks with small search spaces and develop a family of models withthe distilling neural architecture DNA techniques. These proposed modelsnamely a DNA family are capable of resolving multiple dilemmas of theweight-sharing NAS such as scalability efficiency and multi-modalcompatibility. Our proposed DNA models can rate all architecture candidates asopposed to previous works that can only access a subsearch space usingheuristic algorithms. Moreover under a certain computational complexityconstraint our method can seek architectures with different depths and widths.Extensive experimental evaluations show that our models achievestate-of-the-art top-1 accuracy of 78.9 and 83.6 on ImageNet for a mobileconvolutional network and a small vision transformer respectively.Additionally we provide in-depth empirical analysis and insights into neuralarchitecture ratings. Codes available: urlhttps://github.com/changlin31/DNA. |


# stat.ML 

| Item |Content|
| --- |---|
|idx| 2403.01318v1 |
|title| High-Dimensional Tail Index Regression: with An Application to Text Analyses of Viral Posts in Social Media |
|authors| Yuya SasakiJing TaoYulong Wang
|links| http://arxiv.org/abs/2403.01318v1 |
|updated| 2024-03-02 21:37:40 UTC |
|summary| Motivated by the empirical power law of the distributions of credits e.g.the number of likes of viral posts in social media we introduce thehigh-dimensional tail index regression and methods of estimation and inferencefor its parameters. We propose a regularized estimator establish itsconsistency and derive its convergence rate. To conduct inference we proposeto debias the regularized estimate and establish the asymptotic normality ofthe debiased estimator. Simulation studies support our theory. These methodsare applied to text analyses of viral posts in X formerly Twitter concerningLGBTQ. |


| Item |Content|
| --- |---|
|idx| 2403.01315v1 |
|title| Near-optimal Per-Action Regret Bounds for Sleeping Bandits |
|authors| Quan NguyenNishant A. Mehta
|links| http://arxiv.org/abs/2403.01315v1 |
|updated| 2024-03-02 21:22:46 UTC |
|summary| We derive near-optimal per-action regret bounds for sleeping bandits inwhich both the sets of available arms and their losses in every round arechosen by an adversary. In a setting with K total arms and at most Aavailable arms in each round over T rounds the best known upper bound isOKsqrtTAlnK obtained indirectly via minimizing internal sleepingregrets. Compared to the minimax OmegasqrtTA lower bound this upperbound contains an extra multiplicative factor of KlnK. We address this gapby directly minimizing the per-action regret using generalized versions ofEXP3 EXP3-IX and FTRL with Tsallis entropy thereby obtaining near-optimalbounds of order OsqrtTAlnK and OsqrtTsqrtAK. We extend ourresults to the setting of bandits with advice from sleeping expertsgeneralizing EXP4 along the way. This leads to new proofs for a number ofexisting adaptive and tracking regret bounds for standard non-sleeping bandits.Extending our results to the bandit version of experts that report theirconfidences leads to new bounds for the confidence regret that dependsprimarily on the sum of experts confidences. We prove a lower bound showingthat for any minimax optimal algorithms there exists an action whose regret issublinear in T but linear in the number of its active rounds. |


| Item |Content|
| --- |---|
|idx| 2403.01272v1 |
|title| Can a Confident Prior Replace a Cold Posterior? |
|authors| Martin MarekBrooks PaigePavel Izmailov
|links| http://arxiv.org/abs/2403.01272v1 |
|updated| 2024-03-02 17:28:55 UTC |
|summary| Benchmark datasets used for image classification tend to have very low levelsof label noise. When Bayesian neural networks are trained on these datasetsthey often underfit misrepresenting the aleatoric uncertainty of the data. Acommon solution is to cool the posterior which improves fit to the trainingdata but is challenging to interpret from a Bayesian perspective. We explorewhether posterior tempering can be replaced by a confidence-inducing priordistribution. First we introduce a DirClip prior that is practical to sampleand nearly matches the performance of a cold posterior. Second we introduce aconfidence prior that directly approximates a cold likelihood in the limit ofdecreasing temperature but cannot be easily sampled. Lastly we provide severalgeneral insights into confidence-inducing priors such as when they mightdiverge and how fine-tuning can mitigate numerical instability. |


| Item |Content|
| --- |---|
|idx| 2403.01204v1 |
|title| Stochastic gradient descent for streaming linear and rectified linear systems with Massart noise |
|authors| Halyun JeongDeanna NeedellElizaveta Rebrova
|links| http://arxiv.org/abs/2403.01204v1 |
|updated| 2024-03-02 12:45:01 UTC |
|summary| We propose SGD-exp a stochastic gradient descent approach for linear andReLU regressions under Massart noise adversarial semi-random corruption modelfor the fully streaming setting. We show novel nearly linear convergenceguarantees of SGD-exp to the true parameter with up to 50 Massartcorruption rate and with any corruption rate in the case of symmetricoblivious corruptions. This is the first convergence guarantee result forrobust ReLU regression in the streaming setting and it shows the improvedconvergence rate over previous robust methods for L_1 linear regression dueto a choice of an exponentially decaying step size known for its efficiency inpractice. Our analysis is based on the drift analysis of a discrete stochasticprocess which could also be interesting on its own. |


| Item |Content|
| --- |---|
|idx| 2403.01046v1 |
|title| A Library of Mirrors: Deep Neural Nets in Low Dimensions are Convex Lasso Models with Reflection Features |
|authors| Emi ZegerYifei WangAaron MishkinTolga ErgenEmmanuel CandèsMert Pilanci
|links| http://arxiv.org/abs/2403.01046v1 |
|updated| 2024-03-02 00:33:45 UTC |
|summary| We prove that training neural networks on 1-D data is equivalent to solving aconvex Lasso problem with a fixed explicitly defined dictionary matrix offeatures. The specific dictionary depends on the activation and depth. Weconsider 2-layer networks with piecewise linear activations deep narrow ReLUnetworks with up to 4 layers and rectangular and tree networks with signactivation and arbitrary depth. Interestingly in ReLU networks a fourth layercreates features that represent reflections of training data about themselves.The Lasso representation sheds insight to globally optimal networks and thesolution landscape. |


# cs.HC 

| Item |Content|
| --- |---|
|idx| 2403.01335v1 |
|title| Making Hybrid Languages: A Recipe |
|authors| Leif AndersenCameron MoyStephen ChangMatthias Felleisen
|links| http://arxiv.org/abs/2403.01335v1 |
|updated| 2024-03-02 22:53:06 UTC |
|summary| The dominant programming languages support only linear text to express ideas.Visual languages offer graphical representations for entire programs whenviewed with special tools. Hybrid languages with support from existing toolsallow developers to express their ideas with a mix of textual and graphicalsyntax tailored to an application domain. This mix puts both kinds of syntax onequal footing and importantly the enriched language does not disrupt aprogrammers typical workflow. This paper presents a recipe for equippingexisting textual programming languages as well as accompanying IDEs with amechanism for creating and using graphical interactive syntax. It also presentsthe first hybrid language and IDE created using the recipe. |


| Item |Content|
| --- |---|
|idx| 2403.01242v1 |
|title| Augmenting Automation: Intent-Based User Instruction Classification with Machine Learning |
|authors| Lochan BasyalBijay Gaudel
|links| http://arxiv.org/abs/2403.01242v1 |
|updated| 2024-03-02 16:06:03 UTC |
|summary| Electric automation systems offer convenience and efficiency in controllingelectrical circuits and devices. Traditionally these systems rely onpredefined commands for control limiting flexibility and adaptability. In thispaper we propose a novel approach to augment automation by introducingintent-based user instruction classification using machine learning techniques.Our system represents user instructions as intents allowing for dynamiccontrol of electrical circuits without relying on predefined commands. Througha machine learning model trained on a labeled dataset of user instructions oursystem classifies intents from user input enabling a more intuitive andadaptable control scheme. We present the design and implementation of ourintent-based electric automation system detailing the development of themachine learning model for intent classification. Experimental resultsdemonstrate the effectiveness of our approach in enhancing user experience andexpanding the capabilities of electric automation systems. Our work contributesto the advancement of smart technologies by providing a more seamlessinteraction between users and their environments. |


| Item |Content|
| --- |---|
|idx| 2403.01208v1 |
|title| The Science of Data Collection: Insights from Surveys can Improve Machine Learning Models |
|authors| Stephanie EckmanBarbara PlankFrauke Kreuter
|links| http://arxiv.org/abs/2403.01208v1 |
|updated| 2024-03-02 13:34:43 UTC |
|summary| Whether future AI models make the world safer or less safe for humans restsin part on our ability to efficiently collect accurate data from people aboutwhat they want the models to do. However collecting high quality data isdifficult and most AI/ML researchers are not trained in data collectionmethods. The growing emphasis on data-centric AI highlights the potential ofdata to enhance model performance. It also reveals an opportunity to gaininsights from survey methodology the science of collecting high-quality surveydata.  In this position paper we summarize lessons from the survey methodologyliterature and discuss how they can improve the quality of training andfeedback data which in turn improve model performance. Based on the cognitiveresponse process model we formulate specific hypotheses about the aspects oflabel collection that may impact training data quality. We also suggestcollaborative research ideas into how possible biases in data collection can bemitigated making models more accurate and human-centric. |


| Item |Content|
| --- |---|
|idx| 2403.01127v1 |
|title| Towards RehabCoach: Design and Preliminary Evaluation of a Conversational Agent Supporting Unsupervised Therapy after Stroke |
|authors| Giada DevittoriMehdi AkeddarAlexandra RetevoiFabian SchneiderViktoria CvetkovaDaria DinacciAntonella CaliffiPaolo RossiClaudio PetrilloTobias KowatschOlivier Lambercy
|links| http://arxiv.org/abs/2403.01127v1 |
|updated| 2024-03-02 08:18:02 UTC |
|summary| Unsupervised therapy after stroke is a promising way to boost therapy dosewithout significantly increasing the workload on healthcare professionals.However it raises important challenges such as lower adherence to therapy inthe absence of social interaction with therapists. We present the initialprototype of RehabCoach a novel smartphone-based app with conversational agentto support unsupervised therapy. RehabCoach is designed to increase patientsengagement and adherence to therapy and to provide information e.g. aboutstroke health in an interactive and user-friendly manner. We report on thedesign and usability evaluation of the first prototype of RehabCoach assessedby four stroke patients and five healthcare professionals who interacted withthe app in a single testing session. Task completion time and success rateswere measured for 15 representative tasks and participants assessed usabilityvia questionnaires and a semi-structured interview. Results show that it wasfeasible for stroke patients to successfully interact with RehabCoach tasksuccess geq 93  without requiring extensive training. Participantspositively rated the usability of RehabCoach mean mHealth App UsabilityQuestionnaire score: 1.3 for primary users 1.4 for healthcare professionalson a scale from 1 positive evaluation to 7. The feedback collected in thiswork opens the door to further enhance RehabCoach as an interactive digitaltool to support unsupervised rehabilitation. |


| Item |Content|
| --- |---|
|idx| 2403.01090v1 |
|title| Sharing Frissons among Online Video Viewers: Exploring the Design of Affective Communication for Aesthetic Chills |
|authors| Zeyu HuangXinyi CaoYuanhao ZhangXiaojuan Ma
|links| http://dx.doi.org/10.1145/3613904.3642818 |
|updated| 2024-03-02 04:21:21 UTC |
|summary| On online video platforms viewers often lack a channel to sense others andexpress their affective state on the fly compared to co-located group-viewing.This study explored the design of complementary affective communicationspecifically for effortless spontaneous sharing of frissons during videowatching. Also known as aesthetic chills frissons are instantpsycho-physiological reactions like goosebumps and shivers to arousing stimuli.We proposed an approach that unobtrusively detects viewers frissons using skinelectrodermal activity sensors and presents the aggregated data alongsideonline videos. Following a design process of brainstorming focus groupinterview N7 and design iterations we proposed three different designs toencode viewers frisson experiences namely ambient light icon andvibration. A mixed-methods within-subject study N48 suggested that ourapproach offers a non-intrusive and efficient way to share viewers frissonmoments increases the social presence of others as if watching together andcan create affective contagion among viewers. |


# cs.MA 

| Item |Content|
| --- |---|
|idx| 2403.01286v1 |
|title| Summary Paper: Use Case on Building Collaborative Safe Autonomous Systems-A Robotdog for Guiding Visually Impaired People |
|authors| Aman MalhotraSelma Saidi
|links| http://arxiv.org/abs/2403.01286v1 |
|updated| 2024-03-02 18:59:03 UTC |
|summary| This is a summary paper of a use case of a Robotdog dedicated to guidevisually impaired people in complex environment like a smart intersection. Insuch scenarios the Robotdog has to autonomously decide whether it is safe tocross the intersection or not in order to further guide the human. We leveragedata sharing and collaboration between the Robotdog and other autonomoussystems operating in the same environment. We propose a system architecture forautonomous systems through a separation of a collaborative decision layer toenable collective decision making processes where data about the environmentrelevant to the Robotdog decision together with evidences for trustworthinessabout other systems and the environment are shared. |


| Item |Content|
| --- |---|
|idx| 2403.01277v1 |
|title| Optimal Integrated Task and Path Planning and Its Application to Multi-Robot Pickup and Delivery |
|authors| Aman AryanManan ModiIndranil SahaRupak MajumdarSwarup Mohalik
|links| http://arxiv.org/abs/2403.01277v1 |
|updated| 2024-03-02 17:48:40 UTC |
|summary| We propose a generic multi-robot planning mechanism that combines an optimaltask planner and an optimal path planner to provide a scalable solution forcomplex multi-robot planning problems. The Integrated planner through theinteraction of the task planner and the path planner produces optimalcollision-free trajectories for the robots. We illustrate our general algorithmon an object pick-and-drop planning problem in a warehouse scenario where agroup of robots is entrusted with moving objects from one location to anotherin the workspace. We solve the task planning problem by reducing it into anSMT-solving problem and employing the highly advanced SMT solver Z3 to solveit. To generate collision-free movement of the robots we extend thestate-of-the-art algorithm Conflict Based Search with Precedence Constraintswith several domain-specific constraints. We evaluate our integrated task andpath planner extensively on various instances of the object pick-and-dropplanning problem and compare its performance with a state-of-the-artmulti-robot classical planner. Experimental results demonstrate that ourplanning mechanism can deal with complex planning problems and outperforms astate-of-the-art classical planner both in terms of computation time and thequality of the generated plan. |


| Item |Content|
| --- |---|
|idx| 2403.01112v1 |
|title| Efficient Episodic Memory Utilization of Cooperative Multi-Agent Reinforcement Learning |
|authors| Hyungho NaYunkyeong SeoIl-chul Moon
|links| http://arxiv.org/abs/2403.01112v1 |
|updated| 2024-03-02 07:37:05 UTC |
|summary| In cooperative multi-agent reinforcement learning MARL agents aim toachieve a common goal such as defeating enemies or scoring a goal. ExistingMARL algorithms are effective but still require significant learning time andoften get trapped in local optima by complex tasks subsequently failing todiscover a goal-reaching policy. To address this we introduce Efficientepisodic Memory Utilization EMU for MARL with two primary objectives: aaccelerating reinforcement learning by leveraging semantically coherent memoryfrom an episodic buffer and b selectively promoting desirable transitions toprevent local convergence. To achieve a EMU incorporates a trainableencoder/decoder structure alongside MARL creating coherent memory embeddingsthat facilitate exploratory memory recall. To achieve b EMU introduces anovel reward structure called episodic incentive based on the desirability ofstates. This reward improves the TD target in Q-learning and acts as anadditional incentive for desirable transitions. We provide theoretical supportfor the proposed incentive and demonstrate the effectiveness of EMU compared toconventional episodic control. The proposed method is evaluated in StarCraft IIand Google Research Football and empirical results indicate furtherperformance improvement over state-of-the-art methods. |


| Item |Content|
| --- |---|
|idx| 2403.00987v1 |
|title| Composite Distributed Learning and Synchronization of Nonlinear Multi-Agent Systems with Complete Uncertain Dynamics |
|authors| Emadodin JandaghiDalton L. SteinAdam HoburgMingxi ZhouChengzhi Yuan
|links| http://arxiv.org/abs/2403.00987v1 |
|updated| 2024-03-01 21:19:28 UTC |
|summary| This paper addresses the challenging problem of composite synchronization andlearning control in a network of multi-agent robotic manipulator systemsoperating under heterogeneous nonlinear uncertainties within a leader-followerframework. A novel two-layer distributed adaptive learning control strategy isintroduced comprising a first-layer distributed cooperative estimator and asecond-layer decentralized deterministic learning controller. The primaryobjective of the first layer is to facilitate each robotic agents estimationof the leaders information. The second layer is responsible for both enablingindividual robot agents to track desired reference trajectories and accuratelyidentifying and learning their nonlinear uncertain dynamics. The proposeddistributed learning control scheme represents an advancement in the existingliterature due to its ability to manage robotic agents with completelyuncertain dynamics including uncertain mass matrices. This framework allows therobotic control to be environment-independent which can be used in varioussettings from underwater to space where identifying system dynamics parametersis challenging. The stability and parameter convergence of the closed-loopsystem are rigorously analyzed using the Lyapunov method. Numerical simulationsconducted on multi-agent robot manipulators validate the effectiveness of theproposed scheme. The identified nonlinear dynamics can be saved and reusedwhenever the system restarts. |


| Item |Content|
| --- |---|
|idx| 2403.00725v1 |
|title| Cost-Effective Activity Control of Asymptomatic Carriers in Layered Temporal Social Networks |
|authors| Masoumeh MoradianAresh DadlaniRasul KairgeldinAhmad Khonsari
|links| http://arxiv.org/abs/2403.00725v1 |
|updated| 2024-03-01 18:21:20 UTC |
|summary| The robustness of human social networks against epidemic propagation relieson the propensity for physical contact adaptation. During the early phase ofinfection asymptomatic carriers exhibit the same activity level as susceptibleindividuals which presents challenges for incorporating control measures inepidemic projection models. This paper focuses on modeling and cost-efficientactivity control of susceptible and carrier individuals in the context of thesusceptible-carrier-infected-removed SCIR epidemic model over a two-layercontact network. In this model individuals switch from a static contact layerto create new links in a temporal layer based on state-dependent activationrates. We derive conditions for the infection to die out or persist in ahomogeneous network. Considering the significant costs associated with reducingthe activity of susceptible and carrier individuals we formulate anoptimization problem to minimize the disease decay rate while constrained by alimited budget. We propose the use of successive geometric programming SGPapproximation for this optimization task. Through simulation experiments onPoisson random graphs we assess the impact of different parameters on diseaseprevalence. The results demonstrate that our SGP framework achieves a costreduction of nearly 33 compared to conventional methods based on degree andcloseness centrality. |


